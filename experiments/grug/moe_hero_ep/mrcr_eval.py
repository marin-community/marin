# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a hero checkpoint on oracle-normalized paired MRCR and derive its metrics.

The headline recovery ratio is ``(distractor_only NLL - full_context NLL) / (query_only NLL - needle_only NLL)``,
token-micro over aggregate paired contrasts. Its bootstrap resamples the four-condition source tuples within
needle-count and evidence-distance strata. Log scaling acts on the long ``full_context`` and ``distractor_only``
documents and on neither of the short ones, while a uniform qk change acts on all four, so recovery alone does
not compare those arms. Every group therefore also reports each condition's NLL and both gains with intervals.
"""

import dataclasses
import hashlib
import json
import logging
import math
import resource
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from haliax import Axis
from jax._src import config as jax_config
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.loader import DataLoader
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.eval import _calculate_bytes_per_token_type
from levanter.grug.grug_moe import MoeImplementation
from rigging.filesystem.storage_path import StoragePath

from experiments.datasets.mrcr import (
    MRCR_CONTEXT_CAP,
    MRCR_MANIFEST_NAME,
    MRCR_PROMPT_VARIANT,
    MrcrCondition,
    mrcr_format,
)
from experiments.grug.moe_hero_ep.model import _BATCH_AXES, Transformer
from experiments.grug.moe_hero_ep.train import _to_dropless_local

logger = logging.getLogger(__name__)

MRCR_METRIC_PREFIX = f"eval/mrcr/{MRCR_PROMPT_VARIANT}/cap_{MRCR_CONTEXT_CAP}"
MRCR_COMPONENT = "mrcr"
MRCR_BOOTSTRAP_SAMPLES = 10_000
MRCR_EXAMPLE_LOSSES_NAME = "mrcr_example_losses.jsonl"
MRCR_METRICS_NAME = "mrcr_metrics.json"
_MIN_ORACLE_GAIN_NLL = 0.1
_LOG2E = math.log2(math.e)


@dataclass(frozen=True)
class MrcrEvalConfig:
    data_path: str
    output_path: str
    tokenizer: str
    batch_size: int
    seq_len: int = MRCR_CONTEXT_CAP
    max_batches: int | None = None
    """Score only this many batches, as a smoke test. Metrics then cover those sources alone."""
    bootstrap_samples: int = MRCR_BOOTSTRAP_SAMPLES
    bootstrap_seed: int = 0


@dataclass(frozen=True)
class MrcrConditionLoss:
    source_id: str
    n_needles: int
    distance_band: str
    evidence_distance_tokens: int
    condition: str
    scored_tokens: int
    loss_sum: float
    scored_bytes: int


@dataclass(frozen=True)
class MrcrExampleLoss:
    source_id: str
    n_needles: int
    distance_band: str
    evidence_distance_tokens: int
    scored_tokens: int
    scored_bytes: int
    full_context_loss_sum: float
    query_only_loss_sum: float
    needle_only_loss_sum: float
    distractor_only_loss_sum: float


def pair_condition_losses(condition_losses: Iterable[MrcrConditionLoss]) -> list[MrcrExampleLoss]:
    """Group condition rows into source tuples, requiring identical scored tokens and finite losses."""
    by_source: dict[str, dict[str, MrcrConditionLoss]] = defaultdict(dict)
    for row in condition_losses:
        if row.condition in by_source[row.source_id]:
            raise ValueError(f"duplicate condition for source_id {row.source_id}: {row.condition}")
        by_source[row.source_id][row.condition] = row

    paired = []
    for source_id, conditions in by_source.items():
        if set(conditions) != {condition.value for condition in MrcrCondition}:
            raise ValueError(f"incomplete condition set for source_id {source_id}")
        rows = list(conditions.values())
        first = rows[0]
        identity = (first.n_needles, first.distance_band, first.evidence_distance_tokens)
        if any((row.n_needles, row.distance_band, row.evidence_distance_tokens) != identity for row in rows):
            raise ValueError(f"paired metadata differs for source_id {source_id}")
        if len({(row.scored_tokens, row.scored_bytes) for row in rows}) != 1 or first.scored_tokens <= 0:
            raise ValueError(f"unequal or zero scored counts for source_id {source_id}")
        if not all(math.isfinite(row.loss_sum) for row in rows):
            raise ValueError(f"non-finite loss for source_id {source_id}")
        paired.append(
            MrcrExampleLoss(
                source_id=source_id,
                n_needles=first.n_needles,
                distance_band=first.distance_band,
                evidence_distance_tokens=first.evidence_distance_tokens,
                scored_tokens=first.scored_tokens,
                scored_bytes=first.scored_bytes,
                full_context_loss_sum=conditions[MrcrCondition.FULL_CONTEXT].loss_sum,
                query_only_loss_sum=conditions[MrcrCondition.QUERY_ONLY].loss_sum,
                needle_only_loss_sum=conditions[MrcrCondition.NEEDLE_ONLY].loss_sum,
                distractor_only_loss_sum=conditions[MrcrCondition.DISTRACTOR_ONLY].loss_sum,
            )
        )
    return sorted(paired, key=lambda row: (row.n_needles, row.distance_band, row.source_id))


def _bootstrap_seed(seed: int, prefix: str) -> int:
    prefix_hash = int.from_bytes(hashlib.sha256(prefix.encode()).digest()[:8], "big")
    return (seed + prefix_hash) % (2**64)


def _contrast_values(rows: Sequence[MrcrExampleLoss]) -> dict[str, float]:
    tokens = sum(row.scored_tokens for row in rows)
    full = np.asarray([row.full_context_loss_sum for row in rows]) / np.asarray([row.scored_tokens for row in rows])
    query = np.asarray([row.query_only_loss_sum for row in rows]) / np.asarray([row.scored_tokens for row in rows])
    needle = np.asarray([row.needle_only_loss_sum for row in rows]) / np.asarray([row.scored_tokens for row in rows])
    distractor = np.asarray([row.distractor_only_loss_sum for row in rows]) / np.asarray(
        [row.scored_tokens for row in rows]
    )
    context = sum(row.query_only_loss_sum - row.full_context_loss_sum for row in rows) / tokens
    oracle = sum(row.query_only_loss_sum - row.needle_only_loss_sum for row in rows) / tokens
    real = sum(row.distractor_only_loss_sum - row.full_context_loss_sum for row in rows) / tokens
    macro_context = float(np.mean(query - full))
    macro_oracle = float(np.mean(query - needle))
    macro_real = float(np.mean(distractor - full))
    return {
        "micro_context": context,
        "micro_oracle": oracle,
        "micro_real": real,
        "micro_interaction": real - oracle,
        "macro_context": macro_context,
        "macro_oracle": macro_oracle,
        "macro_real": macro_real,
        "macro_interaction": macro_real - macro_oracle,
    }


def _paired_bootstrap(rows: Sequence[MrcrExampleLoss], *, samples: int, seed: int, prefix: str) -> dict[str, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(_bootstrap_seed(seed, prefix)))
    strata: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        strata[(row.n_needles, row.distance_band)].append(index)
    full = np.asarray([row.full_context_loss_sum for row in rows], dtype=np.float64)
    query = np.asarray([row.query_only_loss_sum for row in rows], dtype=np.float64)
    needle = np.asarray([row.needle_only_loss_sum for row in rows], dtype=np.float64)
    distractor = np.asarray([row.distractor_only_loss_sum for row in rows], dtype=np.float64)
    tokens = np.asarray([row.scored_tokens for row in rows], dtype=np.float64)
    sums = {
        name: np.zeros(samples, dtype=np.float64)
        for name in (
            "micro_context",
            "micro_oracle",
            "micro_real",
            "macro_context",
            "macro_oracle",
            "macro_real",
            "tokens",
            MrcrCondition.FULL_CONTEXT,
            MrcrCondition.QUERY_ONLY,
            MrcrCondition.NEEDLE_ONLY,
            MrcrCondition.DISTRACTOR_ONLY,
        )
    }
    total_examples = 0
    for indices in strata.values():
        stratum = np.asarray(indices, dtype=np.int64)
        selected = stratum[rng.integers(0, len(stratum), size=(samples, len(stratum)))]
        sums["micro_context"] += np.sum(query[selected] - full[selected], axis=1)
        sums["micro_oracle"] += np.sum(query[selected] - needle[selected], axis=1)
        sums["micro_real"] += np.sum(distractor[selected] - full[selected], axis=1)
        sums["macro_context"] += np.sum((query[selected] - full[selected]) / tokens[selected], axis=1)
        sums["macro_oracle"] += np.sum((query[selected] - needle[selected]) / tokens[selected], axis=1)
        sums["macro_real"] += np.sum((distractor[selected] - full[selected]) / tokens[selected], axis=1)
        sums["tokens"] += np.sum(tokens[selected], axis=1)
        sums[MrcrCondition.FULL_CONTEXT] += np.sum(full[selected], axis=1)
        sums[MrcrCondition.QUERY_ONLY] += np.sum(query[selected], axis=1)
        sums[MrcrCondition.NEEDLE_ONLY] += np.sum(needle[selected], axis=1)
        sums[MrcrCondition.DISTRACTOR_ONLY] += np.sum(distractor[selected], axis=1)
        total_examples += len(stratum)

    def recovery(real: np.ndarray, oracle: np.ndarray) -> np.ndarray:
        return np.divide(real, oracle, out=np.full(samples, np.nan, dtype=np.float64), where=oracle != 0)

    denominator = sums["tokens"]
    return {
        "micro_context": sums["micro_context"] / denominator,
        "micro_oracle": sums["micro_oracle"] / denominator,
        "micro_real": sums["micro_real"] / denominator,
        "micro_interaction": (sums["micro_real"] - sums["micro_oracle"]) / denominator,
        "micro_recovery": recovery(sums["micro_real"], sums["micro_oracle"]),
        "macro_context": sums["macro_context"] / total_examples,
        "macro_oracle": sums["macro_oracle"] / total_examples,
        "macro_real": sums["macro_real"] / total_examples,
        "macro_interaction": (sums["macro_real"] - sums["macro_oracle"]) / total_examples,
        "macro_recovery": recovery(sums["macro_real"], sums["macro_oracle"]),
        **{condition.value: sums[condition] / denominator for condition in MrcrCondition},
    }


def _percentile_bounds(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, (2.5, 97.5))
    return float(low), float(high)


def _condition_loss_sum(row: MrcrExampleLoss, condition: MrcrCondition) -> float:
    return getattr(row, f"{condition.value}_loss_sum")


def _group_metrics(
    prefix: str, rows: Sequence[MrcrExampleLoss], *, bootstrap_samples: int, bootstrap_seed: int
) -> dict[str, float]:
    if not rows:
        raise ValueError(f"cannot derive metrics for empty group {prefix}")
    tokens = sum(row.scored_tokens for row in rows)
    scored_bytes = sum(row.scored_bytes for row in rows)
    contrasts = _contrast_values(rows)
    bootstrap = _paired_bootstrap(rows, samples=bootstrap_samples, seed=bootstrap_seed, prefix=prefix)
    metrics = {f"{prefix}/scored_tokens": float(tokens), f"{prefix}/examples": float(len(rows))}
    for condition in MrcrCondition:
        loss_sum = sum(_condition_loss_sum(row, condition) for row in rows)
        low, high = _percentile_bounds(bootstrap[condition.value])
        metrics[f"{prefix}/{condition.value}/micro_loss"] = loss_sum / tokens
        metrics[f"{prefix}/{condition.value}/micro_loss_ci95_low"] = low
        metrics[f"{prefix}/{condition.value}/micro_loss_ci95_high"] = high
        metrics[f"{prefix}/{condition.value}/micro_bpb"] = loss_sum / scored_bytes * _LOG2E
        metrics[f"{prefix}/{condition.value}/loss"] = loss_sum / tokens
        metrics[f"{prefix}/{condition.value}/bpb"] = loss_sum / scored_bytes * _LOG2E
    for aggregation in ("micro", "macro"):
        gain = f"{aggregation}_context"
        low, high = _percentile_bounds(bootstrap[gain])
        metrics[f"{prefix}/{gain}_ppl_ratio"] = math.exp(contrasts[gain])
        metrics[f"{prefix}/{gain}_ppl_ratio_ci95_low"] = math.exp(low)
        metrics[f"{prefix}/{gain}_ppl_ratio_ci95_high"] = math.exp(high)
        for contrast in ("context", "oracle", "real", "interaction"):
            name = f"{aggregation}_{contrast}"
            stem = f"{prefix}/{name}_nll" if contrast == "interaction" else f"{prefix}/{name}_gain_nll"
            low, high = _percentile_bounds(bootstrap[name])
            metrics[stem] = contrasts[name]
            metrics[f"{stem}_ci95_low"] = low
            metrics[f"{stem}_ci95_high"] = high
        oracle = contrasts[f"{aggregation}_oracle"]
        oracle_bootstrap = bootstrap[f"{aggregation}_oracle"]
        positive_fraction = float(np.mean(oracle_bootstrap > 0))
        identifiable = (
            oracle >= _MIN_ORACLE_GAIN_NLL
            and _percentile_bounds(oracle_bootstrap)[0] > 0
            and positive_fraction >= 0.99
            and bool(np.all(oracle_bootstrap != 0))
        )
        metrics[f"{prefix}/{aggregation}_recovery_oracle_positive_fraction"] = positive_fraction
        metrics[f"{prefix}/{aggregation}_recovery_identifiable"] = float(identifiable)
        if identifiable:
            low, high = _percentile_bounds(bootstrap[f"{aggregation}_recovery"])
            metrics[f"{prefix}/{aggregation}_needle_recovery_ratio"] = contrasts[f"{aggregation}_real"] / oracle
            metrics[f"{prefix}/{aggregation}_needle_recovery_ratio_ci95_low"] = low
            metrics[f"{prefix}/{aggregation}_needle_recovery_ratio_ci95_high"] = high
    return metrics


def derive_mrcr_metrics(
    rows: Sequence[MrcrExampleLoss], *, bootstrap_samples: int = MRCR_BOOTSTRAP_SAMPLES, bootstrap_seed: int = 0
) -> dict[str, float]:
    """Return aggregate, per-needle-count, per-distance and per-cell metrics under the #8701 names."""
    if not rows:
        raise ValueError("no paired MRCR rows")
    if len({row.source_id for row in rows}) != len(rows):
        raise ValueError("paired MRCR rows contain duplicate source IDs")
    groups: dict[str, list[MrcrExampleLoss]] = defaultdict(list)
    for row in rows:
        groups[f"{MRCR_METRIC_PREFIX}/{row.n_needles}needle/{row.distance_band}"].append(row)
        groups[f"{MRCR_METRIC_PREFIX}/{row.n_needles}needle"].append(row)
        groups[f"{MRCR_METRIC_PREFIX}/{row.distance_band}"].append(row)
    groups[MRCR_METRIC_PREFIX] = list(rows)
    metrics: dict[str, float] = {}
    for prefix, group in sorted(groups.items()):
        metrics.update(_group_metrics(prefix, group, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed))
    return metrics


def _read_manifest(path: StoragePath) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mrcr_data_config(data_path: str, tokenizer: str) -> LmDataConfig:
    component = DatasetComponent(cache_dir=data_path, format=mrcr_format())
    return LmDataConfig(
        tokenizer=tokenizer,
        cache_dir=None,
        components={MRCR_COMPONENT: component},
        train_weights={MRCR_COMPONENT: 0.0},
        auto_build_caches=False,
    )


def _sequence_score_sums(
    model: Transformer,
    tokens: jax.Array,
    loss_weight: jax.Array,
    mask: Any,
    byte_lengths: jax.Array,
    *,
    sharding: NamedSharding,
    mp: jmp.Policy,
    implementation: MoeImplementation,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return each sequence's summed loss, scored-token count and scored-byte count, with the batch axis kept."""
    model = _to_dropless_local(mp.cast_to_compute(model), implementation=implementation)
    losses = model.next_token_loss(tokens, loss_weight, mask=mask, reduction="none", logsumexp_weight=None)
    losses = jax.sharding.reshard(losses, sharding)
    weights = jax.sharding.reshard(loss_weight, sharding).astype(losses.dtype)
    labels = jnp.pad(tokens[:, 1:], ((0, 0), (0, 1)))
    bytes_per_position = byte_lengths.at[labels].get(out_sharding=sharding)
    return (
        jnp.sum(losses * weights, axis=-1),
        jnp.sum(weights, axis=-1),
        jnp.sum(bytes_per_position * weights, axis=-1),
    )


def _memory_summary() -> str:
    """This process's peak device and host memory, for the batch log. CPU devices report no device stats."""
    stats = jax.local_devices()[0].memory_stats()
    device = f"peak_hbm={stats['peak_bytes_in_use'] / 1024**3:.1f}GiB " if stats else ""
    return f"{device}max_rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2:.1f}GiB"


def score_condition_rows(
    params: Transformer,
    config: MrcrEvalConfig,
    *,
    mesh: Mesh,
    mp: jmp.Policy,
    implementation: MoeImplementation,
) -> list[MrcrConditionLoss]:
    """Score every cache row in order and bind each to its manifest line."""
    manifest = _read_manifest(StoragePath(config.data_path) / MRCR_MANIFEST_NAME)
    data = mrcr_data_config(config.data_path, config.tokenizer)
    position = Axis("position", config.seq_len)
    dataset = data.build_token_datasets(data.build_caches("validation"), position, split="validation")[MRCR_COMPONENT]
    if len(dataset.as_sync_dataset()) != len(manifest):
        raise ValueError(f"cache has {len(dataset.as_sync_dataset())} rows, manifest has {len(manifest)}")
    loader = DataLoader(dataset, config.batch_size, mesh=mesh, axis_resources={"batch": _BATCH_AXES})
    byte_lengths = _calculate_bytes_per_token_type(data.the_tokenizer)
    sharding = NamedSharding(mesh, P(_BATCH_AXES, None))
    score = eqx.filter_jit(_sequence_score_sums)

    output: list[MrcrConditionLoss] = []
    batch_start = time.monotonic()
    with jax_config.enable_pgle(False):
        for batch_index, batch in enumerate(loader):
            if config.max_batches is not None and batch_index >= config.max_batches:
                break
            sums = score(
                params,
                batch.tokens,
                batch.loss_weight,
                batch.attn_mask,
                byte_lengths,
                sharding=sharding,
                mp=mp,
                implementation=implementation,
            )
            loss_sums, scored, scored_bytes = (
                np.asarray(multihost_utils.process_allgather(value, tiled=True)).reshape(-1) for value in sums
            )
            # A padded final batch has more rows than the manifest has left; only real rows bind.
            batch_rows = manifest[len(output) : len(output) + config.batch_size]
            logger.info(
                "MRCR batch %d real tokens per row: %s",
                batch_index,
                [record["length_tokens"] for record in batch_rows],
            )
            for offset, record in enumerate(batch_rows):
                if int(scored[offset]) != record["scored_tokens"]:
                    raise ValueError(
                        f"row {len(output)}: scored {int(scored[offset])} tokens, "
                        f"manifest says {record['scored_tokens']}"
                    )
                output.append(
                    MrcrConditionLoss(
                        source_id=record["source_id"],
                        n_needles=record["n_needles"],
                        distance_band=record["distance_band"],
                        evidence_distance_tokens=record["evidence_distance_tokens"],
                        condition=record["condition"],
                        scored_tokens=int(scored[offset]),
                        loss_sum=float(loss_sums[offset]),
                        scored_bytes=int(scored_bytes[offset]),
                    )
                )
            elapsed = time.monotonic() - batch_start
            batch_start = time.monotonic()
            logger.info(
                "MRCR batch %d: %d of %d rows, %.1fs, %s",
                batch_index,
                len(output),
                len(manifest),
                elapsed,
                _memory_summary(),
            )
    return output


def write_mrcr_results(output_path: str, rows: Sequence[MrcrExampleLoss], metrics: dict[str, float], **record: Any):
    output = StoragePath(output_path)
    (output / MRCR_EXAMPLE_LOSSES_NAME).write_text(
        "".join(json.dumps(dataclasses.asdict(row), sort_keys=True) + "\n" for row in rows)
    )
    (output / MRCR_METRICS_NAME).write_text(
        json.dumps({**record, "metrics": dict(sorted(metrics.items()))}, indent=2, sort_keys=True)
    )


def evaluate_mrcr(
    params: Transformer,
    config: MrcrEvalConfig,
    *,
    mesh: Mesh,
    mp: jmp.Policy,
    implementation: MoeImplementation,
    record: dict[str, Any],
) -> dict[str, float]:
    """Score ``params`` and return metrics on process 0, writing its artifact; other processes return nothing."""
    condition_losses = score_condition_rows(params, config, mesh=mesh, mp=mp, implementation=implementation)
    if jax.process_index() != 0:
        return {}
    rows = pair_condition_losses(condition_losses)
    metrics = derive_mrcr_metrics(rows, bootstrap_samples=config.bootstrap_samples, bootstrap_seed=config.bootstrap_seed)
    write_mrcr_results(config.output_path, rows, metrics, **record)
    return metrics
