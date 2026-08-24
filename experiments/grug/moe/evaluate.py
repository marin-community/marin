# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import subprocess
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar, cast

import fsspec
import jax
import jax.numpy as jnp
import jmp
import levanter.tracker
import numpy as np
from fray.cluster import ResourceConfig
from haliax.partitioning import set_mesh
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.checkpoint import load_checkpoint
from levanter.data import DataLoader
from levanter.data.text import GrugLmExample
from levanter.data.text.datasets import LmDataConfig
from levanter.eval import _calculate_bytes_per_token_type
from levanter.grug.sharding import compact_grug_mesh
from levanter.tracker import TrackerConfig

from experiments.datasets.mrcr import MRCR_DATASET_REVISION, MrcrPromptVariant
from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.grug.moe.model import GrugModelConfig, Transformer

CP_BASE_COMMIT = "db7ffddd339dd4db71fbb83ae2555abe3522c894"
_APPROVED_MODEL_OVERRIDES = frozenset(("max_seq_len", "qk_mult", "attention_implementation"))
_BATCH_AXES = ("replica_dcn", "data", "expert")
ParamsT = TypeVar("ParamsT")


@dataclass(frozen=True)
class GrugCheckpointEvalRuntimeConfig:
    mp: str
    tracker: TrackerConfig
    seed: int = 0
    eval_batch_size: int = 256
    replica_axis_size: int = 1
    data_axis_size: int = 256
    context_axis_size: int = 4
    expert_axis_size: int = 1
    model_axis_size: int = 1


@dataclass(frozen=True)
class GrugCheckpointEvalConfig:
    run_id: str
    checkpoint_path: str
    context_cap: int
    prompt_variant: MrcrPromptVariant
    qk_mult: float
    model: GrugModelConfig
    data: LmDataConfig
    dataset_stats_path: str
    dataset_manifest_paths: dict[str, str]
    runtime: GrugCheckpointEvalRuntimeConfig
    output_path: str
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 0


@dataclass(frozen=True)
class MrcrConditionLoss:
    source_id: str
    prompt_variant: str
    context_cap: int
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
    prompt_variant: str
    context_cap: int
    n_needles: int
    distance_band: str
    evidence_distance_tokens: int
    scored_tokens: int
    full_context_loss_sum: float
    query_only_loss_sum: float
    full_context_scored_bytes: int
    query_only_scored_bytes: int

    @property
    def full_context_nll(self) -> float:
        return self.full_context_loss_sum / self.scored_tokens

    @property
    def query_only_nll(self) -> float:
        return self.query_only_loss_sum / self.scored_tokens

    @property
    def context_gain_nll(self) -> float:
        return self.query_only_nll - self.full_context_nll

    def persisted_record(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "prompt_variant": self.prompt_variant,
            "context_cap": self.context_cap,
            "n_needles": self.n_needles,
            "distance_band": self.distance_band,
            "evidence_distance_tokens": self.evidence_distance_tokens,
            "scored_tokens": self.scored_tokens,
            "full_context_loss_sum": self.full_context_loss_sum,
            "query_only_loss_sum": self.query_only_loss_sum,
            "full_context_nll": self.full_context_nll,
            "query_only_nll": self.query_only_nll,
            "context_gain_nll": self.context_gain_nll,
        }


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical_67b_model() -> GrugModelConfig:
    model = MoeMuonHHeuristic().build_model_config(2560, seq_len=65_536)
    return dataclasses.replace(
        model,
        disable_pko=True,
        disable_long_rope=True,
        sliding_window=2048,
        use_array_stacked_blocks=True,
        qk_mult=1.57,
        hybrid_attention_flops_accounting=True,
    )


def normalized_model_config(model: GrugModelConfig) -> dict[str, Any]:
    return {
        field.name: _jsonable(getattr(model, field.name))
        for field in dataclasses.fields(model)
        if field.name not in _APPROVED_MODEL_OVERRIDES
    }


def validate_grug_checkpoint_eval_config(config: GrugCheckpointEvalConfig) -> None:
    if config.context_cap != config.model.max_seq_len:
        raise ValueError("context_cap must equal model.max_seq_len")
    if config.qk_mult != config.model.qk_mult:
        raise ValueError("qk_mult must equal model.qk_mult")
    if config.bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if config.runtime.eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")
    if config.runtime.data_axis_size * config.runtime.replica_axis_size * config.runtime.expert_axis_size != (
        config.runtime.eval_batch_size
    ):
        raise ValueError("eval_batch_size must equal replica_axis_size * data_axis_size * expert_axis_size")
    expected = normalized_model_config(_canonical_67b_model())
    actual = normalized_model_config(config.model)
    mismatched = sorted(field for field in expected if actual[field] != expected[field])
    if mismatched:
        raise ValueError(f"model differs from the canonical 67B config in static fields: {', '.join(mismatched)}")

    variant_prefix = f"{config.prompt_variant.value}/cap_{config.context_cap}/"
    component_names = tuple(config.data.components)
    if not component_names:
        raise ValueError("evaluation data has no components")
    invalid = sorted(name for name in component_names if not name.startswith(variant_prefix))
    if invalid:
        raise ValueError(f"evaluation data contains a different cap or prompt variant: {invalid[0]}")
    expected_manifests = {_cell_name(name) for name in component_names}
    if set(config.dataset_manifest_paths) != expected_manifests:
        raise ValueError("dataset_manifest_paths must contain exactly one manifest for every paired MRCR cell")
    for cell in expected_manifests:
        conditions = {name.rsplit("/", 1)[-1] for name in component_names if _cell_name(name) == cell}
        if conditions != {"full_context", "query_only"}:
            raise ValueError(f"paired MRCR cell is incomplete: {cell}")


def _cell_name(component_name: str) -> str:
    cell, separator, condition = component_name.rpartition("/")
    if not separator or condition not in ("full_context", "query_only"):
        raise ValueError(f"invalid MRCR component name: {component_name}")
    return cell


def pair_mrcr_condition_losses(condition_losses: Iterable[MrcrConditionLoss]) -> list[MrcrExampleLoss]:
    by_source: dict[str, dict[str, MrcrConditionLoss]] = defaultdict(dict)
    for row in condition_losses:
        if row.condition in by_source[row.source_id]:
            raise ValueError(f"duplicate condition for source_id {row.source_id}: {row.condition}")
        by_source[row.source_id][row.condition] = row

    paired: list[MrcrExampleLoss] = []
    for source_id, conditions in by_source.items():
        if set(conditions) != {"full_context", "query_only"}:
            raise ValueError(f"missing pair for source_id {source_id}")
        full = conditions["full_context"]
        query = conditions["query_only"]
        full_identity = dataclasses.astuple(full)[:6]
        query_identity = dataclasses.astuple(query)[:6]
        if full_identity != query_identity:
            raise ValueError(f"paired metadata differs for source_id {source_id}")
        if full.scored_tokens != query.scored_tokens:
            raise ValueError(f"unequal scored-token counts for source_id {full.source_id}")
        if full.scored_tokens <= 0:
            raise ValueError(f"zero scored tokens for source_id {full.source_id}")
        if full.scored_bytes != query.scored_bytes or full.scored_bytes <= 0:
            raise ValueError(f"unequal or zero scored-byte counts for source_id {full.source_id}")
        if not math.isfinite(full.loss_sum) or not math.isfinite(query.loss_sum):
            raise ValueError(f"non-finite loss for source_id {full.source_id}")
        paired.append(
            MrcrExampleLoss(
                source_id=full.source_id,
                prompt_variant=full.prompt_variant,
                context_cap=full.context_cap,
                n_needles=full.n_needles,
                distance_band=full.distance_band,
                evidence_distance_tokens=full.evidence_distance_tokens,
                scored_tokens=full.scored_tokens,
                full_context_loss_sum=full.loss_sum,
                query_only_loss_sum=query.loss_sum,
                full_context_scored_bytes=full.scored_bytes,
                query_only_scored_bytes=query.scored_bytes,
            )
        )
    return sorted(paired, key=lambda row: (row.prompt_variant, row.n_needles, row.distance_band, row.source_id))


def _bootstrap_seed(seed: int, prefix: str) -> int:
    prefix_hash = int.from_bytes(hashlib.sha256(prefix.encode()).digest()[:8], "big")
    return (seed + prefix_hash) % (2**64)


def _gain_values(rows: Sequence[MrcrExampleLoss]) -> tuple[float, float]:
    full_sum = sum(row.full_context_loss_sum for row in rows)
    query_sum = sum(row.query_only_loss_sum for row in rows)
    tokens = sum(row.scored_tokens for row in rows)
    micro = (query_sum - full_sum) / tokens
    macro = float(np.mean([row.context_gain_nll for row in rows]))
    return micro, macro


def _paired_bootstrap(
    rows: Sequence[MrcrExampleLoss], *, samples: int, seed: int, prefix: str
) -> tuple[tuple[float, float], tuple[float, float]]:
    rng = np.random.Generator(np.random.PCG64(_bootstrap_seed(seed, prefix)))
    strata: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        strata[(row.n_needles, row.distance_band)].append(index)
    full = np.asarray([row.full_context_loss_sum for row in rows], dtype=np.float64)
    query = np.asarray([row.query_only_loss_sum for row in rows], dtype=np.float64)
    tokens = np.asarray([row.scored_tokens for row in rows], dtype=np.float64)
    gains = query / tokens - full / tokens
    boot_micro = np.zeros(samples, dtype=np.float64)
    boot_macro_sum = np.zeros(samples, dtype=np.float64)
    total_examples = 0
    for indices in strata.values():
        stratum = np.asarray(indices, dtype=np.int64)
        selected = stratum[rng.integers(0, len(stratum), size=(samples, len(stratum)))]
        boot_micro += np.sum(query[selected] - full[selected], axis=1)
        boot_macro_sum += np.sum(gains[selected], axis=1)
        total_examples += len(stratum)
    denominator = np.zeros(samples, dtype=np.float64)
    # Repeat the deterministic stratum draws so the token denominator matches each numerator draw.
    rng = np.random.Generator(np.random.PCG64(_bootstrap_seed(seed, prefix)))
    for indices in strata.values():
        stratum = np.asarray(indices, dtype=np.int64)
        selected = stratum[rng.integers(0, len(stratum), size=(samples, len(stratum)))]
        denominator += np.sum(tokens[selected], axis=1)
    boot_micro /= denominator
    boot_macro = boot_macro_sum / total_examples
    micro_percentiles = np.percentile(boot_micro, (2.5, 97.5))
    macro_percentiles = np.percentile(boot_macro, (2.5, 97.5))
    micro_bounds = (float(micro_percentiles[0]), float(micro_percentiles[1]))
    macro_bounds = (float(macro_percentiles[0]), float(macro_percentiles[1]))
    return micro_bounds, macro_bounds


def _group_metrics(
    prefix: str,
    rows: Sequence[MrcrExampleLoss],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    include_cell_raw: bool,
) -> dict[str, float]:
    if not rows:
        raise ValueError(f"cannot derive metrics for empty group {prefix}")
    full_loss = sum(row.full_context_loss_sum for row in rows) / sum(row.scored_tokens for row in rows)
    query_loss = sum(row.query_only_loss_sum for row in rows) / sum(row.scored_tokens for row in rows)
    micro, macro = _gain_values(rows)
    micro_ci, macro_ci = _paired_bootstrap(rows, samples=bootstrap_samples, seed=bootstrap_seed, prefix=prefix)
    metrics = {
        f"{prefix}/scored_tokens": float(sum(row.scored_tokens for row in rows)),
        f"{prefix}/examples": float(len(rows)),
        f"{prefix}/micro_context_gain_nll": micro,
        f"{prefix}/micro_context_gain_nll_ci95_low": micro_ci[0],
        f"{prefix}/micro_context_gain_nll_ci95_high": micro_ci[1],
        f"{prefix}/micro_context_ppl_ratio": math.exp(micro),
        f"{prefix}/micro_context_ppl_ratio_ci95_low": math.exp(micro_ci[0]),
        f"{prefix}/micro_context_ppl_ratio_ci95_high": math.exp(micro_ci[1]),
        f"{prefix}/macro_context_gain_nll": macro,
        f"{prefix}/macro_context_gain_nll_ci95_low": macro_ci[0],
        f"{prefix}/macro_context_gain_nll_ci95_high": macro_ci[1],
        f"{prefix}/macro_context_ppl_ratio": math.exp(macro),
        f"{prefix}/macro_context_ppl_ratio_ci95_low": math.exp(macro_ci[0]),
        f"{prefix}/macro_context_ppl_ratio_ci95_high": math.exp(macro_ci[1]),
    }
    if include_cell_raw:
        full_bytes = sum(row.full_context_scored_bytes for row in rows)
        query_bytes = sum(row.query_only_scored_bytes for row in rows)
        metrics.update(
            {
                f"{prefix}/full_context/loss": full_loss,
                f"{prefix}/full_context/bpb": (
                    sum(row.full_context_loss_sum for row in rows) / full_bytes * math.log2(math.e)
                ),
                f"{prefix}/query_only/loss": query_loss,
                f"{prefix}/query_only/bpb": (
                    sum(row.query_only_loss_sum for row in rows) / query_bytes * math.log2(math.e)
                ),
            }
        )
    else:
        metrics[f"{prefix}/full_context/micro_loss"] = full_loss
        metrics[f"{prefix}/query_only/micro_loss"] = query_loss
    return metrics


def derive_mrcr_metrics(
    rows: Sequence[MrcrExampleLoss], *, bootstrap_samples: int = 10_000, bootstrap_seed: int = 0
) -> dict[str, float]:
    if not rows:
        raise ValueError("no paired MRCR rows")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    variants = {row.prompt_variant for row in rows}
    caps = {row.context_cap for row in rows}
    if len(variants) != 1 or len(caps) != 1:
        raise ValueError("one evaluation result must contain one prompt variant and one context cap")
    if len({row.source_id for row in rows}) != len(rows):
        raise ValueError("paired MRCR rows contain duplicate source IDs")
    variant = next(iter(variants))
    cap = next(iter(caps))
    base = f"eval/mrcr/{variant}/cap_{cap}"
    metrics: dict[str, float] = {}
    cells: dict[tuple[int, str], list[MrcrExampleLoss]] = defaultdict(list)
    distances: dict[str, list[MrcrExampleLoss]] = defaultdict(list)
    for row in rows:
        cells[(row.n_needles, row.distance_band)].append(row)
        distances[row.distance_band].append(row)
    for (needles, distance), group in sorted(cells.items()):
        prefix = f"{base}/{needles}needle/{distance}"
        metrics.update(
            _group_metrics(
                prefix,
                group,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                include_cell_raw=True,
            )
        )
    for distance, group in sorted(distances.items()):
        metrics.update(
            _group_metrics(
                f"{base}/{distance}",
                group,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                include_cell_raw=False,
            )
        )
    metrics.update(
        _group_metrics(
            base,
            rows,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            include_cell_raw=False,
        )
    )
    return metrics


def _read_json(path: str) -> Any:
    with fsspec.open(path, "rt") as source:
        return json.load(source)


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    with fsspec.open(path, "rt") as source:
        return [json.loads(line) for line in source if line.strip()]


def _serialized_jsonl(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        for record in records
    )


def _write_idempotent(path: str, content: bytes) -> None:
    fs, _, paths = fsspec.get_fs_token_paths(path)
    plain_path = paths[0]
    if fs.exists(plain_path):
        with fs.open(plain_path, "rb") as existing:
            if existing.read() == content:
                return
        raise ValueError(f"conflicting output already exists: {path}")
    parent = plain_path.rsplit("/", 1)[0]
    if parent:
        fs.makedirs(parent, exist_ok=True)
    temporary = f"{plain_path}.tmp-{uuid.uuid4().hex}"
    try:
        with fs.open(temporary, "wb") as output:
            output.write(content)
        if fs.exists(plain_path):
            with fs.open(plain_path, "rb") as existing:
                if existing.read() != content:
                    raise ValueError(f"conflicting output already exists: {path}")
            return
        fs.mv(temporary, plain_path)
    finally:
        if fs.exists(temporary):
            fs.rm(temporary)


def _validate_existing_output(path: str, content: bytes) -> None:
    fs, _, paths = fsspec.get_fs_token_paths(path)
    plain_path = paths[0]
    if not fs.exists(plain_path):
        return
    with fs.open(plain_path, "rb") as existing:
        if existing.read() != content:
            raise ValueError(f"conflicting output already exists: {path}")


def persist_grug_checkpoint_eval(
    config: GrugCheckpointEvalConfig,
    *,
    checkpoint_step: int,
    paired_rows: Sequence[MrcrExampleLoss],
    metrics: Mapping[str, float],
) -> None:
    ordered_rows = sorted(
        paired_rows, key=lambda row: (row.prompt_variant, row.n_needles, row.distance_band, row.source_id)
    )
    example_content = _serialized_jsonl(row.persisted_record() for row in ordered_rows)
    model_digest = _sha256_json(normalized_model_config(config.model))
    evaluation_digest = _sha256_json(
        {
            "context_cap": config.context_cap,
            "prompt_variant": config.prompt_variant,
            "qk_mult": config.qk_mult,
            "dataset_stats_path": config.dataset_stats_path,
            "dataset_manifest_paths": config.dataset_manifest_paths,
            "bootstrap_samples": config.bootstrap_samples,
            "bootstrap_seed": config.bootstrap_seed,
        }
    )
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    record = {
        "run_id": config.run_id,
        "checkpoint_path": config.checkpoint_path,
        "checkpoint_step": checkpoint_step,
        "context_cap": config.context_cap,
        "prompt_variant": config.prompt_variant.value,
        "qk_mult": config.qk_mult,
        "dataset_revision": MRCR_DATASET_REVISION,
        "model_config_sha256": model_digest,
        "evaluation_config_sha256": evaluation_digest,
        "cp_base_commit": CP_BASE_COMMIT,
        "evaluator_commit": commit,
        "metrics": dict(sorted(metrics.items())),
    }
    metrics_content = _serialized_jsonl((record,))
    examples_path = os.path.join(config.output_path, "mrcr_example_losses.jsonl")
    metrics_path = os.path.join(config.output_path, "eval_metrics.jsonl")
    # Check both destinations before writing either so a conflicting retry cannot leave a half-updated pair.
    _validate_existing_output(examples_path, example_content)
    _validate_existing_output(metrics_path, metrics_content)
    _write_idempotent(examples_path, example_content)
    _write_idempotent(metrics_path, metrics_content)


def _manifest_records(config: GrugCheckpointEvalConfig) -> dict[str, list[dict[str, Any]]]:
    stats = _read_json(config.dataset_stats_path)
    if stats["dataset_revision"] != MRCR_DATASET_REVISION:
        raise ValueError("dataset revision does not match the pinned MRCR revision")
    manifests: dict[str, list[dict[str, Any]]] = {}
    for cell, path in config.dataset_manifest_paths.items():
        records = _read_jsonl(path)
        accepted = stats["accepted"].get(cell)
        if accepted is None or accepted["examples"] != len(records):
            raise ValueError(f"manifest row count disagrees with stats for {cell}")
        if sum(record["scored_tokens"] for record in records) != accepted["scored_tokens"]:
            raise ValueError(f"manifest scored-token count disagrees with stats for {cell}")
        manifests[cell] = records
    return manifests


def summarize_per_example_losses(
    per_position_loss: jax.Array, loss_weight: jax.Array, bytes_per_position: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reduce token losses while preserving one sum and count per sequence."""
    return (
        jnp.sum(per_position_loss * loss_weight, axis=-1),
        jnp.sum(loss_weight, axis=-1),
        jnp.sum(bytes_per_position * loss_weight, axis=-1),
    )


def _evaluate_components(
    config: GrugCheckpointEvalConfig, model: Transformer, mesh: jax.sharding.Mesh
) -> list[MrcrConditionLoss]:
    manifests = _manifest_records(config)
    eval_sets = config.data.tagged_eval_grug_sets(seq_len=config.context_cap)
    byte_lengths = _calculate_bytes_per_token_type(config.data.the_tokenizer)
    if byte_lengths is None:
        raise ValueError("MRCR BPB evaluation requires a tokenizer")
    sharding = NamedSharding(mesh, P(_BATCH_AXES, None))

    @jax.jit
    def loss_sums(batch: GrugLmExample) -> tuple[jax.Array, jax.Array, jax.Array]:
        losses = model.next_token_loss(
            batch.tokens,
            batch.loss_weight,
            mask=batch.attn_mask,
            reduction="none",
            logsumexp_weight=None,
        )
        losses = jax.sharding.reshard(losses, sharding)
        weights = jax.sharding.reshard(batch.loss_weight, sharding)
        target_ids = jnp.roll(batch.tokens, -1, axis=-1)
        bytes_per_position = byte_lengths.at[target_ids].get(out_sharding=sharding)
        return summarize_per_example_losses(losses, weights, bytes_per_position)

    output: list[MrcrConditionLoss] = []
    for dataset, tags in eval_sets:
        component_names = [tag for tag in tags if tag in config.data.components]
        if len(component_names) != 1:
            raise ValueError(f"could not identify one component from tags: {tags}")
        component = component_names[0]
        cell = _cell_name(component)
        condition = component.rsplit("/", 1)[-1]
        manifest = manifests[cell]
        loader = DataLoader(
            dataset,
            config.runtime.eval_batch_size,
            mesh=mesh,
            axis_resources={"batch": _BATCH_AXES},
            pad_final_batch=True,
        )
        ordinal = 0
        for batch in loader:
            batch_loss, batch_tokens, batch_bytes = loss_sums(batch)
            gathered_loss = np.asarray(multihost_utils.process_allgather(batch_loss)).reshape(-1)
            gathered_tokens = np.asarray(multihost_utils.process_allgather(batch_tokens)).reshape(-1)
            gathered_bytes = np.asarray(multihost_utils.process_allgather(batch_bytes)).reshape(-1)
            if jax.process_index() != 0:
                continue
            for loss_sum, scored_tokens, scored_bytes in zip(
                gathered_loss, gathered_tokens, gathered_bytes, strict=True
            ):
                if scored_tokens == 0:
                    continue
                if ordinal >= len(manifest):
                    raise ValueError(f"tokenized cache contains more rows than manifest for {cell}")
                record = manifest[ordinal]
                if int(scored_tokens) != record["scored_tokens"]:
                    raise ValueError(f"assistant-mask sum disagrees with manifest for source_id {record['source_id']}")
                output.append(
                    MrcrConditionLoss(
                        source_id=record["source_id"],
                        prompt_variant=config.prompt_variant.value,
                        context_cap=config.context_cap,
                        n_needles=int(cell.split("/")[2].removesuffix("needle")),
                        distance_band=cell.split("/")[3],
                        evidence_distance_tokens=record["evidence_distance_tokens"],
                        condition=condition,
                        scored_tokens=int(scored_tokens),
                        loss_sum=float(loss_sum),
                        scored_bytes=int(scored_bytes),
                    )
                )
                ordinal += 1
        if jax.process_index() == 0 and ordinal != len(manifest):
            raise ValueError(f"tokenized cache row count disagrees with manifest for {cell}")
    return output


def load_grug_checkpoint_params(
    checkpoint_path: str, *, initialized_params: ParamsT, mesh: jax.sharding.Mesh | None
) -> tuple[int, ParamsT]:
    """Restore only a checkpoint's stored step and model parameters."""
    restored = load_checkpoint(
        {"step": jnp.asarray(0, dtype=jnp.int32), "params": initialized_params},
        checkpoint_path,
        axis_mapping=None,
        mesh=mesh,
        allow_partial=False,
    )
    return int(restored["step"]), cast(ParamsT, restored["params"])


def evaluate_grug_checkpoint(config: GrugCheckpointEvalConfig) -> dict[str, float]:
    """Evaluate one Grug checkpoint without constructing training state."""
    validate_grug_checkpoint_eval_config(config)
    tracker = config.runtime.tracker.init(config.run_id)
    levanter.tracker.set_global_tracker(tracker)
    mesh = compact_grug_mesh(
        expert_axis_size=config.runtime.expert_axis_size,
        replica_axis_size=config.runtime.replica_axis_size,
        model_axis_size=config.runtime.model_axis_size,
        context_axis_size=config.runtime.context_axis_size,
    )
    if mesh.shape["data"] != config.runtime.data_axis_size:
        raise ValueError(
            f"configured data_axis_size={config.runtime.data_axis_size} but resolved mesh has data={mesh.shape['data']}"
        )
    policy = jmp.get_policy(config.runtime.mp)
    with set_mesh(mesh):
        model = jax.jit(lambda key: policy.cast_to_param(Transformer.init(config.model, key=key)))(
            jax.random.PRNGKey(config.runtime.seed)
        )
        checkpoint_step, params = load_grug_checkpoint_params(
            config.checkpoint_path, initialized_params=model, mesh=mesh
        )
        condition_losses = _evaluate_components(config, policy.cast_to_compute(params), mesh)
    if jax.process_index() != 0:
        tracker.finish()
        return {}
    paired = pair_mrcr_condition_losses(condition_losses)
    metrics = derive_mrcr_metrics(
        paired, bootstrap_samples=config.bootstrap_samples, bootstrap_seed=config.bootstrap_seed
    )
    levanter.tracker.log(metrics, step=checkpoint_step)
    persist_grug_checkpoint_eval(config, checkpoint_step=checkpoint_step, paired_rows=paired, metrics=metrics)
    tracker.finish()
    return metrics


def dispatch_grug_checkpoint_eval(
    config: GrugCheckpointEvalConfig,
    *,
    resources: ResourceConfig,
    processes_per_task: int = 1,
) -> None:
    """Dispatch ``evaluate_grug_checkpoint`` through the June Grug Fray runner."""
    if processes_per_task != 1:
        raise ValueError("the current Fray callable runner requires processes_per_task=1")
    dispatch_grug_training_run(
        run_id=f"eval-{config.run_id}",
        config=config,
        local_entrypoint=evaluate_grug_checkpoint,
        resources=resources,
    )
