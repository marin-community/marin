#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate local corruption sensitivity for JPEG representation checkpoints."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass

import equinox as eqx
import fsspec
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from haliax import named_jit
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.base.model import Transformer
from experiments.jpeg_tokenizer.base.data import (
    materialize_token_store,
    open_token_matrix_dataset,
    read_token_store_manifest,
    read_token_store_metadata,
)
from experiments.jpeg_tokenizer.base.eval import causal_loss_mask_from_lengths, summarize_metric
from experiments.jpeg_tokenizer.base.model import JPEG_TOKENIZER_V0_MODEL
from levanter.checkpoint import load_checkpoint
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepresentationRunSpec:
    """One checkpoint/store pair to evaluate."""

    name: str
    checkpoint: str
    token_store: str
    sliding_window: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-spec",
        action="append",
        default=[],
        help=(
            "Comma-separated key=value pairs describing one run. "
            "Required keys: name,checkpoint,store. Optional: sliding_window."
        ),
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=512)
    parser.add_argument("--perturb-fractions", default="0.5")
    parser.add_argument("--horizons", default="1,64,512,4096")
    parser.add_argument(
        "--output-dir",
        default="artifacts/jpeg_tokenizer/analysis/representation_perturbation",
    )
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def parse_run_spec(text: str) -> RepresentationRunSpec:
    fields: dict[str, str] = {}
    for part in text.split(","):
        key, sep, value = part.partition("=")
        if not sep:
            raise ValueError(f"Malformed run spec fragment {part!r}; expected key=value")
        fields[key.strip()] = value.strip()

    required = {"name", "checkpoint", "store"}
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"Run spec is missing required keys: {missing}")

    return RepresentationRunSpec(
        name=fields["name"],
        checkpoint=fields["checkpoint"],
        token_store=fields["store"],
        sliding_window=int(fields["sliding_window"]) if fields.get("sliding_window") else None,
    )


def _iter_batches(num_examples: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, num_examples, batch_size):
        yield start, min(start + batch_size, num_examples)


def _pad_batch(batch: np.ndarray, lengths: np.ndarray, target_batch_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Pad a token batch to the requested batch size by repeating the final example."""

    actual_batch_size = int(batch.shape[0])
    if actual_batch_size <= 0:
        raise ValueError("Expected a non-empty batch")
    if actual_batch_size > target_batch_size:
        raise ValueError(f"Batch size {actual_batch_size} exceeds target batch size {target_batch_size}")
    if actual_batch_size == target_batch_size:
        return batch, lengths, actual_batch_size

    pad_rows = target_batch_size - actual_batch_size
    padded_batch = np.concatenate([batch, np.repeat(batch[-1:, :], pad_rows, axis=0)], axis=0)
    padded_lengths = np.concatenate([lengths, np.repeat(lengths[-1:], pad_rows, axis=0)], axis=0)
    return padded_batch, padded_lengths, actual_batch_size


def _write_json(path: str, payload: object) -> None:
    fs, fs_path = fsspec.core.url_to_fs(path)
    parent = fs_path.rsplit("/", 1)[0]
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(fs_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_text(path: str, text: str) -> None:
    fs, fs_path = fsspec.core.url_to_fs(path)
    parent = fs_path.rsplit("/", 1)[0]
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(fs_path, "w") as handle:
        handle.write(text)


def _forbidden_replacement_ids(metadata_tokenizer_config: dict[str, object]) -> set[int]:
    forbidden: set[int] = set()
    for key in ("pad_token_id", "loss_mask_ignore_id", "eos_token_id"):
        value = metadata_tokenizer_config.get(key)
        if isinstance(value, int):
            forbidden.add(value)
    return forbidden


def _replacement_token_id(original: int, *, vocab_size: int, forbidden_ids: set[int]) -> int:
    if vocab_size <= 1:
        raise ValueError(f"vocab_size must be > 1, got {vocab_size}")
    candidate = (original + 1) % vocab_size
    for _ in range(vocab_size):
        if candidate != original and candidate not in forbidden_ids:
            return candidate
        candidate = (candidate + 1) % vocab_size
    raise ValueError("No valid replacement token id found")


def _perturb_positions(lengths: np.ndarray, *, fraction: float) -> np.ndarray:
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError(f"perturb fraction must be in [0, 1], got {fraction}")
    max_source_positions = np.maximum(lengths - 2, 0)
    return np.floor(max_source_positions.astype(np.float64) * fraction).astype(np.int32)


def _summary(values_nats: np.ndarray) -> dict[str, object]:
    values_bits = values_nats / math.log(2.0)
    return {
        "nats_per_image": summarize_metric(values_nats.tolist()).to_dict(),
        "bits_per_image": summarize_metric(values_bits.tolist()).to_dict(),
    }


def _build_horizon_mask(
    lengths: np.ndarray,
    perturb_positions: np.ndarray,
    *,
    seq_len: int,
    horizon: int,
    include_immediate: bool,
) -> np.ndarray:
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    positions = np.arange(seq_len, dtype=np.int32)[None, :]
    lower = perturb_positions[:, None] + (0 if include_immediate else 1)
    upper = perturb_positions[:, None] + horizon
    valid = positions < (lengths[:, None] - 1)
    return valid & (positions >= lower) & (positions < upper)


def evaluate_run(
    spec: RepresentationRunSpec,
    *,
    trainer: TrainerConfig,
    split: str,
    batch_size: int,
    max_examples: int | None,
    perturb_fractions: list[float],
    horizons: list[int],
    log_every: int,
) -> dict[str, object]:
    local_store = materialize_token_store(spec.token_store)
    metadata = read_token_store_metadata(local_store)
    records = read_token_store_manifest(local_store, split)
    if max_examples is not None:
        records = records[:max_examples]

    dataset = open_token_matrix_dataset(local_store, split)
    tokens = dataset._tokens
    tokens = tokens[: len(records)]

    num_examples = int(tokens.shape[0])
    actual_token_lengths = np.asarray([record.num_tokens for record in records], dtype=np.int32)
    modeled_token_lengths = actual_token_lengths - 1
    forbidden_replacement_ids = _forbidden_replacement_ids(metadata.tokenizer_config)

    with trainer.use_device_mesh():
        mesh = trainer.device_mesh
        batch_axis = trainer.compute_axis_mapping.get(trainer.batch_axis_name, trainer.compute_axis_mapping.get("batch"))
        batch_sharding = NamedSharding(mesh, P(batch_axis, None))

        model_config = dataclasses.replace(
            JPEG_TOKENIZER_V0_MODEL,
            vocab_size=metadata.vocab_size,
            max_seq_len=metadata.seq_len,
            sliding_window=spec.sliding_window,
        )
        with use_cpu_device():
            model = eqx.filter_eval_shape(Transformer.init, model_config, key=jax.random.PRNGKey(0))
            model = load_checkpoint(model, spec.checkpoint, subpath="train_state/params")

        model = hax.shard_with_axis_mapping(model, trainer.parameter_axis_mapping)

        @named_jit(axis_resources=trainer.compute_axis_mapping)
        def eval_batch(
            model: Transformer,
            token_batch: jax.Array,
            loss_mask_batch: jax.Array,
        ) -> jax.Array:
            model = trainer.mp.cast_to_compute(model)
            return model.next_token_loss(
                token_batch,
                loss_mask_batch,
                mask=GrugAttentionMask.causal(),
                reduction="none",
                logsumexp_weight=None,
            )

        clean_sequence_nats: list[np.ndarray] = []
        perturbation_metrics: dict[float, dict[str, list[np.ndarray]]] = {
            fraction: {"delta_total_nats": [], "delta_immediate_nats": []} for fraction in perturb_fractions
        }
        for fraction in perturb_fractions:
            for horizon in horizons:
                perturbation_metrics[fraction][f"delta_h{horizon}_nats"] = []
                perturbation_metrics[fraction][f"delta_tail_h{horizon}_nats"] = []

        for batch_index, (start, end) in enumerate(_iter_batches(num_examples, batch_size)):
            batch = np.asarray(tokens[start:end], dtype=np.int32)
            batch_lengths = actual_token_lengths[start:end]
            batch, batch_lengths, actual_batch_size = _pad_batch(batch, batch_lengths, batch_size)
            loss_mask = causal_loss_mask_from_lengths(batch_lengths.tolist(), seq_len=metadata.seq_len)

            clean_per_pos_loss = jax.device_get(
                eval_batch(
                    model,
                    jax.device_put(batch, batch_sharding),
                    jax.device_put(jnp.asarray(loss_mask, dtype=jnp.float32), batch_sharding),
                )
            )
            clean_per_pos_loss = np.asarray(clean_per_pos_loss, dtype=np.float64)
            clean_sequence_nats.append(np.sum(clean_per_pos_loss[:actual_batch_size], axis=1, dtype=np.float64))

            for fraction in perturb_fractions:
                perturb_positions = _perturb_positions(batch_lengths, fraction=fraction)
                perturbed_batch = np.array(batch, copy=True)
                for row_index, perturb_position in enumerate(perturb_positions.tolist()):
                    original_token = int(perturbed_batch[row_index, perturb_position])
                    perturbed_batch[row_index, perturb_position] = _replacement_token_id(
                        original_token,
                        vocab_size=metadata.vocab_size,
                        forbidden_ids=forbidden_replacement_ids,
                    )

                perturbed_per_pos_loss = jax.device_get(
                    eval_batch(
                        model,
                        jax.device_put(perturbed_batch, batch_sharding),
                        jax.device_put(jnp.asarray(loss_mask, dtype=jnp.float32), batch_sharding),
                    )
                )
                perturbed_per_pos_loss = np.asarray(perturbed_per_pos_loss, dtype=np.float64)
                delta_per_pos_loss = perturbed_per_pos_loss - clean_per_pos_loss
                delta_per_pos_loss = delta_per_pos_loss[:actual_batch_size]
                batch_lengths_actual = batch_lengths[:actual_batch_size]
                perturb_positions_actual = perturb_positions[:actual_batch_size]

                perturbation_metrics[fraction]["delta_total_nats"].append(
                    np.sum(delta_per_pos_loss, axis=1, dtype=np.float64)
                )

                immediate_mask = _build_horizon_mask(
                    batch_lengths_actual,
                    perturb_positions_actual,
                    seq_len=metadata.seq_len,
                    horizon=1,
                    include_immediate=True,
                )
                perturbation_metrics[fraction]["delta_immediate_nats"].append(
                    np.sum(delta_per_pos_loss * immediate_mask, axis=1, dtype=np.float64)
                )

                for horizon in horizons:
                    inclusive_mask = _build_horizon_mask(
                        batch_lengths_actual,
                        perturb_positions_actual,
                        seq_len=metadata.seq_len,
                        horizon=horizon,
                        include_immediate=True,
                    )
                    tail_mask = _build_horizon_mask(
                        batch_lengths_actual,
                        perturb_positions_actual,
                        seq_len=metadata.seq_len,
                        horizon=horizon,
                        include_immediate=False,
                    )
                    perturbation_metrics[fraction][f"delta_h{horizon}_nats"].append(
                        np.sum(delta_per_pos_loss * inclusive_mask, axis=1, dtype=np.float64)
                    )
                    perturbation_metrics[fraction][f"delta_tail_h{horizon}_nats"].append(
                        np.sum(delta_per_pos_loss * tail_mask, axis=1, dtype=np.float64)
                    )

            if (batch_index + 1) % log_every == 0:
                logger.info("Run %s processed %s/%s examples", spec.name, end, num_examples)

    clean_nats = np.concatenate(clean_sequence_nats)
    result: dict[str, object] = {
        "name": spec.name,
        "checkpoint": spec.checkpoint,
        "token_store": spec.token_store,
        "seq_len": metadata.seq_len,
        "vocab_size": metadata.vocab_size,
        "num_examples": num_examples,
        "sliding_window": spec.sliding_window,
        "actual_tokens_per_image": summarize_metric(actual_token_lengths.astype(np.float64).tolist()).to_dict(),
        "modeled_tokens_per_image": summarize_metric(modeled_token_lengths.astype(np.float64).tolist()).to_dict(),
        "clean": _summary(clean_nats),
        "perturbation": {},
    }
    for fraction in perturb_fractions:
        fraction_metrics = perturbation_metrics[fraction]
        fraction_result: dict[str, object] = {
            "delta_total": _summary(np.concatenate(fraction_metrics["delta_total_nats"])),
            "delta_immediate": _summary(np.concatenate(fraction_metrics["delta_immediate_nats"])),
        }
        for horizon in horizons:
            fraction_result[f"delta_h{horizon}"] = _summary(np.concatenate(fraction_metrics[f"delta_h{horizon}_nats"]))
            fraction_result[f"delta_tail_h{horizon}"] = _summary(
                np.concatenate(fraction_metrics[f"delta_tail_h{horizon}_nats"])
            )
        result["perturbation"][str(fraction)] = fraction_result

    return result


def _render_summary(results: list[dict[str, object]], *, perturb_fractions: list[float], horizons: list[int]) -> str:
    lines = [
        "# JPEG Representation Perturbation Sensitivity",
        "",
        "Single-token corruption per image; all values are sequence-level bits/image deltas.",
        "",
        "## Clean Whole-Image Loss",
        "",
    ]
    for run in results:
        clean_bits = run["clean"]["bits_per_image"]["mean"]  # type: ignore[index]
        lines.append(f"- `{run['name']}`: mean bits/image = {clean_bits:.2f}")

    for fraction in perturb_fractions:
        lines.extend(["", f"## Perturbation Fraction {fraction:.2f}", ""])
        header = "| Run | Delta Total | Delta Immediate |"
        separator = "| --- | ---: | ---: |"
        lines.extend([header, separator])
        for run in results:
            metrics = run["perturbation"][str(fraction)]  # type: ignore[index]
            total_bits = metrics["delta_total"]["bits_per_image"]["mean"]  # type: ignore[index]
            immediate_bits = metrics["delta_immediate"]["bits_per_image"]["mean"]  # type: ignore[index]
            lines.append(f"| `{run['name']}` | {total_bits:.2f} | {immediate_bits:.2f} |")

        lines.extend(["", "### Horizon Deltas", ""])
        horizon_header = "| Run | " + " | ".join([f"Delta h{h}" for h in horizons]) + " |"
        horizon_separator = "| --- | " + " | ".join(["---:"] * len(horizons)) + " |"
        lines.extend([horizon_header, horizon_separator])
        for run in results:
            metrics = run["perturbation"][str(fraction)]  # type: ignore[index]
            cells: list[str] = []
            for horizon in horizons:
                cell = metrics[f"delta_h{horizon}"]["bits_per_image"]["mean"]  # type: ignore[index]
                cells.append(f"{cell:.2f}")
            lines.append(f"| `{run['name']}` | " + " | ".join(cells) + " |")

        lines.extend(["", "### Tail-Only Horizon Deltas", ""])
        tail_header = "| Run | " + " | ".join([f"Tail h{h}" for h in horizons]) + " |"
        tail_separator = "| --- | " + " | ".join(["---:"] * len(horizons)) + " |"
        lines.extend([tail_header, tail_separator])
        for run in results:
            metrics = run["perturbation"][str(fraction)]  # type: ignore[index]
            cells = []
            for horizon in horizons:
                cell = metrics[f"delta_tail_h{horizon}"]["bits_per_image"]["mean"]  # type: ignore[index]
                cells.append(f"{cell:.2f}")
            lines.append(f"| `{run['name']}` | " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def main(args: argparse.Namespace | None = None) -> None:
    parsed = parse_args() if args is None else args
    run_specs = [parse_run_spec(text) for text in parsed.run_spec]
    if not run_specs:
        raise ValueError("At least one --run-spec is required")

    perturb_fractions = [float(value) for value in parsed.perturb_fractions.split(",") if value]
    if not perturb_fractions:
        raise ValueError("At least one perturb fraction is required")
    horizons = [int(value) for value in parsed.horizons.split(",") if value]
    if not horizons:
        raise ValueError("At least one horizon is required")

    trainer = TrainerConfig(
        id="jpeg-tokenizer-perturb-eval",
        tracker=NoopConfig(),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        train_batch_size=parsed.batch_size,
        log_jaxprs=False,
        log_xla_hlo=False,
        shutdown_at_exit=False,
    )
    trainer.initialize()

    results: list[dict[str, object]] = []
    for spec in run_specs:
        logger.info("Evaluating %s", spec.name)
        results.append(
            evaluate_run(
                spec,
                trainer=trainer,
                split=parsed.split,
                batch_size=parsed.batch_size,
                max_examples=parsed.max_examples,
                perturb_fractions=perturb_fractions,
                horizons=horizons,
                log_every=parsed.log_every,
            )
        )

    payload = {
        "split": parsed.split,
        "batch_size": parsed.batch_size,
        "max_examples": parsed.max_examples,
        "perturb_fractions": perturb_fractions,
        "horizons": horizons,
        "runs": results,
    }
    _write_json(f"{parsed.output_dir.rstrip('/')}/perturbation_eval.json", payload)
    _write_text(
        f"{parsed.output_dir.rstrip('/')}/summary.md",
        _render_summary(results, perturb_fractions=perturb_fractions, horizons=horizons),
    )
    logger.info("Wrote perturbation evaluation outputs to %s", parsed.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
