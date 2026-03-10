#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate JPEG representation checkpoints with exact per-image sequence losses."""

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
import jax
import jax.numpy as jnp
import numpy as np
import haliax as hax
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
    normalization_unit_name: str | None = None
    normalization_unit_count: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-spec",
        action="append",
        default=[],
        help=(
            "Comma-separated key=value pairs describing one run. "
            "Required keys: name,checkpoint,store. Optional: sliding_window,unit_name,unit_count."
        ),
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--pixels-per-image", type=int, default=256 * 256)
    parser.add_argument(
        "--output-dir",
        default="artifacts/jpeg_tokenizer/analysis/representation_head2head_sequence_loss",
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

    unit_name = fields.get("unit_name")
    unit_count = fields.get("unit_count")
    if (unit_name is None) != (unit_count is None):
        raise ValueError("unit_name and unit_count must either both be provided or both be omitted")

    return RepresentationRunSpec(
        name=fields["name"],
        checkpoint=fields["checkpoint"],
        token_store=fields["store"],
        sliding_window=int(fields["sliding_window"]) if fields.get("sliding_window") else None,
        normalization_unit_name=unit_name,
        normalization_unit_count=int(unit_count) if unit_count is not None else None,
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


def _metric_summary(values_nats: np.ndarray, *, modeled_tokens: np.ndarray, pixels_per_image: int) -> dict[str, object]:
    bits_per_image = values_nats / math.log(2.0)
    bits_per_modeled_token = bits_per_image / modeled_tokens
    bits_per_pixel = bits_per_image / pixels_per_image
    return {
        "nats_per_image": summarize_metric(values_nats.tolist()).to_dict(),
        "bits_per_image": summarize_metric(bits_per_image.tolist()).to_dict(),
        "bits_per_modeled_token": summarize_metric(bits_per_modeled_token.tolist()).to_dict(),
        "bits_per_pixel": summarize_metric(bits_per_pixel.tolist()).to_dict(),
    }


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


def evaluate_run(
    spec: RepresentationRunSpec,
    *,
    trainer: TrainerConfig,
    split: str,
    batch_size: int,
    pixels_per_image: int,
    max_examples: int | None,
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

        collected_sequence_nats: list[np.ndarray] = []
        for batch_index, (start, end) in enumerate(_iter_batches(num_examples, batch_size)):
            batch = np.asarray(tokens[start:end], dtype=np.int32)
            batch_lengths = actual_token_lengths[start:end]
            batch, batch_lengths, actual_batch_size = _pad_batch(batch, batch_lengths, batch_size)
            loss_mask = causal_loss_mask_from_lengths(batch_lengths.tolist(), seq_len=metadata.seq_len)
            per_pos_loss = jax.device_get(
                eval_batch(
                    model,
                    jax.device_put(batch, batch_sharding),
                    jax.device_put(jnp.asarray(loss_mask, dtype=jnp.float32), batch_sharding),
                )
            )
            collected_sequence_nats.append(
                np.asarray(np.sum(per_pos_loss[:actual_batch_size], axis=1), dtype=np.float64)
            )
            if (batch_index + 1) % log_every == 0:
                logger.info("Run %s processed %s/%s examples", spec.name, end, num_examples)

    sequence_nats = np.concatenate(collected_sequence_nats)
    result = {
        "name": spec.name,
        "checkpoint": spec.checkpoint,
        "token_store": spec.token_store,
        "seq_len": metadata.seq_len,
        "vocab_size": metadata.vocab_size,
        "num_examples": num_examples,
        "sliding_window": spec.sliding_window,
        "normalization_unit_name": spec.normalization_unit_name,
        "normalization_unit_count": spec.normalization_unit_count,
        "actual_tokens_per_image": summarize_metric(actual_token_lengths.astype(np.float64).tolist()).to_dict(),
        "modeled_tokens_per_image": summarize_metric(modeled_token_lengths.astype(np.float64).tolist()).to_dict(),
        "metrics": _metric_summary(
            sequence_nats,
            modeled_tokens=modeled_token_lengths.astype(np.float64),
            pixels_per_image=pixels_per_image,
        ),
    }
    if spec.normalization_unit_name and spec.normalization_unit_count:
        bits_per_image = sequence_nats / math.log(2.0)
        bits_per_unit = bits_per_image / spec.normalization_unit_count
        result["metrics"][f"bits_per_{spec.normalization_unit_name}"] = summarize_metric(
            bits_per_unit.tolist()
        ).to_dict()
    return result


def render_summary(results: list[dict[str, object]]) -> str:
    excluded_metrics = {"bits_per_image", "bits_per_pixel", "bits_per_modeled_token"}
    lines = [
        "# JPEG Representation Head-to-Head",
        "",
        "Exact per-image sequence losses computed from final checkpoints.",
        "",
        "## Runs",
        "",
    ]
    for result in results:
        metrics = result["metrics"]
        lines.append(
            f"- `{result['name']}`: seq_len={result['seq_len']}, "
            f"mean actual tokens/image={result['actual_tokens_per_image']['mean']:.2f}, "
            f"mean bits/image={metrics['bits_per_image']['mean']:.2f}, "
            f"mean bits/pixel={metrics['bits_per_pixel']['mean']:.4f}, "
            f"mean bits/modeled-token={metrics['bits_per_modeled_token']['mean']:.4f}"
        )
    lines.extend(["", "## Optional Unit-Normalized Metrics", ""])
    for result in results:
        optional_metrics: list[str] = []
        for key in result["metrics"]:
            if key.startswith("bits_per_") and key not in excluded_metrics:
                optional_metrics.append(key)
        if not optional_metrics:
            continue
        lines.append(f"### `{result['name']}`")
        lines.append("")
        for metric_name in sorted(optional_metrics):
            metric = result["metrics"][metric_name]
            lines.append(
                f"- `{metric_name}`: mean={metric['mean']:.4f}, median={metric['median']:.4f}, p95={metric['p95']:.4f}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.run_spec:
        raise ValueError("At least one --run-spec is required")

    run_specs = [parse_run_spec(text) for text in args.run_spec]
    trainer = TrainerConfig(
        id="jpeg-tokenizer-representation-eval",
        tracker=NoopConfig(),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        train_batch_size=args.batch_size,
        log_jaxprs=False,
        log_xla_hlo=False,
        shutdown_at_exit=False,
    )
    trainer.initialize()

    results = []
    for spec in run_specs:
        logger.info("Evaluating %s", spec.name)
        results.append(
            evaluate_run(
                spec,
                trainer=trainer,
                split=args.split,
                batch_size=args.batch_size,
                pixels_per_image=args.pixels_per_image,
                max_examples=args.max_examples,
                log_every=args.log_every,
            )
        )

    output_dir = args.output_dir.rstrip("/")
    _write_json(f"{output_dir}/representation_eval.json", {"runs": results})
    _write_text(f"{output_dir}/summary.md", render_summary(results))

    jax.effects_barrier()
    logger.info("%s", render_summary(results))
    logger.info("Wrote representation evaluation to %s", output_dir)


if __name__ == "__main__":
    main(parse_args())
