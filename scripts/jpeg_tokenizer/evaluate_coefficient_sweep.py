#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate coefficient-sweep checkpoints with image-level and shared-prefix losses."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

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
from experiments.jpeg_tokenizer.base.data import materialize_token_store, read_token_store_metadata
from experiments.jpeg_tokenizer.base.eval import coefficient_prefix_loss_mask, summarize_metric
from experiments.jpeg_tokenizer.base.model import JPEG_TOKENIZER_V0_MODEL
from levanter.checkpoint import load_checkpoint
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SweepRunSpec:
    """One coefficient-sweep checkpoint to evaluate."""

    name: str
    checkpoint: str
    token_store: str
    seq_len: int
    tokens_per_block: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-spec",
        action="append",
        default=[],
        help=(
            "Comma-separated key=value pairs describing one run. "
            "Required keys: name,checkpoint,store,seq_len,tokens_per_block."
        ),
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--shared-prefixes", default="4,8")
    parser.add_argument("--output-dir", default="artifacts/jpeg_tokenizer/analysis/coefficient_sweep_sequence_loss")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def parse_run_spec(text: str) -> SweepRunSpec:
    fields: dict[str, str] = {}
    for part in text.split(","):
        key, sep, value = part.partition("=")
        if not sep:
            raise ValueError(f"Malformed run spec fragment {part!r}; expected key=value")
        fields[key.strip()] = value.strip()

    required = {"name", "checkpoint", "store", "seq_len", "tokens_per_block"}
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"Run spec is missing required keys: {missing}")

    return SweepRunSpec(
        name=fields["name"],
        checkpoint=fields["checkpoint"],
        token_store=fields["store"],
        seq_len=int(fields["seq_len"]),
        tokens_per_block=int(fields["tokens_per_block"]),
    )


def _iter_batches(num_examples: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, num_examples, batch_size):
        yield start, min(start + batch_size, num_examples)


def _pad_batch(batch: np.ndarray, target_batch_size: int) -> tuple[np.ndarray, int]:
    """Pad a token batch to the requested batch size by repeating the final example."""

    actual_batch_size = int(batch.shape[0])
    if actual_batch_size <= 0:
        raise ValueError("Expected a non-empty batch")
    if actual_batch_size > target_batch_size:
        raise ValueError(f"Batch size {actual_batch_size} exceeds target batch size {target_batch_size}")
    if actual_batch_size == target_batch_size:
        return batch, actual_batch_size

    pad_rows = target_batch_size - actual_batch_size
    padded = np.concatenate([batch, np.repeat(batch[-1:, :], pad_rows, axis=0)], axis=0)
    return padded, actual_batch_size


def _metric_summary(values_nats: np.ndarray, *, blocks_per_image: int) -> dict[str, object]:
    bits_per_image = values_nats / math.log(2.0)
    bits_per_block = bits_per_image / blocks_per_image
    return {
        "nats_per_image": summarize_metric(values_nats.tolist()).to_dict(),
        "bits_per_image": summarize_metric(bits_per_image.tolist()).to_dict(),
        "bits_per_block": summarize_metric(bits_per_block.tolist()).to_dict(),
    }


def evaluate_run(
    spec: SweepRunSpec,
    *,
    trainer: TrainerConfig,
    split: str,
    batch_size: int,
    shared_prefixes: list[int],
    max_examples: int | None,
    log_every: int,
) -> dict[str, object]:
    local_store = materialize_token_store(spec.token_store)
    metadata = read_token_store_metadata(local_store)
    split_info = metadata.splits[split]
    if split_info.seq_len != spec.seq_len:
        raise ValueError(f"Run {spec.name} expected seq_len={spec.seq_len}, found {split_info.seq_len} in token store")

    tokens = np.load(Path(local_store) / split_info.tokens_path, mmap_mode="r")
    if max_examples is not None:
        tokens = tokens[:max_examples]

    num_examples = int(tokens.shape[0])
    blocks_per_image = spec.seq_len // spec.tokens_per_block
    causal_loss_mask = np.zeros(spec.seq_len, dtype=np.float32)
    causal_loss_mask[:-1] = 1.0

    prefix_masks = {
        prefix: coefficient_prefix_loss_mask(
            spec.seq_len,
            tokens_per_block=spec.tokens_per_block,
            prefix_tokens_per_block=prefix,
        )
        for prefix in shared_prefixes
        if prefix <= spec.tokens_per_block
    }

    with trainer.use_device_mesh():
        mesh = trainer.device_mesh
        batch_axis = trainer.compute_axis_mapping.get(trainer.batch_axis_name, trainer.compute_axis_mapping.get("batch"))
        batch_sharding = NamedSharding(mesh, P(batch_axis, None))

        model_config = dataclasses.replace(JPEG_TOKENIZER_V0_MODEL, max_seq_len=spec.seq_len)
        with use_cpu_device():
            model = eqx.filter_eval_shape(Transformer.init, model_config, key=jax.random.PRNGKey(0))
            model = load_checkpoint(model, spec.checkpoint, subpath="train_state/params")

        model = hax.shard_with_axis_mapping(model, trainer.parameter_axis_mapping)

        causal_loss_mask_jax = jnp.asarray(causal_loss_mask, dtype=jnp.float32)
        prefix_masks_jax = {prefix: jnp.asarray(mask, dtype=jnp.float32) for prefix, mask in prefix_masks.items()}

        @named_jit(axis_resources=trainer.compute_axis_mapping)
        def eval_batch(model: Transformer, token_batch: jax.Array) -> dict[str, jax.Array]:
            model = trainer.mp.cast_to_compute(model)
            per_pos_loss = model.next_token_loss(
                token_batch,
                jnp.broadcast_to(causal_loss_mask_jax, token_batch.shape),
                mask=GrugAttentionMask.causal(),
                reduction="none",
                logsumexp_weight=None,
            )
            metrics: dict[str, jax.Array] = {
                "sequence_nats": jnp.sum(per_pos_loss, axis=1),
            }
            for prefix, mask in prefix_masks_jax.items():
                metrics[f"prefix_{prefix}_nats"] = jnp.sum(per_pos_loss * mask[None, :], axis=1)
            return metrics

        collected: dict[str, list[np.ndarray]] = {"sequence_nats": []}
        for prefix in prefix_masks:
            collected[f"prefix_{prefix}_nats"] = []

        for batch_index, (start, end) in enumerate(_iter_batches(num_examples, batch_size)):
            batch = np.asarray(tokens[start:end], dtype=np.int32)
            batch, actual_batch_size = _pad_batch(batch, batch_size)
            outputs = eval_batch(model, jax.device_put(batch, batch_sharding))
            outputs = jax.device_get(outputs)
            for name, values in outputs.items():
                collected[name].append(np.asarray(values[:actual_batch_size], dtype=np.float64))
            if (batch_index + 1) % log_every == 0:
                logger.info("Run %s processed %s/%s examples", spec.name, end, num_examples)

    result = {
        "name": spec.name,
        "checkpoint": spec.checkpoint,
        "token_store": spec.token_store,
        "seq_len": spec.seq_len,
        "tokens_per_block": spec.tokens_per_block,
        "num_examples": num_examples,
        "blocks_per_image": blocks_per_image,
        "metrics": {},
    }
    sequence_nats = np.concatenate(collected["sequence_nats"])
    result["metrics"]["sequence"] = {
        "selected_loss_positions_per_image": int(causal_loss_mask.sum()),
        **_metric_summary(sequence_nats, blocks_per_image=blocks_per_image),
    }
    for prefix in sorted(prefix_masks):
        prefix_nats = np.concatenate(collected[f"prefix_{prefix}_nats"])
        result["metrics"][f"prefix_{prefix}"] = {
            "selected_loss_positions_per_image": int(prefix_masks[prefix].sum()),
            **_metric_summary(prefix_nats, blocks_per_image=blocks_per_image),
        }

    return result


def render_summary(results: list[dict[str, object]]) -> str:
    lines = [
        "# Coefficient Sweep Sequence-Level Evaluation",
        "",
        "Sequence/image-level metrics are computed from per-image total NLL, not mean token loss.",
        "",
        "## Runs",
        "",
    ]

    for result in results:
        metrics = result["metrics"]
        sequence = metrics["sequence"]
        lines.append(
            f"- `{result['name']}`: seq_len={result['seq_len']}, "
            f"tokens_per_block={result['tokens_per_block']}, "
            f"mean bits/image={sequence['bits_per_image']['mean']:.2f}, "
            f"mean bits/block={sequence['bits_per_block']['mean']:.4f}"
        )

    lines.extend(["", "## Shared Prefixes", ""])
    prefix_names = sorted({name for result in results for name in result["metrics"] if name.startswith("prefix_")})
    for prefix_name in prefix_names:
        lines.append(f"### `{prefix_name}`")
        lines.append("")
        for result in results:
            prefix_metrics = result["metrics"].get(prefix_name)
            if prefix_metrics is None:
                continue
            lines.append(
                f"- `{result['name']}`: mean bits/image={prefix_metrics['bits_per_image']['mean']:.2f}, "
                f"median bits/image={prefix_metrics['bits_per_image']['median']:.2f}, "
                f"p95 bits/image={prefix_metrics['bits_per_image']['p95']:.2f}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


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


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.run_spec:
        raise ValueError("At least one --run-spec is required")

    shared_prefixes = [int(value) for value in args.shared_prefixes.split(",") if value]
    run_specs = [parse_run_spec(text) for text in args.run_spec]
    output_dir = args.output_dir.rstrip("/")
    trainer = TrainerConfig(
        id="jpeg-tokenizer-sequence-eval",
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
                shared_prefixes=shared_prefixes,
                max_examples=args.max_examples,
                log_every=args.log_every,
            )
        )

    _write_json(f"{output_dir}/sequence_eval.json", {"runs": results})
    _write_text(f"{output_dir}/summary.md", render_summary(results))

    jax.effects_barrier()
    logger.info("%s", render_summary(results))
    logger.info("Wrote sweep evaluation to %s", output_dir)


if __name__ == "__main__":
    main(parse_args())
