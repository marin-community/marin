# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Stage 0 gate's routing-health check on ``T`` for a Snowball checkpoint.

The gate (DESIGN, "Stage 0 gate", relative form of 2026-10-24): in no layer may the share of
experts below 10% of that layer's mean load on ``T`` exceed the pretrained base's share in the
same layer by more than 2 percentage points. Load is the number of tokens a layer routes to an expert
(top-k selections on the biased router scores). ``T`` documents are packed into fixed
windows with segment ids, so no padding is routed.

Run on one 8-GPU node::

    python -m experiments.post_training.stage0.routing_health \\
        --checkpoint s3://.../hf/step-N --t-file s3://.../T.jsonl --out s3://.../routing.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import equinox as eqx
import fsspec
import jax.numpy as jnp
import jmp
import numpy as np
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef, load_tokenizer
from levanter.grug.attention import AttentionMask
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

logger = logging.getLogger(__name__)

LOW_LOAD_FRACTION_OF_MEAN = 0.10
MAX_LOW_LOAD_EXPERT_SHARE = 0.05
# The relative gate: B may starve at most this much more of a layer's experts than the base.
MAX_LOW_LOAD_SHARE_INCREASE = 0.02
WINDOW_TOKENS = 4096
WINDOWS_PER_BATCH = 8


@dataclass(frozen=True)
class LayerHealth:
    layer: int
    low_load_experts: int
    low_load_share: float
    min_over_mean: float
    max_over_mean: float


def layer_health(loads: np.ndarray) -> list[LayerHealth]:
    """Per layer: experts below 10% of the layer's mean load, and the load spread."""
    health = []
    for layer, row in enumerate(np.asarray(loads, dtype=np.float64)):
        mean = row.mean()
        low = int((row < LOW_LOAD_FRACTION_OF_MEAN * mean).sum())
        health.append(LayerHealth(layer, low, low / row.size, float(row.min() / mean), float(row.max() / mean)))
    return health


def routing_gate(loads: np.ndarray) -> dict:
    """The gate verdict and the per-layer table it was decided on."""
    health = layer_health(loads)
    worst = max(health, key=lambda entry: entry.low_load_share)
    return {
        "passed": worst.low_load_share <= MAX_LOW_LOAD_EXPERT_SHARE,
        "worst_layer": worst.layer,
        "worst_low_load_share": worst.low_load_share,
        "threshold": {"low_load_fraction_of_mean": LOW_LOAD_FRACTION_OF_MEAN, "max_share": MAX_LOW_LOAD_EXPERT_SHARE},
        "layers": [entry.__dict__ for entry in health],
    }


def relative_routing_gate(loads: np.ndarray, base_loads: np.ndarray) -> dict:
    """The relative verdict: each layer's low-load share against the base's in that layer."""
    candidate, base = layer_health(loads), layer_health(base_loads)
    if len(candidate) != len(base):
        raise ValueError("the candidate and the base must have the same layers")
    deltas = [
        {
            "layer": c.layer,
            "share": c.low_load_share,
            "base_share": b.low_load_share,
            "increase": c.low_load_share - b.low_load_share,
        }
        for c, b in zip(candidate, base, strict=True)
    ]
    worst = max(deltas, key=lambda entry: entry["increase"])
    return {
        "passed": worst["increase"] <= MAX_LOW_LOAD_SHARE_INCREASE + 1e-12,
        "worst_layer": worst["layer"],
        "worst_increase": worst["increase"],
        "threshold": {
            "low_load_fraction_of_mean": LOW_LOAD_FRACTION_OF_MEAN,
            "max_increase": MAX_LOW_LOAD_SHARE_INCREASE,
        },
        "layers": deltas,
    }


def packed_windows(documents: Iterable[list[int]], window: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Pack token lists into ``window``-token rows with per-document segment ids.

    A document longer than the window continues in the next row as a new segment; the last
    partial row is dropped, so every routed token is a real token.
    """
    tokens: list[int] = []
    segments: list[int] = []
    for segment, document in enumerate(documents):
        tokens.extend(document)
        segments.extend([segment] * len(document))
        while len(tokens) >= window:
            yield np.asarray(tokens[:window], dtype=np.int32), np.asarray(segments[:window], dtype=np.int32)
            tokens, segments = tokens[window:], segments[window:]


def _t_documents(path: str, tokenizer) -> Iterator[list[int]]:
    with fsspec.open(path, "rt") as handle:
        for line in handle:
            yield tokenizer.encode(json.loads(line)["text"], add_special_tokens=False)


def checkpoint_loads(checkpoint: str, t_file: str, *, max_tokens: int | None) -> tuple[np.ndarray, int]:
    """Summed ``[layers, experts]`` loads of ``checkpoint`` over ``t_file``, and the token count.

    ``max_tokens`` stops early (pipeline checks only; the gate reads all of ``T``).
    """
    ref = RepoRef.from_string(checkpoint)
    converter = HFCheckpointConverter.from_hf(ref)
    config = converter.default_config
    trainer = TrainerConfig(
        mp=jmp.get_policy("p=bfloat16,c=bfloat16"),
        mesh=MeshConfig(axes={"data": 1, "replica": 1, "model": 1, "expert": -1}),
        use_explicit_mesh_axes=config.requires_explicit_mesh_axes,
    )
    tokenizer = load_tokenizer(ref.model_name_or_path, revision=ref.revision)
    total = None
    tokens = 0
    with trainer.use_device_mesh():
        model = converter.load_pretrained(
            config.model_type,
            ref=ref,
            config=config,
            dtype=jnp.bfloat16,
            axis_mapping=trainer.parameter_axis_mapping,
        )
        loads_fn = eqx.filter_jit(
            lambda transformer, ids, segments: transformer.expert_loads(
                ids, AttentionMask(is_causal=True, segment_ids=(segments, segments))
            )
        )
        batch: list[tuple[np.ndarray, np.ndarray]] = []
        for row in packed_windows(_t_documents(t_file, tokenizer), WINDOW_TOKENS):
            batch.append(row)
            if len(batch) < WINDOWS_PER_BATCH:
                continue
            ids = jnp.asarray(np.stack([ids for ids, _ in batch]))
            segments = jnp.asarray(np.stack([seg for _, seg in batch]))
            loads = np.asarray(loads_fn(model.transformer, ids, segments), dtype=np.int64)
            total = loads if total is None else total + loads
            tokens += ids.size
            batch = []
            logger.info("routed %d tokens", tokens)
            if max_tokens is not None and tokens >= max_tokens:
                break
    if total is None:
        raise ValueError(f"{t_file} holds fewer than {WINDOW_TOKENS * WINDOWS_PER_BATCH} tokens")
    return total, tokens


def compare_reports(candidate_path: str, base_path: str, out: str) -> dict:
    """Apply the relative gate to two stored routing reports (each holds its ``loads``)."""
    reports = []
    for path in (candidate_path, base_path):
        with fsspec.open(path, "r") as handle:
            reports.append(json.load(handle))
    candidate, base = reports
    if candidate["t_file"] != base["t_file"] or candidate["tokens"] != base["tokens"]:
        raise ValueError("the candidate and the base must be measured on the same T tokens")
    verdict = {
        "candidate": candidate_path,
        "base": base_path,
        **relative_routing_gate(np.asarray(candidate["loads"]), np.asarray(base["loads"])),
    }
    with fsspec.open(out, "w") as handle:
        json.dump(verdict, handle, indent=2)
    return verdict


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", nargs=2, metavar=("CANDIDATE_JSON", "BASE_JSON"), help="relative gate only")
    parser.add_argument("--checkpoint")
    parser.add_argument("--t-file")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=None, help="stop early; for pipeline checks only")
    args = parser.parse_args()
    if args.compare:
        verdict = compare_reports(*args.compare, args.out)
        logger.info(
            "relative routing gate %s: worst layer %d (+%.3f)",
            "PASSED" if verdict["passed"] else "FAILED",
            verdict["worst_layer"],
            verdict["worst_increase"],
        )
        return
    if args.checkpoint is None or args.t_file is None:
        parser.error("--checkpoint and --t-file are required unless --compare is given")
    loads, tokens = checkpoint_loads(args.checkpoint, args.t_file, max_tokens=args.max_tokens)
    report = {
        "checkpoint": args.checkpoint,
        "t_file": args.t_file,
        "tokens": tokens,
        "complete": args.max_tokens is None,
        **routing_gate(loads),
    }
    report["loads"] = loads.tolist()
    with fsspec.open(args.out, "w") as handle:
        json.dump(report, handle)
    logger.info("routing gate %s: worst layer %d", "PASSED" if report["passed"] else "FAILED", report["worst_layer"])


if __name__ == "__main__":
    main()
