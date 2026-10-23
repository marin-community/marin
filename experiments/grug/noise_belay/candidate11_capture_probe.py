# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Isolate output-dependent capture differences on one GPU without optimizer state."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.attention import AttentionMask

from experiments.grug.fast_track.model import Transformer
from experiments.grug.noise_belay.candidate11 import ROOT, load_model_config
from experiments.grug.noise_belay.candidate11_forward import capture_forward
from experiments.grug.noise_belay.candidate11_gpu_probe import native_forward

GROUPS = (
    ("all", None),
    ("hidden_only", ()),
    ("stages", ("stages",)),
    ("embeddings", ("token_rows", "bigram_rows", "embedding", "bigram_source")),
    ("final", ("final_sources", "final_stream")),
    ("shared", ("shared_preactivations",)),
    (
        "routes",
        ("routing_selected", "routing_weights", "routing_logits", "routed_inputs"),
    ),
    ("weights_only", ("routing_weights",)),
    ("routed_inputs_only", ("routed_inputs",)),
    ("stage_outputs", ("stage:output",)),
    ("stage_partials", ("stage:partial",)),
    ("stage_inputs", ("stage:input",)),
    ("embedding_only", ("embedding",)),
    ("bigram_source_only", ("bigram_source",)),
    ("final_stream_only", ("final_stream",)),
    ("final_sources_only", ("final_sources",)),
)


def capture_subset(model, tokens, mask, *, fields):
    hidden, capture = capture_forward(model, tokens, mask)
    if fields is None:
        return hidden, capture
    return hidden, tuple(
        (
            tuple(getattr(stage, name.removeprefix("stage:")) for stage in capture.stages)
            if name.startswith("stage:")
            else getattr(capture, name)
        )
        for name in fields
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-scale", type=float, required=True)
    parser.add_argument("--groups", nargs="+", choices=[label for label, _ in GROUPS])
    args = parser.parse_args()
    manifest = json.loads((ROOT / "gpu-probe-manifest.json").read_text())
    for name, digest in manifest["experiment_files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Experiment source mismatch: {name}")
    cfg = load_model_config(source_sha256=manifest["model_source_sha256"])
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("Capture isolation requires one GPU")
    mesh = Mesh(
        np.asarray(devices).reshape(1, 1, 1, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    result = {
        "kind": "synthetic_capture_output_isolation",
        "manifest": manifest,
        "batch": 8,
        "sequence": 128,
        "query_scale": args.query_scale,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "comparisons": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with jax.set_mesh(mesh):
        began = time.perf_counter()
        model = eqx.filter_jit(Transformer.init)(cfg, key=jax.random.PRNGKey(0))
        paths, structure = jax.tree_util.tree_flatten_with_path(model)
        keys = jax.random.split(jax.random.PRNGKey(33), len(paths))
        model = jax.tree.unflatten(
            structure,
            [
                (
                    value + args.query_scale * jax.random.normal(key, value.shape)
                    if "attn_res_query" in jax.tree_util.keystr(path)
                    else value
                )
                for (path, value), key in zip(paths, keys, strict=True)
            ],
        )
        model = jmp.get_policy("bf16").cast_to_compute(model)
        jax.block_until_ready(model)
        result["initialization_seconds"] = time.perf_counter() - began
        result["parameter_count"] = sum(x.size for x in jax.tree.leaves(model))
        spec = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
        rng = np.random.default_rng(7)
        tokens = jax.device_put(rng.integers(0, cfg.vocab_size, (8, 128), dtype=np.int32), spec)
        segments = jax.device_put(np.broadcast_to(np.arange(128, dtype=np.int32) // 64, (8, 128)), spec)
        mask = AttentionMask(is_causal=True, segment_ids=(segments, segments))
        native_executable = eqx.filter_jit(native_forward)
        expected = native_executable(model, tokens, mask)
        jax.block_until_ready(expected)
        repeated = native_executable(model, tokens, mask)
        jax.block_until_ready(repeated)
        repeat_delta = jnp.abs(repeated.astype(jnp.float32) - expected.astype(jnp.float32))
        result["native_repeat_max_error"] = float(jnp.max(repeat_delta))
        result["native_repeat_fraction_outside_tolerance"] = float(
            jnp.mean(repeat_delta > 1e-5 + 1e-5 * jnp.abs(expected))
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if result["native_repeat_fraction_outside_tolerance"] > 0:
            result["outcome"] = "native_execution_not_repeatable_at_tolerance"
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(
                json.dumps({k: v for k, v in result.items() if k != "manifest"}),
                flush=True,
            )
            return
        for label, fields in GROUPS:
            if args.groups is not None and label not in args.groups:
                continue
            began = time.perf_counter()
            compiled = eqx.filter_jit(capture_subset)
            hidden, subset = compiled(model, tokens, mask, fields=fields)
            jax.block_until_ready((hidden, subset))
            delta = jnp.abs(hidden.astype(jnp.float32) - expected.astype(jnp.float32))
            row = {
                "group": label,
                "compile_and_call_seconds": time.perf_counter() - began,
                "max_error": float(jnp.max(delta)),
                "mean_error": float(jnp.mean(delta)),
                "fraction_outside_tolerance": float(jnp.mean(delta > 1e-5 + 1e-5 * jnp.abs(expected))),
            }
            repeated, _ = compiled(model, tokens, mask, fields=fields)
            jax.block_until_ready(repeated)
            row["repeat_max_error"] = float(jnp.max(jnp.abs(repeated.astype(jnp.float32) - hidden.astype(jnp.float32))))
            result["comparisons"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(row), flush=True)
            del hidden, subset, compiled
            if label == "all" and row["fraction_outside_tolerance"] == 0:
                result["outcome"] = "not_reproduced_at_reduced_batch"
                break
        result.setdefault("outcome", "output_groups_measured")
        result["device_memory"] = devices[0].memory_stats()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
