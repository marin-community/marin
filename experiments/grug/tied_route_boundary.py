# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect bounded routing traces and compare packed FP4/FP8 expert exchanges."""

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.grug.fast_track.model import ROUTING_SELECTED_KEY, ROUTING_WEIGHTS_KEY, Transformer
from experiments.grug.fast_track.train import build_train_loader
from experiments.grug.noise_belay.candidate11 import REFERENCE, ROOT, load_model_config
from experiments.grug.noise_belay.candidate11_data import native_train_dataset
from experiments.grug.tied_drafter.calibration import MODEL_SOURCE_SHA256
from experiments.grug.tied_drafter_approximations import exchange_function
from experiments.grug.tied_drafter_latency import restored_leaf

TRACE_START = 53100
SAMPLE_ROWS = 16


def plain_hidden(params, batch, mp):
    return mp.cast_to_compute(params)(batch.tokens, mask=batch.attn_mask)[0]


def captured_routes(params, batch, mp):
    hidden, metrics = mp.cast_to_compute(params)(batch.tokens, mask=batch.attn_mask, return_routing=True)
    return hidden, metrics[ROUTING_SELECTED_KEY], metrics[ROUTING_WEIGHTS_KEY]


@eqx.filter_jit
def differences(a, b):
    delta = a.astype(jnp.float32) - b.astype(jnp.float32)
    return {
        "max_absolute": jnp.max(jnp.abs(delta)),
        "relative_rms": jnp.sqrt(jnp.mean(delta**2) / jnp.maximum(jnp.mean(a.astype(jnp.float32) ** 2), 1e-30)),
        "changed_fraction": jnp.mean((delta != 0).astype(jnp.float32)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    record = json.loads(REFERENCE.read_text())["config"]
    for key, value in record["trainer"]["trainer"]["jax_config"].items():
        jax.config.update(key, value)
    cfg = load_model_config(source_sha256=MODEL_SOURCE_SHA256)
    devices = jax.devices("gpu")
    if len(devices) != 8:
        raise ValueError("Use eight H100s")
    mesh = Mesh(
        np.asarray(devices).reshape(1, 1, 8, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    key = jax.random.PRNGKey(0)
    mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "initializing",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint": args.checkpoint,
        "batch_indices": [TRACE_START, TRACE_START + 1],
        "route_rows_per_batch": SAMPLE_ROWS,
        "routes": [],
        "communication": [],
        "scope": (
            "Fixed-checkpoint consecutive-batch route forecasting; no training-update or live cache test. "
            "Packed activation-only exchanges exclude expert-id metadata and combine traffic."
        ),
    }

    def save():
        (args.output / "boundary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    with jax.set_mesh(mesh):
        template = eqx.filter_eval_shape(Transformer.init, cfg, key=key)
        fs, path = filesystem_for(args.checkpoint)
        with fs.open(path, "rb") as stream:
            hero = eqx.tree_deserialise_leaves(stream, template, filter_spec=lambda f, t: restored_leaf(f, t, mesh))
        jax.block_until_ready(hero)
        dataset, schedule, _ = native_train_dataset(ROOT / "tokenizers/hero-bpe-v16384")
        loader = build_train_loader(dataset, batch_schedule=schedule, mesh=mesh, max_buffered_batches=2)
        iterator = iter(loader.iter_from_step(TRACE_START))
        try:
            batches = [next(iterator), next(iterator)]
        finally:
            iterator._batches.stop()
        capture = eqx.filter_jit(captured_routes).lower(hero, batches[0], mp).compile()
        plain = eqx.filter_jit(plain_hidden).lower(hero, batches[0], mp).compile()
        for index, batch in enumerate(batches):
            normal = plain(hero, batch, mp)
            hidden, selected, weights = capture(hero, batch, mp)
            parity = {k: float(v) for k, v in jax.device_get(differences(normal, hidden)).items()}
            repeated_hidden, repeated_routes, _ = capture(hero, batch, mp)
            repeated_equal = bool(jax.device_get(jnp.all(selected == repeated_routes)))
            routes_host, weights_host = jax.device_get((selected, weights))
            counts = np.stack([np.bincount(x.reshape(-1), minlength=cfg.num_experts) for x in routes_host])
            shape = (cfg.num_layers, *batch.tokens.shape, cfg.num_experts_per_token)
            np.savez_compressed(
                args.output / f"routes-{index}.npz",
                routes=routes_host.reshape(shape)[:, :SAMPLE_ROWS].astype(np.uint16),
                weights=weights_host.reshape(shape)[:, :SAMPLE_ROWS].astype(np.float16),
                tokens=np.asarray(jax.device_get(batch.tokens))[:SAMPLE_ROWS].astype(np.uint16),
            )
            result["routes"].append(
                {
                    "batch_index": TRACE_START + index,
                    "full_batch_expert_counts": counts.tolist(),
                    "capture_hidden_difference": parity,
                    "repeated_routes_equal": repeated_equal,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "routes",
                        "batch": TRACE_START + index,
                        "experts_used_per_layer": (counts > 0).sum(axis=1).tolist(),
                        "capture_difference": parity,
                    }
                ),
                flush=True,
            )
            save()
            del normal, hidden, selected, weights, repeated_hidden, repeated_routes, routes_host, weights_host
        del hero, capture, plain, batches
        result["status"] = "communication"
        save()
        spec = P("expert", None, None)
        for rows, width in ((256, 256), (4096, 256), (65536, 256), (4096, 1024), (16384, 1024)):
            x = jax.random.normal(key, (64, rows, width), dtype=jnp.bfloat16, out_sharding=NamedSharding(mesh, spec))
            baseline = exchange_function(spec, "bf16")
            reference = baseline(x)
            jax.block_until_ready(reference)
            for encoding in ("bf16", "fp8", "fp4"):
                function = exchange_function(spec, encoding)
                output = function(x)
                error = {k: float(v) for k, v in jax.device_get(differences(reference, output)).items()}
                for _ in range(3):
                    jax.block_until_ready(function(x))
                times = []
                for _ in range(10):
                    began = time.perf_counter()
                    jax.block_until_ready(function(x))
                    times.append(time.perf_counter() - began)
                row_bytes = {"bf16": width * 2, "fp8": width + 4 * (width // 128), "fp4": width // 2 + width // 16 + 4}[
                    encoding
                ]
                entry = {
                    "encoding": encoding,
                    "rows_per_destination": rows,
                    "width": width,
                    "bytes_per_rank": 8 * rows * row_bytes,
                    "seconds": times,
                    "error": error,
                }
                result["communication"].append(entry)
                print(
                    json.dumps(
                        {
                            "event": "communication",
                            "encoding": encoding,
                            "rows": rows,
                            "width": width,
                            "median_seconds": float(np.median(times)),
                            "error": error,
                        }
                    ),
                    flush=True,
                )
                save()
                del function, output
            del baseline, reference, x
        result["status"] = "complete"
        save()


if __name__ == "__main__":
    main()
