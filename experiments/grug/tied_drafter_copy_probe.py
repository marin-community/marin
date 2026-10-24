# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare native update repeatability with snapshot-method differences in one process."""

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.grug.fast_track.train import build_train_loader
from experiments.grug.noise_belay.candidate11 import REFERENCE, ROOT, load_model_config
from experiments.grug.noise_belay.candidate11_control_probe import native_control
from experiments.grug.noise_belay.candidate11_data import native_train_dataset
from experiments.grug.tied_drafter.calibration import MODEL_SOURCE_SHA256
from experiments.grug.tied_drafter_endpoints import device_snapshot, identical_arrays, load_batches, restore_snapshot


@jax.jit
def differences(left, right):
    values = []
    for a, b in zip(left, right, strict=True):
        delta = jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))
        values.append((jnp.max(delta), jnp.mean(delta), jnp.mean((a != b).astype(jnp.float32))))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    record = json.loads(REFERENCE.read_text())["config"]
    record["trainer"]["grad_accum_microbatches"] = 2
    for name, value in record["trainer"]["trainer"]["jax_config"].items():
        jax.config.update(name, value)
    cfg = load_model_config(source_sha256=MODEL_SOURCE_SHA256)
    devices = jax.devices("gpu")
    mesh = Mesh(
        np.asarray(devices).reshape(1, 1, 8, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "initializing",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "comparisons": {},
        "runs": [],
    }

    def save():
        (args.output / "copy-probe.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    with jax.set_mesh(mesh):
        data, schedule, _ = native_train_dataset(ROOT / "tokenizers/hero-bpe-v16384")
        loader = build_train_loader(data, batch_schedule=schedule, mesh=mesh, max_buffered_batches=2)
        batches = load_batches(loader, 0, 17)
        key = jax.random.split(jax.random.PRNGKey(0), 2)[1]
        leaves, tree, update, _ = native_control(
            cfg, record=record, model_key=key, schedule_steps=record["trainer"]["trainer"]["num_train_steps"]
        )
        leaves = jax.block_until_ready(leaves)
        shardings = [x.sharding for x in leaves]
        paths = [
            jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(jax.tree.unflatten(tree, leaves))
        ]
        executable = update.lower(leaves, batches[0], loop_active=None, router_tie_active=None).compile()
        for batch in batches[:16]:
            leaves, metrics, watches = executable(leaves, batch)
            jax.block_until_ready((leaves, metrics, watches))
        cpu = [np.array(x, copy=True) for x in jax.device_get(leaves)]
        outputs = {}
        for name in ["device_a", "device_b", "host_a", "host_b"]:
            before = time.perf_counter()
            candidate = restore_snapshot(cpu, shardings) if name.startswith("host") else device_snapshot(leaves)
            if not bool(identical_arrays(leaves, candidate)):
                raise AssertionError("Snapshot changed input state")
            result["runs"].append(
                {
                    "name": name,
                    "input_exact": True,
                    "formats_match_source": all(a.format == b.format for a, b in zip(candidate, leaves, strict=True)),
                }
            )
            candidate, metrics, watches = executable(candidate, batches[16])
            jax.block_until_ready((candidate, metrics, watches))
            result["runs"][-1].update(loss=float(metrics["train/loss"]), seconds=time.perf_counter() - before)
            outputs[name] = candidate
            save()
        for a, b in [("device_a", "device_b"), ("host_a", "host_b"), ("device_a", "host_a"), ("device_b", "host_b")]:
            stats = jax.device_get(differences(outputs[a], outputs[b]))
            rows = [
                {"path": p, "max_abs": float(v[0]), "mean_abs": float(v[1]), "different_fraction": float(v[2])}
                for p, v in zip(paths, stats, strict=True)
                if float(v[2]) > 0
            ]
            result["comparisons"][a + "/" + b] = {
                "different_leaves": len(rows),
                "leaves": rows,
                "max_abs": max((r["max_abs"] for r in rows), default=0),
            }
            print(
                json.dumps(
                    {
                        "comparison": a + "/" + b,
                        "different_leaves": len(rows),
                        "largest": sorted(rows, key=lambda r: r["max_abs"], reverse=True)[:8],
                    }
                ),
                flush=True,
            )
            save()
        result["status"] = "complete"
        save()


if __name__ == "__main__":
    main()
