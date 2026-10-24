# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native-only loss calibration on changing data, before any draft treatment."""

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
from jax.sharding import AxisType, Mesh
from jaxlib import xla_client
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.grug.fast_track.model import Transformer
from experiments.grug.fast_track.train import build_train_loader
from experiments.grug.noise_belay.candidate11 import REFERENCE, ROOT, load_model_config
from experiments.grug.noise_belay.candidate11_control_probe import native_control
from experiments.grug.noise_belay.candidate11_data import native_train_dataset
from experiments.grug.tied_drafter.acceptance import fit_progress_forecast, grouped_loss
from experiments.grug.tied_drafter.coordinates import DraftRecipe, coordinate_map, draft_config, restrict_model
from experiments.grug.tied_drafter.runtime import sequence_losses

CHECKPOINTS = (192, 224, 256, 288, 320, 448, 480, 512, 544, 576, 960, 992, 1024, 1056, 1088)
ANCHORS = (256, 512, 1024)
MODEL_SOURCE_SHA256 = "5f39c7bad449de5874a46190d43f87d4063121411d3a3547a20112903477455b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    args.output.mkdir(parents=True, exist_ok=True)
    record = json.loads(REFERENCE.read_text())["config"]
    for key, value in record["trainer"]["trainer"]["jax_config"].items():
        jax.config.update(key, value)
    cfg = load_model_config(source_sha256=MODEL_SOURCE_SHA256)
    devices = jax.devices("gpu")
    if len(devices) != 8:
        raise ValueError("Calibration requires the eight-H100 native hero")
    mesh = Mesh(
        np.asarray(devices).reshape(1, 1, 8, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    result = {
        "status": "initializing",
        "seed": args.seed,
        "gpu_allocator": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "default"),
        "pjrt_gpu_options": xla_client.generate_pjrt_gpu_plugin_options(),
        "kind": "native_control_only_calibration",
        "model_source_sha256": MODEL_SOURCE_SHA256,
        "selection_data_policy": "Reserved training-mixture indices; not a document-disjoint external test set",
        "calibration_batch_indices": list(range(10000, 10004)),
        "check_batch_indices": list(range(10004, 10008)),
        "held_out_target_batch_indices": list(range(20000, 20004)),
        "groups": "Eight adjacent packed sequences per group; nominal group-based uncertainty",
        "checkpoints": [],
        "snapshots": [],
    }

    def save():
        (args.output / "calibration.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    with jax.set_mesh(mesh):
        dataset, schedule, _ = native_train_dataset(ROOT / "tokenizers/hero-bpe-v16384")
        loader = build_train_loader(dataset, batch_schedule=schedule, mesh=mesh, max_buffered_batches=4)
        train = iter(loader.iter_from_step(0))
        first = next(train)
        reserved = {}
        for name, start, count in (("calibration", 10000, 4), ("check", 10004, 4), ("target", 20000, 4)):
            iterator = iter(loader.iter_from_step(start))
            reserved[name] = [next(iterator) for _ in range(count)]
            del iterator
        jax.block_until_ready(reserved)
        result["reserved_token_sha256"] = {
            name: [hashlib.sha256(np.asarray(b.tokens).tobytes()).hexdigest() for b in batches]
            for name, batches in reserved.items()
        }
        key = jax.random.split(jax.random.PRNGKey(args.seed), 2)[1]
        leaves, structure, update, mp = native_control(
            cfg, record=record, model_key=key, schedule_steps=record["trainer"]["trainer"]["num_train_steps"]
        )
        began = time.perf_counter()
        executable = update.lower(leaves, first, loop_active=None, router_tie_active=None).compile()
        result["compile_seconds"] = time.perf_counter() - began
        state = jax.tree.unflatten(structure, leaves)
        began = time.perf_counter()
        forward = eqx.filter_jit(sequence_losses).lower(state.params, first, mp).compile()
        result["forward_compile_seconds"] = time.perf_counter() - began
        template = eqx.filter_eval_shape(Transformer.init, draft_config(cfg, DraftRecipe.WIDTH_QUARTER), key=key)
        mapping = coordinate_map(state.params, template)
        restrict = eqx.filter_jit(restrict_model).lower(state.params, template, mapping).compile()
        del state
        # The first real update warms the executable without retaining a second
        # initial state solely for warmup.
        result["warmup_policy"] = "First real update included in the recorded control trajectory"
        result["memory_before_training"] = [d.memory_stats() for d in devices]
        result["status"] = "training"
        save()
        print(json.dumps({"event": "training_start", "memory": result["memory_before_training"][0]}), flush=True)
        training_seconds = 0.0
        with (args.output / "train.jsonl").open("w") as stream:
            for step in range(1, max(CHECKPOINTS) + 1):
                began = time.perf_counter()
                batch = first if step == 1 else next(train)
                loaded = time.perf_counter()
                leaves, metrics, watches = executable(leaves, batch)
                jax.block_until_ready((leaves, metrics, watches))
                finished = time.perf_counter()
                training_seconds += finished - began
                loss = float(metrics["train/loss"])
                if not np.isfinite(loss):
                    raise FloatingPointError(f"Native loss nonfinite at {step}")
                stream.write(
                    json.dumps(
                        {
                            "step": step,
                            "loss": loss,
                            "training_seconds": training_seconds,
                            "update_seconds": finished - loaded,
                            "data_seconds": loaded - began,
                        }
                    )
                    + "\n"
                )
                if step == 1 or step % 64 == 0:
                    stream.flush()
                    print(json.dumps({"event": "native_update", "step": step, "loss": loss}), flush=True)
                if step == 1:
                    result["memory_after_first_update"] = [d.memory_stats() for d in devices]
                    save()
                if step not in CHECKPOINTS:
                    continue
                state = jax.tree.unflatten(structure, leaves)
                row = {"step": step, "training_seconds": training_seconds, "groups": {}}
                began = time.perf_counter()
                for name, batches in reserved.items():
                    values = [jax.device_get(forward(state.params, b, mp)) for b in batches]
                    losses = np.concatenate([x[0] for x in values])
                    counts = np.concatenate([x[1] for x in values])
                    row["groups"][name] = grouped_loss(losses, counts, sequences_per_group=8).tolist()
                row["verification_seconds"] = time.perf_counter() - began
                if step in ANCHORS:
                    before = forward(state.params, reserved["calibration"][0], mp)
                    after = forward(state.params, reserved["calibration"][0], mp)
                    row["repeatability_max_loss_difference"] = float(
                        np.max(np.abs(np.asarray(before[0]) - np.asarray(after[0])))
                    )
                    began = time.perf_counter()
                    draft = restrict(state.params, template, mapping)
                    # Host copies cannot alias buffers subsequently donated by the hero.
                    arrays = [np.array(x, copy=True) for x in jax.tree.leaves(draft)]
                    result["snapshots"].append(
                        {
                            "step": step,
                            "restriction_and_host_copy_seconds": time.perf_counter() - began,
                            "bytes": sum(x.nbytes for x in arrays),
                        }
                    )
                    del draft, arrays
                result["checkpoints"].append(row)
                del state
                save()
                stream.flush()
                print(
                    json.dumps(
                        {"event": "calibration", "step": step, "loss": loss, "training_seconds": training_seconds}
                    ),
                    flush=True,
                )
        result["forecasts"] = []
        for anchor in ANCHORS:
            window = [x for x in result["checkpoints"] if anchor - 64 <= x["step"] <= anchor + 64]
            fit = [x for x in window if x["step"] <= anchor]
            forecast = fit_progress_forecast(
                np.array([x["step"] for x in fit]), np.array([x["groups"]["calibration"] for x in fit]), confidence=0.975
            )
            conservative = fit_progress_forecast(
                np.array([x["step"] for x in window]), np.array([x["groups"]["check"] for x in window]), confidence=0.975
            )
            reference = next(x for x in window if x["step"] == anchor)
            residuals = [
                {
                    "step": x["step"],
                    "mean_prediction_residual": float(
                        np.mean(x["groups"]["check"])
                        - np.mean(reference["groups"]["check"])
                        + (x["step"] - anchor) * forecast.loss_per_step
                    ),
                }
                for x in window
                if x["step"] > anchor
            ]
            result["forecasts"].append(
                {
                    "anchor": anchor,
                    "fit": asdict(forecast),
                    "independent_check": asdict(conservative),
                    "future_residuals": residuals,
                    "target_loss": float(np.mean(reference["groups"]["target"])),
                }
            )
        result["training_seconds"] = training_seconds
        result["status"] = "complete"
        save()


if __name__ == "__main__":
    main()
