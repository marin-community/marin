# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the enlarged Belay reference at the small checkpoint's matched update count."""

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import draccus
import equinox as eqx
import jax
import numpy as np
from jax.sharding import AxisType, Mesh
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.grug.fast_track.model import GrugModelConfig
from experiments.grug.fast_track.train import build_train_loader
from experiments.grug.noise_belay.candidate11 import REFERENCE, ROOT, load_model_config
from experiments.grug.noise_belay.candidate11_control_probe import native_control
from experiments.grug.noise_belay.candidate11_data import native_train_dataset
from experiments.grug.tied_drafter.calibration import MODEL_SOURCE_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--microbatches", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    record = json.loads(REFERENCE.read_text())["config"]
    record["trainer"]["grad_accum_microbatches"] = args.microbatches
    for name, value in record["trainer"]["trainer"]["jax_config"].items():
        jax.config.update(name, value)
    load_model_config(source_sha256=MODEL_SOURCE_SHA256)
    config_bytes = args.model_config.read_bytes()
    cfg = draccus.decode(GrugModelConfig, json.loads(config_bytes))
    devices = jax.devices("gpu")
    if len(devices) != 8:
        raise ValueError("Use eight H100s with the native expert-parallel mesh")
    mesh = Mesh(
        np.asarray(devices).reshape(1, 1, 8, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "initializing",
        "steps_requested": args.steps,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_source_sha256": MODEL_SOURCE_SHA256,
        "model_config": json.loads(config_bytes),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "batch_size": 128,
        "microbatches": args.microbatches,
        "sequence_length": 4096,
        "hardware": "8 H100, RNO2A interactive",
        "schedule_steps": record["trainer"]["trainer"]["num_train_steps"],
        "checkpoint": args.checkpoint,
        "comparison": "Same 2116 updates and data sequence as small EMA checkpoint; not matched wall time or compute.",
    }

    def save():
        (args.output / "training.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    with jax.set_mesh(mesh):
        dataset, schedule, _ = native_train_dataset(ROOT / "tokenizers/hero-bpe-v16384")
        loader = build_train_loader(dataset, batch_schedule=schedule, mesh=mesh, max_buffered_batches=2)
        iterator = iter(loader.iter_from_step(0))
        try:
            first = next(iterator)
            expected = json.loads(Path(__file__).with_name("tied-large-first-batch.json").read_text())
            for name, array in (
                ("tokens", first.tokens),
                ("loss_weight", first.loss_weight),
                ("segment_ids", first.attn_mask.segment_ids[0]),
            ):
                if hashlib.sha256(np.asarray(array).tobytes()).hexdigest() != expected["array_sha256"][name]:
                    raise ValueError(f"Native first-batch fingerprint mismatch: {name}")
            result["first_batch_verified"] = True
            key = jax.random.split(jax.random.PRNGKey(record["trainer"]["trainer"]["seed"]), 2)[1]
            began = time.perf_counter()
            leaves, structure, update, _ = native_control(
                cfg, record=record, model_key=key, schedule_steps=result["schedule_steps"]
            )
            jax.block_until_ready(leaves)
            result["initialization_seconds"] = time.perf_counter() - began
            state = jax.tree.unflatten(structure, leaves)
            result["parameters"] = sum(x.size for x in jax.tree.leaves(state.params))
            del state
            began = time.perf_counter()
            executable = update.lower(leaves, first, loop_active=None, router_tie_active=None).compile()
            result["compile_seconds"] = time.perf_counter() - began
            result["status"] = "training"
            save()
            began = time.perf_counter()
            with (args.output / "train.jsonl").open("w") as stream:
                for step in range(1, args.steps + 1):
                    start = time.perf_counter()
                    batch = first if step == 1 else next(iterator)
                    loaded = time.perf_counter()
                    leaves, metrics, watches = executable(leaves, batch)
                    jax.block_until_ready((leaves, metrics, watches))
                    end = time.perf_counter()
                    loss = float(metrics["train/loss"])
                    if not np.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss at update {step}")
                    row = {
                        "step": step,
                        "loss": loss,
                        "elapsed_seconds": end - began,
                        "update_seconds": end - loaded,
                        "data_seconds": loaded - start,
                    }
                    stream.write(json.dumps(row) + "\n")
                    if step == 1 or step % 100 == 0 or step == args.steps:
                        stream.flush()
                        print(json.dumps(row), flush=True)
            result.update(
                completed_updates=step,
                training_seconds=time.perf_counter() - began,
                final_training_loss=loss,
                status="saving_checkpoint",
            )
            state = jax.tree.unflatten(structure, leaves)
            if int(state.step) != args.steps:
                raise ValueError("Optimizer step count mismatch")
            ema_start = max(0, result["schedule_steps"] - record["trainer"]["ema_last_steps"])
            if state.ema_params is None or step <= ema_start:
                raise ValueError("Matched comparison requires the scheduled EMA weights")
            save()
            fs, path = filesystem_for(args.checkpoint)
            with fs.open(path, "wb") as stream:
                eqx.tree_serialise_leaves(stream, state.ema_params)
            result.update(status="complete", checkpoint_bytes=fs.size(path), evaluation_weights="ema")
            save()
            print(json.dumps({"event": "complete", "parameters": result["parameters"], "steps": step}), flush=True)
        finally:
            iterator._batches.stop()


if __name__ == "__main__":
    main()
