# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded, real-batch optimizer and verifier timing; no learning-quality claim."""

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
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.grug.fast_track.model import Transformer
from experiments.grug.fast_track.train import build_train_loader
from experiments.grug.noise_belay.candidate11 import REFERENCE, ROOT, load_model_config
from experiments.grug.noise_belay.candidate11_control_probe import native_control
from experiments.grug.noise_belay.candidate11_data import native_train_dataset
from experiments.grug.tied_drafter.coordinates import DraftRecipe, coordinate_map, draft_config, restrict_model
from experiments.grug.tied_drafter.runtime import draft_training_state, sequence_losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=["hero", *DraftRecipe], required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--draft-learning-rate", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    args.output.mkdir(parents=True, exist_ok=True)
    record = json.loads(REFERENCE.read_text())["config"]
    for key, value in record["trainer"]["trainer"]["jax_config"].items():
        jax.config.update(key, value)
    source = Path("experiments/grug/fast_track/model.py")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    cfg = load_model_config(source_sha256="5f39c7bad449de5874a46190d43f87d4063121411d3a3547a20112903477455b")
    devices = jax.devices("gpu")
    if len(devices) != args.gpus:
        raise ValueError(f"Expected {args.gpus} GPUs, got {len(devices)}")
    mesh = Mesh(
        np.asarray(devices).reshape(1, 1, args.gpus, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    result = {
        "status": "initializing",
        "recipe": args.recipe,
        "gpu_count": args.gpus,
        "jax_version": jax.__version__,
        "model_sha256": source_hash,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "updates": [],
        "forward_seconds": [],
        "data": "first native training batch, repeated for latency only",
        "draft_learning_rate": args.draft_learning_rate,
    }

    def save():
        (args.output / "profile.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    with jax.set_mesh(mesh):
        began = time.perf_counter()
        dataset, schedule, _ = native_train_dataset(ROOT / "tokenizers/hero-bpe-v16384")
        loader = build_train_loader(dataset, batch_schedule=schedule, mesh=mesh, max_buffered_batches=2)
        iterator = iter(loader.iter_from_step(0))
        batch = next(iterator)
        jax.block_until_ready(batch)
        result["data_seconds"] = time.perf_counter() - began
        result["batch_shape"] = list(batch.tokens.shape)
        result["tokens_sha256"] = hashlib.sha256(np.asarray(batch.tokens).tobytes()).hexdigest()
        key = jax.random.split(jax.random.PRNGKey(0), 2)[1]
        began = time.perf_counter()
        if args.recipe == "hero":
            leaves, structure, update, mp = native_control(
                cfg,
                record=record,
                model_key=key,
                schedule_steps=record["trainer"]["trainer"]["num_train_steps"],
            )
        else:
            initialized = eqx.filter_jit(Transformer.init)(cfg, key=key)
            template = eqx.filter_eval_shape(Transformer.init, draft_config(cfg, DraftRecipe(args.recipe)), key=key)
            mapping = coordinate_map(initialized, template)
            draft = eqx.filter_jit(restrict_model)(initialized, template, mapping)
            jax.block_until_ready(draft)
            del initialized
            result["mapping"] = [asdict(x) for x in mapping]
            leaves, structure, update, mp = draft_training_state(
                draft,
                learning_rate=args.draft_learning_rate,
                z_loss_weight=record["trainer"]["z_loss_weight"],
            )
            del draft
        jax.block_until_ready(leaves)
        result["initialization_seconds"] = time.perf_counter() - began
        save()
        began = time.perf_counter()
        executable = update.lower(leaves, batch, loop_active=None, router_tie_active=None).compile()
        result["compile_seconds"] = time.perf_counter() - began
        result["xla_cost_analysis"] = executable.cost_analysis()
        result["xla_cost_caveat"] = "Custom calls may omit arithmetic; this is not a complete FLOP count."
        result["status"] = "measuring"
        save()
        for step in range(args.steps):
            began = time.perf_counter()
            leaves, metrics, watches = executable(leaves, batch)
            jax.block_until_ready((leaves, metrics, watches))
            elapsed = time.perf_counter() - began
            loss = float(metrics["train/loss"])
            if not np.isfinite(loss):
                raise FloatingPointError(f"Nonfinite profile loss at step {step}")
            result["updates"].append({"step": step, "seconds": elapsed, "loss": loss})
            save()
            print(json.dumps(result["updates"][-1]), flush=True)
        state = jax.tree.unflatten(structure, leaves)
        began = time.perf_counter()
        forward = eqx.filter_jit(sequence_losses).lower(state.params, batch, mp).compile()
        result["forward_compile_seconds"] = time.perf_counter() - began
        for repeat in range(4):
            began = time.perf_counter()
            losses, counts = forward(state.params, batch, mp)
            jax.block_until_ready((losses, counts))
            elapsed = time.perf_counter() - began
            if repeat:
                result["forward_seconds"].append(elapsed)
        result["per_sequence_loss"] = np.asarray(losses).tolist()
        result["scored_counts"] = np.asarray(counts).tolist()
        result["median_update_seconds"] = float(np.median([x["seconds"] for x in result["updates"][1:]]))
        result["median_forward_seconds"] = float(np.median(result["forward_seconds"]))
        result["device_memory"] = [device.memory_stats() for device in devices]
        result["status"] = "complete"
        save()


if __name__ == "__main__":
    main()
