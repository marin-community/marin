# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete fixed-time native candidate 11 control and full native Paloma evaluation."""

import argparse
import functools
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh
from levanter.eval import eval_model
from levanter.tracker import NoopTracker, current_tracker
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.grug.fast_track.train import (
    GrugEvalConfig,
    _pgle_disabled,
    _reshard_tree_to_mesh,
    _to_dropless_local,
    build_tagged_evaluator,
    build_train_loader,
)
from experiments.grug.noise_belay.candidate11 import REFERENCE, ROOT, load_model_config, load_probe_manifest
from experiments.grug.noise_belay.candidate11_control_probe import native_control
from experiments.grug.noise_belay.candidate11_data import native_train_dataset
from experiments.grug.noise_belay.candidate11_forward import capture_forward
from experiments.grug.noise_belay.candidate11_gpu_probe import native_with_routing
from experiments.grug.noise_belay.candidate11_validation import capture_screen


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-uri", required=True)
    parser.add_argument("--training-seconds", type=float, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = load_probe_manifest()
    record = json.loads(REFERENCE.read_text())["config"]
    for key, value in record["trainer"]["trainer"]["jax_config"].items():
        jax.config.update(key, value)
    cfg = load_model_config(source_sha256=manifest["model_source_sha256"])
    devices = jax.devices()
    if len(devices) != 8 or any(d.platform != "gpu" for d in devices):
        raise ValueError("The control requires eight GPUs")
    axes = ("replica_dcn", "data", "expert", "model")
    mesh = Mesh(
        np.asarray(devices).reshape(1, 1, 8, 1),
        axes,
        axis_types=(AxisType.Explicit,) * 4,
    )
    eval_mesh = Mesh(
        np.asarray(devices).reshape(1, 8, 1, 1),
        axes,
        axis_types=(AxisType.Explicit,) * 4,
    )
    schedule_steps = record["trainer"]["trainer"]["num_train_steps"]
    result = {
        "kind": "native_candidate11_fixed_time_control",
        "status": "initializing",
        "manifest": manifest,
        "training_budget_seconds": args.training_seconds,
        "schedule_steps": schedule_steps,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "jax_version": jax.__version__,
        "jax_memory_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "default"),
        "gpu_count": 8,
        "global_batch": 128,
        "sequence_length": 4096,
        "parameter_count": 1026488855,
        "timing_policy": (
            "480-second wall allowance from first update; includes subsequent data waits, "
            "optimizer and local metric recording; compilation, warmup, validation, "
            "checkpoint and final evaluation separate"
        ),
        "adaptations": [
            "Recorded tokenizer archive resolved by local path",
            "Pinned runtime bug fixes and shared precision boundaries",
            "Stop at wall allowance or original schedule end, whichever comes first",
        ],
    }
    write_json(args.output / "result.json", result)
    started = time.perf_counter()
    with jax.set_mesh(mesh):
        dataset, schedule, data_config = native_train_dataset(ROOT / "tokenizers/hero-bpe-v16384")
        loader = build_train_loader(
            dataset,
            batch_schedule=schedule,
            mesh=mesh,
            max_buffered_batches=record["trainer"]["loader_buffer_batches"],
        )
        iterator = iter(loader.iter_from_step(0))
        first_batch = next(iterator)
        jax.block_until_ready(first_batch)
        first_record = json.loads((ROOT / "data-native-audit/manifest.json").read_text())["batches"][0]
        for name, array in (
            ("tokens", first_batch.tokens),
            ("loss_weight", first_batch.loss_weight),
            ("segment_ids", first_batch.attn_mask.segment_ids[0]),
        ):
            if hashlib.sha256(np.asarray(array).tobytes()).hexdigest() != first_record["array_sha256"][name]:
                raise ValueError(f"First native batch differs: {name}")
        result["first_batch_verified"] = True
        result["dataset_startup_seconds"] = time.perf_counter() - started
        _, key = jax.random.split(jax.random.PRNGKey(record["trainer"]["trainer"]["seed"]), 2)
        began = time.perf_counter()
        leaves, structure, update, mp = native_control(cfg, record=record, model_key=key, schedule_steps=schedule_steps)
        jax.block_until_ready(leaves)
        result["state_initialization_seconds"] = time.perf_counter() - began
        began = time.perf_counter()
        executable = update.lower(leaves, first_batch, loop_active=None, router_tie_active=None).compile()
        result["compile_seconds"] = time.perf_counter() - began
        began = time.perf_counter()
        warm_leaves = [jax.device_put(x, x.sharding, may_alias=False) for x in leaves]
        jax.block_until_ready(warm_leaves)
        warmed, metrics, watches = executable(warm_leaves, first_batch)
        jax.block_until_ready((warmed, metrics, watches))
        del warmed, warm_leaves, metrics, watches
        result["warmup_seconds"] = time.perf_counter() - began
        result["warmup_policy"] = (
            "Real first batch on nonaliasing state copies; discard outputs and retain untouched initial state"
        )
        result["status"] = "training"
        write_json(args.output / "result.json", result)
        step = 0
        predictions = 0.0
        durations = []
        batch = first_batch
        last_update = 0.0
        with (args.output / "metrics.jsonl").open("w") as series:
            training_start = time.perf_counter()
            while step < schedule_steps:
                remaining = args.training_seconds - (time.perf_counter() - training_start)
                if remaining <= max(last_update, 0.001):
                    break
                load_start = time.perf_counter()
                if step:
                    batch = next(iterator)
                load_time = time.perf_counter() - load_start
                if time.perf_counter() - training_start + max(last_update, 0.001) >= args.training_seconds:
                    break
                update_start = time.perf_counter()
                leaves, metrics, watches = executable(leaves, batch)
                jax.block_until_ready((leaves, metrics, watches))
                last_update = time.perf_counter() - update_start
                loss = float(metrics["train/loss"])
                if not np.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training loss at step {step}")
                predictions += float(jnp.sum(batch.loss_weight))
                durations.append(last_update)
                step += 1
                event = {
                    "event": "train",
                    "step": step,
                    "loss": loss,
                    "elapsed_seconds": time.perf_counter() - training_start,
                    "update_seconds": last_update,
                    "data_wait_seconds": load_time,
                    "prediction_weight_total": predictions,
                }
                series.write(json.dumps(event) + "\n")
                if step % 100 == 0 or step == 1:
                    series.flush()
                    print(json.dumps(event), flush=True)
            training_elapsed = time.perf_counter() - training_start
        state = jax.tree.unflatten(structure, leaves)
        if int(state.step) != step:
            raise ValueError("Native state step differs from measured step count")
        result.update(
            status="training_complete",
            completed_updates=step,
            training_wall_seconds=training_elapsed,
            training_gpu_hours=training_elapsed * 8 / 3600,
            tokens_presented=step * 128 * 4096,
            prediction_weight_total=predictions,
            final_training_loss=loss,
            median_update_seconds=float(np.median(durations)),
            training_memory=[d.memory_stats() for d in devices],
        )
        write_json(args.output / "result.json", result)
        print(
            json.dumps({k: v for k, v in result.items() if k not in ("manifest", "training_memory")}),
            flush=True,
        )
        ema_start = max(0, schedule_steps - record["trainer"]["ema_last_steps"])
        selected_model = state.ema_params if state.ema_params is not None and step > ema_start else state.params
        result["evaluation_weights"] = "ema" if selected_model is state.ema_params else "current"
        began = time.perf_counter()
        fs, checkpoint_path = filesystem_for(args.checkpoint_uri)
        with fs.open(checkpoint_path, "wb") as destination:
            eqx.tree_serialise_leaves(destination, selected_model)
        checkpoint = {
            "uri": args.checkpoint_uri,
            "size_bytes": fs.size(checkpoint_path),
            "completed_updates": step,
            "weights": result["evaluation_weights"],
            "format": "equinox serialized model leaves; reconstruct skeleton with pinned config/runtime patch",
        }
        write_json(args.output / "checkpoint.json", checkpoint)
        result["checkpoint_seconds"] = time.perf_counter() - began
        result["checkpoint"] = checkpoint
        write_json(args.output / "result.json", result)
        began = time.perf_counter()
        compute = mp.cast_to_compute(state.params)
        reference, _, routes = eqx.filter_jit(native_with_routing)(compute, first_batch.tokens, first_batch.attn_mask)
        actual, capture = eqx.filter_jit(capture_forward)(compute, first_batch.tokens, first_batch.attn_mask)
        jax.block_until_ready((reference, actual, capture))
        screen = capture_screen(
            compute,
            actual,
            reference,
            first_batch.tokens,
            first_batch.loss_weight,
            capture.routing_selected,
            routes,
        )
        write_json(args.output / "candidate11-validation.json", screen)
        result["capture_screen_passed"] = screen["passed"]
        result["capture_screen_seconds"] = time.perf_counter() - began
        write_json(args.output / "result.json", result)
        del (
            compute,
            reference,
            actual,
            capture,
            leaves,
            state,
            metrics,
            watches,
            executable,
        )
        result["status"] = "evaluating"
        write_json(args.output / "result.json", result)
    began = time.perf_counter()
    with jax.set_mesh(eval_mesh), current_tracker(NoopTracker()):
        evaluation_model = _reshard_tree_to_mesh(selected_model, eval_mesh)
        evaluator = build_tagged_evaluator(
            data_config=data_config,
            max_seq_len=cfg.max_seq_len,
            mesh=eval_mesh,
            eval_cfg=draccus.decode(GrugEvalConfig, record["eval"]),
            mp=mp,
            model_transform=functools.partial(
                _to_dropless_local,
                implementation=record["eval"]["dropless_eval_moe_implementation"],
            ),
        )
        if evaluator is None:
            raise ValueError("Reference evaluation datasets missing")
        with _pgle_disabled():
            scores = eval_model(evaluator, evaluation_model, prefix="eval")
        result["evaluation"] = {k: float(v) for k, v in scores.items()}
    result["evaluation_seconds"] = time.perf_counter() - began
    result["status"] = "complete"
    write_json(args.output / "result.json", result)
    print(
        json.dumps(
            {
                "event": "complete",
                "updates": step,
                "training_wall_seconds": training_elapsed,
                "evaluation": result["evaluation"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
