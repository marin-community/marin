# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Calibrate native candidate 11 optimizer steps on one exact training batch.

The batch is deliberately repeated. This checks runtime/update correctness and
step cost; it does not estimate generalization or replace a paired training run.
"""

import argparse
import hashlib
import json
import os
import time
from functools import partial
from pathlib import Path

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.callbacks.watch import WatchConfig
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask

from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig
from experiments.grug.fast_track.train import (
    _flat_train_step,
    _make_train_step,
    initial_state,
)
from experiments.grug.noise_belay.candidate11 import REFERENCE, ROOT, load_model_config, load_probe_manifest
from experiments.grug.noise_belay.candidate11_capture_probe import GROUPS, capture_subset
from experiments.grug.noise_belay.candidate11_forward import capture_forward
from experiments.grug.noise_belay.candidate11_gpu_probe import (
    capture_hidden_only,
    native_forward,
    native_with_routing,
)
from experiments.grug.noise_belay.candidate11_validation import capture_screen


def native_control(model_config, *, record, model_key, schedule_steps):
    """Build the native state and native update, preserving optimizer and EMA."""
    trainer = record["trainer"]
    mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    optimizer = draccus.decode(GrugMoeMuonHConfig, record["optimizer"]).build(schedule_steps)
    state = eqx.filter_jit(initial_state)(
        model_config,
        optimizer=optimizer,
        mp=mp,
        key=model_key,
        ema_beta=trainer["ema_beta"],
    )
    update = _make_train_step(
        optimizer,
        mp,
        z_loss_weight=trainer["z_loss_weight"],
        ema_beta=trainer["ema_beta"],
        ema_start_step=max(0, schedule_steps - trainer["ema_last_steps"]),
        head_replay_period=trainer["head_replay_period"],
        head_replay_scale=trainer["head_replay_scale"],
        watch_config=draccus.decode(WatchConfig, trainer["trainer"]["watch"]),
        grad_accum_microbatches=trainer["grad_accum_microbatches"],
    )
    leaves, structure = jax.tree.flatten(state)
    return leaves, structure, _flat_train_step(update, structure), mp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-groups", nargs="+", choices=[label for label, _ in GROUPS])
    args = parser.parse_args()
    if args.steps < 2:
        raise ValueError("At least two steps are required to measure steady cost")
    if os.environ.get("GRUG_SKIP_OPTIMIZER"):
        raise ValueError("Optimizer updates must be enabled")
    manifest = load_probe_manifest()
    record = json.loads(REFERENCE.read_text())["config"]
    cfg = load_model_config(source_sha256=manifest["model_source_sha256"])
    devices = jax.devices()
    if len(devices) != 8 or any(d.platform != "gpu" for d in devices):
        raise ValueError("This control calibration requires eight GPUs")
    mesh = Mesh(
        np.asarray(devices).reshape(1, 1, 8, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "kind": "native_optimizer_calibration_repeated_real_batch",
        "manifest": manifest,
        "steps": [],
        "devices": [d.device_kind for d in devices],
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "jax_version": jax.__version__,
        "schedule_steps": record["trainer"]["trainer"]["num_train_steps"],
        "global_batch": 128,
        "sequence": 4096,
    }
    began = time.perf_counter()
    with jax.set_mesh(mesh):
        batch_record = json.loads((ROOT / "data-native-audit/manifest.json").read_text())
        with np.load(ROOT / "data-native-audit/batch-000000.npz") as stored:
            arrays = {k: stored[k] for k in stored.files}
        for name, value in arrays.items():
            if hashlib.sha256(value.tobytes()).hexdigest() != batch_record["batches"][0]["array_sha256"][name]:
                raise ValueError(f"Native batch fingerprint mismatch: {name}")
        sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
        tokens, weight, segments = [
            jax.device_put(arrays[k], sharding) for k in ("tokens", "loss_weight", "segment_ids")
        ]
        batch = GrugLmExample(
            tokens,
            weight,
            AttentionMask(is_causal=True, segment_ids=(segments, segments)),
        )
        result["data_manifest"] = batch_record
        _, model_key = jax.random.split(jax.random.PRNGKey(record["trainer"]["trainer"]["seed"]), 2)
        start = time.perf_counter()
        leaves, structure, update, mp = native_control(
            cfg,
            record=record,
            model_key=model_key,
            schedule_steps=result["schedule_steps"],
        )
        jax.block_until_ready(leaves)
        result["state_initialization_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        executable = update.lower(leaves, batch, loop_active=None, router_tie_active=None).compile()
        result["update_compile_seconds"] = time.perf_counter() - start
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        for step in range(args.steps):
            start = time.perf_counter()
            leaves, metrics, watches = executable(leaves, batch)
            jax.block_until_ready((leaves, metrics, watches))
            elapsed = time.perf_counter() - start
            loss = float(metrics["train/loss"])
            if not np.isfinite(loss):
                raise FloatingPointError(f"Nonfinite control loss after {step} updates")
            row = {"step": step, "step_seconds": elapsed, "loss": loss}
            result["steps"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        state = jax.tree.unflatten(structure, leaves)
        finite = eqx.filter_jit(
            lambda tree: jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree) if eqx.is_array(x)]))
        )(state.params)
        if not bool(finite):
            raise FloatingPointError("Nonfinite trained control parameters")
        model = mp.cast_to_compute(state.params)
        result["median_steady_step_seconds"] = float(np.median([r["step_seconds"] for r in result["steps"][1:]]))
        result["projected_steps_in_480_seconds"] = int(480 / result["median_steady_step_seconds"])
        result["parameters_finite"] = True
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        outputs = {}
        for name, function in (
            ("native", native_forward),
            ("capture", capture_forward),
            ("hidden_only", capture_hidden_only),
            ("routing", native_with_routing),
        ):
            executable = eqx.filter_jit(function).lower(model, tokens, batch.attn_mask).compile()
            if name in ("native", "capture"):
                (args.output.parent / f"{name}-optimized-hlo.txt").write_text(executable.compiled.as_text())
            outputs[name] = executable(model, tokens, batch.attn_mask)
            jax.block_until_ready(outputs[name])
            result["last_validation_stage"] = name
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        expected = outputs["native"]
        actual, capture = outputs["capture"]

        def differences(a, b):
            delta = jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))
            return {
                "max": float(jnp.max(delta)),
                "mean": float(jnp.mean(delta)),
                "fraction_outside_tolerance": float(jnp.mean(delta > 1e-5 + 1e-5 * jnp.abs(b))),
            }

        result["trained_hidden_errors"] = {
            name: differences(value, expected)
            for name, value in (
                ("capture", actual),
                ("hidden_only", outputs["hidden_only"]),
                ("routing", outputs["routing"][0]),
            )
        }
        result["shared_pre_error_per_layer"] = [
            differences(a, b) for a, b in zip(capture.shared_preactivations, outputs["routing"][1], strict=True)
        ]
        result["route_mismatch_fraction_per_layer"] = [
            float(jnp.mean(a != b)) for a, b in zip(capture.routing_selected, outputs["routing"][2], strict=True)
        ]
        result["trained_forward_max_error"] = result["trained_hidden_errors"]["capture"]["max"]
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if args.capture_groups:
            result["capture_group_errors"] = {}
            for label, fields in GROUPS:
                if label not in args.capture_groups:
                    continue
                executable = eqx.filter_jit(partial(capture_subset, fields=fields))
                value, subset = executable(model, tokens, batch.attn_mask)
                jax.block_until_ready((value, subset))
                result["capture_group_errors"][label] = differences(value, expected)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(
                    json.dumps({"group": label, **result["capture_group_errors"][label]}),
                    flush=True,
                )
                del executable, value, subset
        result["bf16_capture_screen"] = capture_screen(
            model,
            actual,
            expected,
            tokens,
            weight,
            capture.routing_selected,
            outputs["routing"][2],
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if not result["bf16_capture_screen"]["passed"]:
            raise AssertionError("Capture failed the BF16 aggregate/loss/routing screen")
        result["device_memory"] = [d.memory_stats() for d in devices]
    result["process_work_seconds"] = time.perf_counter() - began
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("manifest", "data_manifest", "device_memory")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
