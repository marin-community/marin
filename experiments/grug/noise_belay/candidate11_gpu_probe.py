# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded full-architecture GPU forward check; no training or data download."""

import argparse
import json
import os
import time
from pathlib import Path

import equinox as eqx
import jax
import jmp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.attention import AttentionMask

from experiments.grug.fast_track.model import (
    ROUTING_SELECTED_KEY,
    SHARED_PRE_KEY,
    Transformer,
)
from experiments.grug.noise_belay.candidate11 import load_model_config, load_probe_manifest
from experiments.grug.noise_belay.candidate11_boost import measure_boost_hubs
from experiments.grug.noise_belay.candidate11_forward import capture_forward

PROBE_Z_LOSS_WEIGHT = 0.0001


def steady_calls(executable, arguments, repeats):
    """Time repeated synchronized calls after compilation and warmup."""
    durations = []
    for _ in range(repeats):
        began = time.perf_counter()
        output = executable(*arguments)
        jax.block_until_ready(output)
        durations.append(time.perf_counter() - began)
    return output, durations


def native_forward(model, tokens, mask):
    return model(tokens, mask)[0]


def capture_hidden_only(model, tokens, mask):
    return capture_forward(model, tokens, mask)[0]


def native_with_routing(model, tokens, mask):
    hidden, metrics = model(tokens, mask, return_routing=True)
    return hidden, metrics[SHARED_PRE_KEY], metrics[ROUTING_SELECTED_KEY]


def native_training_loss(model, tokens, mask, weights):
    compute = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16").cast_to_compute(model)
    return compute.next_token_loss(
        tokens,
        weights,
        mask=mask,
        reduction="mean",
        logsumexp_weight=PROBE_Z_LOSS_WEIGHT,
        train_terms=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measure-backward", action="store_true")
    parser.add_argument("--measure-boost", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("At least one synchronized timing repeat is required")
    manifest = load_probe_manifest()
    cfg = load_model_config(source_sha256=manifest["model_source_sha256"])
    devices = jax.devices()
    if len(devices) != 8 or any(device.platform != "gpu" for device in devices):
        raise ValueError("This probe requires exactly eight GPUs")
    if args.batch % 8 or args.sequence % 32 or args.sequence > cfg.max_seq_len:
        raise ValueError("Batch must divide over eight GPUs; sequence must fit and align to KDA chunks")
    mesh = Mesh(
        np.asarray(devices).reshape(1, 1, 8, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    result = {
        "kind": "synthetic_forward_parity_probe",
        "manifest": manifest,
        "batch": args.batch,
        "sequence": args.sequence,
        "devices": [device.device_kind for device in devices],
        "jax_version": jax.__version__,
        "repeats": args.repeats,
        "backward_control_benchmark": args.measure_backward,
        "boost_hub_benchmark": args.measure_boost,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with jax.set_mesh(mesh):
        model = eqx.filter_jit(Transformer.init)(cfg, key=jax.random.PRNGKey(0))
        jax.block_until_ready(model)
        result["initialization_seconds"] = time.perf_counter() - started
        result["parameter_count"] = sum(x.size for x in jax.tree.leaves(model) if eqx.is_array(x))
        if result["parameter_count"] != 1_026_488_855:
            raise ValueError(f"Unexpected parameter count: {result['parameter_count']}")
        master_model = model
        model = jmp.get_policy("bf16").cast_to_compute(model)
        shape = (args.batch, args.sequence)
        token_sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
        rng = np.random.default_rng(7)
        tokens = jax.device_put(rng.integers(0, cfg.vocab_size, shape, dtype=np.int32), token_sharding)
        segments = jax.device_put(
            np.broadcast_to(np.arange(args.sequence, dtype=np.int32) // (args.sequence // 2), shape),
            token_sharding,
        )
        mask = AttentionMask(is_causal=True, segment_ids=(segments, segments))
        outputs = []
        diagnostics = {}
        for name, function in (
            ("native", native_forward),
            ("capture", capture_forward),
            ("capture_hidden_only", capture_hidden_only),
            ("native_with_routing", native_with_routing),
        ):
            compiled = eqx.filter_jit(function)
            begin = time.perf_counter()
            executable = compiled.lower(model, tokens, mask).compile()
            output = executable(model, tokens, mask)
            jax.block_until_ready(output)
            result[f"{name}_first_call_seconds"] = time.perf_counter() - begin
            if name in ("native", "capture"):
                (args.output.parent / f"{name}-optimized-hlo.txt").write_text(executable.compiled.as_text())
            output, times = steady_calls(executable, (model, tokens, mask), args.repeats)
            result[f"{name}_steady_seconds"] = times
            result["last_completed_stage"] = name
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            hidden = output if name in ("native", "capture_hidden_only") else output[0]
            outputs.append(np.asarray(hidden, dtype=np.float32))
            if name == "capture":
                capture = output[1]
                diagnostics["capture_shared"] = [np.asarray(x, np.float32) for x in output[1].shared_preactivations]
                diagnostics["capture_selected"] = [np.asarray(x) for x in output[1].routing_selected]
            elif name == "native_with_routing":
                diagnostics["native_shared"] = np.asarray(output[1], np.float32)
                diagnostics["native_selected"] = np.asarray(output[2])
            print(
                json.dumps(
                    {
                        "stage": name,
                        "first_call_seconds": result[f"{name}_first_call_seconds"],
                    }
                ),
                flush=True,
            )
        result["diagnostic_hidden_errors"] = {
            name: {
                "max": float(np.abs(value - outputs[0]).max()),
                "mean": float(np.abs(value - outputs[0]).mean()),
            }
            for name, value in zip(
                ("capture", "capture_hidden_only", "native_with_routing"),
                outputs[1:],
                strict=True,
            )
        }
        result["shared_pre_error_per_layer"] = [
            float(np.abs(a - b).max())
            for a, b in zip(diagnostics["capture_shared"], diagnostics["native_shared"], strict=True)
        ]
        result["route_mismatch_fraction_per_layer"] = [
            float(np.mean(a != b))
            for a, b in zip(
                diagnostics["capture_selected"],
                diagnostics["native_selected"],
                strict=True,
            )
        ]
        difference = np.abs(outputs[0] - outputs[1])
        result["max_absolute_error"] = float(difference.max())
        result["mean_absolute_error"] = float(difference.mean())
        if args.measure_backward:
            np.testing.assert_allclose(outputs[1], outputs[0], atol=1e-5, rtol=1e-5)
            weights = jax.device_put(
                np.broadcast_to(
                    (np.arange(args.sequence) % (args.sequence // 2) != args.sequence // 2 - 1).astype(np.float32),
                    shape,
                ),
                token_sharding,
            )
            loss_values = []
            for name, function in (
                ("training_loss", native_training_loss),
                (
                    "training_loss_and_grad",
                    eqx.filter_value_and_grad(native_training_loss),
                ),
            ):
                begin = time.perf_counter()
                executable = eqx.filter_jit(function).lower(master_model, tokens, mask, weights).compile()
                output = executable(master_model, tokens, mask, weights)
                jax.block_until_ready(output)
                result[f"{name}_first_call_seconds"] = time.perf_counter() - begin
                output, times = steady_calls(executable, (master_model, tokens, mask, weights), args.repeats)
                result[f"{name}_steady_seconds"] = times
                result["last_completed_stage"] = name
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                loss_values.append(float(output if name == "training_loss" else output[0]))
                if name == "training_loss_and_grad":
                    finite = eqx.filter_jit(
                        lambda g: jax.numpy.all(
                            jax.numpy.stack(
                                [jax.numpy.all(jax.numpy.isfinite(x)) for x in jax.tree.leaves(g) if eqx.is_array(x)]
                            )
                        )
                    )(output[1])
                    if not bool(finite):
                        raise ValueError("Backprop control produced non-finite gradients")
                print(
                    json.dumps(
                        {
                            "stage": name,
                            "first_call_seconds": result[f"{name}_first_call_seconds"],
                            "steady_seconds": times,
                        }
                    ),
                    flush=True,
                )
            result["forward_and_backward_losses"] = loss_values
            np.testing.assert_allclose(loss_values[1], loss_values[0], atol=1e-5, rtol=1e-5)
            result["forward_backward_over_forward_loss_ratio"] = float(
                np.median(result["training_loss_and_grad_steady_seconds"])
                / np.median(result["training_loss_steady_seconds"])
            )
        if args.measure_boost:
            np.testing.assert_allclose(outputs[1], outputs[0], atol=1e-5, rtol=1e-5)
            weights = jax.device_put(
                np.broadcast_to(
                    (np.arange(args.sequence) % (args.sequence // 2) != args.sequence // 2 - 1).astype(np.float32),
                    shape,
                ),
                token_sharding,
            )

            def hubs(capture, model, tokens, weights):
                return measure_boost_hubs(
                    capture.stages,
                    tokens,
                    weights,
                    jax.random.PRNGKey(21),
                    head=model.output_proj,
                    norm_gain=model.final_norm.weight,
                    query=model.attn_res_query_final,
                    epsilon=cfg.layer_norm_eps,
                    soft_cap=10.0,
                    relative_step=0.005,
                    token_chunk=64,
                    direction_chunk=16,
                    z_loss_weight=PROBE_Z_LOSS_WEIGHT,
                )

            begin = time.perf_counter()
            executable = eqx.filter_jit(hubs).lower(capture, model, tokens, weights).compile()
            hub_credits = executable(capture, model, tokens, weights)
            jax.block_until_ready(hub_credits)
            result["boost_hubs_first_call_seconds"] = time.perf_counter() - begin
            hub_credits, times = steady_calls(executable, (capture, model, tokens, weights), args.repeats)
            result["boost_hubs_steady_seconds"] = times
            result["boost_hub_max_absolute_credit"] = [float(jax.numpy.max(jax.numpy.abs(x))) for x in hub_credits]
            if not all(np.isfinite(x) and x > 0 for x in result["boost_hub_max_absolute_credit"]):
                raise ValueError("Boost hub measurement produced zero or non-finite hub_credits")
            print(
                json.dumps(
                    {
                        "stage": "boost_hubs",
                        "first_call_seconds": result["boost_hubs_first_call_seconds"],
                        "steady_seconds": times,
                    }
                ),
                flush=True,
            )
        result["device_memory"] = [device.memory_stats() for device in devices]
    result["process_work_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("manifest", "device_memory")}),
        flush=True,
    )
    np.testing.assert_allclose(outputs[1], outputs[0], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    main()
