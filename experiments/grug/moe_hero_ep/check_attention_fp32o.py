# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-GPU SM100 checks for the FP32 saved-output attention backward.

``python -m experiments.grug.moe_hero_ep.check_attention_fp32o`` runs every check, each
forward schedule in a subprocess with a timeout (a bad KV pipeline depth deadlocks):

- the BF16 output and LSE of each FP32-output schedule against the production forward, bitwise;
- dQ/dK/dV of the two-key cancellation case and a random packed case against an FP64 reference;
- finite gradients and forward/backward time at a hero-like 16k shape.
"""

import json
import subprocess
import sys
import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.attention._fa4_cute_backend import fa4_cute_attention_forward, segmented_flash_attention_forward
from levanter.grug.attention._fa4_cute_config import sm100_flash4_cute_kernel_config
from levanter.grug.attention._fa4_cute_saved_output import _fp32_output_forward, fp32_saved_output_attention_forward

SCALE = 128**-0.5
SCHEDULES = {
    "two_stage_overlap": {"q_stage": 2, "overlap_output_with_query": True},
    "one_stage": {"q_stage": 1, "overlap_output_with_query": False},
}
SCHEDULE_TIMEOUT = 900


def _case(name: str):
    if name == "two_key":
        length, heads = 128, 4
        selected = jnp.asarray([31, 63, 95, 127])
        q = jnp.zeros((1, length, heads, 128), jnp.bfloat16).at[:, selected, :, 0].set(jnp.bfloat16(10 * 128**0.5))
        k = jnp.zeros((1, length, heads // 4, 128), jnp.bfloat16).at[:, selected, :, 0].set(1)
        v = jnp.zeros((1, length, heads // 4, 128), jnp.bfloat16).at[:, selected, :, 0].set(1)
        lower = jnp.maximum(jnp.arange(length) - 1, 0)[None].astype(jnp.int32)
        valid = jnp.zeros((1, length), jnp.bool_).at[:, selected].set(True)
    else:
        length, heads, window, q_gain = {
            "packed_1k": (1024, 8, 300, 1.0),
            "peaked_1k": (1024, 8, None, 4.0),
            "hero_global_16k": (16384, 48, None, 1.0),
            "hero_local_16k": (16384, 48, 2048, 1.0),
        }[name]
        keys = jax.random.split(jax.random.PRNGKey(701), 4)
        q = (jax.random.normal(keys[0], (1, length, heads, 128)) * 0.5 * q_gain).astype(jnp.bfloat16)
        k = (jax.random.normal(keys[1], (1, length, heads // 4, 128)) * 0.5 * q_gain).astype(jnp.bfloat16)
        v = jax.random.normal(keys[2], (1, length, heads // 4, 128)).astype(jnp.bfloat16)
        positions = jnp.arange(length)
        # Two packed documents per sequence, as in hero batches.
        doc_start = (positions // (length // 2)) * (length // 2)
        lower = doc_start if window is None else jnp.maximum(doc_start, positions - window + 1)
        lower = lower[None].astype(jnp.int32)
        valid = jnp.ones((1, length), jnp.bool_)
    dout = jax.random.normal(jax.random.PRNGKey(704), q.shape).astype(jnp.bfloat16)
    return q, k, v, lower, valid, dout


def _offset():
    return jnp.zeros((1,), jnp.int32)


def _forward_schedule(schedule: str, case: str) -> dict:
    q, k, v, lower, valid, _ = _case(case)
    config = sm100_flash4_cute_kernel_config()
    base_out, base_lse = jax.jit(
        lambda *a: segmented_flash_attention_forward(*a, softmax_scale=SCALE, kernel_config=config, q_offset=_offset())
    )(q, k, v, lower, valid)
    fwd_config = replace(config.sm100_forward, fp32_saved_output=True, **SCHEDULES[schedule])
    out32, lse = jax.jit(lambda *a: _fp32_output_forward(*a, sm_scale=SCALE, config=fwd_config, q_offset=_offset()))(
        q, k, v, lower, valid
    )
    base_out, base_lse, out32, lse = (np.asarray(x) for x in jax.device_get((base_out, base_lse, out32, lse)))
    cast = np.asarray(jnp.asarray(out32).astype(jnp.bfloat16))
    rows = np.asarray(valid)[0]
    mismatch = cast[:, rows].view(np.uint16) != base_out[:, rows].view(np.uint16)
    lse_valid = lse[:, :, rows]
    return {
        "schedule": schedule,
        "case": case,
        "output_bf16_mismatch": int(mismatch.sum()),
        "output_components": int(mismatch.size),
        "lse_max_abs_diff": float(np.max(np.abs(lse_valid - base_lse[:, :, rows]))),
        "fp32_minus_bf16_max_abs": float(np.max(np.abs(out32[:, rows] - base_out[:, rows].astype(np.float32)))),
    }


def _reference(q, k, v, lower, valid, dout):
    """FP64 attention output and gradients from the same BF16 operands."""
    q, k, v, dout = (np.asarray(jnp.asarray(x, jnp.float32), np.float64)[0] for x in (q, k, v, dout))
    lower, valid = np.asarray(lower)[0], np.asarray(valid)[0]
    length, heads, _ = q.shape
    ratio = heads // k.shape[1]
    out = np.zeros_like(q)
    dq, dk, dv = np.zeros_like(q), np.zeros_like(k), np.zeros_like(v)
    keys = np.arange(length)
    for h in range(heads):
        g = h // ratio
        s = q[:, h] @ k[:, g].T * SCALE
        allowed = (keys[None] >= lower[:, None]) & (keys[None] <= keys[:, None]) & valid[:, None]
        s = np.where(allowed, s, -np.inf)
        m = np.max(np.where(allowed, s, -1e300), axis=1, keepdims=True)
        p = np.where(allowed, np.exp(s - m), 0.0)
        p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-300)
        out[:, h] = p @ v[:, g]
        dp = dout[:, h] @ v[:, g].T
        d = np.sum(dout[:, h] * out[:, h], axis=1, keepdims=True)
        ds = p * (dp - d)
        dq[:, h] = ds @ k[:, g] * SCALE
        dk[:, g] += ds.T @ q[:, h] * SCALE
        dv[:, g] += p.T @ dout[:, h]
    return dq, dk, dv


def _relative_error(actual, reference) -> float:
    actual = np.asarray(jnp.asarray(actual, jnp.float32), np.float64)[0]
    denominator = np.sqrt(np.sum(reference**2))
    return float(np.sqrt(np.sum((actual - reference) ** 2)) / denominator) if denominator > 0 else float("nan")


def _gradients(attention, q, k, v, lower, valid, dout):
    config = sm100_flash4_cute_kernel_config()
    if attention is fp32_saved_output_attention_forward:
        config = replace(
            config, sm100_forward=replace(config.sm100_forward, fp32_saved_output=True, overlap_output_with_query=True)
        )

    def f(q, k, v):
        return attention(q, k, v, lower, valid, sm_scale=SCALE, kernel_config=config, q_offset=_offset())

    @jax.jit
    def run(q, k, v, dout):
        out, vjp = jax.vjp(f, q, k, v)
        return out, *vjp(dout)

    return run, run(q, k, v, dout)


def _gradient_checks(case: str) -> dict:
    q, k, v, lower, valid, dout = _case(case)
    reference = _reference(q, k, v, lower, valid, dout)
    result = {"case": case}
    for name, attention in (("baseline", fa4_cute_attention_forward), ("fp32o", fp32_saved_output_attention_forward)):
        _, (out, dq, dk, dv) = _gradients(attention, q, k, v, lower, valid, dout)
        result[name] = {
            "dq_rel_l2": _relative_error(dq, reference[0]),
            "dk_rel_l2": _relative_error(dk, reference[1]),
            "dv_rel_l2": _relative_error(dv, reference[2]),
        }
        result[f"{name}_output"] = np.asarray(jnp.asarray(out, jnp.float32))
    result["forward_bitwise_equal"] = bool(np.array_equal(result.pop("baseline_output"), result.pop("fp32o_output")))
    return result


def _time(fn, *args, iterations: int = 20) -> float:
    jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - start)
    return float(np.median(samples) * 1e3)


def _hero_checks(case: str) -> dict:
    q, k, v, lower, valid, dout = _case(case)
    result = {"case": case}
    outputs = {}
    for name, attention in (("baseline", fa4_cute_attention_forward), ("fp32o", fp32_saved_output_attention_forward)):
        run, (out, dq, dk, dv) = _gradients(attention, q, k, v, lower, valid, dout)
        outputs[name] = (out, dq, dk)
        finite = all(bool(jnp.all(jnp.isfinite(x.astype(jnp.float32)))) for x in (out, dq, dk, dv))
        result[name] = {"finite": finite, "fwd_bwd_ms": _time(run, q, k, v, dout)}
    result["forward_bitwise_equal"] = bool(
        np.array_equal(
            np.asarray(outputs["baseline"][0]).view(np.uint16), np.asarray(outputs["fp32o"][0]).view(np.uint16)
        )
    )
    for i, grad in ((1, "dq"), (2, "dk")):
        a, b = (np.asarray(jnp.asarray(outputs[n][i], jnp.float32), np.float64) for n in ("baseline", "fp32o"))
        result[f"{grad}_rel_l2_fp32o_vs_baseline"] = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    return result


def _run_child(args: list[str], timeout: int) -> dict:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "experiments.grug.moe_hero_ep.check_attention_fp32o", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"args": args, "error": f"timeout after {timeout}s (deadlock?)"}
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    if proc.returncode != 0 or not lines:
        return {"args": args, "error": f"exit {proc.returncode}", "stderr_tail": proc.stderr[-3000:]}
    return json.loads(lines[-1][len("RESULT ") :])


def main() -> None:
    if len(sys.argv) > 1:
        kind, *rest = sys.argv[1:]
        if kind == "forward":
            result = _forward_schedule(*rest)
        elif kind == "gradients":
            result = _gradient_checks(*rest)
        else:
            result = _hero_checks(*rest)
        print("RESULT " + json.dumps(result), flush=True)
        return
    print(f"devices {jax.devices()}", flush=True)
    for schedule in SCHEDULES:
        for case in ("two_key", "packed_1k", "peaked_1k", "hero_global_16k", "hero_local_16k"):
            print("CHECK " + json.dumps(_run_child(["forward", schedule, case], SCHEDULE_TIMEOUT)), flush=True)
    for case in ("two_key", "packed_1k", "peaked_1k"):
        print("CHECK " + json.dumps(_run_child(["gradients", case], SCHEDULE_TIMEOUT)), flush=True)
    for case in ("hero_global_16k", "hero_local_16k"):
        print("CHECK " + json.dumps(_run_child(["hero", case], SCHEDULE_TIMEOUT)), flush=True)


if __name__ == "__main__":
    main()
