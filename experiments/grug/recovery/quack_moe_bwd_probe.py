#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Production-faithful single-GPU probe of the ragged EP expert MLP (issue #8870).

Runs ``_expert_mlp_quack_wgrad`` forward and backward, the exact six-GEMM QuACK sequence the
ragged all-to-all backend executes per MoE layer, with ``cu_seqlens`` built inside the jitted
step from per-expert active sizes the way ``ep_ragged_all_to_all`` does. Every iteration draws a
new random routing so a warp that decodes a stale ``cu_seqlens`` is visibly wrong, and a watchdog
reports a hang instead of waiting forever.

Before the loop it prints the optimized HLO instruction order of the step so the kernel that
writes ``cu_seqlens`` can be located relative to each grouped-GEMM custom call.

``--fanout`` runs one process per visible GPU, one per ``--pdl-arms`` entry in sequence, and prints
the per-arm median seconds per iteration so the cost of launching without programmatic dependent
launch can be read off the same GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

RESULT_PREFIX = "RESULT_JSON "
HANG_EXIT_CODE = 3

DEFAULT_ROWS = 601_088
DEFAULT_EXPERTS = 6
DEFAULT_HIDDEN = 6_144
DEFAULT_INTERMEDIATE = 3_072


def _print_schedule(compiled_text: str) -> None:
    """Print the entry computation's instruction order, marking custom calls and fusions."""
    entry = re.search(r"\nENTRY [^\n]*\{\n(.*?)\n\}", compiled_text, re.S)
    if entry is None:
        print("SCHEDULE: no ENTRY computation found", flush=True)
        return
    lines = [line.strip() for line in entry.group(1).splitlines() if line.strip()]
    print(f"SCHEDULE: {len(lines)} entry instructions", flush=True)
    for index, line in enumerate(lines):
        name = line.split(" = ", 1)[0]
        if " custom-call(" in line:
            target = re.search(r'custom_call_target="([^"]*)"', line)
            operands = re.search(r"custom-call\(([^)]*)\)", line)
            print(
                f"SCHED {index:4d} CUSTOM {name} target={target.group(1) if target else '?'} "
                f"operands=({operands.group(1) if operands else ''})",
                flush=True,
            )
        elif " fusion(" in line:
            kind = re.search(r"kind=(\w+)", line)
            calls = re.search(r"calls=%?([\w.\-]+)", line)
            shape = line.split(" = ", 1)[1].split(" ", 1)[0]
            operands = re.search(r"fusion\(([^)]*)\)", line)
            print(
                f"SCHED {index:4d} FUSION {name} {shape} kind={kind.group(1) if kind else '?'} "
                f"calls={calls.group(1) if calls else '?'} operands=({operands.group(1) if operands else ''})",
                flush=True,
            )
        elif any(
            op in line for op in (" copy-start(", " copy-done(", " all-to-all", " ragged-all-to-all", " send(", " recv(")
        ):
            print(f"SCHED {index:4d} OTHER {line[:160]}", flush=True)


def _run(args: argparse.Namespace) -> int:
    # Imported here rather than at module level: the ``--fanout`` parent must never import JAX
    # (it would preallocate every GPU), and the QuACK stack only exists in the CUDA 13 GPU extra.
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from levanter.grug._moe import quack_moe_cute  # noqa: PLC0415
    from levanter.grug._moe.sonic_cute import _expert_mlp_quack_wgrad  # noqa: PLC0415

    # The launchers read this constant when they are first built, which happens at the first trace below.
    quack_moe_cute._QUACK_USE_PDL = args.pdl == "on"
    print(f"pdl={args.pdl}", flush=True)

    rows, experts, hidden, inter = args.rows, args.experts, args.hidden, args.intermediate

    def expert_mlp(x, w13_il, w2, active_group_sizes):
        # Mirrors ep_ragged_all_to_all._quack_expert_mlp.
        cu = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(active_group_sizes).astype(jnp.int32)])
        return _expert_mlp_quack_wgrad(x, w13_il, w2, cu)

    def loss(x, w13_il, w2, active_group_sizes, cotangent):
        y = expert_mlp(x, w13_il, w2, active_group_sizes)
        return jnp.sum(y.astype(jnp.float32) * cotangent), y

    @jax.jit
    def step(x, w13_il, w2, active_group_sizes, cotangent):
        (value, y), grads = jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True)(
            x, w13_il, w2, active_group_sizes, cotangent
        )
        return value, y, grads

    key = jax.random.PRNGKey(args.seed)
    k_x, k_w13, k_w2, k_c = jax.random.split(key, 4)
    x = jax.random.normal(k_x, (rows, hidden), jnp.bfloat16)
    w13_il = (jax.random.normal(k_w13, (experts, hidden, 2 * inter), jnp.float32) / np.sqrt(hidden)).astype(jnp.bfloat16)
    w2 = (jax.random.normal(k_w2, (experts, inter, hidden), jnp.float32) / np.sqrt(inter)).astype(jnp.bfloat16)
    cotangent = jax.random.normal(k_c, (rows, hidden), jnp.float32)
    rng = np.random.default_rng(args.seed)

    def random_routing() -> np.ndarray:
        # Dirichlet split of a random fill fraction of the receiver buffer over the local experts.
        fill = rng.uniform(0.05, 0.98)
        fractions = rng.dirichlet(np.ones(experts) * 0.7)
        return np.floor(fractions * fill * rows).astype(np.int32)

    if args.dump_schedule:
        sizes = jnp.asarray(random_routing())
        lowered = step.lower(x, w13_il, w2, sizes, cotangent)
        _print_schedule(lowered.compile().as_text())

    progress = {"time": time.time(), "iteration": -1}
    stop = threading.Event()

    def watchdog() -> None:
        while not stop.wait(1.0):
            if time.time() - progress["time"] > args.timeout:
                print(
                    RESULT_PREFIX
                    + json.dumps(
                        {
                            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
                            "pdl": args.pdl,
                            "outcome": "hang",
                            "iteration": progress["iteration"],
                            "hangs": 1,
                            "nonfinite": 0,
                        }
                    ),
                    flush=True,
                )
                os._exit(HANG_EXIT_CODE)

    threading.Thread(target=watchdog, daemon=True).start()

    nonfinite = 0
    t_start = time.time()
    completed = 0
    for iteration in range(args.iters):
        sizes = jnp.asarray(random_routing())
        progress["iteration"] = iteration
        progress["time"] = time.time()
        value, _, _ = step(x, w13_il, w2, sizes, cotangent)
        value = float(value)
        completed += 1
        if iteration == 0:
            print(f"first step (compile) took {time.time() - t_start:.1f}s", flush=True)
            t_start = time.time()
        if not np.isfinite(value):
            nonfinite += 1
            print(f"NONFINITE iteration={iteration} value={value}", flush=True)
        progress["time"] = time.time()
    elapsed = time.time() - t_start
    stop.set()
    print(
        RESULT_PREFIX
        + json.dumps(
            {
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
                "pdl": args.pdl,
                "outcome": "ok",
                "iteration": completed,
                "hangs": 0,
                "nonfinite": nonfinite,
                "seconds_per_iter": elapsed / max(completed - 1, 1),
            }
        ),
        flush=True,
    )
    return 0


def _fanout(args: argparse.Namespace) -> int:
    listing = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], check=True, text=True, capture_output=True
    )
    gpus = [int(line) for line in listing.stdout.split() if line.strip()]
    passthrough = [
        "--iters",
        str(args.iters),
        "--rows",
        str(args.rows),
        "--experts",
        str(args.experts),
        "--hidden",
        str(args.hidden),
        "--intermediate",
        str(args.intermediate),
        "--timeout",
        str(args.timeout),
    ]
    results: list[dict] = []
    lock = threading.Lock()

    def run_gpu(gpu: int) -> None:
        for pdl in args.pdl_arms:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            cmd = [sys.executable, os.path.abspath(__file__), "--seed", str(args.seed + gpu), "--pdl", pdl, *passthrough]
            if gpu == gpus[0] and args.dump_schedule:
                cmd.append("--dump-schedule")
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            result = None
            tail: list[str] = []
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                tail = [*tail, line][-30:]
                if line.startswith(RESULT_PREFIX):
                    result = json.loads(line[len(RESULT_PREFIX) :])
                print(f"[gpu{gpu} pdl={pdl}] {line}", flush=True)
            code = proc.wait()
            if result is None:
                result = {
                    "gpu": str(gpu),
                    "pdl": pdl,
                    "outcome": f"exit {code}",
                    "hangs": 0,
                    "nonfinite": 0,
                    "tail": tail[-10:],
                }
            with lock:
                results.append(result)

    threads = [threading.Thread(target=run_gpu, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print("SUMMARY_JSON " + json.dumps(results), flush=True)
    for pdl in args.pdl_arms:
        arm = [r for r in results if r.get("pdl") == pdl]
        times = sorted(r["seconds_per_iter"] for r in arm if "seconds_per_iter" in r)
        median = times[len(times) // 2] if times else float("nan")
        lo = times[0] if times else float("nan")
        hi = times[-1] if times else float("nan")
        print(
            f"TOTAL pdl={pdl} runs={len(arm)} hangs={sum(r.get('hangs', 0) for r in arm)} "
            f"nonfinite={sum(r.get('nonfinite', 0) for r in arm)} "
            f"other={sum(1 for r in arm if r['outcome'] not in ('ok', 'hang'))} "
            f"median_seconds_per_iter={median:.5f} min={lo:.5f} max={hi:.5f}",
            flush=True,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fanout", action="store_true")
    parser.add_argument("--dump-schedule", action="store_true")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--experts", type=int, default=DEFAULT_EXPERTS)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--intermediate", type=int, default=DEFAULT_INTERMEDIATE)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pdl", choices=("on", "off"), default="off", help="launch the QuACK GEMMs with programmatic dependent launch"
    )
    parser.add_argument(
        "--pdl-arms", nargs="+", choices=("on", "off"), default=["on", "off"], help="arms to run per GPU under --fanout"
    )
    args = parser.parse_args()
    if args.fanout:
        return _fanout(args)
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
