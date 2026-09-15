#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Production-faithful single-GPU probe of the ragged EP expert MLP (issue #8870).

Runs ``_expert_mlp_quack_wgrad`` forward and backward, the exact six-GEMM QuACK sequence the
ragged all-to-all backend executes per MoE layer, with ``cu_seqlens`` built inside the jitted
step from per-expert active sizes the way ``ep_ragged_all_to_all`` does. Every iteration draws a
new random routing so a warp that decodes a stale ``cu_seqlens`` is visibly wrong, and a watchdog
reports a hang instead of waiting forever.

Before the loop it can print the optimized HLO instruction order of the step so the kernel that
writes ``cu_seqlens`` can be located relative to each grouped-GEMM custom call.

The arms are ``on`` and ``off``, the setting of programmatic dependent launch on the QuACK
launchers. ``--fanout`` runs ``--arms`` in sequence on every visible GPU and prints per-arm median
seconds per iteration, so the cost of launching without PDL can be read off the same GPUs.
"""

from __future__ import annotations

import argparse
import re
import sys
import time

from experiments.grug.recovery.gpu_probe_harness import (
    ProbeResult,
    Progress,
    child_argv,
    current_gpu,
    print_summary,
    run_arms,
)

DEFAULT_ROWS = 601_088
DEFAULT_EXPERTS = 6
DEFAULT_HIDDEN = 6_144
DEFAULT_INTERMEDIATE = 3_072
ARMS = ("on", "off")


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


def _select_pdl(use_pdl: bool) -> None:
    """Point the production launchers at the requested PDL setting before any of them is built."""
    from levanter.grug._moe import quack_moe_cute  # noqa: PLC0415

    for factory in (quack_moe_cute._build_launcher, quack_moe_cute._build_plain_launcher):
        if factory.cache_info().currsize:
            raise RuntimeError(f"{factory.__name__} already built a launcher; PDL must be selected first")
    quack_moe_cute._QUACK_USE_PDL = use_pdl
    print(f"pdl={'on' if use_pdl else 'off'}", flush=True)


def _run(args: argparse.Namespace) -> int:
    # Imported here rather than at module level: the ``--fanout`` parent must never import JAX
    # (it would preallocate every GPU), and the QuACK stack only exists in the CUDA 13 GPU extra.
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from levanter.grug._moe.sonic_cute import _expert_mlp_quack_wgrad  # noqa: PLC0415

    _select_pdl(args.pdl == "on")
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

    progress = Progress(args.pdl, args.timeout)
    nonfinite = 0
    t_start = time.time()
    completed = 0
    for iteration in range(args.iters):
        sizes = jnp.asarray(random_routing())
        progress.touch(iteration)
        value, _, _ = step(x, w13_il, w2, sizes, cotangent)
        value = float(value)
        completed += 1
        if iteration == 0:
            print(f"first step (compile) took {time.time() - t_start:.1f}s", flush=True)
            t_start = time.time()
        if not np.isfinite(value):
            nonfinite += 1
            print(f"NONFINITE iteration={iteration} value={value}", flush=True)
        progress.touch()
    elapsed = time.time() - t_start
    progress.finish()
    print(
        ProbeResult(
            args.pdl,
            current_gpu(),
            "ok",
            iterations=completed,
            nonfinite=nonfinite,
            seconds_per_iter=elapsed / max(completed - 1, 1),
        ).line(),
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
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds without progress that count as a hang")
    parser.add_argument("--arm-budget", type=float, default=900.0, help="seconds per arm process before it is killed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pdl", choices=ARMS, default="off", help="PDL setting for this process's QuACK launchers")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS), help="PDL settings to run per GPU")
    args = parser.parse_args()
    if not args.fanout:
        return _run(args)

    passthrough = [
        f"--{name}={getattr(args, name)}" for name in ("iters", "rows", "experts", "hidden", "intermediate", "timeout")
    ]

    def build_command(gpu: int, arm: str) -> list[str]:
        extra = ["--seed", str(args.seed + gpu), "--pdl", arm, *passthrough]
        if args.dump_schedule and gpu == 0:
            extra.append("--dump-schedule")
        return child_argv(__file__, extra)

    results = run_arms(args.arms, build_command, budget=args.arm_budget)
    print_summary(results, args.arms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
