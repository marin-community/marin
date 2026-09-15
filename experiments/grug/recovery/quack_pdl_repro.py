#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU probe for a QuACK varlen grouped-GEMM hang under programmatic dependent launch.

Issue #8870's September 15 GPU capture found the missing ragged-all-to-all peer stuck inside
QuACK's SM100 grouped GEMM: the MMA and epilogue warps of every resident CTA had retired while
the TMA-load and CLC-scheduler warps were still working on a tile. Every warp role decodes its
work tile independently by reading ``cu_seqlens`` from global memory, and only the load and
scheduler warps execute ``griddepcontrol.wait`` first. Under PDL the other warps can read the
group boundaries before the preceding kernel's writes are visible, and a stale decode splits
the warp roles: one side retires, the other blocks forever on an intra-CTA pipeline.

This script drives that exact kernel (production tile/cluster/CLC settings, production
``cu_seqlens`` construction) with a routing that changes every iteration, so a stale read of
the previous routing is visibly wrong, and counts hangs and corrupted outputs per arm:

  prod      production kernel configuration, launched with PDL
  nopdl     production kernel launched without PDL
  noclc     PDL on, CLC persistence off (static scheduler, single decoder warp)
  patched   PDL on, CLC on, QuACK patched so the MMA/epilogue warps also wait
  constant  PDL on with the same routing every iteration
  nomask    PDL on without the row mask between routing and GEMM

Each arm runs in its own process. ``--fanout`` runs the arm list on every visible GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import tempfile
import time

from experiments.grug.recovery.gpu_probe_harness import (
    ProbeResult,
    Progress,
    child_argv,
    current_gpu,
    print_summary,
    run_arms,
)

# One EP64 rank of the d6144 hero: receiver-buffer rows, local experts, hidden, intermediate.
DEFAULT_ROWS = 601_088
DEFAULT_EXPERTS = 6
DEFAULT_HIDDEN = 6_144
DEFAULT_INTERMEDIATE = 3_072

ARMS = ("prod", "nopdl", "patched", "noclc", "constant", "nomask")

# Fractions of the receiver buffer routed to each local expert. ``BIG`` fills most of the
# buffer, ``SMALL`` a sliver, so a stale decode of one during the other retires warps early
# (stale SMALL) or makes them wait for tiles that never come (stale BIG).
BIG_ROUTING = (0.20, 0.15, 0.25, 0.10, 0.18, 0.07)
SMALL_ROUTING = (0.02, 0.01, 0.03, 0.005, 0.015, 0.01)

MMA_WARP_ANCHOR = "        if warp_idx == self.mma_warp_id:\n"
EPILOGUE_WARP_ANCHOR = "        if warp_idx < self.mma_warp_id:\n"
PDL_WAIT_PATCH = "            if const_expr(self.use_pdl):\n                cute.arch.griddepcontrol_wait()\n"


def _install_patched_quack() -> str:
    """Put a copy of QuACK on ``sys.path`` whose MMA and epilogue warps wait for the prior grid."""
    spec = importlib.util.find_spec("quack")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("quack is not installed")
    source_dir = next(iter(spec.submodule_search_locations))
    root = tempfile.mkdtemp(prefix="quack-pdl-patched-")
    shutil.copytree(source_dir, os.path.join(root, "quack"))
    path = os.path.join(root, "quack", "gemm_sm100.py")
    with open(path) as f:
        text = f.read()
    for anchor in (MMA_WARP_ANCHOR, EPILOGUE_WARP_ANCHOR):
        if text.count(anchor) != 1:
            raise RuntimeError(f"patch anchor not unique in {path}: {anchor!r}")
        text = text.replace(anchor, anchor + PDL_WAIT_PATCH)
    with open(path, "w") as f:
        f.write(text)
    sys.path.insert(0, root)
    return root


def _routing_sizes(rows: int, fractions: tuple[float, ...]) -> list[int]:
    return [int(rows * f) for f in fractions]


def _build_gemm_call(rows: int, inter: int, *, use_pdl: bool, use_clc: bool):
    """Return the production ``dh`` GEMM, ``dy[M, H] @ w2[E, I, H]^T`` grouped over rows, as a JAX call."""
    # Imported here rather than at module level: the ``patched`` arm must put its QuACK copy on
    # ``sys.path`` first, the ``--fanout`` parent must never import JAX (it would preallocate
    # every GPU), and the QuACK stack only exists in the CUDA 13 GPU extra.
    import cutlass  # noqa: PLC0415
    import cutlass.cute as cute  # noqa: PLC0415
    import cutlass.jax as cjax  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import quack  # noqa: PLC0415
    from levanter.cutlass_kernel_cache import cute_launcher_factory, cutlass_call  # noqa: PLC0415
    from levanter.grug._moe.quack_moe_cute import _cute_dtype, _max_active_clusters  # noqa: PLC0415
    from levanter.grug._moe.sonic_cute import _QUACK_GROUPED_KW  # noqa: PLC0415
    from quack.gemm_default_epi import GemmDefaultEpiMixin, GemmDefaultSm100  # noqa: PLC0415
    from quack.gemm_tvm_ffi_utils import make_scheduler_args, make_varlen_args  # noqa: PLC0415

    print(f"quack from {quack.__file__}", flush=True)

    @cute_launcher_factory
    def _build_launcher(
        *, a_dtype, tile_mn, cluster_mnk, max_active_clusters, max_swizzle, use_clc_persistence, use_pdl
    ):
        @cute.jit
        def launcher(stream, mA, mB, mCuSeqlens, mD):
            gemm = GemmDefaultSm100(
                cutlass.Float32,
                a_dtype,
                tile_mn,
                cluster_mnk,
                gather_A=False,
                use_clc_persistence=use_clc_persistence,
                use_pdl=use_pdl,
            )
            epi_args = GemmDefaultEpiMixin.EpilogueArguments()
            scheduler_args = make_scheduler_args(max_active_clusters, max_swizzle, None)
            gemm(mA, mB, mD, None, epi_args, scheduler_args, make_varlen_args(mCuSeqlens, None, None), stream)

        return launcher

    cluster_mnk = _QUACK_GROUPED_KW["cluster_mnk"]
    launcher = _build_launcher(
        a_dtype=_cute_dtype(jnp.bfloat16),
        tile_mn=_QUACK_GROUPED_KW["tile_mn"],
        cluster_mnk=cluster_mnk,
        max_active_clusters=_max_active_clusters(cluster_mnk),
        max_swizzle=8,
        use_clc_persistence=use_clc,
        use_pdl=use_pdl,
    )
    ts = cjax.TensorSpec
    return cutlass_call(
        launcher,
        output_shape_dtype=jax.ShapeDtypeStruct((rows, inter), jnp.bfloat16),
        input_spec=(
            ts(divisibility=(1, 8), static=False),
            ts(mode=(0, 1, 2), divisibility=(1, 1, 8), static=False),
            ts(static=False),
        ),
        output_spec=(ts(divisibility=(1, 8), static=False),),
        use_static_tensors=False,
    )


def _run_arm(args: argparse.Namespace) -> int:
    arm = args.arm
    if arm == "patched":
        print(f"patched quack at {_install_patched_quack()}", flush=True)
    call = _build_gemm_call(args.rows, args.intermediate, use_pdl=arm != "nopdl", use_clc=arm != "noclc")

    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from levanter.grug._moe.common import _zero_inactive_grouped_rows  # noqa: PLC0415

    rows, experts, hidden, inter = args.rows, args.experts, args.hidden, args.intermediate
    mask = arm != "nomask"

    @jax.jit
    def step(dy, w2, sizes):
        cu = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(sizes).astype(jnp.int32)])
        if mask:
            dy = _zero_inactive_grouped_rows(dy, cu)
        return call(dy, w2, cu)

    @jax.jit
    def reference(dy, w2, sizes):
        cu = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(sizes).astype(jnp.int32)])
        if mask:
            dy = _zero_inactive_grouped_rows(dy, cu)
        out = jax.lax.ragged_dot(dy, jnp.swapaxes(w2, 1, 2), sizes, preferred_element_type=jnp.float32)
        return out.astype(jnp.bfloat16)

    def check(out, sizes) -> float:
        active = int(np.sum(np.asarray(sizes)))
        ref = reference(dy, w2, sizes)
        diff = jnp.max(jnp.abs(out[:active].astype(jnp.float32) - ref[:active].astype(jnp.float32)))
        scale = jnp.max(jnp.abs(ref[:active].astype(jnp.float32)))
        return float(diff / scale)

    key = jax.random.PRNGKey(args.seed)
    k_dy, k_w = jax.random.split(key)
    dy = jax.random.normal(k_dy, (rows, hidden), jnp.bfloat16)
    w2 = (jax.random.normal(k_w, (experts, inter, hidden), jnp.float32) / np.sqrt(hidden)).astype(jnp.bfloat16)
    big = jnp.asarray(_routing_sizes(rows, BIG_ROUTING), jnp.int32)
    small = jnp.asarray(_routing_sizes(rows, SMALL_ROUTING), jnp.int32)
    routings = (big, big) if arm == "constant" else (big, small)

    progress = Progress(arm, args.timeout)

    # Compile the kernel and the reference for both routings before timing.
    t_compile = time.time()
    for sizes in routings:
        progress.touch()
        step(dy, w2, sizes).block_until_ready()
        progress.touch()
        reference(dy, w2, sizes).block_until_ready()
    print(f"compiled {arm} in {time.time() - t_compile:.1f}s", flush=True)

    mismatches = 0
    worst_ratio = 0.0
    t_start = time.time()
    completed = 0
    for iteration in range(args.iters):
        sizes = routings[iteration % 2]
        progress.touch(iteration)
        out = step(dy, w2, sizes)
        out.block_until_ready()
        completed += 1
        if iteration % args.check_every == 0:
            progress.touch()
            ratio = check(out, sizes)
            worst_ratio = max(worst_ratio, ratio)
            if ratio > args.mismatch_ratio:
                mismatches += 1
                print(f"MISMATCH arm={arm} iteration={iteration} ratio={ratio:.4f}", flush=True)
        progress.touch()
    elapsed = time.time() - t_start
    progress.finish()
    print(
        ProbeResult(
            arm,
            current_gpu(),
            "ok",
            iterations=completed,
            mismatches=mismatches,
            worst_ratio=worst_ratio,
            seconds_per_iter=elapsed / max(completed, 1),
        ).line(),
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--fanout", action="store_true", help="run --arms on every visible GPU")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--gpus", nargs="+", type=int, default=None)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--experts", type=int, default=DEFAULT_EXPERTS)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--intermediate", type=int, default=DEFAULT_INTERMEDIATE)
    parser.add_argument("--timeout", type=float, default=30.0, help="seconds without progress that count as a hang")
    parser.add_argument("--arm-budget", type=float, default=900.0, help="seconds per arm process before it is killed")
    parser.add_argument("--check-every", type=int, default=5)
    parser.add_argument("--mismatch-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.experts != len(BIG_ROUTING):
        parser.error(f"--experts must be {len(BIG_ROUTING)} to match the routing tables")
    if not args.fanout:
        if args.arm is None:
            parser.error("--arm or --fanout is required")
        return _run_arm(args)

    passthrough = [
        f"--{name.replace('_', '-')}={getattr(args, name)}"
        for name in ("iters", "rows", "experts", "hidden", "intermediate", "timeout", "check_every", "mismatch_ratio")
    ]
    results = run_arms(
        args.arms,
        lambda gpu, arm: child_argv(__file__, ["--arm", arm, "--seed", str(args.seed + gpu), *passthrough]),
        gpus=args.gpus,
        budget=args.arm_budget,
    )
    print_summary(results, args.arms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
