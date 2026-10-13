# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep FA4/CuTe port or native SM100 tile configurations at Grug hero attention shapes.

Times a sliding-window and a full-causal mask against a reference configuration and gates every
candidate on reproducing ``reference_attention`` in float32.

``--backend native-sm100`` sweeps native forward tiles/query stages or backward tiles.
It bypasses the production backward schedule allowlist only within this benchmark. Each native
candidate runs in a fresh process and must pass output/gradient checks before timing. JSONL
records include full configs, package versions, source hashes, and rejected candidates.

For the current hero shapes, use ``--batch 16 --q-heads 48 --kv-heads 6 --sliding-window 2048``.

With ``--backend port``, ``--sweep forward`` varies the forward tile with port backward. ``--sweep backward``
varies the segmented tile, thread count, and path, and lifts the
``_segmented_backward_arches`` allowlist to do it, since that function is narrower than the
kernel's own ``can_implement``.

Lifting the allowlist is a benchmark-only affordance. Measured on GB200 at head dimension 128,
every backward outside it is slower, wrong, or unlaunchable: 192x64 and 256x64 pass
``can_implement`` and return gradients off by four orders of magnitude, and 256 threads does the
same at 128x64 and 128x128.

The two backward paths differ in more than tiles. ``path_arch=120`` runs
``SegmentedFlashAttentionBackwardSm120`` with ``num_stages_Q = num_stages_dO = 1`` at head
dimension 128 -- no double buffering of the Q and dO loads -- and a 4-warp atom layout that wants
128 threads. ``path_arch=80`` runs the SM80 class with both stage counts at 2 and a 2-warp atom
layout, which is how the double-buffering question gets answered without changing library code.
On GB200 the two measure the same at 64x64/128, and every larger tile on the SM80 path exceeds
the 232448-byte shared-memory limit.

One process sweeps one backward path. ``cute_launcher_factory`` memoizes on the launcher's
keyword arguments, and the path is derived inside the launcher rather than passed to it, so
benching two paths at one tile in a single process silently returns the first path's kernel for
both. ``--shard``/``--num-shards`` split the candidate list so one 4-GPU node can run four
disjoint slices concurrently, one process per GPU.
"""

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from levanter.cutlass_kernel_cache import gpu_compute_capability
from levanter.grug.attention import AttentionMask, reference_attention
from levanter.grug.attention import _fa4_cute as fa4_cute
from levanter.grug.attention import _fa4_cute_kernels as fa4_cute_kernels
from levanter.grug.attention._fa4_cute_config import (
    SM100_GQA_RATIOS,
    SM100_HEAD_DIM,
    Flash4CuteKernelConfig,
    Flash4CuteSm100ForwardConfig,
    flash4_cute_kernel_config,
    sm100_flash4_cute_kernel_config,
)

REFERENCE_TILE = (64, 64)
REFERENCE_NUM_THREADS = 128
CANDIDATE_FORWARD_TILES = ((64, 64), (64, 128), (128, 32), (128, 64), (128, 128), (192, 64), (256, 64))
CANDIDATE_BACKWARD_TILES = ((64, 64), (64, 128), (128, 64), (128, 128), (192, 64), (256, 64))
NATIVE_FORWARD_TILES = ((64, 64), (64, 128), (128, 64), (128, 128), (128, 192), (128, 256), (256, 64), (256, 128))
NATIVE_BACKWARD_TILES = ((64, 64), (64, 128), (128, 64), (128, 128), (128, 192), (128, 256), (256, 64), (256, 128))
CANDIDATE_NUM_THREADS = (128, 256)
# Port path 120 uses stages 1/1 at head_dim 128 and 4-warp atoms; 80 is double-buffered.
CANDIDATE_BACKWARD_PATHS = (120, 80)


@dataclass(frozen=True)
class Candidate:
    """A kernel configuration plus the backward path to force it through."""

    config: Flash4CuteKernelConfig
    backward_path: int


def _segment_ids(batch: int, seq_len: int, documents: int) -> jax.Array:
    """Evenly spaced document boundaries, matching the corpus density of ~5 per 4096 tokens."""
    boundaries = np.linspace(0, seq_len, documents + 1).astype(np.int32)[1:-1]
    ids = np.zeros((batch, seq_len), dtype=np.int32)
    for b in range(batch):
        # Stagger each row so tiles do not share identical boundaries.
        ids[b] = np.searchsorted(boundaries, np.arange(seq_len) - b % 64, side="right")
    return jnp.asarray(ids)


def _loss(q, k, v, mask):
    return jnp.sum(fa4_cute.gpu_fa4_cute_attention(q, k, v, mask).astype(jnp.float32) ** 2)


@dataclass(frozen=True)
class BenchResult:
    """One timed configuration. ``backward`` is the grad-of-loss time net of the forward it re-runs."""

    forward: float
    backward: float


@contextlib.contextmanager
def _forced(candidate: Candidate):
    """Force the kernel configuration and the backward path for the duration of the block.

    Overriding ``_segmented_backward_arches`` bypasses its allowlist, so an unsupported
    combination surfaces as the kernel's own rejection rather than the dispatcher's.
    """
    selection = fa4_cute_kernels._BackwardArchSelection(
        path_arch=candidate.backward_path, postprocess_arch=candidate.backward_path
    )
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(fa4_cute, "_segmented_kernel_config", lambda head_dim: candidate.config))
        stack.enter_context(patch.object(fa4_cute_kernels, "_segmented_backward_arches", lambda **_: selection))
        # Benchmark-only escape hatch; the production allowlist remains unchanged.
        if candidate.config.sm100_forward is not None:
            stack.enter_context(patch.object(fa4_cute_kernels, "_validate_sm100_backward_config", lambda config: None))
        yield


def _seconds_per_step(call, steps: int, warmup: int) -> float:
    for _ in range(warmup):
        jax.block_until_ready(call())
    start = time.perf_counter()
    for _ in range(steps):
        jax.block_until_ready(call())
    return (time.perf_counter() - start) / steps


def _bench(candidate: Candidate, q, k, v, mask, steps: int, warmup: int) -> BenchResult:
    with _forced(candidate):
        forward = jax.jit(lambda q, k, v: fa4_cute.gpu_fa4_cute_attention(q, k, v, mask))
        grad = jax.jit(jax.grad(_loss, argnums=(0, 1, 2)))

        out = forward(q, k, v)
        out.block_until_ready()
        grads = grad(q, k, v, mask)
        jax.block_until_ready(grads)

        forward_time = _seconds_per_step(lambda: forward(q, k, v), steps, warmup)
        grad_time = _seconds_per_step(lambda: grad(q, k, v, mask), steps, warmup)

    return BenchResult(forward=forward_time, backward=grad_time - forward_time)


def _check_against_float32_reference(
    candidate: Candidate,
    window: int | None,
    args: argparse.Namespace,
) -> str:
    """Compare a candidate against the float32 reference at the tolerances the GPU tests use.

    Timing runs at hero shapes, where bf16 gradients are large enough that an absolute
    difference between two tile configurations says nothing about correctness. This is the
    check that decides whether a configuration is usable. It reuses the swept head counts and
    document count, adds padded rows/tokens, and shrinks the window so local masking is exercised.
    It shrinks batch and sequence length because the reference materializes
    the full score matrix and is quadratic in sequence length.

    Returns ``"ok"``, or ``"FAIL:"`` followed by the comma-separated tensors that disagreed,
    named among ``out``, ``dq``, ``dk``, and ``dv``.
    """
    seq_len = args.check_seq_len
    key = jax.random.key(11)
    check_batch = 3
    check_window = min(window, max(1, seq_len // (2 * args.documents))) if window is not None else None
    q = jax.random.normal(key, (check_batch, seq_len, args.q_heads, args.head_dim), dtype=jnp.bfloat16)
    kv_shape = (check_batch, seq_len, args.kv_heads, args.head_dim)
    k = jax.random.normal(jax.random.fold_in(key, 1), kv_shape, dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.fold_in(key, 2), kv_shape, dtype=jnp.bfloat16)
    cotangent = jax.random.normal(jax.random.fold_in(key, 3), q.shape, dtype=jnp.bfloat16)
    segment_ids = _segment_ids(check_batch, seq_len, args.documents)
    segment_ids = segment_ids.at[0, :13].set(-1).at[0, -17:].set(-1).at[-1, :].set(-1)
    mask = AttentionMask.causal(sliding_window=check_window).with_segment_ids(segment_ids)

    def fa4_loss(q_arg, k_arg, v_arg):
        out = fa4_cute.gpu_fa4_cute_attention(q_arg, k_arg, v_arg, mask)
        return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

    def ref_loss(q_arg, k_arg, v_arg):
        out = reference_attention(q_arg, k_arg, v_arg, mask, logits_dtype=jnp.float32)
        out = jnp.where((segment_ids >= 0)[..., None, None], out, 0)
        return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

    expected = reference_attention(q, k, v, mask, logits_dtype=jnp.float32)
    expected = jnp.where((segment_ids >= 0)[..., None, None], expected, 0)
    expected_grads = jax.jit(jax.grad(ref_loss, argnums=(0, 1, 2)))(q, k, v)
    with _forced(candidate):
        # A fresh closure prevents JAX from reusing a trace made under another forced config.
        actual = jax.jit(lambda q, k, v: fa4_cute.gpu_fa4_cute_attention(q, k, v, mask))(q, k, v)
        actual_grads = jax.jit(jax.grad(fa4_loss, argnums=(0, 1, 2)))(q, k, v)

    failures = []
    for name, got, want in [
        ("out", actual, expected),
        ("dq", actual_grads[0], expected_grads[0]),
        ("dk", actual_grads[1], expected_grads[1]),
        ("dv", actual_grads[2], expected_grads[2]),
    ]:
        if not bool(jnp.all(jnp.isfinite(got))):
            failures.append(name)
            continue
        try:
            np.testing.assert_allclose(
                np.asarray(got, dtype=np.float32), np.asarray(want, dtype=np.float32), atol=7e-2, rtol=7e-2
            )
        except AssertionError:
            failures.append(name)
    return "ok" if not failures else "FAIL:" + ",".join(failures)


def _build_candidates(
    base: Flash4CuteKernelConfig, sweep: str, backward_path: int, backend: str = "port"
) -> list[Candidate]:
    """Candidates for one sweep axis, holding the other axis at its production value.

    A full cross product of forward and backward tiles wastes most of its runs: the two kernels
    are timed separately, so varying both at once only re-measures the same pairs.
    """
    if backend == "native-sm100":
        assert base.sm100_backward is not None
        if sweep == "forward":
            return [
                Candidate(dataclasses.replace(base, sm100_forward=Flash4CuteSm100ForwardConfig(tile, stage)), 100)
                for tile in NATIVE_FORWARD_TILES
                for stage in (1, 2)
            ]
        return [
            Candidate(dataclasses.replace(base, sm100_backward=dataclasses.replace(base.sm100_backward, tile=tile)), 100)
            for tile in NATIVE_BACKWARD_TILES
        ]
    if sweep == "forward":
        return [
            Candidate(
                dataclasses.replace(
                    base, forward_tile=tile, backward_tile=REFERENCE_TILE, num_threads=REFERENCE_NUM_THREADS
                ),
                backward_path,
            )
            for tile in CANDIDATE_FORWARD_TILES
        ]
    return [
        Candidate(
            dataclasses.replace(
                base, forward_tile=base.forward_tile, backward_tile=tile, num_threads=threads, sm100_backward=None
            ),
            backward_path,
        )
        for tile in CANDIDATE_BACKWARD_TILES
        for threads in CANDIDATE_NUM_THREADS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("port", "native-sm100"), default="port")
    parser.add_argument("--candidate-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--candidate-timeout", type=int, default=300, help="Seconds allowed per isolated candidate.")
    parser.add_argument(
        "--output", type=Path, help="Append JSONL results; defaults to IRIS_OUTPUT_DIR/tile-sweep.jsonl."
    )
    parser.add_argument(
        "--batch", type=int, default=32, help="Per-GPU batch; pass the target training shape explicitly."
    )
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--documents", type=int, default=5)
    parser.add_argument("--sliding-window", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--check-seq-len",
        type=int,
        default=513,
        help="Sequence length for the float32 reference check; the reference is quadratic in it.",
    )
    parser.add_argument(
        "--sweep",
        choices=("forward", "backward"),
        default="forward",
        help="Which tile to vary. 'backward' also varies threads and backward path.",
    )
    parser.add_argument(
        "--backward-path",
        type=int,
        choices=CANDIDATE_BACKWARD_PATHS,
        default=120,
        help=(
            "Backward path to force for every candidate including the reference. One process must "
            "use one path: cute_launcher_factory memoizes on the launcher's keyword arguments, and "
            "the path is derived inside rather than passed, so two paths in one process collide."
        ),
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(f"--shard must be in [0, {args.num_shards}), got {args.shard}")

    if args.output is None and "IRIS_OUTPUT_DIR" in os.environ:
        args.output = Path(os.environ["IRIS_OUTPUT_DIR"]) / "tile-sweep.jsonl"
    if args.backend == "native-sm100" and args.candidate_index is None:
        _run_isolated_candidates(args)
    else:
        _run_sweep(args)


def _run_isolated_candidates(args: argparse.Namespace) -> None:
    # Do not initialize JAX in this parent: every child gets the full GPU memory budget.
    count = len(_build_candidates(sm100_flash4_cute_kernel_config(), args.sweep, args.backward_path, args.backend))
    failed = []
    for index in range(args.shard, count, args.num_shards):
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    # sys.argv[0] is this file's path under both `python path` and `python -m`.
                    sys.argv[0],
                    *sys.argv[1:],
                    "--candidate-index",
                    str(index),
                ],
                check=False,
                timeout=args.candidate_timeout,
            )
        except subprocess.TimeoutExpired:
            failed.append(index)
            _emit(args.output, {"kind": "timeout", "candidate_index": index})
            continue
        if result.returncode:
            failed.append(index)
            _emit(args.output, {"kind": "process_failure", "candidate_index": index, "returncode": result.returncode})
    _emit(
        args.output,
        {
            "kind": "sweep_complete",
            "backend": args.backend,
            "sweep": args.sweep,
            "shard": args.shard,
            "num_shards": args.num_shards,
            "failed_processes": failed,
        },
    )
    if failed:
        raise SystemExit(1)


def _run_sweep(args: argparse.Namespace) -> None:
    if jax.default_backend() != "gpu":
        raise SystemExit("tile_sweep requires the JAX GPU backend.")

    key = jax.random.key(0)
    shape_q = (args.batch, args.seq_len, args.q_heads, args.head_dim)
    shape_kv = (args.batch, args.seq_len, args.kv_heads, args.head_dim)
    q = jax.random.normal(key, shape_q, dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), shape_kv, dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.fold_in(key, 2), shape_kv, dtype=jnp.bfloat16)
    segment_ids = _segment_ids(args.batch, args.seq_len, args.documents)

    arch = gpu_compute_capability()
    base = flash4_cute_kernel_config(args.head_dim, arch=arch)
    if args.backend == "native-sm100":
        ratio, remainder = divmod(args.q_heads, args.kv_heads)
        if arch != 100 or args.head_dim != SM100_HEAD_DIM or remainder or ratio not in SM100_GQA_RATIOS:
            raise ValueError(
                f"Native sweep requires SM100, head dimension {SM100_HEAD_DIM}, and a GQA ratio in {SM100_GQA_RATIOS}."
            )
        base = sm100_flash4_cute_kernel_config()
    else:
        base = dataclasses.replace(base, sm90_backward=None)
    print(f"arch=sm{arch} base_forward_tile={base.forward_tile} base_backward_tile={base.backward_tile}")
    print(f"shape: batch={args.batch} seq={args.seq_len} q_heads={args.q_heads} head_dim={args.head_dim}")

    reference = Candidate(
        config=dataclasses.replace(
            base, forward_tile=REFERENCE_TILE, backward_tile=REFERENCE_TILE, num_threads=REFERENCE_NUM_THREADS
        ),
        backward_path=args.backward_path,
    )
    if args.backend == "native-sm100":
        reference = Candidate(base, 100)
    candidates = _build_candidates(base, args.sweep, args.backward_path, args.backend)
    shard = (
        [candidates[args.candidate_index]]
        if args.candidate_index is not None
        else [c for i, c in enumerate(candidates) if i % args.num_shards == args.shard]
    )
    _emit(
        args.output,
        {
            "kind": "environment",
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "devices": [str(d) for d in jax.devices()],
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("jax", "jaxlib", "flash-attn-4", "nvidia-cutlass-dsl")
            },
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "source_hashes": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__), *Path(fa4_cute.__file__).parent.glob("_fa4_cute*.py")]
            },
        },
    )
    print(f"sweep={args.sweep} candidates={len(candidates)} shard={args.shard}/{args.num_shards} running={len(shard)}")

    for window_name, window in (("sliding", args.sliding_window), ("causal", None)):
        mask = AttentionMask.causal(sliding_window=window).with_segment_ids(segment_ids)
        reference_verdict = _check_against_float32_reference(reference, window, args)
        if reference_verdict != "ok":
            raise RuntimeError(f"Reference failed: {reference_verdict}")
        for candidate in shard:
            record = {
                "candidate_index": args.candidate_index,
                "config": dataclasses.asdict(candidate.config),
                "backward_path": candidate.backward_path,
                "window": window,
                "window_name": window_name,
                "backend": args.backend,
                "sweep": args.sweep,
            }
            try:
                verdict = _check_against_float32_reference(candidate, window, args)
                _emit(args.output, {**record, "kind": "correctness", "verdict": verdict})
                if verdict != "ok":
                    continue
                for round_index in range(args.rounds):
                    # Alternate order to expose drift; repeat the production reference for every candidate.
                    order = [("reference", reference), ("candidate", candidate)]
                    if round_index % 2:
                        order.reverse()
                    for role, measured in order:
                        got = _bench(measured, q, k, v, mask, args.steps, args.warmup)
                        _emit(
                            args.output,
                            {
                                **record,
                                "kind": "timing",
                                "role": role,
                                "config": dataclasses.asdict(measured.config),
                                "backward_path": measured.backward_path,
                                "round": round_index,
                                "forward": got.forward,
                                "backward": got.backward,
                                "total": got.forward + got.backward,
                            },
                        )
            except Exception as exc:
                # Unsupported layouts are expected during a sweep. Preserve the full diagnostic.
                _emit(args.output, {**record, "kind": "error", "error_type": type(exc).__name__, "error": str(exc)})
                # A device error may poison the context. A fresh process handles the next candidate.
                if args.backend == "native-sm100":
                    return
    _emit(args.output, {"kind": "candidate_complete", "candidate_index": args.candidate_index})


def _emit(output: Path | None, record: dict) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a") as stream:
            stream.write(line + "\n")


if __name__ == "__main__":
    main()
