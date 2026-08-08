# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep segmented FA4/CuTe tile configurations at Grug hero attention shapes.

Times a sliding-window and a full-causal mask against a reference configuration and gates every
candidate on reproducing ``reference_attention`` in float32.

``--sweep forward`` varies the forward tile at the production backward. ``--sweep backward``
varies the backward tile, thread count, and backward path, and lifts the
``_segmented_backward_arches`` allowlist to do it: that function restricts compute capability
9.x/10.x/12.x to a 64x64 backward at 128 threads, but the underlying
``SegmentedFlashAttentionBackwardSm120.can_implement`` only requires ``n_block % 16 == 0``,
``num_threads % 32 == 0``, and that shared memory fit. The allowlist is therefore the binding
constraint rather than the kernel, and lifting it in a benchmark is how to find out whether it
needs to bind.

The two backward paths differ in more than tiles. ``path_arch=120`` runs
``SegmentedFlashAttentionBackwardSm120`` with ``num_stages_Q = num_stages_dO = 1`` at head
dimension 128 -- no double buffering of the Q and dO loads -- and a 4-warp atom layout that wants
128 threads. ``path_arch=80`` runs the SM80 class with both stage counts at 2 and a 2-warp atom
layout. Sweeping the path is how the double-buffering question gets answered without changing
library code.

One process sweeps one backward path. ``cute_launcher_factory`` memoizes on the launcher's
keyword arguments, and the path is derived inside the launcher rather than passed to it, so
benching two paths at one tile in a single process silently returns the first path's kernel for
both. ``--shard``/``--num-shards`` split the candidate list so one 4-GPU node can run four
disjoint slices concurrently, one process per GPU.
"""

import argparse
import contextlib
import dataclasses
import time
from dataclasses import dataclass
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.attention import AttentionMask, reference_attention
from levanter.grug.attention import _fa4_cute as fa4_cute
from levanter.grug.attention import _fa4_cute_kernels as fa4_cute_kernels
from levanter.grug.attention._fa4_cute_config import Flash4CuteKernelConfig, flash4_cute_kernel_config

REFERENCE_TILE = (64, 64)
CANDIDATE_FORWARD_TILES = ((64, 64), (64, 128), (128, 32), (128, 64), (128, 128), (192, 64), (256, 64))
CANDIDATE_BACKWARD_TILES = ((64, 64), (64, 128), (128, 64), (128, 128), (192, 64), (256, 64))
CANDIDATE_NUM_THREADS = (128, 256)
# path_arch 120 is production (stages 1/1 at head_dim 128, 4-warp atoms); 80 is double-buffered.
CANDIDATE_BACKWARD_PATHS = (120, 80)


@dataclass(frozen=True)
class Candidate:
    """A kernel configuration plus the backward path to force it through."""

    config: Flash4CuteKernelConfig
    backward_path: int

    def label(self) -> str:
        return (
            f"{self.config.forward_tile!s:>10} {self.config.backward_tile!s:>10} "
            f"{self.config.num_threads:>4} {self.backward_path:>5}"
        )


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

    out: jax.Array
    grads: tuple[jax.Array, ...]
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
    with patch.object(fa4_cute, "_segmented_kernel_config", lambda head_dim: candidate.config):
        with patch.object(fa4_cute_kernels, "_segmented_backward_arches", lambda **_: selection):
            yield


def _bench(candidate: Candidate, q, k, v, mask, steps: int, warmup: int) -> BenchResult:
    with _forced(candidate):
        forward = jax.jit(lambda q, k, v: fa4_cute.gpu_fa4_cute_attention(q, k, v, mask))
        grad = jax.jit(jax.grad(_loss, argnums=(0, 1, 2)))

        out = forward(q, k, v)
        out.block_until_ready()
        grads = grad(q, k, v, mask)
        jax.block_until_ready(grads)

        for _ in range(warmup):
            forward(q, k, v).block_until_ready()
        start = time.perf_counter()
        for _ in range(steps):
            forward(q, k, v).block_until_ready()
        forward_time = (time.perf_counter() - start) / steps

        for _ in range(warmup):
            jax.block_until_ready(grad(q, k, v, mask))
        start = time.perf_counter()
        for _ in range(steps):
            jax.block_until_ready(grad(q, k, v, mask))
        grad_time = (time.perf_counter() - start) / steps

    return BenchResult(out=out, grads=grads, forward=forward_time, backward=grad_time - forward_time)


def _max_diff(a, b) -> float:
    return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))


def _check_against_float32_reference(
    candidate: Candidate,
    window: int | None,
    args: argparse.Namespace,
) -> str:
    """Compare a candidate against the float32 reference at the tolerances the GPU tests use.

    Timing runs at hero shapes, where bf16 gradients are large enough that an absolute
    difference between two tile configurations says nothing about correctness. This is the
    check that decides whether a configuration is usable. It reuses the swept head counts and
    document density but shrinks batch and sequence length, because the reference materializes
    the full score matrix and is quadratic in sequence length.
    """
    seq_len = args.check_seq_len
    key = jax.random.key(11)
    q = jax.random.normal(key, (1, seq_len, args.q_heads, args.head_dim), dtype=jnp.bfloat16)
    kv_shape = (1, seq_len, args.kv_heads, args.head_dim)
    k = jax.random.normal(jax.random.fold_in(key, 1), kv_shape, dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.fold_in(key, 2), kv_shape, dtype=jnp.bfloat16)
    cotangent = jax.random.normal(jax.random.fold_in(key, 3), q.shape, dtype=jnp.bfloat16)
    segment_ids = _segment_ids(1, seq_len, args.documents)
    mask = AttentionMask.causal(sliding_window=window).with_segment_ids(segment_ids)

    def fa4_loss(q_arg, k_arg, v_arg):
        out = fa4_cute.gpu_fa4_cute_attention(q_arg, k_arg, v_arg, mask)
        return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

    def ref_loss(q_arg, k_arg, v_arg):
        out = reference_attention(q_arg, k_arg, v_arg, mask, logits_dtype=jnp.float32)
        return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

    expected = reference_attention(q, k, v, mask, logits_dtype=jnp.float32)
    expected_grads = jax.jit(jax.grad(ref_loss, argnums=(0, 1, 2)))(q, k, v)
    with _forced(candidate):
        actual = jax.jit(fa4_cute.gpu_fa4_cute_attention)(q, k, v, mask)
        actual_grads = jax.jit(jax.grad(fa4_loss, argnums=(0, 1, 2)))(q, k, v)

    failures = []
    for name, got, want in [
        ("out", actual, expected),
        ("dq", actual_grads[0], expected_grads[0]),
        ("dk", actual_grads[1], expected_grads[1]),
        ("dv", actual_grads[2], expected_grads[2]),
    ]:
        try:
            np.testing.assert_allclose(
                np.asarray(got, dtype=np.float32), np.asarray(want, dtype=np.float32), atol=7e-2, rtol=7e-2
            )
        except AssertionError:
            failures.append(name)
    return "ok" if not failures else "FAIL:" + ",".join(failures)


def _build_candidates(base: Flash4CuteKernelConfig, sweep: str, backward_path: int) -> list[Candidate]:
    """Candidates for one sweep axis, holding the other axis at its production value.

    A full cross product of forward and backward tiles wastes most of its runs: the two kernels
    are timed separately, so varying both at once only re-measures the same pairs.
    """
    if sweep == "forward":
        return [
            Candidate(
                dataclasses.replace(base, forward_tile=tile, backward_tile=REFERENCE_TILE, num_threads=128),
                backward_path,
            )
            for tile in CANDIDATE_FORWARD_TILES
        ]
    return [
        Candidate(
            dataclasses.replace(base, forward_tile=base.forward_tile, backward_tile=tile, num_threads=threads),
            backward_path,
        )
        for tile in CANDIDATE_BACKWARD_TILES
        for threads in CANDIDATE_NUM_THREADS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=32, help="Per-GPU batch; the hero runs 32.")
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
        default=512,
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

    if jax.default_backend() != "gpu":
        raise SystemExit("tile_sweep requires the JAX GPU backend.")

    key = jax.random.key(0)
    shape_q = (args.batch, args.seq_len, args.q_heads, args.head_dim)
    shape_kv = (args.batch, args.seq_len, args.kv_heads, args.head_dim)
    q = jax.random.normal(key, shape_q, dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), shape_kv, dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.fold_in(key, 2), shape_kv, dtype=jnp.bfloat16)
    segment_ids = _segment_ids(args.batch, args.seq_len, args.documents)

    arch = fa4_cute._gpu_compute_arch()
    base = flash4_cute_kernel_config(args.head_dim, arch=arch)
    print(f"arch=sm{arch} base_forward_tile={base.forward_tile} base_backward_tile={base.backward_tile}")
    print(f"shape: batch={args.batch} seq={args.seq_len} q_heads={args.q_heads} head_dim={args.head_dim}")

    reference = Candidate(
        config=dataclasses.replace(base, forward_tile=REFERENCE_TILE, backward_tile=REFERENCE_TILE, num_threads=128),
        backward_path=args.backward_path,
    )
    candidates = _build_candidates(base, args.sweep, args.backward_path)
    shard = [c for i, c in enumerate(candidates) if i % args.num_shards == args.shard]
    print(f"sweep={args.sweep} candidates={len(candidates)} shard={args.shard}/{args.num_shards} running={len(shard)}")

    for window_name, window in (("sliding", args.sliding_window), ("causal", None)):
        mask = AttentionMask.causal(sliding_window=window).with_segment_ids(segment_ids)
        ref = _bench(reference, q, k, v, mask, args.steps, args.warmup)
        print(f"\n=== {window_name} (window={window}) ===")
        print(
            f"{'fwd_tile':>10} {'bwd_tile':>10} {'thr':>4} {'path':>5} {'fwd ms':>8} {'bwd ms':>8} "
            f"{'total ms':>9} {'vs ref':>8} {'max|d|':>9}"
        )
        print(
            f"{reference.label()} {ref.forward * 1e3:8.3f} {ref.backward * 1e3:8.3f} "
            f"{(ref.forward + ref.backward) * 1e3:9.3f} {'1.00x':>8} {'ref':>9}"
        )

        for candidate in shard:
            if candidate == reference:
                continue
            try:
                got = _bench(candidate, q, k, v, mask, args.steps, args.warmup)
            except Exception as exc:
                # A rejected tile/thread/shared-memory combination is an expected outcome here, but
                # print the message so a genuine failure is not mistaken for an unsupported config.
                reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                print(f"{candidate.label()} {'-':>8} {'-':>8} {'-':>9} {'-':>8}  {type(exc).__name__}: {reason[:70]}")
                continue
            diff = max(
                [_max_diff(got.out, ref.out)] + [_max_diff(g, r) for g, r in zip(got.grads, ref.grads, strict=True)]
            )
            speedup = (ref.forward + ref.backward) / (got.forward + got.backward)
            verdict = _check_against_float32_reference(candidate, window, args)
            print(
                f"{candidate.label()} {got.forward * 1e3:8.3f} {got.backward * 1e3:8.3f} "
                f"{(got.forward + got.backward) * 1e3:9.3f} {speedup:7.2f}x {diff:9.2e} {verdict}"
            )


if __name__ == "__main__":
    main()
