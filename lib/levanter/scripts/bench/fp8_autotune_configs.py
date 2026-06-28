# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Static experiment design for the FP8 ragged-dot autotuner: shapes and block-config candidates.

Pure data (no jax / haliax) so the multi-GPU orchestrator can enumerate the sweep without importing
jax. The benchmark/worker wraps these dicts into the haliax ``MosaicBlockConfig`` / ``WgradBlockConfig``
dataclasses; the orchestrator passes them straight into worker request specs. Single source of truth so
the single-GPU harness and the sharded orchestrator always sweep the same space.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Shape:
    """A Grug MoE expert-MLP problem size. T = dispatched tokens (seq x top_k), summed over experts."""

    name: str
    tokens: int
    hidden: int  # D
    intermediate: int  # F (per-expert MLP width; the gated up-proj is 2F wide)
    experts: int  # E

    def __str__(self) -> str:
        return f"{self.name}[T{self.tokens}/D{self.hidden}/F{self.intermediate}/E{self.experts}]"


# Experts axis drives raggedness (tokens/expert): target ~1024, scale ~128.
SHAPE_GRID = {
    "small": Shape("small", tokens=4096, hidden=512, intermediate=256, experts=8),
    "target": Shape("target", tokens=8192, hidden=2048, intermediate=5632, experts=8),
    "scale": Shape("scale", tokens=16384, hidden=3072, intermediate=1536, experts=128),
}

# H100 usable shared memory per SM (bytes); prunes block configs whose pipeline cannot fit.
H100_SMEM_BYTES = 227 * 1024

# (block_m, block_n, block_k, max_concurrent_steps, grid_block_n) for the shared forward/dlhs Mosaic
# GEMM. Curated around the GFP8-029 winner (128/128/128 steps=6 grid_block_n=4): baseline, one-axis
# sweeps, and a few interactions.
_MOSAIC_RAW = [
    (128, 128, 128, 6, 4),
    (64, 128, 128, 6, 4),
    (256, 128, 128, 6, 4),
    (128, 64, 128, 6, 4),
    (128, 256, 128, 6, 4),
    (128, 128, 64, 6, 4),
    (128, 128, 256, 2, 4),
    (128, 128, 128, 2, 4),
    (128, 128, 128, 4, 4),
    (128, 128, 128, 8, 4),
    (128, 128, 128, 6, 1),
    (128, 128, 128, 6, 2),
    (128, 128, 128, 6, 8),
    (128, 256, 128, 6, 2),
    (256, 128, 128, 4, 4),
    (64, 128, 128, 8, 4),
    (128, 128, 64, 8, 4),
    (256, 256, 128, 2, 4),
    (64, 64, 128, 8, 8),
]

# Independent f8 weight-gradient kernel candidates (default grid_block_n=2).
_WGRAD_RAW = [
    (128, 128, 128, 6, 2),
    (128, 128, 128, 6, 4),
    (128, 128, 128, 4, 2),
    (128, 128, 64, 6, 2),
    (128, 128, 256, 2, 2),
    (64, 128, 128, 6, 2),
    (128, 64, 128, 6, 2),
    (256, 128, 128, 4, 2),
    (128, 256, 128, 4, 2),
    (128, 128, 128, 8, 2),
    (128, 128, 128, 6, 1),
]

# bf16 Triton baseline: (BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_STAGES). Default 128/128/32/4/4.
_BF16_RAW = [
    (128, 128, 32, 4, 4),
    (128, 128, 64, 4, 4),
    (128, 256, 32, 4, 8),
    (64, 128, 32, 4, 4),
    (256, 128, 32, 8, 4),
    (128, 128, 32, 8, 4),
    (128, 128, 32, 4, 8),
    (128, 128, 32, 4, 2),
    (128, 256, 64, 8, 4),
    (256, 256, 32, 8, 8),
    (128, 128, 128, 4, 4),
]


def smem_per_stage_f8(block_m: int, block_n: int, block_k: int) -> int:
    """Approx bytes/pipeline-stage for an f8 grouped GEMM: one (m,k) + one (k,n) tile, 1 byte/elt."""
    return (block_m * block_k) + (block_k * block_n)


def _dedup_block_dicts(raw):
    """Build block-config dicts from (bm, bn, bk, steps, gbn) tuples; drop smem-infeasible; dedup."""
    out, seen = [], set()
    for bm, bn, bk, steps, gbn in raw:
        if smem_per_stage_f8(bm, bn, bk) * steps > H100_SMEM_BYTES:
            continue
        key = (bm, bn, bk, steps, gbn)
        if key in seen:
            continue
        seen.add(key)
        out.append({"block_m": bm, "block_n": bn, "block_k": bk, "max_concurrent_steps": steps, "grid_block_n": gbn})
    return out


def mosaic_candidate_dicts():
    return _dedup_block_dicts(_MOSAIC_RAW)


def wgrad_candidate_dicts():
    return _dedup_block_dicts(_WGRAD_RAW)


def bf16_candidate_dicts():
    return [
        {"block_m": bm, "block_n": bn, "block_k": bk, "num_warps": w, "num_stages": s}
        for bm, bn, bk, w, s in _BF16_RAW
    ]
