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


# Representative per-device Grug MoE expert ragged-dot shapes for the REAL d2560 model
# (David's run / issue 6044: hidden D=2560, intermediate F=1280, num_experts=256, top_k=4, seq=4096,
# capacity_factor=1). Per device: E_local = 256/EP, tokens/expert = B·S·top_k·EP/(devices·experts).
# Ground-truthed against the GM2560-B16-EP8 profile (kernel grid 128,20,32 → E_local=32, 2F=2560,
# T=16384, 512 tok/expert). The sweep maps the realistic regime: E_local {16,32,64} (EP 16/8/4) and
# tokens/expert {512..4096} (memory-limited GPU profiling → David's EP4 full run ≈1024). All hidden=2560,
# intermediate=1280; `experts` = E_local, `tokens` = E_local · tokens_per_expert (top_k folded in).
SHAPE_GRID = {
    # tokens/expert = 1024 (≈ full-run load), sweeping E_local (EP 16 / 8 / 4):
    "d2560_e16_t1k": Shape("d2560_e16_t1k", tokens=16384, hidden=2560, intermediate=1280, experts=16),
    "d2560_e32_t1k": Shape("d2560_e32_t1k", tokens=32768, hidden=2560, intermediate=1280, experts=32),
    "d2560_e64_t1k": Shape("d2560_e64_t1k", tokens=65536, hidden=2560, intermediate=1280, experts=64),
    # E_local = 32 (EP8, the GPU-realistic case), sweeping tokens/expert 512 → 4096:
    "d2560_e32_t512": Shape("d2560_e32_t512", tokens=16384, hidden=2560, intermediate=1280, experts=32),
    "d2560_e32_t2k": Shape("d2560_e32_t2k", tokens=65536, hidden=2560, intermediate=1280, experts=32),
    "d2560_e32_t4k": Shape("d2560_e32_t4k", tokens=131072, hidden=2560, intermediate=1280, experts=32),
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
    # R2: F=1280 regime (fwd/dlhs GEMM is M-skinny per expert ~512-1024, N=2F=2560, K=2560/1280).
    # Favor wide block_n (cover N=2560 in fewer tiles) and small block_m (less ragged-group padding).
    (64, 256, 128, 4, 4),
    (64, 256, 64, 6, 4),
    (128, 256, 64, 6, 4),
    (64, 256, 128, 4, 2),
    (64, 128, 256, 2, 4),
    (128, 256, 256, 2, 4),
    (256, 256, 64, 4, 4),
    (64, 64, 256, 4, 8),
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
    # R2: F=1280 wgrad (dW = x_t[D,T] x g[T,2F], so K = tokens/expert ~512-1024 is SMALL; M=N=2560).
    # Favor small block_k (short token-reduction) and wide block_m/n over the 2560 output.
    (256, 128, 64, 4, 2),
    (128, 256, 64, 4, 2),
    (256, 256, 64, 4, 2),
    (128, 128, 64, 8, 2),
    (256, 128, 128, 4, 1),
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
