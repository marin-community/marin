# MoE MuonH Matrix Sweep: Research Logbook

## Scope

- Goal: Test whether replacing AdamH with MuonH for every Grug MoE matrix except the lm head improves effective speedup versus the v16 AdamH baseline.
- Primary metrics: `eval/paloma/macro_loss`, `throughput/tokens_per_second`, `throughput/total_tokens`, run state.
- Constraints: Keep model sizing, data, batch size, step count, schedule, z-loss, eval cadence, and lm-head AdamH behavior fixed.
- Issue: https://github.com/marin-community/marin/issues/5596

## Baseline

- Date: 2026-05-09
- Code refs: `experiments/grug/moe/README.md`, `experiments/grug/moe/heuristic.py`, `experiments/grug/moe/launch.py`
- Baseline numbers: use the README compute-optimal table. Gate 1 compares d512 at 2.19e17 FLOPs and d768 at 1.70e18 FLOPs.

## Experiment Log

### 2026-05-09 11:20 - MOE-MH-001 implementation validation

- Hypothesis: MuonH can replace AdamH on Grug MoE matrix-shaped leaves while keeping the lm head on AdamH and vectors/scalars on Adam.
- Command: `uv run pytest -o addopts='' tests/test_grug_moe_optimizer.py tests/test_grug_variant_contracts.py -q`
- Config: `GrugMoeMuonHConfig`, `MUONH_MATRIX_GATE=1` launcher defaults.
- Result: 11 tests passed. The launch-builder smoke check produced gate-1 steps `muonh-matrix-d512-2.19e17` and `muonh-matrix-d768-1.70e18`.
- Interpretation: The optimizer mask, single-device update path, and Grug variant contracts are ready for gate-1 launch.
- Next action: Open PR, launch gate-1 Iris jobs, and monitor W&B runs under group `muonh-matrix-sweep`.
