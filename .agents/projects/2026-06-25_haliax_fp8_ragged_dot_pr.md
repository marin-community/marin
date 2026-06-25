# Plan: extract the haliax FP8 ragged-dot feature into a minimal, reviewable PR

Status: DRAFT (not executed). Source: research branch `research/grug-fp8-h100` (logbook GFP8-029→032).
Goal: a small, self-contained haliax PR that adds genuine-FP8 grouped (ragged) matmul with the H100
Mosaic backend (the shipped ~1.27× MoE-MLP win), reviewable without the experiment history.

## What ships (the minimal feature)

The public entry is `haliax.quantization` → `fp8_scaled_ragged_dot` (delayed-scaling FP8 ragged dot),
backed by the Mosaic-GPU f8 wgmma kernel for the forward + dgrad and bf16 for the weight-gradient.

Files in the PR (final-state, cleaned):
- `lib/haliax/src/haliax/_src/fp8_ragged.py` (NEW) — per-tensor delayed-scaling FP8 ragged dot +
  `custom_vjp`. After cleanup: forward + dgrad on f8, **bf16 weight-gradient** (the shipped recipe).
- `lib/haliax/src/haliax/nn/ragged_dot.py` (+~230) — the `"mosaic"` backend (`_mosaic_pallas_call`,
  `_mosaic_ragged_dot`), `MosaicBlockConfig` with the tuned H100 default, and the `mosaic`
  `Implementation` option. f8 forward/dgrad layouts only (no drhs mosaic branch — see cuts).
- `lib/haliax/src/haliax/_src/fp8.py` (+, trimmed) — only the direct-quant helpers the ragged path
  needs that aren't already on main: `out_dq`, `_new_scale_and_history` (main already has
  `in_q`/`quantize`/`dequantize`). Drop any unrelated dense-fp8 experiments from the diff.
- `lib/haliax/src/haliax/quantization.py` (+) — public dispatch routing the quantized grouped dot to
  `fp8_scaled_ragged_dot` (net-new; main has no ragged path here).
- `lib/haliax/tests/test_fp8_ragged.py` (NEW) + `test_fp8.py` (+) — CPU-runnable behavior tests
  (forward/backward track bf16, delayed-scaling state, unpadded tokens, runs-under-jit).

## What is CUT from the PR (stays on the research branch)

1. **The f8 cast-transpose weight-gradient kernel** `_src/transposed_ragged_dot_mgpu.py` (213 lines) +
   its guarded import + the `_DRHS` branch of `_mosaic_pallas_call` + the f8-wgrad branch in
   `fp8_ragged.py` bwd + the `RAGGED_F8_WGRAD` env toggle. **It is not on the winning path** — the
   shipped recipe runs the wgrad in bf16 (GFP8-030: f8 wgrad loses e2e). Shipping it would add a
   213-line dead-on-arrival kernel; AGENTS.md forbids dead code. Park for a Blackwell follow-up PR.
   - After removal, the mosaic bwd is unconditionally bf16 wgrad (no toggle), and `_mosaic_pallas_call`
     handles only `_DEFAULT`/`_DLHS` (the f8-legal forward/dgrad layouts).
2. **Experiment env knobs introduced by this work** (replace with the shipped defaults):
   - `RAGGED_DOT_F8_COMPUTE` / `_F8_COMPUTE_MODE` → hardcode the shipped `"passthrough"` behavior.
   - `_env_int` + the 11 `RAGGED_DOT_BLOCK_*/NUM_WARPS/NUM_STAGES` overrides → use the
     `_TRITON_DEFAULT_*` constants directly; delete `_env_int`.
   - LEAVE the pre-existing `RAGGED_DOT_IMPL` override alone (already on main — out of scope).
3. **Experiment-log prose** in docstrings/comments: trim `GFP8-029/030/031/032` references to concise
   rationale ("tuned for H100 grouped GEMM: block_k=128 frees smem for a deep pipeline"), no logbook.
4. **Research artifacts**: the GFP8 logbook, every `_s5_*.sh` launcher, the `bench_*.py` benches, and
   the cluster cuDNN-9.12/ptxas/libdevice shims — none of these go in the library PR.

## Extraction mechanics (clean branch, no 182-commit history)

```bash
# fresh branch off current main (NOT the drifted research branch)
git fetch origin && git checkout -b grug-fp8-ragged-dot origin/main

# bring over ONLY the feature files at their final state (one tree checkout, not a rebase)
git checkout research/grug-fp8-h100 -- \
  lib/haliax/src/haliax/_src/fp8_ragged.py \
  lib/haliax/src/haliax/nn/ragged_dot.py \
  lib/haliax/src/haliax/_src/fp8.py \
  lib/haliax/src/haliax/quantization.py \
  lib/haliax/tests/test_fp8_ragged.py \
  lib/haliax/tests/test_fp8.py
# do NOT bring transposed_ragged_dot_mgpu.py

# then apply the cleanup edits (cuts 1-3 above) as normal edits, and squash to 1-2 commits.
```

Then:
- Re-diff `fp8.py`/`quantization.py` against main and **strip any hunks not needed by the ragged
  path** (the +250 in fp8.py likely carries unrelated dense-fp8 changes — keep only `out_dq` +
  `_new_scale_and_history` + the minimal `in_q`/`quantize` deltas the ragged path uses).
- Squash into a single commit (or 2: "feat: fp8 ragged dot scaling" + "feat: mosaic f8 backend") —
  not the GFP8 history.

## Verify before opening

- `./infra/pre-commit.py --all-files --fix` clean; `uv run pyrefly` clean.
- `uv run pytest lib/haliax/tests/test_fp8_ragged.py lib/haliax/tests/test_fp8.py` green on CPU.
  (The mosaic path is H100-only behind a guarded import; CPU CI exercises the triton/xla fallback.
  The Mosaic numerics + perf were validated on H100 — cite that, don't gate CI on H100.)
- `./infra/pre-commit.py --review` against the diff; address every finding.

## PR description (tight — no experiment log)

- What: genuine-FP8 grouped/ragged matmul for the MoE expert GEMMs; per-tensor delayed scaling
  (E4M3 fwd/dgrad, bf16 wgrad), H100 Mosaic-GPU f8 wgmma backend; public via `haliax.quantization`.
- Perf: ~1.27× e2e fwd+bwd on the real MoE-MLP shape vs the bf16-Triton path (H100); ~1.5-1.8×
  per-GEMM dtype-only; grouped kernels at 78-87% of the cuBLAS dense-f8 ceiling. One-line table, no log.
- Numerics: ~6-8% rel-frob vs bf16 (coarse per-tensor E4M3) — fine for the primitive; e2e training
  validation is the follow-up (levanter/Grug integration).
- Deferred (named, not detailed): f8 weight-gradient kernel (loses e2e today, parked for Blackwell);
  per-shape config table (single global config within ~1%).
- Link the research branch / experiment issue for the full story.

## Open decisions for the user

1. **One PR or two?** Recommend ONE (scaling + mosaic together — the value is the mosaic win). Split
   option: PR1 = fp8 ragged scaling on triton/xla (correctness, no speedup); PR2 = mosaic backend.
2. **Drop the f8 wgrad kernel from the PR?** Recommend YES (dead on the winning path). Alternative:
   keep it behind a clean config param (not env) with a "parked for Blackwell" docstring.
3. **Include a benchmark in the PR?** The add-pallas-kernel skill wants a checked-in perf harness, but
   the current benches carry cluster hacks. Option: add one minimal cluster-agnostic bench, or defer.
4. **Where does it land** — confirm the monorepo `lib/haliax` is the target (it is `marin-haliax`,
   native, not a submodule), so a single marin-monorepo PR.

## Rough size after cleanup

~750-900 lines net (down from ~1346): drops the 213-line wgrad kernel, the env-knob plumbing, and the
log prose; keeps fp8_ragged + the mosaic forward/dgrad backend + the direct-quant helpers + tests.
