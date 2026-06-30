# Task: Prototype the simplest complete FP8 `ragged_dot` (MoE grouped matmul) on Hopper H100

You are one of three independent investigators working the SAME problem in isolation.
Do NOT coordinate with anyone. Produce your own from-first-principles solution.

## Goal

In the marin repo we recently added a **dense** FP8 hybrid E4M3/E5M2 forward/backward
`dot_general` (PR #6660). Now build the **analog for `ragged_dot`** — the MoE grouped
matmul (a.k.a. grouped GEMM / `jax.lax.ragged_dot`).

Prototype the **simplest possible complete implementation** that meets ALL of:

1. Supports mixed {E4M3, E5M2}: **E4M3 forward** (activations + weights), **E5M2 backward**
   (output gradients). FP32 accumulation. Forward + backward (custom VJP), TE-style delayed
   per-tensor scaling with amax history, mirroring how PR #6660 structured the dense op.
2. **At least 20% faster** than the current bf16 `ragged_dot` (which is already an optimized
   Pallas/Triton kernel — beating it is the bar) at realistic grug MoE scale on an H100.
3. Numerically sound (forward parity vs bf16 within fp8 tolerance; backward genuinely
   contracts an E5M2 operand).

## Hard constraints

- **Changes only in THIS marin worktree.** If you conclude the goal is impossible without
  patching an external dependency (jax / flax / triton / pallas / tokamax / cublas), you may
  STOP — but then your NOTES must describe the exact patches needed and include the diffs.
- **Do NOT look at any other git branch.** In particular do NOT checkout, diff, read, or `git show`
  anything from `research/grug-fp8-h100` or any `grug-fp8-*` branch, and do NOT read the directory
  `/home/matt/projects/marin` (it is checked out on the forbidden research branch). Work only inside
  your assigned worktree. The point is to start from first principles, unbiased by past attempts.
- **Do NOT open PRs, push branches, or do ANY external communication** (no `gh pr/issue create`,
  no PR/issue comments, no Slack/Discord/email/web posting). Local git commits in your worktree are fine.
- **Do NOT reserve GPUs or run anything on the Iris cluster / cw-us-east-02a.** No GPU jobs at all.
  You have no H100. Instead, write a **runnable benchmark harness** script; the coordinator will run
  it on a shared H100. You MAY run local CPU correctness checks (`uv run`, pallas interpret mode,
  jax on CPU).
- You have network access (uv sync, gh, pip downloads are fine).

## Deliverables (write all into your worktree root)

1. **Prototype code** in `lib/haliax` (new module + wiring as needed): an FP8 `ragged_dot`
   supporting E4M3 fwd / E5M2 bwd with a custom VJP, structured like PR #6660's dense op.
2. **`bench_fp8_ragged_dot.py`** — a single standalone script runnable as
   `uv run python bench_fp8_ragged_dot.py` on one H100. It must:
   - Build realistic grug-MoE-scale inputs (see Orientation), parameterizable via CLI flags.
   - Time the **bf16** `ragged_dot` and your **fp8** `ragged_dot` (forward, and ideally fwd+bwd),
     warm up + average over many iters, and print TFLOP/s and the **fp8/bf16 speedup ratio**.
   - Default to shapes that exercise both expert GEMMs (gate/up and down).
3. **`NOTES.md`** — concise: your approach, key design decisions and why, alternatives you
   rejected and why, correctness validation performed, the expected speedup rationale, risks,
   and (if applicable) exact external-dep patches with diffs. Be specific and dense; this is
   what the coordinator reads to compare and synthesize.

## Orientation (facts, not a prescribed solution — design it yourself)

- BF16 GPU MoE grouped matmul lives in `lib/haliax/src/haliax/nn/ragged_dot.py`:
  a Pallas-Triton kernel (`_triton_ragged_dot_kernel`, fp32 accumulation) with a custom VJP
  using three `RaggedDotDimensionNumbers` layouts (fwd, dlhs, drhs); XLA fallback is
  `jax.lax.ragged_dot_general`. Adapted from openxla/tokamax.
- Dense FP8 reference (the thing you are making an analog of):
  `lib/haliax/src/haliax/_src/fp8.py` and `lib/haliax/src/haliax/quantization.py`
  (`Fp8DotGeneralOp`, `fp8_scaled_dot_general`, `quantized_dot`, `in_q`, `out_dq`,
  `update_fp8_meta`, `quantize`, `dequantize`, `compute_scale`, `compute_amax_history`).
  Also `gh pr diff 6660` shows the full reference diff.
- Call sites: `lib/haliax/src/haliax/nn/linear.py` (`MoELinear`, has a TODO for ragged_dot
  quantization), `lib/levanter/src/levanter/grug/_moe/scatter.py`.
- Realistic grug MoE scale: hidden_dim=3072, intermediate_dim=1536, num_experts=128, top_k=4,
  seq_len=2048, batch=256. After top-4 dispatch tokens ≈ 256*2048*4 ≈ 2.1M.
  - gate/up GEMM: lhs `[~2.1M, 3072]`, rhs `[128, 3072, 3072]`, group_sizes `[128]` → `[~2.1M, 3072]`
  - down GEMM:    lhs `[~2.1M, 1536]`, rhs `[128, 1536, 3072]`, group_sizes `[128]` → `[~2.1M, 3072]`
  (You may benchmark at a smaller token count if memory-bound, but keep the K/N dims realistic
  and report the token count used.)

## Success criterion

A working, numerically-validated fp8 `ragged_dot` prototype + a harness that (when the coordinator
runs it on H100) is expected to show ≥20% speedup over bf16, OR a rigorous writeup proving why that
requires an external-dep patch, with the exact diffs.

When done, ensure your NOTES.md, prototype, and bench script are committed in your worktree.
Print a final one-paragraph summary of what you built and your expected speedup.
