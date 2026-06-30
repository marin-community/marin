# Task: Implement an FP8 `ragged_dot` (MoE grouped matmul) on Hopper H100, from first principles

You are one of three independent investigators working the SAME problem in isolation. Do NOT coordinate with anyone. Produce your own from-first-principles solution. We will compare and synthesize the three approaches afterward.

## What to build

The marin repo recently merged a *dense* FP8 hybrid-E4M3/E5M2 forward/backward `dot_general` (PR #6660: `Fp8DotGeneralOp` in `lib/haliax/src/haliax/quantization.py`, helpers in `lib/haliax/src/haliax/_src/fp8.py`; read it with `gh pr diff 6660` or just read the files). Build the **analog for `ragged_dot`** — the MoE grouped matmul — meeting ALL of:

1. **Mixed FP8:** E4M3 forward (activations + weights), E5M2 backward (output gradients), FP32 accumulation. Forward + backward via a **custom VJP**; the backward must genuinely contract an **E5M2** operand (no all-E4M3 shortcut). **TE-style delayed per-tensor scaling** with amax history. Thread the scale/amax state so the optimizer/EMA state stays clean. Concretely this means a **stateful op object analogous to the dense `Fp8DotGeneralOp`** — an `OverwriteWithGradient` (eqx.Module) subclass holding the scale/amax_history fields, whose updates flow through `partition_for_grad_overwrite`/`apply_updates`. (Delayed scaling needs cross-step state; a pure function can't express it. The *kernel* this object wraps is entirely yours to design.)
2. **Performance:** end-to-end (fwd+bwd) **≥1.2× faster than the bf16 `ragged_dot`** at ≥1024 tokens/expert, at the realistic grug-MoE scale below, on an H100. Faster is better; the forward GEMM should be materially faster than bf16.
3. **Numerics:** forward parity vs bf16 within FP8 tolerance (report relative-Frobenius error, not just `allclose`); gradient parity on small shapes.
4. **Opt-in:** integrate as an opt-in path through `haliax.nn.ragged_dot` (`lib/haliax/src/haliax/nn/ragged_dot.py`). The existing bf16 default (Triton on GPU) MUST remain unchanged.
5. **Genuine ragged (non-uniform) groups — REQUIRED.** Real MoE routing produces a *different, dynamic* token count per expert, so `group_sizes` is **non-uniform** and not a compile-time constant. Your op AND your measured speedup MUST hold for non-uniform groups. A solution that only works — or only clears the bar — for *uniform, equal-size* groups (e.g. reshaping `[tokens,dim] → [experts, tokens_per_expert, dim]` and using a **batched dense GEMM**) does **NOT** satisfy this requirement. Your `bench_fp8_ragged_dot.py` MUST benchmark **non-uniform** `group_sizes` (a realistic skewed/random distribution summing to the same total tokens), and the **≥1.2× fwd+bwd bar must hold there** at ≥1024 *average* tokens/expert, from a genuinely ragged/grouped fp8 kernel (not static-equal-group batching).

## Realistic scale (d2560 grug MoE)

hidden D=2560, intermediate F=1280, num_experts=256, top_k=4. Per-device expert GEMMs:
- w13 (gate+up): `lhs[T_local, 2560] × rhs[E_local, 2560, 2560]` (output 2F = 2560)
- w2 (down): `lhs[T_local, 1280] × rhs[E_local, 1280, 2560]`

Sweep `tokens_per_expert` ∈ {512, 1024, 2048, 4096}; `E_local` ∈ {16, 32, 64}; operating point E_local=64, 1024 tok/expert (so T_local = E_local · tokens_per_expert = 65536). Baseline = `haliax.nn.ragged_dot(..., implementation="triton")` in bf16.

## Environment / H100

You have a dedicated **1×H100 dev pod**. **Read `ENVIRONMENT.md`** for how to run code on it (the `./h100` helper) and how the GPU env is set up. Iterate on real hardware. If you hit cluster-image toolchain errors (missing ptxas/libdevice for Mosaic; `uv sync` python-version rejections; cuDNN-too-old at compile; OOM/exit-137), these are **known environment issues, not part of your task** — ask the coordinator. Microbench-scale jobs only. **NEVER** stop/restart/bounce the cluster; the coordinator manages the pod lifecycle.

## This is solvable — finish it

A complete solution that meets the performance bar **is known to exist on this exact hardware**. Treat the full target — mixed E4M3/E5M2, correct forward + backward gradients, **≥1.2× fwd+bwd at ≥1024 tokens/expert** — as a reachable engineering goal, **not** an open research question.

- If a backend or approach turns out to be a dead end, that is *information*: **pivot and try another**. Do NOT conclude the task is impossible, and do NOT stop at a partial, forward-only, or silently-bf16-fallback result.
- If you get stuck, or are tempted to give up or declare something unsupported, **do not stop** — write down exactly where you are blocked in `NOTES.md` and **ask the coordinator for a hint**. The coordinator can and will unblock you (including external-dependency issues).
- Keep going until you have a working op that meets **all four** requirements above with **measured numbers** from the H100.

## Hard constraints / blindness

- Changes only in THIS worktree.
- Do NOT look at any other git branch. In particular do NOT checkout, diff, read, or `git show` anything from `research/grug-fp8-h100` or any `grug-fp8-*` / `fp8-*` branch, and do NOT read any other checkout under `/home/matt/projects/`. The point is to start from first principles, unbiased by past attempts.
- Do NOT open PRs, push branches, or do any external communication.
- If you believe an external dependency (e.g. a jaxlib/Mosaic limitation) blocks a clean solution, document it precisely in `NOTES.md` and ask the coordinator before building elaborate workarounds.

## Deliverables (in your worktree root)

1. The op in `lib/haliax` (the stateful op + whatever kernel/helpers it needs).
2. A runnable microbench `bench_fp8_ragged_dot.py` — `./h100 python bench_fp8_ragged_dot.py` — printing TFLOP/s and fp8/bf16 speedup across the sweep above, plus the forward relative-Frobenius error vs bf16.
3. `NOTES.md` — your approach, key decisions, numerics, risks, and any external-dependency patches you needed.
