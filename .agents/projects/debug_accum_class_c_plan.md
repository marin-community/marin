# Class C Experiment Plan — debug_accum_tpu_type

Written 2026-04-18T09:20Z during Class B execution. Class C refines based on
what Class B establishes.

## What Class B established (so far, mid-run)

1. **BA (b_init_scale=1e-3): FAILS** — same stuck trajectory as AC. Light
   symmetry break does not rescue c=f32.
2. **BC (A=0, B=rand_small): FAILS** — oscillates at log(2) with no descent.
   Factor-geometry swap doesn't help.
3. **BF (lr=3e-6): PARTIAL** — descends faster (0.60 at step 9 vs AC 0.66)
   but still far from Exp N's 0.32.
4. **BE (warmup=0.0): MILD** — slightly better than AC (0.66 at step 7 vs
   AC 0.66) but same plateau.
5. Pending: BD, BH, BI, BK (noise, CE kernel, no accum, grad cast bf16).
6. Lane 2: BN (matmul precision) v5p-8 running.

## Key unanswered question

**Is c=f32 just a slower version of c=bf16 (LR story), or does it descend
in a different direction (direction story)?**

BF lr=1e-5 (10x baseline, running) will disambiguate:
- If reaches ≤ 0.4: LR story. `c=f32` just needs bigger steps.
- If plateau ≥ 0.5: direction story. `c=f32` grads point "wrong."

## Class C — run after Class B closes

### C0 — Interpretation-setting: BP confirmation

Before pivoting, run **100 steps** of AC and BF to confirm:
- Does AC eventually reach 0.33 given enough steps? (plateau check)
- Does BF-lr3e-6 converge to the good basin if given 100 steps?
- Does BE-wm0 eventually reach 0.33?

If AC plateaus above 0.55 for 100 steps → direction story → f32 is
fundamentally broken.
If AC reaches 0.33 by step 30-50 → LR story → f32 is just slow.

**Run order:**
- CP1: AC recipe, 100 steps (baseline).
- CP2: BF-lr3e-6, 100 steps.
- CP3: BF-lr1e-5, 100 steps (if different from lr-3e-6 result).

### C1 — Isolate the faulty operator (if direction story)

If Class B closes with direction story supported, Class C probes which
operator is faulty at c=f32:

**C1a — matmul precision sweep for LoRA only:** Force `precision="highest"`
(pure f32) specifically on LoRA layers. If training rescues → LoRA's dot
op precision is the culprit. Need to patch `LowRankLinear.__call__` to
pass `dot_general` with `precision=HIGHEST`.

**C1b — LoRA module ablation:** Run with LoRA on only attention, only
MLP, only qkvo. If one family rescues → that submodule is f32-safe, others
are broken.

**C1c — Use unquantized dot:** Force `dot_general=jax.lax.dot_general` with
no reparam in LoRA, bypass any custom quantization path. Tests whether
haliax's `DotGeneralOp.default()` has a quirk at f32.

### C2 — Operator dtype shuffling (if LR story plus a twist)

**C2a — p=bf16, c=f32**: policy params bf16, compute f32. Tests whether
the "c=bf16 works" story is about parameter dtype or compute dtype.

**C2b — p=f32, c=bf16**: params f32, compute bf16. Exp N has this default,
so it should work. But let's confirm.

**C2c — Custom mixed precision**: cast inputs to bf16 inside forward,
keep grads in f32.

### C3 — Optimizer path deep dive (if BK rescues)

If BK rescues (cast grads to bf16 before Adam), Class C digs into Adam's
update formula at f32. Candidate issues:
- Adam's v (second moment) underflows at f32 when grads are ~2.4 L2.
- Division by sqrt(v + eps) where eps is large relative to v.
- NaN propagation on tiny denominators.

### C4 — Reference model dtype (orthogonal)

**C4a — Reference in f32**: currently reference shares base weights cast
to compute_dtype. Try forcing reference-only in bf16.

**C4b — Reference-policy dtype split**: policy computes in f32, reference
computes in bf16. Directly tests the "log-ratio exchange under dtype mismatch"
theory from the old H_ref_mismatch.

### C5 — Width-4 bug 1 probes (Lane 2 continuation)

If BN (matmul precision) doesn't rescue Bug 1:
- **C5a — Submodule ablation on v5p-8**: same as C1b but for Bug 1.
- **C5b — Try older XLA version**: rebuild image with jax/xla from 2 weeks ago
  (before any collective/GSPMD changes).
- **C5c — Explicit shard_map FSDP**: replace GSPMD FSDP with explicit
  shard_map, eliminate compiler opacity.

## Time budget

Remaining after Class B closes (est. 4h): ~5h for Class C.
- CP1-CP3 confirmations: ~1.5h (longer runs).
- C1 or C2 or C3 (whichever branch opens): ~2h.
- C4: ~1h (one run).
- C5: ~1h (if Lane 2 continues).

## Launch policy

All Class C experiments follow Class B standing rules:
- WANDB_MODE=offline (wandb online init is flaky).
- MARIN_DEBUG_SKIP_HF_EXPORT=1.
- AdapterBaseReferenceConfig.
- Multi-region launch where capacity allows.
