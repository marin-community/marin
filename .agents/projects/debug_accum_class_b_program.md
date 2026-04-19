# Class B Experiment Program: LoRA DPO Failure After AB/AC/AD

## Problem

The current investigation has two live failure surfaces that should be treated separately:

1. **Bug 1**: canonical LoRA DPO on `v5p-8` with canonical FSDP (`data:4`, `model:1`), `c=bf16`, `zero_init_b=True` gets stuck near `log(2)` instead of escaping by step 2.
2. **Bug 2**: on an otherwise-good mesh (`v5p-16`, `pd=2`), switching only to `c=f32` also gets stuck under the canonical LoRA init.

AB and AC changed the state of play:

- [experiment_ab_v5p8_fp32_pd4_hlo_s10.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/posttrain/per_stmt_dpo/experiment_ab_v5p8_fp32_pd4_hlo_s10.py) showed that the width-4 grad reductions under `c=f32` are already `f32`, so the old “bf16 collective dtype” story is dead.
- [experiment_ac_v5p16_fp32_pd2_hlo_s10.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/posttrain/per_stmt_dpo/experiment_ac_v5p16_fp32_pd2_hlo_s10.py) showed that `c=f32` is bad even on the known-good `v5p-16,pd=2` recipe.
- AD v3 suggests that `c=f32` is not globally broken: when `zero_init_b=False`, the run descends materially, but that experiment is confounded because `zero_init_b=False` in [lora.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/lora.py:209) uses the full `Linear.init` path for `B`, which is a large perturbation, not a “small symmetry break.”

The Class B program is designed to do two things:

1. **Solve Bug 2 first**, because it contaminates any further `c=f32` interpretation.
2. **Then return to Bug 1** with a narrower search space and better instrumentation.

## Goals

- Determine whether Bug 2 is primarily:
  - a canonical-init / symmetry issue,
  - a reference-path issue,
  - a CE kernel issue,
  - a grad-accum / optimizer-path issue,
  - or something broader in the f32 forward/backward graph.
- Determine whether Bug 1 is primarily:
  - width-4 topology / collective-order specific,
  - local dot/matmul algorithm specific,
  - or localized to a subset of LoRA target modules.
- End the overnight run with at least one of:
  - a working rescue for Bug 2 on the canonical recipe family,
  - a narrowed mechanism with one rejected branch and one promoted branch,
  - or a compact negative-result set that cleanly eliminates a whole class of causes.

## Non-Goals

- Full productionization tonight.
- Broad hyperparameter fishing without mechanistic value.
- Re-litigating AA/AB/AC in the logbook before new evidence arrives.

## Current Facts

- Canonical init is `A=random`, `B=0` when `zero_init_b=True`; see [lora.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/lora.py:223).
- `zero_init_b=False` does **not** mean “small nonzero B”; it means full [Linear.init](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/haliax/src/haliax/nn/linear.py:51) on `B`.
- With canonical init, step-0 trainable gradient is structurally one-sided: `dA = 0`, `dB != 0`. That much is real and already reflected in the logs.
- `_train_step` computes grads and immediately calls `state.take_step(grads, ...)` inside the same compiled function; see [trainer.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/trainer.py:678).
- Microbatch accumulation reshards inside [grad_accum.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/grad_accum.py:121), so accumulation-site interventions must be placed carefully.
- AdapterBase + LoRA + nonzero `B` is guarded in [train_dpo.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/main/train_dpo.py:369), which means some Bug 2 probes either need the debug gate or need `SeparateReferenceConfig`.

## Shared Infra (B0)

Before running most of Class B, land a small instrumentation patch that enables clean probes rather than ad hoc one-off hacks.

### B0.1: Add LoRA init controls

**Files**
- [lora.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/lora.py:135)

**Add**
- `b_init_scale: float | None = None`
- `a_init_mode: Literal["random", "zero"] = "random"`
- `b_init_mode: Literal["zero", "random"] = "zero"` or equivalent compact API

**Why**
- BA/BB need a *small* nonzero `B`.
- BC needs `A=0`, `B=random_small`, while preserving `B @ A = 0` at init.

### B0.2: Add one-shot LoRA grad perturbation controls

**Files**
- [trainer.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/trainer.py:683)

**Add env-gated knobs**
- `MARIN_DEBUG_LORA_GRAD_NOISE_STD`
- `MARIN_DEBUG_LORA_GRAD_NOISE_STEP`
- `MARIN_DEBUG_LORA_GRAD_NOISE_TARGET={A,B,both}`
- optional `MARIN_DEBUG_LORA_GRAD_CAST={none,bf16,f32}`

**Why**
- BD, BK, and optimizer-path isolators should use the same mechanism.

### B0.3: Add LoRA grad artifact dump

**Files**
- [trainer.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/trainer.py:700)

**Add**
- Full or sliced LoRA grad dump for step 0 and step 1 to a local artifact, not just checksums.
- Enough to compute cosine similarity between runs for `lora_A` and `lora_B` by module family.

**Why**
- BJ is much more useful than another round of only `l2`/`sum`.

### B0.4: Mesh reorder hook

**Files**
- likely experiment scripts only, reusing the mesh override style from [experiment_w_v5p8_mesh_s10.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/posttrain/per_stmt_dpo/experiment_w_v5p8_mesh_s10.py)

**Add**
- Explicit logical-to-physical reorder variants for `data:4`.

**Why**
- BL needs a clean way to test topology/order without changing the logical recipe.

## Execution Strategy

Treat this as two lanes:

- **Lane 1: Bug 2 (`c=f32`)** on `v5p-16,pd=2` first. This is the highest-priority lane because it blocks interpretation of any further f32 probes.
- **Lane 2: Bug 1 (`v5p-8,data:4,c=bf16`)** in parallel only if there is spare capacity.

### Overnight Stop Criteria

Stop Lane 1 when one of these happens:

1. A *light* symmetry-break experiment rescues training reproducibly.
2. Three orthogonal f32 hypotheses fail:
   - init/symmetry,
   - reference-path,
   - kernel/accum/optimizer path.

Stop Lane 2 when one of these happens:

1. A topology or precision-algorithm intervention rescues width-4.
2. Two topology-like probes and one local-compute probe all fail, in which case the next move is deeper instrumentation, not more runs.

## Master Table

| Run | Lane | Base | Key change | Hypothesis | Parallel? |
|---|---|---|---|---|---|
| BA | Bug 2 | AC / Exp N | `c=f32`, `zero_init_b=True`, **small** `b_init_scale=1e-3` | A light `B` perturbation rescues canonical f32 | No |
| BB | Bug 2 | BA | scale sweep `1e-4, 1e-3, 1e-2` | There is a usable rescue window, not only “huge random B” | Yes |
| BC | Bug 2 | Exp N | `A=0`, `B=random_small`, `B@A=0` at init | The issue is factor geometry, not merely policy-reference mismatch | No |
| BD | Bug 2 | AC | one-shot noise on `lora_B` grad at first nonzero update | Tiny noise injection rescues f32 canonical init | Yes |
| BE | Bug 2 | AC | `warmup=0.0` and `warmup=0.02` | Zero-LR step 1 is part of the trap | Yes |
| BF | Bug 2 | AC | LR sweep `3e-6`, `1e-5` | Same direction exists at f32; current LR is just too small to escape | Yes |
| BG | Bug 2 | AC | `SeparateReferenceConfig` | f32 failure is AdapterBase-specific | No |
| BH | Bug 2 | AC | pure-XLA CE / no Pallas CE | f32 failure lives in CE kernel path | No |
| BI | Bug 2 | AC | no grad accumulation / changed `pd` with same mesh | f32 failure lives in grad-accum or intra-step reshards | No |
| BJ | Both | N / AC / best rescue | full LoRA grad artifact compare | Early-grad direction localizes the first useful discriminator | Analysis |
| BK | Bug 2 | AC | cast `grads -> bf16` before optimizer | f32 failure is optimizer/update-path, not f32 forward/backward | No |
| BL | Bug 1 | Exp Q | mesh reorder at `data:4` | width-4 failure depends on physical reduction order | Yes |
| BM | Bug 1 | Exp Q | cross-family width-4 on larger pod | width-4 bug is generic vs v5p-specific | No |
| BN | Bug 1 | Exp Q | bf16 matmul precision sweep | local dot algorithm contributes to width-4 failure | Yes |
| BO | Both | Exp Q / AC | target-modules ablation | one module family dominates the failure | Yes |
| BP | Both | best rescue / best null | 100-step confirmation | short-run rescue is real, not transient | No |

## Detailed Experiment Cards

### BA: Small nonzero `B` init under `c=f32`

**Base**
- [debug_r64_matched_pd2_s10.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/posttrain/per_stmt_dpo/debug_r64_matched_pd2_s10.py)
- scientific delta relative to AC only

**Change**
- `mp=jmp.get_policy("p=f32,c=f32")`
- keep `AdapterBaseReferenceConfig`
- keep canonical recipe
- use `b_init_scale=1e-3`, not full `zero_init_b=False`

**Hypothesis**
- The canonical f32 failure is caused by exact `B=0` symmetry plus the first update geometry.
- A *light* `B` perturbation is enough to recover without blowing up step 0.

**Success**
- Step-0 loss remains near `log(2)` or at least not catastrophic.
- Step-2 loss drops clearly below `0.5`.

**If success**
- Run BB to map the usable window.
- Run BP for a 100-step confirmation.

**If fail**
- Run BC next before abandoning the symmetry story.

### BB: `B` init scale sweep

**Scales**
- `1e-4`, `1e-3`, `1e-2`

**Hypothesis**
- There is a narrow rescue window between “too symmetric” and “too disruptive.”

**Interpretation**
- Only `1e-2` works: strong symmetry break needed; weakens the “tiny noise is enough” story.
- `1e-4` or `1e-3` works: strengthens the symmetry-break story substantially.
- None work: move to BC/BD/BH.

### BC: `A=0`, `B=random_small`, still `B@A=0`

**Change**
- Add an init mode that preserves `W' = W_base` at step 0 but routes the first nonzero gradient into `A` instead of `B`.

**Hypothesis**
- The problem is not just “policy equals reference at init.”
- It may be specific to the `A=random, B=0` factor geometry where only `B` gets a gradient first.

**If success**
- Promote “first-updated factor geometry” as the leading mechanism.
- Run BJ immediately to compare `A`-vs-`B` early-grad directions.

**If fail**
- The issue is less likely to be factor-asymmetry-specific; continue to BD/BG/BH.

### BD: One-shot LoRA-grad noise injection

**Change**
- Keep canonical init.
- At the first nonzero optimizer step, add small Gaussian or Rademacher noise to `lora_B` grads only.

**Noise sweep**
- `1e-6`, `1e-5`, `1e-4` relative to grad scale.

**Hypothesis**
- The f32 trap is broken by a tiny perturbation of the first effective update.

**If success**
- This supports a noise/degeneracy story, but not necessarily a bf16-specific story.
- Pair with BJ to see how little directional change was needed.

**If fail**
- Move to BE/BG/BH; the issue is not just “needs a shove.”

### BE: Warmup ablation

**Change**
- Keep AC recipe.
- Try `warmup=0.0` and `warmup=0.02`.

**Hypothesis**
- The zero-LR step 1 may be preserving the exact bad geometry too long.

**If success**
- Promote scheduler/first-update timing as part of Bug 2.
- Combine with BA or BD later.

**If fail**
- Scheduler is not load-bearing.

### BF: Learning-rate sweep

**Change**
- Keep AC recipe.
- Try `3e-6` and `1e-5`.

**Hypothesis**
- The correct escape direction exists at f32, but the current step size is too weak.

**If success**
- This is a practical workaround, but not a mechanism.
- Combine with BJ to see whether the direction was similar and only magnitude changed.

**If fail**
- Move on; do not keep fishing LR.

### BG: `SeparateReferenceConfig` under `c=f32`

**Change**
- Same as AC, but use `SeparateReferenceConfig`.

**Hypothesis**
- The f32 failure is specific to AdapterBase’s coupling to the policy init and reference construction.

**If success**
- The reference path matters more than currently believed.
- Run BJ and compare reference-path HLO/grad traces.

**If fail**
- Bug 2 is not AdapterBase-specific.

### BH: CE kernel isolation

**Change**
- Same as AC.
- Force the pure-XLA CE path / disable the Pallas CE path if possible.

**Hypothesis**
- The f32 failure lives in CE backward numerics or a dtype-specific CE code path.

**If success**
- Focus on the CE kernel, not on LoRA init.

**If fail**
- CE is less likely to be the load-bearing cause of Bug 2.

### BI: Grad accumulation isolation

**Change**
- Same as AC, but alter `pd` / batch geometry to remove or reduce microbatch accumulation while keeping the mesh fixed if possible.

**Hypothesis**
- The f32 failure is in `grad_accum.py`’s accumulation/reshard loop, not in the model forward/backward proper.

**If success**
- Audit [grad_accum.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/grad_accum.py:36) hard.

**If fail**
- Accumulation is not the main Bug 2 driver.

### BJ: Full early-grad artifact compare

**Targets**
- Exp N (good)
- AC (f32 bad)
- best BA/BB/BC/BD rescue, if any

**Hypothesis**
- The first useful discriminator will show up in step-0 or step-1 LoRA grad direction, not only in scalar norms.

**What to compute**
- cosine similarity by module family
- per-module `A` vs `B` norm ratios
- sign agreement on top-k magnitude entries

**If the rescue run matches N more than AC**
- early-grad direction is the right level of explanation

**If it still matches AC**
- the decisive difference happens later: optimizer state, schedule, or second-step dynamics

### BK: Optimizer-path isolation with grad cast

**Change**
- Keep AC recipe.
- Cast grads to `bf16` just before `state.take_step(...)`.

**Hypothesis**
- The failure is in the optimizer/update path consuming f32 grads, not in f32 forward/backward compute.

**If success**
- Audit [trainer_state.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/trainer_state.py:137) and optimizer state interactions.

**If fail**
- The problem is upstream of the optimizer.

### BL: Width-4 mesh reorder on `v5p-8`

**Change**
- Keep Exp Q recipe.
- Reorder which physical devices participate in the `data:4` path without changing the logical recipe.

**Hypothesis**
- Bug 1 depends on physical collective order/topology, not only abstract width 4.

**If success**
- Topology/order is the leading width-4 mechanism.

**If fail**
- Move to BM/BN; width alone still matters but reorder did not.

### BM: Cross-family width-4 emulation on larger pods

**Change**
- Use a larger pod where width-4 can be embedded cleanly, e.g. `v5p-16` or `v6e-16` with mesh settings that expose a `data:4` path and extra replica axes.

**Hypothesis**
- Bug 1 is either:
  - generic to width-4 data parallelism, or
  - specific to `v5p-8` topology/family.

**If width-4 fails cross-family**
- Promote “generic width-4” mechanism.

**If width-4 is only bad on v5p-8**
- Promote v5p-specific topology or compiler-lowering differences.

### BN: bf16 matmul precision sweep on width-4

**Change**
- Keep Exp Q recipe.
- Sweep `jax_default_matmul_precision` or explicit presets if available.

**Hypothesis**
- Bug 1 is influenced by local dot algorithm choice even though collective dtype was a red herring.

**If success**
- Width-4 interacts with local compute, not only collectives.

**If fail**
- Move weight away from local dot precision.

### BO: Target-modules ablation

**Change**
- Run attention-only, MLP-only, and maybe qkvo-only LoRA on:
  - `v5p-8,c=bf16,data:4` for Bug 1
  - `v5p-16,c=f32,pd=2` for Bug 2

**Hypothesis**
- One submodule family dominates one or both failures.

**If one family is sufficient**
- Use that family for all subsequent cheap probes.

**If all families behave similarly**
- The problem is generic to the LoRA update path.

### BP: 100-step confirmation

**Change**
- Take the best rescue and the strongest null.

**Hypothesis**
- Short-run rescue reflects a real training regime shift.

**If rescue holds**
- Promote it to practical workaround status.

**If it degrades back to the bad basin**
- It was only a transient fix.

## Recommended 10-Hour Run Order

### Hour 0-1.5: Land B0

1. Add `b_init_scale` and init-mode controls in [lora.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/lora.py:135).
2. Add grad-noise and grad-cast hooks in [trainer.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/trainer.py:683).
3. Add richer LoRA grad artifact dumping.

### Hour 1.5-4: Solve Bug 2 first

1. Run BA.
2. If BA is promising, launch BB scales in parallel.
3. If BA is not promising, run BC next.
4. In parallel with either BA or BC, run BD if capacity allows.

### Hour 4-6.5: Kill Bug 2 branches aggressively

1. If init/noise branches fail, run BG and BH.
2. If BG and BH both fail, run BI or BK, not both, depending on what the grad artifacts suggest.

### Hour 6.5-9: Return to Bug 1

1. Launch BL and BN in parallel if capacity allows.
2. If one hits, do not keep spraying more width-4 runs; go straight to BP or a tighter confirmation.
3. If both miss, run BM only if a clean width-4 embedding is available on larger pods.

### Hour 9-10: Consolidate

1. Run BJ analysis on the most informative pair or trio.
2. Queue BP if there is a clear best candidate.

## Practical Guidance For The Next Agent

- Do not overinterpret AD v3. It is useful evidence that “f32 can descend when the canonical init is broken strongly,” but it is not a clean symmetry test.
- Do not spend more TPU hours on broad `c=f32` recipe variants until B0 is landed. The small-init and factor-swap probes are much higher information per run.
- Prefer `v5p-16,pd=2` for Bug 2 until you have a clean rescue or a strong null. It is the least confounded baseline we have.
- For Bug 1, prefer interventions that keep the canonical recipe untouched except for one width/topology/precision variable.

## Future Work

- If BA or BC succeeds cleanly, add a permanent research knob rather than carrying env-gated debug code forever.
- If BL identifies a topology-order dependence, build a minimal reproducer around the offending collective pattern outside full DPO.
- If none of BA-BK rescue Bug 2, the next wave should pivot to remat/checkpointing, attention kernel variants, and full per-layer activation diffs rather than more hyperparameter sweeps.
