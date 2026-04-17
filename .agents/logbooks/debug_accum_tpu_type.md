# Debug: v6e-8 vs v5p-8 DPO Training Discrepancy

> **Pointer for the next agent (2026-04-17T00:30Z):** Experiment T has now
> completed with a usable 10-step `v5p-8` full-FT run, and it materially
> changes the hypothesis tree. The strongest current picture is:
> - (leading) **LoRA / adapter-specific path on `v5p-8`** — LoRA update
>   geometry, adapter-only sharding / optimizer, or
>   `AdapterBaseReferenceConfig`
> - (still possible but demoted) LoRA-specific interaction with attention
>   `kv_head` mapping or another sub-CE distributed detail
> - (ruled out) CE kernel tiling / bf16 accumulation / per-chip CE workload
> - (ruled out) TPU family alone, `pd` alone, LoRA rank alone
> - (strongly weakened) broad "the `v5p-8` execution graph itself breaks
>   DPO even without LoRA"
>
> Why: the new Exp T rerun below shows that **`v5p-8` full FT does learn**.
> It is not in the catastrophic bad LoRA basin. See
> `2026-04-17T00:30Z: Experiment T result` immediately below.

## 2026-04-17T01:36Z: Experiment U staged — rerun the bad `v5p-8` LoRA regime with `p=f32,c=f32` at `pd=4`

### What changed in code

Prepared a new experiment script:

- `experiments/posttrain/per_stmt_dpo/experiment_u_v5p8_fp32_pd4_s10.py`

Also threaded mixed precision through the simplified DPO path so debug probes can
set it explicitly instead of editing `defaults.py` inline each time:

- `SimpleDPOConfig` now exposes `mp`
- `default_dpo(...)` now passes `dpo_config.mp` into `TrainerConfig`

This keeps the new probe local and reversible. It does **not** change the
default DPO precision for existing experiments, because the new field defaults
to the prior policy:

- default remains `p=f32,c=bfloat16`

### Why this is the next experiment

David suggested the simplest remaining numeric probe:

> "have you tried putting the whole thing in float32"

Interpreted in Levanter / JMP terms, that means:

- `p=f32,c=f32`

This is materially broader than the earlier CE-only upcast probe:

- it changes the policy forward / backward compute dtype
- it changes the frozen non-trainable base-weight dtype under LoRA
- it changes the reference-path compute dtype
- it changes activations / logits / CE inputs throughout the DPO graph

So Experiment U is a clean test of:

> **Is the bad `v5p-8` LoRA regime specifically tied to bf16 compute rather than to LoRA/reference topology more generally?**

### Exact Experiment U configuration

Experiment U is intentionally a near-copy of the **Experiment Q `pd=4` branch**
with only one scientific knob changed.

Held fixed from Exp Q:

- TPU: `v5p-8`
- data: per-stmt `support_mental_health`
- LoRA: `r=64`, `alpha=64`, `dropout=0.0`, `zero_init_b=True`
- reference path: `AdapterBaseReferenceConfig`
- `train_batch_size=64`
- `num_train_steps=10`
- `steps_per_eval=10`
- `lr=1e-6`
- `lr_schedule="cosine"`
- `warmup=0.1`
- `beta=0.1`
- `train_seq_len=max_seq_len=4096`
- `reference_eval_cache=disabled`
- `max_eval_batches=1`
- debug traces on (`MARIN_DEBUG_LOG_BATCH_INDICES=1`, `MARIN_DEBUG_LOG_STEP_TRACE=1`)

Changed from Exp Q:

- mixed precision: **`p=f32,c=f32`**

Pinned for memory control:

- `per_device_parallelism=4`
- `per_device_eval_parallelism=4`

### Why force `pd=4`

The point of U is to test **whole-graph fp32 compute**, not to maximize HBM
stress. On `v5p-8`, peak HBM is dominated by activation / temporary tensors,
and those scale with the live microbatch (`pd`), not merely with the optimizer's
global batch.

At a rough first-principles level for this 8B LoRA DPO regime:

- `pd=4`, bf16-compute: about mid-20s GiB/chip
- `pd=4`, full-f32-compute: about high-40s GiB/chip
- `pd=8`, bf16-compute: about mid-40s GiB/chip
- `pd=8`, full-f32-compute: about mid-80s GiB/chip

These are ballpark estimates, not measured peaks, but they are directionally
good enough to justify the choice:

- `pd=4` should give a **much safer fit test** on `v5p-8`
- `pd=8` might still fit, but it would mix the precision probe with a much
  tighter HBM regime

So `pd=4` is the right first shot if the goal is to isolate the effect of
whole-graph fp32.

### What U will teach us

If Experiment U **recovers strongly** relative to bad Exp Q / bad `v5p-8`
LoRA runs, then the explanation moves toward:

- bf16-compute sensitivity in the LoRA/reference path on `v5p-8`
- or a numerics issue that only appears once the adapter/base-reference graph is
  run in bf16

If Experiment U is **still bad**, then "just run it in fp32" is mostly ruled
out, and the leading remaining suspects stay structural:

- `AdapterBaseReferenceConfig`
- LoRA-specific param / grad sharding
- LoRA update geometry on the 4-device `v5p-8` mesh
- possibly attention `kv_head` / mesh-mapping interaction

### Launch command

```bash
REGIONS_OVERRIDE=us-east5 \
MARIN_DEBUG_RUN_TAG=ue5a \
uv run python experiments/posttrain/per_stmt_dpo/experiment_u_v5p8_fp32_pd4_s10.py
```

Expected run name pattern:

- `dpo/stmt_dpo/debug/experiment_u_r64_v5p8_pd4_fp32_s10_<tag>`

### Status

- Experiment U is **prepared but not yet launched**
- This is now the cleanest "numeric precision without CE tunnel vision" probe in
  the queue
- If U fails to fit despite the `pd=4` guardrail, that itself is evidence that
  whole-graph fp32 is impractical for this LoRA DPO regime on `v5p-8`

## 2026-04-17T00:30Z: Experiment T result — `v5p-8` full FT **LEARNS**; broad `v5p-8` execution-graph failure is no longer the leading explanation

### Strongest true conclusion

The completed 10-step Experiment T rerun on **April 17, 2026** shows that
`v5p-8` can run **full-FT DPO** and learn in the same *qualitative* regime as
the previously-good full-FT baselines.

This is the cleanest answer yet to the question Exp T was launched to settle:

> **Does the `v5p-8` pathology survive when LoRA is removed?**

The answer is now:

> **No, not in the catastrophic sense seen in the LoRA runs.**

This `v5p-8` full-FT run is not a perfect numeric match to the good 16-chip
full-FT baselines, but it is decisively **closer to the good full-FT regime
than to the bad `v5p-8` LoRA regime**.

The practical implication is strong:

1. **`v5p-8` full FT is feasible.** The run compiled, trained for 10 steps,
   evaluated, and wrote checkpoints / HF export successfully.
2. **The catastrophic stuck-near-`ln(2)` behavior does not reproduce under
   full FT on `v5p-8`.**
3. Therefore the remaining pathology is now much more likely to live in the
   **LoRA / adapter-specific training path** on `v5p-8`:
   - LoRA low-rank update geometry,
   - adapter-only optimizer / sharding behavior,
   - `AdapterBaseReferenceConfig`,
   - or an interaction between those and a still-live sub-CE distributed detail.

This strongly weakens the broader theory:

> "Something generic about the `v5p-8` distributed execution graph breaks DPO,
> even without LoRA."

That theory is no longer the best explanation after this run.

### Run

- `v5p-8` full FT, `bs=32`, `pd=4`, `steps=10`:
  https://wandb.ai/marin-community/dpo/runs/exp_t_v5p8_fullft_bs32_pd4_s10_uc1-rerun-20260416-3-stream-042354
- Run name:
  `exp_t_v5p8_fullft_bs32_pd4_s10_uc1-rerun-20260416-3-stream-042354`
- Worker `output.log` confirms:
  - first train batch started at `2026-04-17T00:17:44Z`
  - step 9 completed by `2026-04-17T00:22:06Z`
  - final eval + checkpoint/HF export completed by `2026-04-17T00:30:25Z`

### Training-relevant config

What is directly established from the run name, worker log, and W&B history:

- TPU: `v5p-8`
- data: per-stmt `support_mental_health`
- **full FT** (no adapter)
- reference path: `SeparateReferenceConfig`
- `train_batch_size=32`
- `per_device_parallelism=4`
- `num_train_steps=10`
- `lr=1e-6`
- `beta=0.1`
- `seed=0`
- `train_seq_len=max_seq_len=4096`
- `stmt_val` + `full_val` eval at the end

Important nuance:

- This is **not** a fully apples-to-apples replay of Exp L / Exp O, because
  the global batch is smaller here (`bs=32` vs the earlier `bs=64`
  full-FT baselines).
- So the right conclusion is **not** "the curves match exactly."
- The right conclusion is the stronger qualitative one: **this run is in the
  same learning regime as the good full-FT runs and far from the bad LoRA
  regime.**

### CE line on this run — descriptive, not causal

The worker log shows:

```
DEBUGCE XLA CE block sizes resolved: device_kind=TPU v5 x.shape=(32768, 4096) w.shape=(4096, 128256) v_block_size=8192 b_block_size=32768 num_v_blocks=16 num_b_blocks=1 explicit_block_sizes=False
```

This matches the "single batch block" CE regime seen in the good
`v5p-16 pd=2` full-FT / LoRA runs and differs from the bad `v5p-8` LoRA
baseline.

But this should **not** be read as reviving the CE hypothesis. Exp R2a already
ruled out CE tiling / CE inter-block accumulation as the load-bearing cause on
the bad LoRA baseline. The CE line here is best treated as a descriptive part
of the run's local execution geometry, not the newly-proved root cause.

### Full 10-step training trajectory (from `DEBUGJ TRACE` worker logs)

| step | loss     | grad_l2  | grad_sum   |
|------|----------|----------|------------|
| 0    | 0.693139 | 31.0029  | -23.8408   |
| 1    | 0.693142 | 29.5883  | +16.2643   |
| 2    | 0.687056 | 28.4906  | -3.7742    |
| 3    | 0.680934 | 27.3795  | -11.7273   |
| 4    | 0.664067 | 28.4214  | -5.8817    |
| 5    | 0.660287 | 28.4812  | +10.0076   |
| 6    | 0.622947 | 27.4912  | +12.2259   |
| 7    | 0.616519 | 26.6690  | -30.4674   |
| 8    | 0.608089 | 27.2849  | +12.3777   |
| 9    | 0.608325 | 25.3038  | -2.6215    |

Qualitative read:

- The run is **not stuck** at `ln(2)`.
- It leaves `ln(2)` by step 2 and reaches `~0.608` by step 9.
- The full-FT gradient norm declines in the expected broad range
  (`31.0 -> 25.3`), unlike the pathological LoRA traces that remain in the
  wrong basin with much weaker descent.

### Side-by-side: Exp T vs good full-FT baselines vs bad `v5p-8` LoRA

| step | Exp T `v5p-8` full FT `bs=32 pd=4` | Good `v5p-16` full FT `pd=4` (Exp L) | Good `v6e-16` full FT `pd=2` (Exp O) | Bad `v5p-8` LoRA `pd=4` (Exp Q) |
|------|------------------------------------|--------------------------------------|--------------------------------------|---------------------------------|
| 0    | 0.693139 | 0.693147 | 0.693163 | 0.693147 |
| 1    | 0.693142 | 0.693147 | 0.693179 | 0.693147 |
| 2    | 0.687056 | 0.688913 | 0.686007 | 0.685125 |
| 3    | 0.680934 | 0.673635 | 0.673826 | 0.682298 |
| 4    | 0.664067 | 0.663108 | 0.667567 | 0.673723 |
| 5    | 0.660287 | 0.656349 | 0.655773 | 0.668946 |
| 6    | 0.622947 | 0.615969 | 0.615281 | 0.667573 |
| 7    | 0.616519 | 0.603591 | 0.601114 | 0.662823 |
| 8    | 0.608089 | 0.593090 | 0.588456 | 0.658715 |
| 9    | 0.608325 | 0.593389 | 0.592088 | 0.660557 |

This table is the core result:

- Exp T is **very close** to the known-good full-FT runs.
- Exp T is **far away** from the bad `v5p-8` LoRA regime by the later steps.
- At step 9, Exp T is only about `~0.015-0.016` worse than the good full-FT
  baselines, but about `~0.052` better than the bad `v5p-8` LoRA run.

### The actual DPO quantity: `delta_pi - delta_ref`

As elsewhere in this logbook, the loss-driving quantity is:

`delta_pi - delta_ref = train/dpo_margin_policy - train/dpo_margin_ref`

For Exp T:

| step | Exp T `delta_pi - delta_ref` |
|------|------------------------------|
| 0    | 0.000153 |
| 1    | 0.000076 |
| 2    | 0.124344 |
| 3    | 0.248108 |
| 4    | 0.593155 |
| 5    | 0.670395 |
| 6    | 1.462914 |
| 7    | 1.605316 |
| 8    | 1.789200 |
| 9    | 1.789017 |

Useful comparison points at step 9:

| run | step-9 `delta_pi - delta_ref` |
|-----|-------------------------------|
| Exp T `v5p-8` full FT | **1.7890** |
| Good `v5p-16` full FT (Exp L) | **2.1208** |
| Bad `v5p-8` LoRA (Exp Q) | **0.6647** |

So Exp T is much closer to the good full-FT regime than to the bad LoRA
regime on the true DPO quantity as well, not just on train loss.

### Validation-set behavior

| split | pre-training | post-step-9/10 eval | Δ |
|-------|-------------:|--------------------:|--:|
| stmt_val | 0.6931 | 0.6116 | -0.0815 |
| full_val | 0.6931 | 0.6913 | -0.0018 |

Interpretation:

- On the statement validation split, Exp T shows clear learning.
- On `full_val`, transfer after only 10 steps is weak, but it is still
  directionally better than the bad `v5p-8` LoRA regime.
- This is normal for a short per-statement probe and does not weaken the main
  conclusion about train-time regime.

### What Exp T rules out, supports, and leaves open

**Rules out / strongly weakens:**

- "The `v5p-8` DPO pathology is broad to full FT as well as LoRA."
- "Something generic about the `v5p-8` execution graph prevents DPO from
  learning, even when LoRA is removed."
- "The next best use of time is more generic full-FT / remat / CE debugging on
  `v5p-8`."

**Strongly supports:**

- The remaining failure surface is now primarily in the **LoRA / adapter
  training path** on `v5p-8`.
- The most likely live culprits are:
  1. **LoRA-specific update geometry / adapter-only optimizer behavior**
  2. **`AdapterBaseReferenceConfig`**, since Exp T uses
     `SeparateReferenceConfig` and learns
  3. a LoRA-specific interaction with a still-live distributed detail such as
     attention `kv_head` mapping or adapter-parameter sharding

**Not yet proved:**

- Whether **LoRA alone** is sufficient to cause the remaining `v5p-8`
  pathology, or whether the real load-bearing variable is specifically
  `LoRA + AdapterBaseReferenceConfig`.
- Whether a `v5p-8` LoRA run with `SeparateReferenceConfig` would learn
  normally.
- Whether adapter-only sharding / optimizer-state handling differs on
  `v5p-8` vs the known-good pods in a way that only matters for LoRA.

### Revised next-best experiments after Exp T

Exp T changes the next-step ranking substantially.

Highest-value next probes now are:

1. **LoRA on `v5p-8` with `SeparateReferenceConfig`**
   - Keep the per-stmt data and `v5p-8` geometry
   - remove `AdapterBaseReferenceConfig`
   - this is the cleanest discriminator between:
     - "LoRA itself is the remaining problem" and
     - "`AdapterBaseReferenceConfig` is the remaining problem"

2. **Adapter parameter / gradient / optimizer sharding dump on `v5p-8`**
   - inspect live LoRA param shardings, grad shardings, and opt-state shardings
   - compare against a known-good LoRA run (`v5p-16` or `v6e-16`)

3. **Only after those:** targeted LoRA-only `kv_head` mapping probe
   - if the LoRA path still looks broken after the reference-config split,
     then attention-axis mapping becomes a cleaner next lever

Lower priority now:

- more generic `v5p-8` full-FT feasibility work
- more CE-kernel work
- broad "maybe remat / FSDP / collectives break everything on `v5p-8`"
  investigation without a LoRA-specific discriminator

---

## 2026-04-16T17:20Z: Experiment T handoff — **XLA compile bug** on `offload`, `recompute` in-flight

**Status update (2026-04-17T00:30Z):** superseded by the completed
10-step Exp T rerun immediately above. The remainder of this section is kept as
historical launch/debug context for the earlier failed `offload` attempt and
the original fallback ladder.

### Executive summary for the next agent

**TL;DR:** Experiment T attempt 2 (`ue5a-i2`, `offload` checkpointing, `bs=32`, `pd=4`) **reached a running TPU worker** at `08:36Z`, then died with an **XLA compile-time check failure** at `08:34Z` wall-clock (exit 139, SIGSEGV). A third attempt with `gradient_checkpointing="recompute"` has been submitted and an auto-restart loop is walking the fallback ladder. Pick up from that attempt.

### What failed on attempt i2 (`offload`, `bs=32`, `pd=4`)

- Iris job id: `/ahmed/debug-t1-full-ft-v5p8-bs32-pd4-offload-ue5a-i2`
- Final state: `state=failed, task=failed, preempt=2`
- Exit: `139` (SIGSEGV)
- XLA stderr (truncated by iris): `F0416 08:34:09.888667  1219 async_dynamic_index_emitter.cc:584] Check failed: intermediate_calc.slice_size % interme...`
- Interpretation: this is an **XLA internal assertion** inside the async dynamic-index emitter, not a preemption, not an HBM OOM, not a process kill. It fires during compile of the full-FT DPO program with `gradient_checkpointing="offload"` on `v5p-8`. The `offload` checkpointing path materializes host-offloaded carries with dynamic-slice shape math, which is where this check lives. So "offload on `v5p-8` + full FT + `SeparateReferenceConfig` + `bs=32 pd=4`" hits the bug; this is *not* a general `v5p-8` compile failure.

### What is running now

- **Currently in-flight**: `/ahmed/debug-t1-full-ft-v5p8-bs32-pd4-recompute-ue5a-i3`
- **Only change vs i2**: `EXPERIMENT_T_CHECKPOINTING=recompute` (was `offload`)
- Submission time: `2026-04-16T17:20Z`
- Parent launch: `iris job run --zone us-east5-a --cpu 1 --memory 3g --enable-extra-resources` is **not** used — parent is plain `--memory 3g` (intentional, see Exp Q ops notes).
- Script: `experiments/posttrain/per_stmt_dpo/experiment_t_v5p8_full_ft_s2.py` (codex-authored, supports env overrides)

### Auto-restart loop (do not duplicate)

An auto-restart bash loop is active in the Claude Code worker (`bg id b57fuy53l`) and writes to `/tmp/t_autorestart.log`. It:

- polls the `train_dpo` child every 10 min
- if it sees `state=failed` *and* no `DEBUGJ TRACE` step progress, it kills the parent and submits the next fallback in the ladder
- if it sees step progress, it stops advancing and just watches

**Fallback ladder encoded in the loop:**

1. i3 (live): `recompute`, `bs=32`, `pd=4`
2. i4 (auto-next on failure): `recompute`, `bs=16`, `pd=4`
3. i5 (auto-next on failure): `offload`, `bs=16`, `pd=4`
4. ladder exhausted → loop stops

The next agent does **not** need to relaunch anything unless:
- the loop is no longer running (check with `pgrep -af t_autorestart`)
- or the ladder is exhausted
- or the next agent wants to try something off-ladder (e.g., `bs=32 pd=2`, or `gradient_checkpointing="default"` with no policy override, or `bs=8`)

### Open behavioral question still waiting on Exp T

None of the Exp T attempts has produced a single `DEBUGJ TRACE` step line yet. So the scientific question the experiment was launched to answer —

> **Does the `v5p-8` pathology survive when LoRA is removed?**

— is still *completely open*. The behavioral comparison to the good full-FT runs (Exp L `v5p-16 pd=4`, Exp L `v6e-16 pd=4`, Exp O `v6e-16 pd=2`) cannot be made until a `v5p-8` full-FT attempt produces at least step 0 and step 2 traces.

### Relevant prior sections in this logbook

- **Exp R2a result (CE kernel branch cleanly ruled out)**: end of logbook at `2026-04-16T22:50Z`. Forced `num_b_blocks=1` at the bad-baseline `B=65536` on `v5p-8 pd=4 bs=64`; still stuck. This is the strongest CE-level rule-out we have. Motivates the full pivot to sub-CE suspects.
- Exp R2 plan (explicit-block-size rationale + three cases + HBM analysis): `2026-04-16T20:00Z`.
- Exp R result (shrinking `v5p-8` CE workload still did not recover it): top of logbook at `2026-04-16T04:52Z`. Note: Exp R did **not** bit-match the good CE tiling; it overshot to `B=16384, num_b_blocks=16`. The clean CE-matching rule-out comes from R2a.
- Root-cause update and Exp T plan: `2026-04-16T13:05Z` ("Root-cause update after Exp R — next best probe is **full FT on `v5p-8`**").
- Exp Q (pd=4 / pd=8) results and ops lessons for `v5p-8` on `us-east5-a`: top of logbook (`2026-04-16T03:17Z`, `2026-04-16T00:50Z`).
- Prior successful `v5p-8` ram tuning: use `ram="150g"` for the child `train_dpo` on this pool; `ram="250g"` will not fit on co-tenanted workers right now. (This is already set in `experiment_t_v5p8_full_ft_s2.py`.)

### What to do when i3 / i4 / i5 completes

1. If any attempt emits `DEBUGJ TRACE step=0` through `step=1` (2-step probe), extract the trace from `iris job logs /ahmed/<job>/train_dpo`, drop into the Exp T results section of this logbook, and compare step-2 loss against the good full-FT baseline.
2. If all three ladder attempts fail in the same XLA way, the next reasonable moves in priority order are:
   - try `EXPERIMENT_T_CHECKPOINTING=default` (no policy override, use the llama_8b default)
   - try `EXPERIMENT_T_PD=2` to change the distributed factorization
   - try `SeparateReferenceConfig` → `AdapterBaseReferenceConfig` swap (this turns Exp T from a pure "is it LoRA?" probe into a "is it the separate-reference graph?" probe; note the interpretation changes)

### Operational regret / lesson

The previous sleep loop (`/tmp/t_sleep_loop.log`) only observed state; it did **not** act on `state=failed`. As a result, Exp T i2 sat in a terminal `failed` state for ~8 hours without a follow-up launch. The new loop acts on failures and walks the ladder, so this class of regret should not recur for Exp T. If the next agent builds a similar watcher for another experiment, include an auto-restart policy from the start.

---

## 2026-04-16T04:52Z: Experiment R result — **shrinking `v5p-8` CE workload still does NOT fix it** (Case 2: no recovery)

### Strongest true conclusion

Experiment R lowered the local XLA CE workload on `v5p-8` substantially by
changing `train_batch_size` from 64 to 32 at `pd=4`, and **the run still stayed
stuck near `ln(2)`**.

This is the "Case 2: no recovery" outcome that the Exp R plan laid out as
the most consequential possibility. It does two things:

1. **Strongly weakens the fused CE kernel batch-blocking hypothesis** as the
   load-bearing cause of the `v5p-8` pathology. We shrank the local CE problem
   far below the bad Exp Q regime and still tracked Exp Q behaviorally.
2. **Pivots the investigation** away from CE-level math toward the broader
   `v5p-8` distributed regime — FSDP sharding, all-gather / reduce-scatter
   topology, attention kv-head mapping, remat scheduling, or the reference
   network's compiled graph.

This is the first Exp Q/R result where **step-0 `grad_l2` itself shifts**
(2.6614 vs 2.4560 in Exp Q / 2.4563 in good `v5p-16 pd=2`). So even the first
forward/backward is producing different accumulated LoRA gradients on `v5p-8
bs=32` than it does on either `v5p-8 bs=64` or `v5p-16 bs=64`. The difference
isn't scale — it's direction: `grad_sum=-1.0642` at step 0 (vs +0.184 in Exp
Q and good runs). That is a first-principles signal that the distributed
compute graph on `v5p-8 bs=32` is arithmetically different, not just
reshaped.

### Run

- `v5p-8 bs=32 pd=4` (us-east5-a, INTERACTIVE band, preemptible pool): https://wandb.ai/marin-community/dpo/runs/experiment_r_r64_v5p8_bs32_pd4_s10_ue5a-i1-423c65

Launch command context:
- parent CPU coordinator pinned via `iris job run --zone us-east5-a --cpu 1 --memory 3g`
- child `train_dpo` task: `v5p-8` preemptible slice in `us-east5-a`
- env: `EXPERIMENT_R_BS=32`, `EXPERIMENT_R_PD=4`, `REGIONS_OVERRIDE=us-east5`, `MARIN_DEBUG_LOG_BATCH_INDICES=1`, `MARIN_DEBUG_LOG_STEP_TRACE=1`, `MARIN_DEBUG_RUN_TAG=ue5a-i1`
- iris job id: `/ahmed/debug-r1-r64-v5p8-bs32-pd4-ue5a-i1`
- experiment script: `experiments/posttrain/per_stmt_dpo/experiment_r_v5p8_bs32_s10.py` (new, cloned from Exp Q)

### Training-relevant config

Identical to the Exp Q pd=4 run (see that section below) except:
- `trainer.train_batch_size = 32` (was 64)
- therefore `grad_accum = 32 / (pd * num_devices) = 32 / (4 * 4) = 2` (was 4)

All other knobs — LoRA `r=64/α=64/zero_init_b=True`, `AdapterBaseReferenceConfig`,
`lr=1e-6`, `β=0.1`, `seed=0`, `seq_len=4096`, 10 steps, `reference_eval_cache=disabled`,
`max_eval_batches=1`, `include_lm_validation=False` — match Exp Q exactly.

### CE kernel shape — reduced sharply, but **did not match** the good `v5p-16 pd=2` regime

Verified `DEBUGCE` line at step 0 from the worker `output.log`:

```
device_kind=TPU v5 x.shape=(16384, 4096) w.shape=(4096, 128256)
v_block_size=8192 b_block_size=1024 num_v_blocks=16 num_b_blocks=16
explicit_block_sizes=False
```

Side-by-side comparison:

| run                         | `x.shape`        | `b_block_size` | `num_b_blocks` |
|-----------------------------|------------------|----------------|----------------|
| Good Exp N `v5p-16 pd=2`    | (32768, 4096)   | 32768          | 1              |
| **Exp R `v5p-8 bs=32 pd=4` (this)** | **(16384, 4096)** | **1024** | **16** |
| Bad Exp Q `v5p-8 pd=8`      | (65536, 4096)   | 1024           | 64             |
| Bad Exp Q `v5p-8 pd=4`      | (65536, 4096)   | 1024           | 64             |

So Exp R did **not** produce a bit-identical local CE problem. Instead, it
overshot in the other direction: the `v5p-8` run saw **half** the good
`v5p-16` local CE rows and still remained in the bad basin. That is still
useful evidence: reducing per-chip CE workload and reducing `num_b_blocks`
from `64 -> 16` did not recover training.

### Full 10-step training trajectory (from `DEBUGJ TRACE` worker logs)

| step | loss     | grad_l2  | grad_sum  |
|------|----------|----------|-----------|
| 0    | 0.693147 | 2.661433 | -1.0642   |
| 1    | 0.693147 | 2.547172 | +1.4175   |
| 2    | 0.688840 | 2.428318 | +0.0763   |
| 3    | 0.682598 | 2.356858 | +0.5037   |
| 4    | 0.675323 | 2.494112 | -0.3843   |
| 5    | 0.672751 | 2.467276 | -0.7983   |
| 6    | 0.665934 | 2.535709 | +0.0361   |
| 7    | 0.668329 | 2.429227 | -1.4167   |
| 8    | 0.660116 | 2.483956 | +0.5612   |
| 9    | 0.662023 | 2.348258 | -2.1968   |

Qualitative pattern is the same bad regime as Exp Q:
- Loss stays near `ln(2)` across 10 steps (only drops ~0.03 total).
- `grad_l2` never decays toward ~1.12; stays in ~2.34-2.66 band.
- Step 1 loss is still exactly `ln(2)` — first update does essentially nothing.

### Side-by-side: Exp R vs Exp Q vs good runs

| step | Exp Q `bs=64 pd=4` | Exp R `bs=32 pd=4` (this) | Good `v5p-16 pd=2` (Exp N) |
|------|--------------------|---------------------------|----------------------------|
| 0    | 0.693147 | 0.693147 | 0.693147 |
| 1    | 0.693147 | 0.693147 | 0.693147 |
| 2    | 0.685125 | **0.688840** | 0.335202 |
| 3    | 0.682298 | **0.682598** | 0.325988 |
| 4    | 0.673723 | **0.675323** | 0.336246 |
| 5    | 0.668946 | **0.672751** | 0.316800 |
| 6    | 0.667573 | **0.665934** | 0.336998 |
| 7    | 0.662823 | **0.668329** | 0.324271 |
| 8    | 0.658715 | **0.660116** | 0.306144 |
| 9    | 0.660557 | **0.662023** | 0.317624 |

Exp R tracks Exp Q (max |Δ| ≈ 0.005), not the good run (|Δ| stays ~0.35).
The bad basin on `v5p-8` is not determined by local CE math.

### Step-0 gradient norms — first shift observed across v5p-8 runs

| run                               | step-0 `grad_l2` | step-0 `grad_sum` |
|-----------------------------------|------------------|-------------------|
| Exp N `v5p-16 pd=2`               | 2.4563 | +0.18380 |
| Exp P `v5p-16 pd=4`               | 2.4622 | (≈+0.18) |
| Exp Q `v5p-8 pd=8`                | 2.4562 | +0.18380 |
| Exp Q `v5p-8 pd=4`                | 2.4560 | +0.18380 |
| **Exp R `v5p-8 bs=32 pd=4` (this)** | **2.6614** | **-1.0642** |

Up to and including Exp Q, every fixed-recipe v5p-* run produced the same
step-0 LoRA gradient to ~4 sig figs — including `grad_sum`. Exp R is the
first to shift `grad_l2` (by about +8%) and `grad_sum` (from positive to
negative, ~6x magnitude change). That means the `bs=32` change on `v5p-8`
actually reaches the numerically-distinct compute graph we were trying to
reproduce — yet the post-init trajectory still tracks the bad basin, not
the good one. The bad basin is therefore attracting from a wider set of
initial gradient directions than we had evidence for before Exp R.

### Validation-set behavior

| eval split | pre-training | post-step-10 | Δ      |
|------------|--------------|--------------|--------|
| stmt_val   | 0.693        | 0.662        | -0.031 |
| full_val   | 0.693        | 0.697        | +0.004 |

Effectively identical to the Exp Q pd=4 validation behavior
(stmt_val 0.693→0.663, full_val 0.693→0.694). No meaningful learning on
the broader distribution.

### What Exp R rules out, supports, and leaves open

**Rules out (strengthened):**

- “We just need to reduce the per-chip token load on `v5p-8`.” Halving the
  global batch size did reduce the local CE workload below the good
  `v5p-16 pd=2` run's level (B=16384 vs good B=32768), and didn't fix it.

**Weakens but does not fully rule out from Exp R alone:**

- “The fused XLA CE batch-blocking / `num_b_blocks` regime on `v5p-8` is
  the load-bearing cause.” Exp R did **not** bit-match the good CE tiling;
  it ended at `num_b_blocks=16` while good is `num_b_blocks=1`. The clean
  rule-out of CE batch-blocking comes from Exp R2a (2026-04-16T22:50Z,
  end of this logbook), which forced `num_b_blocks=1` at the bad-baseline
  `B=65536` and still stayed stuck.

**Strongly supports:**

- The `v5p-8` pathology lives below the CE kernel, in the broader
  distributed execution graph: FSDP all-gather / reduce-scatter pattern,
  attention `kv_head` sharding, reference-network compiled graph, or
  remat/HLO scheduling interactions specific to the `-8` pod topology.
- Step-0 gradients on `v5p-8` are not uniquely determined by the local CE
  shape — something earlier in the forward (or the reference path) also
  differs when `train_batch_size` changes at fixed pod.

**Not yet proved:**

- Which specific distributed-regime knob carries the pathology on `v5p-8`.
  Candidates ordered by probable information density:
  1. attention head sharding — map `kv_head` axis explicitly and see if
     it removes the split (extends the `dpo-lora` branch's TP=4 fix).
  2. FSDP granularity — reduce or eliminate FSDP sharding on the small
     `-8` pod; see if the run recovers.
  3. reference-network path — swap to a `SeparateReferenceConfig` probe on
     `v5p-8 bs=32 pd=4` to check whether the adapter-base reference graph
     is the piece that diverges.
  4. collective algorithm — force a different XLA collective impl via
     `XLA_FLAGS` and observe.

### Operational notes

This Exp R run required **1 preemption** before the clean 10-step attempt:

- **i1 attempt A**: TPU worker `...-0356-0efb6e03-worker-0` took the task and
  then died with `Worker failed: heartbeat stale (>900s since last heartbeat)`
  before reaching a training step.
- **i1 attempt B**: reassigned to `...-0239-624c632f-worker-0` at ~04:27Z.
  `DEBUGCE` logged at 04:27:48Z. Step 0 completed at 04:29:20Z (~85s).
  Steps 1-9 completed by 04:52Z. Status: `state=running, preempt=1` (the
  parent is still finalizing / writing checkpoints at the time of this
  logbook entry but all 10 `DEBUGJ TRACE` step lines are present).

Operational improvements from the Exp Q debugging storm carried into this
run:
- `--memory 3g --cpu 1` on the parent so it lands on a proper CPU ondemand
  worker, not a TPU slice.
- `ram="250g"` for the `v5p-8 pd=4` child so it fits under the ~305 GB
  free-memory budget on co-tenanted preemptible v5p-8 workers.

us-east5-a remains the only region with actual v5p-8 capacity during this
session; `us-central1` still shows 0 slices on
`tpu_v5p-preemptible_8-us-central1-a`.

### Next experiment direction

Per the logbook's pre-registered Case 2 plan:

> if CE is already matched and the run is still bad, stop focusing on CE
> and pivot to topology / sharding / collective investigation.

The most concretely-grounded next probe is:

**Experiment S (proposed) — attention `kv_head` axis mapping probe on
`v5p-8`.**
- same data / LoRA recipe / `bs=32 pd=4` as Exp R so the result is directly
  comparable
- change only the attention `kv_head` axis → `model` axis mapping, to
  mirror the TP fix already present for v6e-8 in the `dpo-lora` branch
  (commit `0b228b3a5 "[dpo] Fix TP=4 attention sharding: map kv_head axis
  to model"`)
- if `v5p-8` shared the same hidden kv-head mapping bug, this should be the
  cleanest first-principles fix

If Exp S does not recover, fall back to a `SeparateReferenceConfig` probe
on `v5p-8 bs=32 pd=4` to isolate the reference-network graph.

---

## 2026-04-16T03:17Z: Experiment Q (pd=4) result — `v5p-8` pathology is **independent of `per_device_parallelism`**

### Strongest true conclusion

Experiment Q pd=4 on `v5p-8` produces a training trajectory that is
**point-for-point identical** (max |Δ| = 0.0024) to the Exp Q pd=8 run. Both
stay stuck near `ln(2)`, while the same LoRA recipe on `v5p-16` escapes
immediately at both pd=2 (Exp N) and pd=4 (Exp P).

This completes the core Exp Q sweep and eliminates `per_device_parallelism`
as even a secondary contributor on the `-8` pod. The remaining pathology is
entirely attributable to the **`v5p-8` pod shape / host topology / sharding
layout** — the only variable that differs between the bad `v5p-8` runs and
the good `v5p-16` runs.

Combined with the earlier experiments:
- **Exp N**: TPU family is not the cause (matched v5p-16 vs v6e-16 track closely)
- **Exp O**: `pd` / local-shape changes don't break full FT on v6e-16
- **Exp P**: `pd` / local-shape changes don't break LoRA on v5p-16
- **Exp Q pd=8**: v5p-8 reproduces the bad regime
- **Exp Q pd=4 (this run)**: v5p-8 stays bad even at lower pd — the failure
  tracks the `-8` pod shape broadly, not just the high-`pd` corner

### Run

- `v5p-8 pd=4` (us-east5-a, INTERACTIVE band, preemptible pool): https://wandb.ai/marin-community/dpo/runs/experiment_q_r64_v5p8_pd4_s10_ue5a-i4-d7d7e1

Launch command context:
- parent CPU coordinator pinned via `iris job run --zone us-east5-a --cpu 1 --memory 3g`
- child `train_dpo` task: `v5p-8` preemptible slice in `us-east5-a`
- env: `EXPERIMENT_Q_PD=4`, `REGIONS_OVERRIDE=us-east5`, `MARIN_DEBUG_LOG_BATCH_INDICES=1`, `MARIN_DEBUG_LOG_STEP_TRACE=1`, `MARIN_DEBUG_RUN_TAG=ue5a-i4`
- iris job id: `/ahmed/debug-q1-r64-v5p8-pd4-ue5a-i4`
- experiment script: `experiments/posttrain/per_stmt_dpo/experiment_q_v5p8_pd_s10.py`

### Training-relevant config

Identical to the Exp Q pd=8 run (see that section below) except:
- `trainer.per_device_parallelism = 4` (was 8)
- `trainer.per_device_eval_parallelism = 4` (was 8)
- `resources.ram = "250g"` (was `"400g"` — lowered to fit on v5p-8 workers
  whose available memory was only ~305 GB due to co-tenancy)
- therefore `grad_accum = train_batch_size / (pd * num_devices) = 64 / (4 * 4) = 4`
  (was `64 / (8 * 4) = 2` at pd=8)

Everything else — LoRA recipe, data, seed, beta, lr, reference config,
reference_eval_cache, max_eval_batches, num_train_steps — is identical.

### Full 10-step training trajectory (from `DEBUGJ TRACE` worker logs)

| step | loss     | grad_l2  |
|------|----------|----------|
| 0    | 0.693147 | 2.456003 |
| 1    | 0.693147 | 2.255161 |
| 2    | 0.685125 | 2.362390 |
| 3    | 0.682298 | 2.366755 |
| 4    | 0.673723 | 2.315102 |
| 5    | 0.668946 | 2.307744 |
| 6    | 0.667573 | 2.237251 |
| 7    | 0.662823 | 2.244449 |
| 8    | 0.658715 | 2.419641 |
| 9    | 0.660557 | 2.272835 |

Same qualitative pattern as pd=8: loss barely drifts from `ln(2)`, gradients
stay in the ~2.24-2.42 band instead of decaying to ~1.12 as in good runs.

### Side-by-side: Exp Q pd=8 vs Exp Q pd=4 on `v5p-8`

| step | `v5p-8 pd=8` | `v5p-8 pd=4` | |Δ|    |
|------|-------------|-------------|---------|
| 0    | 0.693147    | 0.693147    | 0.0000  |
| 1    | 0.693147    | 0.693147    | 0.0000  |
| 2    | 0.685757    | 0.685125    | 0.0006  |
| 3    | 0.682320    | 0.682298    | 0.0000  |
| 4    | 0.673036    | 0.673723    | 0.0007  |
| 5    | 0.669805    | 0.668946    | 0.0009  |
| 6    | 0.668832    | 0.667573    | 0.0013  |
| 7    | 0.665207    | 0.662823    | 0.0024  |
| 8    | 0.660748    | 0.658715    | 0.0020  |
| 9    | 0.662508    | 0.660557    | 0.0020  |

Max delta is 0.0024 at step 7. These are in the same regime to numerical
precision — the `v5p-8` failure is not sensitive to `pd` within the tested
range.

### Side-by-side: `v5p-8` (bad) vs `v5p-16` (good), all at same recipe

| step | `v5p-16 pd=2` (Exp N) | `v5p-16 pd=4` (Exp P) | `v5p-8 pd=8` (Exp Q) | `v5p-8 pd=4` (Exp Q, this) |
|------|-----------------------|-----------------------|----------------------|----------------------------|
| 0    | 0.693147 | 0.693147 | 0.693147 | 0.693147 |
| 1    | 0.693147 | 0.693147 | 0.693147 | 0.693147 |
| 2    | 0.335202 | 0.335283 | 0.685757 | 0.685125 |
| 3    | 0.325988 | 0.327747 | 0.682320 | 0.682298 |
| 4    | 0.336246 | 0.337701 | 0.673036 | 0.673723 |
| 5    | 0.316800 | 0.317172 | 0.669805 | 0.668946 |
| 6    | 0.336998 | 0.336385 | 0.668832 | 0.667573 |
| 7    | 0.324271 | 0.324177 | 0.665207 | 0.662823 |
| 8    | 0.306144 | 0.306423 | 0.660748 | 0.658715 |
| 9    | 0.317624 | 0.316186 | 0.662508 | 0.660557 |

The table tells the whole story: `v5p-16` escapes to ~0.31-0.34 by step 2,
`v5p-8` stays near ~0.66-0.69 regardless of `pd`. The split size is ~0.35
at every step — identical to the original disaster magnitude.

### Step-0 gradient norms (still match good runs)

| run | step-0 `grad_l2` |
|-----|------------------|
| Exp N `v5p-16 pd=2` | 2.4563 |
| Exp P `v5p-16 pd=4` | 2.4622 |
| Exp Q `v5p-8 pd=8`  | 2.4562 |
| **Exp Q `v5p-8 pd=4` (this)** | **2.4560** |

Step-0 init is fine across all four runs — the failure is post-init.

### Validation-set behavior

| eval split | pre-training | post-step-10 | Δ |
|---|---|---|---|
| stmt_val | 0.693 | 0.663 | -0.030 |
| full_val | 0.693 | 0.694 | +0.001 |

Effectively no learning on the broader distribution. Matches the pd=8
validation behavior (stmt_val 0.693→0.656, full_val 0.693→0.692).

### CE kernel shape observation

The `DEBUGCE` log line from this run shows:
```
x.shape=(65536, 4096) b_block_size=1024 num_b_blocks=64
```

This is **identical** to the pd=8 run's CE shape. At pd=4 with
`train_batch_size=64` and 4 devices, `grad_accum` doubles (from 2 to 4)
but the local CE batch dimension `B` per accumulation step stays the same.
So the local CE kernel math is unchanged between pd=8 and pd=4 on `v5p-8` —
consistent with the finding that `pd` doesn't matter here.

### Operational notes

This run required four submission attempts (`ue5a-i1` through `ue5a-i4`)
to get a clean 10-step completion:

- **i1**: first ue5a attempt during the pd=8 run window; preempted before
  reaching step 0 on its `train_dpo` child.
- **i2**: parent stuck in `assigned` state for 20+ min because
  `--memory 16g --enable-extra-resources` caused it to land on a TPU worker
  instead of a CPU node. Container never started.
- **i3**: parent fixed with `--memory 3g` (no `--enable-extra-resources`),
  landed on CPU `e2-highmem-2-ondemand`. But `train_dpo` child requested
  `ram="400g"` while workers only had ~305 GB free (co-tenancy). Scheduler
  message: `Insufficient memory (need 400.0GB, available 304.8GB)`. Child
  never scheduled.
- **i4**: script patched to `ram="250g"` for `pd <= 4`. `train_dpo` child
  placed immediately on `v5p-8` worker in us-east5-a, ran all 10 steps with
  zero preemptions.

For future v5p-8 runs at `pd <= 4`, use `ram="250g"` or lower.

Also: `us-central1` had **zero** v5p-8 capacity during this entire session
(autoscaler pool `tpu_v5p-preemptible_8-us-central1-a` showed 0 slices and
17 consecutive scale-up failures). All successful Exp Q runs came from
`us-east5-a`.

### What the full Exp Q sweep (pd=8 + pd=4) now establishes

The Exp Q sweep is now effectively complete for the primary question. The
optional `pd=2` data point would add one more row to the table, but the
scientific conclusion is already clear:

1. **`v5p-8` pod shape is the root cause** of the remaining LoRA DPO
   pathology, at fixed `r=64, α=64, zero_init_b=True`.
2. **`per_device_parallelism` is not a factor** — pd=8 and pd=4 produce
   identical bad trajectories on `v5p-8`.
3. **`v5p-16` is fine** at both pd=2 and pd=4, so the failure is specific
   to the `-8` pod, not the `v5p` hardware family.
4. **Init is fine** — step-0 gradients match to 4 sig figs across all runs;
   the failure is in how the optimizer updates affect the model post-init.

The remaining open question is **what specifically about the `-8` pod
shape** causes the failure — host topology, sharding layout, collective
communication pattern, or some interaction thereof. That is a deeper
investigation beyond the scope of the Exp Q sweep.

---

## 2026-04-16T00:50Z: Experiment Q (pd=8) result — `v5p-8` **REPRODUCES** the bad regime

### Strongest true conclusion

Experiment Q with `per_device_parallelism=8` on `v5p-8` **reproduces the
old catastrophic stuck-near-ln(2) regime** while holding the LoRA recipe
fixed at exactly the same `r=64, α=64, zero_init_b=True` configuration used
in the recent good Exp N / Exp O / Exp P runs.

In other words:
- `v5p-16` + same recipe + `pd=2`  → escapes ln(2) immediately (Exp N)
- `v5p-16` + same recipe + `pd=4`  → escapes ln(2) immediately (Exp P)
- `v6e-16` + same recipe + `pd=2`  → escapes ln(2) immediately (Exp N)
- **`v5p-8`  + same recipe + `pd=8`  → stays stuck near ln(2) (Exp Q, this run)**

And the `v5p-8 pd=8` trajectory tracks the **old bad `v5p-8 pd=16` Exp K
trajectory** almost point-for-point. So neither `pd`, nor TPU family, nor the
LoRA recipe itself is sufficient to flip the run — the **`v5p-8` pod shape /
host topology / sharding regime** is now the most credible remaining cause.

This is exactly the discriminating outcome the planned Exp Q sweep was
designed to produce, and it collapses the remaining hypothesis space.

### Run

- `v5p-8 pd=8` (us-east5-a, INTERACTIVE band, preemptible pool): https://wandb.ai/marin-community/dpo/runs/experiment_q_r64_v5p8_pd8_s10_ue5a-i1-38dd4c

Launch command context:
- parent CPU coordinator pinned via `iris job run --zone us-east5-a --cpu 1 --memory 16g`
- child `train_dpo` task: `v5p-8` preemptible slice in `us-east5-a`
- env: `EXPERIMENT_Q_PD=8`, `REGIONS_OVERRIDE=us-east5`, `MARIN_DEBUG_LOG_BATCH_INDICES=1`, `MARIN_DEBUG_LOG_STEP_TRACE=1`, `MARIN_DEBUG_RUN_TAG=ue5a-i1`
- iris job id: `/ahmed/debug-q1-r64-v5p8-pd8-ue5a-i1`
- experiment script: `experiments/posttrain/per_stmt_dpo/experiment_q_v5p8_pd_s10.py`

### Training-relevant config, all fixed to the recent good-run recipe

- `trainer.train_batch_size = 64`
- `trainer.per_device_parallelism = 8`
- `trainer.per_device_eval_parallelism = 8`
- `trainer.seed = 0`
- `data_seed = 0`
- `beta = 0.1`
- `lr = 1e-6`
- `lr_schedule = "cosine"`
- `warmup = 0.1`
- `train_seq_len = 4096`
- `max_seq_len = 4096`
- `adapter.r = 64`
- `adapter.alpha = 64`
- `adapter.zero_init_b = True`
- `reference = AdapterBaseReferenceConfig()`
- `reference_eval_cache.mode = "disabled"`
- `max_eval_batches = 1`
- `num_train_steps = 10`
- `include_lm_validation = False`
- data sources: same per-stmt `bloom_v2_singleton/support_mental_health` + speceval full-val that Exp N / O / P used

The only knobs that differ from the good Exp N `v5p-16 pd=2` baseline are
the TPU slice (`v5p-16 → v5p-8`) and `per_device_parallelism (2 → 8)` —
specifically chosen to match the planned Exp Q sweep entry point.

### Full 10-step training trajectory (from `DEBUGJ TRACE` worker logs)

All values are the per-step trace emitted by the training script under
`MARIN_DEBUG_LOG_STEP_TRACE=1`. `grad_l2` here is the LoRA-parameter-only
gradient l2 norm (equal to `gB_l2` at step 0 because `zero_init_b=True`).

| step | loss     | grad_l2  | grad_sum |
|------|----------|----------|----------|
| 0    | 0.693147 | 2.456232 |  0.1838  |
| 1    | 0.693147 | 2.255300 |  0.2208  |
| 2    | 0.685757 | 2.363574 | -0.6084  |
| 3    | 0.682320 | 2.365350 | -0.6799  |
| 4    | 0.673036 | 2.311212 | -0.8281  |
| 5    | 0.669805 | 2.312919 | -1.3175  |
| 6    | 0.668832 | 2.240088 | -0.1497  |
| 7    | 0.665207 | 2.249352 | -1.3066  |
| 8    | 0.660748 | 2.425360 | -0.1287  |
| 9    | 0.662508 | 2.276717 | -1.4925  |

Key qualitative observations from this trajectory:

- Step 0 is `ln(2) = 0.693147` as expected (softmax over two equally scored
  logits at initialization).
- Step 1 is **still exactly `ln(2)`**, i.e. the first update did essentially
  nothing to the DPO loss.
- The loss only drifts down by ~0.03 over 9 update steps (from `0.6931` to
  `0.6625`), an order of magnitude less descent than the good Exp N /
  Exp P runs achieve at the same step count.
- The LoRA gradient l2 norm **does not decay**: it stays in the ~2.24-2.45
  band across all 10 steps, whereas good runs drop their LoRA-only `grad_l2`
  to ~1.12 by step 9 (Exp N / P recorded values in that range).

### Direct comparison with the good matched-geometry runs

Train loss, step by step:

| step | Exp N `v5p-16 pd=2` | Exp P `v5p-16 pd=4` | **Exp Q `v5p-8 pd=8` (this run)** | Exp Q - Exp N |
|------|---------------------|---------------------|-----------------------------------|---------------|
| 0    | 0.693147 | 0.693147 | **0.693147** | 0.000000 |
| 1    | 0.693147 | 0.693147 | **0.693147** | 0.000000 |
| 2    | 0.335202 | 0.335283 | **0.685757** | +0.350555 |
| 3    | 0.325988 | 0.327747 | **0.682320** | +0.356332 |
| 4    | 0.336246 | 0.337701 | **0.673036** | +0.336790 |
| 5    | 0.316800 | 0.317172 | **0.669805** | +0.353005 |
| 6    | 0.336998 | 0.336385 | **0.668832** | +0.331834 |
| 7    | 0.324271 | 0.324177 | **0.665207** | +0.340936 |
| 8    | 0.306144 | 0.306423 | **0.660748** | +0.354604 |
| 9    | 0.317624 | 0.316186 | **0.662508** | +0.344884 |

This is not “a little worse” — it is the same catastrophic split size we saw
in the original `v5p-8` vs `v6e-8` disaster, relative to matched good runs
on the same fixed LoRA recipe.

### Direct comparison with the old bad `v5p-8 pd=16` run (Exp K)

The clean apples-to-apples older bad baseline is the Exp K
`r64_alpha64_s10_v5p8_k1p-6840ce` run, because it uses the **same** fixed
`r=64, α=64, zero_init_b=True` LoRA recipe on a `v5p-8`:

| step | old `v5p-8 pd=16` (Exp K) | new `v5p-8 pd=8` (Exp Q) | |Δ|       |
|------|---------------------------|--------------------------|-------------|
| 2    | 0.685376                  | 0.685757                 | 0.000381    |
| 3    | 0.679647                  | 0.682320                 | 0.002673    |
| 8    | 0.658281                  | 0.660748                 | 0.002467    |

The two `v5p-8` runs, at *different* `per_device_parallelism` values (16 vs 8),
trace essentially the same bad trajectory. Dropping `pd` from 16 to 8 on
`v5p-8` **does not fix it**. That is the key novel fact from this Exp Q data
point.

### Step-0 gradient scale matches good runs (init is fine)

Step-0 LoRA-only `grad_l2` values across all fixed-recipe runs:

| run                          | step-0 `grad_l2` |
|------------------------------|------------------|
| Exp N `v5p-16 pd=2`          | 2.4563 |
| Exp N `v6e-16 pd=2`          | 2.4661 |
| Exp P `v5p-16 pd=4`          | 2.4622 |
| **Exp Q `v5p-8 pd=8` (now)** | **2.4562** |

So init + first forward/backward produces a LoRA gradient of essentially the
same magnitude everywhere — agreement to ~4 significant figures with the
Exp N `v5p-16 pd=2` run. The bad behavior is not an initialization-time
problem. It only appears once we start taking optimizer steps.

However, while the good runs' `grad_l2` decays to ~`1.12` by step 9, the
`v5p-8 pd=8` run's `grad_l2` stays near ~`2.28` at step 9. The optimizer is
receiving a gradient of normal magnitude every step; it just isn't reducing
the loss.

### Validation-set behavior

From the single post-step-10 eval point:

| eval split | pre-training | post-step-10 | Δ      |
|------------|--------------|--------------|--------|
| stmt_val   | 0.693        | 0.656        | -0.037 |
| full_val   | 0.693        | 0.692        | -0.001 |

So the tiny nudge the run manages on train loss barely transfers to
stmt_val and effectively not at all to the broader full_val distribution.

Good Exp N runs, by contrast, move both stmt_val and full_val meaningfully
by step 10 (see Exp N history in W&B).

### What Exp Q (pd=8) rules out, and what it supports

**Rules out (now strengthened):**

- “The old catastrophe was really just about `per_device_parallelism` / local
  CE shape.” We now have `v5p-8 pd=8` behaving as badly as `v5p-8 pd=16`
  despite cutting `pd` in half.
- “The old catastrophe was really TPU family.” Already ruled out by Exp N;
  Exp Q adds that even at the originally-blamed family (`v5p`), the
  `-16` pod is fine and the `-8` pod is not, at matched recipe.
- “The LoRA recipe itself is unstable at `r=64, α=64, zero_init_b=True`.”
  This is the same recipe that trains normally on `v5p-16` and `v6e-16`, so
  the recipe is fine; the environment is not.

**Strongly supports:**

- The remaining pathology is specific to the **`v5p-8` pod shape / host
  topology / sharding layout**, in a way that is not explained by local CE
  tile math or by `per_device_parallelism` alone.
- The failure mode is a post-init stepping problem (updates do not reduce
  loss) rather than an init-time or forward-only problem.

**Not yet proved:**

- Whether `v5p-8` remains bad at `pd=4` and `pd=2`, or whether it recovers as
  `pd` drops further. The planned follow-up sweep entries (`pd=4`, optional
  `pd=2`) are now the most informative next data points.
- Whether `v5p-8` also fails in full fine-tuning, or whether this is a LoRA-
  specific amplifier of the `-8` pod pathology.

### Operational notes — preemption storm

Getting this single successful 10-step run required **3 preemptions** of the
`train_dpo` child task before an attempt survived compile-plus-10-steps. Key
timestamps, reconstructed from `iris job summary` and worker log timestamps:

- 22:42Z — `/ahmed/debug-q1-r64-v5p8-pd8-ue5a-i1` submitted (parent CPU +
  `train_dpo` TPU child).
- Parent CPU stayed pinned to `us-east5-a`; `train_dpo` child bounced across
  multiple `marin-tpu-v5p-preemptible-8-us-east5-a-*-worker-0` slices.
- ~23:28Z — first scheduler message transitioned from
  `tier_blocked by quota-pool tier monotonicity` to autoscaler-demand-routed
  for `tpu_v5p-preemptible_8-us-east5-a`, so the first worker came up.
- First + second worker attempts died with `Worker ... failed: Request timed
  out` before reaching a training step.
- ~00:26Z — third worker (`...-20260416-0026-0cde0ab3-worker-0`) attached.
- 00:44-00:45Z — step 0 completed (~100s, dominated by JIT compile), first
  `stmt_val` and `full_val` evals recorded at 0.693.
- 00:45Z - 00:50Z — steps 1 through 9 completed on this worker.

The parallel `uc1-i1` job in `us-central1` reached step 1 on one of its
attempts, then lost its worker again. Final `preempt` count observed on the
`uc1-i1` child task was **4** by the time Exp Q pd=8 completed on `ue5a-i1`,
and `uc1-i1` never got a clean 10-step run. The `ue5a-i1` run is therefore
the sole source of Exp Q pd=8 step-level data above.

Both jobs were submitted at `PRIORITY_BAND_INTERACTIVE` (the default for
`iris job run`), so on a `preemptible` pool they are kickable by:
- GCP spot reclamation of the preemptible VM itself (independent of Iris), and
- any Iris `PRIORITY_BAND_PRODUCTION` task competing for the same
  `tpu_v5p-preemptible_*` scale group.

During the Exp Q window, the only live `PRIORITY_BAND_PRODUCTION` v5p tasks
observed via `iris rpc controller get-scheduler-state` were from user
`moojink` on a `marin-tpu-v5p-preemptible-32-us-central1-*` slice (an
Iris-scheduled Qwen3-1.7B SFT job), which plausibly contributed to the
autoscaler reshuffling that kept evicting `ue5a-i1` and `uc1-i1` workers.
Moojink's production job was in `us-central1`, not `us-east5-a`, so the
`ue5a-i1` preemptions were most likely GCP spot reclamations rather than
cross-tenant Iris displacement.

Future-agent takeaway: for the remaining planned `v5p-8 pd=4` / `pd=2` Exp Q
sweep points, expect a similar preemption storm. The run script is
idempotent — each surviving attempt re-computes step 0 from the frozen
init, so a preempted attempt is cheap in correctness terms but expensive in
wall-clock (~7-10 min per boot-to-step-0 attempt). Just keep resubmitting or
keep the executor alive; do not switch bands to `production`.

### Next experiment

Per the original planned Exp Q sweep order, the next two entries are:

1. `v5p-8 pd=4` (same recipe, same data, same seed; only `pd: 8 → 4`).
   Expected to further isolate whether `v5p-8` stays pathological as `pd`
   drops.
2. Optional `v5p-8 pd=2`.

If `v5p-8 pd=4` also stays near `ln(2)`, the `v5p-8` pod shape is
confirmed as the remaining root-cause surface, independent of `pd`. If it
recovers, the failure narrows to the high-`pd` corner of `v5p-8`.

---

## 2026-04-15T07:28Z: Experiment N — per-stmt LoRA DPO on matched-family pods with matched local execution geometry

### Strongest true conclusion

Experiment N is the strongest evidence so far that the original catastrophic
`v5p`/`v6e` LoRA-DPO split was **not caused by TPU family by itself**.

When I rerun the same per-statement LoRA DPO recipe on:
- `v5p-16` with `per_device_parallelism=2`
- `v6e-16` with `per_device_parallelism=2`

the two runs track each other very closely from step 0 through step 9. The
old failure mode, where `v6e` escaped `ln(2)` immediately and `v5p` stayed
near `ln(2)`, disappears.

That means:
- TPU-family hardware differences are **not sufficient** to explain the old split
- execution geometry is the **primary driver** of the old split

What this experiment does **not** prove:
- It does **not** prove that LoRA has zero sensitivity to geometry
- It does **not** prove that every component of the old mismatch has been
  individually isolated

The strongest supportable version is:
> the original `v5p-8` / `v6e-8` disaster required the old geometry mismatch;
> once geometry is controlled, the TPU-family split disappears

### Runs

- `v5p-16 pd=2` (us-central1): https://wandb.ai/marin-community/dpo/runs/r64_matched_pd2_s10_v5p16_n1-7a55a1
- `v6e-16 pd=2` (europe-west4): https://wandb.ai/marin-community/dpo/runs/r64_matched_pd2_s10_v6e16_n3-323159
- `v6e-32 pd=1` (us-east1 fallback): https://wandb.ai/marin-community/dpo/runs/r64_matched_pd2_s10_v6e32_n3-8577d8

The `v6e-32 pd=1` run is a useful supporting point because it also lands in
the same learning regime, but the primary comparison for the original
hardware-family question is the matched `v5p-16 pd=2` vs `v6e-16 pd=2` pair.

### Training-relevant config, from script and W&B

Script: `experiments/posttrain/per_stmt_dpo/debug_r64_matched_pd2_s10.py`

Training knobs that match across the primary `v5p-16` and `v6e-16` pair:
- `trainer.train_batch_size = 64`
- `trainer.per_device_parallelism = 2`
- `trainer.per_device_eval_parallelism = 2`
- `trainer.seed = 0`
- `data_seed = 0`
- `beta = 0.1`
- `lr = 1e-6`
- `lr_schedule = "cosine"`
- `warmup = 0.1`
- `train_seq_len = 4096`
- `max_seq_len = 4096`
- `adapter.r = 64`
- `adapter.alpha = 64`
- `adapter.zero_init_b = true`
- `lora.exclude_modules_resolved = ["lm_head"]`
- `reference = AdapterBaseReferenceConfig()`
- `reference_eval_cache.mode = "disabled"`
- `max_eval_batches = 1`

Logical data definition also matches:
- same mirrored train source:
  `preference/bloom_v2_singleton/support_mental_health/train/shard-00000.jsonl.gz`
- same mirrored stmt-val source
- same mirrored full-val source
- same tokenizer / same tokenization names / same permutation type (`feistel`)

What differs across regions in W&B is the concrete tokenized `cache_dir`
prefix (`gs://marin-us-central1/...` vs `gs://marin-us-east1/...`), because the
parent executor resolves outputs in its local region. That is a storage-path
difference, not a logical dataset-definition difference.

One precision note from the W&B payloads:
- the `v5p-16` run still shows non-null `lm_validation_data`
- the `v6e-16` run shows `lm_validation_data = null`

Because `include_lm_validation=False` was passed at the experiment layer, and
because the training curves already match through steps 0-9 before the single
eval point at step 10, I do **not** treat that serialization discrepancy as
material to the Exp N training conclusion. But the earlier wording “identical
except TPU/pd” was too strong and is corrected here.

### What matched

#### 1. Train loss

From W&B history:

| step | v5p-16 pd=2 | v6e-16 pd=2 | |Δ| |
|---|---|---|---|
| 0 | 0.693147 | 0.693147 | 0.000000 |
| 1 | 0.693147 | 0.693147 | 0.000000 |
| 2 | 0.335202 | 0.336385 | 0.001183 |
| 3 | 0.325988 | 0.325428 | 0.000560 |
| 4 | 0.336246 | 0.336330 | 0.000084 |
| 5 | 0.316800 | 0.315582 | 0.001218 |
| 6 | 0.336998 | 0.333553 | 0.003445 |
| 7 | 0.324271 | 0.322128 | 0.002143 |
| 8 | 0.306144 | 0.302488 | 0.003656 |
| 9 | 0.317624 | 0.315265 | 0.002359 |

This is not “close only at step 0.” The two matched-family runs remain close
through the whole 10-step probe.

The fallback `v6e-32 pd=1` run also lands in the same regime:
- step 2: `0.331369`
- step 9: `0.316118`

That further weakens the story that TPU-family or chip-count alone causes the
old pathology.

#### 2. The actual DPO quantity: `delta_pi - delta_ref`

The DPO loss is driven by `delta_pi - delta_ref`, not by
`train/dpo_margin_policy` alone.

Using W&B `train/dpo_margin_policy - train/dpo_margin_ref`:

| step | v5p-16 pd=2 | v6e-16 pd=2 | |Δ| |
|---|---|---|---|
| 0 | 0.0000 | 0.0000 | 0.0000 |
| 1 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 9.4367 | 9.4102 | 0.0264 |
| 3 | 9.7769 | 9.8088 | 0.0319 |
| 4 | 9.4642 | 9.4929 | 0.0287 |
| 5 | 10.2081 | 10.2816 | 0.0735 |
| 6 | 9.4584 | 9.5975 | 0.1392 |
| 7 | 9.8192 | 9.9059 | 0.0866 |
| 8 | 10.5564 | 10.7182 | 0.1618 |
| 9 | 10.2183 | 10.2966 | 0.0782 |

So the matched runs are close on the **true optimization target**, not just
on a surface metric.

#### 3. Step-0 gradient scale

W&B step-0 `grad/norm/total` is also close:
- `v5p-16 pd=2`: `28.8643`
- `v6e-16 pd=2`: `28.9838`
- `v6e-32 pd=1`: `28.9350`

I directly re-read the `v6e-16` `DEBUGJ TRACE` worker log and confirmed its
per-step LoRA-only `grad_l2` trajectory:
- step 0: `2.4661`
- step 1: `2.2592`
- step 2: `1.2170`
- step 9: `1.1154`

For the `v5p-16` side, the strongest directly revalidated evidence in this
pass is the W&B train/loss and DPO-margin history above. The previous draft
used more aggressive “all three DEBUGJ TRACE tables match identically”
language than I can currently support from locally available worker logs, so I
am narrowing the claim accordingly.

### Direct contrast with the old mismatched-geometry pair

The clean apples-to-apples contrast is the earlier `r64/alpha64` pair from
Experiment K, because that used the **same LoRA recipe** but the old
geometry:

- old `v5p-8 pd=16`: https://wandb.ai/marin-community/dpo/runs/r64_alpha64_s10_v5p8_k1p-6840ce
- old `v6e-8 pd=4`: https://wandb.ai/marin-community/dpo/runs/r64_alpha64_s10_v6e8_k1r2-dd49b2

Loss contrast:

| step | old v5p-8 pd=16 | old v6e-8 pd=4 | |Δ| |
|---|---|---|---|
| 2 | 0.685376 | 0.333254 | 0.352123 |
| 3 | 0.679647 | 0.325355 | 0.354293 |
| 8 | 0.658281 | 0.301821 | 0.356460 |

True DPO quantity contrast (`delta_pi - delta_ref`):

| step | old v5p-8 pd=16 | old v6e-8 pd=4 | |Δ| |
|---|---|---|---|
| 2 | 0.1575 | 9.5113 | 9.3538 |
| 3 | 0.2731 | 9.8070 | 9.5339 |
| 8 | 0.7128 | 10.7681 | 10.0553 |

So Exp N does not just “improve the old result a bit.” It collapses a
massive, immediate split into a tiny residual difference.

For completeness, the first symptom we noticed originally was even worse in
the older `r=16, alpha=32` pair:
- old `v5p-8`: `0.6951` at step 2
- old `v6e-8`: `0.4672` at step 2

But that pair is not the cleanest contrast for Exp N because the LoRA recipe
also changed.

### First-principles explanation of CE batch blocking

This is the key code-path explanation for someone who knows JAX but not this
repo.

#### Where CE sits in the DPO path

DPO computes:
- policy log-prob for chosen
- policy log-prob for rejected
- reference log-prob for chosen
- reference log-prob for rejected

in `train_dpo.py` / `dpo.py`.

Each log-prob sum ultimately comes from
`LmHeadModel.compute_next_token_loss(...)`, which calls the fused next-token
cross-entropy path. In `models/loss.py`, that fused CE path flattens **all
non-embedding axes** into a single local batch axis called `__BATCH__` before
calling the kernel.

So the CE kernel does **not** see “number of sequences per device.” It sees a
local matrix:
- `x` with shape `(B, H)`
- `w` with shape `(H, V)`

where:
- `B` = flattened local token rows
- `H` = hidden size
- `V` = vocab size

For fixed-length 4096-token examples, `B` scales with:
- local examples per chip
- times sequence positions per example

That is why `per_device_parallelism` changes the CE kernel’s local problem
shape even if the global `train_batch_size` stays fixed.

#### What “batch blocking” means in this kernel

In `xla.py`, the streaming XLA CE path chooses:
- `v_block_size`: how many vocab columns to process at once
- `b_block_size`: how many flattened batch rows (`B`) to process at once

The backward then does:
1. loop over vocab blocks
2. inside each vocab block, loop over batch blocks
3. slice `x_block` of shape `(b_block_size, H)`
4. compute logits / probs / `delta` of shape `(b_block_size, v_block_size)`
5. accumulate partial `gx_block` and `gw_block_update`

Concretely, in `xla.py`:
- `num_b_blocks = b_dim // batch_block_size`
- the inner loop slices `x_block = dynamic_slice(x, (batch_start, 0), (batch_block_size, h_dim))`
- then forms `delta` and accumulates `gx_inner` / `gw_block`

So changing `B` or `b_block_size` changes:
- how many inner-loop iterations run
- the shapes of the temporary arrays in each iteration
- the order in which partial sums are accumulated
- the HBM footprint of those temporaries

This is what “CE batch blocking” means here.

#### What changed in the old bad pair

Experiment G directly logged the old CE shapes:
- old `v5p-8`: `x.shape=(65536, 4096)`, `b_block_size=1024`, `num_b_blocks=64`
- old `v6e-8`: `x.shape=(16384, 4096)`, `b_block_size=1024`, `num_b_blocks=16`

So even though both were “the same training job” at a high level, they were
executing materially different local CE kernels.

#### Why Exp N is different

In Exp N, the primary pair is:
- `v5p-16 pd=2`
- `v6e-16 pd=2`

Because both have:
- the same chip count
- the same `per_device_parallelism`
- the same sequence length

their local examples-per-chip match, so the flattened CE batch dimension `B`
is much closer by construction. I did **not** re-log the CE kernel in Exp N,
so I am not claiming exact `num_b_blocks` values here. But from the code path
above, matching chip count + matching `pd` on fixed-length data directly
equalizes the local CE problem shape in a way the old `v5p-8 pd=16` vs
`v6e-8 pd=4` pair did not.

This is the first-principles reason execution geometry matters so much in this
investigation.

### Important nuance: “same global microbatch” was not enough

Exp N should **not** be summarized as “just make global microbatch and
grad_accum match.”

We already had Experiment A:
- `v5p-8 pd=8`
- global microbatch = 32
- grad_accum = 2

and it was still bad.

So the real lesson is broader:
- matching global accumulation schedule alone was **not** enough
- matching the **local execution geometry** on matched-family pods was enough

That local geometry includes:
- per-device examples
- flattened CE `B`
- CE batch-block loop structure
- host/chip sharding layout
- probably attention / collective layout as well

### What Exp N rules out and what it supports

#### Ruled out

- TPU-family hardware differences as a **sufficient** explanation
- generic “LoRA DPO diverges across v5p vs v6e”

#### Strongly supported

- execution geometry is the **primary driver** of the original old split

#### Not proved

- that LoRA has zero extra sensitivity to geometry
- that every subcomponent of the old geometry mismatch has been individually isolated
- that full FT would have behaved the same under the exact old bad geometry

So I am retracting the earlier stronger claim that “LoRA amplifier is the
root cause.” The better-supported statement after Exp N is:

> the old `v5p-8` / `v6e-8` catastrophe was primarily an execution-geometry
> problem, and TPU family is not enough to reproduce it once geometry is
> controlled

### Operational note

Two infrastructure fixes were still required to get Exp N to run cleanly:

1. `experiments/paloma.py` and `experiments/evals/exp1600_uncheatable_evals.py`
   now use `mirrored()` for their raw sources, avoiding the executor’s
   cross-region read guard on parent-region CPU jobs.
2. `experiments/defaults.py:default_dpo` now accepts
   `include_lm_validation=False`, which let this short 10-step debug probe
   skip the Paloma / uncheatable LM-validation wiring and launch directly.

Those changes were necessary to run Exp N, but they are not part of the
scientific conclusion above.

### 2026-04-15T21:30Z — Experiment O result: within-family full-FT `pd` ablation on `v6e-16`

#### Why this experiment mattered

Exp N showed that the catastrophic LoRA split disappears when I match the
`v5p` and `v6e` runs on a cleaner local geometry. But that still left an
important causal question open:

> was the old catastrophe explained by `per_device_parallelism` / local kernel
> shape changes alone, or was LoRA unusually sensitive to the old mismatch?

The cleanest way to ask that is to keep hardware fixed and change only `pd`
inside full fine-tuning.

#### Hypothesis

**Hypothesis:** full FT on the per-statement setup is comparatively robust to
the `per_device_parallelism` / local-shape change that existed in the old LoRA
comparison.

Operationally:
- baseline: existing Exp L `v6e-16 pd=4`
- new run: `v6e-16 pd=2`

Keep fixed:
- same per-stmt `support_mental_health` data
- same `SeparateReferenceConfig`
- same `train_batch_size=64`
- same `lr=1e-6`
- same `beta=0.1`
- same `seed=0`
- same `train_seq_len=4096`
- same 10-step probe length

Change only:
- `trainer.per_device_parallelism: 4 -> 2`
- therefore local examples per device drop from `4 -> 2`
- therefore the flattened local CE batch dimension `B` seen by the fused
  next-token CE path drops from roughly `4 * 4096 = 16384` tokens per device to
  roughly `2 * 4096 = 8192`
- and because global `train_batch_size` stays 64, the `pd=2` run now uses
  `grad_accum=2` instead of a single microstep

This is the right first-principles quantity to watch because the fused XLA CE
kernel does not operate on “examples”; it operates on a local 2D tensor
`x.shape = (B, vocab_hidden)` after non-embed axes are flattened. Changing
`pd` changes the local `B`, which is the batch dimension the kernel uses when
choosing how to tile and stream the CE work. In other words: `pd` is not just a
trainer-level knob; it changes the local numerical problem the CE kernel sees.

Important precision note:
- Exp O definitely changes the local CE shape `B`
- Exp O may also change tuned CE batch blocking (`b_block_size`, `num_b_blocks`)
- but I did **not** log the resolved CE block sizes on this run, so this
  experiment rules out the broader “`pd` / local-shape change is sufficient”
  story in full FT, not every specific CE blocking detail in isolation

#### Runs

- Exp L baseline `v6e-16 pd=4`: https://wandb.ai/marin-community/dpo/runs/full_ft_s10_v6e16_l1-008fdb
- Exp O `v6e-16 pd=2`: https://wandb.ai/marin-community/dpo/runs/full_ft_pd2_s10_v6e16_o1ew4rgni4-9219b0

#### Result

The `pd=2` full-FT run closely tracks the earlier `pd=4` full-FT baseline. It
does **not** reproduce anything like the catastrophic LoRA divergence.

Per-step train loss:

| step | `v6e-16 pd=4` | `v6e-16 pd=2` | |Δ| |
|---|---|---|---|
| 0 | 0.693147 | 0.693163 | 0.000016 |
| 1 | 0.693147 | 0.693179 | 0.000032 |
| 2 | 0.692416 | 0.686007 | 0.006409 |
| 3 | 0.678566 | 0.673826 | 0.004740 |
| 4 | 0.664171 | 0.667567 | 0.003396 |
| 5 | 0.655864 | 0.655773 | 0.000091 |
| 6 | 0.618792 | 0.615281 | 0.003512 |
| 7 | 0.606406 | 0.601114 | 0.005291 |
| 8 | 0.597066 | 0.588456 | 0.008610 |

And on the actual DPO quantity `delta_pi - delta_ref`:

| step | `v6e-16 pd=4` | `v6e-16 pd=2` | |Δ| |
|---|---|---|---|
| 0 | 0.0000 | -0.0003 | 0.0003 |
| 1 | 0.0000 | -0.0006 | 0.0006 |
| 2 | 0.0192 | 0.1485 | 0.1293 |
| 3 | 0.2987 | 0.3952 | 0.0965 |
| 4 | 0.5930 | 0.5235 | 0.0695 |
| 5 | 0.7690 | 0.7690 | 0.0000 |
| 6 | 1.5594 | 1.6381 | 0.0787 |
| 7 | 1.8283 | 1.9463 | 0.1180 |
| 8 | 2.0425 | 2.2322 | 0.1897 |

Those are ordinary trajectory differences, not a regime change. In particular,
the `pd=2` run still:
- escapes `ln(2)` immediately
- follows the same qualitative descent as the `pd=4` baseline
- reaches the same full-FT learning regime by step 8

The raw `DEBUGJ TRACE` from the `pd=2` run shows normal full-FT dynamics:
- step 0: `loss=0.69316`, `grad_l2=28.8763`
- step 2: `loss=0.68601`, `grad_l2=27.7550`
- step 5: `loss=0.65577`, `grad_l2=26.7984`
- step 8: `loss=0.58846`, `grad_l2=25.8068`

#### What Exp O rules out

Exp O is strong evidence against the simple story:

> “changing `per_device_parallelism` and the local CE / kernel math is enough
> by itself to cause the old catastrophic divergence”

That story is too strong. Inside full FT on fixed `v6e-16` hardware:
- changing `pd` from `4 -> 2`
- changing local examples per device from `4 -> 2`
- changing the local CE batch shape `B`
- introducing `grad_accum=2`

does **not** cause a catastrophic training split.

#### Strongest true interpretation after Exp O

After Exp N + Exp O, the strongest supportable summary is:

- **TPU family alone is not the cause** of the old LoRA split
- **a pure `pd` / local-shape change is not sufficient** to cause a
  catastrophic split in full FT
- therefore the original `v5p-8 pd=16` vs `v6e-8 pd=4` failure required a
  stronger interaction than “kernel math changed”

What remains live is the interaction:
- old LoRA parameterization
- old broader execution-geometry mismatch
- possibly reference-graph differences

In other words, the old disaster was **not**:
- “v5p hardware breaks DPO”
- and **not** “changing CE local math alone breaks DPO”

It was some interaction specific to the original LoRA setup under the original
broader mismatch.

### 2026-04-15T21:55Z — Experiment P result: fixed-family LoRA `pd` ablation on `v5p-16`

#### Why this experiment mattered

After Exp N and Exp O, the remaining ambiguity was no longer "TPU family or
not." That was already narrowed down.

The live fork was:
- **`pd` / local execution geometry is sufficient to flip LoRA**
- **pod shape / sharding topology is required in addition to `pd`**

The cleanest way to ask that was to keep LoRA on, keep hardware fixed to
`v5p-16`, and change only `per_device_parallelism`.

#### Hypothesis

**Hypothesis:** LoRA on fixed `v5p-16` hardware is materially more sensitive to
the `pd` / local-shape change than full FT was in Exp O.

Operationally:
- baseline: existing good Exp N `v5p-16 pd=2`
- new run: `v5p-16 pd=4`

Keep fixed:
- same per-stmt `support_mental_health` data
- same LoRA recipe `r=64`, `alpha=64`, `zero_init_b=True`
- same `AdapterBaseReferenceConfig`
- same `train_batch_size=64`
- same `lr=1e-6`
- same `beta=0.1`
- same `seed=0`
- same 10-step probe length

Change only:
- `trainer.per_device_parallelism: 2 -> 4`
- local examples per device: `2 -> 4`
- local flattened CE batch dimension `B`: approximately `8192 -> 16384`
- grad accumulation: `2 -> 1`

#### Runs

- Exp N baseline `v5p-16 pd=2`: https://wandb.ai/marin-community/dpo/runs/r64_matched_pd2_s10_v5p16_n1-7a55a1
- Exp P `v5p-16 pd=4`: https://wandb.ai/marin-community/dpo/runs/r64_v5p16_pd4_s10_p1uc1i1-6692c0

#### Result

`v5p-16 pd=4` closely matches the existing good `v5p-16 pd=2` LoRA run. It
does **not** fall back into the old bad `v5p-8` regime.

Per-step train loss:

| step | `v5p-16 pd=2` | `v5p-16 pd=4` | |Δ| |
|---|---|---|---|
| 0 | 0.693147 | 0.693147 | 0.000000 |
| 1 | 0.693147 | 0.693147 | 0.000000 |
| 2 | 0.335202 | 0.335283 | 0.000081 |
| 3 | 0.325988 | 0.327747 | 0.001759 |
| 4 | 0.336246 | 0.337701 | 0.001455 |
| 5 | 0.316800 | 0.317172 | 0.000372 |
| 6 | 0.336998 | 0.336385 | 0.000613 |
| 7 | 0.324271 | 0.324177 | 0.000094 |
| 8 | 0.306144 | 0.306423 | 0.000280 |
| 9 | 0.317624 | 0.316186 | 0.001438 |

The new `pd=4` run is also in the same regime as the good matched-family
`v6e-16 pd=2` run from Exp N:
- `v6e-16 pd=2`: https://wandb.ai/marin-community/dpo/runs/r64_matched_pd2_s10_v6e16_n3-323159

Step-0 / step-9 LoRA-only gradient norms from `DEBUGJ TRACE`:
- `v5p-16 pd=2`: `2.4563 -> 1.1211`
- `v5p-16 pd=4`: `2.4622 -> 1.1188`
- `v6e-16 pd=2`: `2.4661 -> 1.1154`

So this is not just "loss looks vaguely similar." The trainable LoRA gradient
scale is also in the same regime.

#### What Exp P rules out

Exp P is strong evidence against the simple story:

> “LoRA plus a `pd` change / local CE-shape change is enough by itself to
> recreate the old catastrophe”

That story is now too strong. On fixed `v5p-16` hardware, changing
`pd: 2 -> 4`:
- changes local examples per device
- changes local flattened CE batch shape `B`
- changes whether the run uses grad accumulation

and still does **not** create the old bad regime.

#### Strongest true interpretation after Exp P

After Exp N + Exp O + Exp P, the strongest supportable summary is:

- **TPU family alone is not the cause**
- **a pure within-family `pd` / local-shape change is not sufficient** in
  either full FT or LoRA on `v5p-16`
- therefore the original `v5p-8` failures require something more specific to
  the old `-8` setup than just `pd` and local CE shape

The remaining live explanation is now much narrower:
- `v5p-8` pod shape / chip count / host topology / sharding pattern
- or some more extreme `v5p-8` geometry regime (`pd=8` / `pd=16`) that does
  not generalize to `v5p-16`

### Highest-info next experiment after Exp P

#### Reasoning

After Exp N + Exp O + Exp P, the investigation is much narrower than it was
originally:

- `v5p` vs `v6e` hardware family is not enough to explain the split
- `v5p-16` LoRA is robust at fixed `r=64`, `alpha=64` under both `pd=2` and
  `pd=4`
- within-family `pd` / local CE-shape changes are therefore not sufficient, by
  themselves, to recreate the old bad regime on `v5p-16`

So the remaining live question is no longer "does LoRA dislike `pd` in
general?" It is:

> is there something specifically pathological about the **`v5p-8` regime** at
> fixed `r=64`, `alpha=64`?

That is the next clean discriminator.

#### Planned Experiment Q — fixed-`r64/α64` LoRA sweep on `v5p-8`

> **Status (2026-04-16T03:17Z):** the Exp Q sweep is effectively complete.
> Both `v5p-8 pd=8` (2026-04-16T00:50Z) and `v5p-8 pd=4` (2026-04-16T03:17Z)
> reproduce the bad regime with near-identical trajectories (max |Δ| = 0.0024).
> See the two Experiment Q result sections at the top of this logbook for
> full traces, side-by-side comparisons, W&B links, and operational logs.
> The optional `pd=2` entry is deprioritized — the conclusion is already
> clear: the pathology tracks the `v5p-8` pod shape, not `pd`.

**Goal:** hold the LoRA recipe fixed at the same settings used in the recent
good `v5p-16` / `v6e-16` runs, and sweep `v5p-8` across progressively smaller
`per_device_parallelism` values.

Keep fixed across the sweep:
- per-stmt `support_mental_health` singleton data
- `LoraAdaptationConfig(r=64, alpha=64, zero_init_b=True, target_modules=None)`
- `AdapterBaseReferenceConfig`
- `train_batch_size=64`
- `lr=1e-6`
- `beta=0.1`
- `seed=0`
- `num_train_steps=10`
- `MARIN_DEBUG_LOG_BATCH_INDICES=1`
- `MARIN_DEBUG_LOG_STEP_TRACE=1`

Sweep order:
1. `v5p-8` with fixed-`r64/α64` at `pd=8`
2. `v5p-8` with fixed-`r64/α64` at `pd=4`
3. Optional follow-up: `v5p-8` with fixed-`r64/α64` at `pd=2`

Why this order:
- we already know `v5p-16` is robust under `pd=2` and `pd=4`
- we already have one bad `v5p-8` run at fixed `r=64`, `alpha=64`
- the most important remaining cleanup is to determine whether `v5p-8` stays
  bad as `pd` is reduced, or whether the failure is limited to the higher-`pd`
  end of the `v5p-8` regime

#### Hypothesis

The strongest current hypothesis is:

- `v5p-16` is robust under the recent fixed-`r64/α64` LoRA probes
- the remaining pathology is specific to `v5p-8`
- the `v5p-8` sweep will tell us whether the failure tracks the `-8` pod shape
  broadly, or only the higher-`pd` corner of that regime

Interpretation:
- if `v5p-8` remains bad across the sweep, then the leading explanation is that
  something about the `-8` pod shape / host topology / sharding regime is the
  real remaining culprit
- if `v5p-8` improves as `pd` drops, then the surviving issue is the
  high-`pd` corner of the `v5p-8` regime rather than `v5p-8` broadly

#### Historical `v5p-8` references for future agents

These are the prior `v5p-8` LoRA DPO runs that motivated the current sweep.
Use the W&B config for the exact `pd` / LoRA settings if needed:

- original fresh bad `v5p-8` run: https://wandb.ai/marin-community/dpo/runs/smh_lr1em06_s70_v5p8-964129
- forced-geometry `v5p-8` follow-up: https://wandb.ai/marin-community/dpo/runs/smh_lr1em06_s70_v5p8_pd8-0498ec
- fixed-`r64/α64` bad `v5p-8` run: https://wandb.ai/marin-community/dpo/runs/r64_alpha64_s10_v5p8_k1p-6840ce

---

## 2026-04-14: Experiment K — r=64/α=64 LoRA on per-stmt data — **does not remove the early split**

Goal: test whether the checked-in Marin LoRA recipe (`r=64, α=64` — matches
`experiments/tune_lora/README.md`) removes the per-stmt v5p/v6e split seen
with the earlier recipe (`r=16, α=32`). All other knobs
(data=support_mental_health singleton, lr=1e-6, β=0.1, seed=0,
zero_init_b=True, target_modules=None) were held constant.

Important caveat: this is **not** a pure rank-only test. LoRA applies scale
`α/r`, so the old recipe had scale `32/16 = 2` while the new recipe has
scale `64/64 = 1`. Experiment K therefore tests the checked-in `r64/α64`
recipe, not "rank-64 with everything else identical."

**Result: v5p and v6e still split on the DPO objective in the first 8-10
steps.** The checked-in `r64/α64` recipe is not an immediate fix.

**W&B runs (paired, r=64 α=64, per-stmt, 10 steps):**
- v5p-8 (us-central1): https://wandb.ai/marin-community/dpo/runs/r64_alpha64_s10_v5p8_k1p-6840ce
- v6e-8 (europe-west4): https://wandb.ai/marin-community/dpo/runs/r64_alpha64_s10_v6e8_k1r2-dd49b2

### Codex analysis — `train/dpo_margin_policy` is not the loss, and the
loss-driving quantity is still far apart

Re-reading `lib/levanter/src/levanter/dpo.py:119,123`, the DPO loss is

    softplus(−β · (δ_π − δ_ref))

with `δ_π = logp_π(chosen) − logp_π(rejected)` and `δ_ref` likewise for the
reference. The logged `train/dpo_margin_policy` is **only `mean(δ_π)`** —
not the loss-driving quantity. The loss depends on `δ_π − δ_ref`.

Pulling `δ_π`, `δ_ref` from W&B on both runs shows the two TPUs are already
producing very different DPO signals by step 2-3:

| step | run | loss | `δ_π` | `δ_ref` | `δ_π−δ_ref` | `β·(δ_π−δ_ref)` |
|---|---|---|---|---|---|---|
| 3 | v5p | 0.6796 | −119.2961 | −119.5692 | **+0.273** | 0.027 |
| 3 | v6e | 0.3254 | −109.7875 | −119.5945 | **+9.807** | 0.981 |
| 8 | v5p | 0.6583 | — | — | **+0.713** | 0.071 |
| 8 | v6e | 0.3018 | — | — | **+10.768** | 1.077 |

Reward metrics tell the same story at step 3:
- v5p: chosen reward `+0.0198`, rejected reward `−0.0075`
- v6e: chosen reward `+0.6852`, rejected reward `−0.2955`

`v6e` is moving the policy much further from the reference than `v5p` by
step 3. This is not just a plotting artifact from looking at
`train/dpo_margin_policy` in isolation; it is visible in the actual
loss-driving quantity `δ_π−δ_ref`.

Revised interpretation: the checked-in `r64/α64` recipe does not fix the
early per-stmt LoRA-DPO split. This keeps the culprit in the
per-stmt-LoRA regime, but does **not** isolate whether the dominant factor is
LoRA rank, LoRA scale, singleton-data sensitivity, or some numerical
property of the early DPO update.

### Next experiments to fill Codex's 2×2 isolation matrix

|   | per-stmt (singleton) | full bloom_v2 |
|---|---|---|
| **LoRA** | ❌ diverges early (`r16/α32` and `r64/α64`) | ❓ **Exp M** |
| **Full FT** | ❓ **Exp L** | ✅ closer match (full bloom_speceval_v2) |

---

#### Experiment L — per-stmt full DPO, paired

**Hypothesis:** removing LoRA while keeping the tiny singleton dataset tells
us whether the v5p↔v6e divergence depends on LoRA specifically or on the
singleton-dataset shape (one repeated concept, very homogeneous
chosen/rejected distribution). L-matches → LoRA is the amplifier. L-splits
→ singleton dataset drives the pathology on its own.

- Data: `preference/bloom_v2_singleton/support_mental_health/` (same as the
  pathological LoRA exp 1a `smh_lr1em06_s35`)
- Model: `marin-8b-instruct` policy + `marin-8b-instruct` `SeparateReferenceConfig`
  (full fine-tune; no adapter)
- Training: **first pass = `num_train_steps=10`**, `train_batch_size=64`,
  `lr=1e-6`, `lr_schedule=cosine`, `warmup=0.1`, `beta=0.1`, `seed=0`,
  `train_seq_len=4096`, `validation_split_fraction=None`
- Eval: once at step 10 on `stmt_val` + `full_val`
- Debug env: `MARIN_DEBUG_LOG_BATCH_INDICES=1`, `MARIN_DEBUG_LOG_STEP_TRACE=1`
- Resources: full-FT 8B (≈128 GB static / N-chip FSDP) doesn't fit on
  v5p-8 / v6e-8 for paired compare at batch 64. Preferred plan is to run
  the smallest slices that keep geometry close — ideally **v5p-16 pd=4**
  and **v6e-16 pd=4**. Only escalate to **v6e-32** if v6e-16 does not fit,
  and record that as a weaker comparison because chip count then differs.
- Follow-up: if the 10-step probe is ambiguous, extend the same pair to
  35 steps. Script: `experiments/posttrain/per_stmt_dpo/debug_full_ft_s10.py`
  (TBD; extendable to s35).

---

#### Experiment M — full-data LoRA DPO, paired v5p-8 vs v6e-8

**Hypothesis:** keeping LoRA while swapping to the full 109k-example
`bloom_speceval_v2` preference dataset tells us whether the v5p↔v6e
divergence depends on data variety. M-matches → singleton dataset drives
the pathology. M-splits → LoRA-DPO regime itself is fragile regardless of
dataset.

- Data: `preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/`
  (same as Marin's full-dataset DPO and the `tune_lora` runs)
- Model: `marin-8b-instruct` policy with LoRA
  `r=64, α=64, zero_init_b=True, target_modules=None`,
  `AdapterBaseReferenceConfig`
- Training: `num_train_steps=10` (split was visible by step 2 on per-stmt),
  `train_batch_size=64`, `lr=1e-6`, `lr_schedule=cosine`, `warmup=0.1`,
  `beta=0.1`, `seed=0`, `train_seq_len=4096`, `validation_split_fraction=None`
- Eval: once at step 10 on the bloom v2 val set
- Debug env: `MARIN_DEBUG_LOG_BATCH_INDICES=1`,
  `MARIN_DEBUG_LOG_STEP_TRACE=1`
- Resources: LoRA keeps HBM footprint small. `pd=4` on v6e-8 (same as
  existing LoRA per-stmt configs), `pd=-1` (auto) on v5p-8.
- Implementation note: reuse the existing `experiments/tune_lora/common.py`
  codepath rather than cloning from `per_stmt_dpo`, so this stays aligned
  with the checked-in full-data LoRA recipe.
- Script: `experiments/tune_lora/debug_full_data_lora_r64_s10.py` (TBD).

---

#### Decision table for L + M outcomes

| Exp L (per-stmt full-FT) | Exp M (full-data LoRA) | Interpretation |
|---|---|---|
| matches | matches | LoRA × singleton interaction is the specific pathology |
| matches | splits | LoRA itself is the amplifier regardless of dataset |
| splits | matches | singleton dataset is the amplifier regardless of model class |
| splits | splits | something fundamental about early DPO updates on 8B is v5p/v6e sensitive |

Expected: **L matches, M splits**. That would make LoRA-DPO the leading
amplifier. It would strongly de-prioritize generic CE/attention kernel
hunting, though not mathematically rule out numerical effects inside the
LoRA-DPO update path.

---

## 2026-04-14: Full DPO on v6e-32 — divergence does NOT reproduce

**Key finding:** When running **full (non-LoRA) DPO** with batch=64 on v6e-32
and comparing against the v5p-16 seed=0 full-DPO baseline, the loss curves
track each other almost perfectly through at least step 144.

**W&B runs (compare side-by-side):**
- v6e-32 (this run): https://wandb.ai/marin-community/dpo/runs/bloom_speceval_v2_v6e32_pd2_lr5e-07-7f1c19
- v5p-16 seed0 (baseline): https://wandb.ai/marin-community/dpo/runs/bloom_speceval_v2_beta0.1_seed0_b64_v5p16-68f963
- v5p-16 seed1 (sibling): https://wandb.ai/marin-community/dpo/runs/bloom_speceval_v2_beta0.1_seed1_b64_v5p16-c50842
- v5p-16 seed2 (sibling): https://wandb.ai/marin-community/dpo/runs/bloom_speceval_v2_beta0.1_seed2_b64_v5p16-2272c8

| Step | v5p-16 seed0 | v6e-32 (r1) | Δ |
|---|---|---|---|
| ~144 | 0.3937 | 0.39112 | 0.003 (< 1%) |
| ~314 (last observed) | — | 0.0513 | — |

**Interpretation (initial, confounded):** The hardware-level bf16 numerical
divergence that wrecks LoRA-DPO (10× learning-speed gap) **does not visibly
affect full fine-tune** with the same data/seed.

### ⚠️ Codex caveat: three confounds stacked in the "LoRA bad vs full good" comparison

The two run families we've compared so far are **not** a clean model-class ablation:

| Run family | Data | Model | LoRA `r` | α |
|---|---|---|---|---|
| Per-stmt DPO (pathological v5p vs v6e) | 1 singleton stmt (~6k ex) | LoRA | **16** | 32 |
| `tune_lora` full-data LoRA | bloom v2 full (~109k ex) | LoRA | **64** | 64 |
| v6e-32 full DPO (this run) | bloom v2 full (~109k ex) | full FT | n/a | n/a |

Differences: (1) LoRA-vs-full-FT, (2) singleton-vs-full dataset, (3) rank 16 vs 64.
Any of these — or combinations — could explain why the v5p↔v6e split
appears only in the per-stmt LoRA runs.

**Mechanism hypothesis (why LoRA-DPO is more fragile than full-FT DPO):**
- DPO at init is on a flat landscape (policy=ref → `softplus(0) = ln(2)`,
  so `dL/dlogit = 0.5` for every example).
- In LoRA with `zero_init_b=True`, only `lora_B` absorbs gradient; `lora_A`
  is frozen at init-random. `lm_head` is excluded from LoRA by default.
- Full FT can spread the DPO signal across all 8B params; LoRA has to route
  it through a tiny low-rank branch with a zero-initialized B, fixed random A,
  and no `lm_head` adaptation.
- So small bf16 differences in `delta_pi - delta_ref` get projected by
  a low-rank basis into very different `lora_B` updates — amplifying per-chip
  numerical noise into direction-level learning-dynamics differences.

### Codex-suggested isolation matrix

|   | per-stmt (singleton) | full bloom_v2 |
|---|---|---|
| **LoRA** | ❌ diverges (r=16) | ❓ **missing** |
| **Full FT** | ❓ **missing** | ✅ matches |

The highest-info next run is **full bloom_v2 LoRA DPO on v5p vs v6e**
(same rank as pathological case, same 2-step trace instrumentation).
If it still splits → pathology is LoRA-regime specific. If it matches →
singleton dataset is a major driver.

**Config that works on v6e-32** (`experiments/posttrain/full_dpo/v6e32_full_dpo.py`):
- `train_batch_size=64`, `per_device_parallelism=2` (32 chips × 2 ex/chip)
- `SeparateReferenceConfig`, `reference_model_path="marin-community/marin-8b-instruct"`
- No adapter → full fine-tune of all 8B params
- `beta=0.1`, `lr=5e-7`, cosine schedule, `warmup=0.1`, `num_epochs=1.0`
- `seed=0`, `train_seq_len=4096`
- **Throughput: 6.2–7.7 sec/step** post-warmup (JIT compile warm-up ~65s for step 0)
- **HBM OK** — no OOM on v6e-32 (peak host RAM ~37 GB/worker, ~22 GB steady)

**Napkin math per chip** at batch=64 / FSDP over 32 chips:
- policy bf16 16 GB / 32 = 0.5 GB; reference bf16 0.5 GB
- grads fp32 1 GB; Adam m/v fp32 2 GB; DCN replication ~8 GB
- activations w/ gc ~10 GB/chip at pd=2 → ~22 GB total (fits in 32 GB HBM)

**Preemption incident (first run `/ahmed/full-dpo-v6e32-r1`):**
- Child ran 8h 54m, reached step 947 before parent CPU coordinator was
  preempted twice. Re-scheduled parent landed on `us-central1` CPU while
  original wrote checkpoints to `gs://marin-us-central2/.../-7f1c19/`.
  Executor re-resolved output to us-central1 (empty), child respawn then failed.
- **Salvageable on GCS:** `step-947` streaming checkpoint + `step-500` HF export,
  both under `gs://marin-us-central2/checkpoints/dpo/full_dpo/bloom_speceval_v2_v6e32_pd2_lr5e-07-7f1c19/`.
- **Lesson:** long-running DPO parent CPU jobs need region pinning
  (`--zone us-central2-b` or `-e MARIN_PREFIX gs://marin-us-central2`) so that
  preemption-driven reschedules don't orphan checkpoints across regions.

---

## Problem

DPO LoRA training runs with matching hyperparameters produce dramatically
different learning curves on v6e-8 vs v5p-8. v6e-8 learns much faster.
The effect is consistent across all revalidated exp 1a pairs and directionally
present in exps 1b and 2a (though those comparisons have caveats — see below).

## Current Leading Hypothesis (refined after Experiment J)

**Hardware-level numerical divergence in the forward/backward pass.** On
identical inputs and identical initial model state, v5p-8 and v6e-8 produce
measurably different trainable gradients at step 0. The bf16 matmul /
attention numerics differ between the two chip architectures.

### Key facts established by Experiment J (2-step trace):

- **Same batch indices at step 0** (both TPUs: sha256=`7a61ce53d17eb721`, indices [0-63])
- **Same initial param values** (pA_l2 differs in 6th decimal: 59.075542 vs 59.075538)
- **Same loss at step 0** (exactly `0.6931471824645996` on both — but this is
  tautological: LoRA zero-init means policy=reference, so every example has
  `softplus(0) = ln(2)` regardless of data)
- **Different gradients at step 0**:

| Metric | v6e-8 | v5p-8 | Δ |
|---|---|---|---|
| grad_l2 | 2.5308258 | 2.5316045 | 0.00031% |
| grad_sum | -0.8846889 | **-0.7291931** | **0.155 absolute** |
| upd_l2 | 0.004686407 | 0.004685373 | ~match |
| upd_sum | -0.001111864 | -0.0000680 | huge relative |

Sentinel lora_B grad_sum on gate_proj: v6e=0.269, v5p=0.353 (+31%).

**Interpretation:** Gradients have the same L2 magnitude (within 0.0003%)
but **differ significantly in direction** (signed sums diverge). The
forward pass computes `delta_pi - delta_ref` slightly differently on each
chip. At init (delta=0), `d(softplus(-x))/dx = -sigmoid(-x) = -0.5`,
so tiny per-example `delta` differences directly produce tiny gradient-
direction differences. These compound: DPO at initialization sits on a
flat loss landscape where small gradient-direction shifts land the model
in different basins.

### Earlier hypotheses ruled out by Experiment J:

- **Microbatch code path** (ruled out by Experiment A earlier)
- **CE v_block_size heuristic** (ruled out by Experiment G: same on both)
- **Data ordering** (ruled out by Experiment J: identical batch sha256)
- **CE `num_b_blocks` blocking** (Experiment J also rules this out as the
  PRIMARY cause — gradients differ globally, not just in the lm_head gw which
  is frozen for LoRA anyway)

## Shared Config (verified identical in W&B)

- lr=1e-6, cosine schedule, warmup=0.1
- LoRA r=16, alpha=32, dropout=0, zero_init_b=True
- beta=0.1, seed=0, data_seed=0
- max_grad_norm=1.0, train_batch_size=64
- AdapterBaseReferenceConfig (reference = base model with LoRA disabled)
- model: marin-community/marin-8b-instruct

## Execution Differences (verified at runtime via Experiment G)

| | v6e-8 | v5p-8 |
|---|---|---|
| Chips | 8 | 4 |
| `per_device_parallelism` | 4 (explicit) | -1 → 16 (auto) |
| Gradient accumulation | 2 micro-steps of 32 | none (full batch of 64) |
| Per-device examples | 4 | 16 |
| CE kernel B (examples × seq_len) | **16,384** | **65,536** |
| CE kernel `device_kind` | `TPU v6 lite` | `TPU v5` |
| CE `v_block_size` | **8192** | **8192** (same!) |
| CE `b_block_size` | **1024** | **1024** (same!) |
| CE `num_v_blocks` (padded 128256) | **16** | **16** (same!) |
| **CE `num_b_blocks`** | **16** | **64** |

---

## Evidence

### Primary: Fresh s70 Pairs (cleanest comparison)

**Established.** Both start fresh (no checkpoint resume).

**Train loss: lr=1e-6, s70 pair**

| Step | v6e-8 (`fbac2a`) | v5p-8 (`964129`) | Gap |
|---|---|---|---|
| 0 | 0.6931 | 0.6931 | 0.000 |
| 1 | 0.6931 | 0.6931 | 0.000 |
| 2 | **0.4672** | 0.6951 | -0.228 |
| 5 | 0.3861 | 0.6883 | -0.302 |
| 10 | 0.3688 | 0.6710 | -0.302 |
| ~69 | 0.320 | 0.574 | -0.254 |

**Final eval:** v6e-8 stmt=0.402 full=0.613 | v5p-8 stmt=0.601 full=0.675

**Train loss: lr=1e-7, s70 pair**

| Step | v6e-8 (`4ac830`) | v5p-8 (`f3d60b`) | Gap |
|---|---|---|---|
| 0 | 0.6931 | 0.6931 | 0.000 |
| 2 | 0.6806 | 0.6943 | -0.014 |
| 5 | 0.4606 | 0.6932 | -0.233 |
| 10 | 0.3862 | 0.6917 | -0.306 |

The divergence appears at step 2-3 on fresh runs across multiple LR settings.

**Additional matched pairs (summary-level):**

| Pair | v5p train - v6e train | v5p stmt - v6e stmt | v5p full - v6e full |
|---|---|---|---|
| lr5e7_s70 | +0.293 | +0.224 | +0.065 |
| lr5e7_s140 | +0.240 | +0.196 | +0.061 |
| lr1e6_s70 | +0.253 | +0.199 | +0.062 |

### Secondary: s140 Pair (has resume confound)

v5p-8 s140 (`cc7957`) resumed from step 94. v6e-8 s140 (`a6a62e`) started
fresh. Still shows the same direction but should not be treated as flagship.

### Cross-Experiment Evidence (incomplete)

The pattern is directionally present in other experiment types but these
comparisons are less clean:
- **1b**: v6e-8 s420 vs v5p-8 s210 — different step counts, not matched
- **2a**: v5p-8 run was still in progress at last check
- **2b**: no v5p-8 comparator available

### Initialization (step 0)

**Established.** Both runs start in the same state:

| Metric | v6e-8 | v5p-8 | Diff |
|---|---|---|---|
| train/loss | 0.693147 | 0.693147 | 0.000 |
| grad/norm/total | 28.8546 | 28.8568 | 0.008% |
| LoRA-only L2 grad norm (448 tensors) | 2.5308 | 2.5316 | 0.03% |

LoRA A init is provably identical (deterministic from `seed=0`, device-independent
pytree traversal). LoRA B is zeros (`zero_init_b=True`). LR schedule is identical
at every logged step.

**Caveat:** Near-equal step-0 gradient norms do not prove equal full gradients.
The gradients include per-token contributions from the CE kernel, which uses
different block sizes on v5p vs v6e. The DPO loss at initialization is
uniformly ln(2) per example (policy = reference), so block-size differences
would not manifest in the step-0 loss or its gradients. They would appear
starting at step 1 once per-example losses diverge.

### Parameter Norms (stay close despite loss divergence)

**Established (from Codex revalidation on fresh s70 pair).**

| Step | v5p-8 lora_B norm | v6e-8 lora_B norm |
|---|---|---|
| 10 | 0.001166 | 0.001139 |
| 60 | 0.006556 | 0.006408 |

Parameters are not in wildly different states. The loss divergence is not
caused by dramatically different update magnitudes. This is consistent with
the CE hypothesis: slightly different gradients (from different block-size
tiling) compound into large loss differences on this tiny, sensitive dataset.

### Gradient Norm Evolution

| Step | v6e-8 total | v5p-8 total |
|---|---|---|
| 0 | 28.855 | 28.857 |
| 10 | 16.021 | 28.000 |
| 50 | 14.086 | 23.819 |
| 130 | 12.659 | 19.978 |

v6e-8 norms drop rapidly (learning). v5p-8 norms barely decrease (stuck).

---

## W&B Run Index

| Experiment | v6e-8 run | v5p-8 run | Notes |
|---|---|---|---|
| 1a lr1e6 s70 | `smh_lr1em06_s70_v6e8-fbac2a` | `smh_lr1em06_s70_v5p8-964129` | **cleanest pair** |
| 1a lr1e7 s70 | `smh_lr1em07_s70_v6e8-4ac830` | `smh_lr1em07_s70_v5p8-f3d60b` | **cleanest pair** |
| 1a lr5e7 s70 | `smh_lr5em07_s70_v6e8-7ae68d` | `smh_lr5em07_s70_v5p8-86109f` | |
| 1a lr5e7 s140 | `smh_lr5em07_s140_v6e8-ddb5ce` | `smh_lr5em07_s140_v5p8-5f928a` | |
| 1a lr1e6 s140 | `smh_lr1em06_s140_v6e8-a6a62e` | `smh_lr1em06_s140_v5p8-cc7957` | v5p-8 RESUMED step 94 |
| 1b lr1e6 | `3stmt_lr1em06_s420_v6e8-fd4e55` | `3stmt_lr1em06_s210_v5p8-7fe6a8` | different step counts |
| 2a lr1e6 s140 | `support-mental-health_lr1em06_s140_v6e8-004b5d` | `support-mental-health_lr1em06_s140_v5p8-d9567e` | v5p-8 incomplete |
| Diag: v5p pd=8 | — | `smh_lr1em06_s70_v5p8_pd8-0498ec` | Experiment A |

All in W&B project `marin-community/dpo`.

---

## What Has Been Ruled Out

### 1. Microbatch / gradient accumulation code path
**Ruled out by Experiment A.** Forcing v5p-8 to use the same microbatch regime
as v6e-8 (pd=8 → 2× grad accum, microbatch=32) produced identical behavior to
original v5p-8 (pd=16, no grad accum). Both v5p-8 curves are slow learners.

### 2. Config mismatch
**Established.** W&B config diff shows only per_device_parallelism and derived paths.

### 3. LR schedule
**Established.** Identical values at every step.

### 4. Gradient clipping asymmetry
**Established.** `optax.clip_by_global_norm(1.0)` operates on trainable (LoRA)
grads only. Both runs have LoRA grad norm ~2.53 at step 0.

### 5. Checkpoint resume as root cause
**Established.** Fresh s70 pairs show the same divergence.

### 6. Data batch composition
**Established (local test).** The Feistel permutation with `seed=0` produces
identical cache indices for batch 0 regardless of device count (verified
locally in `debug_data_order.py`). Not yet verified end-to-end on TPU.

### 7. LoRA initialization
**Established.** Key derivation from `seed=0` is deterministic and
device-independent. Both runs start with identical LoRA A weights.

## Confounds (real but not root cause)

### Eval batch-size weighting
`eval_loss_loop` (`callbacks/__init__.py:32-88`) uses unweighted mean over
batches. Eval batch sizes differ (v6e-8: 32, v5p-8: 64). Step-0 eval values
match exactly, so this only matters when per-example losses vary. Does not
explain the training-loss divergence.

### Reference eval cache metadata
`dpo.py:227-244`: cache path hash matches but metadata comparison fails
(dict vs dataclass). Both TPU types rebuild the cache. Wastes compute but
does not affect training dynamics.

---

## Ranked Explanations (updated after Experiment J)

### 1. Strongest: Hardware-level bf16 numerical differences in forward/backward

**Confirmed by Experiment J.** On identical batches with identical initial
weights, v5p-8 and v6e-8 produce trainable gradients that:
- agree in L2 magnitude to ~0.0003%
- disagree significantly in signed direction (grad_sum differs by 0.155 absolute)

The forward pass computes `delta_pi - delta_ref` slightly differently per
example because of bf16 matmul / attention numerics that differ between
chip architectures. Since DPO at init sits on a flat loss landscape (every
example loss = ln(2)), tiny per-example logit differences directly
translate into gradient-direction differences that compound over training.

The source is somewhere in the forward/backward pass of the model itself —
before the CE kernel's accumulation steps. Most likely candidates:
- Splash attention numerics (hardcoded block_size=512 but matmul precision
  might differ between v5p MXU and v6e MXU)
- LoRA/Linear matmul precision in the model body
- bf16 accumulation in any reduction (e.g., LayerNorm, attention softmax)

### 2. Lower: CE kernel numerics

Different `num_b_blocks` (16 vs 64) in the CE backward means different bf16
accumulation patterns for the lm_head gradient (gw). But for LoRA training,
gw is FROZEN (`trainables_only(grads, is_trainable)` discards it). So the
gw accumulation difference doesn't affect LoRA training dynamics directly.
The gx gradient (activations) has the SAME number of accumulation steps on
both (16, since num_v_blocks=16 on both). So CE isn't the primary cause.

### 3. Eliminated by Experiment J

- ~~Data ordering / batch composition~~ (confirmed same batch sha256)
- ~~Initial model weights~~ (param norms match to 6 decimals)
- ~~Microbatch code path~~ (ruled out by Experiment A)
- ~~CE `v_block_size` hardware heuristic~~ (ruled out by Experiment G)

---

## Experiment Plan (revised after Experiment J)

The primary mystery is solved: **hardware numerics differ**. Remaining
questions are about which part of the model pipeline introduces the
divergence and whether we can fix it for training stability.

### Experiment K: Force fp32 matmul precision (HIGHEST INFO NEXT)

Add `precision="highest"` (or `precision=jax.lax.Precision.HIGHEST`) to
the model's matmuls / attention / CE kernel. This forces fp32 accumulation
in matmuls, eliminating the bf16 tolerance differences between v5p MXU and
v6e MXU.

If v5p-8 with fp32 precision matches v6e-8 at step 0 gradients:
→ **confirms bf16 matmul precision is the source.**
If still differs: attention or other non-matmul op is the source.

Cost: slower compute but we only need 2 steps.

### Experiment L: Isolate forward-pass log-probs (diagnostic)

Instrument `logp_sum()` in `dpo.py:130` to dump a checksum of
`logp_pi_chosen` and `logp_pi_rejected` BEFORE the delta/loss computation.
If log-probs differ between TPUs → the forward pass is the source.
If they match but gradients differ → the backward pass is the source.

This narrows the hunt: forward vs backward, then within each.

### Experiment M: Disable Splash attention

Fall back to the reference (non-splash) attention kernel to test if
attention is the culprit. If swapping attention implementation closes
the gap → attention numerics are the cause.

### Experiment E: v6e-8 pd=2 (SECONDARY, largely obsolete)

Was meant to test the CE block-size hypothesis. Now that we know gradients
differ on same batch regardless of blocking, this is lower value. Could
still be useful to confirm v6e stays fast across configs.

### Experiments G, I, H (SUPERSEDED)

G was done (confirmed block sizes). I and H were framed around the CE
block-size hypothesis which Experiment J rules out as the primary cause.
Not worth running.

---

## Code References

| Component | File | Lines |
|---|---|---|
| CE kernel (XLA) | `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/xla.py` | 101-350 |
| CE block size heuristic | `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/tuned_block_sizes.py` | 779-818 |
| CE API + impl selection | `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/api.py` | 80-194 |
| CE loss flattening | `lib/levanter/src/levanter/models/loss.py` | 245 |
| DPO loss → CE path | `lib/levanter/src/levanter/dpo.py` | 130-132 |
| Gradient accumulation | `lib/levanter/src/levanter/grad_accum.py` | 36-169 |
| Trainer step | `lib/levanter/src/levanter/trainer.py` | 678-716 |
| Gradient filtering | `lib/levanter/src/levanter/trainer_state.py` | 236-268 |
| Eval loss loop | `lib/levanter/src/levanter/callbacks/__init__.py` | 32-88 |
| Experiment config | `experiments/posttrain/per_stmt_dpo/common.py` | 74-93 |

---

## Experiment Log

### 2026-04-13T20:10Z — Experiment A: v5p-8 with forced pd=8

**Script:** `experiments/posttrain/per_stmt_dpo/debug_accum_v5p8_pd8.py`
**Iris job:** `/ahmed/debug-accum-v5p8-pd8`
**W&B run:** `smh_lr1em06_s70_v5p8_pd8-0498ec`
**Config:** v5p-8, pd=8, microbatch=32, 2× grad accum, lr=1e-6, s70

**Result: MATCHES ORIGINAL v5p-8 (slow). Does NOT match v6e-8.**

At step 21: v5p pd=8 = 0.6397, v5p pd=16 = 0.6406, v6e pd=4 ≈ 0.37.
The two v5p curves track each other exactly. This rules out the microbatch
code path as the cause. Under the CE hypothesis, this is expected: changing
pd on v5p does not change its hardware-dependent CE v_block (still 16384).

### 2026-04-13T20:30Z — LoRA initialization verified identical

Key derivation from `seed=0` is deterministic and device-independent.
`adapter_key = [928981903, 3453687069]`. Pytree traversal order is the same
on both TPU types. LoRA A weights are provably identical at init.

### 2026-04-13T20:45Z — Data ordering verified identical (local)

`debug_data_order.py` confirmed the Feistel permutation produces identical
cache indices for batch 0 regardless of device count (4 or 8 devices both
get indices `[1536, 625, 1162, 271, ...]` for seq 0-63). Not yet verified
end-to-end on real TPU hardware.

### 2026-04-13T21:30Z — CE kernel hypothesis identified

David (senior engineer) suggested checking kernel block sizes. Investigation
revealed hardware-dependent `v_block_size` in XLA fused CE backward. This is
now the leading hypothesis. See "Ranked Explanations" section above.

### 2026-04-13T23:30Z — Experiment G run: CE block sizes logged

Added DEBUGSTART/DEBUGEND markers in `xla.py` to print resolved block sizes
on first kernel call. Results from `smh_lr1em06_s35_*_fp32_upcast` runs:

**v6e-8:**
- device_kind: `TPU v6 lite`
- x.shape: (16384, 4096)
- v_block_size: 8192, b_block_size: 1024
- num_v_blocks: 16, **num_b_blocks: 16**

**v5p-8:**
- device_kind: `TPU v5`
- x.shape: (65536, 4096)
- v_block_size: 8192, b_block_size: 1024
- num_v_blocks: 16, **num_b_blocks: 64**

**Key finding:** `v_block_size` is IDENTICAL on both (8192). The earlier
hypothesis that v5p gets 16384 was wrong — the heuristic checks
`device_key == "TPU v5p"` but v5p reports as `"TPU v5"`, so the
hardware-specific branch never fires.

The real difference is `num_b_blocks`: 16 vs 64. This comes from the
per-device flattened batch being 4× larger on v5p (65536 vs 16384).

### 2026-04-13T23:52Z — Experiment: fp32 upcast of gw_block (PARTIAL IMPROVEMENT)

Patched `xla.py:338-352` to accumulate `gw_block` in fp32 across batch blocks,
casting down to bf16 only once when writing into the final `gw` slice.

Launched fresh 35-step runs on both TPUs:
- `smh_lr1em06_s35_v6e8_fp32_upcast-a0befb` (v6e-8)
- `smh_lr1em06_s35_v5p8_fp32_upcast-1ea0d9` (v5p-8)

**Early signal (v5p-8 at step 12): loss=0.6597**
- Baseline v5p-8 at step ~12: ~0.68
- Modest improvement (~0.02) but still nowhere near v6e-8 (~0.37 at step 12)

**Preliminary conclusion:** Upcasting only the `gw_block` accumulation is
NOT sufficient to close the gap. Either:
1. There's additional bf16 accumulation elsewhere in the CE kernel (`gx`
   still accumulates across vocab blocks in bf16 at `xla.py:347`), OR
2. The CE backward isn't the primary source of divergence at all

**Resolution (confirmed in Experiment J): option 2.** The CE backward
isn't the primary cause. gw is frozen in LoRA (not applied to model), and
gx accumulates 16 times on BOTH TPUs, not 64 vs 16.

### 2026-04-14T01:47Z — Experiment J: 2-step deterministic trace (BREAKTHROUGH)

**Script:** `experiments/posttrain/per_stmt_dpo/debug_two_step_trace.py`
**Instrumentation:** DEBUGSTART/DEBUGEND blocks in `loader.py:452` and
`trainer.py:698` guarded by env vars:
- `MARIN_DEBUG_LOG_BATCH_INDICES=1` → logs first 5 batches' indices + sha256
- `MARIN_DEBUG_LOG_STEP_TRACE=1` → `jax.debug.print` emits full trace

**Jobs:**
- `/ahmed/debug-j-trace-v6e8` (W&B: `two_step_trace_v6e8-*`)
- `/ahmed/debug-j-trace-v5p8-r2` (W&B: `two_step_trace_v5p8-*`)

**Result: SAME DATA, DIFFERENT GRADIENTS.**

Step 0 train batch: both TPUs loaded indices [0-63] with identical
sha256=`7a61ce53d17eb721`. Permutation, data loader, and sharding are
device-count-independent as expected.

Step 0 trainable gradient trace:

| Metric | v6e-8 | v5p-8 | Δ |
|---|---|---|---|
| loss | 0.6931471824645996 | 0.6931471824645996 | exact (tautological at init) |
| grad_l2 | 2.5308258533477783 | 2.531604528427124 | 0.00031% |
| **grad_sum** | **-0.8846888542175293** | **-0.7291930913925171** | **0.155** |
| upd_l2 | 0.004686407 | 0.004685373 | ~match |
| upd_sum | -0.001111864 | -0.0000680 | **huge relative** |
| pA_l2 | 59.07554 | 59.07554 | match to 6 dec |
| pA_sum | -18.20846 | -18.20843 | match to 4 dec |

Sentinel gradients (all are lora_B; lora_A gradients are all 0 at init
because lora_B=0 → no signal flows through lora_A):

| Module | v6e grad_l2 | v5p grad_l2 | v6e grad_sum | v5p grad_sum |
|---|---|---|---|---|
| q_proj | 0.5416715 | 0.5418293 | 0.4532508 | 0.4209823 |
| gate_proj | 0.8171922 | 0.8172969 | 0.2690451 | 0.3527192 |
| o_proj | 0.7086674 | 0.7090713 | -0.8998485 | -0.9008402 |

**Pattern:** L2 norms match to ~4 decimals. Signed sums diverge
significantly. Gradients have **same magnitude but different direction**.

**Interpretation:** The forward pass is computing `delta_pi - delta_ref`
slightly differently per example on the two chip architectures. At init
(delta=0), `d(softplus(-x))/dx = -0.5`, so tiny per-example delta
differences translate directly into gradient-direction differences.

**This is hardware-level bf16 numerical divergence** in the forward/
backward pass. Not data, not init, not grad accum, not CE block sizes.

**Why it compounds into dramatic training differences:** DPO at init sits
on a perfectly flat loss landscape (every example loss = ln(2)). Any small
gradient direction shift lands the model in a different basin after step 1.

See "Ranked Explanations" above for updated hypothesis and "Experiment
Plan" for next experiments (K: force fp32 precision; L: isolate
forward-pass log-probs; M: disable Splash attention).

### 2026-04-14T04:55Z — Experiment J2: 1-step LoRA factor trace (PARTIAL SUCCESS)

Goal: For each LoRA module, log on step 0 (per senior engineer David's suggestion):
1. Forward factor `z = lora_A(x)` checksum/L2
2. Upstream cotangent `dL/d(lora_B_output)` checksum/L2
3. Compare against existing grad_B numbers

**Implementation:**
- `lib/levanter/src/levanter/lora.py` instrumented with DEBUGSTART/DEBUGEND blocks:
  - `jax.debug.print` logs `z` stats after lora_A (forward, stage=`z_after_lora_A`)
  - `jax.custom_vjp` identity op wraps lora_B output; backward supposed to print cotangent
  - Module tag uses INPUT axes to distinguish modules (embed4096/heads32_head_size128/mlp14336)
- `experiments/posttrain/per_stmt_dpo/debug_lora_factor_trace.py` — 1-step run
- Gated by `MARIN_DEBUG_LORA_FACTOR_TRACE=1`

**Issues encountered:**

1. **Initial eval consumed prints.** Initial eval (full_val, 503-batch Paloma/LM eval)
   triggered ~112k LoRA forward calls. With `jax.debug.print` forcing host-device sync,
   this made eval take >1h. First runs (r2, r3) never reached training step.
   - **Fix:** Added `max_eval_batches` to SimpleDPOConfig and plumbed through default_dpo.
     `max_eval_batches=0` crashes with "AsyncDataset has length 0". `max_eval_batches=1`
     works — only ~1800 LoRA calls during eval instead of 112k.

2. **Backward cotangent trace NOT emitting.** v6e-8-r4 run SUCCEEDED with training step
   complete (loss=0.693 logged). Produced 917 DEBUGJ LORA_FWD lines. But **0 DEBUGJ
   LORA_BWD lines**. The `jax.custom_vjp` bwd with `jax.debug.print` is not outputting.
   Possible causes:
   - `jax.debug.print` inside custom_vjp bwd may be DCE'd by XLA
   - NamedArray ↔ jax.Array roundtrip may break the VJP chain
   - Custom_vjp bwd in autodiff graph may not mark prints as required side effects

3. **DEBUGJ TRACE (trainer.py) and DEBUGJ BATCH (loader.py) also NOT firing.** These
   are runtime env var checks (not import-time like lora.py's `_DBG_LORA_FACTOR_TRACE`).
   Env var may not propagate from Iris executor parent → train_dpo sub-job at runtime,
   OR the check is before a code path that doesn't execute. Needs investigation.

**What we have from v6e-8-r4:**
- 917 DEBUGJ LORA_FWD entries with module tags
- Sample values: mod=embed4096 l2≈400-700, sum varies; mod=heads32_head_size128 l2≈600-900;
  mod=mlp14336 l2≈80-520

**What we DON'T have:**
- LORA_BWD cotangent values (custom_vjp bwd silent)
- v5p-8 counterpart (TPU capacity exhausted, jobs pending ~30 min)

**Next step:** Get the backward to actually emit. Options:
- Use `jax.debug.print(..., ordered=True)` to force execution
- Use `jax.experimental.io_callback` instead of debug.print
- Use `jax.vjp` directly in training loop to capture gradients in forward-pass style
- Add a sentinel read of the cotangent that makes it "live" in JAX's eyes

### 2026-04-14T05:15Z — Experiment J3: cleaner approach per Codex critique

Codex rightly critiqued the J2 approach:
- `jax.custom_vjp` inside a hot library path (scan/vmap/remat/partitioning) is
  too fragile as a debugging probe
- Shape-derived tags are not stable module identities
- Treating `ordered=True` as forcing execution is a misconception (it only
  orders existing effects)
- Overcommits to bf16 story despite `mp=jmp.get_policy("p=f32,c=bfloat16")`
  making params f32

New plan: forward-factor trace only + trainer.py's existing sentinel-grad
trace = enough to localize divergence.

**Implementation changes:**

1. Removed custom_vjp backward machinery from `lora.py` entirely.

2. Added `debug_name: str = eqx.field(static=True, default="")` to
   `LowRankLinear`. Propagated via `LoraLinear.init(debug_name=...)` ←
   `_loraize_module(key_path)`. So each LoRA module now has a stable path
   like `transformer.stacked.self_attn.q_proj`.

3. Forward instrumentation restricted to sentinel modules (q_proj,
   gate_proj, o_proj) by substring match on `debug_name`. Avoids the
   noisy 917-line output; expect ~32 layers × 3 sentinels × 2 forwards
   = 192 prints per training step.

4. Relying on trainer.py's existing grad/update/param checksum trace
   (from Experiment J) to capture sentinel grad_B values. We already
   have the per-module grad_l2/grad_sum from Experiment J — those are
   the "backward" numbers Codex says we need.

**Logic to apply when data lands:**
- If `z_after_lora_A` matches across TPUs for same module → forward path
  to that module is identical → divergence enters IN lora_B backward or
  further upstream in the gradient chain
- If `z_after_lora_A` differs → forward path is already diverging (attention
  or earlier LoRA modules)

**Jobs launched** (2026-04-14T05:15Z):
- `/ahmed/debug-j3-v6e8`
- `/ahmed/debug-j3-v5p8-central1`
- `/ahmed/debug-j3-v5p8-east5`

All with `MARIN_DEBUG_LORA_FACTOR_TRACE=1`. 1 training step, `max_eval_batches=1`.

### 2026-04-14T11:55Z — Experiment L result: per-stmt full-FT DPO on v5p-16 vs v6e-16 (EARLY MATCH, BUT NOT A PURE LORA-ONLY ABLATION)

**Runs:**
- v6e-16 full FT DPO: https://wandb.ai/marin-community/dpo/runs/full_ft_s10_v6e16_l1-008fdb
- v5p-16 full FT DPO: https://wandb.ai/marin-community/dpo/runs/full_ft_s10_v5p16_l1-bb2be5

**State at analysis time:**
- `full_ft_s10_v5p16_l1-bb2be5`: finished at W&B `_step=9`
- `full_ft_s10_v6e16_l1-008fdb`: crashed at W&B `_step=8` before the scheduled
  step-10 eval; W&B had not uploaded `output.log` yet, so the strongest
  cross-TPU evidence here is the scalar history plus the finished v5p worker log

**Config parity (verified from W&B config):**
- same singleton per-stmt dataset:
  `preference/bloom_v2_singleton/support_mental_health/`
- same `train_batch_size=64`, `lr=1e-6`, `beta=0.1`, `seed=0`,
  `train_seq_len=4096`, `validation_split_fraction=None`
- same mixed precision policy: params `f32`, compute/output `bf16`
- same `per_device_parallelism=4`
- no adapter (`adapter=null`) — full fine-tune with `SeparateReferenceConfig`

**Important scalar result:** the catastrophic v5p↔v6e split does **not**
reproduce in full FT on the same per-stmt dataset.

Recall from `dpo.py:119` that the loss-driving quantity is
`delta_pi - delta_ref`, not `dpo_margin_policy` alone.

Through the overlapping prefix (steps 0-8), the two full-FT runs stay close on
both loss and `delta_pi - delta_ref`:

| step | v6e loss | v5p loss | v6e `delta_pi-delta_ref` | v5p `delta_pi-delta_ref` |
|---|---|---|---|---|
| 0 | 0.693147 | 0.693147 | 0.000 | 0.000 |
| 2 | 0.692416 | 0.688913 | 0.019 | 0.087 |
| 3 | 0.678566 | 0.673635 | 0.299 | 0.396 |
| 4 | 0.664171 | 0.663108 | 0.593 | 0.613 |
| 5 | 0.655864 | 0.656349 | 0.769 | 0.754 |
| 6 | 0.618792 | 0.615969 | 1.559 | 1.620 |
| 7 | 0.606406 | 0.603591 | 1.828 | 1.889 |
| 8 | 0.597066 | 0.593090 | 2.043 | 2.127 |

Max difference through step 8:
- `max |Δ(train/loss)| = 0.00493`
- `max |Δ(delta_pi-delta_ref)| = 0.0977`

Compare that to the pathological LoRA `r64/alpha64` pair from Experiment K:
- `max |Δ(train/loss)| = 0.35646`
- `max |Δ(delta_pi-delta_ref)| = 10.0553`

So on the same per-stmt dataset, the full-FT discrepancy is roughly:
- **~72x smaller on loss**
- **~103x smaller on the loss-driving DPO quantity**

This is the strongest evidence so far that the earlier catastrophic split is
**not generic to v5p vs v6e DPO**, and that the LoRA regime is the main
amplifier.

**Reward metrics tell the same story.** By step 8:
- v6e full FT: chosen reward `+0.1426`, rejected reward `-0.0616`
- v5p full FT: chosen reward `+0.1504`, rejected reward `-0.0623`

By contrast, the LoRA pair had already separated dramatically by step 3:
- v6e LoRA: chosen reward `+0.6852`, rejected reward `-0.2955`
- v5p LoRA: chosen reward `+0.0198`, rejected reward `-0.0075`

### Full-FT gradient evidence

The finished v5p worker log confirms normal full-FT learning dynamics:
- step 0: `loss=0.693147`, `grad_l2=28.826`, `grad_sum=3.7428`
- step 2: `loss=0.688913`, `grad_l2=27.871`, `grad_sum=1.4474`
- step 3: `loss=0.673635`, `grad_l2=27.496`, `grad_sum=-5.8480`
- step 8: `loss=0.593090`, `grad_l2=25.897`, `grad_sum=2.0256`

These come from the `DEBUGJ TRACE` lines in the finished `v5p` `output.log`.
The LoRA-specific sentinel slots (`gA`, `gB`, `pA`, `pB`) are all zero here
because this is a non-LoRA run; the useful fields are the global
`grad_l2/grad_sum/upd_l2/upd_sum/param_l2/param_sum`.

Step-0 gradient norms also match closely across TPU types in W&B:
- v6e: `grad/norm/total = 28.8673`
- v5p: `grad/norm/total = 28.8260`

Across the full per-parameter grad-norm tree at step 0:
- 582 comparable grad-norm keys
- median relative difference: `~2.97e-6`
- p95 relative difference: `~0.0092`
- max relative difference: `~0.051`

So full FT still has small TPU-level numeric differences, but they do **not**
turn into the qualitatively different early trajectory seen in LoRA.

### Concrete difference vs LoRA that matters: execution geometry is cleaner here

Experiment L is **not** a pure "remove LoRA, hold everything else fixed"
ablation. It also made the execution geometry much more apples-to-apples:

| Run family | v6e setup | v5p setup |
|---|---|---|
| Bad per-stmt LoRA | `v6e-8`, `pd=4` | `v5p-8`, `pd=16` |
| Exp L full FT | `v6e-16`, `pd=4` | `v5p-16`, `pd=4` |

This matters for the CE kernel path. In the finished `v5p-16` full-FT log:
- `DEBUGCE XLA CE block sizes resolved: device_kind=TPU v5`
- `x.shape=(32768, 4096)`
- `v_block_size=8192`
- `b_block_size=32768`
- `num_v_blocks=16`
- `num_b_blocks=1`

That is a very different regime from the bad LoRA pair in Experiment G, where:
- v6e had `num_b_blocks=16`
- v5p had `num_b_blocks=64`

So Exp L changed **both**:
1. model/update parameterization (full FT instead of LoRA), and
2. local CE execution geometry / batch blocking

This means Exp L strongly supports "LoRA is the amplifier," but does **not**
mathematically prove "LoRA is the only thing that changed."

### Strongest current mechanism story

Claude's mechanism explanation is directionally right and now fits the evidence
well:

- In LoRA with `zero_init_b=True`, the first useful adapter update is forced
  through `lora_B`, while `lora_A` stays fixed at its random initialization.
- LoRA excludes `lm_head` by default, so the model cannot absorb the DPO signal
  in the most direct place where logit differences are expressed.
- This means tiny chip-level numeric differences in the early policy/reference
  log-prob computation get projected through a fixed low-rank basis, making the
  first update much more sensitive to those differences.
- Full FT lacks that bottleneck: the update is spread across the full policy,
  including `lm_head`, so small TPU numeric differences are diluted rather than
  amplified.

That is the best current explanation for:
- why Experiment J saw "same batch, different trainable gradients" in LoRA, and
- why Experiment L stays well aligned despite the underlying TPU numerics not
  being perfectly identical.

### What Exp L establishes vs what it does not

**Established:**
- per-stmt **full FT DPO** does **not** show the catastrophic v5p↔v6e split in
  the first 8-10 steps
- the per-stmt dataset by itself is **not** sufficient to force the earlier
  LoRA-style failure mode
- generic "v5p vs v6e breaks DPO" is no longer a credible explanation

**Not yet isolated:**
- whether the bad behavior is primarily:
  1. LoRA parameterization,
  2. LoRA interacting with the earlier `v5p-8` vs `v6e-8` execution geometry,
  3. or both

### Revised 2×2 matrix status

|   | per-stmt (singleton) | full bloom_v2 |
|---|---|---|
| **LoRA** | ❌ diverges early (`r16/alpha32`, `r64/alpha64`) | ❓ **still missing** |
| **Full FT** | ✅ **matches early on v5p-16 vs v6e-16** | ✅ closer match |

**Important caveat on the new green check:** the per-stmt full-FT result
used much closer TPU geometry than the bad LoRA pair, so the cell should be
read as "matches early under a cleaner paired setup," not "pure LoRA-only
ablation complete."

### Highest-info next experiments after Exp L

Two experiments are now high value for different reasons:

1. **Full-data LoRA DPO (Exp M)** — fills the remaining matrix quadrant.
   - If it matches: singleton-data interaction is a major part of the problem.
   - If it splits: LoRA-DPO itself is fragile across TPU families.

2. **Per-stmt LoRA on matched geometry (`v5p-16 pd=4` vs `v6e-16 pd=4`)**
   - This is now the cleanest way to separate:
     - "LoRA is the core amplifier" from
     - "the earlier `v5p-8` vs `v6e-8` geometry mismatch was doing more damage than we thought."

If only one follow-up can be run, this second experiment is now arguably the
highest-info next step because Exp L showed the importance of geometry matching.

### 2026-04-14T12:20Z — Experiment M re-spec: matched-geometry LoRA on the full statement distribution

The earlier Exp M spec was too weak because it changed dataset **and** left
geometry loose (`v5p-8 pd=-1` vs `v6e-8 pd=4`). After Exp L, that is no longer
acceptable: geometry matching matters enough that Exp M should be specified as a
clean paired run, not a convenience run.

**Revised goal:** test whether LoRA still shows a TPU-family split when trained
on the **full 46-statement Bloom v2 preference distribution** rather than the
singleton `support_mental_health` subset, while keeping execution geometry as
close as possible across v5p and v6e.

This is the remaining missing quadrant in the matrix:

|   | per-stmt (singleton) | full 46-statement distribution |
|---|---|---|
| **LoRA** | ❌ diverges early | ❓ **Experiment M (re-spec below)** |
| **Full FT** | ✅ matches early | ✅ matches / closer match |

#### Experiment M (re-spec)

**Hypothesis:** if LoRA is trained on the full statement distribution under
matched TPU geometry, then:
- **M matches** → singleton-data interaction was a major part of the earlier pathology
- **M splits** → LoRA-DPO itself is fragile across TPU families even when the
  data distribution is broad and geometry is controlled

**Data (train + val):**
- Train: `preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/`
  This is the full Bloom v2 / full 46-statement preference distribution used by
  the repo's broader DPO experiments.
- Validation: the usual bloom v2 val set already used in the existing LoRA/full-FT runs

**Model/reference:**
- policy: `marin-community/marin-8b-instruct`
- adapter: LoRA with the checked-in Marin recipe
  - `r=64`
  - `alpha=64`
  - `zero_init_b=True`
  - `target_modules=None`
  - default `exclude_modules` (so `lm_head` remains excluded unless changed explicitly)
- reference: `AdapterBaseReferenceConfig`

**Training knobs:**
- `num_train_steps=10` for the first pass
- `train_batch_size=64`
- `per_device_parallelism=4` on **both** TPUs
- `per_device_eval_parallelism=4` on **both** TPUs unless memory forces a different eval setting
- `lr=1e-6`
- `lr_schedule=cosine`
- `warmup=0.1`
- `beta=0.1`
- `seed=0`
- `train_seq_len=4096`
- `validation_split_fraction=None`
- one eval at step 10

**TPU pairing (preferred):**
- `v5p-16 pd=4`
- `v6e-16 pd=4`

This mirrors Experiment L's cleaner geometry and avoids repeating the older
`v5p-8 pd=16` vs `v6e-8 pd=4` mismatch. If `v6e-16` does not fit or is
unavailable, escalate only with an explicit note that the comparison weakened.

**Debug instrumentation for first pass:**
- `MARIN_DEBUG_LOG_BATCH_INDICES=1`
- `MARIN_DEBUG_LOG_STEP_TRACE=1`

Do **not** enable `MARIN_DEBUG_LORA_FACTOR_TRACE` on the first pass. The first
goal is simply to determine whether the scalar/trace split survives on the full
distribution under matched geometry.

**Implementation note:**
- reuse the `experiments/tune_lora/` codepath rather than cloning from
  `per_stmt_dpo`
- add a dedicated debug script under `experiments/tune_lora/` so the run stays
  aligned with the checked-in full-distribution LoRA recipe while still forcing
  the matched `pd=4` geometry and debug env vars

**Interpretation priority after M lands:**
1. If Exp M matches: singleton-data interaction becomes the leading remaining explanation.
2. If Exp M splits: LoRA-DPO itself remains the leading amplifier, and the next
   best probe is per-stmt LoRA on matched `v5p-16` / `v6e-16` if not already run.

### 2026-04-14T13:20Z — Experiment M(pd=4) launch blocked by v6e-16 HBM OOM

The first implementation of Exp M used the intended matched geometry:
- `v5p-16 pd=4`
- `v6e-16 pd=4`
- LoRA `r=64`, `alpha=64`, `zero_init_b=True`
- `AdapterBaseReferenceConfig`

That config **did not compile on v6e-16**. The TPU worker failed with:

> `RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in`
> `memory space hbm. Used 42.82G of 31.25G hbm. Exceeded hbm capacity by 11.57G.`

This is surprising at first glance because Experiment L's **full FT** run fit on
`v6e-16 pd=4`. But the clean interpretation is **not** "LoRA should always be
smaller than full FT, therefore this must be impossible." The LoRA DPO runtime
here is a different compiled graph from Exp L in two important ways:

1. **Reference-model path differs.**
   - Exp L used `SeparateReferenceConfig`, so training state was a materialized
     `DpoModel(policy, reference)` and the reference model was loaded
     separately.
   - Exp M uses `AdapterBaseReferenceConfig`, so the loss function constructs
     the reference on the fly from the policy via
     `reference_provider.model_for(policy_model)` in `train_dpo.py`.
   - For LoRA, that reference is produced by unwrapping the policy's LoRA
     modules back to their wrapped base linears via `unwrap_lora_modules(...)`.

2. **LoRA adds extra per-layer compute/activation structure.**
   - A LoRA-adapted linear computes `wrapped(x) + lora_B(lora_A(x))`.
   - That means extra branch structure and extra rank-`r` intermediates (`z =
     lora_A(x)`) at many linears, even though the trainable parameter count is
     much smaller than full FT.

So the strongest current explanation for the OOM is:
- **not** "full FT has one forward but LoRA has two" — DPO performs
  policy/reference chosen/rejected work in both cases
- but rather that **LoRA + AdapterBaseReferenceConfig** induces a different,
  more memory-hungry compiled graph than **full FT + SeparateReferenceConfig**
  on `v6e-16 pd=4`

This is also a reminder that Exp L and Exp M are not only "full FT vs LoRA";
they currently differ on **reference configuration** as well:
- Exp L: `SeparateReferenceConfig`
- Exp M: `AdapterBaseReferenceConfig`

That reference-type difference was already deliberate for Exp M because:
- `AdapterBaseReferenceConfig` is the canonical LoRA DPO path in this repo
- switching Exp M to `SeparateReferenceConfig` would add another 8B reference
  copy and create a different, less standard LoRA setup

### Revised recommendation after the OOM

The cleanest next move is:
- **drop Exp M to `pd=2` on both `v5p-16` and `v6e-16`**

Why this is the best fallback:
- keeps TPU family pairing matched (`v5p-16` vs `v6e-16`)
- changes only local batch / activation pressure
- avoids escalating to `v6e-32`, which would reintroduce a chip-count mismatch
- avoids changing reference type, which would weaken the "canonical LoRA path"
  interpretation

Tradeoff:
- `train_batch_size=64` with `pd=2` on 16-chip pods implies `microbatch=32`
  and therefore `grad_accum=2`

That is acceptable for a 10-step probe. It is a weaker geometry match than the
ideal `pd=4` / no-accum run, but it is still much cleaner than falling back to:
- `v6e-32`, or
- the old `v5p-8` vs `v6e-8` mismatch

**Updated Exp M execution plan:**
- first retry with `v5p-16 pd=2` vs `v6e-16 pd=2`
- keep `AdapterBaseReferenceConfig`
- keep LoRA `r=64`, `alpha=64`, `zero_init_b=True`
- keep the same debug env vars (`BATCH_INDICES`, `STEP_TRACE`)
- record explicitly that Exp M is now "matched family / matched pd, with grad
  accumulation on both sides"

---

## Root cause hypothesis after Exp Q

### Strongest current explanation

The best-supported root-cause story at this point is:

1. The catastrophic split is **not** caused by TPU family (`v5p` vs `v6e`).
   Exp N ruled that out by showing matched `v5p-16` and `v6e-16` LoRA runs
   track closely.
2. The split is **not** caused by a simple within-family `pd` change. Exp O
   showed full FT on `v6e-16` is robust to `pd: 4 -> 2`, and Exp P showed LoRA
   on `v5p-16` is robust to `pd: 2 -> 4`.
3. The remaining failure is specific to the **`v5p-8` distributed execution
   regime** under this LoRA DPO setup.

The most concrete low-level difference we have actually measured is the local
cross-entropy problem shape:

- **good `v5p-16 pd=2`** logged
  `DEBUGCE ... x.shape=(32768, 4096) ... b_block_size=32768 ... num_b_blocks=1`
- **bad `v5p-8 pd=8`** logged
  `DEBUGCE ... x.shape=(65536, 4096) ... b_block_size=1024 ... num_b_blocks=64`
- **bad `v5p-8 pd=4`** logged the **same** line as the `pd=8` run:
  `x.shape=(65536, 4096) ... b_block_size=1024 ... num_b_blocks=64`

So the strongest current hypothesis is:

> `v5p-8` with `train_batch_size=64` is running a materially different local CE
> kernel and broader distributed program from the good `v5p-16` regime, and
> that difference perturbs the **direction** of the early LoRA update enough to
> send training into the bad basin.

This explanation fits the full body of evidence better than the earlier
alternatives:

- **not hardware-family numerics alone**: `v5p-16` and `v6e-16` match
- **not LoRA rank alone**: `v5p-8` is still bad at fixed `r=64, α=64`
- **not `pd` alone**: `v5p-16` stays good at both `pd=2` and `pd=4`
- **not the `v5p-8` `pd=8 -> 4` change**: Exp Q showed the local CE shape did
  not change at all, and the bad trajectory stayed bad

The part I still consider an inference rather than a proof is **which** piece
of the `v5p-8` execution regime is load-bearing:

- the CE kernel shape / batch blocking
- the broader FSDP sharding / all-gather / reduce-scatter topology
- or both together

But the CE difference is currently the most concrete, first-principles
mechanism we have actually observed in logs.

### Why this can happen even though `v5p-16` is "just more chips"

`v5p-16` is not merely "the same run with more throughput." It is a different
distributed factorization of the training step:

- each chip holds smaller shards
- each chip contributes fewer token rows to the local CE kernel
- all-gather / reduce-scatter trees differ
- remat / HLO scheduling choices can differ
- temporary activation shapes differ

For full FT, those differences seem to wash out. For this LoRA DPO setup, they
appear to matter because the early update is very brittle:

- `lora_B` starts at zero
- `lora_A` is fixed random
- the initial DPO signal is nearly symmetric (`δ_π - δ_ref ≈ 0` at step 0)
- small directional differences in early gradients can therefore pick a
  different low-rank update direction

That is why the current best explanation is **directional instability in the
early LoRA update**, not a simple scale or convergence-speed issue.

### Next best experiment: make `v5p-8` match the good `v5p-16` local execution

The highest-information next move is **not** another `pd` sweep on `v5p-8`.
Exp Q already showed `pd=8` and `pd=4` leave `v5p-8` in the same bad regime,
and both logged the same CE shape.

The next best experiment is:

#### Experiment R — `v5p-8` local-shape matching probe

> **Status (2026-04-16T04:52Z):** executed and resolved to Case 2 (no
> recovery). The `bs=32 pd=4` run on `v5p-8` did **not** match the good
> `v5p-16 pd=2` `DEBUGCE` line exactly; instead it reduced the local CE
> problem from `(65536, 4096), num_b_blocks=64` down to
> `(16384, 4096), num_b_blocks=16`, and still stayed stuck near `ln(2)`.
> See the "2026-04-16T04:52Z: Experiment R result" section at the top of
> this logbook for the corrected side-by-side table, full 10-step trace,
> step-0 gradient-shift finding, W&B link, and pivot direction
> (topology / sharding / collective). The optional Follow-up R3
> `bs=16 pd=4` sweep is deprioritized; the load-bearing variable is no
> longer expected to be per-chip token load.

**Goal:** try to make the bad `v5p-8` run look like the good `v5p-16 pd=2`
run at the level that matters most in logs: the local CE problem shape seen by
the fused XLA cross-entropy kernel.

This is **not** a "match the global batch size" experiment. It is a "match the
per-chip local execution shape as closely as we can without changing hardware"
experiment.

##### Why this is the right next move

At fixed `train_batch_size=64` we measured:

- good `v5p-16 pd=2`:
  - `DEBUGCE x.shape=(32768, 4096)`
  - `b_block_size=32768`
  - `num_b_blocks=1`
- bad `v5p-8 pd=8` and `pd=4`:
  - `DEBUGCE x.shape=(65536, 4096)`
  - `b_block_size=1024`
  - `num_b_blocks=64`

So the cleanest remaining test is to reduce the global batch on `v5p-8` until
the per-chip CE problem hopefully moves toward the good `v5p-16` regime.

##### Exact config

Hold fixed from the recent LoRA debug runs:
- TPU: `v5p-8`
- data: per-stmt `support_mental_health`
- tokenizer: `marin-community/marin-tokenizer`
- model: `marin-community/marin-8b-instruct`
- LoRA: `r=64`, `alpha=64`, `dropout=0.0`, `zero_init_b=True`,
  `target_modules=None`
- reference: `AdapterBaseReferenceConfig`
- `lr=1e-6`
- `lr_schedule="cosine"`
- `warmup=0.1`
- `beta=0.1`
- `seed=0`
- `train_seq_len=max_seq_len=4096`
- `num_train_steps=10`
- `steps_per_eval=10`
- `max_eval_batches=1`
- `ReferenceEvalCacheConfig(mode="disabled")`
- `MARIN_DEBUG_LOG_BATCH_INDICES=1`
- `MARIN_DEBUG_LOG_STEP_TRACE=1`

Change only:
- `train_batch_size: 64 -> 32`

Recommended first launch:
- `v5p-8`
- `train_batch_size=32`
- `per_device_parallelism=4`

Why start with `pd=4`:
- that path already ran cleanly in Exp Q
- it keeps the parent/child operational story simple
- it avoids conflating this experiment with another `pd` sweep

##### What this is trying to match

The target is **not** the full `v5p-16` training config. The target is the
good `v5p-16 pd=2` **local CE regime**:

- desired `DEBUGCE` trend:
  - `x.shape` moving from `(65536, 4096)` toward `(32768, 4096)`
  - `b_block_size` moving from `1024` toward the full-batch case
  - `num_b_blocks` moving from `64` toward `1`

If `train_batch_size=32` on `v5p-8 pd=4` does that, then we have successfully
matched the most concrete measured difference between the bad and good runs.

##### Required checks during the run

1. **Check `DEBUGCE` first.**
   This experiment is only informative if the local CE shape actually changes.

   Required log lines:
   - `x.shape`
   - `b_block_size`
   - `num_b_blocks`

2. **Check `DEBUGJ TRACE` on steps 0, 2, and 9.**
   We care about:
   - `loss`
   - `grad_l2`
   - `gB_l2`
   - `pB_l2`

3. **Check W&B on the true DPO quantity.**
   Compare:
   - `train/loss`
   - `train/dpo_margin_policy`
   - `train/dpo_margin_ref`
   - therefore `delta_pi - delta_ref`

##### Success / failure criteria

**Strong recovery**
- `DEBUGCE` moves toward the good `v5p-16` regime
- and step-2 `train/loss` drops into the good band (`~0.33-0.35`)
- and `delta_pi - delta_ref` jumps into the good band (`~9-10`)

**No recovery**
- `DEBUGCE` changes as intended
- but step-2 `train/loss` still stays near `~0.68`
- and `delta_pi - delta_ref` stays tiny (`~0.1-0.7`)

**Ambiguous partial recovery**
- `DEBUGCE` changes
- and the loss improves materially from the bad `v5p-8` runs
- but still does not reach the good `v5p-16` regime

##### Interpretation

**Case 1: strong recovery**

This would be the cleanest evidence so far that the leading root cause is the
per-chip local workload / local CE shape on `v5p-8`, not some deeper
`v5p-8`-specific collective bug.

**Follow-up R1:**
- rerun on `v5p-8 bs=32 pd=8`
- goal: see whether recovery persists across `pd` once the per-chip token load
  is reduced

**Case 2: no recovery**

This would mean we successfully matched the most obvious CE-level difference
yet the run still failed. At that point CE batch blocking is probably not the
load-bearing cause, and the remaining culprit is deeper in the `v5p-8`
distributed regime.

**Follow-up R2:**
- add one more probe that keeps `v5p-8 bs=32` but **forces explicit CE block
  sizes** to match the good `v5p-16` line as closely as possible, if the
  runtime heuristic still does something unexpected
- if CE is already matched and the run is still bad, stop focusing on CE and
  pivot to topology / sharding / collective investigation

**Case 3: ambiguous partial recovery**

This would mean local CE shape is a contributor but not the whole story.

**Follow-up R3:**
- complete a two-point local-load sweep on `v5p-8`
- compare:
  - `bs=64 pd=4` (known bad)
  - `bs=32 pd=4`
  - optional `bs=16 pd=4`
- track whether the run improves smoothly as local workload falls

##### Why this is higher information than another plain `pd` sweep

Exp Q already showed:
- `v5p-8 pd=8` bad
- `v5p-8 pd=4` bad
- same `DEBUGCE` line in both runs

So another plain `pd` change is unlikely to teach much unless it actually moves
the local CE problem. `train_batch_size=32` is the first next experiment that
directly tests the strongest remaining hypothesis from measured logs.

---

## 2026-04-16T13:05Z: Root-cause update after Exp R — next best probe is **full FT on `v5p-8`**, not another LoRA kernel guess

### What we have now established

At this point the investigation has narrowed materially:

- **Not TPU family alone:** `v5p-16 pd=2` and `v6e-16 pd=2` LoRA DPO track
  closely (Exp N).
- **Not plain `pd` alone:** `v5p-16` stays good at both `pd=2` and `pd=4`
  (Exp P), and full FT on `v6e-16` stays good at both `pd=4` and `pd=2`
  (Exp O).
- **Not LoRA rank / scale alone:** `v5p-8` is still bad at fixed
  `r=64, alpha=64` (Exp Q).
- **Not CE batch blocking as the primary cause:** Exp R reduced the `v5p-8`
  local CE problem from `(65536, 4096), num_b_blocks=64` down to
  `(16384, 4096), num_b_blocks=16`, and the run still stayed near `ln(2)`.

So the remaining live question is no longer "is the CE kernel wrong?" The
remaining live question is:

> **Is the `v5p-8` pathology specific to LoRA / `AdapterBaseReferenceConfig`,
> or is it a broader property of the `v5p-8` distributed regime even under
> full fine-tuning?**

That is the cleanest next discriminator.

### Why `v6e-8` full FT is not required

It would be nice to have a symmetric `v6e-8` full-FT run, but it is not
necessary for the next narrowing step.

We already know:
- `v5p-16` full FT works
- `v6e-16` full FT works
- `v5p-16` and `v6e-16` full FT track closely
- `v5p-8` LoRA is the only regime that still looks uniquely pathological

Therefore the next high-information question is simply:

> **Can `v5p-8` do full FT DPO at all, and if it can, does it learn normally?**

If the answer is yes, the remaining pathology is much more likely to be
specific to the LoRA + adapter-base-reference path on `v5p-8`.
If the answer is no, then the bug is broader to the `v5p-8` execution regime.

### Important correction to the "match the batch size" intuition

It is tempting to say:

> if `v5p-16` and `v6e-16` both work, then as long as `v5p-8` matches their
> batch size it should also track

But that is **not** guaranteed.

`v5p-8` vs `v5p-16` is not just a batch-size change. It changes the whole
distributed factorization:

- data-axis size
- parameter shard size
- all-gather / reduce-scatter trees
- replica / DCN topology
- local activation shapes
- optimizer-state shards per chip

Exp R already demonstrated this principle: even after shrinking the local CE
shape substantially, `v5p-8` still stayed bad. So "same global batch" is too
weak a matching criterion by itself.

### Memory / checkpointing reality

One reason we have not already run full FT on `v5p-8` is memory pressure.
However, the correct statement is:

- we have **not proved** that `v5p-8` full FT is impossible
- we only know that the existing full-FT experiments were done on 16-chip pods

Relevant code fact:
- `llama_8b` already uses `gradient_checkpointing=True` by default in
  `lib/levanter/src/levanter/models/llama.py`
- Levanter also supports more aggressive checkpointing policies:
  - `"offload"`: offload carries and inputs to host
  - `"recompute"` / `"full"`: don't save carries or block internals

So the correct next move is not to assume "full FT won't fit on `v5p-8`."
The correct next move is to test feasibility with a short run and an explicit
fallback ladder.

### Next best experiment: **Experiment T — `v5p-8` full-FT feasibility + behavior probe**

**Goal:** determine whether the `v5p-8` pathology is LoRA-specific or broader.

#### Baseline comparison runs

Compare against the already-good full-FT runs:
- Exp L `v5p-16 pd=4`
- Exp L `v6e-16 pd=4`
- Exp O `v6e-16 pd=2`

We do **not** need a `v6e-8` full-FT run to answer the immediate question.

#### T0: minimal feasible first attempt

Use the Exp L full-FT recipe, but target `v5p-8` and reduce global batch:

- TPU: `v5p-8`
- data: per-stmt `support_mental_health`
- model: `marin-community/marin-8b-instruct`
- **no adapter** (full FT)
- reference: `SeparateReferenceConfig`
- `train_batch_size=32`
- `per_device_parallelism=4`
- `per_device_eval_parallelism=4`
- `num_train_steps=2` for the first compile/feasibility probe
- `steps_per_eval=2`
- `lr=1e-6`
- `beta=0.1`
- `seed=0`
- `train_seq_len=max_seq_len=4096`
- `max_eval_batches=1`
- `ReferenceEvalCacheConfig(mode="disabled")`
- debug env:
  - `MARIN_DEBUG_LOG_BATCH_INDICES=1`
  - `MARIN_DEBUG_LOG_STEP_TRACE=1`

Model config:
- start with plain `llama_8b`
- note: this already means `gradient_checkpointing=True`

#### T0 fallback ladder if it OOMs

If the first compile fails on HBM:

1. **T0b:** rerun with
   `dataclasses.replace(llama_8b, gradient_checkpointing="offload")`
   This is the most practical "aggressive checkpointing" fallback already used
   elsewhere in the repo.

2. **T0c:** if `"offload"` still fails, rerun with
   `dataclasses.replace(llama_8b, gradient_checkpointing="recompute")`
   This is more compute-heavy but minimizes saved activations further.

3. **T0d:** only if needed, drop `train_batch_size` from `32 -> 16`
   while keeping everything else fixed.

The right order is important:
- first change the checkpointing policy
- only then reduce batch again
- do not mix several memory changes at once on the first retry

#### What to look for

If `v5p-8` full FT **fits and learns**:
- step-2 loss should leave `ln(2)` clearly
- `DEBUGJ TRACE` grad norms should look like the good full-FT regime, not the
  bad LoRA regime
- then the pathology is much more likely to be specific to:
  - LoRA low-rank update geometry
  - `AdapterBaseReferenceConfig`
  - or their interaction with `v5p-8`

If `v5p-8` full FT **fits but is still bad**:
- the bug is broader than LoRA
- at that point the `v5p-8` distributed regime itself becomes the primary
  suspect

If `v5p-8` full FT **does not fit even after offload / recompute**:
- that is itself useful information
- it means the cleanest behavior comparison on `v5p-8` may have to stay in
  LoRA-space or move to a smaller seq/batch probe

### Why this is higher value than another kernel-specific guess

After Exp R, another naive CE-specific probe is lower leverage. We already
shrunk the local CE regime substantially and did not recover. If we revisit CE,
the next honest version would be an **exact-match** probe, not another broad
guess.

The next most valuable piece of information is not "which kernel might still be
different?" It is:

> **Does the `v5p-8` pathology survive when LoRA is removed?**

That question is more important than any individual kernel hypothesis because
it cleanly partitions the remaining search space.

### 2026-04-16T13:18Z — Experiment T launch plan (pending submission)

Concrete script prepared:
- `experiments/posttrain/per_stmt_dpo/experiment_t_v5p8_full_ft_s2.py`

Initial launch target:
- `v5p-8`
- per-stmt `support_mental_health`
- **full FT** (no adapter)
- `SeparateReferenceConfig`
- `train_batch_size=32`
- `per_device_parallelism=4`
- `num_train_steps=2`
- `steps_per_eval=2`
- `seed=0`, `lr=1e-6`, `beta=0.1`
- `gradient_checkpointing="offload"` for the first attempt

Why `offload` first:
- `llama_8b` already uses normal gradient checkpointing
- the immediate goal here is feasibility on `v5p-8`, not a purity contest
- if plain checkpointing would OOM, losing a launch cycle tells us less than
  getting a concrete behavioral answer

Planned parent/child launch shape:
- parent: Iris interactive CPU coordinator in `us-east5-a`
- child: `v5p-8` TPU in `us-east5`
- env:
  - `REGIONS_OVERRIDE=us-east5`
  - `EXPERIMENT_T_BS=32`
  - `EXPERIMENT_T_PD=4`
  - `EXPERIMENT_T_STEPS=2`
  - `EXPERIMENT_T_CHECKPOINTING=offload`
  - `MARIN_DEBUG_RUN_TAG=ue5a-i1`

Immediate success criteria:
- child `train_dpo` spawns
- compile completes
- at least one `DEBUGJ TRACE` line is emitted

If that works, the next step is to compare the step-2 loss / grad trace
against the good `v5p-16` / `v6e-16` full-FT runs.

Submission record:
- iris parent job: `/ahmed/debug-t1-full-ft-v5p8-bs32-pd4-offload-ue5a-i1`
- experiment/run name: `exp_t_v5p8_fullft_bs32_pd4_offload_s2_ue5a-i1`
- expected checkpoint base path:
  `checkpoints/exp_t_v5p8_fullft_bs32_pd4_offload_s2_ue5a-i1`
- submit mode: `interactive`
- parent launch command shape:
  - `iris job run --zone us-east5-a --cpu 1 --memory 3g`
  - env:
    - `REGIONS_OVERRIDE=us-east5`
    - `EXPERIMENT_T_BS=32`
    - `EXPERIMENT_T_PD=4`
    - `EXPERIMENT_T_STEPS=2`
    - `EXPERIMENT_T_CHECKPOINTING=offload`
    - `MARIN_DEBUG_RUN_TAG=ue5a-i1`
- submit status: accepted by Iris controller
- immediate post-submit state: parent `running`, task `assigned`
- at the time of this log entry, the parent has **not yet spawned**
  `/train_dpo`, so there is no TPU-child status or W&B run id yet

---

## 2026-04-16T20:00Z: Experiment R2 — explicit CE block-size probe on `v5p-8`

### Context: what Exp R did and did not show

The Exp R writeup (2026-04-16T04:52Z) has been corrected in place to
reflect the actual `DEBUGCE` line from
`experiment_r_r64_v5p8_bs32_pd4_s10_ue5a-i1-423c65`:

```
device_kind=TPU v5 x.shape=(16384, 4096) w.shape=(4096, 128256)
v_block_size=8192 b_block_size=1024 num_v_blocks=16 num_b_blocks=16
explicit_block_sizes=False
```

i.e. `x.shape=(16384, 4096)`, `b_block_size=1024`, `num_b_blocks=16` —
**not** the good `v5p-16 pd=2` line's `(32768, 4096) / 32768 / 1`.
Dropping `train_batch_size: 64 → 32` on v5p-8 pd=4 halved `B` per chip to
*below* the good run's level, and the heuristic picked 16 batch blocks
rather than matching the good run's single block.

Exp R's useful conclusion: shrinking per-chip CE workload below the good
run's level does not recover training on v5p-8. But that alone doesn't
rule out CE batch-blocking as the cause — we didn't bit-match the good
regime, we overshot it. A clean direct test of the batch-blocking
hypothesis was still needed.

Side-by-side Exp R vs good/bad baselines at the CE-kernel level:

| Run | `x.shape` | `b_block_size` | `num_b_blocks` | Outcome |
|-----|-----------|----------------|----------------|---------|
| Good v5p-16 pd=2 bs=64 (Exp N)  | (32768, 4096) | 32768 | 1  | escapes ln(2) by step 2 |
| Bad  v5p-8  pd=4 bs=64 (Exp Q)  | (65536, 4096) | 1024  | 64 | stuck near ln(2) |
| Exp R v5p-8 pd=4 bs=32          | (16384, 4096) | 1024  | 16 | stuck near ln(2) |

### Why Exp R2

The Exp R approach (reduce `bs` to change CE tiling) is *indirect*. Changing
`bs` simultaneously changes three things:

1. per-chip token count `B` (which changes the CE kernel input shape),
2. `grad_accum` (at fixed global batch, `grad_accum = bs / (pd × chips)`),
3. the heuristic's choice of `b_block_size` / `num_b_blocks` (as a function
   of `B` and `device_kind`).

To isolate "is CE tiling the load-bearing variable?", the cleaner probe is
to **hold everything else fixed at the Exp Q bad baseline** (`bs=64`, `pd=4`,
same microbatch, same grad_accum, same x.shape) and change **only** the CE
tiling by forcing explicit block sizes through the kernel call path.

The XLA CE kernel already accepts explicit block-size overrides — the
`explicit_block_sizes=False` field in every current DEBUGCE line is the
switch. Setting `explicit_block_sizes=True` with chosen `b_block_size` /
`v_block_size` / `num_b_blocks` bypasses the heuristic.

### Hypothesis

**Strong version:** the load-bearing difference between good `v5p-16 pd=2`
and bad `v5p-8 pd=4` is the CE backward's bf16 accumulation pattern across
many batch blocks. `v5p-16 pd=2` runs with `num_b_blocks=1` (no inter-block
accumulation). `v5p-8 pd=4` runs with `num_b_blocks=64` (63-way bf16
accumulation of `gw_block` / `gx_block` inner tiles). If we eliminate that
accumulation on v5p-8 while holding everything else fixed, v5p-8 recovers.

**Weak version:** the accumulation count doesn't matter; per-tile compute
shape does. Then matching `b_block` to the good run's `32768` (with
`num_b_blocks=2` to cover `B=65536`) would be the relevant knob.

**Null:** neither matters. CE tiling is not the load-bearing variable, and
the failure lives somewhere below CE (FSDP / attention / reference-network
graph), consistent with the prevailing hypothesis after Exp R.

### The geometric impossibility of a fully bit-identical CE match

To match the good v5p-16 pd=2 CE kernel exactly, we'd need all four of:

- `x.shape = (32768, 4096)` — determined by per-chip `B`
- `v_block_size = 8192`
- `b_block_size = 32768`
- `num_b_blocks = 1`

On v5p-8 with `bs=64, pd=4`, per-chip `B = 65536`. To cover `B=65536` with a
single batch block, `b_block_size` must be `65536`, not `32768`. Conversely,
if we pin `b_block_size = 32768`, `num_b_blocks` must be `2`, not `1`. You
cannot both (a) keep `B=65536` and (b) have per-tile math identical to the
good run and (c) have a single batch block. One of those has to give.

The honest options are:

- **Give up `num_b_blocks=1`** (Case B below): match per-tile math
  (`b_block=32768`), accept 2-way inter-tile accumulation.
- **Give up matching tile size** (Case A below): force `num_b_blocks=1`
  with a tile of size 65536, accept that the tile is 2× larger than the
  good run's.
- **Give up `B=65536`** (rerun Exp R more carefully at `bs=32 pd=2` on
  v5p-8 — see "alternative" below).

### Three cases

#### Case A — eliminate inter-block accumulation

Force: `b_block_size=65536, num_b_blocks=1, v_block_size=8192, num_v_blocks=16`.

Matches good run on: `num_b_blocks=1`, `num_v_blocks=16`, `v_block_size`.
Differs from good run on: `b_block_size` (65536 vs 32768), `x.shape`.

Tests the **strong hypothesis** most cleanly. If v5p-8 still stays near
`ln(2)` under Case A, then bf16 accumulation across batch blocks in the CE
backward is **ruled out** as the load-bearing cause, even with everything
else held at the bad baseline.

#### Case B — match per-tile compute shape

Force: `b_block_size=32768, num_b_blocks=2, v_block_size=8192, num_v_blocks=16`.

Matches good run on: `b_block_size`, `v_block_size`, `num_v_blocks`.
Differs from good run on: `num_b_blocks` (2 vs 1), `x.shape`.

Per-tile CE math is bit-identical to the good run. Only difference is a
single 2-way bf16 reduction of the two tile outputs. If the weak hypothesis
were true (per-tile compute shape matters, not accumulation count), this
would recover.

#### Case C — impossible

Forcing `b_block_size=32768, num_b_blocks=1` would only cover 32768 of the
65536 rows. Kernel would either error out or silently compute on half the
batch. Skip.

### HBM feasibility

Concurrent CE backward temporaries per chip (bf16, `H=4096`, `V_pad=128256`,
`v_block=8192`):

| Array | Shape | At `b_block=65536` | At `b_block=32768` | At `b_block=1024` (current bad) |
|-------|-------|--------------------|--------------------|--------------------------------|
| `x_block`          | (b_block, H)         | 512 MiB | 256 MiB | 8 MiB |
| `delta` (softmax)  | (b_block, v_block)   | 1024 MiB| 512 MiB | 16 MiB |
| `gx_inner`         | (b_block, H)         | 512 MiB | 256 MiB | 8 MiB |
| `gw_block_update`  | (H, v_block)         | 64 MiB  | 64 MiB  | 64 MiB |
| **Peak CE temps**  |                      | **~2.1 GiB** | **~1.1 GiB** | **~0.1 GiB** |

Baseline HBM on v5p-8 (4 chips, 95 GiB/chip) for this config:

- policy + reference (Llama-8B shared under `AdapterBaseReferenceConfig`),
  FSDP-sharded over 4 chips: ~4 GiB/chip for weights
- LoRA params + AdamW states (LoRA-only): <100 MiB/chip
- activations with `gradient_checkpointing=True` default: ~10-20 GiB/chip
- scratch / collectives / comms buffers: few GiB

Estimated current usage at `b_block=1024`: ~20-30 GiB/chip. Adding ~2 GiB
for Case A's CE temps or ~1 GiB for Case B's lands around 22-32 GiB/chip,
well inside the 95 GiB budget.

**Verdict: no OOM expected.** Prudent to do a 1-step compile-only probe
first to confirm, since the compile-time HBM estimator can be off by a few
GiB under certain conditions.

### Recommended order

1. **R2a — Case A** (`b_block=65536, num_b_blocks=1`). Strongest probe of
   "is CE backward bf16 accumulation the cause?" at the full bad-baseline
   geometry. ~10-minute run.

2. **R2b — Case B** (`b_block=32768, num_b_blocks=2`). Runs only if R2a
   stays stuck. Probes "does per-tile compute shape matter even with
   accumulation present?" Also ~10-minute run.

3. **R2c — alternative, if the "everything below CE" story is right and
   neither R2a nor R2b recovers:** shift focus to FSDP sharding / attention
   kv-head mapping / reference-network graph. (Already covered by the
   post-Exp-R pivot plan at 2026-04-16T13:05Z — Exp T is the current
   in-flight probe for that.)

### Config parity required

Hold fixed from Exp Q bad baseline (the apples-to-apples anchor):

- TPU: `v5p-8`, 4 chips, `us-east5`
- data: per-stmt `support_mental_health` singleton
- tokenizer: `marin-community/marin-tokenizer`
- model: `marin-community/marin-8b-instruct`
- LoRA: `r=64, alpha=64, zero_init_b=True, target_modules=None`
- reference: `AdapterBaseReferenceConfig`
- `train_batch_size=64`
- `per_device_parallelism=4`
- `per_device_eval_parallelism=4`
- `lr=1e-6, lr_schedule="cosine", warmup=0.1`
- `beta=0.1, seed=0`
- `train_seq_len=max_seq_len=4096`
- `num_train_steps=10, steps_per_eval=10`
- `max_eval_batches=1, ReferenceEvalCacheConfig(mode="disabled")`
- `MARIN_DEBUG_LOG_BATCH_INDICES=1`
- `MARIN_DEBUG_LOG_STEP_TRACE=1`

Change only:

- `explicit_block_sizes=True` in the CE kernel call path
- `b_block_size` and `num_b_blocks` to the Case-A or Case-B values
- `v_block_size=8192, num_v_blocks=16` held identical to heuristic pick

### Implementation notes

- The CE kernel lives in
  `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/xla.py`
  and its API shim is at
  `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/api.py`
  (lines 80-194).
- The block-size heuristic is in
  `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/tuned_block_sizes.py`
  (lines 779-818).
- The cleanest plumbing is probably to add a kwarg to the CE call path that
  gets threaded down to the kernel; an environment-variable shim
  (`MARIN_DEBUG_CE_B_BLOCK=65536`, etc.) keeps the experiment change
  minimal and avoids touching core Levanter code in the worktree.
- Verify that `explicit_block_sizes=True` appears in the new DEBUGCE line
  before trusting any downstream result — that's the signal that the
  override actually took effect.

### Success / failure criteria

**Case A (R2a) recovery:**
- DEBUGCE: `b_block_size=65536, num_b_blocks=1, explicit_block_sizes=True`
- step-2 `train/loss` drops into the good band (~0.33-0.35)
- `delta_pi - delta_ref` jumps into the good band (~9-10)
→ "CE backward bf16 inter-block accumulation is the load-bearing cause."
Warrants a focused fix at the CE kernel level.

**Case A (R2a) no recovery:**
- DEBUGCE confirms override took effect
- step-2 `train/loss` still sits near `~0.68-0.69`
→ CE backward bf16 accumulation across batch blocks is **ruled out**
as the load-bearing cause, even with a direct test. Pivot fully to the
sub-CE suspects (FSDP, attention, reference graph, remat).

**Case B (R2b) differential:** only meaningful if Case A stayed stuck.
If Case B recovers while Case A didn't, per-tile compute shape matters in a
way that accumulation count alone doesn't capture. Interesting but less
likely than the binary Case A outcome.

### Why this is higher information than rerunning Exp R at a cleaner bs/pd

At `v5p-8 bs=32 pd=2` (2 examples/chip/microstep × 4 multiplier × 4096 =
32768 per chip), we would naturally land at `x.shape=(32768, 4096)` and
presumably `b_block=32768, num_b_blocks=1` via the heuristic — bit-identical
CE line to the good run. That experiment is also worth running (call it R3
if needed). But it still changes `bs`, `pd`, and `grad_accum` simultaneously
vs the Exp Q bad baseline.

R2 keeps `bs`, `pd`, `grad_accum`, microbatch, and step-0 batch composition
identical to Exp Q, and toggles *only* the CE tiling. That's a stricter
isolation and directly addresses the remaining live question without
introducing new confounds.

### Open question this doesn't address

If both R2 cases stay stuck, we still haven't tested the other leading
suspects (attention kv-head mapping, FSDP sharding, reference-network
graph). Those are the Exp S / Exp T / follow-up agenda. R2 is a quick,
low-cost probe to close out the CE-level hypothesis cleanly before
committing to those more invasive experiments.

---

## 2026-04-16T22:50Z: Experiment R2a result — **CE backward bf16 accumulation is NOT the cause**

### Strongest true conclusion

With `b_block_size=65536, num_b_blocks=1` forced on `v5p-8 pd=4 bs=64` —
i.e. eliminating all inter-batch-block bf16 accumulation in the fused CE
backward, with every other knob held identical to the Exp Q bad baseline —
training **still stays stuck near ln(2)**.

This is a **Case 2 (no recovery)** outcome on the Exp R2 planned
discriminator. It rules out "CE backward bf16 accumulation across batch
blocks is the load-bearing cause" as a load-bearing explanation of the
v5p-8 pathology. Combined with Exp R's CE x.shape reduction (also no
recovery), this closes out the CE-kernel-level hypothesis cleanly and
pivots the investigation fully to sub-CE suspects.

### Run

- `experiment_r2a_r64_v5p8_bs64_pd4_s10_uc1-i1-8dac8a`:
  https://wandb.ai/marin-community/dpo/runs/experiment_r2a_r64_v5p8_bs64_pd4_s10_uc1-i1-8dac8a
- region: `us-central1-a` (parent pinned), v5p-8 TPU child
- env: `EXPERIMENT_R2A_BS=64`, `EXPERIMENT_R2A_PD=4`,
  `EXPERIMENT_R2A_CE_B_BLOCK_SIZE=65536`, `EXPERIMENT_R2A_CE_V_BLOCK_SIZE=8192`,
  `MARIN_DEBUG_RUN_TAG=uc1-i1`

### Override verified

Both diagnostic prints appear in the worker log:

```
DEBUGCE_OVERRIDE: env override taken MARIN_DEBUG_CE_B_BLOCK_SIZE=65536 MARIN_DEBUG_CE_V_BLOCK_SIZE=8192 -> BlockSizes(b_block_size=65536, v_block_size=8192)
DEBUGCE XLA CE block sizes resolved: device_kind=TPU v5 x.shape=(65536, 4096) w.shape=(4096, 128256) v_block_size=8192 b_block_size=65536 num_v_blocks=16 num_b_blocks=1 explicit_block_sizes=True
```

So the CE kernel saw the exact bad-baseline `x.shape=(65536, 4096)` and ran
it as a single 65536-row tile instead of the heuristic's 64-tile schedule.
Same data ordering as prior v5p-8 runs: step-0 batch `sha256=7a61ce53d17eb721`,
bit-identical to Exp J, Exp Q, and the original bad run.

### Step-0 aggregate gradient — indistinguishable from Exp Q

The forced single-tile CE backward produces essentially the same step-0
gradient as the 64-tile heuristic backward:

| Run | `num_b_blocks` | step-0 `grad_l2` | step-0 `grad_sum` |
|-----|----------------|-----------------:|------------------:|
| Exp Q `v5p-8 pd=4 bs=64` (heuristic, bad)    | 64 | 2.456003 | +0.18959 |
| Exp Q `v5p-8 pd=8 bs=64` (heuristic, bad)    | 64 | 2.456232 | +0.18380 |
| **Exp R2a `v5p-8 pd=4 bs=64` (forced, this)**| **1**  | **2.456294** | **+0.18907** |
| Good `v5p-16 pd=2 bs=64`                     | 1  | 2.456284 | +0.27991 |

Relative deltas vs Exp Q pd=4:
- `grad_l2`: +0.012% (2.456294 − 2.456003)
- `grad_sum`: −0.27% (0.18907 − 0.18959)

These are noise-level. Collapsing the bf16 inter-block accumulation from
64-way to 1-way **did not** materially change the step-0 gradient on
v5p-8. If the accumulation pattern were the numerically-load-bearing
difference between the bad v5p-8 regime and the good v5p-16 regime, we
would expect step-0 grads to shift — they don't.

Interestingly, `grad_l2` on R2a is closer to the good v5p-16 pd=2
baseline (2.456284) than to Exp Q (2.456003), but `grad_sum` sits firmly
in the bad-pool direction (+0.189 vs good-pool +0.280). So R2a
step-0 grads align with the good run in magnitude and with the bad pool
in direction — and training tracks the bad pool. Consistent with the
broader finding from Exp N/Q/R that step-0 per-module grad direction is
not a reliable predictor of training outcome.

### Training trajectory — tracks Exp Q, not the good v5p-16 run

W&B scalar history through step 8 (run still in flight but trajectory is
unambiguous):

| step | Exp Q `v5p-8 pd=4` (bad) | **Exp R2a `v5p-8 pd=4 b_block=65536` (this)** | Good `v5p-16 pd=2` |
|------|--------------------------:|----------------------------------------------:|-------------------:|
| 0    | 0.693147 | 0.693147 | 0.693147 |
| 1    | 0.693147 | 0.693147 | 0.693147 |
| 2    | 0.685125 | **0.684550** | 0.335202 |
| 3    | 0.682298 | **0.682521** | 0.325988 |
| 4    | 0.673723 | **0.676150** | 0.336246 |
| 5    | 0.668946 | **0.669656** | 0.316800 |
| 6    | 0.667573 | **0.669083** | 0.336998 |
| 7    | 0.662823 | **0.663766** | 0.324271 |
| 8    | 0.658715 | **0.659077** | 0.306144 |

R2a tracks Exp Q point-for-point (max |Δloss| ≤ 0.003 across all 8 logged
steps). It is firmly in the bad basin, not escaping to the good basin.

True DPO quantity (`delta_pi − delta_ref`) from W&B:

| step | R2a `delta_pi − delta_ref` |
|------|---------------------------:|
| 0    |  0.0000 |
| 1    |  0.0000 |
| 2    | +0.1741 |
| 3    | +0.2154 |
| 4    | +0.3449 |
| 5    | +0.4791 |
| 6    | +0.4901 |
| 7    | +0.5989 |
| 8    | +0.6963 |

Compare to good Exp N `v5p-16 pd=2` which jumps to `~9.4` by step 2 and
stays in the `~9-10` band. R2a stays in the `~0.1-0.7` bad band, same as
Exp Q and Exp R.

### Validation-set behavior at step 10

| split | pre-training | post-step-10 | Δ |
|-------|-------------:|-------------:|--:|
| stmt_val | 0.6931 | 0.6931 | 0.0 |
| full_val | 0.6931 | 0.6931 | 0.0 |

No meaningful generalization — identical to the Exp Q / Exp R stuck
regime. `dpo_accuracy` on both val splits is exactly 0 at eval time.

### What R2a rules out

**Strong hypothesis ruled out:** the CE backward's bf16 accumulation of
`gw_block` / `gx_block` across batch blocks is **not** the load-bearing
cause of the v5p-8 pathology. We forced that accumulation to zero-way
(one tile, no inter-block reduction) and the run stayed stuck.

**Corollary strengthened from Exp R:** CE kernel tiling choice in
general — tile size, number of batch blocks, accumulation order — is not
what's breaking v5p-8. Exp R showed reducing per-chip B below the
good-run's level didn't help. Exp R2a shows forcing `num_b_blocks=1` at
the bad-baseline B doesn't help either. Between the two, the
CE-kernel-level hypothesis is cleanly closed.

### What this leaves live

The remaining live suspects for the `v5p-8` pathology, now free of CE
confounds:

1. **Attention `kv_head` axis sharding** — v5p-8 and v5p-16 have
   different numbers of KV-head-axis-capable chips; the `dpo-lora`
   branch's existing TP=4 fix on v6e-8 (`0b228b3a5`) maps `kv_head` to
   the model axis, which may behave differently on v5p's smaller pod
   topology.
2. **FSDP granularity** — on 4 chips vs 8/16, all-gather /
   reduce-scatter trees differ; remat / HLO scheduling choices differ;
   optimizer-state shards-per-chip differ. Any of these could perturb
   the early LoRA update direction.
3. **`AdapterBaseReferenceConfig` reference-network graph on v5p-8** —
   the adapter-base reference path re-uses the policy with adapters
   disabled (via `unwrap_lora_modules`). The resulting compiled graph
   could differ in unexpected ways on v5p-8 vs the 16-chip pods.
4. **Remat / HLO scheduling interactions** — the llama_8b default uses
   `gradient_checkpointing=True`; the recompute graph on v5p-8's
   topology may lay out differently.

These are Exp S / Exp T / deeper-pivot territory. Exp T (full-FT on
v5p-8) is already the in-flight probe for "is this LoRA-specific or
broader to the v5p-8 execution graph?" — see the 2026-04-16T17:20Z
handoff at the top of the logbook for current status.

### R2b (b_block=32768, num_b_blocks=2) deprioritized

The R2 plan flagged R2b as a follow-up only if R2a stayed stuck and we
wanted to also rule out per-tile compute shape as a confound. With
R2a's step-0 grads landing within 0.012% of Exp Q's, it is very unlikely
that changing from `num_b_blocks=1` to `num_b_blocks=2` with a
`b_block=32768` tile would shift the outcome. Skip R2b and move
directly to the sub-CE probes.

### Operational notes

- Launch: us-central1-a parent, v5p-8 child (first submission that
  landed; the us-east5-a parallel submission can be killed if still
  pending — one clean run on this probe is sufficient).
- Runtime: 10 steps completed in roughly the same wall-clock as the Exp
  Q `pd=4` run. The `b_block=65536` tile is ~2 GiB in CE temporaries
  (vs ~33 MiB at b_block=1024); no OOM observed, confirming the HBM
  budget estimates in the R2 plan above.
- Kernel patch: `xla.py` now reads `MARIN_DEBUG_CE_B_BLOCK_SIZE` /
  `MARIN_DEBUG_CE_V_BLOCK_SIZE` and constructs a `BlockSizes` override
  when both are set. Backwards-compatible when env vars are unset.
  `DEBUGCE_OVERRIDE:` line fires exactly once per process at the first
  CE call if the override is taken; falls through silently otherwise.
- Experiment script:
  `experiments/posttrain/per_stmt_dpo/experiment_r2a_v5p8_pd4_explicit_ce_s10.py`.

### Updated post-R2 hypothesis ranking

1. **(leading)** Something in the v5p-8 distributed execution graph
   below the CE kernel — FSDP / attention sharding / reference-network /
   remat — perturbs the early LoRA update direction. Exp T's full-FT
   probe on v5p-8 will narrow "LoRA-specific interaction" vs "broader
   v5p-8 execution graph bug."
2. **(ruled out by Exp R + R2a)** CE kernel tiling, bf16 accumulation,
   per-chip CE workload.
3. **(ruled out by Exp N/O/P)** TPU family alone, `per_device_parallelism`
   alone, LoRA rank alone.

The investigation has successfully collapsed the CE-kernel branch of the
hypothesis tree.
