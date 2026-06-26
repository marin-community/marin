# True midtraining — resume Delphi pretrains at 80% with the data swap

> ## 🚨 CRITICAL — WRONG 1e20 BASE IN THIS PLAN — DO NOT LAUNCH AS-IS 🚨
>
> **Discovered 2026-05-14.** This plan hardcodes `isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5` as the 1e20 base (see §3, §4, §6.2, §7.3 — multiple sites). **This is the WRONG checkpoint.** It is from a deprecated v5 isoflop sweep, NOT the canonical Delphi v6 family. Will Held (Delphi lead) confirmed.
>
> The canonical Delphi 3e20 base is:
>
>     isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6
>     gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6/
>     Registered: experiments/exp1337_eval_suite.py:180
>
> 1e21 and 1e22 anchors in this plan ARE correct (the `-v5-` in their names is an unrelated experiment-iteration tag; their `LABEL` is `adamh_scaling_v6`).
>
> **Full post-mortem:** [`.agents/ops/2026-05-14-wrong-1e20-base-v5-vs-v6.md`](../ops/2026-05-14-wrong-1e20-base-v5-vs-v6.md)
>
> **Rule (never make this mistake again):** Marin scaling-law / Delphi / isoflop base checkpoints MUST be sourced from one of, in order:
> 1. `experiments/exp1337_eval_suite.py` EVAL_BASES dict (lines 174-186)
> 2. `MARIN_SCALING_SUITES["nemotron-completed-adamh"]` in `experiments/isoflop_sweep.py`
> 3. `experiments/exp1337_delphi_suite.py`
> 4. https://huggingface.co/collections/marin-community/delphi
>
> NEVER pick a base by `gsutil ls`-then-grep on `gs://marin-us-central2/checkpoints/isoflop/` — the bucket preserves every deprecated generation (v1 → v8 all live there). If the registry doesn't include the scale you want, STOP and ping Will Held before substituting.
>
> **The `v5` trap:** the string "v5" surfaces in three unrelated places. (1) `adamh_scaling_v5` suffix on isoflop step names = deprecated heuristic generation. (2) `adamh_scaling_v6` suffix = current canonical. (3) `-v5-XXXXXX` suffix on `adamh-scaling-ladder-nemotron-optimal-*` step names = unrelated experiment-iteration tag hardcoded in `exp1337_delphi_suite.py:232`; those runs use the v6 heuristic despite the `-v5-` in their names.

---

**Current status (codex 5.5 2026-05-16T00:19:24Z):** this plan did launch
after the original 2026-05-09 planning entry. Live W&B/GCS status: wrong-base
1e20 finished and is contaminated; valid 1e21 finished; valid 1e22 crashed
and needs resume from `true-midtrain-1e22-p33m67-step30000-v5p64`. See the
Codex status block at the end of this logbook.
**Author-agent:** 2026-05-09. Built on top of `.agents/logbooks/midtraining_delphi.md` and `.agents/projects/delphi_midtraining.md`.

---

## 1. Goal

Hijack the **decay phase** of each Delphi/AdamH pretrain run with the math midtraining data mixture. Concretely: pick up each scale's pretrain at the start of its WSD decay phase (≈ step 80%), keep the optimizer state, the LR schedule, and every other hyperparameter identical to the original pretrain, and **only swap the data mixture** to the math/pretrain blend used in the prior K=0.20 sweep. This is "TRUE midtraining" in the sense that the LR schedule and opt_state are continuations of the real pretrain run — not a freshly-warmed-up second-stage cooldown.

### Why this differs from the existing K=0.20 sweep (the previous logbook)

| Axis | Prior sweep (`exp_delphi_math_10b_midtrain.py`) | This plan |
|---|---|---|
| Init mode | `CheckpointInitMode.MODEL_ONLY` (weights-only) | Natural resume: pretrain checkpoint pre-staged into our run's own `output_path` |
| Optimizer state | Fresh (count=0, fresh momentum) | Preserved from pretrain |
| LR schedule | Fresh AdamH config: 500-step warmup → linear decay over remaining 4.5–7.5k steps; LR-factor × peak knob | Original AdamH config built by `completed_adamh_heuristic.build_optimizer_config(...)`; no LR factor, peak LR exactly as pretrain |
| `num_train_steps` | K=0.20 of pretrain tokens (~9.4k / 4.4k / 7.6k) | Original pretrain `num_train_steps` (47,064 / 22,057 / 38,235) |
| Effective midtraining duration | All `num_train_steps` (warmup + decay on math) | The remaining `(1 - 0.8) × num_train_steps` ≈ 20% of pretrain compute (decay-on-math) plus a small stable-phase pre-roll on math |
| Cells | 4 LR × 3 mix × 3 scale = 36 | 3 mix × 3 scale = **9** (no LR axis; the schedule is fixed by the pretrain) |
| Compute footprint | Each cell ≈ 20% of its pretrain | Each cell ≈ 20% of its pretrain (similar) |

The token-count budget happens to coincide (K=0.20 ≡ the 20% decay tail of WSD). The difference is the **schedule shape** and the **opt_state**: the prior sweep effectively re-warmed and re-decayed; this plan plays out the actual pretrain decay on the new data.

---

## 2. Pretrain WSD schedule — confirmed

Source of truth: `experiments/scaling_law_sweeps/completed_adamh.py` lines 127–131:

```python
min_lr_ratio: float = 0.0
warmup:        float = 0.1   # fraction of num_train_steps
lr_schedule:   str   = "linear"
decay:         float = 0.2   # fraction of num_train_steps
```

`exp1337_delphi_suite.py:run_optimal_training` builds the AdamH config straight from `completed_adamh_heuristic.build_optimizer_config(...)` with no overrides. So the production pretrain schedule is:

```
phase     │ step range (fraction)            │ LR
──────────┼──────────────────────────────────┼───────────────
warmup    │ [0,            0.10·N)           │ 0   → peak (linear)
stable    │ [0.10·N,       0.80·N)           │ peak
decay     │ [0.80·N,       N)                │ peak → 0 (linear, min_lr_ratio=0)
```

**The 80% step is exactly the start of the decay phase.** That is what makes "resume from 80%" the natural cut point.

The optimizer config additionally fixes (`completed_adamh.py:115–125`):

| Field | Value | Notes |
|---|---|---|
| `beta1` | 0.9 | constant |
| `beta2` | `clip(0.9999^(B/64), 0.9, 0.9999)` | per-base; constant token half-life |
| `epsilon` | `1.85e-8 · sqrt(r0/r)` | per-base |
| `learning_rate` (peak) | `0.00630 · sqrt(B/64) · (T0/T)^0.3` | per-base; clipped to `max_lr=0.01` |
| `adam_lr` (peak) | `0.000656 · sqrt(r/r0)` | per-base; clipped to `max_lr=0.01` |
| `max_grad_norm` | 0.1 | constant |
| `z_loss_weight` | 1.0e-7 | constant |
| `nesterov` | False | constant |

Reference `(B0=64, T0=2.5e9, seq_len=4096)`. We **do not hard-code** these values in our experiment file — we call `completed_adamh_heuristic.build_optimizer_config(batch_size=B, tokens=T)` per scale so any future heuristic change is picked up automatically.

---

## 3. Per-scale numbers — verified from `delphi_midtraining.md` § 3, § 4, § 8

> **WRONG 1e20 BASE — historical only.** The 1e20 row below is the bad
> `d2048-L21` / `adamh_scaling_v5` substitution from the incident. Do not use
> it for new launches; use `experiments/delphi_models.py` and stop until the
> intended true-midtraining 3e20/v6 optimizer-state plan is verified.

| Scale | Run name (canonical) | Hidden | Layers | Params | BS | `num_train_steps` (N) | Pretrain tokens (T) | 80% step (= start of decay) | Decay steps remaining (20%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **1e20** ¹ **[WRONG — historical only]** | `isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5` **[WRONG]** | 2048 | 21 | 1.9 B | 128 | **47,064** | 24.67 B | **37,651** | 9,413 |
| **1e21-v5** | `adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021` | 2560 | 26 | 3.4 B | 512 | **22,057** | 46.27 B | **17,646** | 4,411 |
| **1e22-v5** | `adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e` | 3840 | 37 | 9.7 B | 1024 | **38,235** | 160.37 B | **30,588** | 7,647 |

¹ **WRONG — see post-mortem.** The 1e20 slot used the wrong 3e20 isoflop
point (`d2048-L21`). The valid Delphi 3e20 bucket winner is `d2304-L23` /
`adamh_scaling_v6`; this old substitution is preserved only to explain the
historical mistake.

The **decay-step counts (9,413 / 4,411 / 7,647) match the K=0.20 midtraining step counts exactly**, because both are `0.20 × num_train_steps`. Confirms that K=0.20 was implicitly set to "as many tokens as the original WSD decay tail."

---

## 4. Available checkpoints — exact GCS paths and 80% targets

Production pretrains saved permanent checkpoints with `keep=[dict(every=5000)]` plus a final-step write (`exp1337_delphi_suite.py:174–177`). For 1e20 the cadence is every 10000 (different file). Verified by `gcloud storage ls` on 2026-05-09:

### 1e20 — `gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/checkpoints/` **[WRONG — historical only]**

> Do not use this checkpoint. It is the bad v5 isoflop ablation from the
> 2026-05-14 incident, not canonical Delphi.

```
step-10000/   step-20000/   step-30000/   step-40000/   step-46915/  (final)
```

| Candidate | Step | % of N | Phase | Δ vs target 37,651 |
|---|---:|---:|---|---:|
| `step-30000` | 30,000 | 63.7% | stable | −7,651 (−16.3%) |
| `step-40000` | 40,000 | 85.0% | early decay (`LR ≈ 0.75 × peak`) | +2,349 (+5.0%) |

### 1e21-v5 — `gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/checkpoints/`

```
step-5000/   step-10000/   step-15000/   step-20000/   step-21979/  (final)
```

| Candidate | Step | % of N | Phase | Δ vs target 17,646 |
|---|---:|---:|---|---:|
| `step-15000` | 15,000 | 68.0% | stable | −2,646 (−12.0%) |
| `step-20000` | 20,000 | 90.7% | mid-decay (`LR ≈ 0.46 × peak`) | +2,354 (+10.7%) |

### 1e22-v5 — `gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/`

```
step-5000/   step-10000/   step-15000/   step-20000/   step-25000/   step-30000/   step-35000/   step-38206/  (final)
```

| Candidate | Step | % of N | Phase | Δ vs target 30,588 |
|---|---:|---:|---|---:|
| `step-30000` | 30,000 | 78.5% | stable, 588 steps before decay | **−588 (−1.5%)** ← essentially at 80% |
| `step-35000` | 35,000 | 91.5% | mid-decay (`LR ≈ 0.42 × peak`) | +4,412 (+11.5%) |

### Final choice (user-confirmed 2026-05-09): **start at-or-after decay where possible; use the closest pre-decay only when it's the closest option**

| Scale | Chosen checkpoint | Resume step | Phase at resume | Remaining steps | Stable-on-math | Decay-on-math |
|---|---|---:|---|---:|---:|---:|
| 1e20 **[WRONG — historical only]** | `step-40000` | 40,000 | mid-decay (LR ≈ 0.75·peak) | 7,064 | 0 | 7,064 (= 75.0% of cooldown) |
| 1e21 | `step-20000` | 20,000 | mid-decay (LR ≈ 0.466·peak) | 2,057 | 0 | 2,057 (= 46.6% of cooldown) |
| 1e22 | `step-30000` | 30,000 | late-stable (LR = peak; 588 steps before decay starts) | 8,235 | 588 | 7,647 (= 100% of cooldown) |

**Rationale:**
* For 1e20 and 1e21, the closest checkpoint to the cooldown-start (post-decay) is the post-decay-start option.
* For 1e22 the closest checkpoint is `step-30000` which is 588 steps *before* decay starts — only 1.5% off, and it's the only way to capture the entire cooldown phase on math for that scale.
* Result: 1e22 gets the cleanest "full cooldown on math" signal; 1e20 and 1e21 catch only part of the cooldown. Mixed pre/post pattern across scales is intentional given the cadence.

### Token counts per cell (p33m67 only — 3 cells)

| Scale | (N − resume) | × BS | × seq_len | **Tokens / cell** |
|---|---:|---:|---:|---:|
| 1e20 **[WRONG — historical only]** | 7,064 | 128 | 4,096 | **3.70 B** |
| 1e21 | 2,057 | 512 | 4,096 | **4.31 B** |
| 1e22 | 8,235 | 1,024 | 4,096 | **34.54 B** |
| | | | **3-cell total** | **42.55 B** |

---

## 5. Resume mechanics — why the natural-resume path, not `initialize_from_checkpoint_path`

Levanter's `train_lm.py` (lines 203–215) has an `initialize_from_checkpoint_path` branch with two modes:

```python
case CheckpointInitMode.MODEL_ONLY:
    loaded_model = load_checkpoint(state.model, checkpoint_path, subpath="model")
    state = dataclasses.replace(state, model=loaded_model)
case CheckpointInitMode.FULL_STATE:
    state = load_checkpoint(state, checkpoint_path)
    state = dataclasses.replace(state, step=jnp.array(0))   # ← outer step reset
```

**Neither of these works for what we want.**

* `MODEL_ONLY`: drops the opt_state — exactly what the prior "fake midtraining" sweep did. Wrong.
* `FULL_STATE`: keeps the opt_state (so the schedule's `count` is preserved at, e.g., 30,000) **but resets `state.step` to 0**. Levanter then trains until `state.step >= num_train_steps`. If we set `num_train_steps = 47,064` (the pretrain target), training runs for **47,064 NEW steps** with `opt_state.count` going 30,000 → 77,064. Past `count = 47,063` the schedule output clamps to `min_lr_ratio × peak = 0`, so the last 30,000 of those 47,064 steps train at zero LR — a 2.7× compute waste. If instead we set `num_train_steps = 17,064` (just the remaining tail), the schedule is rebuilt with `decay_start = 13,651`, but `opt_state.count = 30,000` is already past the end — schedule clamps to `min_lr` immediately. **This is the same flat-LR bug that hit us on 2026-04-23.**

**Correct mechanism: use the natural resume path.** When Levanter starts up, it calls `latest_checkpoint_path(output_path)` (the *output* directory, not `initialize_from_checkpoint_path`). If that finds a checkpoint, the trainer loads it the standard way: `state.step` is restored from the checkpoint (NOT reset), opt_state is restored, training continues from where it left off until `state.step >= num_train_steps`.

So the historical recipe was (it mechanically worked for the wrong-base 1e20
cell, but do not reuse that 1e20 source):

1. Pre-stage the pretrain checkpoint into our run's own `output_path`:
   ```
   gs://marin-us-east5/checkpoints/<our-cell-name>/checkpoints/step-40000/
   ```
2. Launch with `output_path = gs://marin-us-east5/checkpoints/<our-cell-name>` and **no** `initialize_from_checkpoint_path`.
3. Set `num_train_steps = 47064` (the original pretrain target) and the AdamH config built by the heuristic for `(BS=128, T=24.67e9)`.
4. Trainer finds `step-40000`, restores `state.step = 40000` and `opt_state.count = 40000`, and trains for 47064 − 40000 = 7064 more steps.
5. Schedule built with `num_train_steps=47064` rebuilds the WSD profile correctly: stable in [4707, 37651), decay in [37651, 47063]. Resume at count=40000 lands inside decay at LR ≈ 0.75·peak; LR walks down to 0 over the remaining 7,064 steps.

This mechanism is **identical to a normal preemption recovery** — Levanter doesn't even know the checkpoint started life in another directory.

### Data loader behaviour on resume

The data loader is built fresh from `config.data` every job startup (`train_lm.py:191` — `config.data.train_set(...)`). Resuming from a checkpoint does NOT serialize/restore the dataset config. So changing `LMMixtureDatasetConfig` between pretrain and our run gives us a fresh data stream from the new mixture, deterministically seeded by `data_seed` (or `trainer.seed`). No hack needed — just point `data` at the math mixture.

The only caveat is that the per-component Feistel block-shuffle is seeded by the cache content + the data seed; a different mixture reads a different cache and produces a different stream. That is what we want.

---

## 6. Pre-staging plan — copy pretrain checkpoint into per-cell output paths

### 6.1 Region choice

The pretrain checkpoints live in `us-central2` (`gs://marin-us-central2/...`). v5p TPUs live in `us-central1` and `us-east5`. We pick **us-east5** as the home for the new runs because:
* 1e22 base was already copied to `us-east5` for the prior sweep (precedent + we know it works).
* `us-east5` v5p-256 capacity has been comparable to `us-central1` historically.
* Keeps the run-region pin from `[experiments] Pin Delphi midtraining jobs by region` happy.

If `us-east5` v5p capacity is wiped (cf. 2026-05-04 GCP eviction in the main logbook), fall back to `us-central1` and stage the checkpoints there instead. The plan is region-agnostic; just keep `output_path` and `staging_region` consistent.

### 6.2 Staging strategy

> **WRONG 1e20 BASE — do not run the 1e20 stanza below.** The first copy command
> is preserved as historical context only. A corrected 3e20/v6 true-midtraining
> plan needs verified native checkpoint step and optimizer-state metadata first.

Two-phase to minimise cross-region egress (which is the dominant cost — see `AGENTS.md`):

**Phase A: cross-region copy once per scale** (3 transfers total, us-central2 → us-east5)

```bash
# WRONG 1e20 BASE - do not run; historical record only.
gcloud storage cp -r \
  gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/checkpoints/step-40000 \
  gs://marin-us-east5/midtrain-bases/delphi-1e20-iso-d2048-L21/step-40000

gcloud storage cp -r \
  gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/checkpoints/step-20000 \
  gs://marin-us-east5/midtrain-bases/delphi-1e21-v5-019021/step-20000

gcloud storage cp -r \
  gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-30000 \
  gs://marin-us-east5/midtrain-bases/delphi-1e22-v5-025b0e/step-30000
```

Estimated sizes (Levanter TensorStore checkpoint = roughly 12–16 bytes/param for fp32 weights + opt_state):
* 1e20: 1.9 B params × ~14 B/param ≈ **27 GB** **[WRONG base — historical only]**
* 1e21: 3.4 B params × ~14 B/param ≈ **48 GB**
* 1e22: 9.7 B params × ~14 B/param ≈ **136 GB**

Cross-region GCS egress is $0.02/GB inside the same continent (us-central2 → us-east5). Total Phase A: (27 + 48 + 136) × $0.02 ≈ **$4.22**. Round up to ~$10 with overhead. Acceptable.

**Phase B: in-region fan-out into 9 cell-specific output_paths** (cheap, ~0 egress)

For each of the 9 (scale × mix) cells, copy the staged base checkpoint into the cell's own `output_path/checkpoints/step-X/`:

```bash
# WRONG 1e20 BASE - historical example only; do not run.
SRC=gs://marin-us-east5/midtrain-bases/delphi-1e20-iso-d2048-L21/step-40000
DST=gs://marin-us-east5/checkpoints/delphi-true-1e20-p33m67-step40000/checkpoints/step-40000
gcloud storage cp -r "$SRC" "$DST"
```

In-region copies are billed only for operation count, ~$0.005 per 1k operations — negligible at this scale.

The `<HASH>` is the marin executor StepSpec hash assigned at `executor_main` time. We will need to compute this *first* (or use `MIDTRAIN_RESUME_OUTPUT_PATH` style naming where we set the path explicitly), THEN do the Phase B copy. See § 7.4 for the launch ordering.

### 6.3 Checkpoint integrity validation

After each `gcloud storage cp -r` completes, verify the destination has `manifest.ocdbt`, `metadata.json`, and the `d/` keystore:

```bash
gcloud storage ls "$DST/" | grep -E '(manifest\.ocdbt|metadata\.json|d/)'
```

All three must be present. The TensorStore restore will silently NaN if `metadata.json` is missing or shape-drifted. Compare `metadata.json.step` field against the expected step number.

---

## 7. New experiment file — design

### 7.1 File path

`experiments/exp_delphi_true_midtrain.py` (new, ~250 lines). Keep `exp_delphi_math_10b_midtrain.py` untouched — the prior sweep is the comparison baseline and its outputs are already useful.

### 7.2 Why a new file rather than an env-var on the existing one

The existing experiment is built around the K=0.20 + LR-factor + fresh-warmup mental model and has many guards (LR factor validation, midtrain budget vs token override, MODEL_ONLY hard-coded, `_enforce_run_id` derived from a particular run-name template). Forcing it to also handle "natural resume from absolute-step pretrain checkpoint" via env-vars would re-introduce the kind of hidden-state coupling that caused the cron-driven resume-namespace incident. A clean new file with its own assumptions is safer.

### 7.3 Skeleton

> **WRONG 1e20 BASE — skeleton is historical.** The checked-in
> `experiments/exp_delphi_true_midtrain.py` has since been changed to expose
> only `1e21` and `1e22`; do not restore the 1e20 block below.

```python
"""True midtraining: resume Delphi pretrains at start-of-decay; only data swaps.

For each (scale × mix) cell, this script:

* Loads the original AdamH config from completed_adamh_heuristic.build_optimizer_config
  (peak LR, beta2, epsilon — verbatim from pretrain).
* Sets num_train_steps to the original pretrain count (so warmup=10%/decay=20%
  windows match the original WSD schedule).
* Uses the data mixture from experiments/midtraining_mixes.py (math + pretrain replay).
* Does NOT pass initialize_from_checkpoint_path — Levanter resumes from
  output_path/checkpoints/step-X via the natural recovery path. The pretrain
  checkpoint is pre-staged into output_path before launch by an external
  gcloud cp (see § 6).

Launch one cell at a time. Required env vars per launch:
  TRUE_MIDTRAIN_SELECT_SCALE  ∈ {1e21, 1e22}  # 1e20 in this historical skeleton was wrong
  TRUE_MIDTRAIN_SELECT_MIX    ∈ {p33m67, p50m50, p67m33}

Optional:
  TRUE_MIDTRAIN_RESUME_STEP_OVERRIDE  (default per-scale; for ad-hoc retries)
"""

import os
from dataclasses import dataclass
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import CheckpointerConfig
from levanter.main import train_lm
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

from experiments.midtraining_mixes import (
    PRETRAIN_33P_MATH_67P_HIGHQUALITY_NEMO_MATH_NAME,
    PRETRAIN_50P_MATH_50P_HIGHQUALITY_NEMO_MATH_NAME,
    PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME,
    midtraining_mix_by_name,
    log_partition_summary,
)
from experiments.midtrain_data_safety import assert_val_train_disjoint
from experiments.scaling_law_sweeps.completed_adamh import completed_adamh_heuristic, SEQ_LEN

# ----------------------------------------------------------------------------
# Per-scale pretrain definitions — MUST match the actual pretrain configs
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class PretrainSpec:
    scale_tag: str
    pretrain_run_name: str          # for documentation only
    pretrain_num_train_steps: int   # N — used to rebuild the WSD schedule exactly
    pretrain_tokens: int            # T — used by the heuristic
    batch_size: int                 # B — used by the heuristic
    resume_from_step: int           # the staged checkpoint step
    staged_base_path: str           # where Phase A staged the pretrain ckpt
    tpu_type: str                   # the slice we'll use for midtraining
    tensor_parallel_size: int       # match the pretrain TP

PRETRAINS: dict[str, PretrainSpec] = {
    # WRONG 1e20 BASE - historical only. Do not restore this entry.
    "1e20": PretrainSpec(
        scale_tag="1e20",
        pretrain_run_name="isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5",
        pretrain_num_train_steps=47_064,
        pretrain_tokens=24_672_337_920,        # 47064 × 128 × 4096
        batch_size=128,
        resume_from_step=40_000,                # 25% into the cooldown phase
        staged_base_path="gs://marin-us-east5/midtrain-bases/delphi-1e20-iso-d2048-L21",
        tpu_type="v5p-32",
        tensor_parallel_size=1,
    ),
    "1e21": PretrainSpec(
        scale_tag="1e21",
        pretrain_run_name="adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021",
        pretrain_num_train_steps=22_057,
        pretrain_tokens=46_267_858_944,        # 22057 × 512 × 4096
        batch_size=512,
        resume_from_step=20_000,                # 53% into the cooldown phase
        staged_base_path="gs://marin-us-east5/midtrain-bases/delphi-1e21-v5-019021",
        tpu_type="v5p-64",
        tensor_parallel_size=1,
    ),
    "1e22": PretrainSpec(
        scale_tag="1e22",
        pretrain_run_name="adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e",
        pretrain_num_train_steps=38_235,
        pretrain_tokens=160_362_270_720,       # 38235 × 1024 × 4096
        batch_size=1024,
        resume_from_step=30_000,                # 588 steps before decay starts (only viable closest-to-cooldown ckpt)
        staged_base_path="gs://marin-us-east5/midtrain-bases/delphi-1e22-v5-025b0e",
        tpu_type="v5p-256",                    # native; v5p-128/64 also OK with grad accum
        tensor_parallel_size=2,
    ),
}

MIXES = {
    "p33m67": PRETRAIN_33P_MATH_67P_HIGHQUALITY_NEMO_MATH_NAME,
    "p50m50": PRETRAIN_50P_MATH_50P_HIGHQUALITY_NEMO_MATH_NAME,
    "p67m33": PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME,
}

def build_cell(scale_tag: str, mix_tag: str) -> ExecutorStep:
    spec = PRETRAINS[scale_tag]
    mix_name = MIXES[mix_tag]

    optimizer = completed_adamh_heuristic.build_optimizer_config(
        batch_size=spec.batch_size, tokens=spec.pretrain_tokens,
    )
    # Verify the schedule we're about to use matches what the pretrain used.
    assert optimizer.warmup == 0.1, f"pretrain heuristic drifted: warmup={optimizer.warmup}"
    assert optimizer.decay == 0.2, f"pretrain heuristic drifted: decay={optimizer.decay}"
    assert optimizer.min_lr_ratio == 0.0, f"pretrain heuristic drifted: min_lr_ratio={optimizer.min_lr_ratio}"

    mix_spec = midtraining_mix_by_name(mix_name)              # MidtrainMixSpec
    data_config = mix_spec.build_lm_data_config(...)          # see signature in midtraining_mixes.py

    run_name = f"delphi-true-{scale_tag}-{mix_tag}-step{spec.resume_from_step}"

    inner_config = train_lm.TrainLmConfig(
        data=data_config,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                entity="marin-community",
                project="delphi-midtraining",
                tags=[
                    "true-midtraining",
                    f"scale={scale_tag}",
                    f"mix={mix_tag}",
                    f"resume_step={spec.resume_from_step}",
                    f"pretrain={spec.pretrain_run_name}",
                ],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=spec.batch_size,
            per_device_parallelism=-1,
            num_train_steps=spec.pretrain_num_train_steps,        # ← CRITICAL: original N
            steps_per_eval=200,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=2000)],                         # tighter cadence; runs are short
            ),
            mesh=MeshConfig(
                axes={"data": -1, "replica": 1, "model": spec.tensor_parallel_size},
                compute_mapping={
                    "token":        (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                },
            ),
            seed=0,
            allow_nondivisible_batch_size=True,
        ),
        train_seq_len=SEQ_LEN,
        model=completed_adamh_heuristic._build_model_config(
            hidden_size={"1e21": 2560, "1e22": 3840}[scale_tag],  # historical 1e20 entry was wrong
            seq_len=SEQ_LEN,
        ),
        optimizer=optimizer,
        # NOTE: no initialize_from_checkpoint_path — pretrain ckpt is pre-staged
        # into output_path; Levanter resumes via the natural recovery path.
        z_loss_weight=optimizer.z_loss_weight if hasattr(optimizer, 'z_loss_weight') else 1.0e-7,
        hf_save_steps=2000,
    )

    pod_config = TrainLmOnPodConfig(
        train_config=inner_config,
        resources=ResourceConfig.with_tpu(spec.tpu_type),
        output_path=this_output_path(),
    )
    return ExecutorStep(name=run_name, fn=run_levanter_train_lm, config=pod_config)

if __name__ == "__main__":
    scale = os.environ.get("TRUE_MIDTRAIN_SELECT_SCALE")
    mix   = os.environ.get("TRUE_MIDTRAIN_SELECT_MIX")
    if scale is None or mix is None:
        # No selection → enumerate all 9 for dry-run inspection only.
        steps = [build_cell(s, m) for s in PRETRAINS for m in MIXES]
        print(f"Built {len(steps)} cells (dry-run; do not executor_main on this list).")
        for s in steps: print(" ", s.name)
    else:
        if scale not in PRETRAINS or mix not in MIXES:
            raise ValueError(f"bad selectors: scale={scale}, mix={mix}")
        cell = build_cell(scale, mix)
        executor_main(steps=[cell])
```

### 7.4 Critical launch ordering

Because `output_path` includes the executor StepSpec hash (which depends on the `TrainLmOnPodConfig` contents), we must:

1. **Pre-compute the hash by dry-running the script with the env-vars set:**
   ```bash
   # WRONG 1e20 BASE - historical only; do not run.
   TRUE_MIDTRAIN_SELECT_SCALE=1e20 TRUE_MIDTRAIN_SELECT_MIX=p33m67 \
     uv run python -c "
   import experiments.exp_delphi_true_midtrain as m, os
   os.environ.setdefault('TRUE_MIDTRAIN_SELECT_SCALE', '1e20')
   os.environ.setdefault('TRUE_MIDTRAIN_SELECT_MIX',   'p33m67')
   step = m.build_cell('1e20', 'p33m67')
   print(step.with_output_path)        # use the marin API to expose the hash
   "
   ```
2. **Phase B copy:** stage the pretrain checkpoint into the resolved output_path:
   ```bash
   gcloud storage cp -r \
     gs://marin-us-east5/midtrain-bases/delphi-1e20-iso-d2048-L21/step-30000 \
     gs://marin-us-east5/checkpoints/<resolved-hash>/checkpoints/step-30000
   ```
3. **Verify:**
   ```bash
   gcloud storage ls gs://marin-us-east5/checkpoints/<resolved-hash>/checkpoints/step-30000/ \
     | grep -E '(manifest\.ocdbt|metadata\.json|d/)'
   ```
4. **Launch:**
   ```bash
   # WRONG 1e20 BASE - historical only; do not run.
   uv run iris --cluster=marin job run --cpu 1 --memory 3GB --disk 9GB \
     --region us-east5 --priority interactive --no-wait \
     --job-name delphi-true-1e20-p33m67-$(date -u +%Y%m%d-%H%M%S) \
     -e WANDB_API_KEY "$WANDB_API_KEY" \
     -e MIDTRAIN_MAX_TASK_FAILURES 100 \
     -e TRUE_MIDTRAIN_SELECT_SCALE 1e20 \
     -e TRUE_MIDTRAIN_SELECT_MIX p33m67 \
     -- python experiments/exp_delphi_true_midtrain.py
   ```
5. **Verify in startup logs:** `Resuming training from step 30000` should appear within the first 60 s after the train_lm child boots. If `Starting from scratch` appears instead, the staged checkpoint path is wrong — kill the job before any wasted compute.

The "compute hash → stage → verify → launch" cycle is per-cell. 9 cells = 9 invocations. Tedious but mechanical; can be wrapped in a small `scripts/launch_true_midtrain_cell.sh` helper if needed.

### 7.5 Alternative — explicit `output_path` override

To avoid the hash-derivation dance, set an explicit name via `with_output_path`:

```python
output_path=f"gs://marin-us-east5/checkpoints/delphi-true-{scale_tag}-{mix_tag}-step{spec.resume_from_step}",
```

This makes the pre-staging path predictable: the directory is fixed by `(scale, mix, step)` triple, no hash. Trade-off: we lose the marin executor's automatic input-fingerprinting for cache-invalidation, but for this 9-cell sweep cache invalidation isn't a concern (no upstream data steps that change). **Recommend this approach** to keep the staging script simple.

---

## 8. Cell layout — narrowed to p33m67 only (user-confirmed 2026-05-09)

User has narrowed the sweep to **p33m67 only** (33% pretrain replay + 67% math). The script still supports the other two mixes (`p50m50`, `p67m33`) via `TRUE_MIDTRAIN_SELECT_MIX`; we just don't launch them. Easy to revisit later without code changes.

> **WRONG 1e20 BASE — historical only.** Any 1e20 throughput, cost, launch-name,
> or checkpoint row in this section is derived from the bad v5 isoflop
> substitution and must not be used for a new true-midtraining launch.

| # | Scale | Mix | Run name (`true-midtrain-…`) | TPU (default) | Resume from |
|---:|---|---|---|---|---|
| 1 | 1e20 **[WRONG — historical only]** | p33m67 | `true-midtrain-1e20-p33m67-step40000` | v5p-32 | `step-40000` of `isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5` **[WRONG]** |
| 2 | 1e21 | p33m67 | `true-midtrain-1e21-p33m67-step20000` | v5p-64 | `step-20000` of `adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021` |
| 3 | 1e22 | p33m67 | `true-midtrain-1e22-p33m67-step30000` | v5p-256 | `step-30000` of `adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e` |

### Wall-clock estimates — per scale × per TPU type

**Throughput basis — anchor measurements from `.agents/logbooks/midtraining_delphi.md`:**

| Anchor | TPU | BS | Step time | Logbook ref |
|---|---|---:|---:|---|
| 1e20 base (1.9 B), BS=512 | v5p-64 | 512 | 4.4–4.5 s | line 661, 956 — v10 |
| 1e20 base (1.9 B), BS=512 | v5p-32 | 512 | ~7.9 s | derived from K=0.20 sweep wall-time |
| 1e21 base (3.4 B), BS=512 | v5p-64 | 512 | ~7.3 s | K=0.20 sweep ~9 h for 4,411 steps |
| 1e21 base (3.4 B), BS=512 | v5p-256 | 512 | **1.6–1.8 s** ✅ | line 2888-3214, multiple monitor ticks (lr=0.67/0.83/0.5 v5p-256 pilot) |
| 1e22 base (9.7 B), BS=1024 | v5p-64 | 1024 | **35 s** ✅ (with grad-accum=4) | line 8549, May-9 resume |
| 1e22 base (9.7 B), BS=1024 | v5p-512 | 1024 | **4.1–4.4 s** ✅ | line 2938-3175, multiple monitor ticks (lr=0.67/0.83 v5p-512 sweep) |

Throughput scales close to linearly with chip count for these (large-enough) per-device batches. Anchor → derive other points by chip-count ratio. Per-device-batch <4 starts to lose efficiency to per-step overhead; flag those rows.

---

#### **1e20 cell** — 7,064 steps, BS=128, 3.70 B tokens

Per-device-batch shrinks as we add chips, so larger TPUs hit overhead floor.

| TPU | Chips | Per-dev-batch | Step time | **Wall-clock (steady)** | + 50% buffer | Chip-hours | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| v5p-32 | 16 | 8 | ~2.5 s | **4.9 h** | ~7.4 h | 78 | default; healthy per-device-batch |
| v5p-64 | 32 | 4 | ~1.5 s | **2.9 h** | ~4.4 h | 94 | per-device-batch borderline; some overhead |
| v5p-128 | 64 | 2 | ~1.2 s | **2.4 h** | ~3.6 h | 152 | overhead-bound; diminishing returns |
| v5p-256 | 128 | 1 | ~1.0 s | **2.0 h** | ~3.0 h | 256 | overhead-dominated; not recommended |

**Recommended for 1e20: v5p-32** (cheapest in chip-hours, healthy per-device-batch). Speedup from v5p-32 → v5p-64 is roughly 2× wall-clock for 1.2× chip-hours — worth it if v5p-32 is contended.

---

#### **1e21 cell** — 2,057 steps, BS=512, 4.31 B tokens

Linear scaling holds well across TPU sizes (v5p-256 anchor confirms).

| TPU | Chips | Per-dev-batch | Step time | **Wall-clock (steady)** | + 50% buffer | Chip-hours | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| v5p-64 | 32 | 16 | ~7.3 s | **4.2 h** | ~6.3 h | 134 | default; matches K=0.20 sweep measurement |
| v5p-128 | 64 | 8 | ~3.6 s | **2.1 h** | ~3.1 h | 134 | linear scaling; same chip-hours |
| v5p-256 | 128 | 4 | **1.7 s** ✅ | **0.97 h** | ~1.5 h | 124 | measured directly, 4.3× faster than v5p-64 |
| v5p-512 | 256 | 2 | ~0.9 s | **0.51 h** | ~0.8 h | 131 | per-device-batch=2 starts losing efficiency |

**Recommended for 1e21: v5p-128 or v5p-256** — chip-hours are flat (linear scaling), so pick whichever is available. v5p-256 finishes in <1.5 h per cell with buffer.

---

#### **1e22 cell** — 8,235 steps, BS=1024, 34.54 B tokens (the dominant cost)

Linear scaling holds across the full v5p ladder (both v5p-64 and v5p-512 are measured anchors).

| TPU | Chips | Per-dev-batch | Step time | **Wall-clock (steady)** | + 50% buffer | Chip-hours | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| v5p-64 | 32 | 32 | **35 s** ✅ (grad-accum=4) | **80.1 h** | ~120 h | 2,563 | painful; only as a fallback |
| v5p-128 | 64 | 16 | ~17.5 s | **40.0 h** | ~60 h | 2,560 | half-speed of v5p-256; same chip-hours |
| v5p-256 | 128 | 8 | ~8.75 s | **20.0 h** | ~30 h | 2,560 | default; same chip-hours as v5p-128 |
| v5p-512 | 256 | 4 | **4.2 s** ✅ | **9.6 h** | ~14.4 h | 2,458 | measured; ~2× faster than v5p-256, slightly cheaper chip-hours |

**Recommended for 1e22: v5p-512 if available** (≈10 h vs ≈20 h on v5p-256, no chip-hour penalty). Falls back to v5p-256 cleanly. Avoid v5p-64 unless it's the only thing left — wall-time is 4× worse.

---

### Sweep-level wall-clock and chip-hour roll-up — 3 cells (p33m67 only)

Three preset profiles depending on what's available and how much you care about wall-time:

**Profile A — fast (largest TPU per scale)**

| Scale | TPU | Per-cell + buffer | Per-scale chip-hours |
|---|---|---:|---:|
| 1e20 | v5p-64 | 4.4 h | 94 |
| 1e21 | v5p-256 | 1.5 h | 124 |
| 1e22 | v5p-512 | 14.4 h | 2,458 |
| | | **Critical path: ~14.4 h (1e22)** | **Total: 2,676 chip-h** |

**Profile B — default (per-scale defaults, recommended)**

| Scale | TPU | Per-cell + buffer | Per-scale chip-hours |
|---|---|---:|---:|
| 1e20 | v5p-32 | 7.4 h | 78 |
| 1e21 | v5p-64 | 6.3 h | 134 |
| 1e22 | v5p-256 | 30 h | 2,560 |
| | | **Critical path: ~30 h (1e22)** | **Total: 2,772 chip-h** |

**Profile C — cheap-and-slow (smallest viable TPU per scale)**

| Scale | TPU | Per-cell + buffer | Per-scale chip-hours |
|---|---|---:|---:|
| 1e20 | v5p-32 | 7.4 h | 78 |
| 1e21 | v5p-64 | 6.3 h | 134 |
| 1e22 | v5p-128 | 60 h | 2,560 |
| | | **Critical path: ~60 h (1e22)** | **Total: 2,772 chip-h** |

Notes on the table:
* Critical path = max wall-clock across cells, assuming all 3 cells launch concurrently on independent TPU slices. The 1e22 cell dominates the wall-clock; 1e20 and 1e21 finish in well under a quarter of the time the 1e22 cell takes.
* Add ~30 min/cell for iris coordinator + TPU acquisition + JAX init + checkpoint load. Eval/HF-export pauses (every ~750-2,000 steps) add 5–15 min total per cell.
* Chip-hours assume "v5p-N" means N cores = N/2 chips (Marin/Iris convention). Cost: ~$4-5/chip-hour on-demand, ~30% of that on preemptible.
* For comparison: the prior 36-cell K=0.20 sweep cost ~30,000 chip-hours; this 3-cell true-midtraining sweep is ~9% of that. The 1e22 cell dominates (~92% of chip-hours regardless of profile).

---

## 9. Optimizer config — verbatim from heuristic, no LR factor knob

Every cell uses:

```python
optimizer = completed_adamh_heuristic.build_optimizer_config(
    batch_size=spec.batch_size,
    tokens=spec.pretrain_tokens,
)
```

This produces, per scale (computed from the `(B, T)` of each pretrain):

| Scale | peak `learning_rate` | peak `adam_lr` | `beta2` | `epsilon` |
|---|---:|---:|---:|---:|
| 1e20 (B=128, T=24.67e9) | 4.483e-3 | 7.382e-5 | 0.99980 | 4.11e-8 |
| 1e21 (B=512, T=46.27e9) | 7.425e-3 | 4.314e-4 | 0.99920 | 2.81e-8 |
| 1e22 (B=1024, T=160.37e9) | ~7.23e-3 | ~3.28e-4 | ~0.99840 | ~3.70e-8 |

(The 1e20 and 1e21 numbers are confirmed against the W&B configs in `.agents/logbooks/midtraining_delphi.md` § "Base models". 1e22 numbers are computed from the heuristic; **verify against W&B before launching the 1e22 cells** — `wandb.Api().run("marin-community/marin/<run-name>").config` to confirm.)

The schedule is added on top: `warmup=0.1`, `decay=0.2`, `min_lr_ratio=0.0`, `lr_schedule="linear"`. With `num_train_steps` set to the original pretrain count, the schedule replays exactly the pretrain's WSD profile from the resume step onward.

**Assertions in the experiment file** (already in the skeleton above):
```python
assert optimizer.warmup == 0.1
assert optimizer.decay == 0.2
assert optimizer.min_lr_ratio == 0.0
```

These trip if the heuristic ever drifts from the documented WSD. Refuse to launch if any fails.

---

## 10. Data mixture — reuse `experiments/midtraining_mixes.py`

The three mixes from the prior sweep are exactly the right candidates:

* `PRETRAIN_33P_MATH_67P_HIGHQUALITY_NEMO_MATH_NAME` ("p33m67")
* `PRETRAIN_50P_MATH_50P_HIGHQUALITY_NEMO_MATH_NAME` ("p50m50")
* `PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME` ("p67m33")

Each blends:
* **Pretrain replay** = `nemotron_mix` (Nemotron-CC HQ + starcoderdata 25% + proofpile_2 5.5%) — the *same* mix the Delphi pretrains used. Continuity guarantee: the model has seen this distribution before.
* **Math** = `nemotron_cc_math_v1/4plus` — the BOS-correct Phi-4-cleaned, non-Qwen math corpus. ~52 B tokens; far above the largest cell's 35 B math token budget, so no epoch repeats.

The per-component held-out validation slice (12,500 sequences ≈ 51 M tokens carved from the math component via Levanter's `num_validation_sequences` + Feistel split) is automatic via `MidtrainMixSpec.build_lm_data_config(...)`. Same plumbing as the prior sweep.

**No code changes needed in `midtraining_mixes.py`** — we consume it as-is.

---

## 11. Safety guards

Reuse the four-layer safety stack from the prior sweep:

1. **Layer 1 (config-time):** `MidtrainMixSpec.__post_init__` → `validate_midtrain_spec` checks weights sum to 1.0, no name collisions, val_sequences in (0, ∞).
2. **Layer 2 (built-config):** `assert_lm_data_config_safe` confirms the resulting `LmDataConfig` has `train_weights` summing to 1.0, val carve-out names exist, `shuffle_before_trainval_split=True`, training shuffle is `BlockShuffleConfig`.
3. **Layer 3 (launch-time):** `assert_val_train_disjoint` hash-samples train + ALL of val, asserts empty intersection. Catches Levanter slice/shuffle refactors.
4. **Layer 4 (run startup):** `log_partition_summary` logs val sequence counts and weights to W&B.

Plus **new guards specific to true-midtraining:**

5. **Heuristic-drift guard:** the three `assert optimizer.{warmup,decay,min_lr_ratio} == ...` checks in § 9. Refuse to launch if any fails.
6. **Resume-step floor guard:** before launch, the script must verify the staged checkpoint actually exists at the expected step:
   ```python
   from levanter.checkpoint import discover_latest_checkpoint
   discovered = discover_latest_checkpoint(output_path)
   assert discovered is not None, "Pre-staging did not complete; refusing to launch"
   assert discovered.step == spec.resume_from_step, (
       f"Found checkpoint at step {discovered.step}; expected {spec.resume_from_step}"
   )
   ```
7. **`num_train_steps` matches pretrain:** documented invariant. The script prints both the heuristic-computed `optimizer` and the configured `num_train_steps` at startup, so a human can eyeball-confirm.
8. **Wandb run-id from the run name** (not from a hash): with the explicit `output_path` strategy from § 7.5, `run_id = basename(output_path)` and `_enforce_run_id` passes naturally. No `MIDTRAIN_RESUME_OUTPUT_PATH` mechanics needed.

---

## 12. Open questions for the user — please confirm before I do anything

1. ~~**Checkpoint choice for 1e20 and 1e21.**~~ **DECIDED 2026-05-09:** start at-or-after decay where possible.
   * 1e20: `step-40000` (85.0%, 25% into cooldown, LR=0.75·peak; 7,064 steps, **3.70 B tokens / cell**)
   * 1e21: `step-20000` (90.7%, 53% into cooldown, LR=0.466·peak; 2,057 steps, **4.31 B tokens / cell**)
   * 1e22: `step-30000` (78.5%, 588 steps before decay starts, LR=peak; 8,235 steps, **34.54 B tokens / cell**)

   Mixed pre/post pattern is intentional — 1e22 has no post-decay checkpoint near cooldown-start, so its closest option is the just-pre-decay one.

2. **Region.** Default to `us-east5` for staging + training. Switch to `us-central1` if v5p capacity is wiped again. Confirm OK.

3. **Mixes.** Reuse the same three mixes as the prior sweep (`p33m67`, `p50m50`, `p67m33`). Confirm OK or specify a different set.

4. **Cell count.** 9 cells (3 mix × 3 scale). No LR axis, since we're playing out the pretrain schedule unchanged. If the user wants an additional knob — e.g., 0.5× / 1.0× / 1.5× LR multiplier on top of the pretrain peak as a sensitivity test — we can add it, but recommend NOT to in the first pass.

5. **Pretrain mix continuity.** `nemotron_mix` (Nemotron-CC HQ + starcoderdata 25% + proofpile_2 5.5%) is the same mix the Delphi pretrains used. The user's instruction said "nemotron cc for pretraining" — confirming `nemotron_mix` (which is the Delphi/Mantis pretrain mix; it's mostly Nemotron-CC HQ but does include starcoderdata + proofpile_2) is the intended interpretation. If the user wants pure Nemotron-CC HQ with no starcoderdata/proofpile_2 in the replay, that's a different MidtrainMixSpec and we'd need to extend `experiments/midtraining_mixes.py`.

6. **Cost approval.** Phase A cross-region copy is ~$10 in GCS egress. Acceptable per the user's prior "forget cross-region costs" approval, but flagging explicitly.

7. **Compute capacity.** ~2 days wall-clock for the full sweep at full parallelism. If GCP wipes v5p-256 again mid-1e22, the recipe can drop to v5p-128 or v5p-64 with grad accum (`per_device_parallelism=4`), same as the prior sweep.

8. **Branch / commit hygiene.** This is a new experiment file plus a small staging script. Plan to:
   * commit `.agents/logbooks/true_midtraining.md` (this file) immediately
   * commit `experiments/exp_delphi_true_midtrain.py` once the user approves the design
   * commit `scripts/stage_true_midtrain_checkpoints.sh` (the Phase A + B helper) once written
   * NOT push until the user signs off on the cell list

---

## 13. Launch plan — every guard lifted from `exp_delphi_math_10b_midtrain.py`

The prior K=0.20 sweep wasted multiple cycles of compute on broken resumes (cron-driven hash drift, MIDTRAIN_OUTPUT_PATH_OVERRIDE namespace splits, post-eviction wrong-namespace recoveries). Every guard the prior file accreted as a result of those incidents must be present here, plus extra guards specific to TRUE midtraining (where every cell **is** a resume from day one — there is no "fresh start" path).

### 13.1 WandB run-name convention (MANDATORY)

Every run name and every wandb display name **must start with `true-midtrain-`**. Concretely:

```
true-midtrain-{scale}-{mix}-step{resume_step}
```

Examples:
* `true-midtrain-1e20-p33m67-step40000`
* `true-midtrain-1e21-p67m33-step20000`
* `true-midtrain-1e22-p50m50-step30000`

Constraints:
* `{scale}` ∈ `{1e20, 1e21, 1e22}` — the `_BASE_OUTPUT_TAGS` short tag from the original file.
* `{mix}` ∈ `{p33m67, p50m50, p67m33}` — the `_MIX_OUTPUT_TAGS` short tag (no `math` since every cell uses a non-trivial pretrain replay).
* `{resume_step}` is the literal pretrain step number staged into `output_path`.
* Total length must stay under **64 characters** (W&B display-name cap; the original file enforces this in `_build_run_name`). Longest example above: `true-midtrain-1e20-p33m67-step40000` = 35 chars. Well under the limit.
* No suffix needed by default (no LR knob → no per-cell variant).

The run name is also the output_path basename and the wandb run id — so `_enforce_run_id` (`lib/marin/src/marin/training/training.py:190`) trivially passes.

W&B project: `marin-community/delphi-midtraining` (same as the prior sweep — keeps every Delphi-related run discoverable in one place).

W&B tags (mirrors the original tag block, plus a `true-midtraining` tag for filtering):
```python
tags=(
    "true-midtraining",                                 # ← discriminator vs. prior K=0.20 sweep
    "midtraining",
    f"base={base_tag}",                                 # e.g. "base=1e22-v5"
    f"midtraining_mix={mix_name}",
    "midtraining-mix",
    f"resume_step={spec.resume_from_step}",
    f"pretrain_run={spec.pretrain_run_name}",
    f"batch_size={batch_size}",
    f"seq_len={seq_len}",
    f"tpu_type={compute_config.tpu_type}",
    f"per_device_parallelism={compute_config.per_device_parallelism}",
    f"tensor_parallel_size={compute_config.tensor_parallel_size}",
    f"pretrain_tokens={base.pretrain_tokens}",
    f"num_train_steps={spec.pretrain_num_train_steps}",
    f"peak_lr={optimizer.learning_rate:.3e}",           # original pretrain peak; no LR factor
    f"adam_lr={optimizer.adam_lr:.3e}",
    "adamh",
    "delphi-midtrain",
)
```

### 13.2 Safety guards — ported verbatim from `exp_delphi_math_10b_midtrain.py`

These guards exist because each one stopped a real, observed failure mode in the prior sweep. The new file MUST carry the same guard logic, even where the failure mode seems unlikely now.

| # | Guard (origin file) | Why it matters here | Implementation sketch in `exp_delphi_true_midtrain.py` |
|---|---|---|---|
| G1 | **Mirror-staged checkpoints with explicit `budget_gb`.** `mirrored("...", budget_gb=N)` per base (30 / 50 / 150 GB). Wrapped via `MirroredValue`. | Cross-region staging needs explicit budget — relying on the default 10 GB limit silently OOMs the mirror cache for a 1e22 ckpt. Caused the v5p-256 pilot failure on 5/2. | Each `PretrainSpec` carries `staged_base_path: str | MirroredValue[str]`; default to `mirrored(<rel-path>, budget_gb=N)` per scale. |
| G2 | **Region pinning via `_midtrain_tpu_resources` + `MIDTRAIN_COORDINATOR_REGIONS = ("us-central1", "us-east5")`.** Coordinator can land in either; child `train_lm` resources are pinned to coordinator's resolved region. Zone-string normalization (`us-east5-a` → `us-east5`). | The 4/27 cross-region hash-drift incident landed because parent-retried in a different region than the child. The 5/4 GCP eviction recovery used the same guard. | Port `_normalize_region`, `_selected_train_region`, `_midtrain_tpu_resources` verbatim. Same env-var override `MIDTRAIN_TRAIN_REGION` (or `TRUE_MIDTRAIN_TRAIN_REGION` for clarity). |
| G3 | **Typed `V5PComputeConfig` allowlist per base.** `compute_config(tpu_type)` raises if the requested TPU isn't in the allowlist for that scale. | Submitting 1e22 to v5p-32 (no per-device-batch headroom) silently OOMs late in step. The allowlist enforces "this base + this TPU was tested." | Each `PretrainSpec` carries `v5p_compute: tuple[V5PComputeConfig, ...]`. Reuse the same per-scale allowlists from `BASES` in the original file (1e22 needs `per_device_parallelism=4` on every TPU; 1e20 needs `tensor_parallel_size=2` on v5p-512). |
| G4 | **Hyperparameters from W&B config, not heuristic recompute.** `peak_lr`, `peak_adam_lr`, `beta2`, `epsilon` hard-coded per base from each pretrain run's `wandb.config`. | Prevents drift between heuristic implementation and the actual values the pretrain weights were optimized against. CRITICAL for true midtraining: even tiny ε/β2 mismatch is a different optimizer than the pretrain. | Lift the four numeric fields verbatim from `BASES` in the original. Add an `assert` that compares them against `completed_adamh_heuristic.build_optimizer_config(BS, T)` for forward-compat sanity (logs a warning on mismatch but does not block). |
| G5 | **`MIDTRAIN_OUTPUT_PATH_OVERRIDE` is hard-rejected.** Raises ValueError at module import. | Cron-driven sweep on 5/2 used this env var and produced 12 wandb namespace fragmentations. Banned permanently. | Same hard-reject in the new file. Substitute the new env var name `TRUE_MIDTRAIN_RESUME_OUTPUT_PATH` for the recovery path. |
| G6 | **Resume identity is derived from one source of truth: `<env>_RESUME_OUTPUT_PATH`.** Run name = wandb id = output_path basename. `_resume_run_id_from_output_path` extracts the id; `_resume_identity_env_vars` builds `RUN_ID` / `WANDB_RUN_ID` / `WANDB_RESUME=allow`. | The hash-drift / wandb-fragmentation incidents always traced back to multiple sources of identity disagreeing. One source of truth = no possible disagreement. | Port `_resume_run_id_from_output_path`, `_resume_identity_env_vars`, `_with_resume_identity` verbatim. **For true midtraining, every cell uses this path** (every cell IS a resume from a pre-staged ckpt). |
| G7 | **`_validate_resume_output_path_matches_run`.** The resume path's basename must equal (or have as prefix) the run name this script would generate for the selected `(base, mix, lr)` triple. | Stops the "I gave you the wrong resume path" failure mode — e.g. trying to resume `1e21/p33m67` from a `1e21/p67m33` namespace. | Port and adapt to the new naming scheme. The resume-path basename must equal `true-midtrain-{scale}-{mix}-step{resume_step}`. |
| G8 | **`_verify_resume_checkpoint_namespace` pre-flight.** Scans permanent + temp checkpoint paths under `output_path`; finds `latest_checkpoint`; refuses to launch if `step < MIDTRAIN_EXPECT_RESUME_MIN_STEP`. | The 4/26 incidents (`p33m67/lr0.5` failed at preemption recovery; `p33m67/lr0.67` lost ~6,000 steps to namespace drift) were caught only post-hoc. Pre-flight closes that gap. | **For true midtraining this guard is mandatory on every launch**, not opt-in. Set `expected_min_step = spec.resume_from_step` (the pre-staged step). Refuse to launch if no checkpoint or step < expected. |
| G9 | **`MIDTRAIN_EXPECT_RESUME_MIN_STEP` requirement.** `_validate_resume_env_contract` raises unless `MIDTRAIN_ALLOW_EMPTY_RESUME=1` is also set. | Forces the operator to be explicit about what step they expect the resume to land on, instead of "whatever's there." | Same here — but `expected_min_step` is implicit (from `PretrainSpec.resume_from_step`), not env-var-driven. The env var becomes optional only as a launch-time override. |
| G10 | **Run-name length cap (64 chars) enforced at `_build_run_name`.** | W&B silently truncates long display names, splitting one logical run across two display names. Caused real confusion in the prior sweep. | Same enforcement. All current names are <40 chars (`true-midtrain-1e20-p33m67-step40000` = 35), but the check is cheap. |
| G11 | **Duplicate run-name detection.** `_build_runs` raises if it produces two runs with the same name. | Caught a real bug in `_build_runs` early in the prior sweep. | Same check. |
| G12 | **`reset_data_loader_on_init=True`.** Fresh data iterator from the new mixture (math+replay), not pretrain's iterator. | We're switching distributions; pretrain's iterator state is meaningless. | Same here. |
| G13 | **`STEPS_PER_EVAL=200`.** Frequent eval gives early-warning if a cell goes off the rails (loss exploding, NaN, mode collapse). | Caught the flat-LR bug within the first 200 steps of the 4/23 sweep. Same value here. | Identical. |
| G14 | **Permanent checkpoint cadence = `max(50, 0.10 × num_train_steps)`.** ~10 evenly-spaced rollback points per cell. HF export uses the same cadence. | Lets us recover from preemption without losing more than ~10% of the cell. | Identical. |
| G15 | **Pre-flight `assert_val_train_disjoint` against the math-only check config.** Layer 3 of the four-layer safety stack from `experiments/midtrain_data_safety.py`. | Catches Levanter slice/shuffle refactors that could cause val to leak into train. | Same call. The math-only check config (`full_highquality_nemo_math`) is shared across all cells; one launch-time check is sufficient. |
| G16 | **`log_partition_summary` at __main__.** Writes val sequence counts and mix weights to stdout/W&B. | Layer 4 of the safety stack — gives the operator a verifiable record of what mixture this run is actually using. | Identical. |
| G17 | **`MidtrainMixSpec` four-layer safety (Layers 1+2 at module import).** `validate_midtrain_spec` (post_init) + `assert_lm_data_config_safe` (build-time). | The mix-config validation that prevented broken weights / sums-not-1.0 / shuffle bugs in the prior sweep. | Reused as-is by importing `experiments.midtraining_mixes.midtraining_mix_by_name`. |

### 13.3 Guards specific to true midtraining (NEW)

| # | New guard | Why |
|---|---|---|
| N1 | **Schedule-fields heuristic-drift guard.** Refuse to launch if `completed_adamh_heuristic` returns anything other than `warmup=0.1, decay=0.2, min_lr_ratio=0.0`. | The whole recipe assumes the pretrain WSD profile. If somebody ever changes the heuristic, the assumption silently breaks. Hard-fail at module import. |
| N2 | **`num_train_steps` invariant.** `num_train_steps == spec.pretrain_num_train_steps` for the selected base. Refuse to launch otherwise. | Setting `num_train_steps` to anything other than the original pretrain target re-creates the 4/23 flat-LR bug (schedule built with wrong N, opt_state.count past end → clamps to min_lr). |
| N3 | **Resume-step-step floor invariant.** `pre-staged checkpoint's step == spec.resume_from_step` exactly. Refuse if the staged ckpt is at a different step than the spec says. | Catches "I staged the wrong file" before the cell trains a single step. The original guard G8 only checks `>=`; here we want exact match. |
| N4 | **Pre-staged checkpoint integrity.** Before launch, verify `manifest.ocdbt`, `metadata.json`, and `d/` keystore all exist under `staged_path/checkpoints/step-X/`. Verify `metadata.json` contains `step == spec.resume_from_step`. | TensorStore silently NaN-restores on missing arrays; we need to fail loudly before the run starts. |
| N5 | **No `initialize_from_checkpoint_path`.** Set this to `None` explicitly. The natural-resume path is the only resume mechanism. | The original file uses `initialize_from_checkpoint_path` for the prior sweep. For true midtraining we MUST not, because that branch resets `state.step=0` (which re-triggers the flat-LR bug). |
| N6 | **No LR factor.** `LR_FACTORS = (1.0,)`. Refuse if anyone tries to override via env var. | LR is fixed by the pretrain schedule continuation. Adding a multiplier would break "TRUE midtraining." |
| N7 | **WandB run-name discriminator.** Hard-assert `name.startswith("true-midtrain-")` before generating any ExecutorStep. | Per user instruction — every wandb run must be obviously distinguishable from the prior K=0.20 sweep. |

### 13.4 Per-cell launch sequence — narrowed to 3 cells (p33m67 only)

The 3 chosen cells (one per scale, all p33m67) each run through this sequence. With only one mix, Phase A staging and Phase B fan-out collapse: each scale's staged base only needs to fan out to a single cell output_path.

> **WRONG 1e20 BASE — historical only.** The 1e20 staging bullet below is not a
> valid launch instruction after the 2026-05-14 incident.

**Phase A — one-time per scale (3 invocations total)**

1. **Cross-region copy of pretrain ckpt to a `midtrain-bases` staging path.**
   ```bash
   gcloud storage cp -r \
     gs://marin-us-central2/<pretrain-run>/checkpoints/step-<resume_step> \
     gs://marin-us-east5/midtrain-bases/<scale-tag>/step-<resume_step>
   ```
   Per scale (~$10 total egress for all 3):
   - 1e20: `step-40000` of `isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5` (~27 GB)
     **[WRONG base — historical only; do not stage]**
   - 1e21: `step-20000` of `adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021` (~48 GB)
   - 1e22: `step-30000` of `adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e` (~136 GB)

2. **Verify integrity** (Phase A1 sanity):
   ```bash
   gcloud storage ls gs://marin-us-east5/midtrain-bases/<scale-tag>/step-<resume_step>/ | grep -E '(manifest\.ocdbt|metadata\.json|d/)'
   ```
   All three must be present per scale. Read `metadata.json.step` and confirm it equals the expected step.

**Phase B — per cell (3 invocations total, all `mix=p33m67`)**

3. **Compute the run name** (deterministic; no executor hashing involved):
   ```
   true-midtrain-{scale}-p33m67-step{resume_step}
   ```

4. **Compute the cell's `output_path`** (use the run name as the basename so `_enforce_run_id` passes):
   ```
   gs://marin-us-east5/checkpoints/true-midtrain-{scale}-p33m67-step{resume_step}
   ```

5. **In-region fan-out copy** of the staged base into the cell's own `output_path`:
   ```bash
   gcloud storage cp -r \
     gs://marin-us-east5/midtrain-bases/<scale-tag>/step-<resume_step> \
     gs://marin-us-east5/checkpoints/true-midtrain-<scale>-p33m67-step<resume_step>/checkpoints/step-<resume_step>
   ```
   In-region copies are ~free (operation count only). Total: 3 fan-outs (27 GB + 48 GB + 136 GB ≈ 211 GB total). Wall-time ~2–10 min per copy.

6. **Verify the cell's checkpoint integrity** (Phase B6 sanity):
   ```bash
   gcloud storage ls gs://marin-us-east5/checkpoints/true-midtrain-<scale>-<mix>-step<resume_step>/checkpoints/step-<resume_step>/ \
     | grep -E '(manifest\.ocdbt|metadata\.json|d/)'
   ```
   All three must be present.

7. **Pre-flight the experiment script** (dry-run, no executor_main):
   ```bash
   TRUE_MIDTRAIN_SELECT_SCALE=<scale> TRUE_MIDTRAIN_SELECT_MIX=<mix> \
   TRUE_MIDTRAIN_RESUME_OUTPUT_PATH=gs://marin-us-east5/checkpoints/true-midtrain-<scale>-<mix>-step<resume_step> \
     uv run python experiments/exp_delphi_true_midtrain.py --dry-run
   ```
   Expected output: prints the resolved run name, optimizer config (peak_lr, peak_adam_lr, beta2, ε), num_train_steps, resume_from_step, and the tags. Any guard failure (G1–G17, N1–N7) raises here, before any TPU is touched.

8. **Launch the iris job** (one cell per coordinator — no shared coordinators, per the post-v10 lesson in the prior logbook):
   ```bash
   uv run iris --cluster=marin job run \
     --cpu 1 --memory 3GB --disk 9GB \
     --region us-central1 --region us-east5 \
     --priority interactive \
     --no-wait \
     --job-name true-midtrain-<scale>-<mix>-step<resume_step>-$(date -u +%Y%m%d-%H%M%S) \
     -e WANDB_API_KEY "$WANDB_API_KEY" \
     -e MIDTRAIN_MAX_TASK_FAILURES 100 \
     -e TRUE_MIDTRAIN_SELECT_SCALE <scale> \
     -e TRUE_MIDTRAIN_SELECT_MIX <mix> \
     -e TRUE_MIDTRAIN_RESUME_OUTPUT_PATH gs://marin-us-east5/checkpoints/true-midtrain-<scale>-<mix>-step<resume_step> \
     -e TRUE_MIDTRAIN_EXPECT_RESUME_STEP <resume_step> \
     -- python experiments/exp_delphi_true_midtrain.py
   ```
   Notes:
   * `--job-name` always includes a timestamp so we never reuse a prior killed-job name (lesson from the v9 zombie-reattach bug).
   * `MIDTRAIN_MAX_TASK_FAILURES=100` absorbs the v5p placement-collision dispatch race (verified working 5/9).
   * `TRUE_MIDTRAIN_EXPECT_RESUME_STEP` (NEW; replaces the prior `MIDTRAIN_EXPECT_RESUME_MIN_STEP`) is the *exact* expected step (not a floor) for guard N3.

9. **Babysit per the user's standing rule** ("babysit every job I submit"):
   * Poll iris state every 30 s for the first 5 min after submission.
   * Then every 60 s for 10 min.
   * Then every 5 min until the train_lm child is `state=running` with `preemptions=0` for ≥30 s.
   * Verify the startup log line `Resuming training from step <resume_step>`. If `Starting from scratch` appears, kill the job — the staged checkpoint is in the wrong place.

10. **Post-launch verification at first eval** (~step `resume_step + 200`):
    * `train/loss` should be in the same neighborhood as the pretrain at that step (small jump because of distribution shift, but not catastrophic).
    * `optim/learning_rate` should equal `pretrain_peak_lr × (1 - (resume_step - 0.8·N) / (0.2·N))` — i.e. the LR walk-down is continuing from the pretrain's WSD schedule. If the LR is at peak (1.0) or at min_lr (0.0), the resume is broken.
    * `Paloma c4_en` eval loss should be close to the pretrain's eval loss at that step.

### 13.5 Failure-mode playbook

| Symptom | Likely cause | Fix |
|---|---|---|
| `Starting from scratch` in startup log | Pre-staged checkpoint not in the cell's `output_path/checkpoints/<step-X>/` (typo, wrong region, fan-out skipped). | Kill, re-run Phase B5–B7, relaunch. |
| `optim/learning_rate` at peak (1.0×) at step >> resume_step | `num_train_steps` was set wrong; schedule sees us in stable phase forever. | Kill, check `num_train_steps == spec.pretrain_num_train_steps`. |
| `optim/learning_rate` at min_lr (0.0) from step 0 | `num_train_steps` was set too low (say K=0.20 budget); opt_state.count past schedule end → clamps. This is the 4/23 flat-LR bug. | Kill, set `num_train_steps` to the original pretrain target. |
| Train loss explodes within first 100 steps | Distribution shift too aggressive at high LR. (Unlikely given we start in mid-decay, but possible for 1e22 stable-phase pre-roll.) | Investigate; possibly add a gentler start by trimming the stable-phase pre-roll. |
| `TransferBudgetExceeded` during ckpt load | `mirrored(..., budget_gb=N)` set too low for the scale. | Bump `budget_gb` per scale (G1 default values are 30/50/150 already; should suffice). |
| All 3 cells of a scale die at JAX init with port-8476 race | v5p placement-collision recurrence (#5470). | `MIDTRAIN_MAX_TASK_FAILURES=100` should absorb it. If not, serialize submissions with ≥30 s gaps. |
| Iris parent dies on 1st preemption | `max_task_failures` not threaded through. | Verify `lib/marin/src/marin/training/training.py:_submit_training_job` reads `MIDTRAIN_MAX_TASK_FAILURES` (the 5/7 patch). |
| Cross-region hash-drift (parent retries in different region than child started in) | Region pin guard (G2) failed, OR Marin executor StepSpec hash includes region-specific paths. | Should be fixed by upstream #5223 (now in branch). If it recurs, kill and use `TRUE_MIDTRAIN_TRAIN_REGION` env to force a specific region. |
| WandB shows two runs for the same cell | WandB run id splitting. Should be impossible if guard G6 fires (output_path basename = run_id). | Check `_enforce_run_id` and the resume identity env vars. |

### 13.6 Decisions still pending user sign-off before any expensive action

1. **Profile choice (A/B/C from §8) → fixes which TPU slice each scale targets.** With only 3 cells the chip-hour cost is roughly 1/3 of the original 9-cell plan:
   * Profile A (fast): v5p-64 / v5p-256 / v5p-512 → **~14.4 h critical path, 2,676 chip-h**.
   * Profile B (default, recommended): v5p-32 / v5p-64 / v5p-256 → **~30 h critical path, 2,772 chip-h**.
   * Profile C (cheap-and-slow): v5p-32 / v5p-64 / v5p-128 → **~60 h critical path, 2,772 chip-h**.

2. **Region default.** Plan defaults to `us-east5` (where the May-9 1e22 cells just succeeded). Confirm or override.

3. **Pretrain replay = `nemotron_mix`** (Nemotron-CC HQ + 25% starcoderdata + 5.5% proofpile_2) — same mix the Delphi pretrains used. If the user's "nemotron cc for pretraining" instruction meant pure Nemotron-CC HQ, we need a new MidtrainMixSpec variant. Confirm `nemotron_mix` is acceptable.

4. **Phase A staging cost ($10 cross-region GCS egress) approved?** Same as before — Phase A copies one ckpt per scale regardless of how many mixes we run.

5. ~~**OK to write `experiments/exp_delphi_true_midtrain.py` next** (free; no execution)?~~ **DONE 2026-05-09.** File is written, imports clean, generates 3 cells when filtered to `p33m67` (or 9 cells when no selectors).

I will not run any `gcloud cp` or `iris job run` until each of (1)–(4) has an explicit answer.

---

## 14. Status — chronological action log

| Date (UTC) | Action | Result |
|---|---|---|
| 2026-05-09 | Confirmed AdamH WSD schedule from `experiments/scaling_law_sweeps/completed_adamh.py:127-131`: `warmup=0.1, decay=0.2, min_lr_ratio=0.0, lr_schedule="linear"`. | 80% step = start of decay phase. |
| 2026-05-09 | Confirmed `exp1337_delphi_suite.py:run_optimal_training` builds the AdamH config straight from the heuristic with no overrides. | Production schedule = heuristic defaults verbatim. |
| 2026-05-09 | Listed all `gs://marin-us-central2/.../checkpoints/` for canonical 1e20/1e21/1e22 runs and their seed replicates. | Cadence: every 5,000 steps for 1e21/1e22 (every 10,000 for 1e20) plus a final-step write. No checkpoint at exactly the 80% step for any scale. |
| 2026-05-09 | Confirmed Levanter `CheckpointInitMode.FULL_STATE` resets `state.step` to 0, which combined with the loaded `opt_state.count` triggers the same flat-LR bug we hit on 2026-04-23. | Cannot use `initialize_from_checkpoint_path` — must use the natural-resume path (pre-stage checkpoint into `output_path`). |
| 2026-05-09 | Confirmed `experiments/midtraining_mixes.py` already exposes the three mixes we need. No changes required. | Reuse as-is. |
| 2026-05-09 | Wrote this plan to `.agents/logbooks/true_midtraining.md`. | Awaiting user sign-off on § 12 before any code or staging. |
| 2026-05-09 | User confirmed checkpoint choice: 1e20 `step-40000`, 1e21 `step-20000`, 1e22 `step-30000`. Updated §4, §5, §6, §7.3, §8, §12. | Per-cell tokens: 3.70 B / 4.31 B / 34.54 B. 9-cell total = 127.67 B tokens. |
| 2026-05-09 | Computed per-cell wall-clock from prior-sweep throughputs (`midtraining_delphi.md` lines 661, 956, 8549). | 1e20 ≈ 4.3 h, 1e21 ≈ 4.2 h, 1e22 ≈ 20 h steady-state per cell; ~30 h with preemption buffer. Sweep critical path = 1e22 ≈ 32 h if all 9 cells launch concurrently. Total chip-hours ≈ 8,289 (28% of the prior 36-cell sweep). |
| 2026-05-09 | Expanded §8 with per-TPU wall-clock × chip-hour estimates for every viable scale × TPU combo. Added direct measurements for 1e21 on v5p-256 (1.7 s/step, line 2888-3214) and 1e22 on v5p-512 (4.1-4.4 s/step, line 2938-3175). | Three roll-up profiles: A (fast, v5p-64/256/512) ≈ 16 h critical path; B (default, v5p-32/64/256) ≈ 32 h; C (cheap, v5p-32/64/128) ≈ 64 h. Chip-hours roughly equal across profiles (~8,000-8,300) thanks to linear scaling; profile choice trades wall-clock vs TPU-availability risk. |
| 2026-05-09 | Wrote §13 launch plan. Cataloged 17 ported guards (G1-G17) and 7 new guards (N1-N7). Defined wandb naming convention (`true-midtrain-{scale}-{mix}-step{resume_step}`), per-cell launch sequence (Phase A staging + Phase B per-cell fan-out + dry-run + iris launch + babysit + first-eval verify), and failure-mode playbook. | All guards and the new naming convention are mandatory; nothing to launch until §13.6 user sign-off. |
| 2026-05-09 | Wrote `experiments/exp_delphi_true_midtrain.py` with all 17+7 guards. | New file; not yet executed or staged. Awaiting §13.6 sign-off. |
| 2026-05-09 | Verified `exp_delphi_true_midtrain.py` imports cleanly and enumerates 9 cells in dry-run. Fixed one bug: G7/N7 startswith assertion now strips the `checkpoints/` prefix that `default_train` prepends. | All 9 cells materialise with `true-midtrain-` prefix. |
| 2026-05-09 | User narrowed sweep to **p33m67 only** (1 mix × 3 scales = 3 cells). Updated §4 (token table), §8 (cell list, profiles), §13 (Phase B fan-out 9→3), §13.6 (open Q's). | Script unchanged (still supports all 3 mixes via `TRUE_MIDTRAIN_SELECT_MIX`). 3-cell total = 42.55 B tokens. Profile-B chip-hours ≈ 2,772 (was 8,316 for 9 cells). |

# codex 5.5 2026-05-15T23:47:49Z

Hardened `experiments/delphi_models.py` after the Delphi wrong-base incident.
This was a code-only registry cleanup; no GCS data movement, staging, training,
or Iris launch happened.

What changed:

- Split model identity by use case:
  - `hf_repo` / pinned HF revision: public weights for eval, inference, and
    `CheckpointInitMode.MODEL_ONLY` continued-pretraining starts.
  - `gcs_run_root`: native Levanter run root for full-state checkpoint workflows.
  - `verified_checkpoint_path`: concrete native checkpoint path for model-only
    Levanter loading when an experiment intentionally uses
    `initialize_from_checkpoint_path`.
- Pinned the current HuggingFace `refs/heads/main` SHAs for the verified Delphi
  repos using `git ls-remote` on 2026-05-15, so `download_step()` no longer
  follows a moving `main` branch.
- Registered only the 9 verified models:
  `3e18`, `9e18`, `2e19`, `3e19`, `9e19`, `2e20`, `3e20`, `1e21`, `1e22`.
- Removed the partially-filled `DELPHI_1E23` object from `ALL_DELPHI_MODELS`.
  Kept only `DELPHI_1E23_HF_REPO`, `DELPHI_1E23_HF_REVISION`, and
  `DELPHI_1E23_GCS_RUN_ROOT` until architecture and checkpoint-step metadata
  are verified.
- Replaced float-keyed lookup with `get_delphi_model(flops_key: str)` and
  `DELPHI_BY_FLOPS_KEY`. `get_delphi_model("3e+20")` resolves to
  `DELPHI_3E20`; `get_delphi_model("1e20")` raises because there is no 1e20
  Delphi model.
- Added construction-time guards for:
  - banned v5 isoflop paths such as
    `isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5`;
  - `gcs_run_root` accidentally being a concrete `.../checkpoints/step-N` path;
  - missing pinned HF revisions;
  - non-positive architecture/checkpoint metadata.

Validation run:

```bash
./infra/pre-commit.py --fix experiments/delphi_models.py
uv run python -m py_compile experiments/delphi_models.py
uv run python experiments/delphi_models.py
uv run python - <<'PY'
from experiments.delphi_models import DELPHI_3E20, get_delphi_model

assert DELPHI_3E20.hf_revision == "b585d913d6f72cf86482a7adc71cc77af4b910f3"
assert get_delphi_model("3e+20") is DELPHI_3E20
try:
    get_delphi_model("1e20")
except ValueError as e:
    assert "There is no 1e20 Delphi model" in str(e)
else:
    raise AssertionError("1e20 lookup should fail")
PY
```

Remaining blocker before any true-midtraining launch:

- `experiments/exp_delphi_true_midtrain.py` still reflects the pre-incident
  1e20-style plan and must not be launched as-is. Before launch, decide whether
  to drop the 1e20 row entirely or replace it with the canonical `DELPHI_3E20`
  v6 bucket winner and verified optimizer/run metadata.

# codex 5.5 2026-05-16T00:00:43Z

Closed the remaining wrong-base cleanup items from the post-mortem thread. This
was a code/documentation hygiene pass only; no Iris jobs, training launches, GCS
copies, or checkpoint staging happened.

What changed:

- `experiments/exp_delphi_math_10b_midtrain.py` no longer has a launchable
  `1e20-iso-d2048-L21` base. The math continued-pretraining sweep now imports
  verified `DELPHI_1E21` / `DELPHI_1E22` checkpoint metadata from
  `experiments/delphi_models.py`, and the old 1e20 slot is only a warning
  comment.
- `experiments/exp_delphi_true_midtrain.py` no longer exposes a 1e20 true-
  midtraining cell. Until native checkpoint step and optimizer-state metadata
  are verified for the v6 3e20 bucket winner, true-midtraining is fail-closed
  to `1e21` and `1e22`.
- `scripts/_verify_mirror_stage.py` now derives its mirror target from
  `DELPHI_3E20.verified_checkpoint_path`, so the mirror verifier exercises the
  canonical `d2304-L23` / `adamh_scaling_v6` checkpoint instead of the wrong v5
  ablation.
- The analysis/reporting scripts now label the `1e20` rows as historical
  v5-isoflop 3e20 rows, not Delphi. Internal names like `CANONICAL_DONE` and
  `canonical_for_cell` were renamed so generated artifacts do not quietly
  reintroduce the bad framing.
- This logbook now has inline **WRONG 1e20 BASE** markers in the historical
  sections that previously mentioned the bad v5 checkpoint without local
  context (§3, §4, §6.2, §7.3, and the §8/§13 launch/cost rows).
- Claude memory files were updated to point future sessions at
  `experiments/delphi_models.py`, `get_delphi_model()`, and
  `assert_not_banned()` instead of old float-key or GCS-grep patterns.
- Posted the correction on GitHub issue #4547:
  https://github.com/marin-community/marin/issues/4547#issuecomment-4464605119

Validation run:

```bash
./infra/pre-commit.py --fix experiments/delphi_models.py \
  experiments/exp_delphi_math_10b_midtrain.py \
  experiments/exp_delphi_true_midtrain.py \
  scripts/_verify_mirror_stage.py \
  scripts/analysis/plot_midtrain_curves.py \
  scripts/analysis/interactive_midtrain_prefix_plot.py \
  scripts/analysis/midtrain_loss_predictor.py \
  scripts/analysis/download_midtrain_wandb.py \
  scripts/analysis/delphi_midtrain_scaling_analysis.py
uv run python -m py_compile experiments/delphi_models.py \
  experiments/exp_delphi_math_10b_midtrain.py \
  experiments/exp_delphi_true_midtrain.py \
  scripts/_verify_mirror_stage.py \
  scripts/analysis/plot_midtrain_curves.py \
  scripts/analysis/interactive_midtrain_prefix_plot.py \
  scripts/analysis/midtrain_loss_predictor.py \
  scripts/analysis/download_midtrain_wandb.py \
  scripts/analysis/delphi_midtrain_scaling_analysis.py
uv run python - <<'PY'
from experiments import exp_delphi_math_10b_midtrain as math_midtrain
from experiments import exp_delphi_true_midtrain as true_midtrain
from scripts import _verify_mirror_stage as mirror_stage

assert "1e20-iso-d2048-L21" not in math_midtrain.BASES
assert set(math_midtrain.BASES) == {"1e21-v5", "1e22-v5"}
assert set(true_midtrain.PRETRAINS) == {"1e21", "1e22"}
assert "d2304-L23" in mirror_stage.CKPT_REL
assert "adamh_scaling_v6" in mirror_stage.CKPT_REL
print("ok")
PY
./infra/pre-commit.py --fix .agents/logbooks/true_midtraining.md
```

The final grep check still finds the wrong v5 path only in explicit caveats,
the banned-path registry guard, memory notes, and historical W&B run names used
for offline analysis lookup. It is no longer a launchable checkpoint path in
the midtraining experiment files.

# codex 5.5 2026-05-16T00:13:31Z

Clarified and hardened the optimizer-state split between the two Delphi
midtraining experiment files.

Policy now enforced in code:

- `experiments/exp_delphi_math_10b_midtrain.py` is continued pretraining from
  base weights only. It must set `initialize_from_checkpoint_path` and
  `CheckpointInitMode.MODEL_ONLY`, so the pretrain optimizer state is dropped
  and the AdamH schedule starts from a fresh count. Same-run preemption resume
  still restores that midtraining run's own optimizer state before the init
  branch is reached.
- `experiments/exp_delphi_true_midtrain.py` is true midtraining. It must not
  use `initialize_from_checkpoint_path`; the pretrain checkpoint must be
  pre-staged under the run's own `output_path/checkpoints/` namespace so
  Levanter natural resume restores model, optimizer state, and step together.
- The true-midtraining `executor_main` path now refuses no-selector launches.
  No-selector mode is dry-run/introspection only; real launches require
  `TRUE_MIDTRAIN_SELECT_SCALE`, `TRUE_MIDTRAIN_SELECT_MIX`,
  `TRUE_MIDTRAIN_RESUME_OUTPUT_PATH`, and
  `TRUE_MIDTRAIN_EXPECT_RESUME_STEP`.

Validation run:

```bash
uv run pytest experiments/test_default_train_init_mode.py -q
uv run python -m py_compile experiments/exp_delphi_math_10b_midtrain.py \
  experiments/exp_delphi_true_midtrain.py \
  experiments/test_default_train_init_mode.py
./infra/pre-commit.py --fix experiments/exp_delphi_math_10b_midtrain.py \
  experiments/exp_delphi_true_midtrain.py \
  experiments/test_default_train_init_mode.py
./infra/pre-commit.py --fix .agents/logbooks/true_midtraining.md
git diff --check
```

# codex 5.5 2026-05-16T00:19:24Z

Corrected stale true-midtraining status after checking live W&B and GCS. The
older top-level `PLAN — nothing has been launched yet` statement was stale:
true-midtraining runs were launched after the initial plan.

Live status from `marin-community/delphi-midtraining` and GCS:

| Run | Status | W&B step | Latest Levanter checkpoint | HF exports | Notes |
|---|---|---:|---|---|---|
| `true-midtrain-1e20-p33m67-step40000` | finished | 47063 | `gs://marin-us-east5/checkpoints/true-midtrain-1e20-p33m67-step40000/checkpoints/step-47063` | through `hf/step-47063` | **Wrong-base / contaminated**: this used the deprecated v5 isoflop 1e20 stand-in. Keep only as historical evidence. |
| `true-midtrain-1e21-p33m67-step20000-v5p64` | finished | 22056 | `gs://marin-us-east5/checkpoints/true-midtrain-1e21-p33m67-step20000-v5p64/checkpoints/step-22056` | through `hf/step-22056` | Valid true-midtraining cell. |
| `true-midtrain-1e22-p33m67-step30000` | crashed | W&B list showed ~30045; direct lookup later returned `CommError` | `gs://marin-us-east5/tmp/ttl=14d/checkpoints-temp/marin-us-east5/checkpoints/true-midtrain-1e22-p33m67-step30000/checkpoints/step-30031` | none found | Early failed attempt; do not resume this unless intentionally recovering that namespace. |
| `true-midtrain-1e22-p33m67-step30000-v5p64` | crashed | 33436 | `gs://marin-us-east5/tmp/ttl=14d/checkpoints-temp/marin-us-east5/checkpoints/true-midtrain-1e22-p33m67-step30000-v5p64/checkpoints/step-33419` | `hf/step-30584` only | Valid 1e22 attempt, incomplete. This is the resume target unless a newer checkpoint is discovered. |

Conclusion:

- The valid post-incident true-midtraining set is `1e21` and `1e22` only.
- `1e21` finished.
- `1e22` did **not** finish. Resume from
  `gs://marin-us-east5/checkpoints/true-midtrain-1e22-p33m67-step30000-v5p64`
  with `TRUE_MIDTRAIN_EXPECT_RESUME_STEP=30000`; Levanter should pick the
  latest temp checkpoint (`step-33419` at this check) via natural resume.
- The finished `1e20` run is not a valid Delphi true-midtraining result because
  it used the wrong v5 base.

Status commands run:

```bash
uv run python - <<'PY'
import os
import wandb
from levanter.checkpoint import discover_latest_checkpoint
from marin.training.training import temporary_checkpoint_base_path

api = wandb.Api(timeout=60)
for name in [
    "true-midtrain-1e20-p33m67-step40000",
    "true-midtrain-1e21-p33m67-step20000-v5p64",
    "true-midtrain-1e22-p33m67-step30000",
    "true-midtrain-1e22-p33m67-step30000-v5p64",
]:
    try:
        run = api.run(f"marin-community/delphi-midtraining/{name}")
        print(name, run.state, run.summary.get("global_step", run.summary.get("_step")))
    except Exception as exc:
        print(name, f"wandb lookup failed: {type(exc).__name__}: {exc}")
    out = f"gs://marin-us-east5/checkpoints/{name}"
    print(discover_latest_checkpoint(os.path.join(out, "checkpoints"), temporary_checkpoint_base_path(out)))
PY
```
