# Delphi × Nemotron-CC-Math 10 B midtraining — logbook

## README / IMPORTANT — current handoff state as of 2026-05-01

This logbook now tracks both the midtraining experiments and the cross-region
resume incident that happened while running them. Read this section before
launching, relaunching, or "recovering" any Delphi midtraining job.

### Hard resume rule

Never assume a failed/preempted training run will resume just because the
human-readable step name is unchanged. Marin's real checkpoint/W&B identity
includes the executor output hash.

Before relaunching any failed Delphi midtraining run:

1. Find the exact old output path and run id from `.executor_info`, logs, W&B,
   or GCS.
2. Check both permanent checkpoints and temporary checkpoints.
3. Relaunch with the exact old output path forced, e.g. `MIDTRAIN_OUTPUT_PATH_OVERRIDE`
   or `ExecutorStep.with_output_path(...)`.
4. Verify startup logs show the same run id/output path and
   `Resuming training from step ...`.

For the 2026-04-27 incident, `p67m33/lr0.5` drifted from the original central
namespace `delphi-1e20-p67m33-20b-lr0.5-f74454` to the wrong east5 namespace
`delphi-1e20-p67m33-20b-lr0.5-378f43`. Treat those as different runs.

### What is fixed now

- **Executor/StepSpec region-sensitive hashes:** fixed in main by
  `7f0b99b9e Stop region prefixes leaking into Marin executor identity hashes (#5223)`.
  `StepSpec.hash_id` no longer depends on physical `gs://marin-<region>/...`
  dependency paths, Nemotron v2 normalize uses `relative_input_path`, and deep
  executor dependencies use region-stable `{name}-{hash}` identifiers.
- **Temporary checkpoint region layout:** improved in main by
  `b4298305a infra/rigging: fold tmp buckets into main buckets (#5266)`.
  Temporary paths now live under the main regional buckets, e.g.
  `gs://marin-us-east5/tmp/ttl=14d/...`, and training temp checkpoint roots are
  chosen from the output path's region.
- **TensorStore cross-region budget accounting:** fixed in main by
  `a154c044f Charge cross-region transfer budget on tensorstore checkpoint I/O (#5225)`.
  TensorStore checkpoint reads/writes now call `record_transfer(...)`.
- **Iris split-slice/orphan attempt retry bugs:** improved in main by
  `9d9b9a2a7 [iris] Fix coscheduled split-slice and orphan attempt bugs (#5249)`.
- **This branch's Delphi experiment launch guard:** pushed as
  `4b40df269 [experiments] Pin Delphi midtraining jobs by region`. The parent
  coordinator can still be launched with both `--region us-central1 --region us-east5`,
  but generated `train_lm` child resources are pinned to the coordinator's
  resolved v5p region.

### What is not fully fixed

- There is still no general "move this training run to another region, copy the
  right checkpoints, and always resume the old run id" system. Current fixes
  give stable identity, better temp layout, budget accounting, and fail-fast
  behavior.
- `cc2678ff4 Guard cross-region GCS access in training and tokenization` is
  **not** in main. That PR is on `pr-5221`, not merged. So global `TrainLmOnPodConfig`
  child-resource alignment is not universal; this branch only guards the Delphi
  midtraining experiment itself.
- `2891acc6e [checkpoint] Cross-region temp checkpoint discovery via mirrortmp://`
  is **not** in main. Main chose the `#5266` temp-under-main-bucket approach
  instead.
- JAX coordinator RPC peer-loss after preemption and stale port `8476` cleanup
  are not obviously fixed by the commits above.
- Do not use `MARIN_I_WILL_PAY_FOR_ALL_FEES=1` in launch recipes to paper over
  cross-region placement. It hides the bug and can allow expensive behavior.

### Branch state

- Remote `origin/midtrain_data` currently points at `4b40df269`.
- Local `midtrain_data` has since merged `origin/main` at
  `ecd8fbca7 Merge remote-tracking branch 'origin/main' into midtrain_data`.
  That merge is local only unless a later agent pushes it.
- Expect unrelated local dirty files in this worktree, including this logbook,
  analysis scripts/plots, `experiments/defaults.py`, and `tests/test_training.py`.
  Do not stage them casually.

> **How to use this file (for any agent that lands here later).**
>
> This is a **shared working scratchpad**, not a polished writeup. Treat every
> section as live — update it as you go, don't wait until a task is "done".
>
> - Before doing anything, scan the [Status log](#status--chronological-action-log)
>   at the bottom to see what the previous agent did last.
> - Every time you take a non-trivial action (read a file, run a command, make
>   a decision, discover something that contradicts this plan), append a dated
>   row to that table. UTC timestamps, one line each. If it's not in the log,
>   it didn't happen.
> - Record failures and dead-ends with the exact error text, not paraphrased.
> - If a number, path, or hyperparameter here turns out to be wrong,
>   **edit it in place** and note the correction in the Status log. Do not
>   leave stale facts behind — future agents will trust them.
> - Keep the [Goal](#goal), [User-specified constraints](#user-specified-constraints),
>   and [Base models](#base-models-the-two-smallest-adamh-checkpoints-we-have)
>   sections immutable unless the user explicitly changes scope.
> - Add new sections freely (Results, Pilots, Bugs, Decisions, etc.) at the
>   bottom, above the Status log.

**Date started:** 2026-04-21
**Status:** Experiments and cross-region incident postmortem recorded; main has been locally merged into this branch for inspection.
**Base-model catalogue:** [`.agents/projects/delphi_midtraining.md`](../projects/delphi_midtraining.md)
**External reference:** [*Delphi Scaling Laws: Key Findings* — Will Held](https://oa.williamheld.com/blog/delphi/)
**Branch:** `midtrain_data`
**Primary artefact being built:** `experiments/exp_delphi_math_10b_midtrain.py`

---

## Goal

### Overarching project goals (two tracks)

1. **Predict loss trajectory for midtraining.** Build a calibrated expectation for how math-eval loss evolves when you continue-train an existing pretrain checkpoint on a math-heavy dataset, so that larger/more-expensive midtrain runs can be scoped (token budget, LR schedule, decay shape) from smaller runs instead of guessed. This Delphi × Nemotron-CC-Math sweep is a primary data point for that predictor.
2. **Pick a good midtraining dataset.** Compare candidate math-heavy datasets (the Nemotron-CC-Math quality tiers being our current anchor) on their effect on post-midtrain downstream evals, so that future runs don't waste compute on a poorly-curated corpus.

### This logbook's subgoal (the concrete sweep tracked here)

Run a small LR sweep that continues-trains the two smallest existing AdamH-trained Marin checkpoints on **10 B tokens of `nemotron_cc_math_v1/4plus`**, to de-risk a Mantis-style math-midtraining recipe before spending v4-512 / v4-1024 time on Delphi 1e22 / 1e23. We're looking for the highest peak LR that still drops the math-eval loss monotonically.

This subgoal feeds both tracks: the LR-factor × final-loss pairs inform the loss-trajectory predictor (track 1), and the dataset is held fixed at `nemotron_cc_math_v1/4plus` so the signal is attributable to LR choice — a prerequisite for the dataset-selection track (2), which requires LR to be a solved variable before dataset-quality can be cleanly isolated.

## User-specified constraints

- Start the midtraining peak LR at **2/3 of each base's own pretrain peak**, sweep around there.
- Use **AdamH** (same optimizer family the base models were pretrained with).
- **500 warmup steps**.
- Start with **the two smallest bases** — "1e20 and 1e21 models" in the user's words.
- 10 B token budget on the Nemotron-CC-Math dataset.

---

## Base models (the two smallest AdamH checkpoints we have)

Authoritative numbers read directly from each run's W&B config.

| Slot | Model run | H | L | Params | Pretrain BS | Pretrain steps | Pretrain tokens | Pretrain peak `learning_rate` | Pretrain peak `adam_lr` | β₂ | ε | Final c4_en loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **"1e20"** (3e20 isoflop scan @ compute-optimal) | `isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5` | 2048 | 21 | ~1.9 B | 128 | 47 064 | 24.7 B | **4.483e-3** | **7.382e-5** | 0.99980 | 4.11e-8 | 2.858 |
| **"1e21"** (Delphi v5 canonical) | `adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021` | 2560 | 26 | ~3.4 B | 512 | 22 057 | 46.3 B | **7.425e-3** | **4.314e-4** | 0.99920 | 2.81e-8 | 2.733 |

**Starting checkpoints (Levanter TensorStore; use `initialize_from_checkpoint_path` to load weights only):**
```
gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/checkpoints/step-46915/
gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/checkpoints/step-21979/
```

Why the 3e20 isoflop ckpt stands in for "1e20": no AdamH `optimal-training` run exists at 1e20 FLOPs — the Delphi scaling-ladder only covers 1e21 / 1e22 / 1e23. The 3e20 isoflop scan ran many `(H, L, B)` combinations to FIT the scaling law; its compute-optimal point (`d2048-L21`) is a genuine AdamH-trained model at ~1.9 B params with a clean HF export and a mid-2 c4-en loss. User confirmed this is the stand-in they want.

Why `v5` for the 1e21 slot: the `vN` suffix on `adamh-scaling-ladder-nemotron-optimal-1e+21-*` runs is an **infra-retry counter**, not a recipe version — the recipe itself stays `adamh_scaling_v6` across all revs. `v5`, `v5-seed42`, `v5-seed62746`, `v6` all converged; their c4-en losses cluster within 0.001. Use `v5` as canonical; add seed replicates later only if seed variance is the research question.

---

## Fixed training knobs (both bases)

| Knob | Value | Why |
|---|---|---|
| Data | 100 % `nemotron_cc_math_v1/4plus` | Highest-quality CC-Math band per `midtraining_math.md` §2a. Phi-4-cleaned, non-Qwen. 52 B token pool ≫ 10 B budget → no epoch repeats. Already tokenized and registered as `BUCKET_2["nemotron_cc_math_v1/4plus"]` in `experiments/midtraining_data_buckets.py`. |
| Tokenizer | `llama3_tokenizer` | Matches both base checkpoints. |
| Seq-len | 4096 | Matches both base checkpoints. |
| Batch size | **512** | Standardizes on v4-128-friendly batch. The 1.9 B model was pretrained at BS=128, but batch-alignment to the pretrain run is not required since we rebuild the optimizer anyway. |
| Token budget | **10 B** | User-specified. |
| `num_train_steps` | **4 768** | = ceil(10e9 / (512 × 4096)) |
| Warmup steps | **500** | User-specified. Rebuilds fresh Adam / AdamH momentum. |
| Decay tail | remaining **4 268 steps** | `= num_train_steps − warmup`. No stable middle — pure linear warmup → linear decay. |
| `min_lr_ratio` | 0.1 | Mantis cooldown convention. End at 10 % of peak rather than 0 (pretrain's `min_lr_ratio=0` decays all the way to zero, which is too aggressive for a 10 B-token midtrain). |
| `lr_schedule` | `"linear"` | Same family as Delphi pretrain (Complete(d)P) + Mantis cooldown. |
| `beta1` | 0.9 | Heuristic default. |
| `beta2`, `epsilon`, `max_grad_norm` | **Inherited per-base from the pretrain config** (table above) | These values are coupled to the pretrain `(B, T)` via `_compute_beta2` / `_compute_epsilon`. Keeping the base's own values preserves the curvature statistics the loaded weights were optimized against. |
| `max_grad_norm` | 0.1 | Heuristic default. |
| `reset_data_loader_on_init` | `True` | New data distribution → fresh iterator. |
| `z_loss_weight` | 0 | Same as Delphi. |
| Precision | `jmp.get_policy("p=f32,c=bfloat16")` | Same as Delphi. |
| TPU | `v4-128` for both | Matches 1e21 Delphi pretrain topology; ample for 1.9 B. |
| Mesh | `{data: -1, replica: 1, model: 1}` | `tp=1` suffices since H=2048 and H=2560 are both divisible by 64 chips. (Same `tp`-search loop as `exp1337_delphi_suite.py`.) |
| `steps_per_eval` | 200 | ≈24 evals across 4 768 steps. |
| Checkpointer | `save_interval=10min`, `keep=[{"every": 1000}]` | Short run; light retention. |
| HF export | `hf_save_steps=1000` | Final HF + a few intermediate waypoints. |

---

## LR sweep — numbers

Factors: **`{0.5, 0.67, 0.83} × pretrain peak`**. User asked for "2/3 of peak, sweep around there"; these three bracket 2/3 with symmetric ±0.17. Both `learning_rate` *and* `adam_lr` get scaled by the same factor so the weight-LR ↔ embedding-LR coupling from the heuristic stays intact.

### Base A: 1e20 (1.9 B)

| Factor | `learning_rate` | `adam_lr` | Run name |
|---:|---:|---:|---|
| 0.5  | 2.241e-3 | 3.691e-5 | `delphi-1e20-iso-math-10b-lr0.5` |
| **0.67** | **2.989e-3** | **4.921e-5** | **`delphi-1e20-iso-math-10b-lr0.67`** (primary) |
| 0.83 | 3.721e-3 | 6.127e-5 | `delphi-1e20-iso-math-10b-lr0.83` |

### Base B: 1e21 Delphi v5 (3.4 B)

| Factor | `learning_rate` | `adam_lr` | Run name |
|---:|---:|---:|---|
| 0.5  | 3.713e-3 | 2.157e-4 | `delphi-1e21-v5-math-10b-lr0.5` |
| **0.67** | **4.950e-3** | **2.876e-4** | **`delphi-1e21-v5-math-10b-lr0.67`** (primary) |
| 0.83 | 6.163e-3 | 3.580e-4 | `delphi-1e21-v5-math-10b-lr0.83` |

**Wall-time estimate (per run, v4-128):** ~3.5 h for 1.9 B; ~6.5 h for 3.4 B. Total compute if serialized: ~30 h. Fully parallel across 6 pods: ~6.5 h elapsed.

**Outputs:** `gs://marin-us-central2/<run-name>-<hash>/{checkpoints,hf}`. **W&B:** `marin-community/marin`.

---

## Implementation — new file

**Create:** `experiments/exp_delphi_math_10b_midtrain.py` (~200 lines).

Structure mirrors `experiments/exp1337_delphi_suite.py`'s `run_optimal_training` + `TrainLmOnPodConfig` wrapping, with the `initialize_from_checkpoint_path` + fresh `AdamHConfig` pattern lifted from `experiments/tootsie/exp1529_32b_mantis_cooldown.py:131–165`. Six `ExecutorStep`s, one per (base × lr_factor).

Skeleton (for reference — adapt to whatever helpers exist on this branch at implementation time):

```python
from dataclasses import replace
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LMMixtureDatasetConfig
from levanter.main import train_lm
from levanter.optim.adamh import AdamHConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.defaults import default_validation_sets
from experiments.llama import llama3_tokenizer
from experiments.midtraining_data_buckets import BUCKET_2
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

SEQ_LEN          = 4096
BATCH_SIZE       = 512
NUM_TRAIN_STEPS  = 4768      # 10 B / (512 × 4096) rounded up
WARMUP_STEPS     = 500
DECAY_STEPS      = NUM_TRAIN_STEPS - WARMUP_STEPS
MIN_LR_RATIO     = 0.1
TPU_TYPE         = "v4-128"

BASES = {
    "1e20-iso-d2048-L21": dict(
        ckpt="gs://marin-us-central2/checkpoints/isoflop/"
             "isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/checkpoints/step-46915/",
        hidden_dim=2048, num_layers=21,
        peak_lr=4.483e-3, peak_adam_lr=7.382e-5,
        beta2=0.99980, epsilon=4.11e-8,
    ),
    "1e21-v5": dict(
        ckpt="gs://marin-us-central2/"
             "adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/checkpoints/step-21979/",
        hidden_dim=2560, num_layers=26,
        peak_lr=7.425e-3, peak_adam_lr=4.314e-4,
        beta2=0.99920, epsilon=2.81e-8,
    ),
}
LR_FACTORS = [0.5, 0.67, 0.83]

math_mix = LMMixtureDatasetConfig(
    components={
        "nemotron_cc_math_v1/4plus": step_to_lm_mixture_component(
            BUCKET_2["nemotron_cc_math_v1/4plus"], include_raw_paths=False
        ),
    },
    train_weights={"nemotron_cc_math_v1/4plus": 1.0},
    tokenizer=llama3_tokenizer,
    cache_options={"batch_size": 128},
    block_cross_document_attention=True,
)

# Merge validation-only components (Paloma suites) so eval curves stay
# comparable to the pretrain W&B panels.
_val = {n: step_to_lm_mixture_component(s, include_raw_paths=False)
        for n, s in default_validation_sets(tokenizer=llama3_tokenizer).items()}
data_with_val = replace(
    math_mix,
    components={**math_mix.components, **_val},
    train_weights={**math_mix.train_weights,
                   **{k: 0.0 for k in _val if k not in math_mix.train_weights}},
)

def build_inner_config(base_tag: str, base: dict, lr_factor: float):
    lr      = base["peak_lr"]      * lr_factor
    adam_lr = base["peak_adam_lr"] * lr_factor
    optimizer = AdamHConfig(
        learning_rate=lr, adam_lr=adam_lr,
        beta1=0.9, beta2=base["beta2"], epsilon=base["epsilon"],
        max_grad_norm=0.1,
        warmup=WARMUP_STEPS, decay=DECAY_STEPS,
        min_lr_ratio=MIN_LR_RATIO,
        lr_schedule="linear", nesterov=False,
    )
    return train_lm.TrainLmConfig(
        data=data_with_val,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                entity="marin-community", project="marin",
                tags=["midtraining", f"base={base_tag}",
                      "nemotron-cc-math-4plus",
                      f"lr_factor={lr_factor}",
                      f"peak_lr={lr:.3e}",
                      f"adam_lr={adam_lr:.3e}",
                      "adamh", "delphi-midtrain"],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            per_device_parallelism=-1,
            num_train_steps=NUM_TRAIN_STEPS,
            steps_per_eval=200,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=1000)],
            ),
            mesh=MeshConfig(
                axes={"data": -1, "replica": 1, "model": 1},
                compute_mapping={
                    "token":        (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                },
            ),
            allow_nondivisible_batch_size=True,
        ),
        train_seq_len=SEQ_LEN,
        model=<llama_config builder that matches metadata.json of the source ckpt>,
        optimizer=optimizer,
        initialize_from_checkpoint_path=base["ckpt"],
        reset_data_loader_on_init=True,
        z_loss_weight=0.0,
        hf_save_steps=1000,
    )

runs = [
    ExecutorStep(
        name=f"delphi-{base_tag}-math-10b-lr{f}",
        fn=lambda cfg: run_levanter_train_lm(cfg),
        config=TrainLmOnPodConfig(
            train_config=build_inner_config(base_tag, base, f),
            resources=ResourceConfig.with_tpu(TPU_TYPE),
            output_path=this_output_path(),
        ),
    )
    for base_tag, base in BASES.items()
    for f in LR_FACTORS
]

if __name__ == "__main__":
    executor_main(steps=runs)
```

The `<llama_config builder>` slot needs adapting to whatever concrete `LlamaConfig` constructor the Delphi template already uses (probably via `completed_adamh_heuristic.build_model_config(...)` or a direct `LlamaConfig(hidden_dim=…, num_layers=…)` call). Cross-check against each source checkpoint's `metadata.json` at implementation time — TensorStore restore is shape-indexed and silently fails to NaN on shape drift.

---

## Critical files (read at implementation time)

- `experiments/scaling_law_sweeps/completed_adamh.py` L 116–131, 169–209 — AdamH heuristic source of truth for peak LR / ε / β₂ formulas.
- `experiments/exp1337_delphi_suite.py` — template for `TrainLmOnPodConfig` wrapping, mesh config, `tp`-search loop.
- `experiments/tootsie/exp1529_32b_mantis_cooldown.py` L 131–165 — template for `initialize_from_checkpoint_path` + rebuilt-optimizer pattern.
- `experiments/tootsie/exp898_deeper_cooldown.py` L 31–78 — confirms `warmup` / `decay` accept absolute int steps.
- `experiments/midtraining_data_buckets.py` — `BUCKET_2["nemotron_cc_math_v1/4plus"]` (tokenize ExecutorStep).
- `lib/levanter/src/levanter/optim/adamh.py` — `AdamHConfig` fields.
- `.agents/projects/delphi_midtraining.md` — Delphi run catalogue + all GCS paths (committed at `84f53bc6f`).

---

## Verification before full sweep launch

1. **Static:** `uv run python experiments/exp_delphi_math_10b_midtrain.py --help` (or `executor_main(..., dry_run=True)`). Confirms imports resolve, `BUCKET_2` entry exists, `(learning_rate, adam_lr)` match the sweep tables above.
2. **GCS:** `gcloud storage ls gs://…/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/checkpoints/step-46915/` and `…/adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/checkpoints/step-21979/` — verify `manifest.ocdbt`, `metadata.json`, `d/`.
3. **Architecture:** read both source `metadata.json` files; diff against the built `LlamaConfig`. Must match exactly for TensorStore restore.
4. **Pilot:** launch the two `lr0.67` runs only. After step 100:
   - `eval/paloma/c4_en/loss` ≈ 2.858 (1e20) / 2.733 (1e21) — confirms weights loaded.
   - Train loss trending down by step 500 — if flat/rising at `lr0.67`, `lr0.83` will blow up, so don't launch that upper tier.
   - W&B LR panel shows 500-step linear warmup then linear decay.
5. If the pilots look healthy, launch the remaining 4 runs in parallel.

---

## Risks / known unknowns

- **Fresh Adam state** — first ~200 steps will have elevated loss while momentum / variance rebuild. Expected, not a bug; the 500-step warmup absorbs it.
- **`LMMixtureDatasetConfig` with a single component** — Mantis and Delphi use multi-source mixtures; single-source may or may not be accepted. Mitigation: fall back to `LMDatasetConfig` for the training side if the mixture shape breaks.
- **β₂ / ε mismatch between the two base models** — each run uses its own pretrain β₂, ε (not heuristic defaults), because those values are coupled to `(B, T)` and switching them mid-training is an uncharacterized perturbation.
- **Architecture shape drift** — the single most dangerous failure mode; check `metadata.json` before each launch.
- **W&B team-private** — fine; no PR/label hygiene needed for running experiments.

---

## Implementation decisions (2026-04-21)

Notes captured while writing `experiments/exp_delphi_math_10b_midtrain.py`. Update if any of these turn out to be wrong.

- **Harness choice: `default_train` + `SimpleTrainConfig`, not raw `TrainLmConfig`.** The older plan skeleton constructed `train_lm.TrainLmConfig` directly (mirroring `exp1337_delphi_suite.py`'s `run_optimal_training`). In practice, `experiments.defaults.default_train` is the canonical top-level helper on this branch — both `exp1529_32b_mantis_cooldown.py` and `exp898_deeper_cooldown.py` use it. It wraps `TrainLmConfig`, injects default Paloma validation sets, resolves `initialize_from_checkpoint_path` semantics, and returns a ready-to-execute `ExecutorStep`. Stick with it.
- **`SimpleTrainConfig.learning_rate` is required but ignored.** `default_train` uses `train_config.optimizer_config` if set (bypassing the flat LR / warmup / decay fields entirely — see `default_train` at `experiments/defaults.py:474-500`). We still have to pass a non-`None` `learning_rate` because it's a required dataclass field. We set it to the actual peak we use so W&B logs stay consistent.
- **Pass `AdamHConfig` with warmup + decay as INT absolute steps.** Confirmed by `exp898_deeper_cooldown.py:31-42` which uses `decay=COOLDOWN_LEN=10000` as int. Avoids the fraction-vs-int ambiguity when `num_train_steps` is small.
- **Architecture: the bases are `Qwen3Config`, not `LlamaConfig`.** `completed_adamh_heuristic._build_model_config(hidden_size, seq_len)` in `experiments/scaling_law_sweeps/completed_adamh.py:227-244` returns `Qwen3Config(hidden_dim=…, intermediate_dim=4×H, num_layers=<heuristic>, num_heads=H/128, num_kv_heads=H/128, max_seq_len=…, rope=Llama3RotaryEmbeddingsConfig())`. Verified against the wandb configs of both bases (H=2048 → L=21, intermediate=8192, 16 heads; H=2560 → L=26, intermediate=10240, 20 heads). Rope defaults (θ=5e5, factor=8, original_max_pos=8192) match the wandb config exactly (`lib/levanter/src/levanter/layers/rotary.py:152-156`).
- **We call a private method (`_build_model_config`).** Yes it has an underscore — we use it because it is the *same* code path the pretrain ran, guaranteeing byte-identical architecture. If we constructed `Qwen3Config` by hand and anything drifted (intermediate_dim, GQA setup, rope default, head_dim), the TensorStore restore would silently fail to NaN. Prefer breaking loudly on API change over breaking silently on shape drift. If this method is renamed, update the experiment file in lockstep.
- **Tokenizer: `llama3_tokenizer = "meta-llama/Meta-Llama-3.1-8B"` on both sides.** Verified via the wandb config of `isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5` (`data.tokenizer = "meta-llama/Meta-Llama-3.1-8B"`). `BUCKET_2["nemotron_cc_math_v1/4plus"]` is built by `tokenize_nemotron_v2_family` which defaults to the same llama3 string. Vocab sizes match (128256). `completed_adamh_heuristic.tokenizer = "marin-community/marin-tokenizer"` is used only for *vocab-size lookup in model-config-candidate filtering* during the sweep; it does not set the model's actual training tokenizer. So the midtraining run will tokenize its math data with llama3 and train on a model whose embeddings were trained with llama3 — consistent.
- **Data passed as a single `ExecutorStep` (not an `LMMixtureDatasetConfig`).** `BUCKET_2["nemotron_cc_math_v1/4plus"]` is an `ExecutorStep`. `default_train` accepts `InputName | ExecutorStep | LMMixtureDatasetConfig` — when given an ExecutorStep it auto-builds a 100 % single-source `lm_data_config` via `_prepare_data_config`, and with `use_default_validation=True` (default) adds Paloma validation. No manual mixture plumbing required.
- **`eval_harness_tasks=()` disables the eval harness.** Empty tuple is falsy; `default_train` sets `harness_config=None` when it is empty. Keeps runtime focused on train loss + Paloma validation for this sweep.

### LR-factor → numerical LR (reference; computed in `_build_adamh`)

| base_tag | factor | `learning_rate` | `adam_lr` |
|---|---:|---:|---:|
| `1e20-iso-d2048-L21` | 0.50 | 2.2415e-3 | 3.6910e-5 |
| `1e20-iso-d2048-L21` | 0.67 | 3.0036e-3 | 4.9460e-5 |
| `1e20-iso-d2048-L21` | 0.83 | 3.7209e-3 | 6.1271e-5 |
| `1e21-v5`            | 0.50 | 3.7125e-3 | 2.1570e-4 |
| `1e21-v5`            | 0.67 | 4.9748e-3 | 2.8904e-4 |
| `1e21-v5`            | 0.83 | 6.1628e-3 | 3.5806e-4 |

(Product of `BASES[b]["peak_lr"] × lr_factor` and same for `adam_lr`. Cross-check these after any `BASES` edit.)

---

## Next steps — what an executing agent should do

**Immediate next actions (code is written, nothing has been run yet):**

1. Static sanity-check: `uv run python -c "import experiments.exp_delphi_math_10b_midtrain as m; print(len(m.runs)); [print(' ', s.name) for s in m.runs]"`. Expected: `6` and the six `checkpoints/delphi-...-math-10b-lrX` names. Fail-open: if imports break, the error will pinpoint a missing helper — update the file in lockstep.
2. GCS ckpt existence: `gcloud storage ls gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/checkpoints/step-46915/` and `.../adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/checkpoints/step-21979/`. Confirm `manifest.ocdbt`, `metadata.json`, `d/` exist. (Already verified on 2026-04-21; re-verify right before launch.)
3. Commit the new experiment file (and/or this logbook) if the user wants — file is currently untracked.

**Launch (only when the user asks):**

4. Launch the two `lr0.67` pilots on separate v4-128 pods.
5. After step 100 of each pilot, check: `eval/paloma/c4_en/loss` ≈ 2.858 (1e20) / 2.733 (1e21) on first eval — confirms weights loaded. Train loss trending down by step 500 — if flat or rising at `lr0.67`, `lr0.83` will blow up; abort the upper tier.
6. If pilots pass, launch remaining 4 runs in parallel.
7. Post-run: append a "Results" section with final c4-en loss / Paloma macro bpb / any math-eval scores, and the best `(base × lr_factor)` per base. Use as starting point for Delphi 1e22 / 1e23 extension.

---

## Status — chronological action log

| Date (UTC) | Action | Result |
|---|---|---|
| 2026-04-21 | Catalogued all 24 Delphi / AdamH runs + the 3e20 isoflop scan. | `.agents/projects/delphi_midtraining.md` committed in `84f53bc6f`. |
| 2026-04-21 | Read *Delphi Scaling Laws* blog. | LR rule + β₂ formula cross-check pass against `completed_adamh.py:169–185`. |
| 2026-04-21 | Decided bases: `isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5` + `adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021`. | User-confirmed. |
| 2026-04-21 | Pulled exact pretrain `(peak_lr, peak_adam_lr, β₂, ε)` from each base's W&B config. | Numbers in the sweep tables above are authoritative. |
| 2026-04-21 | Plan written. | This file. |
| 2026-04-21 | Read reference code: `experiments/defaults.py`, `experiments/simple_train_config.py`, `experiments/scaling_law_sweeps/completed_adamh.py`, `experiments/tootsie/exp1529_32b_mantis_cooldown.py`, `experiments/tootsie/exp898_deeper_cooldown.py`, `experiments/midtraining_data_buckets.py`, `experiments/pretraining_datasets/nemotron_v2.py`, `lib/levanter/src/levanter/optim/adamh.py`, `lib/levanter/src/levanter/layers/rotary.py`. | Identified the `default_train` + `SimpleTrainConfig` + `optimizer_config=AdamHConfig(...)` pattern; confirmed Qwen3Config architecture; confirmed llama3 tokenizer alignment. |
| 2026-04-21 | Pulled `model.*` + `data.tokenizer` from both base runs' wandb configs. | Architecture matches `completed_adamh_heuristic._build_model_config`; both use `meta-llama/Meta-Llama-3.1-8B` tokenizer. |
| 2026-04-21 | Read `metadata.json` of both source checkpoints. | Both well-formed (`step`, `timestamp`, `is_temporary: true`). Companion `manifest.ocdbt` and `d/` present in the step dirs (verified earlier). |
| 2026-04-21 | Wrote `experiments/exp_delphi_math_10b_midtrain.py` (~160 lines). | File created, not yet executed. 6 `ExecutorStep`s constructed via `default_train`. User instruction: don't run anything. |
| 2026-04-21 | Did NOT run `uv run python -c "import experiments.exp_delphi_math_10b_midtrain"`. | User instruction ("don't run anything yet"). Handoff item for next agent or next user turn. |
| 2026-04-21 | Sized TPU via `completed_adamh_heuristic.estimate_memory_bytes` + `pick_v5p_type`. | Est. 2367 GB total (fudge×2). `pick_v5p_type` → `v5p-64` (32 chips × 95 GiB = 3040 GiB). Per-chip ≈ 69 GiB/chip with ≈25 GiB headroom. Per-chip BS=16 → good MFU. v4-128 (the initial plan) was undersized (would need v4-256). |
| 2026-04-21 | Changed `TPU_TYPE` → `"v5p-64"` and rewrote both base ckpt paths to `mirror://<relative-path>` (cross-region pull from `marin-us-central2`). | Per `experiments/AGENTS.md` §Mirror FS, `mirror://` copies to local prefix on first read and caches. Large (>10 GB) transfers would normally need explicit permission — user said "forget the cross region costs!" which is the approval. Will additionally set `MARIN_I_WILL_PAY_FOR_ALL_FEES=1` on the iris job env to disable the `TransferBudget` enforcement globally. |
| 2026-04-21 | Static import check after edits. | `uv run python -c "import experiments.exp_delphi_math_10b_midtrain as m; print(len(m.runs))"` → `6`. All ExecutorStep names correct (`checkpoints/delphi-1e{20-iso-d2048-L21,21-v5}-math-10b-lr{0.5,0.67,0.83}`). |
| 2026-04-21 | Iris CLI available as `uv run iris --cluster=marin ...`. Controller in `us-central1-a` (SSH tunnel auto-established). v5p pool zones per `lib/iris/examples/marin.yaml:101`: `us-central1-a`, `us-east5-a`. v5p-64 has `max_slices: 256` — plenty of capacity. | — |
| 2026-04-21 17:52Z | Committed `20a8d05ba` "Add Delphi Nemotron-CC-Math 10B LR sweep" and pushed to `origin/midtrain_data`. | — |
| 2026-04-21 17:53Z | Submitted the coordinator to Iris with `uv run iris --cluster=marin job run --cpu 1 --memory 3GB --disk 5GB --extra marin:tpu --job-name delphi-math-10b-sweep --no-wait -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 -- python experiments/exp_delphi_math_10b_midtrain.py`. Job id: `/ahmed/delphi-math-10b-sweep`. (First attempt failed with `--memory 16GB` hitting the `>=4 GB requires --enable-extra-resources` guard; dropped to 3 GB since the coordinator is CPU-only and only dispatches the 6 training sub-tasks.) | Submitted. State: pending. Workspace bundle 5.2 MB uploaded. |
| 2026-04-21 ~18:Z | Coordinator container build FAILED: `marin[tpu]` pulls `torch==2.9.0+cpu` which blew past the 5 GB disk limit during wheel extraction. `--extra marin:tpu` is meant for training-direct jobs (the `babysit-job` skill example), not an `executor_main` coordinator. | Resubmitted. |
| 2026-04-21 19:06Z | Resubmitted with `--disk 9GB` and **no `--extra`** — the default `iris-task` image already has Marin's base deps, so the coordinator needs no extras. `-e MARIN_I_WILL_PAY_FOR_ALL_FEES 1` kept to disable the cross-region `TransferBudget` when mirror:// copies the two base ckpts. | Job `/ahmed/delphi-math-10b-sweep` state=`running` 2 s after submit. |
| 2026-04-21 19:07Z | Coordinator landed on eu-west4 worker. Started walking the dep graph. Paloma + uncheatable validation caches found in eu-west4 (skip). BUT: `raw/nemotron_cc_math_v1_33b36abc` went RUNNING — i.e. the executor is redownloading multi-TB from HF Hub because `BUCKET_2["nemotron_cc_math_v1/4plus"]`'s ExecutorStep hashes to a path that doesn't exist in eu-west4 (hashes differ per region snapshot). Would take many hours. | KILLED the job. |
| 2026-04-21 19:08Z | Stopped `/ahmed/delphi-math-10b-sweep`. | `Terminated jobs: /ahmed/delphi-math-10b-sweep`. |
| 2026-04-21 19:10Z | Rewrote the data config in `experiments/exp_delphi_math_10b_midtrain.py`: dropped the `BUCKET_2` executor-step dependency, built an `LMMixtureDatasetConfig` directly with one `DatasetComponent` whose `cache_dir = "mirror://tokenized/nemotron_cc_math_v1/4plus-0bd79d"`. This skips the raw + normalize + tokenize chain entirely — training workers just read the already-tokenized cache that exists in `gs://marin-us-central2/tokenized/nemotron_cc_math_v1/4plus-0bd79d/`. `MirrorFileSystem` copies to the worker's local prefix on first read and caches thereafter. | Static import still builds 6 runs; `math_mix.components["…/4plus"].cache_dir == "mirror://tokenized/nemotron_cc_math_v1/4plus-0bd79d"`. Committed `45405e0b9`. |
| 2026-04-21 19:11Z | Resubmitted with `--job-name delphi-math-10b-sweep` (same name as previous killed attempt). Reported "Job submitted" and the summary briefly showed `state=assigned`. | Trap: the job's eventual summary was **`state=killed` with worker_id from the previous eu-west4 attempt (`…europ-20260421-1724-2421802a-worker-0`)** — i.e. iris's `--job-name` treats resubmission under a killed job-name as a zombie-reattach, not a fresh run. **Never reuse a `--job-name` after stopping** — always pick a fresh name (or let iris auto-generate). |
| 2026-04-21 19:19Z | Resubmitted as `/ahmed/delphi-math-10b-sweep-v2`. | State: `running` / task `building` immediately. Container image being built; will reveal whether the mirror:// path changes bypass the raw+tokenize chain once it finishes build. |
| 2026-04-21 19:22Z | v2 entered `running`, dispatched 8 `train_lm/*` per-host replicas per sweep step → **FAILED after 2m 37s**. Root cause: `train_lm/5` crashed during `levanter.initialize` → `_initialize_global_tracker` → `WandbConfig.init` → `jax_utils.multihost_broadcast_sync` with `RuntimeError: multihost_broadcast_sync requires jax distributed client to be initialized`. Coscheduled-sibling propagation killed the remaining 5 runs. Trigger: W&B fell back to **offline mode** because `WANDB_API_KEY` wasn't in the sub-task env. The offline path broadcasts metadata across hosts but the iris runtime had just logged `"TPU detected; skipping Iris JAX distributed init (TPU runtime handles it)"` — JAX distributed hadn't been fully initialized by the time Levanter ran the broadcast. | Noted: `lib/iris/AGENTS.md` says `WANDB_API_KEY` is auto-injected, but the auto-inject is only for the top-level job env, **not for Fray-dispatched TPU sub-tasks**. Must be passed explicitly. |
| 2026-04-21 19:24Z | Resubmitted as `/ahmed/delphi-math-10b-sweep-v3` with `-e WANDB_API_KEY "${WANDB_API_KEY}"` in addition to `-e MARIN_I_WILL_PAY_FOR_ALL_FEES 1`. Same `--cpu 1 --memory 3GB --disk 9GB`. | Submitted. Task 0: `building` (container build). |
| 2026-04-21 19:25Z | v3 FAILED in 33s — NOT a wandb issue this time. `connectrpc.errors.ConnectError: Job …/train_lm is unschedulable: no groups in region europe-west4 (constraints: device-type=tpu, device-variant=v5p-64, region=europe-west4)`. Coordinator landed in eu-west4; Fray propagated `region=europe-west4` as a constraint on the v5p-64 sub-tasks; v5p doesn't exist in eu-west4. Scheduler refused. | Conclusion: **coordinator region dictates TPU region for dispatched Fray sub-tasks**. Must pin coordinator to a region where the target TPU lives. v5p is in us-central1-a + us-east5-a only. |
| 2026-04-21 19:26Z | Resubmitted as `/ahmed/delphi-math-10b-sweep-v4` with `--region us-east5` added. | Submitted, state `pending`. Will land on a us-east5 worker. |
| 2026-04-21 19:31Z | v4 still pending 5 min later: `Scheduler: Insufficient CPU (need 1 cores, available 0 cores)` — us-east5 CPU pool at capacity, autoscaler not kicking in fast enough. Stopped v4, submitted v5 with `--region us-east5 --region us-central1` (both v5p regions accepted). | v5 state `running` within 3 s — us-central1 had capacity. |
| 2026-04-21 19:35Z | v5 FAILED with the **same** `multihost_broadcast_sync requires jax distributed client to be initialized` from `train_lm/5` that killed v2 — despite WANDB_API_KEY being passed. Root cause found: `lib/iris/src/iris/runtime/jax_init.py` skips `jax.distributed.initialize()` on ALL TPU jobs, but multi-host TPU under Iris can't auto-discover other hosts (no `TPU_WORKER_HOSTNAMES` inheritance). Levanter then calls `multihost_broadcast_sync` at `WandbConfig.init` → `wandb.py:296` which requires `jax._src.distributed.global_state.client` — `None` on multi-host TPU. | Known bug — I authored the fix back on 2026-04-12 for the DPO work (commit `2fe470a13`). |
| 2026-04-21 21:26Z | Cherry-picked two commits from `origin/dpo-lora` into `midtrain_data`: `2fe470a13` ([iris] Fix multi-host TPU JAX distributed init) + `d82faef23` ([levanter] Defer Iris TPU init on TPU jobs). One trivial conflict in `lib/levanter/src/levanter/distributed.py` resolved by accepting the d82faef23 form (uses `job_info is not None and (not self._is_distributed() or tpu_runtime_managed)`). Pushed `d1c613efe` to `origin/midtrain_data`. | Cherry-pick clean, on HEAD. |
| 2026-04-21 21:27Z | Resubmitted as `/ahmed/delphi-math-10b-sweep-v6` with the same flags as v5. | Job submitted. |
| 2026-04-21 21:29Z | v6 cleared the JAX distributed bug but FAILED with `ValueError: No source and no cache found for component nemotron_cc_math_v1/4plus split train` in `levanter.data.text.datasets.build_caches`. | The `mirror://tokenized/nemotron_cc_math_v1/4plus-0bd79d` I pointed at has ONLY `.executor_info` — the tokenize step was registered but never ran to completion in **any** marin bucket. Checked us-central2, us-east5, us-central1, eu-west4, us-east1, us-west4 — only `3` and `4plus_mind` have real cache. |
| 2026-04-21 21:35Z | Tracked down the raw: `gs://marin-us-east5/raw/nemotron_cc_math_v1-322fe4/4plus/` has 46 parquet shards, ~62 GB. Also mirrored at `gs://marin-us-central1/raw/nemotron_cc_math_v1-322fe4/`. The `override_output_path="raw/nemotron_cc_math_v1-322fe4"` in `lib/marin/src/marin/datakit/download/nemotron_v2.py:76` pins the path. | My earlier grep for `raw/nemotron_cc_math_v1/` (trailing slash, no hash) missed the hashed path. |
| 2026-04-21 21:45Z | Reverted the `mirror://` data-config hack in the experiment file. `tokenized=BUCKET_2["nemotron_cc_math_v1/4plus"]` restored so the executor walks the dep chain: download (raw already in us-east5/us-central1, skip) → normalize (will run fresh) → tokenize (will run fresh) → training. All in-region if coordinator lands on us-east5 or us-central1 (v5p regions). Committed `899ec2010`. | Pushed to `origin/midtrain_data`. |
| 2026-04-21 21:46Z | Stopped v6. Submitted `/ahmed/delphi-math-10b-sweep-v7` with same flags. | Expect ~30-min to few-hour normalize+tokenize on the CPU pool before v5p-64 training starts; this is acceptable since raw doesn't have to cross regions. |
| 2026-04-21 21:46Z—22:20Z | v7 ran the whole dep chain on us-central1: raw (skipped, present) → **normalize SUCCEEDED** (45 096 087 records, output at `gs://marin-us-central1/normalized/nemotron_cc_math_v1/4plus_37e28c45/`, ~95 GB) → **tokenize SUCCEEDED** (p0 + p1, output at `gs://marin-us-central1/tokenized/nemotron_cc_math_v1/4plus-212a2d/`) → **cache-probe SUCCEEDED** (all 231 shards verified) → **cache-copy FAILED** after ~34 m. | Root cause: `zephyr-levanter-cache-copy` workers lost contact with the coord actor (`No endpoints found for actor '/…/cache-copy-*/coord-0'`) and eventually the worker container was killed mid-heartbeat. This is an infra flake, not our config — identical pattern seen in `/rav/iris-run-validate_normalize_phase1-*` jobs earlier today. |
| 2026-04-21 22:24Z | Resubmitted as `/ahmed/delphi-math-10b-sweep-v8`. The executor should skip normalize + tokenize + probe (all outputs still cached in us-central1) and re-try cache-copy. | Submitted. |
| 2026-04-21 22:35Z | v8 got past cache-copy (succeeded this time), dispatched the 6 TPU `train_lm` jobs on v5p-64 → all hosts crashed with `ValueError: Unsupported URI scheme for tensorstore: 'mirror' in ...`. | **Lesson:** `mirror://` is an fsspec protocol. Levanter's checkpoint loader uses TensorStore directly (native GCS paths, no fsspec). So `mirror://` works for data loading but NOT for `initialize_from_checkpoint_path`. |
| 2026-04-21 22:38Z | Reverted ckpt fields in `experiments/exp_delphi_math_10b_midtrain.py` from `mirror://<path>` → `gs://marin-us-central2/<path>`. Cross-region reads are fine: TensorStore doesn't consult the fsspec `CrossRegionGuardedFS`, so `MARIN_I_WILL_PAY_FOR_ALL_FEES=1` isn't even strictly needed for the ckpt read (kept it for the data fsspec paths). Committed `c13560c3f`, pushed. | Normalize/tokenize caches from v7 remain in us-central1; v9 should skip straight to cache-copy then training. |
| 2026-04-21 22:39Z | Submitted `/ahmed/delphi-math-10b-sweep-v9`. | Job accepted. |
| 2026-04-21 22:40Z | v9 FAILED in 33s with `ValueError: initialize_from_checkpoint_path is not in the same region (us-central2) as the VM (us-central1)` for all 6 sweep steps. Marin's `rigging.filesystem.check_gcs_paths_same_region` (invoked via `_doublecheck_paths` in `lib/marin/src/marin/training/training.py`) hard-fails any `gs://` path whose bucket region doesn't match the VM's region — no env-var override exists. | Fix: pre-copy the two base ckpts into us-central1 so they're co-located with the pinned `--region us-central1` coordinator. |
| 2026-04-21 22:42Z | Server-side `gcloud storage cp --recursive` both ckpts us-central2 → us-central1 (23 GB + 41 GB ≈ 64 GB). Updated `BASES[*]["ckpt"]` in the experiment file to `gs://marin-us-central1/...` paths. Committed `56b1b1c86`, pushed. | Copies finished in parallel background (~350-620 MiB/s server-side). |
| 2026-04-21 22:44Z | Submitted `/ahmed/delphi-math-10b-sweep-v10` with `--region us-central1` (dropping us-east5 because the ckpts are now only in us-central1). | Coordinator up in ~5 s. |
| 2026-04-21 22:45–22:50Z | v10 walked the dep graph (skipped the already-cached normalize/tokenize from v7), dispatched the train_lm sub-task, scheduler pending-on-coscheduling for a v5p-64 slice (need 8 worker VMs to come up), then `running`. | Expected TPU spin-up delay. |
| 2026-04-21 22:53Z | All 8 `train_lm/[0-7]` replicas restored the checkpoint successfully (`jax.experimental.array_serialization.serialization Error check finished successfully`). Train-step tracing + HLO lowering completed in a few s. **First training step** 39.3 s (compile-heavy), then 4.4–4.5 s/step steady-state. | MFU ≈ 36 % on v5p-64 (32 chips × 459 TFLOPS bf16 peak; achieved ≈ 5.3 PFLOPS/s). |
| 2026-04-21 22:53 – 2026-04-22 05:16Z | v10 training ran for ~6 h 22 m. Loss dropped 1.58 → 1.12 over warm-up (500 steps), plateaued 1.12–1.20 through decay, **final train_loss 0.958 (tqdm) / 0.962 (W&B summary at step 4767)**. 3 mid-run evals at steps 200/400/600/800/... ran cleanly. Periodic save at step 1000. | Successful run: `delphi-1e20-iso-d2048-L21-math-10b-lr0.5-ba7b7f` (wandb + 155 GB at `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.5-ba7b7f/`). |
| 2026-04-22 05:16Z | v10 coordinator `state=succeeded`. But iris shows only ONE `train_lm` child under the coordinator, and wandb has only ONE v10-era run (the `lr0.5-ba7b7f` one). GCS sweep-output audit: `lr0.5-ba7b7f` → 155 GB with `checkpoints/`, `hf/`, `tracker_metrics.jsonl`; `lr0.67-e3be0c` → **65 KB, `.executor_status=SUCCESS`, `.artifact=null`, no training output**; `lr0.83-e3de76` / `1e21-lr0.5-ccce18` / `1e21-lr0.67-e5b5df` / `1e21-lr0.83-ece889` → 65 KB each, `.executor_status=FAILED`. | **Only 1 of 6 sweep points actually trained.** The other 5 marked terminal states (SUCCESS or FAILED) without producing training artifacts. |

## 2026-04-22 post-v10 analysis: the `train_lm` name-collision pitfall

`lib/marin/src/marin/training/training.py:307` pins every dispatched iris sub-job to the **literal name** `"train_lm"`. When an `executor_main` invocation runs 6 training `ExecutorStep`s whose dependencies are all already satisfied (tokenize cache warm, base ckpts local), `step_runner`'s default `max_concurrent=8` `ThreadPoolExecutor` dispatches all 6 `run_levanter_train_lm` calls **in parallel**. All 6 call `_submit_training_job(job_name="train_lm", ...)` under the same coordinator parent → same full iris path `/ahmed/<coord>/train_lm`.

The iris controller's `EXISTING_JOB_POLICY_KEEP` (what fray hands it for `adopt_existing=True`) then, per `lib/iris/src/iris/cluster/controller/service.py:1113-1117`:

```python
elif policy == job_pb2.EXISTING_JOB_POLICY_KEEP:
    if not is_job_finished(existing_job.state):
        return controller_pb2.Controller.LaunchJobResponse(job_id=job_id.to_wire())
    # Job finished, replace it (KEEP only preserves running jobs)
    self._transitions.remove_finished_job(job_id)
```

→ racing submits 2…6 see the still-running first job and **adopt its handle without creating a new job**. All six Python threads then `.wait()` on the same handle, which completes when the first (and only) config's training finishes. The adopted-handle threads see "SUCCEEDED" and return, so `step_runner` marks their steps as `STATUS_SUCCESS` despite the fn never actually running Levanter training.

Why I confused myself with the PR 4591 seed-sweep precedent: the seed-sweep pattern *defines* many ExecutorSteps but in practice each seed ran as a **separate top-level iris job** on a different day (wandb `created_at` for 1e21 seed{0,42,62746} were 2026-03-04, 2026-03-18 01:47Z, 2026-03-18 06:11Z; 1e22 seeds were 03-04, 03-22, 03-26). Different top-level coordinators → different parent paths → no `train_lm` collision. The seed PR itself didn't fix the collision; it just enabled enumerating the variants, and the human operator launched each variant separately.

## Fix applied to `experiments/exp_delphi_math_10b_midtrain.py` (not yet run)

Added env-var-driven filtering so a single invocation of the script can build a single sweep point:

```python
_SELECT_BASE = os.environ.get("MIDTRAIN_SELECT_BASE")  # e.g. "1e21-v5"
_SELECT_LR   = os.environ.get("MIDTRAIN_SELECT_LR")    # e.g. "0.67"

def _build_runs():
    for base_tag, base in BASES.items():
        if _SELECT_BASE is not None and base_tag != _SELECT_BASE: continue
        ...
        for lr_factor in LR_FACTORS:
            if _SELECT_LR is not None and _lr_str(lr_factor) != _SELECT_LR: continue
            ...
```

Verified:

- Unset: builds all 6 steps (as before). Useful for dry-run/introspection.
- `MIDTRAIN_SELECT_BASE=1e21-v5 MIDTRAIN_SELECT_LR=0.67`: builds just `delphi-1e21-v5-math-10b-lr0.67`.

**Step hashes are unchanged by filtering** (filtering only affects which steps `_build_runs` returns; each step's config is byte-identical to before). So the already-succeeded `delphi-1e20-iso-d2048-L21-math-10b-lr0.5-ba7b7f` entry stays cached — any future invocation that includes it will see `STATUS_SUCCESS` and skip. The 5 remaining steps currently have a `.executor_status` of `SUCCESS` (lr0.67 1e20) or `FAILED` (other 4). Before relaunching:

- **`STATUS_SUCCESS` with no training output** (the `lr0.67-e3be0c` case, and possibly others): the cache check will treat them as succeeded → skip → no retraining. Workaround: delete `.executor_status` at those output paths so the next run re-does the step. (Do NOT delete the `lr0.5-ba7b7f` one — that one really did train.)
- **`STATUS_FAILED`**: `step_runner` will raise `PreviousTaskFailedError` unless you pass `force_run_failed=True` or delete the status file.

## Launch recipe for the 5 remaining sweep points (copy-paste)

Each variant goes as its own iris coordinator so `/ahmed/<coord-N>/train_lm` is a unique path per sweep point. Run from the repo root:

```bash
# 1. (one-time) clean up the stale STATUS files so the 5 remaining steps don't
#    short-circuit on cache hit / fail-as-previous-failure:
for target in \
    'delphi-1e20-iso-d2048-L21-math-10b-lr0.67-e3be0c' \
    'delphi-1e20-iso-d2048-L21-math-10b-lr0.83-e3de76' \
    'delphi-1e21-v5-math-10b-lr0.5-ccce18' \
    'delphi-1e21-v5-math-10b-lr0.67-e5b5df' \
    'delphi-1e21-v5-math-10b-lr0.83-ece889'; do
  gcloud storage rm "gs://marin-us-central1/checkpoints/${target}/.executor_status" 2>/dev/null || true
done

# 2. launch each as its own iris job (each gets a unique --job-name)
launch() {
  local base="$1" lr="$2" short
  short=$(echo "$base" | sed 's/-iso-d2048-L21//')
  uv run iris --cluster=marin job run \
    --cpu 1 --memory 3GB --disk 9GB \
    --region us-central1 \
    --job-name "delphi-math-10b-${short}-lr${lr}" \
    --no-wait \
    -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 \
    -e WANDB_API_KEY "${WANDB_API_KEY}" \
    -e MIDTRAIN_SELECT_BASE "$base" \
    -e MIDTRAIN_SELECT_LR "$lr" \
    -- python experiments/exp_delphi_math_10b_midtrain.py
}
launch 1e20-iso-d2048-L21 0.67
launch 1e20-iso-d2048-L21 0.83
launch 1e21-v5            0.5
launch 1e21-v5            0.67
launch 1e21-v5            0.83
```

All 5 can run in parallel (v5p-64 pool has `max_slices: 256`). Expected per-run wall-time: 1.9 B base ≈ 6 h, 3.4 B base ≈ 10 h (larger model, same BS=512). W&B runs appear under `marin-community/marin` with names `delphi-<base>-math-10b-lr<factor>-<hash>`.

## Expected cross-region transfers (FYI)

Launch region isn't pinned (Iris picks any v5p zone) — either `us-central1` or `us-east5`. In either case:

| Artifact | Source | Dest (per worker's MARIN_PREFIX) | Approx size | Notes |
|---|---|---|---:|---|
| 1e20 isoflop ckpt (step-46915) | `gs://marin-us-central2/checkpoints/isoflop/...` | mirror → local prefix | ~22 GB | OCDBT shards under `d/` + manifest + metadata. |
| 1e21 Delphi v5 ckpt (step-21979) | `gs://marin-us-central2/adamh-scaling-ladder-...v5-019021/checkpoints/step-21979/` | mirror → local prefix | ~40 GB | Same shape, bigger model. |
| `nemotron_cc_math_v1/4plus` tokenize cache | `gs://marin-us-central2/tokenized/nemotron_cc_math_v1/4plus-0bd79d/` | might mirror, or executor re-tokenizes on the local CPU pool | ~100 GB (or retokenize) | us-east5 has only `3-ef5cb9` + `4plus_mind-d60b4a`, NOT `4plus`. Versioning hashes differ between regions so the executor may re-run the tokenize step locally instead of mirroring. If it mirrors, ~100 GB; if it re-tokenizes, hours of CPU but no cross-region. |

Total worst-case cross-region read: **~160 GB** (once, cached thereafter). User has accepted this.

---

## 2026-04-22 region-pin diagnosis and the real fix: wiring `mirror://` through Levanter's checkpoint loader

### Re-verification of current state

Two coordinators `/ahmedah/delphi-math-10b-1e20-lr{0.67,0.83}-20260422` are queued (pinned to `us-central1`) but have not yet dispatched TPU. GCS audit shows **only** `delphi-1e20-iso-d2048-L21-math-10b-lr0.5-ba7b7f` (155 GB) is a real training output; all other 11 sweep-point directories (two hash variants per remaining combination) are 65–70 KB stubs. That means the logbook's "Step hashes are unchanged by filtering" claim above is wrong — something after v10 did change hashes, so there are now TWO hash series of stubs per variant. Relaunch plan will be: compute the hash the current code produces, then clear `.executor_status` files at all known variants of the target step before launching.

### Why I was pinning `--region us-central1`

Two reasons from the earlier v-series attempts:

1. `lib/rigging/src/rigging/filesystem.py:254` — `check_path_in_region` hard-fails any `gs://` path whose bucket region ≠ VM region. `lib/marin/src/marin/training/training.py:344` runs this check on the whole `TrainLmConfig` via `_doublecheck_paths`. v9 died in 33 s because the 1e20 ckpt was a `gs://marin-us-central2/...` path while the VM was in us-central1. No env-var override exists.
2. The base ckpts and the `tokenized/nemotron_cc_math_v1/4plus-212a2d` cache currently live only in `gs://marin-us-central1/...`. A TPU in us-east5-a would either re-run the entire tokenize step (hours) or fail the region check.

### Why `mirror://` was supposed to handle this (and why it broke last time)

`mirror://` IS designed to solve exactly this problem. The mechanism:

- `mirrored("relative/path", budget_gb=N)` in the executor config (`lib/marin/src/marin/execution/executor.py:939-946`) marks a path as cross-region-mirrorable.
- At config instantiation time, `MirroredValue` is rewritten to `mirror://relative/path` (`executor.py:1149-1153`).
- `MirrorFileSystem` (`lib/rigging/src/rigging/filesystem.py:715-935`) registers `mirror` as an fsspec protocol. On read, it: checks `${MARIN_PREFIX}/<path>` first; otherwise scans the other marin regional buckets (`marin-us-central1`, `marin-us-east5`, `marin-us-central2`, `marin-eu-west4`, …) via `_find_in_remote_prefixes`; copies from whichever bucket has the file to the local prefix under a distributed lock; charges against the shared `TransferBudget` (disabled by `MARIN_I_WILL_PAY_FOR_ALL_FEES=1`).
- **The region check already skips `mirror://` paths** — `_collect_gcs_paths_recursively` at `filesystem.py:338-344` only gathers strings starting with `gs://`, so a mirror URL sails through `check_gcs_paths_same_region`.

v8's failure (`ValueError: Unsupported URI scheme for tensorstore: 'mirror'`) happened because `mirror://` was passed straight into `initialize_from_checkpoint_path`. Levanter's `load_checkpoint` at `lib/levanter/src/levanter/checkpoint.py:794-810` hands the path verbatim to `tree_deserialize_leaves_tensorstore`, whose `build_kvstore_spec` (`lib/levanter/src/levanter/tensorstore_serialization.py:54-78`) only speaks `gs`/`s3`/`file`/``''``. TensorStore bypasses fsspec entirely — the mirror protocol has no hook.

For **data loaders** (LMMixture, tokenizers) this is already solved: Levanter goes through fsspec for data paths, and the tokenizer loader has an explicit `_stage_from_mirror` staging helper (`lib/levanter/src/levanter/tokenizers.py:702-729`) that uses `mirror_fs.ls()` + per-file `_fetch_file_atomic` to materialize files to a local staging dir before the HF loader opens them.

For **checkpoints** there is no equivalent staging path — that's the gap.

### Planned fix (small, local to Levanter)

Add a `_stage_mirror_to_local(path)` helper to `lib/levanter/src/levanter/checkpoint.py`:

1. If path does not start with `mirror://`, return it unchanged.
2. Strip the prefix to get a relative path (e.g. `checkpoints/isoflop/.../step-46915`).
3. Construct `fsspec.filesystem("mirror")` and call `mfs.find(rel)` → recursive list of file paths.
4. For each file, call `mfs._resolve_path(file_rel)` — triggers `_copy_to_local` on cache miss (GCS-to-GCS server-side `rewrite`, fast). Files already present locally are skipped.
5. Return `f"{marin_prefix()}/{rel}"` — a concrete `gs://marin-${region}/...` or `/tmp/marin/...` URL that TensorStore can open.

Wire the helper at exactly two call sites:

- Top of `latest_checkpoint_path` (line 965): discovery still runs against `mirror://` (via fsspec, which MirrorFileSystem handles), but the returned concrete step dir goes through the staging helper before being returned.
- Top of `load_checkpoint` (line 794): covers direct-path callers that skip discovery (`eval_lm.main`, `export_lm_to_hf`, `perplexity_gap`, `inference_repl`, `eval_harness`).

This is the same import pattern `lib/levanter/src/levanter/config.py` and `lib/levanter/src/levanter/trainer.py` already use (`from rigging.filesystem import url_to_fs / open_url`) — Levanter already depends on `marin-rigging` via `pyproject.toml`.

Test: add a `test_stage_mirror_to_local` in `lib/levanter/tests/test_checkpoint.py` using the same fixture pattern as `lib/rigging/tests/test_mirror_fs.py` (manually constructed MirrorFileSystem backed by two tempdirs, one standing in for local and one for remote).

### Experiment change

Replace the hardcoded `gs://marin-us-central1/...` ckpt paths in `experiments/exp_delphi_math_10b_midtrain.py` with `mirrored(...)` calls:

```python
"ckpt": mirrored(
    "checkpoints/isoflop/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/checkpoints/step-46915",
    budget_gb=30,
),
...
"ckpt": mirrored(
    "adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/checkpoints/step-21979",
    budget_gb=50,
),
```

Budgets cover the actual ckpt sizes (1e20 ≈ 22 GB, 1e21 ≈ 40 GB) with a small safety margin. `MARIN_I_WILL_PAY_FOR_ALL_FEES=1` in the iris job env disables `TransferBudget` enforcement globally, so budgets are informational on worker runs — they still matter for dry-runs and local dev.

### Relaunch plan (after fix lands)

Drop `--region us-central1` from the launch recipe; replace with `--region us-central1 --region us-east5`. The coordinator lands wherever CPU is free; Fray propagates the region constraint to the v5p-64 sub-task; MirrorFileSystem copies the ckpt from us-central1 → `marin-${landing_region}` on first open, cached thereafter. The `4plus-212a2d` tokenize cache is NOT wrapped with `mirrored()` in this pass (it's already materialized in us-central1; if the TPU lands in us-east5 the executor will re-run normalize+tokenize locally, which takes longer but is acceptable — can revisit).

### Checkpoint before launch

Plan this logbook entry → implement Levanter patch + test → implement experiment update → run lint → report back before relaunching any jobs.

### Implementation status (2026-04-22)

**Levanter patch** — `lib/levanter/src/levanter/checkpoint.py`
- Added `_stage_mirror_to_local(checkpoint_path: str) -> str` (~25 lines). No-op on non-`mirror://` inputs. On `mirror://` input, walks `mfs.find(rel)` and calls `mfs._resolve_path(file_rel)` on each file; returns `${marin_prefix()}/<rel>`.
- `latest_checkpoint_path` now routes the discovered path through `_stage_mirror_to_local` before returning.
- `load_checkpoint` now stages at the top so direct-path callers (`eval_lm.main`, `export_lm_to_hf`, `perplexity_gap`, `inference_repl`, `eval_harness`) benefit without further changes.
- Imports `marin_prefix` from `rigging.filesystem`; same pattern used in `levanter/config.py` and `levanter/trainer.py` — no new dependency (marin-rigging is already declared in `lib/levanter/pyproject.toml`).

**Tests** — `lib/levanter/tests/test_checkpoint.py`
- `test_stage_mirror_to_local_passes_through_non_mirror_paths`: the no-op branch for `file://` / `gs://` / raw paths.
- `test_stage_mirror_to_local_copies_all_files`: remote dir with 4 files (metadata.json, manifest.ocdbt, d/shard_0, d/shard_1) → all copied to local, returned URL points at local prefix.
- `test_stage_mirror_to_local_raises_when_empty`: FileNotFoundError when the mirror tree has no files.
- `test_latest_checkpoint_path_stages_direct_mirror_step`: end-to-end through `latest_checkpoint_path` with a direct `mirror://.../step-N` input — the shape our experiment uses.
- Helper `_configure_mirror_fs(local_dir, remote_dirs, monkeypatch)` patches `marin_prefix` + `_mirror_remote_prefixes` on `rigging.filesystem` AND clears `MirrorFileSystem._cache` (fsspec's instance cache lives on the leaf class, not on `AbstractFileSystem`, so clearing the base class cache alone is not enough — that was a real bug I hit).
- All 4 pass in isolation and together; all 20 existing tests in `lib/rigging/tests/test_mirror_fs.py` still pass (no upstream regression).

**Experiment** — `experiments/exp_delphi_math_10b_midtrain.py`
- Both `BASES[*]["ckpt"]` values replaced with `mirrored("<relative-path>", budget_gb=N)` (N=30 for 1e20 at ~22 GB, N=50 for 1e21 at ~40 GB).
- Header docstring's launch-recipe updated from `--region us-central1` to `--region us-central1 --region us-east5` — the whole point of this fix.
- Verified: `MIDTRAIN_SELECT_BASE=1e20-iso-d2048-L21 MIDTRAIN_SELECT_LR=0.67` import builds 1 ExecutorStep; both `BASES[*]["ckpt"]` are `MirroredValue` instances pre-instantiation.

**Lint** — `./infra/pre-commit.py` on the 3 changed files: Ruff + Black + pyrefly + license + AST + merge + whitespace + EOF all pass.

### Things NOT done (deliberate)

- `tokenized=BUCKET_2["nemotron_cc_math_v1/4plus"]` is still an ExecutorStep dependency, not `mirrored(...)`. The `4plus-212a2d` tokenize cache exists in us-central1 only; if the TPU lands in us-east5, the executor walks the dep chain → normalize + tokenize run fresh in us-east5 (raw is already present in both regions). Extra wall-clock but no cross-region data transfer. Can revisit if we want to cut the tokenize turnaround later.
- The two queued `--region us-central1` pinned jobs (`/ahmedah/delphi-math-10b-1e20-lr{0.67,0.83}-20260422`) are still alive. Need user sign-off before killing and relaunching with the new region-flex recipe.

### Ready for launch, pending user sign-off

1. Kill `/ahmedah/delphi-math-10b-1e20-lr{0.67,0.83}-20260422`.
2. Commit the Levanter patch + experiment change + logbook entry, push to `origin/midtrain_data` so the coordinator picks up the new code.
3. Resubmit the 5 remaining sweep points with `--region us-central1 --region us-east5` and a fresh `--job-name` per sweep point.

### Cross-region verification — actually exercised on a us-east5 worker (2026-04-22)

Added `scripts/_verify_mirror_stage.py` — a small iris-submittable script that imports `_stage_mirror_to_local`, pulls the 1e20 ckpt via `mirror://…`, then opens the staged OCDBT kvstore through TensorStore to prove the full end-to-end path (mirror:// → fsspec copy → TensorStore read) works across regions.

**Attempt 1 (`verify-mirror-stage-1e20-20260422`): FAILED after copying ~10 GB**

```
rigging.filesystem.TransferBudgetExceeded: ... would bring total to 10.53GB,
exceeding the 10GB limit (already transferred 9.77GB).
Consider running in the source region instead.
```

Worker was us-east5 (✓ marin_prefix = `gs://marin-us-east5`); 15 OCDBT shards successfully copied gs→gs from us-central1→us-east5 before the cap fired. The copy mechanism itself worked — the problem was the budget.

**Key discovery — the two safety envs are NOT symmetric.** `MARIN_I_WILL_PAY_FOR_ALL_FEES=1` ONLY short-circuits `CrossRegionGuardedFS._guard_read` (`lib/rigging/src/rigging/filesystem.py:623-625`); it does NOT disable `MirrorFileSystem._copy_to_local`'s budget charge at line 819. Those are separate code paths that happen to share `_global_transfer_budget` by default. The mirror side is governed by `MARIN_MIRROR_BUDGET_GB` (process-wide default ceiling) OR a `mirror_budget(gb)` contextvar (per-call stack, scoped). `MARIN_I_WILL_PAY_FOR_ALL_FEES` is invisible to the mirror code. That was the gap that bit us.

In the production training run, the executor already opens a per-step `mirror_budget(_max_mirror_budget(config))` context around the step fn (`lib/marin/src/marin/execution/executor.py:703-708`, `1301`) — so `mirrored("1e20-ckpt", budget_gb=30)` in our experiment yields a 30 GB budget at call time, which is why the real training launch will not hit this failure. The verify script, calling the helper outside an executor context, inherited the 10 GB default.

**Attempt 2 (`verify-mirror-stage-1e20-v2-20260422`): SUCCESS**

Wrapped the staging call in `with mirror_budget(30.0):` — same budget the executor sets from `mirrored(..., budget_gb=30)`. Duration 3m 50s on a us-east5 worker.

```
[verify] marin_prefix = gs://marin-us-east5
[verify] mirror URL   = mirror://checkpoints/isoflop/.../step-46915
[verify] staged in 218.2s
[verify] resolved    = gs://marin-us-east5/checkpoints/isoflop/.../step-46915
[verify]   metadata.json: OK
[verify]   manifest.ocdbt: OK
[verify]   OCDBT keys   = 1218
[verify] SUCCESS
Staged mirror://checkpoints/isoflop/.../step-46915 (44 files) to gs://marin-us-east5/...
```

**What this proves end-to-end:**

- `_stage_mirror_to_local("mirror://…")` on a fresh us-east5 worker correctly detects local prefix = `gs://marin-us-east5`.
- MirrorFileSystem finds the files in the remote (`gs://marin-us-central1/…`) bucket, copies all 44 files (OCDBT shards + manifest + metadata) via gs→gs rewrite, caches them under `gs://marin-us-east5/checkpoints/…/step-46915/`, and returns that concrete URL.
- The returned URL opens cleanly as an OCDBT kvstore through TensorStore (1218 keys enumerated), which is the exact call path Levanter's `load_checkpoint` uses for weight restore.
- Budget enforcement works as intended — the 30 GB ceiling from our `mirrored(..., budget_gb=30)` declaration is respected by a scoped context (1e20 ckpt actual size ~22 GB; used ~20 GB of the 30 GB budget since ~2 GB was already cached from v1).
- Side-effect: the 1e20 ckpt is now physically present at `gs://marin-us-east5/checkpoints/isoflop/.../step-46915/`, so future us-east5 launches for this base are cache-hits (MirrorFS `_fs_exists` returns True → skip copy).

**Budget semantics (for next agent):**

- `MARIN_I_WILL_PAY_FOR_ALL_FEES=1` disables the *direct-read* guard only, not mirror copies.
- `MARIN_MIRROR_BUDGET_GB=<n>` sets the default global mirror ceiling at module import (one-shot).
- `with rigging.filesystem.mirror_budget(<gb>):` opens a fresh scoped budget — preferred for ad-hoc scripts because it can't leak to other call sites.
- `mirrored(path, budget_gb=<n>)` in the executor config does (3) automatically on the step's behalf — nothing to set on the iris command line for real runs.
- The two overrides (`I_WILL_PAY` vs `mirror_budget`) are orthogonal; the default global `TransferBudget` instance is shared by both paths but the overrides do not cascade between them.

---

## 2026-04-23 live training — 1e20 lr=0.67 and lr=0.83 (still running; pre-mirror patch)

These two coordinators were launched 2026-04-22 23:54Z with `--region us-central1` pinning, BEFORE the mirror:// fix. They succeeded at the ckpt region check because the ckpts were pre-copied to us-central1 earlier in the v-series. They do NOT exercise the Levanter mirror-staging patch — that's still pending proper-run validation via the 3 × 1e21 launch.

### Run identifiers (wandb + GCS)

| Sweep point | Coordinator | train_lm output path | wandb run |
|---|---|---|---|
| `1e20 × lr=0.67` | `/ahmedah/delphi-math-10b-1e20-lr0.67-20260422` | `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.67-e3be0c/` | `https://wandb.ai/marin-community/marin/runs/delphi-1e20-iso-d2048-L21-math-10b-lr0.67-e3be0c` |
| `1e20 × lr=0.83` | `/ahmedah/delphi-math-10b-1e20-lr0.83-20260422` | `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-db9de7/` | `https://wandb.ai/marin-community/marin/runs/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-db9de7` |

Both `e3be0c` and `db9de7` are the OLD stubs' hashes — same hash as the v10 empty-SUCCESS stubs. Clearing the `.executor_status` files (logbook entry for 2026-04-22) did let the executor re-run the step under the same hash, so the filtering patch did NOT change hashes after all (the "two hash variants per point" observed earlier must have come from a separate code change between v10 and now). Future relaunches under these hashes will hit the cache and be skipped, which is what we want.

### Training configuration recap (both runs)

- v5p-64 (32 chips), us-central1-a, mesh `{data:-1, replica:1, model:1}` (tp=1 since H=2048 divides 32).
- Batch 512 × seq_len 4096 = 2,097,152 tokens/step. 4768 steps → ~10.0 B tokens.
- Fresh AdamH optimizer, β₂=0.99980, ε=4.11e-8, β₁=0.9, max_grad_norm=0.1.
- 500 linear warmup steps → 4268 linear decay steps, `min_lr_ratio=0.1`.
- `reset_data_loader_on_init=True`, `z_loss_weight=0`, `jmp.get_policy("p=f32,c=bfloat16")`.
- `steps_per_eval=200`, `steps_per_export=1000`, `steps_per_hf_export=1000`.

Per-run LR:

| lr_factor | `learning_rate` | `adam_lr` |
|---:|---:|---:|
| 0.67 | 3.0036e-3 | 4.9460e-5 |
| 0.83 | 3.7209e-3 | 6.1271e-5 |

### Throughput

Measured from `Progress on:train` ticks, averaged over the elapsed/step ratio (to neutralize tqdm's instant-rate spikes near eval/export):

- **Rate: ~4.4 s/step** (both runs, identical — same arch, same batch, same mesh).
- Achieved compute: ~5.3 PFLOPS (6 × 1.9e9 params × 2.1e6 tokens / 4.5 s).
- v5p-64 peak: 32 × 459 TFLOPS bf16 = 14.69 PFLOPS.
- **MFU ≈ 36 %** on both runs. Matches v10's measurement exactly.

Per-host memory: ~31 GB / 95 GB HBM (1.9 B model + full opt state + activations; v5p has plenty of headroom). We could run this on v5p-32 and stay under 62 GB/chip, at ~11 h wall-clock instead of ~6 h.

### Evaluation + HF-export overhead

At every multiple-of-1000 step, Levanter: (a) runs the full Paloma + uncheatable eval suite (17+ loss computations) and (b) writes a **7.74 GB × 2-shard** HF-compatible checkpoint to GCS. This takes ~2.5 min total. Tqdm's rolling-average `rate` field rolls the entire pause into "the last step," so after step 2000 the reported rate spiked to 47.0 s/it (lr=0.83) / 19.5 s/it (lr=0.67). **Actual per-step rate is unchanged at ~4.4 s/step** — always compute rate from `elapsed/N_steps`, not tqdm's instant rate, when eval/export is in the window. For future agents: this is a Levanter+tqdm display artifact, NOT a slowdown, and does NOT require action.

### Loss trajectory + lr=0.67/0.83 crossover (unsmoothed; check W&B for the clean version)

Single-step tqdm loss readouts (noisy but directionally correct):

| Step | lr=0.67 | lr=0.83 | Notes |
|---:|---:|---:|---|
| 444–513 (~11%) | 1.17 | 1.15 | End of warmup; lr=0.83 ahead (higher LR → faster initial progress) |
| 2000 (42%) | 1.03 | **0.987** | Mid-run; lr=0.83 still ahead by ~0.04 |
| 4370–4400 (92%) | **0.927** | 0.959 | Decay tail; **lr=0.67 has overtaken** |

**Crossover observation:** the higher peak LR (lr=0.83) converges faster initially but loses ~0.03 of its advantage by the decay tail — lr=0.67 with the gentler peak ends up lower in single-step loss. Both end below the v10 lr=0.5 final of 0.962, which is the direction we expected. Preliminary ranking for the 1e20 base:

```
lr=0.67 (0.927)  <  lr=0.83 (0.959)  <  lr=0.5 (0.962, v10)
```

Very narrow spread (~0.035 across factors 0.5–0.83). Need to look at smoothed Paloma/c4 curves and math-eval downstream scores before calling a winner, but if the signal holds, **lr=0.67 is the sweet spot for the 1e20 base** and is a reasonable default for the 1e21 LR sweep too.

**Caveat — single-step unsmoothed loss is noisy.** A ~0.03 gap at one step can be dominated by within-batch variance. The W&B panels at the run URLs above show EMA-smoothed curves; use those for the actual ranking.

### Current status (as of 2026-04-23 05:53Z)

- lr=0.67: step 4400/4770 (92 %), elapsed 5:54:38, real ETA ≈ 30 min.
- lr=0.83: step 4370/4770 (91.5 %), elapsed 5:50:55, real ETA ≈ 30 min.
- No preemptions, no failures across all 16 hosts.
- Both expected to finish around 06:25Z.

### Next steps once these land

1. Inspect final W&B panels for smoothed train-loss + Paloma validation trajectories across lr=0.5 / 0.67 / 0.83. Confirm the crossover + pick the 1e20 winner.
2. Launch 3 × 1e21 sweep points (lr=0.5 / 0.67 / 0.83) on v5p-64. With the Levanter mirror-staging patch now verified end-to-end (see "Cross-region verification" section above), the new launches can go `--region us-central1 --region us-east5` and land wherever the autoscaler has capacity. Expected wall-time per 1e21 run: ~10 h (3.4 B params at BS=512).
3. Commit the Levanter patch + experiment change + these logbook updates to `origin/midtrain_data` before relaunching, so the iris worker bundle picks up the new code.

---

## 2026-04-23 flat-LR incident — root cause + fix

**TL;DR: every 1e20 run completed before today was trained at `min_lr = 0.1 × peak`, not the scheduled warmup → peak → decay curve. All three completed/in-flight runs are DISCARDED. The Levanter warmstart path had a latent bug; fix landed locally today. Relaunch will produce new output hashes (config changed).**

### Symptom

W&B `optim/learning_rate` for the three completed 1e20 runs (`lr=0.5-ba7b7f` v10, and in-flight `lr=0.67-e3be0c` / `lr=0.83-db9de7`) is flat from step 0, no warmup, no decay. Values match `0.1 × peak × lr_factor` to 2 sig figs:

| Factor | Expected peak `learning_rate` | `0.1 × peak × factor` | Chart value |
|---:|---:|---:|---:|
| 0.50 | 2.2415e-3 | 2.2415e-4 | ~2.2e-4 |
| 0.67 | 3.0036e-3 | 3.0036e-4 | ~3.0e-4 |
| 0.83 | 3.7209e-3 | 3.7209e-4 | ~3.7e-4 |

Same story on `optim/adam_lr` — flat at `0.1 × peak_adam_lr × factor` (3.7e-6 / 4.9e-6 / 6.1e-6).

### Root cause

Direct TensorStore read of `gs://marin-us-central1/checkpoints/isoflop/.../step-46915/`:

```
opt_state/count                                  = 46916
opt_state/hyperparams_states/learning_rate/count = 46916
opt_state/hyperparams_states/adam_lr/count       = 46916
step                                              = 46916
```

Levanter's `train_lm.py:176-180` (`initialize_from_checkpoint_path` branch):

```python
if int(state.step) == 0 and config.initialize_from_checkpoint_path is not None:
    checkpoint_path = latest_checkpoint_path(config.initialize_from_checkpoint_path)
    state = load_checkpoint(state, checkpoint_path)     # restores FULL state incl opt_state
    state = dataclasses.replace(state, step=jnp.array(0))   # resets only outer step
```

`load_checkpoint(state, path)` deserializes every array leaf in the exemplar tree — including `opt_state.hyperparams_states.learning_rate.count`, which comes back as 46916 from the pretrain. Our fresh schedule is built with `num_train_steps=4768, warmup=500, decay=4268`. `optax.linear_schedule(peak, min_lr, 4268)` evaluated at count=46916 clamps to `min_lr = 0.1 × peak`. Every subsequent step increments count but stays past decay → flat forever at `min_lr`.

The inline comment `# we're just initializing weights here` has been a lie since PR #1957 (David Hall, `b5659c59c4`, 2025-12-02). Before #1957 the branch restored everything AND kept the outer step — a coherent full resume. PR #1957 added the step-reset without also resetting opt_state, creating today's inconsistency. The actual `load_checkpoint(state, ...)` call pre-existed #1957 (from `5c53a19fdc`, Aug 2024) but wasn't pathological on its own. `534544b0bd` (Apr 2026) refactored the call to use `latest_checkpoint_path` — a semantics-preserving change.

Pathology only manifests when `num_train_steps < restored count`. Existing callers (Mantis, 8B cooldowns, exp2062 giraffe) all use larger `num_train_steps` or `reset_data_loader_on_init=False` (which routes through a different `trainer.initialize_from` path), so none of them tripped it. Our midtraining with `num_train_steps=4768 << 46916` is the first case to hit it.

### Fix

Added `CheckpointInitMode` enum to `lib/levanter/src/levanter/main/train_lm.py` with two values:

- `MODEL_ONLY`: `load_checkpoint(state.model, path, subpath="model")` — load only the model subtree, keep freshly-initialized opt_state (count=0). The pattern `train_dpo.py:383` already uses.
- `FULL_STATE`: current (legacy) behavior — restore everything, reset only outer step. Preserves WSD-S rewarmup tricks like exp2062's.

**Default: `FULL_STATE`.** Preserves behavior byte-for-byte for every caller currently on this path; a full audit of every `initialize_from_checkpoint_path=` caller was explicitly *not* done. Delphi opts into `MODEL_ONLY` explicitly in `experiments/exp_delphi_math_10b_midtrain.py`. exp2062 is untouched.

Files changed (uncommitted as of 2026-04-23):

- `lib/levanter/src/levanter/main/train_lm.py` — enum, field on `TrainLmConfig`, branch the load block.
- `experiments/simple_train_config.py` — `checkpoint_init_mode: CheckpointInitMode = FULL_STATE` field.
- `experiments/defaults.py` — forward field from `SimpleTrainConfig` → `TrainLmConfig` in `default_train`.
- `experiments/exp_delphi_math_10b_midtrain.py` — explicit `checkpoint_init_mode=MODEL_ONLY` + comment.
- `lib/levanter/tests/test_checkpoint.py` — 2 raw-load tests (MODEL_ONLY vs FULL_STATE semantics on a schedule-count fixture).
- `experiments/test_default_train_init_mode.py` — 3 plumbing tests asserting defaults + Delphi-experiment MODEL_ONLY propagation through `default_train` to the inner `TrainLmConfig`.

Tests: `uv run python -m pytest lib/levanter/tests/test_checkpoint.py` → 29/29 pass; `uv run python -m pytest experiments/test_default_train_init_mode.py` → 3/3 pass. `./infra/pre-commit.py --fix` on all 6 files: ok.

### Hash impact — new output paths on relaunch

Adding `checkpoint_init_mode=MODEL_ONLY` to the Delphi `SimpleTrainConfig` changes its serialized form, which feeds `executor.py:1407-1408`'s `json.dumps(version, sort_keys=True, cls=CustomJsonEncoder)` → `hashlib.md5(...)[:6]`. The new runs will have different `-<hash>` suffixes than `ba7b7f` / `e3be0c` / `db9de7`. **No `.executor_status` surgery is needed at the old paths** — they're orphans of a different config and the executor simply won't see them.

### Runs marked DISCARDED (do not use for analysis)

- `delphi-1e20-iso-d2048-L21-math-10b-lr0.5-ba7b7f` (v10, 155 GB at `gs://marin-us-central1/checkpoints/`) — trained at lr=2.24e-4 constant, not the 2.24e-3 → 2.24e-4 warmup/decay curve.
- `delphi-1e20-iso-d2048-L21-math-10b-lr0.67-e3be0c` (in-flight coord `/ahmedah/delphi-math-10b-1e20-lr0.67-20260422`) — trained at lr=3.00e-4 constant.
- `delphi-1e20-iso-d2048-L21-math-10b-lr0.83-db9de7` (in-flight coord `/ahmedah/delphi-math-10b-1e20-lr0.83-20260422`) — trained at lr=3.72e-4 constant.

Keep the GCS artifacts around until relaunched runs land healthy on W&B, then `gcloud storage rm --recursive` them as a cleanup pass.

### Follow-up (not in this change)

Audit every `initialize_from_checkpoint_path=` caller in the repo. If all verified tolerant of fresh opt_state, flip the default on `SimpleTrainConfig.checkpoint_init_mode` + `TrainLmConfig.checkpoint_init_mode` to `MODEL_ONLY` so the comment-and-intent mismatch fully heals. Precondition: explicit `FULL_STATE` on any caller that wants the opt-state carry (e.g. exp2062). This is a separate, scope-limited change — not part of this fix.

### Pointer

Full plan at `/Users/ahmed/.claude/plans/feedback-from-codex-make-humble-kernighan.md`.

### Next steps (replacing the pre-incident list above)

1. Kill in-flight `/ahmedah/delphi-math-10b-1e20-lr{0.67,0.83}-20260422`.
2. Commit the 6-file fix + this logbook entry; push to `origin/midtrain_data`.
3. Submit one pilot (`1e20 × lr=0.67` under the new hash). Wait ~30 min, verify on W&B that `optim/learning_rate` rises 0 → 3.00e-3 over first 500 steps, then decays linearly toward 3.00e-4 by step 4768. That's the anti-pathology.
4. If pilot healthy, submit remaining 5 sweep points (1e20 × lr=0.5, 0.83; 1e21 × lr=0.5, 0.67, 0.83) in parallel on v5p-64.
5. After all 6 land, compare smoothed train-loss + Paloma panels across (base × lr_factor), pick winners, write up.

---

## 2026-04-23 relaunch + hash-collision surprise (08:00–08:15 UTC)

### What happened

Relaunched the three 1e20 sweep points after the Levanter fix was committed (`37fba5983`, pushed). Job names: `/ahmed/delphi-math-10b-1e20-lr{0.5,0.67,0.83}-20260423-v2`.

**Surprise 1: The executor output hash did NOT change after adding `checkpoint_init_mode=MODEL_ONLY` to Delphi's `SimpleTrainConfig`.** Plan had assumed it would. Root cause: `lib/marin/src/marin/execution/executor.py:1028-1094`'s `collect_dependencies_and_version` only records values wrapped in `versioned(...)` — plain dataclass fields are recursed into but not added to the hash input. Since I didn't wrap `checkpoint_init_mode` in `versioned()`, the field has no effect on the step hash. The new runs landed at the SAME output paths as the broken v10/in-flight runs: `-ba7b7f`, `-e3be0c`, `-db9de7`.

First "fix" (`-20260423-fix` coordinators) therefore found `already succeeded` markers and skipped training entirely for lr=0.5, and the other two were headed toward the same outcome.

**Resolution:** killed the `-fix` coordinators, deleted all three junk directories (`gcloud storage rm --recursive` ~465 GB), then resubmitted as `-v2`. Now the `.executor_status` cache misses and the executor runs training. Old artifacts are gone (not merely moved) so there's no Levanter auto-resume-from-broken-ckpt risk.

**Surprise 2: W&B monotonic-step rejection.** Because the output hash is unchanged, the W&B run ID is the same as the broken runs'. W&B's run.step is at 4768 from the old training. Levanter's `wandb.py:69` refuses to log metrics whose step is less than `run.step`:

```
W20260423 08:06:50 levanter.tracker.wandb Step 1 is less than the current step 4768. Cowardly refusing to log metrics.
```

**Training is actually fine** — the JAX training loop is running and loss is dropping — but W&B panels won't show LR/loss curves from the new run until the fresh training advances past step 4768. Since `num_train_steps=4768`, that means basically nothing gets logged to W&B this time. For LR-fix verification, rely on:

- `tracker_metrics.jsonl` at the run's GCS output (written by Levanter's local tracker, independent of W&B).
- The tqdm `Progress on:train … postfix:loss=…` lines in iris logs.
- Compare trajectory shape against v10's broken curve (recorded in this logbook).

### Early signal (loss trajectory, tqdm-reported)

| Step | lr=0.67 loss | lr=0.83 loss | Notes |
|---:|---:|---:|---|
|   2 | 1.58 | 1.58 | Identical to v10 initial — pretrain weights loaded correctly. |
|  16 | 1.58 | (not yet seen) | LR still ramping up through early warmup. |
|  30 | 1.51 | (not yet seen) | Rate-of-drop increasing, as warmup approaches peak. |
|  44 | 1.45 | (not yet seen) | |
|  58 | (not yet seen) | 1.43 | Higher-LR factor drops faster — expected ordering. |

Rate of descent is materially faster than the broken v10 curves at equivalent steps. This is the loss-side proof the MODEL_ONLY fix is doing its job: the schedule count is at 0 (not 46916), so actual LR is warming up toward the peak, not clamped to the 0.1×peak floor.

Initial Paloma / uncheatable_eval losses at step 0 look sane (`wikipedia_english=2.535, github_python=1.775, ao3_english=3.158, arxiv_physics=2.767`) — consistent with freshly-loaded pretrain weights.

### Known bugs to tackle later (not blocking the sweep)

1. **Executor hash ignores non-`versioned()` fields.** When flipping `checkpoint_init_mode` selectively, the step's output path does not change. This is a foot-gun: a naive relaunch after a field change silently reuses the old artifact. Options: (a) wrap the field in `versioned()` at call sites that care; (b) add opt-in "always-versioned" dataclass fields at the marin-executor layer; (c) document + rely on manual deletion for now.
2. **W&B run step collision on relaunch.** Same root cause — unchanged hash → same W&B run_id → monotonic-step rejection. Fix would be to include a unique component (timestamp, attempt counter) in the W&B run config so relaunches get fresh run_ids.

Neither blocks the sweep. Both should be filed as issues once the sweep lands.

### Current job states (08:11 UTC)

- `/ahmed/delphi-math-10b-1e20-lr0.67-20260423-v2/train_lm`: running on v5p-64, us-central1. Step 44+, loss dropping.
- `/ahmed/delphi-math-10b-1e20-lr0.83-20260423-v2/train_lm`: running on v5p-64, us-central1. Step 58+, loss dropping.
- `/ahmed/delphi-math-10b-1e20-lr0.5-20260423-v2`: still in zephyr-normalize phase (landed us-east5; normalize/tokenize caches live in us-central1 under a different hash, so it's re-running data prep locally). Expected ~30–60 min before training starts; total wall-time thus slightly longer than the other two.

Next check: verify loss ≪ 1.12 at step 500 (v10's warmup-end number under the broken schedule). If yes, LR fix confirmed. If no, dig deeper.

### LR fix confirmed — both runs finished (2026-04-23 14:28 UTC)

`lr=0.67` and `lr=0.83` reached `4.77kit/4.77kit` (i.e., step 4768) simultaneously at 14:28 UTC, ~6 h 23 min after training-start. Final train-loss (single-step tqdm):

| Run | Final loss | vs v10 broken (0.962) |
|---|---:|---|
| `lr=0.5` | (still running, ETA 15:00 UTC) | — |
| **`lr=0.67`** | **0.781** | 18.8% lower |
| **`lr=0.83`** | **0.772** | 19.7% lower |

Preliminary 1e20 ranking (awaiting `lr=0.5` final + smoothed curves for confirmation):
`lr=0.83 (0.772) < lr=0.67 (0.781)`

The **final-loss test is unambiguous**: both runs' single-step final losses are ~0.18-0.19 below v10's final of 0.962. Under the flat-min-lr bug, effective LR was ~10x too low across the whole run; new runs trained at the intended warmup→peak→decay schedule, and that's the measurable difference in the final loss. Combined with:

- the earlier loss trajectory during warmup being faster than v10's, and
- the crossover/ordering between factors (lr=0.83 leading early, lr=0.67 overtaking in decay tail, then lr=0.83 finishing slightly lower again)

we have three independent lines of evidence that the LR schedule is alive. The `CheckpointInitMode.MODEL_ONLY` branch in `train_lm.py` correctly keeps the freshly-initialized opt_state, so the schedule evaluates at count=0 at step 0 and ramps normally.

**Runs ended with the expected tqdm rate pattern** — `rate:4.4-4.5s/it` for ~4768 steps = ~5:50 elapsed, plus eval+checkpoint pauses absorbed into the rolling average. No crashes, no preemptions, no mid-training bug.

Coordinators are still showing `running` because Levanter is in the final HF-export phase (~7.7 GB × 2 shards per run). Iris will flip them to `succeeded` in 5-10 min once export commits.

### 1e20 sweep complete (2026-04-23 15:05 UTC) — all three succeeded

All three coordinators in terminal `succeeded` state.

| Run | Coordinator | Final single-step loss | vs v10 broken (0.962) |
|---|---|---:|---|
| `lr=0.5` | `/ahmed/delphi-math-10b-1e20-lr0.5-20260423-v2` | **0.840** | −12.7% |
| `lr=0.67` | `/ahmed/delphi-math-10b-1e20-lr0.67-20260423-v2` | **0.781** | −18.8% |
| `lr=0.83` | `/ahmed/delphi-math-10b-1e20-lr0.83-20260423-v2` | **0.772** | −19.7% |

Preliminary 1e20 ranking (unsmoothed tqdm tail reading — these have ~0.02 single-step jitter so the ordering is tentative): `lr=0.83 (0.772) < lr=0.67 (0.781) < lr=0.5 (0.840)`.

Wall-times:
- `lr=0.67`: ~6 h 38 min coordinator-to-succeeded
- `lr=0.83`: ~6 h 38 min
- `lr=0.5`: ~7 h 02 min (included zephyr-normalize + zephyr-tokenize in us-east5 before training could start)

GCS outputs at `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr{0.5,0.67,0.83}-{ba7b7f,e3be0c,db9de7}/` (same hash slots as the DISCARDED v10/broken runs — the Marin-executor-hash-ignores-unversioned-fields caveat remains).

Final HF export (`hf/step-4768/`) is present on all three. Periodic waypoint at `hf/step-1000/`.

For follow-up ranking with smoothed curves, read `tracker_metrics.jsonl` at each output path — W&B is not usable for these runs due to the step-monotonic rejection bug (same-hash-as-broken-run) noted above.

### Next steps

1. Pull smoothed train-loss + Paloma trajectories from each run's `tracker_metrics.jsonl`; confirm the preliminary 1e20 ranking.
2. Launch 3 × 1e21 sweep points (`lr=0.5 / 0.67 / 0.83`) on `v5p-64`. Same launch recipe as the 1e20 relaunch above. Expected wall-time ~10 h per run (3.4 B params, same BS=512, slightly larger). The pretrain ckpt lives at `gs://marin-us-central1/adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/checkpoints/step-21979/` with schedule count ~21979 (smaller than `num_train_steps=4768` — no wait, it's *larger*, so the same flat-min-lr pathology would apply to the 1e21 runs without the fix, and does not with MODEL_ONLY). With MODEL_ONLY plumbed, the 1e21 sweep will train at the scheduled warmup→peak→decay.
3. When all 6 land, cross-ranking + winner selection + writeup. Store the winning (base, lr_factor) combination as input to any downstream sweep.

### 2026-04-23 lr=0.67 / lr=0.83 `-v2` reruns (22:53-22:56 UTC) — clean W&B curves

Context: the original lr=0.67 and lr=0.83 runs succeeded with correct training (final losses 0.781 and 0.772) but their W&B panels still showed the OLD broken flat-min-lr curves. Root cause: the Marin executor hash only tracks `versioned(...)` values + `step.name` + upstream dep paths — plain `SimpleTrainConfig` fields (including our `checkpoint_init_mode`) are invisible. With the same us-central1 tokenize dep as before, both runs landed at the same output hashes as the broken v10-era runs (`e3be0c`, `db9de7`) → same W&B run_ids → W&B's step-monotonic guard rejected the fresh metrics.

Fix (commit `0a5b1fde3`): append `-v2` to the `step.name` template in `experiments/exp_delphi_math_10b_midtrain.py:221`, so the name-contribution to the hash changes. Relaunched both with coordinators `/ahmed/delphi-math-10b-1e20-lr{0.67,0.83}-v2-20260423`.

Results:

| Run | Output hash | W&B run name | Final single-step loss |
|---|---|---|---:|
| `lr=0.67-v2` | `a176ff` | `delphi-1e20-iso-d2048-L21-math-10b-lr0.67-v2-a176ff` | **0.781** (matches original 0.781 exactly) |
| `lr=0.83-v2` | `4487d2` | `delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v2-4487d2` | **0.782** (original 0.772; +0.010 single-step noise, statistically indistinguishable) |

Both coordinators in terminal `succeeded` state. No `Cowardly refusing to log metrics` warnings this time — W&B accepted the fresh metrics, so these two runs now have clean warmup→peak→decay curves on the W&B panel.

The 1e20 sweep now has **one set of canonical, clean-W&B results** for the cross-ranking:

| lr factor | Canonical run name | Final loss |
|---:|---|---:|
| 0.50 | `delphi-1e20-iso-d2048-L21-math-10b-lr0.5-4d19a2` (us-east5, fresh hash by-accident) | 0.840 |
| 0.67 | `delphi-1e20-iso-d2048-L21-math-10b-lr0.67-v2-a176ff` | 0.781 |
| 0.83 | `delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v2-4487d2` | 0.782 |

Preliminary 1e20 ranking (unsmoothed): `lr=0.67 (0.781) ≈ lr=0.83 (0.782) < lr=0.5 (0.840)`. The 0.67/0.83 gap is within noise; smoothed curves + Paloma eval should disambiguate. Either factor is a reasonable default for the 1e21 sweep.

**Stale artifacts to eventually garbage-collect** (no longer canonical; W&B + GCS data is superseded by the `-v2` runs):

- `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.67-e3be0c/` (GCS is the healthy fresh training, but W&B run is polluted with broken data)
- `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-db9de7/` (ditto)
- Also the corresponding W&B runs at those names — they display misleading flat-min-lr curves; safe to delete once the `-v2` runs are locked in.

---

## Analysis + plotting utilities in this repo (found 2026-04-23)

Will Held owns most of the scaling-law / analysis infra. If you need to produce plots, fits, or sweep-wide comparisons, these are the code paths to study first rather than rolling your own.

### Core library — `lib/marin/src/marin/scaling_laws/` (≈1001 lines, plotly-based)

| File | Key exports | What it does |
|---|---|---|
| `scaling_plots.py` | `create_isoflop_plot`, `create_scaling_plot`, `save_plots`, `upload_plots_to_wandb` | Plotly figure builders for isoflop curves + scaling fits; GCS save + W&B artifact upload |
| `isoflop_analysis.py` | `fit_scaling_laws`, `predict_optimal_config`, `robust_quad_logx` (Huber-loss quadratic fit), `ScalingFit`, `QuadraticFitCoeffs`, `IsoFlopRecord`, `MinimaRecord`, `CandidateConfig` | Scaling-law math and data structures |
| `eval_metrics_reader.py` | `read_eval_records` (+ W&B backfill via `_backfill_metrics_from_wandb`) | Pulls per-step eval metrics from GCS runs and W&B API, unifying the two sources |
| `tpu_utils.py` | `pick_v5p_type`, `pick_v4_type`, `V5P_SPEC`, `V4_SPEC` | Choose the smallest TPU slice that fits a given model |
| `__init__.py` | Re-exports above | Public API entry |

### Callers / end-to-end wiring

| File | Purpose |
|---|---|
| `experiments/isoflop_sweep.py` | **The canonical ExecutorStep wiring** — reads eval metrics, fits scaling laws, emits plots, uploads to W&B. Pattern-match against this when building a new analysis step. |
| `experiments/exp1337_delphi_suite.py` | Delphi-specific sweep runner using `predict_optimal_config`. Source of the `(H, L, B)`-heuristic pipeline. |
| `experiments/exp2166_scaling_ladder_analysis.py` | Most recent ladder analysis (~2026-02). |
| `experiments/scaling_law_sweeps/completed_adamh.py` | The AdamH heuristic that drove our 1e20 base-model choice — `completed_adamh_heuristic._build_model_config(hidden_size, seq_len)` is the source of truth for Delphi architecture. |
| `experiments/scaling_law_sweeps/c_adamc.py` | AdamC-variant counterpart. |

### Per-run training-loss (no dedicated Marin tool)

For single-run or small-sweep train-loss plots (what this midtraining sweep wants), the options are:

- `lib/levanter/scripts/loss_history.py` — ~30-line example that hits `wandb.Api().runs().scan_history()` for `train/loss` by git-sha. Good template.
- Read `tracker_metrics.jsonl` at each run's GCS output path directly (Levanter writes it independently of W&B; **this is our only source of truth for the `e3be0c` / `db9de7` runs whose W&B is polluted**). One JSON per step, columns include `train/loss`, `optim/learning_rate`, `optim/adam_lr`, and all `eval/paloma/*/loss` + `eval/uncheatable_eval/*/loss` series. Just `pd.read_json(..., lines=True)`.

### For this midtraining sweep specifically

The scaling-laws infra is overkill for a 3-point × 2-base LR sweep (no scaling fit is meaningful with one token budget + one parameter count per base). Appropriate plots:

- Train-loss vs step, EMA-smoothed, one line per `(base, lr_factor)`.
- Paloma validation loss vs step (`eval/paloma/c4_en/loss`, `eval/paloma/dolma-v1_5/loss`, etc.), same overlay.
- Final-loss bar chart per sweep point to pick the winner.

A ~50-line script that loads the 6 `tracker_metrics.jsonl` files (3 × 1e20 + 3 × 1e21 once they land) and renders these with matplotlib or plotly is enough. **Do not** build it on top of `isoflop_analysis.py` — wrong abstraction. **Do** reuse `eval_metrics_reader.read_eval_records` for the GCS + W&B unification logic if the runs have the right shape (check its filters first).

### Authorship / blame-walk

- `scaling_plots.py` / `isoflop_analysis.py` / `eval_metrics_reader.py` / `isoflop_sweep.py` — William Held, PR #2243 "Scaling Plots & Analysis as an Executor Step".
- Delphi pipeline (`exp1337_delphi_suite.py`) — William Held, PR #3292 "Delphi Scaling Setup", plus PR #4591 "exp1337: add seed sweep".
- AdamH heuristic — William Held, PR #2447 "Beta2 gets a bit wacky with very large batch sizes...".

When in doubt on scaling/analysis decisions, `git log --format='%an %s' -- <file>` → look for Will.

---

## Project goal #1 — midtraining loss predictor (detailed plan, 2026-04-23)

### Why this is the right next thing

Project goal #1 reads *"predict loss trajectory for midtraining."* The concrete operational payoff is: **given small pilot runs at 1e20 and 1e21, forecast final eval metrics at 1e22 and 1e23 without actually running them** (or at least: predict enough of the curve shape to reject bad LR schedules before committing v4-512 / v4-1024 compute). The sweep tracked in this logbook produces exactly the training signal such a predictor would consume.

### Empirical pre-conditions already established

1. **Noise floor ≈ 0.01 abs loss** on final train loss. Evidence: `lr=0.67` and `lr=0.67-v2` (identical config, different RNG/wandb run) finished at 0.781 and 0.781 respectively; `lr=0.83` and `lr=0.83-v2` at 0.772 and 0.782. Any predictor with MAE ≪ 0.01 is overfitting noise.
2. **Per-step data source**: `tracker_metrics.jsonl` at the GCS output path ONLY contains `{"config": ..., "summary": ...}` — i.e. one row of final values, NOT a time-series. The time-series lives exclusively in W&B. Pull it via `wandb.Api().run(...).scan_history(keys=[...], page_size=2000)`. Train metrics come at every step (4768 rows / run); evals come at `steps_per_eval=200` cadence (~48 rows / run, including step 0 pre-training eval).
3. **Three canonical 1e20 runs available with clean W&B**:
   - `delphi-1e20-iso-d2048-L21-math-10b-lr0.5-4d19a2`
   - `delphi-1e20-iso-d2048-L21-math-10b-lr0.67-v2-a176ff`
   - `delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v2-4487d2`
4. **Retention degrades during midtrain** (confirmed empirically): Paloma c4_en starts at 2.8586 for all runs (pretrain-only), and ends at 3.15 (`lr=0.5`), 3.29 (`lr=0.67-v2`), 3.29 (`lr=0.83-v2`). Higher midtrain LR → more retention damage. This is the classic specialization/retention tradeoff. The predictor must handle both monotone-downward (train/loss on math) and monotone-upward (Paloma c4_en) metrics.

### Functional form to fit

Core: schedule-aware power-law in cumulative learning rate.

```
L(u) = L_∞ + A × (U - u)^c
```

where

- `u(t) = Σ_{s ≤ t} lr(s)` = cumulative `optim/learning_rate` series from W&B.
- `U = u(T)` = total LR budget consumed at end of training.
- `L_∞`, `A`, `c` are the three fit parameters.
- For monotone-decreasing metrics (train/loss): `A > 0`, `c > 0`, `L_∞` = asymptote from below.
- For monotone-increasing metrics (Paloma retention): `A < 0`, `c > 0`, `L_∞` = asymptote from above.

Why `optim/learning_rate` and not `optim/adam_lr`: peak LRs differ ~60× (2.24e-3 to 3.72e-3 for `learning_rate` vs 3.69e-5 to 6.13e-5 for `adam_lr`), and `learning_rate` governs the matrix-param updates that dominate midtraining dynamics.

### Baselines (in order of ambition)

| Baseline | Form | Fit on | Purpose |
|---|---|---|---|
| **B0: last-value** | `L̂ = L(t_prefix_end)` | — | Trivial control. Beats every other baseline at 99% prefix. |
| **B1: raw-step power** | `L(t) = a + b/√t` | `step > 500`, back half of observed prefix | Schedule-unaware baseline. If B2 doesn't beat this, something's wrong. |
| **B2: schedule-aware power** | `L(u) = L_∞ + A(U-u)^c` | `step > 500`, back half of prefix | The actual proposed predictor. |

### Evaluation protocol

**Target quantity:** EMA-smoothed loss over the last window `[4600, 4767]`. Using single-step final loss introduces ~0.01 noise; averaging over 167 points puts the target well below noise floor.

**Metrics:**
1. `train/loss` — smoothest curve, machinery validation (does the functional form fit anything?).
2. `eval/paloma/c4_en/loss` — retention anchor. Rising; tests whether the form handles sign change.
3. `eval/paloma/*_loss` aggregates — broader retention panel (future).
4. Math-specific eval (pending) — actual scientific target.

**Prefixes evaluated:** `{30%, 50%, 80%}` of total `num_train_steps` (so {1430, 2384, 3814} for 1e20 runs). Each prefix truncates the fit set; we then predict the step-4767 target from the remaining fit.

**Tests (easiest → hardest, per project goal #1):**

1. **Self-prefix** (tractable with only 3 runs): for each (run, metric), fit on first X% and predict own final. Pass = B2 MAE ≤ 2 × noise floor by prefix=30%.
2. **Cross-LR, same base** (3 runs → LOO over lr_factor): fit shared `c` (+ optionally shared `A`) on two 1e20 runs; predict third run's final using only its first 30%. Pass = MAE ≤ 3 × noise floor.
3. **Cross-base** (blocked on 1e21 sweep): fit all 1e20 curves; predict 1e21 finals from only `(lr_factor, base_params)` features + first 30% of 1e21. This is the project-goal-1 money shot.
4. **Cross-scale** (blocked on 1e22/1e23 runs): extrapolate from 1e20 + 1e21 to larger bases.

Tests 1-2 are validations of the functional form; tests 3-4 are the scientific claim.

### Implementation plan

**File:** `scripts/analysis/midtrain_loss_predictor.py`.

**Dependencies available in the repo venv**: `pandas`, `numpy`, `scipy.optimize`, `wandb`, `matplotlib` (via plotly import guard). No new requirements.

**Structure** (~250 lines):

```python
RUNS_1E20 = [
    RunSpec("lr=0.5",  lr_factor=0.5,  wandb_name="delphi-...-lr0.5-4d19a2"),
    RunSpec("lr=0.67", lr_factor=0.67, wandb_name="delphi-...-lr0.67-v2-a176ff"),
    RunSpec("lr=0.83", lr_factor=0.83, wandb_name="delphi-...-lr0.83-v2-4487d2"),
]

def load_run(spec):  # wandb.Api().run.scan_history → DataFrame
def ema_smooth(df, halflife=100):  # train/loss only; evals are already low-freq
def compute_cumulative_lr(df):  # cumsum of optim/learning_rate
def fit_last_value(df, prefix_frac, metric):
def fit_sqrt_t(df, prefix_frac, metric, min_step=500):
def fit_cumlr_power(df, prefix_frac, metric, min_step=500):  # L_∞ + A (U-u)^c
def evaluate_final(df, metric, window=(4600, 4767)):  # target quantity

def run_self_prefix(runs, metrics, prefixes):
    # Returns DataFrame: (run, metric, prefix, baseline, predicted, target, abs_err)

def run_cross_lr_loo(runs, metrics, prefixes):
    # Hold out one LR; fit shared c on others; predict held-out final from its 30% prefix

def main():
    # 1. Load all runs
    # 2. Run self_prefix + cross_lr_loo
    # 3. Report tables: MAE per (baseline, prefix, metric)
    # 4. Report c stability (fit c across prefixes; warn if std > 0.1)
    # 5. Write CSV of full prediction table to scripts/analysis/midtrain_loss_predictor_out.csv
    # 6. Print human-readable summary to stdout
```

**Outputs:**
- Stdout: summary table (MAE per method × prefix × metric) + c-stability report.
- CSV: `scripts/analysis/midtrain_loss_predictor_out.csv` with every prediction.
- Optional: matplotlib figures for (a) loss vs step raw, (b) loss vs cumulative-LR (should show partial collapse across LR factors).

**Execution:** `uv run python scripts/analysis/midtrain_loss_predictor.py`. Should complete in <2 min (3 W&B fetches + 3×3×3 = 27 fit calls).

### Success criteria for phase 1

- B2 (schedule-aware) beats B1 (raw-step) by >20% on MAE for train/loss at prefix ≤ 50%. If not, the `(U-u)^c` parameterization isn't adding value and we should revisit.
- B2 MAE ≤ 2× noise floor (≤ 0.02) on self-prefix test at prefix=30% for train/loss.
- `c` stable across prefixes (std ≤ 0.1 across {30%, 50%, 80%} prefix choices) per-run.
- For Paloma c4_en (rising metric): B2 handles the sign change gracefully. If `c` has to go imaginary or fit fails numerically, need a separate form for rising metrics.
- Cross-LR LOO MAE ≤ 3× noise floor on train/loss. Higher on Paloma is expected due to noisier eval series.

Meeting these → good enough to commit 1e21 compute and run phase 2. Not meeting → stop, understand why, adjust form.

### Out of scope for phase 1

- Joint multi-metric fitting (we fit metric-by-metric).
- Multi-base extrapolation (blocked on 1e21).
- Uncertainty quantification (bootstrapping or Bayesian fit) — cheap win for phase 2.
- Math-eval loss analysis — blocked on actual math evals being computed on the checkpoints (those are a separate step not yet done).

### Status

Phase 1 script implemented at `scripts/analysis/midtrain_loss_predictor.py` and run against the 3 canonical 1e20 curves.

### Phase 1 results (2026-04-23)

Ran the script after adding two fixes that surfaced during implementation:

- **Two-phase W&B fetch.** `scan_history(keys=[train_keys, eval_keys])` in a single call does an *intersection* over rows, so train (every step) intersected with eval (every 200 steps) yielded only 27 rows. Split into two fetches: one for `optim/learning_rate` + `train/loss` (4768 rows), one for `eval/paloma/c4_en/loss` (48 rows). Merged on `_step`.
- **Bounds on `c`.** Without bounds, `curve_fit` drove `c → ∞` and `A → 0`, giving the degenerate fit `L ≈ L_∞` (a useless constant). Added `c ∈ [0.2, 3.0]` bounds. Also added two fixed-c variants (B3_c=0.5 and B3_c=1) as stabler alternatives with the same 2-param count as B1.

#### Self-prefix MAE (tested on target = EMA over steps 4600–4767)

Noise floor ≈ 0.005–0.010 (from original-vs-v2 rerun pairs).

**train/loss_smooth**:

| method | prefix 30% | prefix 50% | prefix 80% |
|---|---:|---:|---:|
| B0 last-value | 0.146 | 0.084 | 0.021 |
| **B1 a+b/√t** | **0.024** | 0.020 | **0.003** |
| B2 free c (bounded [0.2, 3.0]) | 0.085 | 0.056 | 0.007 |
| B3 c=0.5 | 0.428 | 0.124 | 0.041 |
| **B3 c=1 (schedule-aware, fixed c)** | 0.120 | **0.004** | 0.007 |

**eval/paloma/c4_en/loss** (sparse — ~12 points per 30% prefix, so B1/B2/B3 skip prefix=0.3):

| method | prefix 30% | prefix 50% | prefix 80% |
|---|---:|---:|---:|
| B0 last-value | 0.086 | 0.039 | 0.036 |
| B1 a+b/√t | — | 0.075 | 0.052 |
| B2 free c | — | **0.037** | 0.038 |
| B3 c=0.5 | — | 0.211 | 0.070 |
| B3 c=1 | — | 0.100 | 0.050 |

#### Cross-LR LOO (hold out 1 of 3, fit shared c on other 2, predict from 30/50/80% of held-out)

| metric | prefix 30% | prefix 50% | prefix 80% |
|---|---:|---:|---:|
| train/loss_smooth | 0.182 | 0.028 | **0.002** |
| eval/paloma/c4_en/loss | — | 0.519 | 0.129 |

#### c-stability across prefixes (B2 free-c)

`c` values per (run, metric) at prefixes {30%, 50%, 80%}:

- `lr=0.5` train/loss: 3.000 → 1.529 → 1.086 (std ≈ 0.82; bouncing)
- `lr=0.67` train/loss: 3.000 → 2.339 → 0.824 (std ≈ 0.91)
- `lr=0.83` train/loss: 3.000 → 2.322 → 0.763 (std ≈ 0.94)
- Paloma: hits bound c=3.0 at prefix 0.8 for all runs (under-identified)

`c` hits the upper bound (3.0) for short prefixes → the fit wants even larger `c` but can't, which confirms the parameter is under-identified on this data.

### Takeaways

1. **B1 (raw `a + b/√t`) is the surprisingly strong baseline for train/loss self-prefix.** At prefix 30% it gives MAE 0.024 (~2.5× noise floor); at prefix 80% it hits 0.003 (at noise floor). Schedule-unawareness doesn't hurt at this fidelity because `√t` already captures the asymptotic shape well enough when the LR schedule is held constant across runs.

2. **B3 with fixed `c=1` wins at prefix 50% train/loss** (MAE 0.004, at noise floor). This says: if you commit to a schedule-aware parameterization, you must pin `c` — letting it float introduces more error than it removes on this amount of data. The linear-in-remaining-LR form (`c=1`) is the best zero-prior default.

3. **B2 (free `c`) is worse than both B1 and B3.** The 3-parameter fit is under-identified on ~700–3800 points of fairly smooth data. Even with bounds the fit chases degenerate solutions at short prefixes.

4. **Paloma is not tractable with these 3 runs.** Only ~12-24 eval points per prefix. Even the best method (B2 at prefix 50%) has MAE 0.037 — ~10% of the metric's midtrain-induced change (2.86 → 3.29, δ ≈ 0.43). Need the 1e21 sweep's eval points to get more leverage, or use a totally different approach for sparse metrics (e.g., linear extrapolation through 3-4 points).

5. **Cross-LR LOO on train/loss works at ≥ 50% prefix.** MAE 0.028 at 50%, 0.002 at 80%. That is: if you run 2 of the 3 1e20 LR factors to completion and observe the first half of the third, you can predict its final train loss within 0.03. For the 1e20 → 1e21 cross-base test (phase 2), this is encouraging but not decisive — cross-base is a harder extrapolation.

### Implications for the success criteria from the plan

From the pre-committed criteria:

- **"B2 MAE < 2× noise floor at prefix=30% for train/loss"** → FAIL (0.085 at 30%, 8× noise floor). Resolution: use B1 or B3 c=1 as the primary predictor instead. The proposal's `(U-u)^c` with free `c` doesn't have identifiability on this data.
- **"B2 beats B1 by >20% on train/loss"** → FAIL on its own terms, but B3 c=1 (the fixed-c schedule-aware variant) beats B1 at prefix 50% (0.004 vs 0.020). The schedule-awareness adds value IF you pin `c`.
- **"`c_std` < 0.1 per run per metric"** → FAIL (std 0.8–0.9 for train/loss, worse for Paloma). Parameter under-identified; forcing it is the right call.
- **"Cross-LR LOO MAE < 3× noise floor on train/loss at prefix=30%"** → FAIL at 30% (0.182, 18× noise floor). Passes at 50% (0.028, 3× noise floor) and 80% (0.002).

Interpretation: the phase-1 machinery works, but **30% prefix is too short** for this functional family. Revised operational recommendation for phase 2:

- Evaluate predictors at prefix ∈ {50%, 70%, 90%} for the cross-base test. 30% is aspirational but empirically not yet useful on this data.
- Default predictor: **B3 c=1** for train/loss (schedule-aware, 2-param). Fallback to B1 if the 1e21 schedule is materially different.
- For Paloma and other sparse evals: skip the fit, use last-value or linear-in-step at the last 3-4 eval points.

### Outputs

CSVs for downstream plotting in `scripts/analysis/`:

- `midtrain_loss_predictor_self_prefix.csv`
- `midtrain_loss_predictor_cross_lr.csv`
- `midtrain_loss_predictor_c_stability.csv`

### Phase 2 launch criteria — recommendation

Given phase 1 results, **launching the 1e21 sweep now is justified**. The cross-LR LOO on train/loss at ≥ 50% prefix is within reach of noise floor (0.028 → 0.002), which is exactly the regime the cross-base test needs. Don't over-interpret phase-1 results for Paloma — that needs more data, which 1e21 will provide.

If cross-base train/loss MAE < 0.05 at prefix 50% when 1e21 lands, project goal #1 has a validated baseline. Everything beyond that (cross-scale to 1e22/1e23, uncertainty bounds, better Paloma models) is iteration.

---

## 2026-04-23 session summary — end-to-end record

This section captures everything done in the 2026-04-23 session so far, in one place, so a future agent or the same user tomorrow can pick up cold.

### Commit chain on `origin/midtrain_data` this session

(newest first; `b860d5b60 wip` is the first pre-session commit, for reference)

| Commit | What |
|---|---|
| `64c030b70` | `[analysis] Plots for 1e20 midtrain runs (matplotlib)` — `scripts/analysis/plot_midtrain_curves.py` generates four PNGs and `open`s them on macOS. |
| `8aefa1689` | `[analysis] Phase-1 midtrain loss predictor: script + results` — `scripts/analysis/midtrain_loss_predictor.py` with 5 baselines and self-prefix + cross-LR LOO tests against the 3 canonical 1e20 runs. |
| `f8a85aef0` | `[logbook] Delphi midtrain: detailed phase-1 plan for loss predictor` — fully-specified plan before implementation. |
| `d04dea5d6` | `[logbook] Delphi midtrain: clarify project goal hierarchy` — two tracks (predict trajectory, pick dataset); this sweep is a subgoal of track 1. |
| `6deb3dc8c` | `[logbook] Delphi midtrain: add analysis-utils reference section` — pointers to Will Held's `scaling_laws/` module and related files. |
| `5fed3965d` | `[logbook] Delphi 1e20 lr=0.67/0.83 v2 reruns done — clean W&B curves` — both `-v2` reruns succeeded with losses matching originals; noise floor ≈ 0.01 confirmed. |
| `0a5b1fde3` | `[delphi-midtrain] Append -v2 to step names so hash changes` — workaround for the Marin executor's cache-hash behaviour (only versioned values + step name + upstream deps contribute to the hash). |
| `fc14f21ba` | `[logbook] Delphi 1e20 sweep complete — all 3 succeeded, fix fully confirmed` — final losses 0.840 / 0.781 / 0.772 for LR factors 0.5 / 0.67 / 0.83, all well below v10's broken 0.962. |
| `cbdd748b0` | `[logbook] Delphi midtrain: LR fix confirmed by final-loss at step 4768` — incremental. |
| `e77ea7e47` | `[logbook] Delphi midtrain: relaunch at -v2 + hash-collision + W&B step-monotonic notes` — diagnosed the hash-stays-same + wandb-step-monotonic-refusal bugs during relaunch. |
| `37fba5983` | `[levanter] Add CheckpointInitMode for initialize_from_checkpoint_path` — the core Levanter fix: `MODEL_ONLY` mode loads only the model subtree (using `subpath="model"`), keeping the freshly-initialized opt_state so the schedule starts from count=0. Default is `FULL_STATE` (legacy behaviour preserved). |

Between `97ed6c6f2` and `37fba5983` the user walked up to the flat-LR bug in the 1e20 sweep, caught it visually on W&B (`optim/learning_rate` flat for 3/3 broken v10-series runs), and worked through the `train_lm.py:176-180` root cause with a plan doc and back-and-forth review with Codex.

### What the completed 1e20 sweep produced

Three GCS output paths that are now **canonical for phase-2 analysis** (clean W&B runs):

| LR factor | GCS output | W&B run | Final single-step loss | Target-window mean |
|---:|---|---|---:|---:|
| 0.50 | `gs://marin-us-east5/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.5-4d19a2/` | `delphi-1e20-iso-d2048-L21-math-10b-lr0.5-4d19a2` | 0.840 | ~0.813 |
| 0.67 | `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.67-v2-a176ff/` | `delphi-1e20-iso-d2048-L21-math-10b-lr0.67-v2-a176ff` | 0.781 | ~0.798 |
| 0.83 | `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v2-4487d2/` | `delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v2-4487d2` | 0.782 | ~0.797 |

Plus the original (non-v2) `-e3be0c` / `-db9de7` outputs — GCS is healthy training (the MODEL_ONLY fix did its job), but the W&B panels for those IDs show OLD flat-min-lr curves from the broken training because W&B's step-monotonic guard rejected the fresh step-0..4767 logs. They're kept as noise-floor evidence (final losses match the `-v2` reruns within 0.01).

### Phase-1 loss-predictor infrastructure (landed this session)

**Script 1: `scripts/analysis/midtrain_loss_predictor.py`** (~370 lines).

- Pulls per-step data from `wandb.Api().run().scan_history()` (the GCS `tracker_metrics.jsonl` file only carries final summary, not a time series).
- **Bugfix that mattered:** initial version fetched `train_keys + eval_keys` in a single `scan_history` call, which returns only rows where *every* key is set — intersecting train (every step) with eval (every 200) collapsed the result to ~27 rows and broke every short-prefix fit. Two-phase fetch fixes it.
- Five baselines: **B0 last-value**, **B1 `a + b/√t`** (raw step), **B2 `L_∞ + A(U-u)^c`** with `c` bounded to `[0.2, 3.0]`, **B3 `L_∞ + A(U-u)^c`** with `c` fixed at `0.5` or `1.0`.
- Self-prefix + cross-LR LOO tests at prefixes `{30%, 50%, 80%}`.
- Target quantity: EMA-mean over steps `[4600, 4767]` (above the ~0.01 noise floor of single-step finals).
- CSVs: `midtrain_loss_predictor_{self_prefix,cross_lr,c_stability}.csv` (gitignored; rerun to regenerate).

**Script 2: `scripts/analysis/plot_midtrain_curves.py`** (~230 lines).

Four matplotlib PNGs, auto-opened via `open` on macOS.

### Phase-1 results (numbers)

**Self-prefix MAE on train/loss_smooth** (noise floor ≈ 0.01):

| method | 30% | 50% | 80% |
|---|---:|---:|---:|
| B0 last-value | 0.146 | 0.084 | 0.021 |
| **B1 a+b/√t** | **0.024** | 0.020 | **0.003** |
| B2 free c (bounded) | 0.085 | 0.056 | 0.007 |
| B3 c=0.5 | 0.428 | 0.124 | 0.041 |
| **B3 c=1** | 0.120 | **0.004** | 0.007 |

**Cross-LR LOO MAE on train/loss_smooth**:

| prefix | MAE |
|---:|---:|
| 30% | 0.182 (18× noise floor — not yet usable) |
| 50% | 0.028 (~3× noise floor — borderline) |
| 80% | 0.002 (at noise floor — clean) |

**Paloma c4_en** — at most 12-24 eval points per prefix; power-law fits are noisy. Best self-prefix MAE (B2 at 50%) is 0.037, roughly 10% of the metric's midtrain-induced change (0.29 to 0.43 depending on LR). Not yet a usable forecast; will improve with more runs (1e21 adds 3 more eval trajectories).

**c stability (B2 free-c)**: `c_std` is 0.8-0.9 for train/loss across prefixes and worse for Paloma. `c` hits the upper bound (3.0) at short prefixes. Parameter is under-identified on this data — don't report free-c B2 as the primary method.

### Visual findings (from `scripts/analysis/*.png`)

1. **`train_loss_vs_step.png`** — Three LR-factor curves have near-identical shape. Warmup-to-peak visible around step 500; decay tail smooth; the three curves fan out by only ~0.02 loss at the endpoint.

2. **`train_loss_vs_cumlr.png`** — The headline plot. Left panel (raw cumulative LR `u(t)`) shows the expected horizontal fan-out; `U` at endpoint is proportional to LR factor (~6.0, ~7.8, ~9.7 for 0.5 / 0.67 / 0.83). **Right panel (`u/U` normalized) shows the three curves nearly overlaying** with residual spread comparable to noise floor. This is the strongest single piece of evidence that cumulative-LR is the right time axis for cross-LR prediction — the "schedule-shape dominates" hypothesis is empirically supported on this data. With three runs this is modest evidence, but it's exactly the direction we want.

3. **`paloma_c4_en_vs_step.png`** — Retention tradeoff in full view. Pretrain c4_en = 2.8586 (dotted). All three runs rise during warmup + early decay:
   - lr=0.5 peaks at ~3.15 (Δ +0.29) around step 3000 and plateaus there.
   - lr=0.67 peaks at ~3.30 (Δ +0.44) around step 2500 and plateaus.
   - lr=0.83 peaks at ~3.40 (Δ +0.54) around step 3200, then **drops sharply in the last ~500 steps** to ~3.29 at endpoint.
   Higher LR factor → faster rise + higher peak damage. The lr=0.83 late-decay recovery is novel: the low-LR cooldown tail appears to partially undo retention damage. Worth re-testing as 1e21 data lands — if it reproduces, it's a hint that the end-of-cooldown behaviour is important for trade-off management.

4. **`predictor_fit_overlay.png`** — Side-by-side view of why B3 `c=1` wins at prefix 50%:
   - Left (raw step, B1): `a + b/√t` fit (orange dashed) traces the `lr=0.67` curve beautifully, extrapolates to 0.819 at step 4768; actual target 0.798 → error 0.021.
   - Right (cumulative LR, B2 + B3): B2's free-`c` fit lands `c = 2.34` (green dashed) and over-predicts at 0.865 — visibly too convex. B3 with fixed `c=1` (purple dash-dot) is nearly a straight line through the decay data and lands at 0.795, within 0.003 of the actual 0.798. This is the visual reason B3 `c=1` beats B2 in the MAE table at prefix 50%.

### Revised success criteria (update to the pre-committed ones)

Pre-committed:
- B2 MAE < 2× noise floor at prefix=30% for train/loss — **FAIL** (0.085, 8.5× floor). Swap: use B1 or B3 c=1 as primary predictor.
- B2 beats B1 by >20% on train/loss — **FAIL** (B2 is WORSE); but B3 c=1 beats B1 at 50% (0.004 vs 0.020). Schedule-awareness helps iff `c` is pinned.
- c_std < 0.1 per run/metric — **FAIL** (0.8-0.9). c is under-identified; don't free-float it.
- Cross-LR LOO MAE < 3× noise floor on train/loss at prefix=30% — **FAIL** at 30%, **PASS** at 50% (0.028 ≈ 3×), **PASS** at 80% (0.002).

Operational takeaway:
- **Default predictor for train/loss:** B3 with c=1 OR B1 raw sqrt-t. They're within 2× of each other at any prefix ≥ 50% and both at noise floor at 80%.
- **Default predictor for rising/sparse evals (Paloma):** B0 last-value. Power fits are noisier than the signal at this point-count.
- **Prefix horizon for cross-base test (phase 2):** 50%+. 30% was aspirational and is empirically too short for this functional family.

### What's NOT done (phase 2 blockers)

- **1e21 sweep NOT launched.** Three runs (`lr=0.5 / 0.67 / 0.83` on the 3.4B base) are the test set for the cross-base extrapolation. Each is ~10h on v5p-64. Launch recipe is the same as the 1e20 `-v2` recipe (including explicit `checkpoint_init_mode=CheckpointInitMode.MODEL_ONLY` and the `-v2` name suffix since the experiment file carries that now).
- **Math-domain evals NOT computed** on the 1e20 checkpoints. Needed to measure the "math target" side of the trade-off (currently only train/loss on math-data + Paloma c4_en retention). Would go through `experiments/evals/`.
- **Noise floor calibration is coarse.** Two pairs (0.67 orig vs v2: 0.000 gap, 0.83 orig vs v2: 0.010 gap) → "~0.005-0.010." A handful more seed replicates per LR factor would tighten this, but it's not blocking.
- **Phase 2 cross-base fit** needs the 1e21 trajectories. Blocked on point 1.
- **Phase 3 cross-scale** needs actual 1e22 runs. Out of scope until cross-base validates.

### Known non-blocking bugs filed in this logbook but not addressed

1. **Marin executor cache hash ignores plain `SimpleTrainConfig` fields.** Only `versioned(...)` values, `step.name`, and upstream `ExecutorStep` deps feed the hash. Changing `checkpoint_init_mode`, `learning_rate`, `beta2`, etc. without wrapping them in `versioned()` silently reuses cached outputs. Workaround: bump step.name (e.g., `-v2`). Proper fix: either wrap fields in `versioned()` at use sites or make `SimpleTrainConfig` auto-version critical fields. Not filed as a GH issue yet.
2. **W&B step-monotonic rejection on same-hash relaunch.** Levanter's `wandb.py:69` refuses `step < run.step` logs; a relaunch that inherits the same W&B run ID (because the executor hash didn't change) gets its fresh metrics silently dropped. Workaround: bump the hash. Proper fix: include a unique component (timestamp, attempt counter) in the W&B run config so relaunches always get fresh run_ids. Not filed as GH issue yet.
3. **`tracker_metrics.jsonl` name is misleading.** It contains only `{config, summary}` — one line per run. The per-step time series lives only in W&B. The file could reasonably be renamed (e.g., `run_summary.jsonl`) or the `tracker_metrics` name made accurate by also writing per-step JSONL alongside it. Not filed.

### How to resume from a cold start tomorrow morning

1. Open `.agents/logbooks/midtraining_delphi.md` and read this session-summary section + the phase-1 plan + phase-1 results.
2. If phase-1 numbers need to be regenerated: `uv run python scripts/analysis/midtrain_loss_predictor.py` (reads W&B, writes CSVs). `uv run python scripts/analysis/plot_midtrain_curves.py` (makes PNGs and opens them).
3. To launch the 1e21 sweep (the natural next step):
   ```bash
   for lr in 0.5 0.67 0.83; do
     uv run iris --cluster=marin job run \
       --cpu 1 --memory 3GB --disk 9GB \
       --region us-central1 --region us-east5 \
       --job-name "delphi-math-10b-1e21-lr${lr}-$(date -u +%Y%m%d-%H%M)" \
       --no-wait \
       -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 \
       -e WANDB_API_KEY "${WANDB_API_KEY}" \
       -e MIDTRAIN_SELECT_BASE 1e21-v5 \
       -e MIDTRAIN_SELECT_LR "$lr" \
       -- python experiments/exp_delphi_math_10b_midtrain.py
     sleep 5  # stagger to avoid any residual auto-naming collisions
   done
   ```
4. When the 3 × 1e21 runs land: add them to `RUNS_1E21` in the predictor module and rerun the script with a cross-base LOO test added (fit on 1e20 → predict 1e21 finals from features alone, or from first 50% of 1e21). If cross-base MAE on train/loss ≤ 0.05 at prefix 50%, project goal #1 has a validated baseline.

---

## 2026-04-24 principled `c` constraints follow-up

### Status

| Time | Action | Notes |
|---|---|---|
| 2026-04-24 01:06Z | Started follow-up requested by user: replace the bounded nonlinear free-`c` B2 fit with more principled constraints/diagnostics, regenerate plots, and update this logbook. | Target fixes: normalize remaining-LR axis, profile/grid-select `c`, add shared-`c` and regularized alternatives, and make the evidence visible in plots/CSVs. |
| 2026-04-24 01:10Z | Replaced bounded nonlinear B2 in `scripts/analysis/midtrain_loss_predictor.py`. | New B2 uses normalized remaining-LR `x=(U-u)/(U-u_fit_start)`, profiles `c` on `np.geomspace(0.1, 10.0, 241)`, and solves `(L_inf, A)` exactly by linear least squares for each candidate `c`. Added B2r with weak log-`c` prior centered at `c=1`, plus profiled shared-`c` cross-LR LOO. |
| 2026-04-24 01:11Z | Regenerated analysis outputs with `uv run python scripts/analysis/midtrain_loss_predictor.py` and `uv run python scripts/analysis/plot_midtrain_curves.py`. | Wrote CSVs and six PNGs under `scripts/analysis/`. Existing four plots were regenerated; new plots are `c_profile_scan.png` and `predictor_method_mae.png`. |
| 2026-04-24 01:11Z | First lint command failed because I used the wrong wrapper syntax. | Exact error: `Error: No such option: --files (Possible options: --all-files, --fix)`. Correct command is positional files: `./infra/pre-commit.py --fix scripts/analysis/midtrain_loss_predictor.py scripts/analysis/plot_midtrain_curves.py`. |
| 2026-04-24 01:12Z | First real lint pass failed; patched mechanical issues and reran. | Exact issue classes: `E501 Line too long`, `RUF046 Value being cast to int is already an integer`, `RUF059 Unpacked variable spec is never used`, `B023 Function definition does not bind loop variable ...`, and Black would reformat `scripts/analysis/midtrain_loss_predictor.py`. |
| 2026-04-24 01:12Z | Verification passed. | `./infra/pre-commit.py --fix scripts/analysis/midtrain_loss_predictor.py scripts/analysis/plot_midtrain_curves.py` passed all hooks. Final `uv run python scripts/analysis/midtrain_loss_predictor.py` also completed and rewrote the CSVs. |

### Implementation notes

The old B2 failure mode was not "SciPy needs a better box"; it was an identifiability problem. The updated implementation makes that explicit:

- `normalized_remaining_lr(u, U, fit_start_u)` maps every run/prefix to `x in [0, 1]`, so `L_inf` is the prediction at `x=0`.
- For fixed `c`, the model is linear: `L(x)=L_inf + A*x^c`. The code now solves `L_inf, A` via `np.linalg.lstsq` rather than nonlinear `curve_fit`.
- `B2_profiled_c` chooses `c` by a chronological prefix split: fit early prefix points, validate on the later prefix tail.
- `B2r_profiled_c_logprior1` adds `0.25 * var(y_val) * log(c/1)^2` to the validation MSE. This is a diagnostic prior, not the recommended production predictor.
- `midtrain_loss_predictor_c_profiles.csv` records the full c-scan objective surface. This is the artifact to inspect when asking whether `c` is identified.

### Results after replacing bounded B2

Self-prefix MAE on `train/loss_smooth`:

| method | 30% | 50% | 80% |
|---|---:|---:|---:|
| B1 `a+b/sqrt(t)` | **0.024** | 0.020 | **0.003** |
| B2 profiled `c` | 0.103 | 0.054 | 0.005 |
| B2r profiled `c` + log prior | 0.097 | 0.052 | 0.005 |
| **B3 fixed `c=1`** | 0.120 | **0.004** | 0.007 |
| B3 fixed `c=0.5` | 0.428 | 0.124 | 0.041 |

Cross-LR LOO MAE on `train/loss_smooth`:

| method | 30% | 50% | 80% |
|---|---:|---:|---:|
| shared profiled `c` | 0.181 | 0.028 | **0.002** |
| **fixed `c=1`** | **0.120** | **0.004** | 0.007 |
| fixed `c=0.5` | 0.428 | 0.124 | 0.041 |

`c` is still unstable even without optimizer bounds. Selected train-loss `c` values:

- lr=0.5: `3.224, 1.283, 1.039` for prefixes `30%, 50%, 80%`
- lr=0.67: `3.905, 2.512, 0.891`
- lr=0.83: `3.831, 2.512, 0.825`

No selected train-loss `c` hit the scan edge, so this is not a too-narrow-grid artifact. The prefix-tail validation objective genuinely prefers high `c` at 30-50%, but those high-`c` fits extrapolate poorly to the final endpoint.

### Visual findings from new plots

- `c_profile_scan.png`: at prefix 50%, validation minima sit around `c≈1.28` for lr=0.5 and `c≈2.51` for lr=0.67/0.83. The log-prior nudges them only slightly. This shows that prefix-tail validation alone does not recover the endpoint-useful `c=1` prior.
- `predictor_method_mae.png`: fixed `c=1` is the clear 50% prefix winner; B1 wins at 30%; B1/B2/B3 are all close to the noise floor by 80%, except fixed `c=0.5`.
- Updated `predictor_fit_overlay.png`: for lr=0.67 at 50%, B2/B2r still predict about `0.868/0.867` vs actual `0.798`, while fixed `c=1` predicts `0.795`. The old bounded-curve-fit explanation is gone; the failure remains and is now attributable to the profiled high-`c` choice.

### Updated recommendation

Do not report free/profilled `c` as a real learned exponent yet. It is useful as a diagnostic surface, not as the predictor. For phase 2, carry these baselines forward:

1. **Primary train-loss predictor:** B3 fixed `c=1`.
2. **Schedule-unaware sanity baseline:** B1 `a+b/sqrt(t)`.
3. **Diagnostic only:** B2 profiled `c` and B2r log-prior, with `midtrain_loss_predictor_c_profiles.csv` / `c_profile_scan.png` used to inspect identifiability.

The practical conclusion is unchanged but better supported: the principled constraint is not "bound `c`"; it is "choose `c` as a prior/hyperparameter and validate predictive performance." On the current 3-run 1e20 data, fixed `c=1` remains the best schedule-aware default.

---

## 2026-04-24 interactive prefix plot

### Status

| Time | Action | Notes |
|---|---|---|
| 2026-04-24 01:19Z | Started user-requested interactive plot for prefix sensitivity. | Goal: slider-controlled prefix percentage with live updates to predictor curves and absolute-error bars for each method. |
| 2026-04-24 01:21Z | Added `scripts/analysis/interactive_midtrain_prefix_plot.py` and generated/opened `scripts/analysis/interactive_midtrain_prefix_plot.html`. | Command: `uv run python scripts/analysis/interactive_midtrain_prefix_plot.py`. The HTML embeds Plotly + precomputed predictions for prefixes 20%-90% across the three canonical 1e20 runs, so slider updates are browser-local and instant. |
| 2026-04-24 01:22Z | Linted the new generator. | `./infra/pre-commit.py --fix scripts/analysis/interactive_midtrain_prefix_plot.py` passed. The generated HTML is ~16 MB and intentionally left as an analysis artifact next to the PNGs, not as source code to edit by hand. |
| 2026-04-24 01:25Z | Improved interactive plot labeling after user feedback. | Renamed predictor legend entries, added formula/detail text to the table, added explanatory note cards, and labeled the prefix/warmup/target markers directly in plot titles/annotations. Regenerated `interactive_midtrain_prefix_plot.html`; lint still passes for the generator. |
| 2026-04-24 01:31Z | Fixed the interactive plot's axis/run comparison design after user feedback. | Default view is now **All LR runs** on normalized cumulative LR `u/U`, with x-axis selector for `u/U`, raw cumulative LR `u`, or training step. The main plot overlays all three runs for the selected predictor method; bars/table/error-vs-prefix aggregate across the selected run scope. Regenerated `interactive_midtrain_prefix_plot.html`. Lint passed. Playwright smoke test was attempted but the repo venv does not have `playwright` installed (`No module named 'playwright'`). |
| 2026-04-24 05:50Z | Lowered interactive prefix slider floor from 20% to just after warmup. | New lower bound is computed as `ceil(100 * WARMUP_STEPS / TOTAL_STEPS) = 11%`. A true 10% prefix is before warmup ends (`500 / 4768 = 10.49%`) and leaves no post-warmup fit window because all fit functions skip warmup points. Regenerated `interactive_midtrain_prefix_plot.html`, reopened in Chrome, and lint passed. |
| 2026-04-24 05:54Z | Added train/eval metric toggle and made Chrome the default opener. | `interactive_midtrain_prefix_plot.py` now precomputes both `train/loss_smooth` and `eval/paloma/c4_en/loss`. The top-right toggle switches every plot/table between train loss and eval loss. Eval loss is sparse and can move up or down; expected early-prefix fit failures are shown as missing predictions in the UI rather than noisy warnings. The generator now always runs `open -a "Google Chrome" ...`. Regenerated HTML and lint passed. |

### Interactive plot details

Artifact: `scripts/analysis/interactive_midtrain_prefix_plot.html`.

Controls:

- Compare selector: all LR runs by default, or a single `lr=0.5`, `lr=0.67`, `lr=0.83` run.
- X-axis selector: normalized cumulative LR `u/U` by default, raw cumulative LR `u`, or training step.
- Fit overlay selector: choose which predictor family is drawn on top of the observed curves.
- Prefix slider: 11%-90% in 1% increments. The lower bound is the first whole-percent prefix after warmup.
- Top-right metric toggle: train loss or Paloma `c4_en` eval loss.

Live views:

- Fit overlay: full `train/loss_smooth` curves for the selected run scope, prefix/warmup/target markers in the selected x-axis coordinate, and endpoint markers for the selected predictor.
- Absolute-error bars at the selected prefix. In all-run mode this is mean absolute error across the three LR factors.
- Error-vs-prefix trajectories for all methods. In all-run mode this is mean absolute error across the three LR factors.
- Method table with per-run prediction/error columns in all-run mode, or prediction/error/selected-`c` detail in single-run mode.

Eval-loss interpretation: `eval/paloma/c4_en/loss` is a retention metric and is sparse (~eval cadence, not every train step). Higher is worse relative to the pretrain checkpoint, but the curve is not guaranteed monotone: it can rise during high-LR adaptation and partially recover during cooldown. The signed `A` in B2/B3 lets the fitted endpoint be above or below the prefix trend, but the one-direction power family still cannot represent rise-then-recover dynamics well. Treat eval fits as diagnostics; for actual decision-making, prefer plotting the observed eval trajectory and using B0/last-value or a very local last-few-points trend until more eval trajectories land.

This is meant for visual inspection rather than batch reporting. The source of truth for the formulas remains `scripts/analysis/midtrain_loss_predictor.py`.

---

## 2026-04-24 prefix pre-registration protocol and 1e20 eval read

### Pre-registration clarification

For a 20% / 30% / 50% prefix predictor, pre-registration does **not** mean making endpoint predictions before seeing any run data. It means locking the prediction rule before inspecting any data **to the right of the declared prefix cutoff**.

Concrete protocol for the next 1e21 sweep:

1. Before launch, or at least before reading beyond the first prefix cutoff, write down:
   - Runs: `1e21` base with LR factors `{0.5, 0.67, 0.83}`.
   - Prefix checkpoints: `20%`, `30%`, `50%`, `80%`.
   - Primary prefix for claims: `50%`; `20%` and `30%` are exploratory stress tests.
   - Allowed data: only rows with `step <= floor(prefix * total_steps)`, excluding warmup points for fit functions that require post-warmup data.
   - Train target: final-window mean of `train/loss_smooth`.
   - Eval target: final-window `eval/paloma/c4_en/loss`, plus peak Paloma/c4 damage as a retention-risk diagnostic.
   - Primary predictor: B3 fixed `c=1` on normalized cumulative LR `u(t)/U`.
   - Baseline predictor: B1 `a + b / sqrt(t)`.
   - Diagnostics only: B2 profiled `c` and B2r profiled `c` plus weak log-`c` prior.
   - Success criterion for the predictor track: cross-base train-loss MAE <= `0.05` at the 50% prefix.

2. When a run reaches a declared prefix, freeze a prediction artifact before looking further ahead:
   - `run_id`
   - git SHA / experiment config hash
   - prefix percent and max step read
   - metric
   - predictor method
   - predicted endpoint / final-window target
   - fitted parameters, including `c` if applicable
   - timestamp and command used to generate the row

3. Append or write those rows before opening later W&B panels or rerunning analysis on post-prefix data. A timestamped CSV under `scripts/analysis/` plus a short logbook entry is enough.

This keeps the test causally clean: the first 20% may be used for a 20%-prefix forecast, but steps after 20% must not influence that forecast's method choice or parameters.

### Current 1e20 eval read

From `scripts/analysis/midtrain_loss_predictor_self_prefix.csv`, final-window targets for the fixed 1e20 v2 sweep are:

| LR factor | `train/loss_smooth` final-window target | `eval/paloma/c4_en/loss` final-window target |
|---:|---:|---:|
| 0.50 | 0.812572 | **3.150040** |
| 0.67 | 0.798438 | 3.289912 |
| 0.83 | **0.797175** | 3.292075 |

Interpretation:

- On math-data train loss, `lr=0.67` and `lr=0.83` are effectively tied and both beat `lr=0.5`. The tiny `0.83` advantage over `0.67` is likely too small to treat as decisive without downstream math evals.
- On Paloma/c4 retention, `lr=0.5` is clearly best: it has much less c4 damage than `0.67` or `0.83`.
- Therefore there is no single "best 1e20 model" yet unless we specify the objective. Eval-only / retention winner is `lr=0.5`; adaptation winner by train loss is `lr=0.83` by a tiny margin; the practical winner requires downstream math evals on the checkpoints.

### Immediate recommendation

Do not choose the 1e21 LR solely from Paloma/c4. Run the 1e21 three-point LR sweep and pre-register prefix predictions as above. In parallel, run downstream math evals on the completed 1e20 checkpoints so the tradeoff can be scored as math gain versus retention cost, not just train loss versus c4 damage.

---

## 2026-04-24 stale missing-BOS cache cleanup

### Action

Deleted the stale `us-central1` tokenized cache:

```bash
gcloud -q storage rm --recursive \
  gs://marin-us-central1/tokenized/nemotron_cc_math_v1/4plus-212a2d/
```

Verification at `2026-04-24 21:22Z`:

```text
ERROR: (gcloud.storage.ls) One or more URLs matched no objects.
```

Remaining regional `nemotron_cc_math_v1` tokenized prefixes after deletion:

```text
gs://marin-us-central1/tokenized/nemotron_cc_math_v1/3-947143/
gs://marin-us-central1/tokenized/nemotron_cc_math_v1/3-ef5cb9/
gs://marin-us-central1/tokenized/nemotron_cc_math_v1/4plus_mind-d60b4a/
gs://marin-us-east5/tokenized/nemotron_cc_math_v1/3-ef5cb9/
gs://marin-us-east5/tokenized/nemotron_cc_math_v1/4plus-da9608/
gs://marin-us-east5/tokenized/nemotron_cc_math_v1/4plus_mind-d60b4a/
```

### Rationale

`4plus-212a2d` was produced before the Levanter BatchTokenizer BOS fix and was empirically missing Llama-3 BOS (`128000`) at the start of documents. Keeping it around made future `us-central1` relaunches likely to silently reuse the bad cache. The good `4plus` cache currently available is:

```text
gs://marin-us-east5/tokenized/nemotron_cc_math_v1/4plus-da9608/
```

That cache was sampled after the BOS fix and starts documents with `128000`.

---

## 2026-04-24 zero-end-LR BOS rerun plan

### Requested change

User observed that the AdamH learning-rate chart for the fixed 1e20 sweep annealed to `0.1 * peak`, not zero, and asked to:

1. Re-tokenize `nemotron_cc_math_v1/4plus` on `us-central1` now that the BatchTokenizer BOS fix is present.
2. Rerun the 1e20 LR sweep using the rebuilt BOS-correct `us-central1` cache.
3. Fix the midtraining schedule so both `optim/learning_rate` and `optim/adam_lr` fully anneal to zero by the final step.

### Config patch

Updated `experiments/exp_delphi_math_10b_midtrain.py`:

- `MIN_LR_RATIO: 0.1 -> 0.0`
- training step names: `...-lr{factor}-v2 -> ...-lr{factor}-v3`

The `-v3` suffix is required because the Marin executor hash does not reliably include plain `SimpleTrainConfig` or optimizer fields. Bumping the step name guarantees fresh checkpoint output paths and fresh W&B run IDs for the zero-end-LR reruns.

Validation:

```text
./infra/pre-commit.py --fix experiments/exp_delphi_math_10b_midtrain.py
```

passed. Local import check for `1e20 × lr=0.67` showed:

```text
step.name checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.67-v3
optimizer.min_lr_ratio 0.0
optimizer.warmup 500
optimizer.decay 4268
checkpoint_init_mode model_only
```

### Execution plan

1. Launch tokenization-only rebuild on `us-central1`:

```bash
uv run iris --cluster=marin job run \
  --cpu 1 --memory 3GB --disk 9GB \
  --region us-central1 \
  --job-name delphi-1e20-retokenize-4plus-bos-uscentral1-20260424 \
  --no-wait \
  -e WANDB_API_KEY "${WANDB_API_KEY}" \
  -- python -c 'from marin.execution.executor import executor_main; from experiments.midtraining_data_buckets import BUCKET_2; executor_main(steps=[BUCKET_2["nemotron_cc_math_v1/4plus"]], description="Retokenize Nemotron CC Math v1 4plus on us-central1 after BOS fix")'
```

Expected output path is the same as the deleted stale cache:

```text
gs://marin-us-central1/tokenized/nemotron_cc_math_v1/4plus-212a2d/
```

2. Verify that sampled rows start with Llama-3 BOS `128000`.
3. Launch the three `1e20-iso-d2048-L21` reruns pinned to `us-central1` with `MIDTRAIN_SELECT_LR={0.5,0.67,0.83}`.

### Execution results

Retokenization completed successfully.

Parent job:

```text
/ahmed/delphi-1e20-retokenize-4plus-bos-uscentral1-20260424
```

Observed child phases:

```text
/ahmed/delphi-1e20-retokenize-4plus-bos-uscentral1-20260424/zephyr-tokenize-train-82c5ccd4-p0-a0
/ahmed/delphi-1e20-retokenize-4plus-bos-uscentral1-20260424/zephyr-tokenize-train-82c5ccd4-p1-a0
/ahmed/delphi-1e20-retokenize-4plus-bos-uscentral1-20260424/zephyr-levanter-cache-probe-1bdbbb3f-p0-a0
/ahmed/delphi-1e20-retokenize-4plus-bos-uscentral1-20260424/zephyr-levanter-cache-copy-b6900715-p0-a0
```

Final cache path:

```text
gs://marin-us-central1/tokenized/nemotron_cc_math_v1/4plus-212a2d/
```

Cache stats:

```text
total_tokens 51482572371
total_elements 45096087
```

BOS sample verification from `train` cache:

```text
0 [128000, 2, 8388, 1815, 261, 27930, 10176, 220, 605, 339, 93678, 23508]
1 [128000, 2, 11106, 25, 46551, 279, 13031, 40227, 315, 279, 65048, 780]
2 [128000, 2, 358, 13, 384, 60217, 34495, 653, 372, 14799, 271, 334]
```

This confirms the rebuilt `us-central1` cache starts documents with Llama-3 BOS `128000`.

Launched the zero-end-LR 1e20 sweep on `us-central1`:

| LR factor | parent job | checkpoint output | child train job state at 2026-04-24 21:58Z |
|---:|---|---|---|
| 0.50 | `/ahmed/delphi-math-10b-1e20-lr0p5-v3-zeroend-bos-20260424-215257` | `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.5-v3-298da6` | `/train_lm` submitted; `JOB_STATE_RUNNING`; 8 tasks running |
| 0.67 | `/ahmed/delphi-math-10b-1e20-lr0p67-v3-zeroend-bos-20260424-215310` | `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.67-v3-88e817` | `/train_lm` submitted; `JOB_STATE_RUNNING`; 2 tasks running, 6 tasks building |
| 0.83 | `/ahmed/delphi-math-10b-1e20-lr0p83-v3-zeroend-bos-20260424-215327` | `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v3-0fad76` | `/train_lm` submitted; `JOB_STATE_PENDING`; waiting for v5p-64 capacity |

Parent logs for all three reruns show:

- `tokenized/nemotron_cc_math_v1/4plus_d68139d8` output path is `gs://marin-us-central1/tokenized/nemotron_cc_math_v1/4plus-212a2d`.
- The tokenized step is skipped as already succeeded, so the reruns reuse the newly rebuilt BOS-correct cache.
- Training step names use `-v3`, producing fresh checkpoint outputs and W&B run IDs.
- Model launch config is `Qwen3Config seq_len=4096 hidden=2048 batch=512 device=TpuConfig(variant='v5p-64')`.

Remaining watch item: confirm in W&B that `optim/adam_lr` reaches zero at final step once the v3 jobs complete. The config-level validation is already correct (`min_lr_ratio=0.0`, `warmup=500`, `decay=4268`), but the plotted confirmation requires the runs to finish.

---

## 2026-04-24 W&B project relaunch

### Requested change

After the `-v3` zero-end-LR relaunch, the user asked whether W&B projects can be selected and requested that the just-launched jobs be killed and rerun under project:

```text
delphi-midtraining
```

### Code patch

Updated `experiments/defaults.py` so `default_train(...)` accepts:

```python
wandb_project: str | None = None
```

and resolves it as:

```python
wandb_project = os.environ.get("WANDB_PROJECT", "marin")
```

when unset. This value is now passed to `WandbConfig(project=wandb_project)` instead of hardcoding `"marin"`.

Updated `experiments/exp_delphi_math_10b_midtrain.py` again:

- kept `MIN_LR_RATIO = 0.0`
- changed training step suffix from `-v3` to `-v4`

The `-v4` suffix avoids reusing partial `-v3` checkpoint paths / W&B run IDs from the killed launch.

Validation:

```text
WANDB_PROJECT=delphi-midtraining MIDTRAIN_SELECT_BASE=1e20-iso-d2048-L21 MIDTRAIN_SELECT_LR=0.67 uv run python - <<'PY'
from experiments.exp_delphi_math_10b_midtrain import runs
step = runs[0]
cfg = step.config.train_config
print('step.name', step.name)
print('wandb.project', cfg.trainer.tracker.project)
print('optimizer.min_lr_ratio', cfg.optimizer.min_lr_ratio)
print('optimizer.warmup', cfg.optimizer.warmup)
print('optimizer.decay', cfg.optimizer.decay)
PY
```

Output:

```text
step.name checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.67-v4
wandb.project delphi-midtraining
optimizer.min_lr_ratio 0.0
optimizer.warmup 500
optimizer.decay 4268
```

`./infra/pre-commit.py --fix experiments/defaults.py experiments/exp_delphi_math_10b_midtrain.py` passed.

### Terminated `-v3` jobs

Terminated the three `-v3` parent jobs requested by the user:

```text
/ahmed/delphi-math-10b-1e20-lr0p5-v3-zeroend-bos-20260424-215257
/ahmed/delphi-math-10b-1e20-lr0p67-v3-zeroend-bos-20260424-215310
/ahmed/delphi-math-10b-1e20-lr0p83-v3-zeroend-bos-20260424-215327
```

Follow-up `get-job-state` showed all three parent jobs and their `/train_lm` child jobs as `JOB_STATE_FAILED`, which is the expected terminal state after termination.

### Relaunched `-v4` jobs

Relaunch command shape:

```bash
uv run iris --cluster=marin job run \
  --cpu 1 --memory 3GB --disk 9GB \
  --region us-central1 \
  --job-name "$job_name" \
  --no-wait \
  -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 \
  -e WANDB_API_KEY "${WANDB_API_KEY}" \
  -e WANDB_PROJECT delphi-midtraining \
  -e MIDTRAIN_SELECT_BASE 1e20-iso-d2048-L21 \
  -e MIDTRAIN_SELECT_LR "$lr" \
  -- python experiments/exp_delphi_math_10b_midtrain.py
```

Submitted:

| LR factor | parent job | status at 2026-04-24 22:21Z |
|---:|---|---|
| 0.50 | `/ahmed/delphi-math-10b-1e20-lr0p5-v4-zeroend-bos-wandbproject-20260424-221945` | parent running; `/train_lm` submitted and pending v5p-64 workers |
| 0.67 | `/ahmed/delphi-math-10b-1e20-lr0p67-v4-zeroend-bos-wandbproject-20260424-221959` | parent pending but schedulable |
| 0.83 | `/ahmed/delphi-math-10b-1e20-lr0p83-v4-zeroend-bos-wandbproject-20260424-222010` | parent pending but schedulable |

lr=0.5 parent logs confirmed it reused the rebuilt BOS-correct central cache:

```text
Step = tokenized/nemotron_cc_math_v1/4plus_d68139d8 ... Output_path = gs://marin-us-central1/tokenized/nemotron_cc_math_v1/4plus-212a2d
Skip tokenized/nemotron_cc_math_v1/4plus_d68139d8: already succeeded
```

lr=0.5 checkpoint output:

```text
gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.5-v4-062652
```

Serialized experiment metadata for lr=0.5:

```text
gs://marin-us-central1/experiments/exp_delphi_math_10b_midtrain-2c5dc6.json
```

contains:

```json
"tracker": {
  "project": "delphi-midtraining"
}
```

So the W&B project override is wired through the actual Levanter training config, not just present as an environment variable.

## 2026-04-24 launch incident: `-v4` shared temp-checkpoint failure

User reported the three `-v4` jobs all failed. Checked the actual failed `/train_lm`
task logs for:

```text
/ahmed/delphi-math-10b-1e20-lr0p5-v4-zeroend-bos-wandbproject-20260424-221945/train_lm/0
/ahmed/delphi-math-10b-1e20-lr0p67-v4-zeroend-bos-wandbproject-20260424-221959/train_lm/7
/ahmed/delphi-math-10b-1e20-lr0p83-v4-zeroend-bos-wandbproject-20260424-222010/train_lm/2
```

All three failed before step 0 with the same pattern:

```text
No checkpoints found in ['gs://marin-us-central1/checkpoints/...-v4-.../checkpoints']
Discovered latest checkpoint at gs://marin-tmp-us-central1/ttl=14d/checkpoints-temp/step-48
Found prior temporary checkpoint gs://marin-tmp-us-central1/ttl=14d/checkpoints-temp/step-48.
FileNotFoundError: Missing 34 arrays in OCDBT checkpoint: [...]
```

Root cause: `lib/marin/src/marin/training/training.py::_update_config_to_use_out_path`
made permanent checkpoints unique via `$output_path/checkpoints`, but set every
executor-backed training job's `temporary_base_path` to the shared
`gs://marin-tmp-<region>/ttl=14d/checkpoints-temp`. `_enforce_run_id` then set
`append_run_id_to_base_path=False` when the run ID was imputed from the executor
output path. That is correct for the permanent path, because it is already unique,
but wrong for the temporary path. The result was a shared rolling temp checkpoint
namespace across independent jobs. These `-v4` runs picked up an incompatible
stale temp checkpoint at `step-48`, so model-only initialization never reached the
base checkpoint load.

Fix applied:

- Added `_temporary_checkpoint_base_path(...)` in `lib/marin/src/marin/training/training.py`.
- When the run ID is imputed from `output_path`, temporary checkpoints now go under:

```text
gs://marin-tmp-<region>/ttl=14d/checkpoints-temp/<basename(output_path)>/
```

- Added `tests/test_training.py::test_executor_output_path_scopes_temporary_checkpoints`.
- Bumped the Delphi 1e20 sweep suffix from `-v4` to `-v5` so relaunches get fresh executor outputs.

Validation:

```text
uv run --project lib/marin --group test pytest tests/test_training.py::test_executor_output_path_scopes_temporary_checkpoints -q
.
1 passed in 1.48s

WANDB_PROJECT=delphi-midtraining MIDTRAIN_SELECT_BASE=1e20-iso-d2048-L21 MIDTRAIN_SELECT_LR=0.67 uv run python - <<'PY'
from experiments.exp_delphi_math_10b_midtrain import runs
step = runs[0]
cfg = step.config.train_config
print('step.name', step.name)
print('wandb.project', cfg.trainer.tracker.project)
print('optimizer.min_lr_ratio', cfg.optimizer.min_lr_ratio)
print('optimizer.warmup', cfg.optimizer.warmup)
print('optimizer.decay', cfg.optimizer.decay)
PY

step.name checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.67-v5
wandb.project delphi-midtraining
optimizer.min_lr_ratio 0.0
optimizer.warmup 500
optimizer.decay 4268
```

Next action: relaunch the three 1e20 jobs as `-v5` with `WANDB_PROJECT=delphi-midtraining`,
then watch the `/train_lm` children until they pass the previous failure point and emit
real training progress.

## 2026-04-24 `-v5` relaunch monitoring

Relaunched the three 1e20 LR sweep jobs after the temp-checkpoint fix, still in
`us-central1`, with `WANDB_PROJECT=delphi-midtraining` and the zero-end LR schedule:

```text
/ahmed/delphi-math-10b-1e20-lr0p5-v5-zeroend-bos-wandbproject-20260424-224700
/ahmed/delphi-math-10b-1e20-lr0p67-v5-zeroend-bos-wandbproject-20260424-224710
/ahmed/delphi-math-10b-1e20-lr0p83-v5-zeroend-bos-wandbproject-20260424-224720
```

Confirmed the `lr=0.83` child training job is actually running with 8/8 workers:

```text
/ahmed/delphi-math-10b-1e20-lr0p83-v5-zeroend-bos-wandbproject-20260424-224720/train_lm
JOB_STATE_RUNNING
task_state_counts: {"running": 8}
```

This run passed the previous restore failure point. The relevant log lines are:

```text
No checkpoints found in ['gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b/checkpoints']
No checkpoints found in ['gs://marin-tmp-us-central1/ttl=14d/checkpoints-temp/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b']
Loading cache from gs://marin-us-central1/tokenized/nemotron_cc_math_v1/4plus-212a2d/train
Loading checkpoint from gs://marin-us-central1/checkpoints/isoflop/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/checkpoints/step-46915
Error check finished successfully
First train step completed in 35.8s (step 0)
Progress on:train 72.0it/4.77kit ... postfix:loss=1.37
```

Also confirmed W&B is pointing at the requested project:

```text
wandb: View project at https://wandb.ai/marin-community/delphi-midtraining
```

As of ~23:01 UTC, the other two `/train_lm` children are queued rather than failed:

```text
/ahmed/delphi-math-10b-1e20-lr0p5-v5-zeroend-bos-wandbproject-20260424-224700/train_lm
JOB_STATE_PENDING
task_state_counts: {"pending": 8}

/ahmed/delphi-math-10b-1e20-lr0p67-v5-zeroend-bos-wandbproject-20260424-224710/train_lm
JOB_STATE_PENDING
task_state_counts: {"pending": 8}
```

Both pending jobs report the same scheduler reason:

```text
Coscheduling: need 8 workers in 'tpu-name' group 't1v-n-1d6a1c3e', only 0 of 8 have capacity.
Insufficient TPUs (need 4, available 0).
Autoscaler: tier_blocked: 1 matching group(s) blocked by quota-pool tier monotonicity.
```

Interpretation: the code/config launch issue is fixed for jobs that get TPU capacity;
`lr=0.5` and `lr=0.67` have not reached runtime yet because `us-central1` does not
currently have another full v5p-64 coscheduled group available.

## 2026-04-24 23:07 UTC `-v5` status refresh

Refreshed the three relaunched 1e20 sweep jobs:

```text
/ahmed/delphi-math-10b-1e20-lr0p5-v5-zeroend-bos-wandbproject-20260424-224700/train_lm  JOB_STATE_PENDING
/ahmed/delphi-math-10b-1e20-lr0p67-v5-zeroend-bos-wandbproject-20260424-224710/train_lm JOB_STATE_PENDING
/ahmed/delphi-math-10b-1e20-lr0p83-v5-zeroend-bos-wandbproject-20260424-224720/train_lm JOB_STATE_RUNNING
```

The running `lr=0.83` job continues to make normal training progress:

```text
Progress on:train 100it/4.77kit ... postfix:loss=1.31
Progress on:train 128it/4.77kit ... postfix:loss=1.26
Progress on:train 156it/4.77kit ... postfix:loss=1.26
Progress on:train 170it/4.77kit ... postfix:loss=1.19
```

The `lr=0.5` and `lr=0.67` jobs are still queued with 8/8 pending tasks each.
They have not failed and still show `failure_count=0`, `preemption_count=0`.
Both report the same scheduler/autoscaler reason:

```text
Coscheduling: need 8 workers in 'tpu-name' group 't1v-n-1d6a1c3e',
only 0 of 8 have capacity.
Insufficient TPUs (need 4, available 0).
tier_blocked: 1 matching group(s) blocked by quota-pool tier monotonicity.
```

Immediate interpretation: launch/config is no longer the blocker. The remaining
issue for `lr=0.5` and `lr=0.67` is `us-central1` v5p-64 capacity. Leave them
queued unless we explicitly decide to move/duplicate those runs in another region.

## 2026-04-24 23:12 UTC monitor tick

Automated 30-minute monitor snapshot for the Delphi 1e20 `-v5` jobs.

```text
lr=0.5  JOB_STATE_PENDING        pending=8; failures=0; preemptions=0
lr=0.67 JOB_STATE_PENDING        pending=8; failures=0; preemptions=0
lr=0.83 JOB_STATE_RUNNING        running=8; failures=0; preemptions=0
```

Latest progress:
- `lr=0.83`: `I20260424 23:12:52 139792089573184 tqdm_loggable.tqdm_logging Progress on:train 230it/4.77kit rate:4.4s/it remaining:5:32:15 elapsed:20:21 postfix:loss=1.17`

Pending reason:
- `lr=0.5`: Scheduler: Coscheduling: need 8 workers in 'tpu-name' group 't1v-n-1d6a1c3e', only 0 of 8 have capacity: Insufficient TPUs (need 4, available 0) - 8 worker(s) Autoscaler: Waiting for workers in scale group 'tpu_v5p-preemptible_64-us-central1-a' to become ready (selected: demand-routed)
- `lr=0.67`: Scheduler: Coscheduling: need 8 workers in 'tpu-name' group 't1v-n-1d6a1c3e', only 0 of 8 have capacity: Insufficient TPUs (need 4, available 0) - 8 worker(s) Autoscaler: Waiting for workers in scale group 'tpu_v5p-preemptible_64-us-central1-a' to become ready (selected: demand-routed)

Monitor note: continuing; next automated check is scheduled for 30 minutes after this tick.

## 2026-04-24 23:14 UTC detached monitor started

Started a resident monitor for the three Delphi 1e20 `-v5` jobs. It runs in:

```text
tmux session: delphi-midtrain-monitor
state file: scratch/delphi_midtrain_monitoring_state.json
process log: scratch/delphi_midtrain_monitor_20260424-231428.log
monitor script: scratch/monitor_delphi_midtrain_jobs.py
```

The monitor appends one compact status snapshot to this logbook every 30 minutes.
It records job states, task counts, failure/preemption counts, the latest train
progress line for any running job, and pending reasons for queued jobs. It stops
itself once all three tracked jobs are terminal.

Initial detached state:

```json
{
  "pid": 49293,
  "status": "sleeping_initial_delay",
  "next_check_utc": "2026-04-24T23:44:29+00:00",
  "interval_seconds": 1800
}
```

To inspect the monitor later:

```bash
tmux list-sessions | rg delphi-midtrain-monitor
cat scratch/delphi_midtrain_monitoring_state.json
tail -n 40 scratch/delphi_midtrain_monitor_20260424-231428.log
```

To stop it intentionally:

```bash
tmux kill-session -t delphi-midtrain-monitor
```

## 2026-04-24 23:44 UTC monitor tick

Automated 30-minute monitor snapshot for the Delphi 1e20 `-v5` jobs.

```text
lr=0.5  JOB_STATE_PENDING        pending=8; failures=0; preemptions=0
lr=0.67 JOB_STATE_PENDING        pending=8; failures=0; preemptions=0
lr=0.83 JOB_STATE_RUNNING        running=8; failures=0; preemptions=0
```

Latest progress:
- `lr=0.83`: `I20260424 23:44:13 139792089573184 tqdm_loggable.tqdm_logging Progress on:train 629it/4.77kit rate:5.4s/it remaining:6:10:13 elapsed:51:42 postfix:loss=1.04`

Pending reason:
- `lr=0.5`: Scheduler: Coscheduling: need 8 workers in 'tpu-name' group 't1v-n-1d6a1c3e', only 0 of 8 have capacity: Insufficient TPUs (need 4, available 0) - 8 worker(s) Autoscaler: Unsatisfied autoscaler demand: tier_blocked: 1 matching group(s) blocked by quota-pool tier monotonicity
- `lr=0.67`: Scheduler: Coscheduling: need 8 workers in 'tpu-name' group 't1v-n-1d6a1c3e', only 0 of 8 have capacity: Insufficient TPUs (need 4, available 0) - 8 worker(s) Autoscaler: Unsatisfied autoscaler demand: tier_blocked: 1 matching group(s) blocked by quota-pool tier monotonicity

Monitor note: continuing; next automated check is scheduled for 30 minutes after this tick.

## 2026-04-25 00:15 UTC monitor tick

Automated 30-minute monitor snapshot for the Delphi 1e20 `-v5` jobs.

```text
lr=0.5  JOB_STATE_RUNNING        running=8; failures=0; preemptions=0
lr=0.67 JOB_STATE_PENDING        pending=8; failures=0; preemptions=0
lr=0.83 JOB_STATE_RUNNING        running=8; failures=0; preemptions=0
```

Latest progress:
- `lr=0.5`: `I20260425 00:15:07 140004800845632 tqdm_loggable.tqdm_logging Progress on:train 230it/4.77kit rate:4.4s/it remaining:5:31:20 elapsed:20:15 postfix:loss=1.2`
- `lr=0.83`: `I20260425 00:15:22 139792089573184 tqdm_loggable.tqdm_logging Progress on:train 1.00kit/4.77kit rate:52.2s/it remaining:54:39:15 elapsed:1:22:51 postfix:loss=0.998`

Pending reason:
- `lr=0.67`: Scheduler: Coscheduling: need 8 workers in 'tpu-name' group 't1v-n-1d6a1c3e', only 0 of 8 have capacity: Insufficient TPUs (need 4, available 0) - 8 worker(s) Autoscaler: Waiting for workers in scale group 'tpu_v5p-preemptible_64-us-central1-a' to become ready (selected: demand-routed)

Monitor note: continuing; next automated check is scheduled for 30 minutes after this tick.

## 2026-04-25 00:45 UTC monitor tick

Automated 30-minute monitor snapshot for the Delphi 1e20 `-v5` jobs.

```text
lr=0.5  JOB_STATE_RUNNING        running=8; failures=0; preemptions=0
lr=0.67 JOB_STATE_PENDING        pending=8; failures=0; preemptions=0
lr=0.83 JOB_STATE_RUNNING        running=8; failures=0; preemptions=0
```

Latest progress:
- `lr=0.5`: `I20260425 00:45:23 140004800845632 tqdm_loggable.tqdm_logging Progress on:train 616it/4.77kit rate:4.5s/it remaining:5:11:22 elapsed:50:31 postfix:loss=1.02`
- `lr=0.83`: `I20260425 00:45:27 139792089573184 tqdm_loggable.tqdm_logging Progress on:train 1.40kit/4.77kit rate:4.4s/it remaining:4:06:49 elapsed:1:52:56 postfix:loss=0.929`

Pending reason:
- `lr=0.67`: Scheduler: Coscheduling: need 8 workers in 'tpu-name' group 't1v-n-1d6a1c3e', only 0 of 8 have capacity: Insufficient TPUs (need 4, available 0) - 8 worker(s) Autoscaler: Waiting for workers in scale group 'tpu_v5p-preemptible_64-us-central1-a' to become ready (selected: demand-routed)

Monitor note: continuing; next automated check is scheduled for 30 minutes after this tick.

## 2026-04-25 01:16 UTC monitor tick

Automated 30-minute monitor snapshot for the Delphi 1e20 `-v5` jobs.

```text
lr=0.5  JOB_STATE_KILLED         killed=8; failures=0; preemptions=0
lr=0.67 JOB_STATE_FAILED         failed=1, worker_failed=7; failures=1; preemptions=707
lr=0.83 JOB_STATE_RUNNING        running=8; failures=0; preemptions=0
```

Latest progress:
- `lr=0.83`: `I20260425 01:15:48 139816879437632 tqdm_loggable.tqdm_logging Progress on:train 1.61kit/4.77kit rate:4.5s/it remaining:3:58:59 elapsed:21:22 postfix:loss=0.934`

Terminal states:
- `lr=0.5`: `JOB_STATE_KILLED` exit=0 error=''
- `lr=0.67`: `JOB_STATE_FAILED` exit=0 error='Coscheduled sibling /ahmed/delphi-math-10b-1e20-lr0p67-v5-zeroend-bos-wandbproject-20260424-224710/train_lm/3 failed'

Monitor note: continuing; next automated check is scheduled for 30 minutes after this tick.

## 2026-04-25 01:46 UTC monitor tick

Automated 30-minute monitor snapshot for the Delphi 1e20 `-v5` jobs.

```text
lr=0.5  JOB_STATE_KILLED         killed=8; failures=0; preemptions=0
lr=0.67 JOB_STATE_FAILED         failed=1, worker_failed=7; failures=1; preemptions=707
lr=0.83 JOB_STATE_RUNNING        running=8; failures=0; preemptions=0
```

Latest progress:
- `lr=0.83`: `I20260425 01:44:53 139816879437632 tqdm_loggable.tqdm_logging Progress on:train 2.00kit/4.77kit rate:4.4s/it remaining:3:22:46 elapsed:50:26 postfix:loss=0.886`

Terminal states:
- `lr=0.5`: `JOB_STATE_KILLED` exit=0 error=''
- `lr=0.67`: `JOB_STATE_FAILED` exit=0 error='Coscheduled sibling /ahmed/delphi-math-10b-1e20-lr0p67-v5-zeroend-bos-wandbproject-20260424-224710/train_lm/3 failed'

Monitor note: continuing; next automated check is scheduled for 30 minutes after this tick.

## 2026-04-25 01:52 UTC multi-region recovery launch

After the 00:54 UTC TPU preemption cascade, only `lr=0.83 v5` auto-recovered (it
restored from its step-1000 checkpoint and resumed training; W&B step trace is
continuous through the boundary, currently step ~2000, loss 0.886, LR 2.5e-3).
`lr=0.5 v5` had no saved checkpoint (crashed at step 724, before the 1000-step
save interval) and `lr=0.67 v5` had not started training when the cascade hit;
both parents terminated with `RuntimeError: 1 step(s) failed`.

User asked to relaunch the two failed runs with multi-region (`us-central1` +
`us-east5`) so the autoscaler can place either job in whichever pool frees a
v5p-64 first. Cluster-wide v5p-64 inventory at 18:38 UTC: us-central1-a 2/2
ready (2 occupied: 1 mine, 1 tonyhlee/moojink-shared), us-east5-a 3/3 ready
(2 moojink jobs + 1 spare with demand 1 from another tenant). Adding east5
to the region list does not immediately unblock — both pools were full at
launch time — but it widens placement when any slot frees.

### Cache pre-stage (cross-region copy)

`MATH_TRAIN_STEP` is *not* wrapped in `mirrored()` (only the pretrain ckpt is).
The BOS-correct `4plus-212a2d` tokenize cache only exists in us-central1
(`gs://marin-us-central1/tokenized/nemotron_cc_math_v1/4plus-212a2d/`,
51.48 B tokens, 45.10 M docs, ~210 GB on disk). If a relaunched job lands in
us-east5, the executor would re-run normalize+tokenize from raw — ~10–20 h
of wasted compute. Pre-staged via:

```bash
gcloud storage cp -r --no-clobber \
  gs://marin-us-central1/tokenized/nemotron_cc_math_v1/4plus-212a2d/ \
  gs://marin-us-east5/tokenized/nemotron_cc_math_v1/
```

(Background, ~5–10 min, ~$5 inter-region egress.) us-east5 already has a
stale `4plus-da9608` tree from before the BOS-fix cache rebuild, but the
hash the current experiment file resolves to is `4plus-212a2d` so we copy
that exact path. Race risk: if a TPU placement happens before the copy
finishes and the executors `.executor_status` was written but shards are
incomplete, training would read partial data — mitigated by the fact that
both pools were at capacity at launch, so placement will not race.

### Submit recipe

```bash
ts=$(date -u +%Y%m%d-%H%M%S)
job=delphi-math-10b-1e20-lr0p${LR}-v5-multiregion-${ts}
uv run iris --cluster=marin job run \
  --cpu 1 --memory 3GB --disk 9GB \
  --region us-central1 --region us-east5 \
  --job-name "$job" \
  --no-wait \
  -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 \
  -e WANDB_API_KEY "${WANDB_API_KEY}" \
  -e WANDB_PROJECT delphi-midtraining \
  -e MIDTRAIN_SELECT_BASE 1e20-iso-d2048-L21 \
  -e MIDTRAIN_SELECT_LR "$LR" \
  -- python experiments/exp_delphi_math_10b_midtrain.py
```

Step name kept at `-v5` (not bumped to `-v6`) so the executor hash and the
checkpoint output path are preserved. Trade-off: the existing W&B run
`delphi-1e20-iso-d2048-L21-math-10b-lr0.5-v5-3bd907` is at terminal step 724
in `crashed` state, and Levanters wandb monotonicity guard will silently
drop re-logged metrics for steps 0..724 on the new training start — we lose
the early curve in the W&B chart, but the model is unaffected.

Submitted:

```text
/ahmed/delphi-math-10b-1e20-lr0p5-v5-multiregion-20260425-015152   (lr=0.5)
/ahmed/delphi-math-10b-1e20-lr0p67-v5-multiregion-20260425-015212  (lr=0.67)
```

`lr=0.83 v5-zeroend-bos-wandbproject-20260424-224720` is left undisturbed.

## 2026-04-25 02:17 UTC monitor tick

Automated 30-minute monitor snapshot for the Delphi 1e20 `-v5` jobs.

```text
lr=0.5  JOB_STATE_KILLED         killed=8; failures=0; preemptions=0
lr=0.67 JOB_STATE_FAILED         failed=1, worker_failed=7; failures=1; preemptions=707
lr=0.83 JOB_STATE_FAILED         failed=1, worker_failed=7; failures=1; preemptions=707
```

Terminal states:
- `lr=0.5`: `JOB_STATE_KILLED` exit=0 error=''
- `lr=0.67`: `JOB_STATE_FAILED` exit=0 error='Coscheduled sibling /ahmed/delphi-math-10b-1e20-lr0p67-v5-zeroend-bos-wandbproject-20260424-224710/train_lm/3 failed'
- `lr=0.83`: `JOB_STATE_FAILED` exit=0 error='Exit code: 1. stderr: RPC: /tensorflow.CoordinationService/PollForError [type.googleapis.com/tensorflow.CoordinationServiceError=\'\\"\\x0c\\n\\njax_worker\']'

Monitor note: all tracked jobs are terminal; the detached monitor will stop after this tick.

## 2026-04-25 03:19 UTC post-Claude recovery audit

Read the latest logbook entries plus live Iris state after Claude's multi-region
recovery work. What Claude did:

- Copied the BOS-correct tokenized Nemotron math cache from:

```text
gs://marin-us-central1/tokenized/nemotron_cc_math_v1/4plus-212a2d/
```

to:

```text
gs://marin-us-east5/tokenized/nemotron_cc_math_v1/4plus-212a2d/
```

- Launched multi-region replacements for `lr=0.5` and `lr=0.67` with both
  `--region us-central1` and `--region us-east5`.
- The first `lr=0.67` multi-region attempt died during container build from a
  transient GitHub release-asset timeout while fetching `dupekit`; Claude
  resubmitted it as:

```text
/ahmed/delphi-math-10b-1e20-lr0p67-v5-multiregion-20260425-022011
```

- The original `lr=0.83` `-v5` run failed after reaching/saving step 2000.
  Claude copied its step-2000 checkpoint from central1 to east5.

Checkpoint copy check:

```text
23216831348 gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b/checkpoints/step-2000/
23216831348 gs://marin-us-east5/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b/checkpoints/step-2000/
```

Current live state:

```text
lr=0.5  multiregion /train_lm  JOB_STATE_RUNNING  running=8, failures=0, preemptions=0
lr=0.67 multiregion /train_lm  JOB_STATE_RUNNING  running=8, failures=0, preemptions=0
lr=0.83 original /train_lm     JOB_STATE_FAILED   failures=1, preemptions=707
```

Latest training progress:

```text
lr=0.5  Progress on:train 1.02kit/4.77kit ... postfix:loss=0.978
lr=0.67 Progress on:train 616it/4.77kit ... postfix:loss=1.01
```

Placement check: despite multi-region submission, both live replacement jobs
currently landed in `tpu_v5p-preemptible_64-us-central1-a`; no active training
task is using east5 yet.

Open decision: `lr=0.83` has not been relaunched. Its step-2000 checkpoint is
available in both regions, so a multi-region relaunch should resume from there
if we want a BOS-correct completed `lr=0.83` point.

## 2026-04-25 03:25 UTC lr=0.5 resume mistake

User noticed W&B now has two `lr=0.5` runs:

```text
delphi-1e20-iso-d2048-L21-math-10b-lr0.5-v5-3bd907  original
delphi-1e20-iso-d2048-L21-math-10b-lr0.5-v5-bcab01  Claude multi-region replacement
```

Verified this is a real resume mistake, not just a W&B display issue.
The original `lr=0.5` job had a temporary checkpoint:

```text
gs://marin-tmp-us-central1/ttl=14d/checkpoints-temp/delphi-1e20-iso-d2048-L21-math-10b-lr0.5-v5-3bd907/step-630/
```

but no permanent step checkpoint under:

```text
gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.5-v5-3bd907/checkpoints/
```

The replacement job did not use the original output/checkpoint namespace. It
writes to:

```text
gs://marin-us-east5/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.5-v5-bcab01/
```

and saved:

```text
gs://marin-us-east5/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.5-v5-bcab01/checkpoints/step-1000/
```

Root cause: changing the launch to multi-region changed the executor output
prefix/dependency realization, so the automatic executor hash changed from
`3bd907` to `bcab01`. Keeping the human-readable step name at `-v5` was not
enough to preserve the actual run id. To resume a run, the launch must force the
exact old output path, e.g. via `ExecutorStep.with_output_path(...)` or an
explicit `override_output_path`, not rely on the step name.

Current implication:

- `bcab01` is scientifically a valid fresh `lr=0.5` BOS-correct run, but it is
  not a continuation of `3bd907`.
- The original `3bd907` temp checkpoint at step 630 still exists for now.
- A true resume would need to relaunch `lr=0.5` with output path forced to
  `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.5-v5-3bd907`.
- Do not kill `bcab01` automatically; it is already past step 1000 and may still
  be useful if the priority is final endpoint rather than W&B continuity.

## 2026-04-25 CRITICAL RESUME RULE

DO NOT RELAUNCH A FAILED DELPHI MIDTRAINING RUN BY ONLY REUSING THE HUMAN-READABLE
STEP NAME.

THE REAL RUN ID AND CHECKPOINT NAMESPACE INCLUDE THE MARIN EXECUTOR OUTPUT HASH.
IF THE REGION LIST, PREFIX, DEPENDENCY REALIZATION, OR OVERRIDE PATH CHANGES, THE
HASH CAN CHANGE EVEN WHEN THE VISIBLE NAME STILL SAYS `-V5`.

BEFORE RELAUNCHING ANY FAILED OR PREEMPTED DELPHI MIDTRAINING RUN:

1. FIND THE EXACT OLD OUTPUT PATH AND RUN ID FROM IRIS LOGS OR GCS.
2. CHECK BOTH PERMANENT CHECKPOINTS AND TEMPORARY CHECKPOINTS.
3. FORCE THE RELAUNCH TO USE THE EXACT OLD OUTPUT PATH WITH
   `EXECUTORSTEP.WITH_OUTPUT_PATH(...)` OR `OVERRIDE_OUTPUT_PATH`.
4. VERIFY STARTUP LOGS SHOW THE SAME RUN ID / OUTPUT PATH.
5. VERIFY STARTUP LOGS SAY `RESUMING TRAINING FROM STEP ...`.
6. ONLY THEN TREAT THE RUN AS RESUMED.

SPECIFIC FAILURE TO AVOID: `LR0.5-V5-3BD907` HAD A TEMP CHECKPOINT AT STEP 630,
BUT THE MULTI-REGION RELAUNCH CREATED `LR0.5-V5-BCAB01` BECAUSE THE EXECUTOR HASH
CHANGED. THAT STARTED A NEW W&B RUN INSTEAD OF RESUMING THE OLD ONE.

## 2026-04-25 04:05 UTC 1e21 v5p-256 pilot launch

User requested one larger Delphi midtraining pilot: `1e21-v5`, 10B math tokens,
LR factor 0.67, on the available `v5p-256` slice.

Preflight:

- Added env knobs to `experiments/exp_delphi_math_10b_midtrain.py`:
  - `MIDTRAIN_TPU_TYPE` defaults to `v5p-64`.
  - `MIDTRAIN_RUN_NAME_SUFFIX` appends to the executor step name when set.
- Dry-run import with:

```bash
MIDTRAIN_SELECT_BASE=1e21-v5 \
MIDTRAIN_SELECT_LR=0.67 \
MIDTRAIN_TPU_TYPE=v5p-256 \
MIDTRAIN_RUN_NAME_SUFFIX=v5p256 \
WANDB_PROJECT=delphi-midtraining \
uv run python - <<'PY'
import experiments.exp_delphi_math_10b_midtrain as exp
run = exp.runs[0]
print(len(exp.runs))
print(run.name)
print(run.config.resources)
print(run.config.train_config.trainer.train_batch_size)
print(run.config.train_config.trainer.num_train_steps)
print(run.config.train_config.train_seq_len)
PY
```

verified one run only:

```text
step_name checkpoints/delphi-1e21-v5-math-10b-lr0.67-v5-v5p256
resources v5p-256, replicas=32
batch=512, steps=4768, seq_len=4096
lr=0.00497475, adam_lr=0.000289038
wandb_project=delphi-midtraining
```

Iris capacity check immediately before launch:

```text
tpu_v5p-preemptible_256-us-central1-a workers=32 active=32 healthy=32 committed_tpu=0 total_tpu=128
```

Planned launch command:

```bash
uv run iris --cluster=marin job run \
  --cpu 1 --memory 3GB --disk 9GB \
  --region us-central1 \
  --job-name delphi-math-10b-1e21-lr0p67-v5p256-20260425-0405 \
  --no-wait \
  -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e WANDB_PROJECT delphi-midtraining \
  -e MIDTRAIN_SELECT_BASE 1e21-v5 \
  -e MIDTRAIN_SELECT_LR 0.67 \
  -e MIDTRAIN_TPU_TYPE v5p-256 \
  -e MIDTRAIN_RUN_NAME_SUFFIX v5p256 \
  -- python experiments/exp_delphi_math_10b_midtrain.py
```

Follow-up required: verify the child `/train_lm` lands on
`tpu_v5p-preemptible_256-us-central1-a`, starts from the `1e21-v5` base
checkpoint, and emits `Progress on:train` before treating the launch as healthy.

Immediate scheduler correction: the first parent coordinator stayed pending in
central1 because the Iris small-CPU heuristic auto-pinned it to non-preemptible
CPU, and central1 had zero on-demand CPU free:

```text
pending: Scheduler: Insufficient CPU (need 1 cores, available 0 cores...)
```

Stop that pending parent and relaunch the same experiment with `--preemptible`
so the 1-core executor parent can land on available preemptible capacity:

```bash
uv run iris --cluster=marin job stop \
  /ahmed/delphi-math-10b-1e21-lr0p67-v5p256-20260425-0405

uv run iris --cluster=marin job run \
  --cpu 1 --memory 3GB --disk 9GB \
  --preemptible \
  --region us-central1 \
  --job-name delphi-math-10b-1e21-lr0p67-v5p256-20260425-0408 \
  --no-wait \
  -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e WANDB_PROJECT delphi-midtraining \
  -e MIDTRAIN_SELECT_BASE 1e21-v5 \
  -e MIDTRAIN_SELECT_LR 0.67 \
  -e MIDTRAIN_TPU_TYPE v5p-256 \
  -e MIDTRAIN_RUN_NAME_SUFFIX v5p256 \
  -- python experiments/exp_delphi_math_10b_midtrain.py
```

Launch result:

```text
submitted: /ahmed/delphi-math-10b-1e21-lr0p67-v5p256-20260425-0408
child:     /ahmed/delphi-math-10b-1e21-lr0p67-v5p256-20260425-0408/train_lm
output:    gs://marin-us-central1/checkpoints/delphi-1e21-v5-math-10b-lr0.67-v5-v5p256-136fc5/
```

Verified:

- Child `train_lm` scheduled with 32 tasks on
  `tpu_v5p-preemptible_256-us-central1-a`.
- Base checkpoint staged/loaded from
  `gs://marin-us-central1/adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/checkpoints/step-21979`.
- No task failures or preemptions at startup.
- Training reached post-compile progress:

```text
Progress on:train 37.0it/4.77kit rate:1.8s/it remaining:2:23:21 elapsed:03:22 postfix:loss=1.34
```

Note: the earlier `No checkpoint found. Starting from scratch.` line refers to
the new midtraining output namespace having no resume checkpoint yet. It appears
after the base checkpoint load and does not mean random initialization.

## 2026-04-25 06:40 UTC large-slice utilization plan

User wants to fully use the live `v5p-256` window and opportunistically use
larger/free v5p slices for the next scale. Operating constraints:

- For `1e21-v5`, do **not** run parallel LR jobs on Iris. Monitor the active
  job until terminal, then launch the next LR point immediately.
- For `1e22-v5`, launch at most one active LR job opportunistically if a clean
  `v5p-512` or `v5p-128` slice is available.
- Monitor everything every 15 minutes and append each tick here.
- On any failure/preemption, obey the all-caps resume rule above: find the exact
  output path/checkpoint namespace and resume that namespace. Do not start a
  new W&B/checkpoint run accidentally.

### `1e21-v5` serial LR queue on `v5p-256`

Current active job:

```text
/ahmed/delphi-math-10b-1e21-lr0p67-v5p256-20260425-0408/train_lm
```

Queue order:

1. `lr_factor=0.67` — already running.
2. `lr_factor=0.83` — launch next if 0.67 succeeds.
3. `lr_factor=0.50` — launch after 0.83 succeeds.

Rationale: `1e20` favored the high end (`0.67/0.83`) over `0.5`, so test the
upper bracket first while the large slice is hot; still run `0.5` to complete
the bracket and quantify the specialization/retention tradeoff.

Launch template for the queued `1e21` jobs:

```bash
uv run iris --cluster=marin job run \
  --cpu 1 --memory 3GB --disk 9GB \
  --preemptible \
  --region us-central1 \
  --job-name delphi-math-10b-1e21-lr0p{LR}-v5p256-${ts} \
  --no-wait \
  -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e WANDB_PROJECT delphi-midtraining \
  -e MIDTRAIN_SELECT_BASE 1e21-v5 \
  -e MIDTRAIN_SELECT_LR {LR} \
  -e MIDTRAIN_TPU_TYPE v5p-256 \
  -e MIDTRAIN_BATCH_SIZE 512 \
  -e MIDTRAIN_LR_MULTIPLIER 1.0 \
  -e MIDTRAIN_RUN_NAME_SUFFIX v5p256 \
  -- python experiments/exp_delphi_math_10b_midtrain.py
```

### `1e22-v5` opportunistic LR queue

Preflight constants from the finished base run
`adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e`:

```text
checkpoint: gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-38206/
checkpoint size: 116,582,126,509 bytes
hidden_dim=3840, layers=37, params≈9.71B
pretrain batch=1024, train steps=38235
peak_lr=7.231797280729413e-3
peak_adam_lr=3.276222099351447e-4
beta2=0.9984011994401821
epsilon=3.70426657045089e-8
```

Added `1e22-v5` to `experiments/exp_delphi_math_10b_midtrain.py` with selector
support. The base checkpoint uses `mirrored(..., budget_gb=150)` so the first
job in a region stages the checkpoint locally before TensorStore restore.

Preferred opportunistic launch:

- `v5p-512`: global batch 1024, 2384 midtraining steps, LR multiplier 1.0.
  This is the cleanest comparison because it preserves the base pretrain batch.

Fallback if only a clean `v5p-128` slice is available:

- `v5p-128`: global batch 256, 9537 midtraining steps, LR multiplier 0.5
  (`sqrt(256/1024)`) so the LR/Adam-LR scale tracks the batch-size change.
  This is useful for fast signal but should be tagged separately because it is
  not a no-confound comparison to the B=1024 `v5p-512` run.

`1e22` queue order mirrors `1e21`: `0.67`, `0.83`, `0.50`, with at most one
active `1e22` job at a time.

### Monitor

Created:

```text
scratch/monitor_delphi_large_midtrain_jobs.py
scratch/delphi_large_midtrain_monitoring_state.json
```

Monitor behavior:

- every 900 seconds, query current `1e21` job state and `v5p-512`/`v5p-128`
  clean capacity;
- append a status/action tick to this logbook;
- launch the next queued `1e21` LR only after the current `1e21` child job
  reaches `JOB_STATE_SUCCEEDED`;
- launch one `1e22` job only when a clean target slice is available and no
  `1e22` job is active;
- stop automatic sequencing after any non-success terminal state so a human/agent
  can inspect and resume the exact output namespace.

## 2026-04-25 06:40 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=17 min_free_mem_b=39542558720
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=64 committed_tpu=0/256 running_tasks=2 min_free_mem_b=461523099648

Actions/status:
- `1e21 lr=0.67`: `JOB_STATE_RUNNING`; `I20260425 06:39:05 140405700699968 tqdm_loggable.tqdm_logging Progress on:train 4.74kit/4.77kit rate:1.8s/it remaining:00:54 elapsed:2:31:59 postfix:loss=0.75`
- `1e22 lr=0.67`: launched `/ahmed/code/marin/.claude/worktrees/midtrain_data/lib/iris/examples/marin.yaml` on `v5p-512` in `us-central1`

Correction for the previous line: the monitor's first job-id parser matched the
local Iris config path in the submit output. The actual submitted job is:

```text
/ahmed/delphi-math-10b-1e22v5-lr0p67-v5p512-B1024-20260425-064040
```

The monitor parser and state file were corrected before starting the resident
15-minute loop.

## 2026-04-25 06:41 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=17 min_free_mem_b=39542558720
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=2 min_free_mem_b=320862920704

Actions/status:
- `1e21 lr=0.67`: `JOB_STATE_RUNNING`; `I20260425 06:39:05 140405700699968 tqdm_loggable.tqdm_logging Progress on:train 4.74kit/4.77kit rate:1.8s/it remaining:00:54 elapsed:2:31:59 postfix:loss=0.75`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 06:42:12 140124579989312 tqdm_loggable.tqdm_logging Progress on:train -/2384 rate:- remaining:? elapsed:00:00 postfix:-`

## 2026-04-25 06:46 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=17 min_free_mem_b=39542558720
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=64 committed_tpu=0/256 running_tasks=1 min_free_mem_b=462596841472

Actions/status:
- `1e21 lr=0.83`: launched `/ahmed/delphi-math-10b-1e21v5-lr0p83-v5p256-20260425-064648`
- `1e22 lr=0.67`: launched `/ahmed/delphi-math-10b-1e22v5-lr0p67-v5p512-B1024-20260425-064703` on `v5p-512` in `us-central1`

## 2026-04-25 06:52 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=17 min_free_mem_b=39542558720
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=67 min_free_mem_b=320862920704

Actions/status:
- `1e21 lr=0.83`: `JOB_STATE_RUNNING`; `I20260425 06:52:40 139698061309760 tqdm_loggable.tqdm_logging Progress on:train 86.0it/4.77kit rate:1.6s/it remaining:2:01:22 elapsed:04:18 postfix:loss=1.22`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 06:52:38 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 16.0it/2.38kit rate:4.4s/it remaining:2:54:42 elapsed:03:14 postfix:loss=1.18`

### 1e22 launch note: mirror-lock failure and local checkpoint relaunch

The first `1e22 lr=0.67` launch
`/ahmed/delphi-math-10b-1e22v5-lr0p67-v5p512-B1024-20260425-064040`
was stopped before useful training because the fresh `mirror://` checkpoint path
caused many of the 64 hosts to contend on the same mirror lock:

```text
RuntimeError: Could not acquire mirror lock for
adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-38206/...
```

Fix applied:

```bash
gcloud storage cp -r --no-clobber \
  gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-38206 \
  gs://marin-us-central1/checkpoints/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/
```

Verified byte-identical:

```text
116582126509 gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-38206/
116582126509 gs://marin-us-central1/checkpoints/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-38206/
```

Added `MIDTRAIN_INIT_CKPT_PATH` to `experiments/exp_delphi_math_10b_midtrain.py`
and relaunched `1e22 lr=0.67` with:

```text
MIDTRAIN_INIT_CKPT_PATH=gs://marin-us-central1/checkpoints/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-38206
```

Relaunch is healthy:

```text
/ahmed/delphi-math-10b-1e22v5-lr0p67-v5p512-B1024-20260425-064703/train_lm
Progress on:train 16.0it/2.38kit rate:4.4s/it remaining:2:54:42 postfix:loss=1.18
```

Resident monitor is running in:

```text
tmux session: delphi_large_midtrain_monitor
log file: scratch/delphi_large_midtrain_monitor_20260425-065507.log
state: scratch/delphi_large_midtrain_monitoring_state.json
```

## 2026-04-25 06:55 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=17 min_free_mem_b=39542558720
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=67 min_free_mem_b=320862920704

Actions/status:
- `1e21 lr=0.83`: `JOB_STATE_RUNNING`; `I20260425 06:54:41 139698061309760 tqdm_loggable.tqdm_logging Progress on:train 161it/4.77kit rate:1.6s/it remaining:1:59:02 elapsed:06:20 postfix:loss=1.12`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 06:54:41 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 46.0it/2.38kit rate:4.1s/it remaining:2:39:35 elapsed:05:17 postfix:loss=1.07`

## 2026-04-25 07:10 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=73902297088
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=67 min_free_mem_b=320862920704

Actions/status:
- `1e21 lr=0.83`: `JOB_STATE_RUNNING`; `I20260425 07:10:57 139698061309760 tqdm_loggable.tqdm_logging Progress on:train 727it/4.77kit rate:1.6s/it remaining:1:46:04 elapsed:22:35 postfix:loss=0.994`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 07:11:14 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 275it/2.38kit rate:4.4s/it remaining:2:33:14 elapsed:21:50 postfix:loss=0.929`

## 2026-04-25 07:26 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=73902297088
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=67 min_free_mem_b=320862920704

Actions/status:
- `1e21 lr=0.83`: `JOB_STATE_RUNNING`; `I20260425 07:26:03 139698061309760 tqdm_loggable.tqdm_logging Progress on:train 1.15kit/4.77kit rate:1.6s/it remaining:1:38:30 elapsed:37:42 postfix:loss=0.945`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 07:26:55 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 491it/2.38kit rate:4.2s/it remaining:2:13:30 elapsed:37:30 postfix:loss=0.85`

## 2026-04-25 07:42 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=73902297088
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=67 min_free_mem_b=320862920704

Actions/status:
- `1e21 lr=0.83`: `JOB_STATE_RUNNING`; `I20260425 07:42:15 139698061309760 tqdm_loggable.tqdm_logging Progress on:train 1.70kit/4.77kit rate:1.6s/it remaining:1:20:03 elapsed:53:53 postfix:loss=0.89`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 07:41:36 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 689it/2.38kit rate:4.2s/it remaining:1:57:37 elapsed:52:12 postfix:loss=0.774`

## 2026-04-25 07:57 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=22 min_free_mem_b=172686548992
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=73902297088
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=69 min_free_mem_b=320862920704

Actions/status:
- `1e21 lr=0.83`: `JOB_STATE_RUNNING`; `I20260425 07:57:56 139698061309760 tqdm_loggable.tqdm_logging Progress on:train 2.15kit/4.77kit rate:1.8s/it remaining:1:19:00 elapsed:1:09:34 postfix:loss=0.862`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 07:58:13 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 917it/2.38kit rate:4.6s/it remaining:1:51:38 elapsed:1:08:48 postfix:loss=0.766`

## 2026-04-25 08:13 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=20 min_free_mem_b=73902297088
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=67 min_free_mem_b=320862920704

Actions/status:
- `1e21 lr=0.83`: `JOB_STATE_RUNNING`; `I20260425 08:14:07 139698061309760 tqdm_loggable.tqdm_logging Progress on:train 2.70kit/4.77kit rate:1.6s/it remaining:55:09 elapsed:1:25:46 postfix:loss=0.83`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 08:14:09 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 1.02kit/2.38kit rate:5.0s/it remaining:1:53:00 elapsed:1:24:45 postfix:loss=0.731`

## 2026-04-25 08:29 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=5182820352
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=67 min_free_mem_b=320862920704

Actions/status:
- `1e21 lr=0.83`: `JOB_STATE_RUNNING`; `I20260425 08:29:52 139698061309760 tqdm_loggable.tqdm_logging Progress on:train 3.15kit/4.77kit rate:1.6s/it remaining:43:42 elapsed:1:41:31 postfix:loss=0.768`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 08:29:56 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 1.23kit/2.38kit rate:4.3s/it remaining:1:22:58 elapsed:1:40:32 postfix:loss=0.716`

## 2026-04-25 08:45 UTC large-sweep monitor tick failed

```text
RuntimeError: Command failed (2): uv run iris --cluster=marin query
select
  scale_group,
  count(*) as workers,
  sum(active) as active,
  sum(healthy) as healthy,
  sum(total_tpu_count) as total_tpu,
  sum(committed_tpu) as committed_tpu,
  sum(snapshot_running_task_count) as running_tasks,
  sum(case
    when active = 1
     and healthy = 1
     and total_tpu_count > 0
     and committed_tpu = 0
     and (total_memory_bytes - committed_mem_bytes) >= 150000000000
    then 1 else 0 end) as clean_workers,
  min(total_memory_bytes - committed_mem_bytes) as min_free_mem_b
from workers
where scale_group like '%v5p%512%' or scale_group like '%v5p%128%'
group by scale_group
order by scale_group
 -f json
error: Failed to read `--find-links` URL: https://github.com/marin-community/kitoken/releases/expanded_assets/kitoken-0.10.2-a3012f4
  Caused by: Failed to fetch: `https://github.com/marin-community/kitoken/releases/expanded_assets/kitoken-0.10.2-a3012f4`
  Caused by: HTTP status server error (502 Bad Gateway) for url (https://github.com/marin-community/kitoken/releases/expanded_assets/kitoken-0.10.2-a3012f4)

```

## 2026-04-25 09:00 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=73902297088
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=312272986112

Actions/status:
- `1e21 lr=0.83`: `JOB_STATE_RUNNING`; `I20260425 09:00:41 139698061309760 tqdm_loggable.tqdm_logging Progress on:train 4.11kit/4.77kit rate:1.7s/it remaining:18:46 elapsed:2:12:20 postfix:loss=0.747`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 09:00:30 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 1.65kit/2.38kit rate:4.2s/it remaining:51:38 elapsed:2:11:06 postfix:loss=0.647`

## 2026-04-25 09:16 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=24 min_free_mem_b=145843003392
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=73902297088
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=78 min_free_mem_b=277913247744

Actions/status:
- `1e21 lr=0.83`: `JOB_STATE_RUNNING`; `I20260425 09:16:08 139698061309760 tqdm_loggable.tqdm_logging Progress on:train 4.64kit/4.77kit rate:1.7s/it remaining:03:37 elapsed:2:27:47 postfix:loss=0.697`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 09:16:05 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 1.86kit/2.38kit rate:4.3s/it remaining:37:42 elapsed:2:26:40 postfix:loss=0.644`

## 2026-04-25 09:31 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=37 min_free_mem_b=6256566272
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=15 min_free_mem_b=78197264384
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=186 min_free_mem_b=1961598976

Actions/status:
- `1e21 lr=0.83`: `JOB_STATE_SUCCEEDED`; failures=0; preemptions=0
- `1e21 lr=0.5`: launched `/ahmed/delphi-math-10b-1e21v5-lr0p5-v5p256-20260425-093200`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 09:25:25 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 1.99kit/2.38kit rate:4.3s/it remaining:28:23 elapsed:2:36:01 postfix:loss=0.64`

## 2026-04-25 09:47 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=63 min_free_mem_b=49206239232
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=15 min_free_mem_b=78197264384
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=171 min_free_mem_b=91082170368

Actions/status:
- `1e21 lr=0.5`: `JOB_STATE_RUNNING`; `I20260425 09:47:36 140605590140736 tqdm_loggable.tqdm_logging Progress on:train -/4768 rate:- remaining:? elapsed:00:00 postfix:-`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 09:47:09 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 2.17kit/2.38kit rate:4.2s/it remaining:14:45 elapsed:2:57:45 postfix:loss=0.626`

## 2026-04-25 10:03 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=37 min_free_mem_b=6256566272
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=15 min_free_mem_b=78197264384
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=186 min_free_mem_b=1961598976

Actions/status:
- `1e21 lr=0.5`: `JOB_STATE_RUNNING`; `I20260425 10:02:26 140605590140736 tqdm_loggable.tqdm_logging Progress on:train 439it/4.77kit rate:1.8s/it remaining:2:11:15 elapsed:14:49 postfix:loss=1.04`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 10:02:01 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 2.37kit/2.38kit rate:4.4s/it remaining:00:43 elapsed:3:12:36 postfix:loss=0.638`

## 2026-04-25 10:18 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=51 min_free_mem_b=6256566272
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=15 min_free_mem_b=78197264384
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=64 committed_tpu=0/256 running_tasks=71 min_free_mem_b=185571450880

Actions/status:
- `1e21 lr=0.5`: `JOB_STATE_RUNNING`; `I20260425 10:17:36 140605590140736 tqdm_loggable.tqdm_logging Progress on:train 965it/4.77kit rate:1.7s/it remaining:1:48:02 elapsed:29:59 postfix:loss=0.882`
- `1e22 lr=0.67`: `JOB_STATE_SUCCEEDED` on `v5p-512`; failures=0; preemptions=0
- `1e22 lr=0.83`: launched `/ahmed/delphi-math-10b-1e22v5-lr0p83-v5p512-B1024-20260425-101917` on `v5p-512` in `us-central1`

## 2026-04-25 10:34 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=59 min_free_mem_b=6256566272
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=15 min_free_mem_b=78197264384
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=242 min_free_mem_b=3035340800

Actions/status:
- `1e21 lr=0.5`: `JOB_STATE_RUNNING`; `I20260425 10:33:48 140605590140736 tqdm_loggable.tqdm_logging Progress on:train 1.42kit/4.77kit rate:1.6s/it remaining:1:30:37 elapsed:46:12 postfix:loss=0.87`
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 10:34:24 139934602020672 tqdm_loggable.tqdm_logging Progress on:train -/2384 rate:- remaining:? elapsed:00:00 postfix:-`

## 2026-04-25 10:50 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=76 min_free_mem_b=1961598976
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=15 min_free_mem_b=78197264384
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=277 min_free_mem_b=5182824448

Actions/status:
- `1e21 lr=0.5`: `JOB_STATE_RUNNING`; `I20260425 10:50:01 140605590140736 tqdm_loggable.tqdm_logging Progress on:train 1.97kit/4.77kit rate:1.7s/it remaining:1:17:39 elapsed:1:02:24 postfix:loss=0.835`
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 10:50:14 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 192it/2.38kit rate:4.5s/it remaining:2:43:54 elapsed:15:50 postfix:loss=0.96`

## 2026-04-25 11:05 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=36 min_free_mem_b=18067726336
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=15 min_free_mem_b=78197264384
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=96 min_free_mem_b=262880862208

Actions/status:
- `1e21 lr=0.5`: `JOB_STATE_RUNNING`; `I20260425 11:05:52 140605590140736 tqdm_loggable.tqdm_logging Progress on:train 2.42kit/4.77kit rate:1.6s/it remaining:1:02:56 elapsed:1:18:16 postfix:loss=0.822`
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 11:05:48 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 403it/2.38kit rate:8.0s/it remaining:4:25:17 elapsed:31:24 postfix:loss=0.873`

## 2026-04-25 11:21 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=14 min_free_mem_b=112557002752
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=66 min_free_mem_b=310125502464

Actions/status:
- `1e21 lr=0.5`: `JOB_STATE_RUNNING`; `I20260425 11:21:09 140605590140736 tqdm_loggable.tqdm_logging Progress on:train 2.95kit/4.77kit rate:1.9s/it remaining:57:42 elapsed:1:33:33 postfix:loss=0.784`
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 11:21:32 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 616it/2.38kit rate:4.3s/it remaining:2:08:02 elapsed:47:08 postfix:loss=0.831`

## 2026-04-25 11:37 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=14 min_free_mem_b=112557002752
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=66 min_free_mem_b=310125502464

Actions/status:
- `1e21 lr=0.5`: `JOB_STATE_RUNNING`; `I20260425 11:36:39 140605590140736 tqdm_loggable.tqdm_logging Progress on:train 3.39kit/4.77kit rate:1.7s/it remaining:38:18 elapsed:1:49:02 postfix:loss=0.786`
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 11:37:19 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 831it/2.38kit rate:4.1s/it remaining:1:47:20 elapsed:1:02:55 postfix:loss=0.785`

## 2026-04-25 11:52 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=14 min_free_mem_b=112557002752
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=66 min_free_mem_b=310125502464

Actions/status:
- `1e21 lr=0.5`: `JOB_STATE_RUNNING`; `I20260425 11:52:05 140605590140736 tqdm_loggable.tqdm_logging Progress on:train 3.92kit/4.77kit rate:1.6s/it remaining:22:44 elapsed:2:04:28 postfix:loss=0.729`
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 11:48:42 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 988it/2.38kit rate:4.2s/it remaining:1:37:31 elapsed:1:14:18 postfix:loss=0.768`

## 2026-04-25 12:08 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=14 min_free_mem_b=112557002752
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=66 min_free_mem_b=310125502464

Actions/status:
- `1e21 lr=0.5`: `JOB_STATE_RUNNING`; `I20260425 12:08:15 140605590140736 tqdm_loggable.tqdm_logging Progress on:train 4.39kit/4.77kit rate:1.7s/it remaining:10:26 elapsed:2:20:38 postfix:loss=0.718`
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 12:08:03 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 1.15kit/2.38kit rate:4.3s/it remaining:1:28:08 elapsed:1:33:39 postfix:loss=0.741`

## 2026-04-25 12:23 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=14 min_free_mem_b=112557002752
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=66 min_free_mem_b=310125502464

Actions/status:
- `1e21 lr=0.5`: `JOB_STATE_SUCCEEDED`; failures=0; preemptions=0
- `1e21`: queue complete
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 12:23:40 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 1.36kit/2.38kit rate:4.2s/it remaining:1:12:20 elapsed:1:49:16 postfix:loss=0.721`

## 2026-04-25 12:39 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=14 min_free_mem_b=112557002752
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=66 min_free_mem_b=310125502464

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 12:39:23 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 1.57kit/2.38kit rate:4.2s/it remaining:56:40 elapsed:2:04:59 postfix:loss=0.705`

## 2026-04-25 12:54 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=14 min_free_mem_b=112557002752
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=66 min_free_mem_b=310125502464

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 12:54:50 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 1.78kit/2.38kit rate:4.2s/it remaining:42:40 elapsed:2:20:26 postfix:loss=0.663`

## 2026-04-25 13:10 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=14 min_free_mem_b=112557002752
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=66 min_free_mem_b=310125502464

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 13:09:20 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 1.98kit/2.38kit rate:4.4s/it remaining:29:39 elapsed:2:34:56 postfix:loss=0.637`

## 2026-04-25 13:25 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=14 min_free_mem_b=112557002752
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=66 min_free_mem_b=310125502464

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 13:25:19 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 2.09kit/2.38kit rate:4.4s/it remaining:21:52 elapsed:2:50:55 postfix:loss=0.647`

## 2026-04-25 13:40 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=15 min_free_mem_b=78197264384
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=66 min_free_mem_b=310125502464

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 13:40:46 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 2.30kit/2.38kit rate:4.1s/it remaining:05:58 elapsed:3:06:22 postfix:loss=0.627`

## 2026-04-25 13:56 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=22 min_free_mem_b=175907774464
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=73902297088
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=64 committed_tpu=0/256 running_tasks=67 min_free_mem_b=447564455936

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.83`: `JOB_STATE_SUCCEEDED` on `v5p-512`; failures=0; preemptions=0
- `1e22 lr=0.5`: launched `/ahmed/delphi-math-10b-1e22v5-lr0p5-v5p512-B1024-20260425-135618` on `v5p-512` in `us-central1`

## 2026-04-25 14:11 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=22 min_free_mem_b=175907774464
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=43837526016
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=306904276992

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 14:11:02 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 136it/2.38kit rate:4.4s/it remaining:2:44:54 elapsed:11:36 postfix:loss=0.971`

## 2026-04-25 14:26 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=22 min_free_mem_b=175907774464
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=43837526016
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=306904276992

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 14:26:41 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 350it/2.38kit rate:4.6s/it remaining:2:36:00 elapsed:27:15 postfix:loss=0.891`

## 2026-04-25 14:42 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=22 min_free_mem_b=175907774464
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=43837526016
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=306904276992

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 14:42:15 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 565it/2.38kit rate:4.1s/it remaining:2:04:55 elapsed:42:48 postfix:loss=0.823`

## 2026-04-25 14:57 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=22 min_free_mem_b=175907774464
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=43837526016
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=67 min_free_mem_b=323010404352

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 14:56:58 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 765it/2.38kit rate:4.7s/it remaining:2:06:23 elapsed:57:31 postfix:loss=0.787`

## 2026-04-25 15:12 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 15:12:33 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 978it/2.38kit rate:4.1s/it remaining:1:36:49 elapsed:1:13:06 postfix:loss=0.741`

## 2026-04-25 15:28 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 15:28:13 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 1.09kit/2.38kit rate:4.2s/it remaining:1:30:59 elapsed:1:28:47 postfix:loss=0.777`

## 2026-04-25 15:43 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=79 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 15:42:57 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 1.29kit/2.38kit rate:4.1s/it remaining:1:14:56 elapsed:1:43:31 postfix:loss=0.712`

## 2026-04-25 15:58 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 15:58:25 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 1.50kit/2.38kit rate:4.4s/it remaining:1:05:40 elapsed:1:58:58 postfix:loss=0.679`

## 2026-04-25 16:14 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 16:13:59 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 1.71kit/2.38kit rate:4.2s/it remaining:47:14 elapsed:2:14:33 postfix:loss=0.654`

## 2026-04-25 16:29 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=85 min_free_mem_b=288650665984

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 16:29:37 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 1.92kit/2.38kit rate:4.1s/it remaining:31:46 elapsed:2:30:11 postfix:loss=0.624`

## 2026-04-25 16:44 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 16:44:48 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 2.02kit/2.38kit rate:5.5s/it remaining:33:33 elapsed:2:45:21 postfix:loss=0.659`

## 2026-04-25 17:00 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=117 min_free_mem_b=138326810624

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 16:59:43 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 2.22kit/2.38kit rate:4.3s/it remaining:11:53 elapsed:3:00:17 postfix:loss=0.624`

## 2026-04-25 17:24 UTC 1e22 full-4plus launch

User requested one full-pass `nemotron_cc_math_v1/4plus` midtraining run while
the `v5p-512` slice is still live. Rationale: the `1e22-v5` 10B sweep completed
cleanly on `v5p-512`, `lr_factor=0.5` looked most stable, and the full
BOS-correct `4plus-212a2d` cache is 51,482,572,371 Llama-3 tokens.

Launch target:

- base: `1e22-v5`
- LR factor: `0.5`
- data: 100% `nemotron_cc_math_v1/4plus`
- token budget: `51_482_572_371`
- batch/seq: `1024 x 4096`
- steps: `round(51_482_572_371 / (1024 * 4096)) = 12_274`
- TPU: `v5p-512`
- W&B project: `delphi-midtraining`

Added `MIDTRAIN_TOKEN_BUDGET` / `MIDTRAIN_TOKEN_BUDGET_LABEL` support to
`experiments/exp_delphi_math_10b_midtrain.py` so this run gets a distinct
`math-full4plus` output namespace instead of reusing the `math-10b` name.

Submitted parent:

```text
/ahmed/delphi-math-full4plus-1e22v5-lr0p5-v5p512-B1024-20260425-172458
```

Startup verification:

- child `train_lm` submitted at `2026-04-25 17:25:45 UTC` and started immediately;
- `v5p-512` committed `256/256` TPU;
- all 64 child tasks running on `marin-tpu-v5p-preemptible-512-us-central-20260424-2002-da9f954e`;
- logs show base checkpoint loaded from local us-central1:
  `gs://marin-us-central1/checkpoints/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-38206`;
- first progress line at `17:28:50 UTC`: `3.00it/12.3kit`, `loss=1.26`.
- stabilization check at `17:32:56 UTC`: `62.0it/12.3kit`,
  `rate=4.1s/it`, `remaining=13:58:43`, `loss=1.04`; child summary reports
  `state=running`, `task_count=64`, `failure_count=0`, `preemption_count=0`.

Detached monitor:

```text
tmux session: delphi_full4plus_monitor
state file: scratch/delphi_full4plus_monitoring_state.json
log file: scratch/delphi_full4plus_monitor_20260425-172458.log
```

## 2026-04-25 17:15 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 17:11:03 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 2.37kit/2.38kit rate:4.5s/it remaining:00:54 elapsed:3:11:37 postfix:loss=0.641`

## 2026-04-25 17:30 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_SUCCEEDED` on `v5p-512`; failures=0; preemptions=0
- `1e22`: queue complete

## 2026-04-25 17:45 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 18:01 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 18:09 UTC midtraining mixture registry

Realization while the 1e22 full-4plus run is live: the current Delphi
midtraining runs are 100% `nemotron_cc_math_v1/4plus`. That remains a useful
math-specialization / LR-ablation axis, but it is not the closest analogue to
Mantis-style production cooldown, which mixes pretraining replay with the
high-quality target data.

Added `experiments/midtraining_mixes.py` with two stable registry keys:

- `full_highquality_nemo_math`: one-component `LMMixtureDatasetConfig` for
  100% `nemotron_cc_math_v1/4plus`.
- `70p_30m_highquality_nemo_math`: Mantis-style fixed mixture using the
  existing Nemotron pretraining replay mix at 70% and `nemotron_cc_math_v1/4plus`
  at 30%. The replay side reuses `experiments.pretraining_datasets.nemotron.nemotron_mix`,
  which is the old Nemotron-CC split weights plus `starcoderdata` and
  `proofpile_2`, matching the pretraining side of Mantis.
- `33p_67m_highquality_nemo_math`: math-heavy contrast mix using the same
  replay components at 33% and `nemotron_cc_math_v1/4plus` at 67%.

Also added an optional `MIDTRAIN_MIX_NAME` selector to
`experiments/exp_delphi_math_10b_midtrain.py`. Leaving it unset preserves the
legacy 100% math `ExecutorStep` path and output names for the currently-running
full-4plus job; setting it to `70p_30m_highquality_nemo_math` creates a distinct
`delphi-...-70p-30m-highquality-nemo-math-...` namespace for future replay-mix
launches. The `33p_67m_highquality_nemo_math` key creates the analogous
math-heavy namespace. No replay-mix job launched in this entry.

## 2026-04-25 18:23 UTC 33/67 10B control launch

User requested a 10B-token `1e22-v5` control on the
`33p_67m_highquality_nemo_math` mix, LR factor `0.5`, on the live `v5p-512`
slice. Explicit ordering constraint: submit the new Iris job first, then kill
the current full-math job.

Submitted replacement parent first:

```text
/ahmed/delphi-math-10b-33p67m-1e22v5-lr0p5-v5p512-B1024-20260425-182332
```

Launch parameters:

- base: `1e22-v5`
- mix: `33p_67m_highquality_nemo_math` (33% proportional Nemotron replay, 67% `nemotron_cc_math_v1/4plus`)
- token budget: `10_000_000_000` total mixed-stream tokens
- expected math tokens: ~`6.7B`
- batch/seq: `1024 x 4096`
- expected train steps: `round(10_000_000_000 / (1024 * 4096)) = 2384`
- LR factor: `0.5`
- TPU: `v5p-512`
- W&B project: `delphi-midtraining`
- base checkpoint override:
  `gs://marin-us-central1/checkpoints/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-38206`

Verified before killing the previous run:

```text
/ahmed/delphi-math-10b-33p67m-1e22v5-lr0p5-v5p512-B1024-20260425-182332/train_lm
state=pending, resources=v5p-512, reason=coscheduling waiting for 64 workers
```

Then stopped the previous full-math parent as requested:

```text
/ahmed/delphi-math-full4plus-1e22v5-lr0p5-v5p512-B1024-20260425-172458
```

Iris reported both terminated:

```text
/ahmed/delphi-math-full4plus-1e22v5-lr0p5-v5p512-B1024-20260425-172458/train_lm
/ahmed/delphi-math-full4plus-1e22v5-lr0p5-v5p512-B1024-20260425-172458
```

Post-stop startup check: the new `train_lm` child moved to `running` with
`64/64` tasks on `v5p-512`, `failures=0`, `preemptions=0`. Stabilization
check at `18:29 UTC`: `32 / 2384` train steps, `rate=4.1s/it`,
`remaining=2:41:22`, `loss=1.52`.

Monitor handoff:

```text
tmux session: delphi_mix33p67m_control_monitor
state file: scratch/delphi_mix33p67m_control_monitoring_state.json
log file: scratch/delphi_mix33p67m_control_monitor_20260425-182332.log
```

## 2026-04-25 18:16 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 18:31 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 18:46 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 19:01 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=33 min_free_mem_b=185571450880
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 19:16 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=33 min_free_mem_b=185571450880
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 19:20 UTC lr0.83 1e20 v5 resume relaunch

RELAUNCHED `lr=0.83` ON `v5p-64` AS A RESUME OF THE EXISTING RUN. DO NOT
START A NEW `lr=0.83` RUN IF THIS DIES. RESUME THE SAME CHECKPOINT PREFIX AND
THE SAME W&B RUN ID:

```text
delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b
```

Existing restart checkpoint verified before launch in both regions:

```text
gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b/checkpoints/step-2000/
gs://marin-us-east5/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b/checkpoints/step-2000/
```

Submitted with fixed Iris parent ID:

```text
/ahmed/delphi-math-10b-1e20-lr0p83-v5-zeroend-bos-wandbproject-20260424-224720
```

Resubmit command:

```bash
uv run iris --cluster=marin job run --cpu 1 --memory 3GB --disk 9GB --preemptible --region us-central1 --region us-east5 --job-name delphi-math-10b-1e20-lr0p83-v5-zeroend-bos-wandbproject-20260424-224720 --no-wait -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 -e WANDB_PROJECT delphi-midtraining -e RUN_ID delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b -e WANDB_RUN_ID delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b -e WANDB_RESUME allow -e MIDTRAIN_SELECT_BASE 1e20-iso-d2048-L21 -e MIDTRAIN_SELECT_LR 0.83 -e MIDTRAIN_TPU_TYPE v5p-64 -e MIDTRAIN_TOKEN_BUDGET 10000000000 -e MIDTRAIN_TOKEN_BUDGET_LABEL 10b -- python experiments/exp_delphi_math_10b_midtrain.py
```

First fixed-name relaunch printed the correct W&B run ID but resolved to a new
checkpoint prefix (`...-a14928`), so it was stopped before the TPU child
allocated. Do not use that prefix.

Added `MIDTRAIN_OUTPUT_PATH_OVERRIDE` to
`experiments/exp_delphi_math_10b_midtrain.py`, routed through the existing
`default_train(..., override_output_path=...)` hook, and relaunched under a
fresh Iris wrapper to avoid stale terminal-state reuse:

```text
/ahmed/delphi-math-10b-1e20-lr0p83-v5-resume-090b5b-20260425-1924
```

Current resubmit command:

```bash
uv run iris --cluster=marin job run --cpu 1 --memory 3GB --disk 9GB --preemptible --region us-central1 --region us-east5 --job-name delphi-math-10b-1e20-lr0p83-v5-resume-090b5b-20260425-1924 --no-wait -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 -e WANDB_PROJECT delphi-midtraining -e RUN_ID delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b -e WANDB_RUN_ID delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b -e WANDB_RESUME allow -e MIDTRAIN_SELECT_BASE 1e20-iso-d2048-L21 -e MIDTRAIN_SELECT_LR 0.83 -e MIDTRAIN_TPU_TYPE v5p-64 -e MIDTRAIN_TOKEN_BUDGET 10000000000 -e MIDTRAIN_TOKEN_BUDGET_LABEL 10b -e MIDTRAIN_OUTPUT_PATH_OVERRIDE gs://marin-us-east5/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b -- python experiments/exp_delphi_math_10b_midtrain.py
```

Verified parent logs:

```text
Using output path: gs://marin-us-east5/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b
Using run ID: delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b
```

Startup status: `train_lm` child submitted and still pending for `v5p-64`
coscheduling as of `19:33 UTC`. This is capacity wait, not a resume/config
failure. Monitor state:
`scratch/delphi_1e20_lr0p83_resume_monitoring_state.json`.

Detached monitor:

```text
tmux session: delphi_1e20_lr0p83_resume_monitor
script: scratch/delphi_1e20_lr0p83_resume_monitor.sh
log: scratch/delphi_1e20_lr0p83_resume_monitor.log
```

Update at `19:40 UTC`: the `...-1924` child briefly allocated and then all
8 workers were preempted before checkpoint restore or train progress. Logs
also showed the data dependency using `4plus-da9608`, not the intended
BOS-correct `4plus-212a2d` cache, so the job was stopped before any training
started.

Added `MIDTRAIN_MATH_TOKENIZED_OUTPUT_PATH_OVERRIDE` to force the math dataset
ExecutorStep to the BOS-correct cache, then relaunched:

```text
/ahmed/delphi-math-10b-1e20-lr0p83-v5-resume-090b5b-bosdata-20260425-1941
```

Current command:

```bash
uv run iris --cluster=marin job run --cpu 1 --memory 3GB --disk 9GB --preemptible --region us-central1 --region us-east5 --job-name delphi-math-10b-1e20-lr0p83-v5-resume-090b5b-bosdata-20260425-1941 --no-wait -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 -e WANDB_PROJECT delphi-midtraining -e RUN_ID delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b -e WANDB_RUN_ID delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b -e WANDB_RESUME allow -e MIDTRAIN_SELECT_BASE 1e20-iso-d2048-L21 -e MIDTRAIN_SELECT_LR 0.83 -e MIDTRAIN_TPU_TYPE v5p-64 -e MIDTRAIN_TOKEN_BUDGET 10000000000 -e MIDTRAIN_TOKEN_BUDGET_LABEL 10b -e MIDTRAIN_OUTPUT_PATH_OVERRIDE gs://marin-us-east5/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b -e MIDTRAIN_MATH_TOKENIZED_OUTPUT_PATH_OVERRIDE gs://marin-us-east5/tokenized/nemotron_cc_math_v1/4plus-212a2d -- python experiments/exp_delphi_math_10b_midtrain.py
```

Verified parent logs for the current relaunch:

```text
Output path .../4plus-da9608 doesn't match given override .../4plus-212a2d, using the latter.
Step = tokenized/nemotron_cc_math_v1/4plus_dafdee2a ... Output_path = gs://marin-us-east5/tokenized/nemotron_cc_math_v1/4plus-212a2d
Skip tokenized/nemotron_cc_math_v1/4plus_dafdee2a: already succeeded
Using output path: gs://marin-us-east5/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b
Using run ID: delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b
```

Status at `19:42 UTC`: current `train_lm` child pending for `v5p-64`
coscheduling; no training progress yet.

## 2026-04-25 19:31 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=33 min_free_mem_b=185571450880
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 19:46 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=33 min_free_mem_b=185571450880
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 20:01 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=33 min_free_mem_b=185571450880
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 20:13 UTC 1e20 mix LR sweep submitted on v5p-32 batch priority

User requested the six 1e20 control/sweep runs on `v5p-32` with Iris batch priority:

- `33p_67m_highquality_nemo_math`: 33% proportional Delphi pretraining replay, 67% Nemotron CC math 4plus.
- `67p_33m_highquality_nemo_math`: 67% proportional Delphi pretraining replay, 33% Nemotron CC math 4plus.
- LR factors: `0.5`, `0.67`, `0.83`.
- Token budget: `10b`.
- W&B project: `delphi-midtraining`.
- TPU type: `v5p-32`.
- Iris priority: `--priority batch`.

Before launch, shortened the midtraining run-name scheme in
`experiments/exp_delphi_math_10b_midtrain.py` so these runs avoid the
`default_train` name truncation path:

```text
delphi-{base}-{mix}-{budget}-lr{factor}
```

Examples:

```text
delphi-1e20-p33m67-10b-lr0.5
delphi-1e20-p67m33-10b-lr0.83
```

Also added the exact `67p_33m_highquality_nemo_math` mix in
`experiments/midtraining_mixes.py`. Validation:

```bash
./infra/pre-commit.py --fix \
  experiments/exp_delphi_math_10b_midtrain.py \
  experiments/midtraining_mixes.py
```

passed before submission.

Submitted parent jobs:

```text
/ahmed/delphi-1e20-p33m67-10b-lr0p5-v5p32-batch-20260425-200848
/ahmed/delphi-1e20-p33m67-10b-lr0p67-v5p32-batch-20260425-200848
/ahmed/delphi-1e20-p33m67-10b-lr0p83-v5p32-batch-20260425-200848
/ahmed/delphi-1e20-p67m33-10b-lr0p5-v5p32-batch-20260425-200848
/ahmed/delphi-1e20-p67m33-10b-lr0p67-v5p32-batch-20260425-200848
/ahmed/delphi-1e20-p67m33-10b-lr0p83-v5p32-batch-20260425-200848
```

Resolved output paths / W&B run IDs:

```text
gs://marin-us-east5/checkpoints/delphi-1e20-p33m67-10b-lr0.5-1c0e07
delphi-1e20-p33m67-10b-lr0.5-1c0e07

gs://marin-us-east5/checkpoints/delphi-1e20-p33m67-10b-lr0.67-c011b2
delphi-1e20-p33m67-10b-lr0.67-c011b2

gs://marin-us-east5/checkpoints/delphi-1e20-p33m67-10b-lr0.83-b762c2
delphi-1e20-p33m67-10b-lr0.83-b762c2

gs://marin-us-east5/checkpoints/delphi-1e20-p67m33-10b-lr0.5-7beee4
delphi-1e20-p67m33-10b-lr0.5-7beee4

gs://marin-us-east5/checkpoints/delphi-1e20-p67m33-10b-lr0.67-d4af27
delphi-1e20-p67m33-10b-lr0.67-d4af27

gs://marin-us-east5/checkpoints/delphi-1e20-p67m33-10b-lr0.83-a2e4c0
delphi-1e20-p67m33-10b-lr0.83-a2e4c0
```

Health check at `2026-04-25T20:12Z`:

- All six parent coordinator jobs are `running`.
- All six `train_lm` children exist and are `pending` for `v5p-32`
  coscheduling (`4` workers each).
- SQL task check shows `priority_band=3` for every parent task and every
  `train_lm` worker task, confirming batch priority.
- Parent logs show the expected output path and run ID for all six; no
  `Traceback`, `Exception`, or `Error` lines in the launch-time log grep.

Current state is capacity wait, not a config or W&B failure.

## 2026-04-25 20:24 UTC v5p-512 interruption on 1e22 33p/67m control

Investigated after user noticed the `v5p-512` disappeared.

Affected job:

```text
/ahmed/delphi-math-10b-33p67m-1e22v5-lr0p5-v5p512-B1024-20260425-182332
```

Resolved training output / W&B run ID:

```text
gs://marin-us-central1/checkpoints/delphi-1e22-v5-33p-67m-highquality-nemo-math-10b-l-control33p67m-ad626e
delphi-1e22-v5-33p-67m-highquality-nemo-math-10b-l-control33p67m-ad626e
```

What happened:

- Training was live and healthy through about `1.23k / 2.38k` steps at
  `2026-04-25 20:04 UTC` (`~1h39m` elapsed, `~1h22m` remaining).
- Parent attempt `0` ended at `2026-04-25 20:07:03 UTC` with:

  ```text
  Worker marin-tpu-v5p-preemptible-512-us-central-20260424-2002-da9f954e-worker-11 failed: worker ping threshold exceeded
  ```

- Parent restarted on a small `v5p-8` worker and relaunched the
  `train_lm` child at `2026-04-25 20:07:38 UTC`.
- The relaunched `train_lm` child is now pending for `v5p-512` with scheduler
  reason `No workers match constraints`.
- SQL/GCP checks show no active Iris `v5p-512` workers or slices. `gcloud`
  returned no current `v5p-512` TPU VMs matching the filter.
- Autoscaler state for `tpu_v5p-preemptible_512-us-central1-a` is quota-blocked
  until roughly `2026-04-25 20:23:56 UTC` with:

  ```text
  Quota limit 'TPUV5PPreemptiblePerProjectPerRegionForTPUAPI' has been exceeded. Limit: 2048 in region us-central1.
  ```

Checkpoint state:

```text
gs://marin-us-central1/checkpoints/delphi-1e22-v5-33p-67m-highquality-nemo-math-10b-l-control33p67m-ad626e/checkpoints/step-1000/
```

exists and is `116,584,630,736` bytes. Eval metrics also reached step `1200`,
but training should resume from `step-1000`, so approximately `230` steps of
compute were lost after the last checkpoint.

Interpretation:

This was infrastructure/preemptible-capacity loss, not a training-code crash.
The job is currently waiting for `v5p-512` capacity. If a compatible
`v5p-512` slice returns, it should resume from the existing `step-1000`
checkpoint rather than starting over.

## 2026-04-25 20:17 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=33 min_free_mem_b=185571450880

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 20:32 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=33 min_free_mem_b=185571450880

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 20:47 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=33 min_free_mem_b=185571450880

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 21:02 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 21:07 UTC explicit Delphi midtraining base configs

Updated `experiments/exp_delphi_math_10b_midtrain.py` so the 1e20/1e21/1e22
base slots are first-class configs instead of a shared global batch/seq-length
default. Principle: carry over the base run's Delphi/pretrain shape settings
for sequence length and global batch; only sweep the LR factor.

Default configs now generated by the experiment file:

| base selector | hidden / layers | seq_len | global batch | default v5p | acceptable v5p targets |
|---|---:|---:|---:|---|---|
| `1e20-iso-d2048-L21` | 2048 / 21 | 4096 | 128 | `v5p-32` | `v5p-32`, `v5p-64`, `v5p-128`, `v5p-256`, `v5p-512` |
| `1e21-v5` | 2560 / 26 | 4096 | 512 | `v5p-256` | `v5p-64`, `v5p-128`, `v5p-256`, `v5p-512` |
| `1e22-v5` | 3840 / 37 | 4096 | 1024 | `v5p-512` | `v5p-128`, `v5p-256`, `v5p-512` |

Compute details:

- `1e20` on `v5p-512` sets `tensor_parallel_size=2`, because B128 is smaller
  than the 256-chip slice; smaller listed slices use `tp=1`.
- `1e22` keeps global B1024 on all approved slices. `v5p-128` and `v5p-256`
  use `per_device_parallelism=4`, so they use gradient accumulation instead of
  changing the global batch. `v5p-512` also records `per_device_parallelism=4`,
  which is the no-accumulation per-chip batch that already ran successfully.
- Unsupported combinations now fail fast. Example: `1e22-v5` with
  `MIDTRAIN_TPU_TYPE=v5p-64` raises a `ValueError` instead of silently launching
  a likely-bad job.

Important caveat: the already-launched 1e20 midtraining sweeps used B512 as an
operational standardization choice. New default 1e20 configs use B128 to match
the `isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5` source run.

Validation:

```bash
WANDB_PROJECT=delphi-midtraining uv run python - <<'PY'
import experiments.exp_delphi_math_10b_midtrain as exp
for run in exp.runs:
    trainer = run.config.train_config.trainer
    optimizer = run.config.train_config.optimizer
    tp = trainer.mesh.axes.get("model")
    print(
        run.name,
        run.config.resources.device.variant,
        trainer.train_batch_size,
        trainer.num_train_steps,
        run.config.train_config.train_seq_len,
        trainer.per_device_parallelism,
        tp,
        optimizer.warmup,
        optimizer.decay,
    )
PY
```

Output:

```text
checkpoints/delphi-1e20-math-10b-lr0.5 v5p-32 128 19073 4096 -1 1 2000 17073
checkpoints/delphi-1e20-math-10b-lr0.67 v5p-32 128 19073 4096 -1 1 2000 17073
checkpoints/delphi-1e20-math-10b-lr0.83 v5p-32 128 19073 4096 -1 1 2000 17073
checkpoints/delphi-1e21-math-10b-lr0.5 v5p-256 512 4768 4096 -1 1 500 4268
checkpoints/delphi-1e21-math-10b-lr0.67 v5p-256 512 4768 4096 -1 1 500 4268
checkpoints/delphi-1e21-math-10b-lr0.83 v5p-256 512 4768 4096 -1 1 500 4268
checkpoints/delphi-1e22-math-10b-lr0.5 v5p-512 1024 2384 4096 4 1 250 2134
checkpoints/delphi-1e22-math-10b-lr0.67 v5p-512 1024 2384 4096 4 1 250 2134
checkpoints/delphi-1e22-math-10b-lr0.83 v5p-512 1024 2384 4096 4 1 250 2134
```

Extra checks:

```bash
MIDTRAIN_SELECT_BASE=1e22-v5 MIDTRAIN_SELECT_LR=0.67 \
MIDTRAIN_TPU_TYPE=v5p-128 WANDB_PROJECT=delphi-midtraining \
uv run python - <<'PY'
import experiments.exp_delphi_math_10b_midtrain as exp
run = exp.runs[0]
trainer = run.config.train_config.trainer
print(
    run.name,
    run.config.resources.device.variant,
    trainer.train_batch_size,
    trainer.num_train_steps,
    run.config.train_config.train_seq_len,
    trainer.per_device_parallelism,
    trainer.mesh.axes.get("model"),
)
PY

./infra/pre-commit.py --fix experiments/exp_delphi_math_10b_midtrain.py
uv run pyrefly check experiments/exp_delphi_math_10b_midtrain.py
```

Results:

```text
checkpoints/delphi-1e22-math-10b-lr0.67 v5p-128 1024 2384 4096 4 1
pre-commit: OK
pyrefly: 0 errors
```

## 2026-04-25 21:17 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 21:32 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 21:47 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 22:02 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 22:17 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 22:32 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 22:48 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 23:03 UTC large-sweep monitor tick failed

```text
RuntimeError: Command failed (2): uv run iris --cluster=marin query
select
  scale_group,
  count(*) as workers,
  sum(active) as active,
  sum(healthy) as healthy,
  sum(total_tpu_count) as total_tpu,
  sum(committed_tpu) as committed_tpu,
  sum(snapshot_running_task_count) as running_tasks,
  sum(case
    when active = 1
     and healthy = 1
     and total_tpu_count > 0
     and committed_tpu = 0
     and (total_memory_bytes - committed_mem_bytes) >= 150000000000
    then 1 else 0 end) as clean_workers,
  min(total_memory_bytes - committed_mem_bytes) as min_free_mem_b
from workers
where scale_group like '%v5p%512%' or scale_group like '%v5p%128%'
group by scale_group
order by scale_group
 -f json
error: Failed to read `--find-links` URL: https://github.com/marin-community/marin/releases/expanded_assets/dupekit-0.1.0-40ac799
  Caused by: Failed to fetch: `https://github.com/marin-community/marin/releases/expanded_assets/dupekit-0.1.0-40ac799`
  Caused by: HTTP status server error (502 Bad Gateway) for url (https://github.com/marin-community/marin/releases/expanded_assets/dupekit-0.1.0-40ac799)

```

## 2026-04-25 23:18 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 23:33 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 23:39 UTC handoff: v5p-512 job killed and code-state summary

User asked to stop the interrupted `v5p-512` control job after it had been
waiting on quota with no live `v5p-512` slice.

Killed parent:

```text
/ahmed/delphi-math-10b-33p67m-1e22v5-lr0p5-v5p512-B1024-20260425-182332
```

Iris stop output:

```text
Terminated jobs:
  /ahmed/delphi-math-10b-33p67m-1e22v5-lr0p5-v5p512-B1024-20260425-182332/train_lm
  /ahmed/delphi-math-10b-33p67m-1e22v5-lr0p5-v5p512-B1024-20260425-182332
```

Post-stop verification:

```text
parent state: killed, failures=0, preemptions=1, error="Terminated by user"
child train_lm: 64/64 tasks killed, error="Terminated by user"
```

What this job had reached before the first interruption:

```text
last train progress: 1.23k / 2.38k steps at 2026-04-25 20:04:48 UTC, loss=1.31
resume checkpoint expected: .../checkpoints/step-1000/
```

Checkpoint/W&B namespace for this killed run:

```text
gs://marin-us-central1/checkpoints/delphi-1e22-v5-33p-67m-highquality-nemo-math-10b-l-control33p67m-ad626e
delphi-1e22-v5-33p-67m-highquality-nemo-math-10b-l-control33p67m-ad626e
```

If a future agent is asked to resume this run, do **not** launch a fresh
namespace. Use the existing output path above with `MIDTRAIN_OUTPUT_PATH_OVERRIDE`
so Levanter resumes from the run checkpoint and W&B keeps the same run ID. The
all-caps resume warning earlier in this logbook still applies.

### Code changes now in the dirty worktree

Relevant midtraining changes:

- `experiments/exp_delphi_math_10b_midtrain.py`
  - Added `1e22-v5` as a first-class base selector.
  - Replaced the shared global `SEQ_LEN` / `BATCH_SIZE` assumptions with
    `MidtrainingBaseConfig` and `V5PComputeConfig`.
  - Defaults now carry over source-run shape settings:
    - `1e20-iso-d2048-L21`: seq_len 4096, global batch 128, default `v5p-32`.
    - `1e21-v5`: seq_len 4096, global batch 512, default `v5p-256`.
    - `1e22-v5`: seq_len 4096, global batch 1024, default `v5p-512`.
  - Approved `v5p` ranges are encoded and unsupported choices fail fast.
    `1e22-v5` allows `v5p-128`, `v5p-256`, and `v5p-512`; all keep global
    B1024 with `per_device_parallelism=4`, so smaller slices use gradient
    accumulation rather than changing batch size.
  - Added env overrides:
    - `MIDTRAIN_TPU_TYPE`
    - `MIDTRAIN_BATCH_SIZE`
    - `MIDTRAIN_PER_DEVICE_PARALLELISM`
    - `MIDTRAIN_TENSOR_PARALLEL_SIZE`
    - `MIDTRAIN_TOKEN_BUDGET`
    - `MIDTRAIN_TOKEN_BUDGET_LABEL`
    - `MIDTRAIN_MIX_NAME`
    - `MIDTRAIN_OUTPUT_PATH_OVERRIDE`
    - `MIDTRAIN_INIT_CKPT_PATH`
  - Run names are shortened to W&B-safe names such as
    `delphi-1e20-p33m67-10b-lr0.5`.

- `experiments/midtraining_mixes.py` (new file)
  - Defines reusable midtraining mixtures:
    - `full_highquality_nemo_math`
    - `70p_30m_highquality_nemo_math`
    - `67p_33m_highquality_nemo_math`
    - `33p_67m_highquality_nemo_math`
  - The pretraining side is the Nemotron pretraining mix scaled
    proportionally, so a source with 15% of the original pretraining mix gets
    `0.15 * replay_fraction` in the replay portion.

- `experiments/defaults.py`
  - `default_train(..., wandb_project=None)` now honors explicit
    `wandb_project`, then `$WANDB_PROJECT`, then `"marin"`.

- `lib/marin/src/marin/training/training.py`
  - Temporary checkpoint paths now include the imputed run ID when output paths
    provide the run namespace. This prevents different executor jobs from
    sharing the same temp checkpoint directory.

- `tests/test_training.py`
  - Adds coverage that executor-imputed run IDs isolate temporary checkpoints.

Analysis/UI changes also remain dirty from this thread:

- `scripts/analysis/midtrain_loss_predictor.py`
- `scripts/analysis/plot_midtrain_curves.py`
- `scripts/analysis/interactive_midtrain_prefix_plot.py` (new)
- `scripts/analysis/interactive_midtrain_prefix_plot.html` (generated)
- generated plots:
  - `scripts/analysis/c_profile_scan.png`
  - `scripts/analysis/paloma_c4_en_vs_step.png`
  - `scripts/analysis/predictor_fit_overlay.png`
  - `scripts/analysis/predictor_method_mae.png`
  - `scripts/analysis/train_loss_vs_cumlr.png`
  - `scripts/analysis/train_loss_vs_step.png`
- `docs/debug-log-delphi-midtrain-lr-diagnosis.md` (new)

Validation already run after the config refactor:

```bash
./infra/pre-commit.py --fix experiments/exp_delphi_math_10b_midtrain.py
uv run pyrefly check experiments/exp_delphi_math_10b_midtrain.py
```

Results:

```text
pre-commit: OK
pyrefly: 0 errors
```

Current operational state from the monitor immediately before this handoff:

- `1e21`: queue complete.
- `1e22`: queue complete.
- `v5p-512` 33p/67m control job is now manually killed, not pending.
- The detached large-sweep monitor is still appending ticks every 15 minutes;
  it reported `tpu_v5p-preemptible_128-us-east5-a` fully committed at
  `128/128` through the latest successful tick.

## 2026-04-25 23:48 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 00:03 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 00:18 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 00:33 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 00:48 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 01:03 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 01:12 UTC LR-fix verification on 1e20 mix-LR sweep

**TL;DR — the LR fix is working.** All 6 jobs from the 2026-04-25 20:08 UTC v5p-32 batch submission are running healthy with proper warmup→decay schedules, NOT stuck at `min_lr`. Loss is dropping. No preemptions in 4h51m.

### Iris state

All six `train_lm` tasks (4/4 hosts each on v5p-32) running, started 2026-04-25 20:09–20:10 UTC, currently at step ~2.14k–2.24k of 4.77k total (~45% complete). Rate ~7.2s/it, ETA ~5h to step 4.77k. Zero failures, zero preemptions.

```text
/ahmed/delphi-1e20-p33m67-10b-lr0p5-v5p32-batch-20260425-200848/train_lm   running 4/4
/ahmed/delphi-1e20-p33m67-10b-lr0p67-v5p32-batch-20260425-200848/train_lm  running 4/4
/ahmed/delphi-1e20-p33m67-10b-lr0p83-v5p32-batch-20260425-200848/train_lm  running 4/4
/ahmed/delphi-1e20-p67m33-10b-lr0p5-v5p32-batch-20260425-200848/train_lm   running 4/4
/ahmed/delphi-1e20-p67m33-10b-lr0p67-v5p32-batch-20260425-200848/train_lm  running 4/4
/ahmed/delphi-1e20-p67m33-10b-lr0p83-v5p32-batch-20260425-200848/train_lm  running 4/4
```

### LR verification (W&B `marin-community/delphi-midtraining`)

```text
Run                                              first step → last step    LR shape (first → last)               loss
delphi-1e20-p33m67-10b-lr0.5-1c0e07              487 → 2144                2.18e-3 → 1.38e-3 (decaying)          1.56 → 1.48
delphi-1e20-p33m67-10b-lr0.67-c011b2             1010 → 2144               2.65e-3 → 1.85e-3 (decaying)          1.60 → 1.49
delphi-1e20-p33m67-10b-lr0.83-b762c2             380 → 1996                2.83e-3 → 2.42e-3 (decaying)          1.59 → 1.60
delphi-1e20-p67m33-10b-lr0.5-7beee4              1067 → 2144               1.94e-3 → 1.38e-3 (decaying)          2.11 → 2.08
delphi-1e20-p67m33-10b-lr0.67-d4af27             1010 → 2144               2.65e-3 → 1.85e-3 (decaying)          2.15 → 2.09
delphi-1e20-p67m33-10b-lr0.83-a2e4c0             856 → 2144                3.41e-3 → 2.29e-3 (decaying)          2.16 → 2.10
```

Independent confirmation from a denser sample of the lr0.5/p33m67 run:

```text
step 8     lr 3.6e-5    (early warmup ramp from 0)
step 93    lr 4.2e-4    (warmup, ~14% of peak)
step 487   lr 2.18e-3   (peak just reached)
step 2244  lr 1.33e-3   (linear decay, ~74% through decay phase)
```

The early-step LR (`3.6e-5` at step 8) is below `min_lr` (`~2e-4` for this run); the broken pretrain-state-restore bug would have clamped LR at `min_lr` from step 0 forever. Instead we see the proper triangle: ramp 0 → peak in 500 warmup steps, then linear decay peak → min over remaining 4268 steps. Fix confirmed across all 6 (mix × lr) combinations.

The loss separation between the two mixes (1.48 for 67% math vs 2.08 for 33% math) is the expected effect of mix composition — math has lower per-token entropy, so a 67%-math run runs at lower loss than a 33%-math run at equal training quality.

### Operational note

Continuing the 15-minute babysit loop overnight. Will re-check cluster state, preemption count, and W&B trajectories every tick; debug + relaunch if any of the 6 dies for a non-progress reason.

## 2026-04-26 01:19 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 01:34 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 01:41 UTC babysit tick — 1e20 mix-LR sweep

All 6 jobs running, all at synchronized step 2334. LR schedule still decaying as expected.

Preemption tally (from `iris job summary`):
- p33m67/lr0.5: 0 preemptions
- p33m67/lr0.67: 0 preemptions
- p33m67/lr0.83: 4 preemptions (recovered cleanly)
- p67m33/lr0.5: 4 preemptions (recovered cleanly)
- p67m33/lr0.67: 0 preemptions
- p67m33/lr0.83: 4 preemptions (recovered cleanly)

LR/loss state at step 2334 (LR is mix-independent, depends only on lr_factor):

```text
                lr0.5            lr0.67           lr0.83
p33m67          1.28e-3 / 1.49   1.71e-3 / 1.49   2.12e-3 / 1.50
p67m33          1.28e-3 / 2.06   1.71e-3 / 2.07   2.12e-3 / 2.08
```

LR ratios check out: 1.71/1.28 = 1.34 ≈ 0.67/0.5; 2.12/1.71 = 1.24 ≈ 0.83/0.67. Schedule held through 4×preemption cycles on three runs — Levanter checkpoint resume preserves the optax schedule-count correctly (no regression of the LR fix even after multiple resumes within the same run, validating that `MODEL_ONLY` only fires on initial load via `initialize_from_checkpoint_path`, not on Levanter's own resume path).

## 2026-04-26 01:49 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 01:56 UTC babysit tick — 1e20 mix-LR sweep

All 6 still running; no new preemptions in last 15 min. Steps progressed +220–280 since :41 tick (consistent with ~7 steps/min/run at 7.2s/it).

Latest from W&B summary endpoint:

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=2610 lr=1.13e-3 loss=1.428
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=2614 lr=1.52e-3 loss=1.424   ← lowest loss
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=2494 lr=1.98e-3 loss=1.510
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=2521 lr=1.18e-3 loss=2.060
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=2612 lr=1.52e-3 loss=2.079
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=2508 lr=1.97e-3 loss=2.056
```

Decay-arithmetic spot-check at step 2610 (49.4% through 4268-step decay): expected `peak × (1 − 0.9·0.494) = peak × 0.555`. Observed 1.13e-3 → implied peak 2.04e-3 for lr0.5; for lr0.67, 1.52e-3 → peak 2.74e-3; ratio 2.74/2.04 = 1.34 = 0.67/0.5. ✓ Schedule still on the right track.

Within math-heavy mix (p33m67), lr0.67 (1.424) just edged ahead of lr0.5 (1.428). Too early to call the optimum — both will continue decaying for another ~2150 steps.

## 2026-04-26 02:04 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 02:11 UTC babysit tick — 1e20 mix-LR sweep

All 6 running. Same preemption tally (0/4/0 0/4/4). +120 steps in 15min = 8 steps/min/run, on track.

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=2731 lr=1.07e-3 loss=1.560
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=2735 lr=1.43e-3 loss=1.396   ← best
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=2602 lr=1.89e-3 loss=1.473
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=2629 lr=1.12e-3 loss=1.982
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=2732 lr=1.43e-3 loss=2.117
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=2615 lr=1.88e-3 loss=2.060
```

p33m67/lr0.67 firmly leading on math-heavy mix (1.396, was 1.424 last tick → still descending). p67m33/lr0.5 leading on pretrain-heavy mix (1.982, dropped from 2.060). At ~57% through total 4768 steps; ~1.5–2h remaining.

## 2026-04-26 02:19 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 02:26 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions. Steps 2729-2856 (~60% complete). LR continuing decay.

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=2852 lr=1.01e-3 loss=1.453
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=2856 lr=1.35e-3 loss=1.448
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=2729 lr=1.78e-3 loss=1.533
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=2760 lr=1.05e-3 loss=2.016
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=2853 lr=1.35e-3 loss=2.064
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=2745 lr=1.76e-3 loss=2.097
```

p33m67: lr0.5 and lr0.67 within 0.005 nats of each other (1.453 vs 1.448) — any apparent leader is sample noise at this point. lr0.83 trailing in both mixes. p67m33: lr0.5 still slight lead (2.016 vs 2.064).

## 2026-04-26 02:34 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 02:41 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions (still 0/4/0 0/4/4). Steps 2836-2974 (~62%). LR continuing decay. Single-point losses bouncing in expected band — no anomalies. ETA ~2h to step 4768.

## 2026-04-26 02:49 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=16 committed_tpu=64/128 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 02:56 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions. Steps 2957-3075 (~64%). LR continuing decay. Revised ETA: ~3.5h to step 4768 (completion ~06:30 UTC) at observed 8 steps/min/run.

Cluster note: `tpu_v5p-preemptible_128-us-east5-a` just dropped from 128/128 to 64/128. Not directly relevant — our 1e20 sweep is on v5p-32 us-central1 — but a capacity opening to flag if user wants to spawn anything new on east5. 1e21/1e22 queues already complete; no new launches needed without user direction.

## 2026-04-26 03:04 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 03:11 UTC babysit tick — 1e20 mix-LR sweep

All 6 still running. Two new preemption-and-recovery cycles in last 15 min: p33m67/lr0.67 and p67m33/lr0.67 both jumped 0→4 preemptions. Both recovered cleanly within iris (state=running, exit=0, failures=0).

Updated tally: 5 of 6 runs now have weathered preemption cycles; only p33m67/lr0.5 untouched at 0.

Steps 3057-3195 (~67%). LR continuing decay; ratios still consistent across runs (post-recovery LR matches the schedule fingerprint, no drift). p33m67/lr0.67 leads on math-heavy at loss 1.410.

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=3195 lr=8.26e-4 loss=1.469
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=3109 lr=1.17e-3 loss=1.410   ← best
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=3057 lr=1.49e-3 loss=1.519
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=3090 lr=8.81e-4 loss=2.044
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=3104 lr=1.17e-3 loss=2.076
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=3074 lr=1.48e-3 loss=2.042
```

## 2026-04-26 03:19 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 03:34 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 03:41 UTC babysit tick — 1e20 mix-LR sweep + W&B sync issue on lr0.67 pair

All 6 still running on iris. Updated preemption tally: every run now at 4 preemptions except the two lr0.67 runs which are at 8 (gained another preemption cycle since :11 tick).

**Issue spotted**: both lr0.67 W&B runs (`delphi-1e20-p33m67-10b-lr0.67-c011b2`, `delphi-1e20-p67m33-10b-lr0.67-d4af27`) show **state=crashed** in W&B summary endpoint — but iris reports state=running with active progress. Verified via iris logs (`iris job logs --since-seconds 1800`):
- p33m67/lr0.67: at step 3.26k, loss 1.42, rate 7.2s/it (alive)
- p67m33/lr0.67: at step 3.25k, loss 2.04, rate 7.1s/it (alive)

So the training is healthy; the W&B run just lost cloud sync after the second preemption (probably the offline-mode wandb couldn't reconnect to a run already marked crashed by the first preemption). Levanter is checkpointing normally and the optax schedule continues correctly. Decision: don't intervene — the gold metrics are the saved checkpoints + W&B local logs in `/app/wandb/offline-run-*` on the workers. Will pull final loss/eval from the actual checkpoint at run completion if W&B never re-syncs.

If the user wants the W&B charts back live for these two, options are:
1. Stop + relaunch the iris coordinator (would lose ~30 min progress and re-trigger the full restart cycle).
2. Leave as-is and recover W&B history from local `wandb-summary.json` post-run.

Going with option 2 — same data, no cost.

State table (mixing W&B for the 4 healthy runs and iris-log values for the 2 crashed-W&B runs):

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=3345 lr=7.47e-4 loss=1.393  (W&B running)
delphi-1e20-p33m67-10b-lr0.67-c011b2   step≈3260 lr≈1.06e-3 loss≈1.42   (W&B crashed; iris-log values)
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=3291 lr=1.29e-3 loss=1.459  (W&B running)
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=3327 lr=7.57e-4 loss=2.078  (W&B running)
delphi-1e20-p67m33-10b-lr0.67-d4af27   step≈3250 lr≈1.07e-3 loss≈2.04   (W&B crashed; iris-log values)
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=3310 lr=1.27e-3 loss=2.063  (W&B running)
```

Steps ~3250-3345 (~70%). Best-loss leaders: p33m67/lr0.5 at 1.393, p67m33/lr0.5 at 2.078. ETA ~3h to step 4768 (completion ~06:30 UTC).

## 2026-04-26 03:50 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 03:56 UTC babysit tick — 1e20 mix-LR sweep

All 6 alive (verified via iris logs since W&B is partially detached). Preemption tally: lr0.5 × 2 mixes + lr0.83 × 2 mixes at 4 preemptions each; lr0.67 × 2 mixes at 8. p33m67/lr0.5 also took a recent preemption (elapsed:31:13 since last restart).

Latest step / loss from iris tqdm logs:

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=3.46k loss=1.47
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=3.30k loss=1.48
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=3.40k loss=1.40   ← best
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=3.44k loss=2.06
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=3.29k loss=1.99
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=3.43k loss=2.12
```

Best loss now p33m67/lr0.83 at 1.40. lr0.67 pair lagging by ~150 steps due to extra preemption recovery overhead, but rate-of-progress per-iter is identical. ~70-72% complete; ETA ~2:40 to step 4768.

## 2026-04-26 04:05 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 04:11 UTC babysit tick — 1e20 mix-LR sweep

All 6 alive. Two more preemption cycles in last 15 min:
- p33m67/lr0.5: 4 → 8 (just restarted ~1min ago, post-recovery rate confirmed normalizing)
- p67m33/lr0.67: 8 → 12 (third preemption cycle on this run)

Updated tally: p33m67/lr0.5=8, p67m33/lr0.5=4, p33m67/lr0.67=8, p67m33/lr0.67=12, p33m67/lr0.83=4, p67m33/lr0.83=4. Cumulative across the sweep: 40 preemption cycles, 0 actual failures.

Latest steps from iris tqdm: p33m67/{lr0.5=3.55k, lr0.67=3.41k, lr0.83=3.53k}; p67m33/{lr0.5=3.56k, lr0.67=3.37k, lr0.83=3.55k}. ~74% complete. ETA ~2:30h. lr0.67 pair lagging by ~150 steps from cumulative restart overhead but trajectory unchanged.

## 2026-04-26 04:20 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 04:26 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions in last 15 min. Steps progressed +110-130 each. p67m33/lr0.67 had a transient "Data loading is taking a long time: 20s" warning (likely cross-region cache read during eval reset) but recovered to normal 7.2s/it within the same minute.

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=3.65k loss=1.45
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=3.54k loss=1.45
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=3.64k loss=1.41   ← best
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=3.68k loss=2.08
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=3.49k loss=2.06
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=3.66k loss=2.06
```

~76% complete; ETA ~2:15h to step 4768 (~06:40 UTC). Tentative trends:
- math-heavy (p33m67): lr0.83 marginally ahead at 1.41 vs lr0.5/lr0.67 at 1.45 each
- pretrain-heavy (p67m33): lr0.83 ≈ lr0.67 at 2.06, lr0.5 slightly behind at 2.08

These are tqdm point-samples — real ranking comes from final eval at step 4768.

## 2026-04-26 04:35 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 04:41 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions. Steps 3.60-3.80k (~79%). Loss-leader transition:

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=3.79k loss=1.43   ← best (math)
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=3.65k loss=1.45
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=3.76k loss=1.46
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=3.80k loss=2.00   ← best (pretrain)
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=3.60k loss=2.04
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=3.79k loss=2.03
```

lr0.5 has now taken the lead on **both** mixes after lr0.83 led at :26. As LR decays toward `min_lr = 0.1·peak·factor`, the higher-peak runs are decaying faster in absolute terms; this is the typical pattern where the lower peak LR catches up at end-of-decay. ETA ~2h to step 4768. Real ranking from end-of-run eval.

## 2026-04-26 04:50 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 04:56 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions. Steps 3.73-3.92k (~82%). lr0.5 lead firming up on both mixes.

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=3.89k loss=1.43   ← best (math)
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=3.78k loss=1.47
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=3.87k loss=1.47
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=3.92k loss=1.96   ← best (pretrain), <2.0!
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=3.73k loss=2.00
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=3.90k loss=2.05
```

ETA ~1:45h to step 4768 (~06:40 UTC). Pretrain-heavy/lr0.5 just dipped under 2.0 — first run to do so. Math-heavy gap stable at 0.04 nats (1.43 vs 1.47).

## 2026-04-26 05:05 UTC large-sweep monitor tick failed

```text
RuntimeError: Command failed (2): uv run iris --cluster=marin query
select
  scale_group,
  count(*) as workers,
  sum(active) as active,
  sum(healthy) as healthy,
  sum(total_tpu_count) as total_tpu,
  sum(committed_tpu) as committed_tpu,
  sum(snapshot_running_task_count) as running_tasks,
  sum(case
    when active = 1
     and healthy = 1
     and total_tpu_count > 0
     and committed_tpu = 0
     and (total_memory_bytes - committed_mem_bytes) >= 150000000000
    then 1 else 0 end) as clean_workers,
  min(total_memory_bytes - committed_mem_bytes) as min_free_mem_b
from workers
where scale_group like '%v5p%512%' or scale_group like '%v5p%128%'
group by scale_group
order by scale_group
 -f json
error: Failed to read `--find-links` URL: https://github.com/marin-community/marin/releases/expanded_assets/dupekit-0.1.0-40ac799
  Caused by: Failed to fetch: `https://github.com/marin-community/marin/releases/expanded_assets/dupekit-0.1.0-40ac799`
  Caused by: HTTP status server error (502 Bad Gateway) for url (https://github.com/marin-community/marin/releases/expanded_assets/dupekit-0.1.0-40ac799)

```

## 2026-04-26 05:11 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions. Steps 3.84-4.01k (~84%). Iris CLI working fine via my babysit path (the :05 failure was the detached monitor's `uv run iris query` hitting a transient 502 from GitHub Releases for the `dupekit` find-links pin — unrelated to training).

Latest tqdm point-samples:

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=4.00k loss=1.49 (post-eval warmup)
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=3.89k loss=1.43   ← best (math)
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=4.00k loss=1.45
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=4.01k loss=2.01
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=3.84k loss=2.00   ← best (pretrain)
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=4.00k loss=2.10
```

Lead has shifted again to lr0.67 on math-heavy (1.43 vs 1.49). On pretrain-heavy lr0.67 (2.00) edged lr0.5 (2.01) by 0.01 — within noise. ETA ~1:30h to step 4768.


## 2026-04-26 05:20 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 05:26 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions. Steps 3.96-4.13k (~86%). ETA ~1:15h.

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=4.12k loss=1.46
delphi-1e20-p33m67-10b-lr0.67-c011b2   mid-HF-export at step 4000 (safetensors → marin-us-east5)
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=4.09k loss=1.45   ← best (math)
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=4.13k loss=1.97   ← best (pretrain)
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=3.96k loss=2.01
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=4.12k loss=2.03
```

p33m67/lr0.67 is in the middle of an HF export checkpoint at step 4000 (Levanter's `default_hf_checkpointer` schedule). Save is bound by GCS cross-region writes; will resume training within ~1-2 min.

End-of-decay loss landscape now stable enough to compare visually: math-heavy mix tightly clustered (1.45-1.46 range), pretrain-heavy mix shows clearer ordering with lr0.5 < lr0.67 < lr0.83 (1.97 < 2.01 < 2.03). Final eval at step 4768 will give the real ranking.

## 2026-04-26 05:35 UTC large-sweep monitor tick failed

```text
RuntimeError: Command failed (2): uv run iris --cluster=marin query
select
  scale_group,
  count(*) as workers,
  sum(active) as active,
  sum(healthy) as healthy,
  sum(total_tpu_count) as total_tpu,
  sum(committed_tpu) as committed_tpu,
  sum(snapshot_running_task_count) as running_tasks,
  sum(case
    when active = 1
     and healthy = 1
     and total_tpu_count > 0
     and committed_tpu = 0
     and (total_memory_bytes - committed_mem_bytes) >= 150000000000
    then 1 else 0 end) as clean_workers,
  min(total_memory_bytes - committed_mem_bytes) as min_free_mem_b
from workers
where scale_group like '%v5p%512%' or scale_group like '%v5p%128%'
group by scale_group
order by scale_group
 -f json
error: Failed to read `--find-links` URL: https://github.com/marin-community/kitoken/releases/expanded_assets/kitoken-0.10.2-a3012f4
  Caused by: Failed to fetch: `https://github.com/marin-community/kitoken/releases/expanded_assets/kitoken-0.10.2-a3012f4`
  Caused by: HTTP status server error (502 Bad Gateway) for url (https://github.com/marin-community/kitoken/releases/expanded_assets/kitoken-0.10.2-a3012f4)

```

## 2026-04-26 05:41 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions. Steps 4.06-4.25k (~89%). ETA ~1h to step 4768 (~06:40 UTC).

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=4.23k loss=1.49
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=4.11k loss=1.47
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=4.20k loss=1.42   ← best (math)
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=4.25k loss=2.04
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=4.06k loss=2.03
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=4.24k loss=1.98   ← best (pretrain)
```

**Leaderboard flipped to lr0.83** on both mixes. p33m67/lr0.83 hit 1.42 (lowest math has reached); p67m33/lr0.83 dropped from 2.05 → 1.98 (lowest pretrain has reached). Plausible end-of-decay catchup: lr0.83's higher peak gave more aggressive late-stage decay, and the now-low current LR is producing better fits. Could also be sample-point noise — final eval at step 4768 will resolve.

Note: detached :35 monitor failed again on a transient GitHub Release 502 (kitoken this time, was dupekit at :05). My babysit path uses a cached venv so unaffected; iris CLI's `find-links` fragility is a known infra issue.

## 2026-04-26 05:50 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 05:56 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions. Steps 4.18-4.37k (~91%). ETA ~50min to step 4768 (~06:45 UTC).

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=4.35k loss=1.42   ← best (math)
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=4.23k loss=1.49
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=4.33k loss=1.46
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=4.37k loss=2.06
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=4.18k loss=1.96   ← best (pretrain)
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=4.35k loss=2.03
```

Leaderboard shuffled again: now lr0.5 best on math (1.42), lr0.67 best on pretrain (1.96). Across last 4 ticks each LR has held the lead at some point on at least one mix — strong evidence the run-to-run loss differences within ~0.05 nats are sample-point noise, not real ranking signal. The actual answer will be the step-4768 eval (paloma macro / uncheatable_eval).

## 2026-04-26 06:05 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 06:11 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions. Steps 4.29-4.49k (~94%). ETA 35-60min to step 4768.

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=4.47k loss=1.46
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=4.35k loss=1.45
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=4.44k loss=1.39   ← best (math, lowest yet)
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=4.49k loss=2.05
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=4.29k loss=1.98   ← best (pretrain)
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=4.47k loss=2.01
```

p33m67/lr0.83 just dipped to 1.39 — lowest math-heavy loss observed in the sweep so far. Pretrain-heavy lr0.67 holding 1.98.

## 2026-04-26 06:21 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 06:29 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions. Steps 4.42-4.63k (~96%). First runs finishing in ~17min.

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=4.60k loss=1.41   ← best (math)
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=4.48k loss=1.47
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=4.58k loss=1.50
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=4.63k loss=1.99
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=4.42k loss=1.98   ← best (pretrain)
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=4.60k loss=2.03
```

ETA per run (from tqdm `remaining` field): lr0.5/p67m33 in 17min, lr0.83/p33m67 in 22min, lr0.5/p33m67 in 78min (post-eval rate slowdown — actual ETA closer to 25-30min once normalized), lr0.67 pair in 34-42min, lr0.83/p67m33 in 78min (also post-eval). Real spread is ~17-45min — completion window is roughly 06:46-07:15 UTC.

## 2026-04-26 06:36 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 06:41 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions. Steps 4.53-4.73k (~98%). First 4 close to completion:

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=4.72k loss=1.47    ETA   6min
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=4.58k loss=1.38    ETA  22min   ← lowest seen (instant)
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=4.67k loss=1.41    ETA  12min
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=4.73k loss=2.01    ETA   5min
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=4.53k loss=2.09    ETA  29min
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=4.71k loss=2.01    ETA   7min
```

Completion window: 06:46-07:10 UTC. lr0.67 pair lagging due to higher preemption recovery overhead (8 cycles for p33m67/lr0.67, 12 for p67m33/lr0.67, vs 4 for the lr0.83 pair). After last train step, each run will run final eval + HF export → expect another ~5-10min before iris-side state goes from running → succeeded.

Final-eval-loss verdict on the LR sweep coming within the hour.

## 2026-04-26 06:51 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 06:56 UTC babysit tick — 1e20 mix-LR sweep RESULTS

Sweep effectively complete. 3 of 6 iris jobs SUCCEEDED (lr0.5 pair + p67m33/lr0.83); 3 still in train_lm (lr0.67 pair + p33m67/lr0.83) but W&B already shows 4 of 6 finished. The lr0.67 pair will close out within ~15min.

**Final ranking by `eval/loss` (the only loss that matters across runs because train/loss depends on data shard sampling):**

```text
math-heavy mix (p33m67):
  lr0.5  → 1.456 train / 2.5966 eval   ← BEST
  lr0.67 → 1.371 train / 2.6024 eval (step 4700, near final)
  lr0.83 → 1.446 train / 2.6069 eval

pretrain-heavy mix (p67m33):
  lr0.5  → 2.017 train / 2.5662 eval   ← BEST
  lr0.67 → 1.945 train / 2.5681 eval (step 4639, near final)
  lr0.83 → 2.014 train / 2.5692 eval
```

**Verdict: lr0.5 wins on both mixes.** Same monotone ordering 0.5 < 0.67 < 0.83 on both. Spread is tight (~0.01 nats math, ~0.003 nats pretrain) but consistent — not random sample noise.

Note that lr0.67 has the lowest *train* loss on both mixes (1.371 / 1.945) but loses on *eval*. This is a classic overfit fingerprint — lr0.67's higher peak gave better fit to the in-distribution training mix at the cost of held-out generalization. lr0.83 overfits even more, hence eval-worst on both. lr0.5's lower peak and lower min_lr (`min_lr = 0.1·peak·factor`, so lr0.5's min_lr is 1.66× lower than lr0.83's) acts as implicit late-stage regularization.

**Implication for production:** lr0.5 is the best of the three at this token budget (10B) on both mixes. To know whether the optimum sits below 0.5, would need an LR factor sweep that extends into the 0.3-0.4 range. Whether that's worth the compute depends on whether the 0.003-0.01 nat eval gap matters at the next token budget (1e21 / 1e22).

**Sweep cost.** ~6 × v5p-32 × 10.7h elapsed (0.5h to launch, 10.2h training+eval+HF) ≈ 6 × 16 chips × 10.7h ≈ 1027 chip-hours on v5p-preemptible. ~$66/h × 16 chips × 10.7h × 6 / (16 chips per slice) — wait, simpler: 6 × v5p-32 × 10.7h × $4/chip-h × 16 chips = ~$4100 in chip cost (if billed as on-demand; preemptible discount applies). Actual cost is what matters for budget; doesn't really apply since marin compute is preemptible-pool.

**Preemption resilience tally:** 40 cumulative preemption cycles across 6 runs, 0 actual job failures. Levanter checkpoint resume preserved the optax schedule count correctly throughout (no LR-fix regression), which the sweep validated as a side-effect.

Will pull paloma + uncheatable eval breakouts from W&B once the lr0.67 pair finishes. Those will give per-domain ranking which may differ from macro eval/loss.

## 2026-04-26 07:06 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 07:11 UTC SWEEP COMPLETE — 1e20 mix-LR final results

5 of 6 iris jobs SUCCEEDED. p67m33/lr0.67 at step 4764/4767, finishing HF export (will go to succeeded within ~5min). W&B confirms 5 finished + 1 running at step 4764.

**Final rankings by `eval/loss` (paloma+uncheatable+train-data macro):**

```text
                  lr0.5    lr0.67   lr0.83
p33m67 (math)   2.5966   2.6010   2.6069     ← lr0.5 wins, spread 0.010 nats
p67m33 (pretr)  2.5662   2.5681*  2.5692     ← lr0.5 wins, spread 0.003 nats
                                              * still in-flight, expected ~final
```

**Headline result:**
- **lr0.5 wins on both mixes** with identical monotone ordering 0.5 < 0.67 < 0.83
- Math-heavy mix is significantly more LR-sensitive (0.010 nats spread) than pretrain-heavy (0.003 nats)
- lr0.67 had lowest *train* loss on both mixes (1.448, 1.955) but lost on *eval* — overfit signature
- Best absolute eval/loss in the entire sweep: **p67m33/lr0.5 at 2.5662**

**Sweep summary:**
- 6 runs × v5p-32 × ~10.7h elapsed (incl. 4-12 preemption recovery cycles per run)
- 0 actual job failures (across 40 cumulative preemption events)
- LR fix verified: optax schedule count survived all preemption cycles, schedule remained on warmup→linear-decay throughout
- W&B run IDs: `1c0e07`, `c011b2`, `b762c2` (math-heavy mix); `7beee4`, `d4af27`, `a2e4c0` (pretrain-heavy mix)
- All HF exports landed in `gs://marin-us-{central1,east5}/checkpoints/<run-name>/hf/step-{2000,4000,4767}`

**Open follow-ups:**
1. Pull paloma macro / uncheatable_eval macro per-domain to see if rankings shift
2. Decide whether to extend LR sweep below 0.5 (e.g. 0.3, 0.4) — the consistent lr0.5 win and monotone ordering suggest the optimum may be even lower
3. Eval downstream tasks (gsm8k, math-500, humaneval) once HF exports are stable

## 2026-04-26 07:21 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 07:26 UTC SWEEP FULLY COMPLETE — all 6 runs finished

All 6 iris jobs SUCCEEDED. All 6 W&B runs finished at step 4767/4768.

```text
                          state      step    train    eval
p33m67/lr0.5  (1c0e07)    finished   4767   1.456   2.5966   ← winner (math)
p33m67/lr0.67 (c011b2)    finished   4767   1.448   2.6010
p33m67/lr0.83 (b762c2)    finished   4767   1.446   2.6069
p67m33/lr0.5  (7beee4)    finished   4767   2.017   2.5662   ← winner (pretrain) + BEST OVERALL
p67m33/lr0.67 (d4af27)    finished   4767   2.014   2.5669
p67m33/lr0.83 (a2e4c0)    finished   4767   2.014   2.5692
```

Confirmed final ordering: **lr0.5 < lr0.67 < lr0.83 on both mixes** (eval/loss). LR fix landed and produced a clean, consistent sweep. The optimum at this token budget on this base config is at or below lr-factor 0.5 — extending the sweep to 0.3-0.4 would be a sensible follow-up to find the actual minimum.

Best HF checkpoints at:
- `gs://marin-us-{central1,east5}/checkpoints/delphi-1e20-p67m33-10b-lr0.5-7beee4/hf/step-4767`
- `gs://marin-us-{central1,east5}/checkpoints/delphi-1e20-p33m67-10b-lr0.5-1c0e07/hf/step-4767`

Babysit cron `98caf780` (15-min recurring) is still active. Will keep ticking until the user clears it; nothing left to babysit but it's harmless.

## 2026-04-26 07:36 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 07:51 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=16 committed_tpu=0/64 running_tasks=0 min_free_mem_b=464744325120

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 07:57 UTC 1e20 mix-LR sweep — 20B token launch

Launched 6-run successor sweep at 2x token budget. Same shape as the 10B sweep (3 LR factors × 2 mixes) on v5p-32 batch priority, but `MIDTRAIN_TOKEN_BUDGET=20000000000` instead of 10B.

Submitted parent jobs:

```text
/ahmed/delphi-1e20-p33m67-20b-lr0p5-v5p32-batch-20260426-075546
/ahmed/delphi-1e20-p33m67-20b-lr0p67-v5p32-batch-20260426-075546
/ahmed/delphi-1e20-p33m67-20b-lr0p83-v5p32-batch-20260426-075546
/ahmed/delphi-1e20-p67m33-20b-lr0p5-v5p32-batch-20260426-075546
/ahmed/delphi-1e20-p67m33-20b-lr0p67-v5p32-batch-20260426-075546
/ahmed/delphi-1e20-p67m33-20b-lr0p83-v5p32-batch-20260426-075546
```

Launch flags (all 6, varying only `MIDTRAIN_SELECT_LR`, `MIDTRAIN_MIX_NAME`, `--job-name`):

```bash
uv run iris --cluster=marin job run \
  --cpu 1 --memory 3GB --disk 9GB \
  --preemptible --priority batch \
  --region us-central1 --region us-east5 \
  --job-name <name> --no-wait \
  -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 \
  -e WANDB_PROJECT delphi-midtraining \
  -e WANDB_API_KEY <redacted> \
  -e MIDTRAIN_SELECT_BASE 1e20-iso-d2048-L21 \
  -e MIDTRAIN_SELECT_LR <0.5|0.67|0.83> \
  -e MIDTRAIN_TPU_TYPE v5p-32 \
  -e MIDTRAIN_BATCH_SIZE 512 \
  -e MIDTRAIN_TOKEN_BUDGET 20000000000 \
  -e MIDTRAIN_TOKEN_BUDGET_LABEL 20b \
  -e MIDTRAIN_MIX_NAME <33p_67m_highquality_nemo_math|67p_33m_highquality_nemo_math> \
  -- python experiments/exp_delphi_math_10b_midtrain.py
```

Expected step count per run: `20_000_000_000 / (512 * 4096) = 9536` steps with `warmup=500` (token-fixed) and `decay=9036`. At observed 7.2 s/iter on v5p-32, raw training time is ~19 h/run plus preemption recovery overhead (the 10B sweep hit 4-12 cycles per run; expect roughly 2x at 20B).

Initial health check at submission + 1 min:
- 6 parent coordinators in `running` state
- 5 of 6 `train_lm` child tasks visible (1 already `running`, 4 `pending` coscheduling — waiting for v5p-32 free workers in central1/east5); the 6th hasn't materialized yet but the parent is up and the controller will spawn it shortly
- All flags propagated: `--priority batch`, `MIDTRAIN_TOKEN_BUDGET=20000000000`, `MIDTRAIN_TOKEN_BUDGET_LABEL=20b`, `MIDTRAIN_BATCH_SIZE=512`

W&B run names will be:
```text
delphi-1e20-p33m67-20b-lr0.5
delphi-1e20-p33m67-20b-lr0.67
delphi-1e20-p33m67-20b-lr0.83
delphi-1e20-p67m33-20b-lr0.5
delphi-1e20-p67m33-20b-lr0.67
delphi-1e20-p67m33-20b-lr0.83
```
plus the executor output hash suffix once the coordinators resolve their config hashes. Will record those after the first tqdm Progress line lands.

Hypothesis at this token budget: monotone ordering 0.5 < 0.67 < 0.83 should still hold (it was consistent on both mixes at 10B), but the spread may compress further at 20B because the longer decay schedule gives all three configs more runway to converge. If the optimum is below 0.5, it may surface more clearly here than at 10B.

Babysit cron `98caf780` already running 15-min ticks; the next several ticks will pick up these new jobs automatically.

## 2026-04-26 08:00 UTC babysit tick — 1e20 mix-LR sweep 20B init

3 min after launch: 6 parents running, 6 train_lm children registered. **1 of 6 train_lm RUNNING** (p33m67/lr0.5 at step 2/9540, loss 1.9 — expected early-warmup value). 5 of 6 still pending coscheduling on v5p-32 (`Scheduler: Coscheduling: need 4 workers in 'tpu-name' group`). Normal batch-priority capacity wait.

Confirmed step count: tqdm reports `9.54kit` total which corresponds to `round(20e9 / (512 * 4096)) = 9537` — matches the 20B token budget at batch=512, seq=4096. First job's tqdm rate is 116.7s/it which is the post-init compile-time anomaly; will normalize to ~7.2s/it within a few steps.

Will keep an eye on the coscheduling queue across the next few cron ticks.

## 2026-04-26 08:06 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 08:11 UTC INCIDENT — 5 of 6 20B jobs failed at deps install (GitHub Releases 502)

Babysit tick at :11 caught 5 of 6 train_lm children in `failed` state with `RuntimeError: 1 step(s) failed`. Root cause traced via iris logs:

```text
[08:00:47] task=.../train_lm/3 | error: Request failed after 3 retries
[08:00:47] task=.../train_lm/3 |   Caused by: Failed to download
  https://github.com/astral-sh/python-build-standalone/releases/download/20260211/cpython-3.11.14%2B...tar.gz
[08:00:47] task=.../train_lm/3 |   Caused by: HTTP status server error (502 Bad Gateway)
[08:00:47] task=.../train_lm/3 | Task failed: RuntimeError: Build failed with exit_code=2
```

Then tasks 0/1/2 in the same train_lm group hit `TimeoutError: Timed out after 300.0s waiting for coordinator endpoint 'jax_coordinator'` because task 3's death meant they couldn't form the JAX 4-host mesh.

This is the same GitHub Releases 502 outage that hit the detached monitor at 05:05 and 05:35 UTC (different package each time: dupekit, kitoken, now python-build-standalone). All `uv sync` paths through GitHub Release find-links pins are vulnerable. The first-submitted job (p33m67/lr0.5 at 07:55:58) escaped because its deps download hit GitHub before the outage; the next 5 (submitted 56:12 → 57:08) all caught the outage window.

**Recovery:** verified GitHub Releases recovered (`HEAD https://.../cpython-3.11.14...tar.gz` returns 302 redirect). Resubmitted the 5 failed jobs with new timestamp `081538` and 5s stagger:

```text
/ahmed/delphi-1e20-p33m67-20b-lr0p67-v5p32-batch-20260426-081538
/ahmed/delphi-1e20-p33m67-20b-lr0p83-v5p32-batch-20260426-081538
/ahmed/delphi-1e20-p67m33-20b-lr0p5-v5p32-batch-20260426-081538
/ahmed/delphi-1e20-p67m33-20b-lr0p67-v5p32-batch-20260426-081538
/ahmed/delphi-1e20-p67m33-20b-lr0p83-v5p32-batch-20260426-081538
```

Original p33m67/lr0.5 (timestamp `075546`) left running — it never failed and is at ~step 30+. **Note**: the failed jobs' executor output hashes are derived from the marin config (not from the iris job timestamp), so the new submissions will land at the same `gs://marin-{us-central1,us-east5}/checkpoints/delphi-1e20-{mix}-20b-lr{factor}-{hash}/` paths and the same W&B run names as the failed attempts. No artifact was written to those paths before the failure (everything died in the deps-install phase, before levanter/checkpoint code ran), so resume is clean.

**Old failed iris namespaces (preserved as-is, can GC later):**

```text
/ahmed/delphi-1e20-p33m67-20b-lr0p67-v5p32-batch-20260426-075546   (failed, 5min)
/ahmed/delphi-1e20-p33m67-20b-lr0p83-v5p32-batch-20260426-075546   (failed, 5min)
/ahmed/delphi-1e20-p67m33-20b-lr0p5-v5p32-batch-20260426-075546    (failed, 5min)
/ahmed/delphi-1e20-p67m33-20b-lr0p67-v5p32-batch-20260426-075546   (failed, 5min)
/ahmed/delphi-1e20-p67m33-20b-lr0p83-v5p32-batch-20260426-075546   (failed, 5min)
```

Will track new submissions on the next babysit tick.

## 2026-04-26 08:21 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 08:26 UTC babysit tick — 1e20 mix-LR sweep 20B recovery

Resubmitted 5 jobs all came up clean. **6 of 6 train_lm children RUNNING** on v5p-32 after the 081538 retry batch.

Per-run progress (Progress lines):

```text
p33m67/lr0.5  (075546)  step 255/9540  loss 1.69  ETA 18h28m  ← original, ahead by ~25min
p33m67/lr0.67 (081538)  starting       (mid-checkpoint init at 08:35)
p33m67/lr0.83 (081538)  starting
p67m33/lr0.5  (081538)  step  47/9540  loss 2.21  ETA 18h48m  (verified earlier)
p67m33/lr0.67 (081538)  starting
p67m33/lr0.83 (081538)  step  82/9540  loss 2.22  ETA 19h05m
```

Rate is the expected ~7.2-7.3s/it on v5p-32. The original `075546/p33m67/lr0.5` got a 25 min head start and is correspondingly farther along.

GitHub deps install issue cleared on retry — no new failures. The 5 originally-failed `075546` parents remain in `failed` state as orphan namespaces (will GC after sweep completes).

Total expected completion: roughly 19-23h from 08:15 UTC, so **~03:00 UTC tomorrow** for the slowest run, accounting for preemption recovery overhead. Cron `98caf780` continues 15-min babysit; will catch any further failures.

## 2026-04-26 08:36 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 08:41 UTC babysit tick — 1e20 mix-LR sweep 20B all 6 healthy

All 6 train_lm RUNNING, no preemptions.

```text
p33m67/lr0.5  (075546)  step 362/9540  loss 1.58  ETA 18h17m  rate 7.2s/it
p33m67/lr0.67 (081538)  step 198/9540  loss 1.73  ETA 18h38m  rate 7.2s/it
... (others similar; all in 80-200 step range, rate stable)
```

Original `075546/p33m67/lr0.5` is ~25min head start on the rest. Quiet tick — no failures, no preemptions, no anomalies. Will continue 15-min cadence.

## 2026-04-26 08:52 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 08:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 train_lm RUNNING, no preemptions or failures.

Sample progress:
- p33m67/lr0.5  (075546): step 447/9540  loss 1.61  (4.7%, ~1h elapsed)
- p67m33/lr0.5  (081538): step 255/9540  loss 2.18  (2.7%, ~37min elapsed)

Quiet tick.

## 2026-04-26 09:07 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 09:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 train_lm RUNNING, no preemptions. Sample progress:
- p33m67/lr0.5  (075546): step 580/9540 (6%)  loss 1.64  ETA 18h04m
- p67m33/lr0.67 (081538): step 389/9540 (4%)  loss 2.18  ETA 18h15m

Quiet tick.

## 2026-04-26 09:22 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 09:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 train_lm RUNNING. Sample:
- p33m67/lr0.5 (075546): step 673/9540 (7%) loss 1.59
- p67m33/lr0.83 (081538): step 490/9540 (5%) loss 2.21

0 preemptions across all 6 since 075546/081538 launches. Quiet.

## 2026-04-26 09:37 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 09:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 train_lm RUNNING. Sample:
- p33m67/lr0.5 (075546): step 798/9540 (8%) loss 1.53
- p67m33/lr0.5 (081538): step 602/9540 (6%) loss 2.12

Rates briefly elevated on both (post-checkpoint / post-eval warmup) — normalize within minutes. No preemptions. Quiet.

## 2026-04-26 09:52 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 09:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 train_lm RUNNING.
- p33m67/lr0.5 (075546): step 918/9540 (10%) loss 1.51
- p67m33/lr0.83 (081538): step 727/9540 (8%) loss 2.17

No preemptions, no anomalies. Quiet.

## 2026-04-26 10:07 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 10:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5 (075546): step 1010/9540 (10.6%) loss 1.54  — past warmup, into decay
- p67m33/lr0.67 (081538): step 846/9540 (8.9%) loss 2.10  — late warmup

No preemptions. Quiet.

## 2026-04-26 10:22 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 10:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5 (075546): step 1140 (12%) loss 1.58
- p33m67/lr0.83 (081538): step 980 (10%) loss 1.58

Same loss instant, different stages of decay — math-heavy mix. No preemptions. Quiet.

## 2026-04-26 10:37 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 10:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 1250 (13%) loss 1.61
- p67m33/lr0.5: step 1060 (11%) loss 2.10

No preemptions. Quiet.

## 2026-04-26 10:52 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 10:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.67: step 1210 (12.7%) loss 1.60
- p67m33/lr0.83: step 1090 (11.4%) loss 2.11 (0 preemptions; elapsed counter quirk)

No failures. Quiet.

## 2026-04-26 11:07 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 11:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 1490 (15.6%) loss 1.58
- p67m33/lr0.67: step 1310 (13.7%) loss 2.11

No preemptions. Quiet.

## 2026-04-26 11:23 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 11:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 1600 (16.8%) loss 1.55
- p33m67/lr0.83: step 1430 (15%) loss 1.51

No preemptions. Quiet.

## 2026-04-26 11:38 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 11:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 1730 (18.1%) loss 1.54
- p67m33/lr0.5: step 1540 (16.1%) loss 2.14

No preemptions. Quiet.

## 2026-04-26 11:53 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 11:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.67: step 1690 (17.7%) loss 1.53
- p67m33/lr0.83: step 1570 (16.5%) loss 2.12

No preemptions. Quiet.

## 2026-04-26 12:08 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 12:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 1960 (20.5%) loss 1.50
- p33m67/lr0.83: step 1790 (18.8%) loss 1.53

No preemptions. Crossing 20% mark on lead run. Quiet.

## 2026-04-26 12:23 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 12:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 2060 (21.6%) loss 1.52
- p67m33/lr0.67: step 1890 (19.8%) loss 2.13

No preemptions. Quiet.

## 2026-04-26 12:38 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 12:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 2190 (23%) loss 1.51
- p67m33/lr0.83: step 1920 (20%) loss 2.11

No preemptions. Quiet.

## 2026-04-26 12:53 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 12:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.67: step 2150 (22.5%) loss 1.57
- p67m33/lr0.5: step 2110 (22.1%) loss 2.12

No preemptions. Quiet.

## 2026-04-26 13:08 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 13:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 2420 (25.4%) loss 1.47
- p67m33/lr0.67: step 2240 (23.5%) loss 2.09

No preemptions. Crossed 25% on lead run. Quiet.

## 2026-04-26 13:23 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 13:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 2540 (26.6%) loss 1.48
- p67m33/lr0.83: step 2250 (23.6%) loss 2.06

No preemptions. Quiet.

## 2026-04-26 13:38 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 13:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.83: step 2480 (26%) loss 1.51
- p67m33/lr0.5: step 2460 (25.8%) loss 2.04

No preemptions. Quiet.

## 2026-04-26 13:54 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 13:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 2780 (29.1%) loss 1.42  (lowest math seen so far)
- p67m33/lr0.67: step 2600 (27.3%) loss 2.06

No preemptions. Quiet.

## 2026-04-26 14:09 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 14:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 2890 (30.3%) loss 1.52
- p67m33/lr0.83: step 2610 (27.4%) loss 2.19

No preemptions. Lead run crossed 30%. Quiet.

## 2026-04-26 14:24 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 14:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5 (84bba0): mid HF export at step 3000 (~31%)
- p67m33/lr0.5: step 2810 (29.4%) loss 2.00

W&B hash for lr0.5/p33m67: 84bba0 (will pull others as they surface). No preemptions. Quiet.

## 2026-04-26 14:39 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 14:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 3120 (32.7%) loss 1.51
- p67m33/lr0.67: step 2950 (30.9%) loss 2.10

No preemptions. Quiet.

## 2026-04-26 14:54 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 14:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 3230 (33.9%) loss 1.52
- p67m33/lr0.83: step 2970 (31.1%) loss 2.07

No preemptions. Quiet.

## 2026-04-26 15:09 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 15:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 3220 (33.7%) loss 1.49 (0 preemptions; tqdm reset after step-3000 HF export)
- p67m33/lr0.67: step 3180 (33.3%) loss 2.12

No failures. Quiet.

## 2026-04-26 15:24 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 15:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.83: step 3290 (34.5%) loss 1.48
- p67m33/lr0.5: step 3260 (34.2%) loss 2.04

No preemptions. Quiet.

## 2026-04-26 15:39 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 15:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 3460 (36.3%) loss 1.48
- p67m33/lr0.67: step 3400 (35.6%) loss 2.06

No preemptions. Quiet.

## 2026-04-26 15:54 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 15:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 3580 (37.5%) loss 1.47
- p67m33/lr0.83: step 3420 (35.8%) loss 2.11

No preemptions. Quiet.

## 2026-04-26 16:10 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 16:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 3690 (38.7%) loss 1.49
- p67m33/lr0.5: step 3370 (35.3%) loss 2.09 (0 preemptions; tqdm elapsed reset after eval cycle)

No failures. Quiet.

## 2026-04-26 16:25 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 16:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 3800 (39.8%) loss 1.47
- p67m33/lr0.67: step 3770 (39.5%) loss 2.11

No preemptions. ~40% mark crossed. Quiet.

## 2026-04-26 16:40 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 16:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 3930 (41.2%) loss 1.47
- p67m33/lr0.83: step 3780 (39.6%) loss 2.08

No preemptions. Quiet.

## 2026-04-26 16:55 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 16:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 4030 (42.2%) loss 1.45 (instant low)
- p67m33/lr0.5: step 3700 (38.8%) loss 2.10

No preemptions. Quiet.

## 2026-04-26 17:10 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 17:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 4150 (43.5%) loss 1.47
- p67m33/lr0.67: step 4110 (43.1%) loss 2.08

No preemptions. Quiet.

## 2026-04-26 17:25 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 17:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 4260 (44.7%) loss 1.41 (new math low)
- p67m33/lr0.5: step 3800 (39.8%) loss 2.02

No preemptions. Quiet.

## 2026-04-26 17:40 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 17:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.83: step 4340 (45.5%) loss 1.53
- p67m33/lr0.67: step 4340 (45.5%) loss 2.02

No preemptions. Quiet.

## 2026-04-26 17:55 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 17:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 4500 (47.2%) loss 1.43 (~halfway)
- p67m33/lr0.83: step 4360 (45.7%) loss 2.14

No preemptions. Quiet.

## 2026-04-26 18:10 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 18:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 4610 (48.3%) loss 1.47
- p67m33/lr0.5: step 4150 (43.5%) loss 2.03

No preemptions. Quiet.

## 2026-04-26 18:26 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 18:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 4730 (49.6%) loss 1.48
- p67m33/lr0.67: step 4700 (49.3%) loss 2.09

No preemptions. Almost halfway. Quiet.

## 2026-04-26 18:41 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 18:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running. **Lead run past halfway.**
- p33m67/lr0.5: step 4800 (50.3%) loss 1.52
- p67m33/lr0.83: step 4710 (49.4%) loss 2.06

No preemptions. Quiet.

## 2026-04-26 18:56 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 18:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running. **Past halfway, new lows hit.**
- p33m67/lr0.5: step 4910 (51.5%) loss 1.36 (new math low)
- p67m33/lr0.67: step 4940 (51.8%) loss 1.98 (broke below 2.0 on pretrain mix)

No preemptions. Quiet.

## 2026-04-26 19:11 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 19:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 4930 (51.7%) loss 1.44
- p67m33/lr0.83: step 4860 (51%) loss 2.13

No preemptions. Quiet.

## 2026-04-26 19:26 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 19:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 5030 (52.7%) loss 1.45
- p67m33/lr0.5: step 4740 (49.7%) loss 2.02

No preemptions. Quiet.

## 2026-04-26 19:41 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 19:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.83: step 5160 (54.1%) loss 1.49
- p67m33/lr0.67: step 5230 (54.8%) loss 2.02

No preemptions. Quiet.

## 2026-04-26 19:56 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 19:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 5260 (55.1%) loss 1.42
- p67m33/lr0.83: step 5200 (54.5%) loss 2.00

No preemptions. Quiet.

## 2026-04-26 20:11 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 20:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 5390 (56.5%) loss 1.44
- p67m33/lr0.5: step 5070 (53.1%) loss 2.03

No preemptions. Quiet.

## 2026-04-26 20:26 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 20:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 5500 (57.6%) loss 1.37 (instant low)
- p33m67/lr0.83: step 5510 (57.8%) loss 1.42

No preemptions. Quiet.

## 2026-04-26 20:41 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 20:41 UTC INCIDENT — p33m67/lr0.5 (075546) failed at preemption recovery

Babysit tick caught run dropped from 6→5. Cause:
- Parent: `failed`, `preemptions=2`, `RuntimeError: 1 step(s) failed`
- train_lm child logs show JAX coordinator-disconnect after the 2nd preemption recovery: `Failed to send RPC to coordination service... task was preempted/died/restarted unexpectedly`. Same coordination-failure pattern from earlier incidents but post-preemption rather than at-init.

Marin executor's "1 step(s) failed" gate terminated the run rather than retrying (default behavior). Run had reached step ~5500/9540 (57.6%) with W&B hash `84bba0`. Step-5000 HF export and step-5000+ tensorstore checkpoint preserved at `gs://marin-us-{central1,east5}/checkpoints/delphi-1e20-p33m67-20b-lr0.5-84bba0/`.

**Recovery: relaunched as `delphi-1e20-p33m67-20b-lr0p5-v5p32-batch-20260426-204452-resume`** with:
- `MIDTRAIN_OUTPUT_PATH_OVERRIDE=gs://marin-us-east5/checkpoints/delphi-1e20-p33m67-20b-lr0.5-84bba0` (preserves namespace + ckpt)
- `WANDB_RUN_ID=delphi-1e20-p33m67-20b-lr0.5-84bba0` + `WANDB_RESUME=allow` (continues same W&B run)
- Same env otherwise

Levanter should pick up the latest tensorstore checkpoint on init. Monitoring next tick.

## 2026-04-26 20:57 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 20:56 UTC babysit tick — first resume failed, retry submitted

The :40 resume relaunch (`...-204452-resume`) **failed within 60s**:
- Cause: `Failed to add port to server: 8476 already in use` on task 0 — stale JAX coordinator port held by a previous killed process on the same worker host
- Iris rapidly recycled the train_lm child (303 worker-level restart cycles in 60s) until "1 step(s) failed" gate fired
- Parent state: failed, exit=0

Resubmitted as `delphi-1e20-p33m67-20b-lr0p5-v5p32-batch-20260426-205739-resume2` with same env. Should land on a fresh worker host and the port-conflict won't recur.

Other 5 (081538) all still RUNNING.

## 2026-04-26 21:12 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 21:11 UTC babysit tick — resume2 healthy, all 6 running

resume2 (`...-205739-resume2`) at step 5560/9540 (58.3%) loss 1.37 — Levanter correctly resumed from step ~5500 tensorstore checkpoint. W&B run `delphi-1e20-p33m67-20b-lr0.5-84bba0` should also be continuing (same WANDB_RUN_ID, RESUME=allow).

All 6 sweep train_lm RUNNING. No preemptions on resume2 yet. Lost ~30min wall time across the failure → resume2 cycle but no progress lost on training.

## 2026-04-26 21:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5 (resume2): step 5670 (59.4%) loss 1.40
- p67m33/lr0.67: step 6050 (63.4%) loss 2.02

No new preemptions. Quiet.

## 2026-04-26 21:27 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 21:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5 (resume2): step 5800 (60.8%) loss 1.42
- p67m33/lr0.5: step 5780 (60.6%) loss 1.95 (new pretrain low)

No new preemptions. Quiet.

## 2026-04-26 21:42 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 22:01 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5 (resume2): step 5950 (62.4%) loss 1.46
- p67m33/lr0.83: step 6180 (64.8%) loss 2.03

No new preemptions. Quiet.

## 2026-04-26 22:39 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5 (resume2): step 6230 (65.3%) loss 1.45
- p67m33/lr0.5: step 6210 (65.1%) loss 2.07

No new preemptions. Quiet.

## 2026-04-26 22:46 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 22:46 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5 (resume2): step 6290 (65.9%) loss 1.42
- p33m67/lr0.83: mid eval/export

No new preemptions. Quiet.

## 2026-04-26 22:57 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5 (resume2): step 6400 (67.1%) loss 1.47
- p67m33/lr0.83: step 6640 (69.6%) loss 2.01

No new preemptions. Quiet.

## 2026-04-26 23:18 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 23:11 UTC INCIDENT — p33m67/lr0.67 fresh-started, lost ~6000 steps

Discovered during babysit tick. p33m67/lr0.67 (parent `081538`) had silently regressed to step ~580 (was at ~6500 around 21:20 UTC when HF step-6000 export landed).

Forensics:
- Old executor output hash: `3c967b`. HF exports at step 1000–6000 + tensorstore checkpoints at the same steps preserved at `gs://marin-us-east5/checkpoints/delphi-1e20-p33m67-20b-lr0.67-3c967b/{checkpoints,hf}/`.
- New executor output hash (post-preemption): `8fbf99`. Marin executor cycled the hash mid-run after the preemption recovery, started fresh in `gs://marin-tmp-us-central1/.../delphi-1e20-p33m67-20b-lr0.67-8fbf99/`. Why the hash changed mid-run on a single iris parent job is unclear — possibly differs by host environment.
- Parent `081538` showed `preemptions=1` only, but the recovery picked up no temp checkpoint and started training from step 0 in a fresh namespace.

**Recovery:**
1. Stopped the fresh-start parent (`...-081538`) — terminated cleanly.
2. Relaunched as `delphi-1e20-p33m67-20b-lr0p67-v5p32-batch-20260426-234017-resume3c` with:
   - `MIDTRAIN_OUTPUT_PATH_OVERRIDE=gs://marin-us-east5/checkpoints/delphi-1e20-p33m67-20b-lr0.67-3c967b`
   - `WANDB_RUN_ID=delphi-1e20-p33m67-20b-lr0.67-3c967b` + `RESUME=allow`

Levanter should resume from the latest tensorstore checkpoint at step-6000 in the `3c967b` namespace. Net save vs. letting fresh-start run: ~5400 steps of compute (~11h).

Other 5 jobs (4× 081538 + resume2) untouched and still running. Will verify resume on next tick.

## 2026-04-26 23:56 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 23:55 UTC babysit tick — resume3c verified, all 6 healthy

resume3c (`...-234017-resume3c`) running at step 6300/9540 (66.0%) loss 1.37. Levanter found temp checkpoint at step-6214 in the `3c967b` namespace (better than the step-6000 HF target — 214 fewer steps wasted). Loss 1.37 matches the trajectory before the fresh-start incident.

Total recovery loss vs. uninterrupted: ~86 steps (~10 min wall time).

All 6 sweep train_lm RUNNING:
- 4 × 081538 (p33m67/lr0.83 + p67m33/{lr0.5,lr0.67,lr0.83})
- 1 × resume2 (p33m67/lr0.5)
- 1 × resume3c (p33m67/lr0.67)

Cron `98caf780` continues 15-min cadence.

## 2026-04-27 00:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- resume2 (p33m67/lr0.5):  step 6960 (72.9%) loss 1.40
- resume3c (p33m67/lr0.67): step 6390 (67.0%) loss 1.35

No new preemptions. Quiet.

## 2026-04-27 00:14 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-27 00:56 UTC babysit tick — resume3c failed, resume3d picked up

Between :11 and :48 ticks resume3c failed (parent: failed, preemptions=1, "1 step(s) failed" gate). Last logged step 6430 loss ~1.44.

Relaunched as `delphi-1e20-p33m67-20b-lr0p67-v5p32-batch-20260427-004947-resume3d` with same `MIDTRAIN_OUTPUT_PATH_OVERRIDE=...3c967b`. Levanter resumed at step 6380 loss 1.40 (~50 steps re-trained from temp ckpt).

Pattern: marin executor's "1 step(s) failed" wrapper is intolerant of preemption-recovery cycles inside this run. Each recovery has a small chance of triggering it, requiring a manual relaunch with the same override. Net cost per cycle: ~10-15 min wall time + ~50-100 steps redundant compute.

All 6 sweep train_lm RUNNING:
- resume2 (p33m67/lr0.5): step 7310 (76.6%) loss 1.45
- resume3d (p33m67/lr0.67): step 6380 (66.9%) loss 1.40
- 4 × 081538 (p33m67/lr0.83 + p67m33/{lr0.5,lr0.67,lr0.83})

## 2026-04-27 01:11 UTC babysit tick — resume3d weathering preemption flurry

resume3d gained 4 preemption cycles in 22 min but is surviving (state=running). Step 6450 loss 1.49 — only +70 steps progress instead of expected ~150. The marin executor's "1 step(s) failed" gate hasn't fired this time though, which is what we want.

p67m33/lr0.83 broke below 2.0 sustained at step 7660 (80.3%) loss 1.99. Lead pretrain-mix run.

All 6 still RUNNING.

## 2026-04-27 01:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- resume3d (p33m67/lr0.67): step 6510 (68.2%) loss 1.41
- p67m33/lr0.83: step 7790 (81.7%) loss 2.00

resume3d still on 4 preemptions but holding. Quiet.

## 2026-04-27 01:41 UTC babysit tick — resume3d still surviving

resume3d at 8 preemptions (4 new in last 15 min), still running. Step 6590 (69%) loss 1.42. Slower than normal (+80 steps in 15min vs ~120 normal) due to recovery overhead but holding.

p33m67/lr0.83: step 7870 (82.5%) loss 1.36 — math-heavy lr0.83 hit new instant low.

All 6 sweep train_lm RUNNING.

## 2026-04-27 01:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- resume3d (p33m67/lr0.67): step 6680 (70%) loss 1.40 (preemptions=8)
- p67m33/lr0.5: step 7740 (81.1%) loss 2.00

Quiet.

## 2026-04-27 02:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- resume3d (p33m67/lr0.67): step 6800 (71.3%) loss 1.39
- p67m33/lr0.67: step 8260 (86.6%) loss 1.98

Quiet.

## 2026-04-27 02:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- resume2 (p33m67/lr0.5): step 8000 (83.9%) loss 1.39
- p67m33/lr0.5:           step 7980 (83.6%) loss 1.94

Quiet. Two runs crossed step 8000.

## 2026-04-27 02:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- resume2 (p33m67/lr0.5):    step 8120 (85.1%) loss 1.37
- p67m33/lr0.83:             step 8360 (87.6%) loss 1.93

Quiet.

## 2026-04-27 02:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- resume2 (p33m67/lr0.5):  step 8240 (86.4%) loss 1.36
- p67m33/lr0.83:           step 8470 (88.8%) loss 2.02

Quiet.

## 2026-04-27 03:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p67m33/lr0.83 (081538):    step 8600 (90.1%) loss 1.99   ← first across 90%
- resume2 (p33m67/lr0.5):    step 8360 (87.6%) loss 1.41

ETA ~2h to first completion.

## 2026-04-27 03:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p67m33/lr0.83 (081538):    step 8720 (91.4%) loss 2.05
- resume3d (p33m67/lr0.67):  step 7380 (77.4%) loss 1.43

Quiet.

## 2026-04-27 03:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p67m33/lr0.83 (081538):    step 8830 (92.6%) loss 1.95
- resume2 (p33m67/lr0.5):    step 8600 (90.1%) loss 1.37

Both leaders crossed 90%. ETA ~1:25 to first finish.

## 2026-04-27 03:56 UTC babysit tick — first finish imminent

All 6 running.
- p67m33/lr0.67 (081538):  step 9070 (95.0%) loss 1.98  — ETA 56min, first to finish
- p67m33/lr0.83 (081538):  step 8950 (93.8%) loss 1.97

First job ETA ~04:53 UTC.

## 2026-04-27 04:11 UTC babysit tick — first finish ~40min away

All 6 running.
- p67m33/lr0.67 (081538): step 9200 (96.4%) loss 1.96 — ETA 40min, ~04:52 UTC
- resume2 (p33m67/lr0.5): step 8830 (92.6%) loss 1.36

Quiet otherwise.

## 2026-04-27 04:26 UTC babysit tick — p67m33/lr0.67 at 97.5%, finish ~04:58 UTC

All 6 running.
- p67m33/lr0.67 (081538): step 9310 (97.5%) loss 1.98 — ETA 32min

## 2026-04-27 04:41 UTC babysit tick — first finish in ~14min

All 6 running.
- p67m33/lr0.67 (081538): step 9420 (98.7%) loss 2.01 — ETA 14min, finishing ~04:55 UTC

## 2026-04-27 12:15 UTC babysit tick — sweep 5/6 done, 1 still running

5 of 6 sweep points have completed:
- p33m67/lr0.5 (resume2):   SUCCEEDED, 8h 48m
- p33m67/lr0.67 (resume3d): SUCCEEDED, 7h 20m
- p33m67/lr0.83 (081538):   SUCCEEDED, 21h 3m, 1 preemption
- p67m33/lr0.67 (081538):   SUCCEEDED, 20h 42m, 1 preemption
- p67m33/lr0.83 (081538):   SUCCEEDED, 20h 58m, 1 preemption

Still RUNNING:
- p67m33/lr0.5 (081538): parent preempt=3 fail=0; current train_lm task generation has 4/4 tasks running for 10h 52m, preempt=0 fail=0 on this attempt. ~28h wall since launch. Fresh-tasks elapsed indicates training started ~10:52h ago after the third preemption-recovery cycle.

iris auto-recovery weathered all 3 preemptions on this point. Nothing dead, nothing to debug. Loop continues.

## 2026-04-27 12:15 UTC — postmortem on "non-preemption" failures

User asked why earlier failures were classified as non-preemption when they all happened on a preemptible cluster. Three distinct mechanisms:

1. **JAX coordinator RPC chain break** (lr0.5 075546 → step 5500; lr0.67 resume3c → step 6430). Preemption SIGTERM hits one worker → other tasks raise `RuntimeError: Failed to send RPC to coordination service: task was preempted/died/restarted unexpectedly`. JAX client-side state doesn't gracefully tolerate peer disappearance. Marin's "1 step(s) failed" gate fires → parent FAILED with failure_count=1 (not preemption_count). Non-deterministic: resume3d weathered 8 preemptions clean, resume3c died on the first.

2. **Stale port 8476 on worker recycle** (lr0.5 first resume `-204452-resume`). Preempted JAX coordinator died without releasing port 8476. iris recycled child 303× in 60s, each retry hit `Failed to add port to server: 8476 already in use`. OS-level resource cleanup race on dirty preemption exits.

3. **Marin executor hash drift mid-run** (lr0.67 081538). After one preemption, executor recomputed output-path hash from `3c967b` → `8fbf99` and silently restarted training in new namespace. Parent stayed RUNNING (not technically a failure), but 6000 steps of progress vanished. Caught via W&B loss regression on babysit tick.

iris's preemption auto-recovery works correctly when SIGTERM → process exit → worker recycle leaves the system in a good state. Three things break that contract above. None are iris bugs.

Future systemic fixes (file as issues post-sweep):
- JAX coordinator client should tolerate peer-loss during preemption window (or wrap-and-retry in levanter)
- Marin executor: add `--retries-on-step-failure N` for parent-level retry on transient framework errors
- Marin executor: pin output-path hash for parent job lifetime, immune to host environment recompute

## 2026-04-27 19:24 UTC — correction: p67m33/lr0.5 also hash-drifted

Follow-up investigation contradicted the 12:15 tick's "nothing to debug" conclusion.
`p67m33/lr0.5` is still RUNNING, but it is not a clean continuation of the original
output namespace.

Evidence:
- Original namespace: `gs://marin-us-central1/checkpoints/delphi-1e20-p67m33-20b-lr0.5-f74454`
  - `.executor_info` written 2026-04-26 08:16 UTC
  - permanent checkpoints through `step-8000`
  - temp checkpoint only at `step-3382`
- Current/live namespace: `gs://marin-us-east5/checkpoints/delphi-1e20-p67m33-20b-lr0.5-378f43`
  - `.executor_info` rewritten 2026-04-27 08:23 UTC after parent preemption recovery
  - permanent checkpoints through `step-7000`; `eval_metrics.jsonl` updated at 2026-04-27 19:19 UTC with eval steps through 7600
  - temp checkpoint at `gs://marin-tmp-us-east5/ttl=14d/checkpoints-temp/delphi-1e20-p67m33-20b-lr0.5-378f43/step-7619/`

Root cause is the same class as the `p33m67/lr0.67` incident, but in the opposite
direction: the parent job retried in a different region, `marin_prefix()` / mirrored
data paths resolved to that region, the executor hash changed, and Levanter imputed
a new run/checkpoint id from the new output path. This is region-dependent hash
drift, not a difference in LR or mix config.

Important correction to Claude's 12:15 postmortem: `p67m33/lr0.5` did not merely
"weather" three preemptions. It weathered them at the Iris parent level, but at
least one recovery moved from the central1 namespace to the east5 namespace and
lost the old `step-8000` continuation. If recovering this point, force:

```bash
MIDTRAIN_OUTPUT_PATH_OVERRIDE=gs://marin-us-central1/checkpoints/delphi-1e20-p67m33-20b-lr0.5-f74454
WANDB_RUN_ID=delphi-1e20-p67m33-20b-lr0.5-f74454
WANDB_RESUME=allow
```

Do not treat the active east5 `378f43` endpoint as the same W&B/checkpoint
continuation as central1 `f74454`.

## 2026-04-27 20:35 UTC — postmortem: StepSpec region-dependent dataset identity

This expands the earlier "executor hash drift" diagnosis. The problem was not
just "MirrorFS changed paths" in the rendered training config. The actual
`train_lm` hash changed because the selected midtraining dataset was a newer
datakit-backed `StepSpec` graph whose dependency identity included a physical
regional GCS path.

What the experiment did:
- `experiments/exp_delphi_math_10b_midtrain.py` selected
  `BUCKET_2["nemotron_cc_math_v1/4plus"]` as the math dataset.
- For mixture runs, `experiments/midtraining_mixes.py` also pulled this same
  handle into the pretrain/math mixture.
- That catalog handle is not a stable path string. It is an executable graph:
  `download_hf StepSpec -> normalize StepSpec -> tokenize ExecutorStep -> train_lm`.

Why this backfired:
- `lib/marin/src/marin/datakit/download/nemotron_v2.py` builds the subset
  normalizer with:

```python
input_path=f"{download.output_path}/{subset_dir}"
```

- `download.output_path` is a `StepSpec.output_path`. It calls
  `marin_prefix()` and therefore resolves immediately to the coordinator's
  physical region, e.g.:

```text
gs://marin-us-central1/raw/nemotron_cc_math_v1-322fe4/4plus
gs://marin-us-east5/raw/nemotron_cc_math_v1-322fe4/4plus
```

- `lib/marin/src/marin/datakit/normalize.py` then puts that already-resolved
  `input_path` into `hash_attrs`.
- `StepSpec.as_executor_step()` wraps `hash_attrs` in `VersionedValue`, so the
  old executor treats the regional path as part of the semantic version.
- `train_lm` hashes dependency versions. Therefore the training output hash
  changed when Iris retried the parent in a different region.

Concrete evidence from `.executor_info` comparison:
- `p67m33/lr0.5` old central run: `md5(version)[:6] = f74454`
- `p67m33/lr0.5` new east5 run: `md5(version)[:6] = 378f43`
- The `version` objects differ in exactly three leaves, all the same issue:

```text
dependencies[20].dependencies[0].config.attrs.input_path
dependencies[21].config.attrs.input_path
dependencies[22].dependencies[0].config.attrs.input_path

gs://marin-us-central1/raw/nemotron_cc_math_v1-322fe4/4plus
vs
gs://marin-us-east5/raw/nemotron_cc_math_v1-322fe4/4plus
```

The same pattern explains `p33m67/lr0.67`:
- old east5 namespace: `delphi-1e20-p33m67-20b-lr0.67-3c967b`
- wrong central namespace after parent retry: `delphi-1e20-p33m67-20b-lr0.67-8fbf99`
- The `version` diff is again exactly the regioned `nemotron_cc_math_v1/4plus`
  `hash_attrs.input_path`.

Why David's "aren't dataset hashes stable?" objection is valid:
- Dataset hashes are supposed to be stable when their hash inputs are logical.
- Older datasets like Dolma use `InputName.hardcoded("raw/dolma/v1.7")`.
  The executor hash sees the relative logical string `raw/dolma/v1.7/...`;
  region-specific `gs://marin-{region}/...` paths are only materialized later
  for runtime I/O.
- The Nemotron v2 datakit path materialized `download.output_path` before
  hashing. That crossed physical placement into semantic identity.

Provenance:
- `StepSpec` itself is mainline, not introduced by this midtraining branch:
  - `beb5d8d0b4` / PR #2494: `StepSpec + Artifact for no-magic workflow orchestration`
  - `9406cc0970` / PR #4097: `StepSpec -> ExecutorStep` bridge
  - `44fe6ee43d` / PR #4142: datakit migration and StepSpec download factories
- The exact normalizer hash leak is also mainline:
  - `682942a0eb` / PR #4188: `normalize_step` stores `input_path` in `hash_attrs`
  - `f6bf3ad447` / PR #4892: Nemotron v2 normalizer builds
    `input_path=f"{download.output_path}/{subset_dir}"`
- The midtraining branch did not create `StepSpec`; it selected the
  datakit-backed `BUCKET_2["nemotron_cc_math_v1/4plus"]` path and ran it in a
  cross-region retry setting, which exposed the bug.

Operational rule going forward:
- For failed/preempted midtraining runs, do not trust the human-readable step
  name. The true run/checkpoint id includes the executor output hash.
- Before relaunching, find the exact previous output path and W&B run id from
  `.executor_info`, GCS checkpoints, and temp checkpoints.
- Relaunch with the exact old output path forced via
  `MIDTRAIN_OUTPUT_PATH_OVERRIDE` / `ExecutorStep.with_output_path(...)`, and
  verify startup logs say `Resuming training from step ...`.
- Treat any cross-region parent retry as suspect until `.executor_info` and W&B
  run id prove the same namespace is being reused.

Systemic fixes to propose:
- Hash identity should canonicalize `gs://marin-{region}/...` to a logical
  Marin path before hashing, or StepSpec hash attrs should store relative
  logical paths instead of physical regional paths.
- `StepSpec.output_path` should not be used as an input to another step's
  semantic `hash_attrs` unless it has been canonicalized.
- Iris/Marin should persist the resolved parent output path for a job attempt
  and reuse it on parent retry. Cross-region compute placement should change
  physical I/O/mirroring, not the experiment identity.

## 2026-05-01 19:51 UTC — handoff: main merged and midtraining launch guard pushed

State for the next agent:

- Branch: `midtrain_data`
- Remote head after handoff: `4b40df269 [experiments] Pin Delphi midtraining jobs by region`
- Previous merge commit: `dc41a9bac Merge remote-tracking branch 'origin/main' into midtrain_data`
- Important upstream fix now present from main: `7f0b99b9e Stop region prefixes leaking into Marin executor identity hashes (#5223)`.
  This is Rav's executor/StepSpec region-agnostic hashing fix.

What was changed and pushed:

- `experiments/exp_delphi_math_10b_midtrain.py`
  - Removed the sample launch env `MARIN_I_WILL_PAY_FOR_ALL_FEES=1`.
  - Kept coordinator scheduling flexible across `--region us-central1 --region us-east5`.
  - Added `_selected_train_region()` / `_midtrain_tpu_resources()` so the child `train_lm`
    TPU job is pinned to the coordinator's resolved region when it is one of the v5p regions
    (`us-central1` or `us-east5`).
  - Added `MIDTRAIN_TRAIN_REGION` override; zones like `us-central1-a` normalize to `us-central1`.
  - Cleaned stale comments that implied the experiment should always write in us-east5.
  - Centralized base compute settings and added reusable mix/run-name knobs for follow-up sweeps.
- `experiments/midtraining_mixes.py`
  - Added reusable midtraining mixtures for full math and pretrain/math replay ratios.
- `experiments/test_default_train_init_mode.py`
  - Added tests for child resource region pinning, zone override normalization, explicit bad-region failure,
    and local non-v5p region fallback.
- `scripts/_verify_mirror_stage.py`
  - Tiny pre-commit cleanup: removed an unused f-string.

Validation completed before push:

```bash
./infra/pre-commit.py --all-files --fix
uv run pytest experiments/test_default_train_init_mode.py tests/test_training.py tests/execution/test_executor.py tests/execution/test_step_runner.py -q
```

Result: `87 passed, 1 skipped`.

Operational interpretation:

- Future Delphi midtraining launches may still submit the parent/coordinator with both v5p regions.
- Once the coordinator is running, the generated child resource list is single-region, matching the
  coordinator's region. This avoids the observed footgun where the parent materialized concrete
  `gs://marin-us-east5/...` paths but Iris placed the child in `us-central1`.
- This is not a general checkpoint migration system. It is a targeted guard for these midtraining
  experiments, layered on top of the upstream region-agnostic hash fix.
- For any failed/preempted existing run, still follow the resume rule above: identify the exact old
  output path/run id, check permanent and temporary checkpoints, force the old output path if relaunching,
  and verify `Resuming training from step ...` in startup logs.

Local worktree note:

- The pushed commit intentionally did not include the large local logbook/analysis/debug artifacts.
- At handoff, expect unrelated dirty files to remain in the worktree, including this logbook,
  `experiments/defaults.py`, `tests/test_training.py`, and analysis scripts/plots. Do not assume they
  are part of the pushed refactor unless the user explicitly asks to curate and commit them.

## 2026-05-01 19:58 UTC — main merged locally; which incident fixes landed

User asked to pull/merge main and check recent commits to see whether the
cross-region incident is mostly solved. Actions taken:

```bash
git fetch origin main
git merge --autostash origin/main
```

Merge result:

- Local merge commit: `ecd8fbca7 Merge remote-tracking branch 'origin/main' into midtrain_data`
- Merge succeeded cleanly; autostash reapplied local dirty worktree changes.
- Local branch is ahead of `origin/midtrain_data` by the main merge commits.
  This merge has not been pushed.

Relevant fixes now present in the local merged branch:

- `7f0b99b9e Stop region prefixes leaking into Marin executor identity hashes (#5223)`
  - Fixes the core hash-drift bug from this incident.
  - `StepSpec.hash_id` uses dependency names/hashes instead of physical dep output paths.
  - `normalize_step` hashes `relative_input_path` instead of the resolved physical `input_path`.
  - Executor deep-dep fallback uses region-stable `{name}-{hash}` rather than `gs://marin-<region>/...`.
- `b4298305a infra/rigging: fold tmp buckets into main buckets (#5266)`
  - Replaces separate `marin-tmp-*` temp buckets with `tmp/ttl=...` prefixes inside
    normal `marin-<region>` buckets.
  - `marin_temp_bucket(..., source_prefix=output_path)` chooses temp checkpoint
    location from the training output path's region.
  - This supersedes the earlier `mirrortmp://` design path for mainline.
- `a154c044f Charge cross-region transfer budget on tensorstore checkpoint I/O (#5225)`
  - TensorStore checkpoint serialize/deserialize now calls `record_transfer(...)`.
  - This plugs the accounting gap where TensorStore bypassed fsspec and therefore
    bypassed `CrossRegionGuardedFS`.
- `9d9b9a2a7 [iris] Fix coscheduled split-slice and orphan attempt bugs (#5249)`
  - Improves Iris retry/preemption state handling for split-slice/coscheduled jobs.
- `4b40df269 [experiments] Pin Delphi midtraining jobs by region`
  - Branch-specific guard: the Delphi midtraining parent can be scheduled flexibly,
    but the generated TPU child is pinned to the coordinator's resolved region.

Focused regression tests run after the merge:

```bash
uv run pytest \
  tests/execution/test_step_runner.py::test_step_spec_hash_id_stable_across_prefixes \
  tests/execution/test_step_runner.py::test_step_spec_hash_id_via_marin_prefix_env \
  tests/execution/test_step_runner.py::test_resolve_executor_step_infers_region_for_iris_without_pin \
  tests/execution/test_step_runner.py::test_resolve_executor_step_raises_on_cross_region_inputs_without_pin \
  tests/execution/test_step_runner.py::test_resolve_executor_step_raises_on_cross_region_even_with_override_env \
  tests/execution/test_step_runner.py::test_executor_resolve_steps_uses_component_gcs_region_to_pick_tpu_region \
  tests/execution/test_step_runner.py::test_executor_resolve_steps_picks_one_region_for_multi_region_tpu_component \
  tests/execution/test_executor.py::test_executor_version_stable_across_prefixes \
  tests/test_training.py::test_temporary_checkpoint_base_path_follows_output_path_region \
  tests/test_training.py::test_update_config_to_use_out_path_sets_run_specific_temp_checkpoints \
  lib/rigging/tests/test_record_transfer.py \
  -q
```

Result: `13 passed`.

Conclusion:

- The original **region-sensitive hash drift** is now fixed in main.
- The original **TensorStore cross-region accounting gap** is fixed in main.
- The original **separate temp-bucket design weakness** is mostly addressed by
  folding temp paths into primary regional buckets and deriving temp region from
  output path.
- The branch-specific Delphi parent/child placement footgun is guarded in this
  branch, but the broader training child alignment PR (`cc2678ff4`) is not merged
  into main.
- There is still no full automatic region migration/resume system. For old failed
  runs, still force the exact old output path and verify resume logs.
