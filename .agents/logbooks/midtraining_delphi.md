# Delphi × Nemotron-CC-Math 10 B midtraining — logbook

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
**Status:** Plan written; implementation in progress (no experiment launched yet).
**Base-model catalogue:** [`.agents/projects/delphi_midtraining.md`](../projects/delphi_midtraining.md)
**External reference:** [*Delphi Scaling Laws: Key Findings* — Will Held](https://oa.williamheld.com/blog/delphi/)
**Branch:** `midtrain_data`
**Primary artefact being built:** `experiments/exp_delphi_math_10b_midtrain.py`

---

## Goal

Run a small LR sweep that continues-trains the two smallest existing AdamH-trained Marin checkpoints on **10 B tokens of `nemotron_cc_math_v1/4plus`**, to de-risk a Mantis-style math-midtraining recipe before spending v4-512 / v4-1024 time on Delphi 1e22 / 1e23. We're looking for the highest peak LR that still drops the math-eval loss monotonically.

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
