# ARTIFACT_PORT.md — Porting the DPO/LoRA experiments to main's ArtifactSteps executor

> **Mission for the next agent:** make the ~160 DPO/LoRA experiment scripts on
> this branch (`dpo-lora`, pushed to remote `dpo-lora-artificat`) *runnable*
> against **current `main`**, whose execution framework was rewritten out from
> under them. Keep all of main's library logic; adapt only the experiment
> scripts and their private helpers.

---

## 1. TL;DR

- This branch = **current `main`** (library/infra/login all current) **+** the full
  body of DPO/LoRA experiment work (160 scripts) preserved as **reference
  artifacts**.
- The scripts **do not run** today. They target the *old* Marin Executor
  (`marin.execution.executor.executor_main`) and old helper modules
  (`experiments/defaults.py::default_dpo`), which `main` **deleted** in
  **#6649 "Replace the Executor with lazy ArtifactSteps"** (`f7f5535c39`).
- The job is a **framework port**, not import fixups: rewrite the scripts to the
  new ArtifactSteps model, plus a handful of symbol renames.
- **Do not** modify anything under `lib/` (levanter/haliax/marin). Those are
  main's canonical, upstreamed versions. Only touch `experiments/`.

---

## 2. How we got here (essential context)

| Commit | What it did |
|---|---|
| `2ff3dea28f` (#2460, "First Attempt at DPO") | Added `default_dpo` to `experiments/defaults.py`. |
| `e7c8d06367` (#4637 head, tag `pr4637-head`) | Upstreamed the DPO/LoRA **library** logic (levanter `train_dpo`, `adaptor`, `dpo`, preference data). This is the merged PR. |
| `f7f5535c39` (#6649, "Replace the Executor with lazy ArtifactSteps") | **Deleted the old Executor and `experiments/defaults.py` (850 lines, incl. `default_dpo`/`default_tokenize`).** This is why the scripts broke. |
| `36fd7f894a` (this branch) | Merged `main` into the dpo-lora worktree: library/framework = main's current; DPO experiments kept. |
| `76e5126258` (this branch) | Restored `experiments/defaults.py`, `marin_models.py`, `paloma.py` from pre-merge so imports *resolve* (but still target the old API). |

Useful refs already in the repo:
- Tag **`dpo-lora-backup-premerge`** (`5182ab1f3c`) — the original worktree
  (1143 commits behind main) with everything against the old framework. Use it
  to see how a script was *originally* wired.
- Tag **`pr4637-head`** (`e7c8d06`) — the original merged PR (5 experiment files).

---

## 3. Current branch layout

DPO/LoRA experiment scripts (160 files), all currently old-API:
- `experiments/posttrain/per_stmt_dpo/` — ~96 files: per-statement DPO, LoRA-vs-full-FT
  sweeps, LR sweeps (`exp1a/1b/2a/2b_lr*_s*.py`), and a large `debug_*`/`experiment_*`
  series from a v5p-8 LoRA bf16 all-reduce numerics investigation.
- `experiments/posttrain/full_dpo/` — full-finetune DPO (`v6e32`, `v6e64`).
- `experiments/sweep_dpo/`, `experiments/tune_lora/` — parameter sweeps.
- `experiments/dpo_bloom_speceval*.py`, `experiments/dpo_ultrafeedback.py`,
  `experiments/eval_dpo.py`, `experiments/test_dpo_generation_config.py`.

Restored old helpers the scripts import (also old-API — port or replace these):
- `experiments/defaults.py` — `default_dpo` (@663), `default_tokenize` (@261).
- `experiments/marin_models.py` — `marin_tokenizer`.
- `experiments/paloma.py`.

DPO configs / docs kept (data, not code — usually fine): `lib/levanter/config/dpo/*.yaml`,
`lib/levanter/docs/guides/{DPO,LoRA-DPO}-Training.md`, `dpo_bloom_plots/`,
`.agents/logbooks/`, `docs/debug-log-*dpo*.md`.

---

## 4. The core problem: old Executor → ArtifactSteps

Every leaf script ends with the old entrypoint:

```python
from marin.execution.executor import executor_main
...
if __name__ == "__main__":
    executor_main(steps=[...])
```

`executor_main` and the `ExecutorStep` graph API were **replaced** by the lazy
**ArtifactSteps** model in #6649. **Before writing any code**, the next agent
must learn the new model:

1. Read **`experiments/AGENTS.md`** on `main` — it documents how experiments are
   authored now.
2. Read the diff of **#6649** (`git show f7f5535c39 -- lib/marin/src/marin/execution/`)
   to see exactly what `executor_main`/`ExecutorStep` became.
3. Open a **currently-working `main` experiment** end-to-end as a template
   (find one: `git grep -l "ArtifactStep\|def main(" main -- experiments/` — pick a
   training experiment, not a datakit op). Mirror its step-declaration and
   run-entry pattern.

---

## 5. Concrete symbol map (old → new)

| Old (this branch, broken) | New (current main) | Notes |
|---|---|---|
| `from marin.execution.executor import executor_main` | **ArtifactSteps API** (#6649) | The real work. Study `experiments/AGENTS.md` + a live experiment. |
| `from experiments.defaults import default_dpo, default_tokenize` | **no drop-in** — removed with the Executor | Reimplement as small DPO-scoped helpers against ArtifactSteps (see §6). `default_train` logic now lives in `experiments/simple_train_config.py`. |
| `from experiments.marin_models import marin_tokenizer` | `from experiments.marin_tokenizer import marin_tokenizer` | Pure module rename (main renamed the file). |
| `from levanter.adaptation import LoraAdaptationConfig` | `from levanter.adaptor import LoraAdaptorConfig` | `LoraAdaptorConfig(LoraConfig, AdaptorConfig)` in `lib/levanter/src/levanter/adaptor/__init__.py:161`; LoRA fields (`r`, `alpha`, `target_modules`) in `adaptor/lora.py::LoraConfig` (@105). Export fields (`peft_save_path`, `merged_hf_save_path`, `hf_upload`, …) moved onto `AdaptorConfig`/`AdaptorExportConfig` — re-check field names. |
| `from levanter.data.text import PreferenceChatLmDatasetFormat` | `from levanter.data.text.preference import PreferenceChatLmDatasetFormat` | Class exists; just **not re-exported** from the package `__init__`. |
| `from marin.processing.tokenize import lm_data_config` | `from marin.processing.tokenize import lm_mixture_data_config` | Renamed. Confirm the signature matches the single-source call sites. |
| `experiments.simple_dpo_config.DPO_EVAL_PARALLELISM` | **removed** | Old value was a `dict[str, int]`. Reintroduce the constant in the DPO helper module or inline at the 2 call sites (grab old value from `dpo-lora-backup-premerge:experiments/simple_dpo_config.py:18`). |
| `from experiments.paloma import ...` | `from experiments.datasets.paloma import ...` | main moved paloma into the dataset registry. Only `defaults.py` imports it. |

**Confirmed still valid on main (imports OK, but check for signature drift):**
`levanter.dpo.ReferenceEvalCacheConfig`, `levanter.main.train_dpo.{AdapterBaseReferenceConfig, SeparateReferenceConfig}`,
`levanter.utils.mesh.MeshConfig`, `experiments.llama.{llama_8b, LLAMA3_CHAT_STOP_TOKEN_IDS}`,
`experiments.models.llama_3_1_8b`, `experiments.simple_dpo_config.SimpleDPOConfig`,
`fray.cluster.ResourceConfig`.

---

## 6. Recommended strategy (do it in this order)

**The linchpin:** most leaf scripts are thin. ~60 do
`from experiments.posttrain.per_stmt_dpo.common import (...)` and ~15 just call
`run_exp1a/1b/2a/2b(...)` from that `common.py`. **Porting `common.py` fixes the
bulk of the fan-out.** Same for `experiments/posttrain/full_dpo/common.py`.

1. **Learn the framework** (§4). Don't skip — everything depends on it.
2. **Port the shared helpers first.** Create a DPO-scoped helper module (prefer
   `experiments/posttrain/dpo_defaults.py` over resurrecting the 893-line
   monolith `experiments/defaults.py`): reimplement `default_dpo` /
   `default_tokenize` as ArtifactSteps builders, and add `DPO_EVAL_PARALLELISM`.
   Then delete the restored `experiments/{defaults,marin_models,paloma}.py` once
   nothing imports them. Fix the §5 renames everywhere with a scripted sweep.
3. **Port `per_stmt_dpo/common.py` and `full_dpo/common.py`** to build ArtifactSteps
   and expose the same `run_expNx(...)` entrypoints the leaves expect.
4. **Port one representative leaf end-to-end** (e.g.
   `experiments/posttrain/per_stmt_dpo/exp1a_lr1e6_s35.py`) and get it to build
   its step graph without launching anything (§7).
5. **Sweep the remaining leaves** — mostly mechanical once `common.py` is done.
   Prioritize the `exp1a/1b/2a/2b` sweeps and `full_dpo`; the `debug_*` /
   `experiment_[a-z]*` series are from a finished numerics investigation — port
   them last or mark as archived if low value.
6. **Verify** (§7), then lint (`./infra/pre-commit.py --all-files --fix`) and typecheck
   (`uv run pyrefly`).

---

## 7. Verification (do NOT burn TPUs)

"Runnable" = imports resolve, config/step-graph builds, dry-run planning works.
Do not launch large multi-region TPU jobs to prove the port.

- **Import smoke:** `uv run python -c "import experiments.posttrain.per_stmt_dpo.exp1a_lr1e6_s35"`
  should succeed (top-level config construction runs on import).
- **Step-graph dry run:** whatever the new ArtifactSteps API's plan/dry-run
  entrypoint is (learn it in §4) — build the graph without executing.
- **Tiny smoke only if needed:** the smallest config (a `*_s2`/`*_s10` on `v5p-8`
  or CPU) for a couple of steps. Iris/TPU is free for this project, but keep it
  minimal and single-region for a smoke.
- Follow root `TESTING.md` + `lib/*/AGENTS.md` for any tests you add.

---

## 8. Guardrails

- **Never touch `lib/`** (levanter/haliax/marin). It is main's upstreamed logic.
  If a script needs a library change, that's a signal you've mis-mapped an API —
  re-check §5 against current main, don't patch the library.
- **Stay current with main** — the whole point of the merge was working
  login/infra. Do not resurrect the old Executor or old `lib/` versions.
- **No backward-compat shims** (house rule): update call sites, don't wrap.
- Follow `AGENTS.md` house style; `agent-generated` label on any PR; agent PR/issue
  *comments* start with 🤖 (never in commit messages or PR bodies).

---

## 9. Handy commands

```bash
WT=.  # run from the dpo-lora worktree

# See a script's original old-framework wiring
git show dpo-lora-backup-premerge:experiments/posttrain/per_stmt_dpo/common.py

# What #6649 did to the executor
git show f7f5535c39 -- lib/marin/src/marin/execution/

# Find a live main experiment to mimic
git grep -l "def main(" main -- experiments/ | grep -v datakit | head

# Recover the old DPO_EVAL_PARALLELISM value
git show dpo-lora-backup-premerge:experiments/simple_dpo_config.py | sed -n '15,30p'

# All old-API entrypoints still to port
git grep -l "executor_main" -- experiments/ | wc -l

# Sweep a rename across the scripts (example)
git grep -l "from experiments.marin_models import marin_tokenizer" -- experiments/ \
  | xargs sed -i 's/from experiments.marin_models import/from experiments.marin_tokenizer import/'
```

## 10. Definition of done

- Every `experiments/` DPO/LoRA script imports cleanly and builds its
  ArtifactSteps graph against current `main`.
- No references to `executor_main`, `experiments.defaults`, `experiments.marin_models`,
  `levanter.adaptation`, or `lm_data_config` remain under `experiments/`.
- One representative script verified running a minimal smoke.
- `./infra/pre-commit.py --all-files --fix` and `uv run pyrefly` pass.
- Nothing under `lib/` changed.
