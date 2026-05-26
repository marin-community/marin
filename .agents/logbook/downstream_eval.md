# Downstream Eval Logbook

**Branch:** `aa_fork_rk_downstream`
**Forked from:** `rohith-working-agent` at `a93fcf4cca` (`downstream scaling stuff`)
**Started:** 2026-05-13
**Owner:** Ahmed
**Initial draft:** Claude, after the fork
**Expanded:** 2026-05-13

## TL;DR

This branch is a fork of Rohith's `experiments/downstream_scaling/` work. The
package adds a small downstream-eval framework with three artifacts:
`prompts.jsonl.gz`, `completions.jsonl.gz`, and `grades.jsonl.gz`.

The only launcher currently present is IID sampling on masked GSM8K across the
Delphi isoflop ladder:

- 11 mask fractions: `0.0, 0.1, ..., 1.0`
- 10 Delphi checkpoints: `3e18` through `1e23`
- 256 GSM8K test problems per condition
- 32 samples per problem
- 8192 completion requests per model/mask condition
- 110 final grade targets, backed by 110 completion steps and 11 prompt steps

Artifacts do exist. They are under `gs://marin-us-east5`, not the current local
`MARIN_PREFIX=gs://marin-us-central2`. A status scan on 2026-05-13 found:

- Prompt status files in `gs://marin-us-east5/downstream_scaling/evals/prompts/`:
  `12 SUCCESS` total, including the 11 masked-GSM8K prompt versions plus one
  dummy prompt artifact.
- Completion/grade status files under
  `gs://marin-us-east5/downstream_scaling/evals/delphi/masked_gsm8k/iid/`:
  `134 SUCCESS`, `39 FAILED`, `12 RUNNING`.
- Expected completion/grade outputs for the launcher: 220. The scan found 185
  `.executor_status` files, so 35 completion/grade outputs had no status file
  at scan time.

Treat `RUNNING` status files as executor state, not proof of live jobs. Check
Iris before relaunching or cleaning anything up.

## Scope

The code in this branch does two things:

1. Defines an eval artifact contract and composition API under
   `experiments/downstream_scaling/evals/framework/`.
2. Provides tasks and completion algorithms that can be composed into Marin
   `ExecutorStep`s.

The code present is broader than the launched experiment. `mask.py` and
`rerank.py` exist as completion algorithms, and `math500.py` exists as a task,
but the only launcher in this package is
`experiments/downstream_scaling/evals/run_delphi_masked_gsm8k_iid.py`.

## Research Question

The launched experiment asks how Delphi-scale models use a corrupted GSM8K
solution hint.

The masked GSM8K task masks target-solution hints as well as few-shot material.
For each few-shot example, it writes:

1. the question,
2. a token-masked version of the solution prefix,
3. then the full original target solution.

For the target problem, it writes the question and a token-masked version of
the target solution prefix, then asks the model to continue. The mask fraction
therefore controls how much of the target solution hint is visible, from no
masking at `mask_00` to all hint tokens replaced by `<mask>` at `mask_10`.

The natural metrics are per-completion GSM8K exact match, per-condition mean
accuracy, and pass@k-style curves over 32 samples. The current code writes
per-completion grade rows only. There is no reducer or plotting script in
`experiments/downstream_scaling/` yet.

The broader framework seems intended to support comparing IID, mask-hint, and
rerank decoding, but only IID has a launcher here.

## Code Map

### Guide And Framework

- `experiments/downstream_scaling/evals/GUIDE.md`
  - Canonical design note.
  - Defines the three-stage graph:

    ```text
    task -> prompts.jsonl.gz
    model + prompts + completion algorithm -> completions.jsonl.gz
    task + prompts + completions -> grades.jsonl.gz
    ```

- `experiments/downstream_scaling/evals/framework/core.py`
  - Defines the `EvalTask` and `CompletionAlgorithm` protocols.
  - `make_eval_step(...)` creates prompt, completion, and grade steps and
    returns the final grade step.
  - `task.make_prompts_step()` is intentionally stable across model evals so
    the same task config reuses prompt artifacts.

- `experiments/downstream_scaling/evals/framework/schema.py`
  - Defines `PromptRow`, `CompletionRow`, and `GradeRow` as `TypedDict`s.
  - Provides gzip JSONL readers.
  - Readers validate shape and reject duplicate `id`s.
  - Public artifact filenames are constants:
    `prompts.jsonl.gz`, `completions.jsonl.gz`, `grades.jsonl.gz`.

- `experiments/downstream_scaling/evals/utils.py`
  - `version_path(...)` wraps raw string paths with `versioned(...)`.
  - `InputName` and `MirroredValue` inputs pass through unchanged.

### Tasks

| File | Purpose | Notes |
| --- | --- | --- |
| `evals/tasks/gsm8k.py` | Standard GSM8K prompt and grade task | Uses `lm_eval.tasks.get_task_dict(["gsm8k"])`; grading applies the task filters and uses `flexible_extract` when present. |
| `evals/tasks/gsm8k_masked.py` | Masked-solution GSM8K prompt task | Uses `load_tokenizer(tokenizer_path)`, token-level masking, and reuses `GSM8KTask.make_grade_step`. |
| `evals/tasks/math500.py` | MATH-500 task | Uses `HuggingFaceH4/MATH-500` and `safe_grade(..., grader="sympy")`; default prompt prefix includes the strawberry `\boxed{3}` example. |
| `evals/tasks/dummy.py` | Smoke-test task | Writes simple prompts and always-correct grades. |

Important task behavior:

- `GSM8KTaskConfig.grade_workers` is operational and not versioned.
- `MaskedGSM8KTaskConfig.mask_fraction`, `mask_text`, `num_fewshot`,
  `fewshot_seed`, `n_problems`, and `tokenizer_path` affect prompt identity and
  are versioned.
- `MaskedGSM8KTask.make_grade_step = GSM8KTask.make_grade_step` works because
  `MaskedGSM8KTaskConfig` has `grade_workers`.
- `Math500TaskConfig.prompt_prefix` and `prompt_suffix` are versioned. The
  strawberry prefix may be intentional priming, but that intent is not recorded.

### Completion Algorithms

| File | Purpose | Launcher usage |
| --- | --- | --- |
| `evals/algorithms/iid.py` | IID sampling in one remote TPU step, with chunk files and a Zephyr aggregation pass | Used by `run_delphi_masked_gsm8k_iid.py`. |
| `evals/algorithms/iid_zephyr.py` | IID sampling sharded through Zephyr workers | Present, not used by the launcher. |
| `evals/algorithms/mask.py` | Hint generation, hint grading, mask-prompt construction, final IID sampling | Present, no launcher in this package. |
| `evals/algorithms/rerank.py` | Proposal generation plus scoring-model rerank decode | Present, no launcher in this package. |

IID details:

- `IIDSamplingConfig` is semantic and versioned: `n_samples`,
  `temperature`, `top_p`, `top_k`, `max_tokens`, `seed`, `stop`.
- `IIDExecutionConfig.num_workers` and `worker_resources` are operational.
- `chunk_size` is versioned because it determines chunk layout and resume
  namespace.
- For the launched GSM8K config, each completion step has
  `256 * 32 / 512 = 16` chunk files.
- Chunks are written to
  `chunks/chunk_size=512/chunk-000000.jsonl.gz` etc., with matching `.SUCCESS`
  markers.
- The final completion artifact is assembled by a Zephyr `group_by`, sorted by
  internal `completion_index`.

TPU/vLLM details:

- `iid.py` and `iid_zephyr.py` set:
  - `MARIN_VLLM_MODE=native`
  - `VLLM_ENABLE_V1_MULTIPROCESSING=0`
  - `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`
  - `VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION=1`
  - `VLLM_TPU_SKIP_PRECOMPILE=1`
- `iid.py` and `iid_zephyr.py` patch
  `tpu_inference.kernels.ragged_paged_attention.v3.kernel.get_tuned_block_sizes`
  to return `max(1, bkv_p // 2)`. The root cause is not documented.
- `mask.py` inherits the IID behavior for hint and final completion substeps.
- `rerank.py` imports the vLLM env var block, but does not apply the IID
  ragged-paged-attention patch in this file.

Rerank-specific details:

- `rerank.py` uses a proposal model and a scoring model.
- Scoring is backed by `experiments.rerank_decode.scorer.VLLMLogprobScorerTPU`.
- The scorer uses custom collective RPC methods such as `score_suffixes` and
  `accept_scoring_suffix`.
- `experiments/rerank_decode/TODO.md` says the original vLLM logprob scorer was
  too slow because it recomputed full prefixes. Treat the rerank path as
  experimental until it has a dedicated smoke run.

## Model Registry

`experiments/downstream_scaling/models/delphi.py` defines
`DELPHI_CHECKPOINTS`:

| Key | Relative path family |
| --- | --- |
| `3e18` | `checkpoints/isoflop/isoflop-3e+18-d1024-L11-B8-adamh_scaling_v6/hf` |
| `9e18` | `checkpoints/isoflop/isoflop-9e+18-d1152-L12-B16-adamh_scaling_v6/hf` |
| `2e19` | `checkpoints/isoflop/isoflop-2e+19-d1408-L15-B16-adamh_scaling_v6/hf` |
| `3e19` | `checkpoints/isoflop/isoflop-3e+19-d1536-L16-B32-adamh_scaling_v6/hf` |
| `9e19` | `checkpoints/isoflop/isoflop-9e+19-d1792-L18-B64-adamh_scaling_v6/hf` |
| `2e20` | `checkpoints/isoflop/isoflop-2e+20-d2048-L21-B64-adamh_scaling_v6/hf` |
| `3e20` | `checkpoints/isoflop/isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6/hf` |
| `1e21` | `adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/hf` |
| `1e22` | `adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/hf` |
| `1e23` | `adamh-scaling-ladder-nemotron-optimal-1e+23-v5-27f2fb/hf` |

The launcher passes these through `InputName.hardcoded(checkpoint)`, so they are
relative to the executor prefix. In the successful `us-east5` run, for example,
`3e18` resolves to:

```text
gs://marin-us-east5/checkpoints/isoflop/isoflop-3e+18-d1024-L11-B8-adamh_scaling_v6/hf
```

Do not assume cache hits across regions. Running with `MARIN_PREFIX` set to
`gs://marin-us-central2` computes the same logical names but different concrete
GCS paths.

## Launcher: Masked GSM8K IID

Entry point:

```text
experiments/downstream_scaling/evals/run_delphi_masked_gsm8k_iid.py
```

Static config:

| Setting | Value |
| --- | --- |
| `N_SAMPLES` | `32` |
| `N_PROBLEMS` | `256` |
| `NUM_WORKERS` | `1` |
| `TPU_TYPE` | `v5p-8` |
| `TEMPERATURE` | `0.6` |
| `TOP_P` | `1.0` |
| `TOP_K` | `1000` |
| `MAX_TOKENS` | `512` |
| `SEED` | `42` |
| `STOP_TOKENS` | `("Question:", "</s>", "<|im_end|>")` |
| `NUM_FEWSHOT` | `5` |
| `FEWSHOT_SEED` | `1234` |
| `MASK_FRACTIONS` | `tuple(i / 10 for i in range(11))` |
| `MASK_TEXT` | `"<mask>"` |
| tokenizer | `experiments.llama.llama3_tokenizer`, resolved in executor info as `meta-llama/Meta-Llama-3.1-8B` |

Output-name pattern:

```text
downstream_scaling/evals/delphi/masked_gsm8k/iid/mask_{i:02d}/{slug}
```

The launcher returns grade steps from `make_eval_step(...)`, but executor
resolution produces:

| Step kind | Count |
| --- | ---: |
| masked-GSM8K prompt steps | 11 |
| completion steps | 110 |
| grade steps | 110 |
| total unique output paths | 231 |

Full run size:

```text
11 masks * 10 checkpoints * 256 problems * 32 samples = 901,120 generated completions
```

Operational caution: `executor_main` defaults to running all ready steps unless
bounded by executor config. Launching the full script without a concurrency
limit may request far more TPU capacity than intended.

## Artifact Status: 2026-05-13

Observed source of truth for the launched run:

```text
gs://marin-us-east5/downstream_scaling/evals/delphi/masked_gsm8k/iid/
gs://marin-us-east5/downstream_scaling/evals/prompts/
```

Current local environment:

```text
MARIN_PREFIX=gs://marin-us-central2
```

`gs://marin-us-central2/downstream_scaling/evals/...` had no matching objects
when checked on 2026-05-13.

Prompt status:

```text
SUCCESS 12
```

The 12 prompt artifacts include 11 masked-GSM8K versions and one dummy prompt
artifact.

Completion/grade status under the launched IID path:

```text
SUCCESS 134
FAILED 39
RUNNING 12
```

Breakdown of non-success statuses:

```text
failed_completions=25
failed_grades=14
running_completions=10
running_grades=2
```

Examples:

- `mask_00/3e18/completions-8f3fee` is `SUCCESS`.
- `mask_00/3e18/grade-93eea0` is `SUCCESS`.
- `mask_10/1e23/completions-36f454` is `FAILED`.
- `mask_10/1e23/grade-a33cb8` had no `.executor_status` in the sampled check,
  consistent with a failed or incomplete dependency.

The status scan used only object listings and small status-file reads:

```bash
gcloud storage ls -r 'gs://marin-us-east5/downstream_scaling/evals/delphi/masked_gsm8k/iid/**/.executor_status' \
  | xargs -P 16 -n 1 sh -c 's=$(gcloud storage cat "$0" 2>/dev/null || echo ERROR); printf "%s\t%s\n" "$s" "$0"' \
  > /tmp/downstream_east5_statuses.tsv

awk -F '\t' '{count[$1]++} END {for (s in count) print s, count[s]}' \
  /tmp/downstream_east5_statuses.tsv | sort
```

## Known Gaps And Risks

- No reducer exists for per-condition metrics. `grades.jsonl.gz` is the final
  artifact today, so accuracy/pass@k tables need a new read-only analysis step.
- No tests exist under `experiments/downstream_scaling/`. A syntax pass with
  `uv run python -m compileall -q experiments/downstream_scaling` passed on
  2026-05-13, but that does not validate vLLM, TPU, Zephyr, or grader behavior.
- `iid.py` has an undocumented `bkv_p // 2` TPU inference patch. Find the error
  it avoids before removing or changing it.
- Several `RUNNING` status files are present in GCS. They may be stale executor
  states. Check Iris job state before assuming jobs are still live.
- Central-region artifacts are confusing:
  - `us-east5` has the useful launched artifacts.
  - `us-central2` had no matching downstream-scaling artifacts at scan time.
  - `us-central1` has some matching prefixes, but prompt status files checked
    there were failed or incomplete. Do not use `us-central1` as source of
    truth without a fresh scan.
- `math500.py` includes the strawberry prompt prefix by default. That could be a
  deliberate boxed-answer priming example or leftover debugging.
- `mask.py` and `rerank.py` have no launchers in this package. Treat them as
  unlaunched framework pieces until proven otherwise.
- `rerank.py` depends on experimental scorer RPC behavior in
  `experiments/rerank_decode/scorer.py`.
- `write_jsonl(..., skip_existing=True)` and chunk `.SUCCESS` files make resume
  behavior dependent on partial outputs. Inspect existing chunks before deleting
  or force-rerunning failed completion steps.

## Suggested Next Work

1. Build a read-only summary script for existing `us-east5` grades:
   - read `grades.jsonl.gz` for each successful grade output,
   - compute mean per-completion accuracy,
   - compute pass@1/pass@8/pass@16/pass@32 or equivalent,
   - emit a CSV keyed by `mask_fraction`, checkpoint key, and output path.
2. Check failed and running outputs before relaunch:
   - list `.executor_status`,
   - inspect `.executor_info`,
   - verify whether any Iris jobs are actually active,
   - preserve completion chunk outputs for failed completion steps unless
     intentionally restarting from scratch.
3. Add small local tests for schema and non-TPU logic:
   - duplicate-id rejection in readers,
   - completion order restoration from `completion_index`,
   - masked GSM8K prompt construction on a tiny fake task/tokenizer boundary if
     possible without mocking internal logger behavior.
4. Run one dev-TPU smoke test before broad relaunch:
   - `smoke_iid_vllm_tpu.py` validates vLLM load and one generation,
   - `time_iid_vllm_chunk.py` estimates chunk latency.
5. Decide whether the next experiment is:
   - finish the existing IID grid,
   - add result aggregation and plots first,
   - or write launchers for mask-hint/rerank after validating those algorithms.

## Useful Commands

Dry-run-style path resolution without launching jobs:

```bash
uv run python - <<'PY'
from collections import Counter
from marin.execution.executor import Executor
from experiments.downstream_scaling.evals.run_delphi_masked_gsm8k_iid import build_steps

prefix = "gs://marin-us-east5"
executor = Executor(prefix=prefix, executor_info_base_path=prefix + "/experiments")
for step in build_steps("v5p-8"):
    executor.compute_version(step, is_pseudo_dep=False)

paths = [executor.output_paths[step] for step in executor.steps]
print(f"executor_steps={len(executor.steps)}")
print(f"unique_output_paths={len(set(paths))}")
print(Counter(
    "prompts" if "/prompts/" in path
    else "completions" if "/completions-" in path
    else "grades" if "/grade-" in path
    else "other"
    for path in paths
))
PY
```

Check prompt status files:

```bash
gcloud storage ls -r 'gs://marin-us-east5/downstream_scaling/evals/prompts/**/.executor_status' \
  | xargs -P 8 -n 1 sh -c 's=$(gcloud storage cat "$0" 2>/dev/null || echo ERROR); printf "%s\t%s\n" "$s" "$0"'
```

Check one successful completion artifact:

```bash
gcloud storage ls -r \
  'gs://marin-us-east5/downstream_scaling/evals/delphi/masked_gsm8k/iid/mask_00/3e18/completions-8f3fee/**' \
  | head -50
```

## Timeline

### 2026-05-13

- Forked `rohith-working-agent` at `a93fcf4cca` into
  `aa_fork_rk_downstream` and pushed to origin.
- Claude wrote the first `.agents/logbook/downstream_eval.md` summary after the
  fork.
- Re-read `experiments/downstream_scaling/` and expanded this logbook into a
  more complete handoff.
- Verified the branch contains only Rohith's downstream-scaling commit plus the
  untracked logbook.
- Confirmed `gs://marin-us-east5` contains launched masked-GSM8K IID artifacts.
- Ran `uv run python -m compileall -q experiments/downstream_scaling`; syntax
  compilation passed. The generated local `__pycache__` files were removed after
  the check.

### 2026-05-13 — Pre-spend: SFT on GSM8K Q+A (planned)

Goal — extend `experiments/downstream_scaling/` with an **SFT half** so we can answer:

> Does midtraining on `nemotron_cc_math_v1/4plus` (math-heavy `p33m67` vs replay-heavy `p67m33`) help downstream GSM8K accuracy after a short SFT pass, and how does that interact with the Delphi pretrain scale?

Plan locked at `/Users/ahmed/.claude/plans/make-a-plan-effervescent-rivest.md`.

**Scope**: 16 SFT runs + 16 evals = 32 ExecutorSteps.
- 10 base ladder: 3e18, 9e18, 2e19, 3e19, 9e19, 2e20, 3e20, 1e20_iso, 1e21, 1e22 (Rohith's Delphi minus 1e23, plus `1e20_iso = isoflop-3e+20-d2048-L21-B128`).
- 6 midtrained (issue #4547, best-LR per mix):
  - `1e20_p33m67_lr0.67`, `1e20_p67m33_lr0.33`
  - `1e21_p33m67_lr0.67`, `1e21_p67m33_lr0.33`
  - `1e22_p33m67_lr0.67`, `1e22_p67m33_lr0.33`

**SFT config** — `SimpleSFTConfig(train_batch_size=64, max_seq_len=1024, num_train_steps=360, learning_rate=5e-6, warmup=0.03, lr_schedule="linear", decay=0.9, steps_per_hf_export=360, pad_tokenizer_to_match_model=True)`. v5p-8 for sub-1e22, v5p-32 for 1e22-scale.

**Data format** — canonical lm-eval GSM8K shape:
- Training chat template: `Question: {q}\nAnswer:{% generation %} {a}{% endgeneration %}`
- Eval prompt: `Question: {q}\nAnswer:` (zero-shot, no fewshot)
- Train target: raw GSM8K `answer` field including `<<x=y>>` annotations and `#### N` marker.
- Grader: Rohith's existing `_grade_gsm8k_shard` (lm-eval `flexible_extract` + `strict_match` filters) — unchanged.

**Expected artifact paths** (under `MARIN_PREFIX=gs://marin-us-east5`):
- SFT checkpoints: `checkpoints/downstream_scaling/sft/delphi/gsm8k_qa/{slug}/hf/step-360/`
- Eval prompts: `downstream_scaling/sft/data/gsm8k_qa_*/prompts.jsonl.gz`
- Eval completions: `downstream_scaling/evals/delphi_sft/gsm8k_qa/iid/{slug}/completions-*/`
- Eval grades: `downstream_scaling/evals/delphi_sft/gsm8k_qa/iid/{slug}/grade-*/grades.jsonl.gz`

**Expected cost** — ~14 TPU-hours total (~8 base + ~4 midtrain + ~2 eval), multiplied by preemption-retry overhead.

**Open blocker before main launch** — resolve the 1e20_iso base path. Plan calls for reading `initialize_from_hf` from one 1e20 midtrain W&B run.

**Expected smoke result** — 3e18 after 10 SFT steps scores near 0% on GSM8K-256. Smoke verifies wiring, not quality.

**Expected full-sweep findings** (hypothesis to falsify):
- Post-SFT mean accuracy monotonically increases with base-pretrain scale: 3e18 → 1e22.
- SFT initialized from `p33m67` (math-heavy midtrain) > SFT initialized from `p67m33` (replay-heavy midtrain) > SFT initialized from the raw base pretrain, at each of 1e20/1e21/1e22.
- All post-SFT cells exceed Rohith's existing `mask_00` numbers (which evaluate the bare base pretrain at 5-shot solution-primed `mask_fraction=0.0`, no SFT applied).

### 2026-05-13 — Smoke run (`aa-smoke-sft-gsm8k-3e18-v5`): success after 5 attempts

End-to-end smoke landed. Pipeline verified: `gsm8k_raw` → `transform` → `tokenize` → `SFT (3e18, 10 steps)` → `IID completions` → `grade`. Final artifact:

- Grades: `gs://marin-us-east5/downstream_scaling/evals/smoke_sft/gsm8k_qa/iid/3e18/grade-60898d/grades.jsonl.gz`
- SFT'd HF checkpoint: `gs://marin-us-east5/checkpoints/downstream_scaling/sft/smoke/delphi/gsm8k_qa/3e18-db520a/hf/step-9/`
- W&B run: https://wandb.ai/marin-community/marin/runs/3e18-db520a
- Smoke accuracy: **12/256 ≈ 4.7%** (8 prompts × 32 samples; expected near-zero since 10 SFT steps on a 3e18 model is far from converged). Sample extractions: `$60.00`, `$2`, `$2.50.` — flexible_extract grader is pulling numbers out of unrelated dollar-amount responses on most prompts. Pipeline correctness, not quality, is what the smoke verifies.

#### Bug findings + fixes

1. **1e20_iso base path was wrong (off by `v6` vs `v5`).** The path I put into `models/delphi_extra.py` from the plan (`.../isoflop-3e+20-d2048-L21-B128-adamh_scaling_v6/hf`) doesn't exist. Resolved by reading `.executor_info` on one of the 1e20 midtrain runs (`delphi-1e20-p33m67-4p94b-lr0.67-7c32da`), which reveals `initialize_from_checkpoint_path = mirror://checkpoints/isoflop/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/checkpoints/step-46915`. Correct suffix is **`adamh_scaling_v5`**, not `v6`. The HF version of this base lives **only in `gs://marin-us-central2`** (not east5) — final HF step `step-47063`. Updated `delphi_extra.py` to use an absolute `gs://marin-us-central2/...` URI, and updated `sft/train.py:_resolve_latest_hf_checkpoint` to handle `://` URIs by skipping the MARIN_PREFIX prefix join.

2. **`iid.py:_load_vllm` patched a function that no longer exists.** Rohith's patch references `rpa_kernel.get_tuned_block_sizes` (old API), but current `tpu_inference.kernels.ragged_paged_attention.v3.kernel` exposes `get_default_block_sizes`. Crashed the IID completion child with `AttributeError`. Rohith already wrote the corrected patch in `smoke_iid_vllm_tpu.py:25-44` (halves `bq_sz`, `bq_csz`, `bkv_sz`, `bkv_csz` for non-DECODE cases). Updated `iid.py:_load_vllm` to mirror that. This change does NOT re-version anything — the patch is inside the worker function, not in any `versioned(...)` field.

3. **`rope_scaling`-vs-`max_position_embeddings` is non-fatal.** transformers prints a stderr warning about `original_max_position_embeddings=8192 > max_position_embeddings=4096` on Delphi base configs. Levanter's `HFCheckpointConverter` tolerates it and produces a clean `Qwen3Config`. Confirmed not a blocker for SFT init. **vLLM also tolerates it** — the SFT'd checkpoint loaded cleanly and generated 256 samples without rope-scaling failure.

#### Launch invocation (5 attempts, fixes accumulated)

The final working invocation:

```bash
uv run iris --cluster=marin job run --no-wait \
    --job-name aa-smoke-sft-gsm8k-3e18-v5 \
    --zone us-east5-a \
    --tpu v5p-8 --enable-extra-resources \
    --cpu 32 --memory 128GB --disk 50GB \
    --extra tpu \
    --max-retries 2 \
    -e MARIN_PREFIX gs://marin-us-east5 \
    -- python experiments/downstream_scaling/smoke_sft_gsm8k_qa.py
```

Attempt-by-attempt failure modes (all on the SAME smoke script with progressive flag fixes):

| Attempt | Flags changed | Failure | Cause |
|---|---|---|---|
| v1 | `--cpu 1 --memory 4G --extra cpu` (no TPU) | `RuntimeError: No accelerator found` in `levanter/main/train_lm.py:112` | SFT step's `_run_iris_job` ran training **inline in the coordinator process** instead of dispatching as a child Iris job. Coordinator had no TPU. |
| v2 | same as v1, retry | identical | (v1 was killed by `--max-retries-preemption` typo; resubmitted same flags) |
| v3 | added `--tpu v5p-8`, dropped `--extra cpu` | `Failed to open libtpu.so: cannot open shared object file` | TPU host has no `libtpu` system library installed. Container `default_task_image: ghcr.io/marin-community/iris-task:latest` doesn't bundle TPU JAX. |
| v4 | added `--extra tpu` (installs `jax==0.9.2`, `libtpu==0.0.38`) | `Container was OOM killed by the kernel` | Container `--memory 1GB` default insufficient for SFT process memory footprint. |
| v5 | added `--cpu 32 --memory 128GB --disk 50GB` (match `SimpleTrainConfig.resources`), updated `iid.py` to use `get_default_block_sizes` | **SUCCESS** | n/a |

#### Open framework question: why doesn't `_run_iris_job` actually submit a child Iris job for SFT?

Per `step_runner.py:340-346`:

```python
if step.resources is not None:
    _run_iris_job(step, output_path)
elif isinstance(step.fn, RemoteCallable):
    _run_remote_step(step, output_path)
else:
    result = step.fn(output_path)
```

For the SFT step from `default_sft → default_train`:
- `step.fn = run_levanter_train_lm` (plain function, not `@remote`)
- `step.resources = ResourceConfig.with_tpu("v5p-8")` (set)

Path SHOULD be `_run_iris_job` → `_submit_iris_job` → `fray_client.current_client().submit(JobRequest(...))` → new Iris child job with `cpu=32, ram=128g, device=TpuConfig(v5p-8)`. But empirically the training ran **inline in the coordinator's process** (confirmed via task-id prefix and host name `marin-cpu-vm-e2-highmem-2-ondemand-us-ea-...`). No child Iris job appeared in `iris job list --prefix /ahmed/aa-smoke-sft-gsm8k-3e18`.

In contrast, the IID completion step (`@remote`-wrapped) DID dispatch as a child Iris job (`.../downstream_scaling-evals-smoke_sft-gsm8k_qa-iid-3e18-completions_dfc7d948-03b8dee2`). So `@remote` dispatch works; plain-function-with-resources dispatch does not.

**Hypothesis (unverified)**: PR #5279 (`[executor] Run executor inside training job; remove Iris region inheritance`, May 7 2026) made training steps run inline in whatever job hosts the executor, with `materialize(config)` inside `run_levanter_train_lm` handling upstream-deps materialization. That would explain the observed behavior. If true, the **NEW intended launch pattern** is to:
- Submit the entrypoint with `--tpu <variant>` (or use `training_step.fn(training_step.config)` directly, which calls @remote → submits with TPU).
- Let the executor walk and run training in-process.

For the **16-cell prod sweep this is a problem**: serial execution of 16 SFT runs each on its own dedicated v5p-8 lifetime is wasteful. Worth either:
- Asking Russell (PR #5279 author) for the canonical sweep pattern under the new design.
- Or biting the bullet and re-implementing the SFT step as `@remote` so each cell gets its own child Iris job dispatched from a CPU coordinator.

For now the smoke proves the per-cell pipeline correctness end-to-end. Production launch can use the new launch invocation (v5 flags) for a single cell at a time, or wait for the framework fix.

#### Files changed during smoke iteration

- `experiments/downstream_scaling/models/delphi_extra.py` — corrected `1e20_iso` path: absolute `gs://marin-us-central2/.../adamh_scaling_v5/hf` URI (the HF version exists only in central2).
- `experiments/downstream_scaling/sft/train.py:_resolve_latest_hf_checkpoint` — accept absolute `://` URIs and skip MARIN_PREFIX join.
- `experiments/downstream_scaling/evals/algorithms/iid.py:_load_vllm` — replace stale `get_tuned_block_sizes` patch with `get_default_block_sizes` patch (matches `smoke_iid_vllm_tpu.py`).

### 2026-05-14 — Production 1e20 SFT comparison (SFT from base vs SFT from math-midtrain)

After the smoke landed, ran the first real SFT experiment. **Both cells were SFT'd**; the only thing that varies between them is the checkpoint SFT was initialized from. Cell A: SFT initialized from the raw **Delphi 1.9B base pretrain** at the 3e+20 FLOP IsoFLOP point. Cell B: SFT initialized from **the same base after `p33m67 lr=0.67` math midtraining**. Both used the same GSM8K Q+A SFT recipe, same tokenizer, same eval config. Then re-ran the same pair with `pack=1` and 1 epoch to disentangle training budget from packing. Both pairs succeeded end-to-end. **All artifacts on GCS; zero code committed yet — see the "Code state" section below for the dirty-tree summary.**

#### Headline results

**All four cells below are post-SFT.** The only axis that varies between rows is which checkpoint SFT was initialized from (raw base pretrain vs the same base after math midtraining). Both rows use identical SFT recipe, identical tokenizer, identical eval config. **No pre-SFT zero-shot numbers exist** for either checkpoint — see "Missing baselines" below.

| SFT-init checkpoint | 19 packed epochs (pack=True, 360 steps) | 1 unpacked epoch (pack=1, 117 steps) |
|---|---|---|
| base 1.9B pretrain (slug `1e20_iso`) | **2.73%** mean / **37.5%** pass@32 | 1.86% / 30.5% |
| math-midtrain 1.9B (slug `1e20_p33m67_lr0.67`) | **18.82%** mean / **66.4%** pass@32 | 13.94% / 65.2% |

- Initializing SFT from the math-midtrained checkpoint instead of the raw base gives a **~7× uplift in post-SFT mean accuracy** at this scale, robust across training budgets (7.5× at 1 unpacked epoch, 6.9× at 19 packed epochs).
- 19 packed-epochs beats 1 unpacked-epoch on mean accuracy for both rows. GSM8K's 7.5k train split is small enough that more SFT doesn't overfit on this format.
- Pass@32 saturates earlier than mean accuracy (math-midtrain row: 65.2 → 66.4, +1.2pp from extra training vs +4.88pp on mean). At 32 samples we already cover most "in-reach" problems early; later training improves per-sample quality, not coverage.

#### Missing baselines

What this comparison does **not** measure:

1. **Raw base pretrain at GSM8K Q+A zero-shot, no SFT** — no datapoint for "what does the bare 1.9B base score on `Question: {q}\nAnswer:` with our T=0.6, n=32 sampling config?"
2. **Raw math-midtrain checkpoint at GSM8K Q+A zero-shot, no SFT** — same gap on the midtrain side. So we cannot decompose the 18.82% post-SFT mean into "midtraining alone gave us X" plus "the SFT pass added Y."
3. **Apples-to-apples vs Rohith's `mask_00` cell** — his 5-shot solution-primed eval is a different task: same grader, different difficulty. See "Eval-task mismatch with Rohith's matrix" below.

Each of (1) and (2) is one eval-only step (vLLM only, no retrain), ~22 min on v5p-8. Wire by pointing `make_eval_step` at the raw HF checkpoint paths in `models/delphi_extra.py` and `models/midtrain.py` instead of at SFT-step outputs.

#### What we actually trained on (the "1e20" naming clarification)

Ahmed labels his midtrain base "1e20" by W&B convention. **The actual model is 3e+20 FLOPs (not 1e20).** Specifics:

- Path: `gs://marin-us-east5/checkpoints/isoflop/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/hf` (final step `step-47063` in central2 — the HF version of this checkpoint ONLY exists in `gs://marin-us-central2`, not east5; we handle this with the `mirror://` rewrite in `_to_mirror_uri`).
- Architecture: d=2048, L=21, num_heads=16, vocab=128,256 → **1.9B params**.
- Training: 3e+20 FLOPs (6ND) ≈ 24.7B tokens with batch 128.
- Released HF id: [`marin-community/delphi-3e20-1.9Bparams-24.7Btokens`](https://huggingface.co/marin-community/delphi-3e20-1.9Bparams-24.7Btokens). It IS published as part of canonical Delphi (88 base models).
- **Off-compute-optimum**: the 3e20 bucket-winner (per the HF collection's own annotation) is `delphi-3e20-2.5Bparams-18.6Btokens` — our base is one IsoFLOP bucket below the optimum. Token-to-param ratio ~13 vs the optimum's ~7.4. Mildly overtrained for its param count — standard small-model-for-deployment shape.

#### Sweep version glossary (corrected from earlier notes)

Three sweep series exist on Nemotron data; only the AdamH ones are canonical Delphi:

| Sweep | Recipe | Used by | Notes |
|---|---|---|---|
| `nemo-wider-depth-adapt` | Cautious AdamC (`c_adamc.py`) | HF `marin-community/nemotron-isoflop-models` collection (Oct 2025) | The **broken "Attempt 1"** from Will's blog. Diverged at 1e23. Not part of canonical Delphi. |
| `adamh_scaling_v5` | AdamH (`completed_adamh.py`) | Ahmed's midtrain bases at 1e20 (the `isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5` checkpoint above); also the published-only-via-HF large-scale held-outs (`adamh-scaling-ladder-nemotron-optimal-1e+2{1,2,3}-v5-*`) | Earlier iteration of AdamH. Includes the d2048-L21 point at 3e20 that Ahmed used. |
| `adamh_scaling_v6` | AdamH (`completed_adamh.py`, same file, retuned via Vizier) | Current `experiments/isoflop_sweep.py:229` `nemotron-completed-adamh` sweep; Rohith's `models/delphi.py` small-scale entries 3e18→3e20 | Newer iteration; re-ran small scales. The HF Delphi release unions both v5 and v6 under cleaner names by (FLOPs × params × tokens). |

v5 and v6 are both AdamH ("Attempt 2") iterations — **not** Attempt-1 vs Attempt-2. Both are valid Delphi artifacts. v5 at 3e20 uses d2048-L21 (1.9B); v6 at 3e20 uses d2304-L23 (2.5B, the bucket-winner).

#### Eval-task mismatch with Rohith's existing matrix

Our `GSM8KQATask` is **zero-shot Q+A** — model gets `Question: {q}\nAnswer:` and must produce the full CoT plus `#### N`.

Rohith's `MaskedGSM8KTask` at `mask_fraction=0.0` (his `mask_00` baseline cell) is **5-shot solution-primed completion** — 5 few-shot exemplars followed by the target's full CoT, and the model produces only the final `#### N` line.

Same grader (lm-eval `flexible_extract` / `strict_match` regex over `#### N`). **Sampling config is byte-identical between the two** (n=32, T=0.6, top_p=1.0, top_k=1000, max_tokens=512, seed=42, n_problems=256, v5p-8) except for one extra stop token in our launcher (`<|end_of_text|>`). Different *task difficulty*. **Our 1e20 numbers are NOT directly comparable to Rohith's `mask_00` numbers.** For apples-to-apples either:

- Run our SFT'd checkpoints through Rohith's `MaskedGSM8KTask` at `mask_fraction=0.0`, OR
- Run the base Delphi 1.9B through our `GSM8KQATask` (zero-shot baseline, no SFT).

##### Verified prompt format from `gs://marin-us-east5/downstream_scaling/evals/prompts/masked_gsm8k-42fd11/prompts.jsonl.gz` (the mask_00 artifact)

Pulled an actual prompt to ground-truth the format. At `mask_fraction=0.0`:

```text
Question: <fewshot_q_0>
Answer: <fewshot_CoT_0>
<fewshot_CoT_0 AGAIN — see "doubled CoT" note below>
#### <fewshot_a_0>

... (5 few-shot exemplars total)

Question: <target_q>
Answer: <target_CoT, fully written out>
```

The model continues with `#### N`. For example, the first test problem's target CoT ends with `"She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market."` — the model just has to output `#### 18`. **That is pattern matching from a visible answer, not math reasoning.** `mask_00` is therefore much easier than what "5-shot GSM8K baseline" usually means in lm-eval-harness (where the model produces the full CoT from `Question: X\nAnswer:`).

##### Two prompt-construction observations worth knowing before any compatibility eval

These are observations from reading `gsm8k_masked.py:107-130` and verifying against a `mask_05` prompt artifact (`gs://marin-us-east5/downstream_scaling/evals/prompts/masked_gsm8k-8f0376/prompts.jsonl.gz`):

1. **Few-shot exemplars contain the CoT body twice.** Lines 119-120:
   ```python
   prompt += task.doc_to_text(fewshot_doc) + task.config.target_delimiter + masked_solution + "\n"
   prompt += task.doc_to_target(fewshot_doc) + task.config.fewshot_delimiter
   ```
   First the (possibly masked) CoT body, then the *unmasked* full answer (`doc_to_target` returns CoT + `#### N`). For `mask_fraction=0.0` the two copies are identical and the duplication is just redundant. For `mask_fraction > 0` the first copy is masked but the second copy is the clean answer — meaning the few-shot exemplars are effectively unmasked regardless of `mask_fraction`. Only the target's CoT is meaningfully masked.

2. **Therefore the mask axis only affects the target's CoT.** The few-shot demonstrations stay clean across all 11 mask conditions, only the target prefix gets perturbed. This is plausibly intentional (the few-shot is teaching "masked-noise → clean-answer", and the model is then asked to do that mapping on a noisy target), but it is not documented in the code. Any analysis that assumes the few-shot also degrades with `mask_fraction` is wrong.

##### Sampling config cross-reference

| Aspect | Rohith mask_00 — `evals/run_delphi_masked_gsm8k_iid.py:25-35` | Our SFT eval — `run_delphi_sft_gsm8k_qa.py:45-52` |
|---|---|---|
| `n_samples` | 32 | 32 |
| `temperature` | 0.6 | 0.6 |
| `top_p` / `top_k` | 1.0 / 1000 | 1.0 / 1000 |
| `max_tokens` | 512 | 512 |
| `seed` | 42 | 42 |
| `stop` | `("Question:", "</s>", "<|im_end|>")` — 3 stops | `("Question:", "</s>", "<|im_end|>", "<|end_of_text|>")` — 4 stops |
| `num_fewshot` | **5** (`fewshot_seed=1234`) | **0** |
| `n_problems` | 256 | 256 |
| What the model produces | only `#### N` | full CoT + `#### N` |

So if we want to drop our SFT'd checkpoints into Rohith's pipeline for an apples-to-apples comparison, the only delta to manage is `num_fewshot` and the prompt format — the sampling/grader/n_problems all already line up.

#### Bugs discovered + fixed today (all unhashed in iid.py and sft/train.py)

Five bugs surfaced. **None of the fixes are committed yet.**

**1. Levanter `pack=False` triggers `max_segments_per_example=0` validation error.**

At `lib/levanter/src/levanter/data/text/datasets.py:485-487`:
```python
max_segments = (
    64 if effective_pack is True else (int(effective_pack) if isinstance(effective_pack, int) else 1)
)
```
Since `bool` subclasses `int`, `isinstance(False, int)` is True and `int(False) = 0` → `max_segments=0` → ChatDataset constructor raises `ValueError: max_segments_per_example must be a positive integer`. Workaround: pass `pack=1` (literal int) instead of `pack=False`. Routes through `int(1)=1` correctly. Same behaviour as no-pack (max 1 Q+A pair per sequence).

**2. Cross-region SFT init fails the `check_gcs_paths_same_region` guard.**

`marin.training.training` validates that `initialize_from_hf` lives in the same GCS region as the VM. The 1e20_iso HF checkpoint only exists in `gs://marin-us-central2/`; v5p-8 in us-east5-a → cross-region read → `ValueError`. Fix: rewrite `gs://marin-<region>/...` URLs to `mirror://...` so the check (which only looks at `gs://` paths) ignores them, and Levanter's MirrorFileSystem auto-copies to the local region on first access. Helper at `experiments/downstream_scaling/sft/train.py:_to_mirror_uri`.

**3. vLLM trips on `q_norm` because Levanter's SFT export carries forward the upstream PR #3092 `LlamaForCausalLM` metadata bug.**

The base Delphi (and midtrained) HF configs ship `architectures: ["LlamaForCausalLM"]` but the weights are Qwen3 (have `q_norm`/`k_norm`). vLLM's LlamaForCausalLM loader doesn't know what to do with `q_norm` → `AttributeError: model.layers.20.self_attn.q_norm is not a valid param path`. Levanter's HF export propagates whatever `architectures` the base config had — so SFT'd `1e20_iso` ALSO has `LlamaForCausalLM` and fails. p33m67's base already had `Qwen3ForCausalLM` (Ahmed fixed it in his midtrain export), so SFT'd p33m67 succeeded. Fix: pass `hf_overrides={"architectures": ["Qwen3ForCausalLM"]}` to `vllm.LLM(...)` in `iid.py:_load_vllm` — forces the right loader regardless of config. `model_type: qwen3` was correct in the config already; only the architectures key was wrong.

**4. `iid.py` patched a function that no longer exists in current `tpu_inference`.**

Rohith's patch was `rpa_kernel.get_tuned_block_sizes` (old API). Current API: `get_default_block_sizes` with different return shape (dict of sizes, not tuple). Rebased the patch to match `smoke_iid_vllm_tpu.py:25-44` — halves `bq_sz`, `bq_csz`, `bkv_sz`, `bkv_csz` for non-DECODE cases. Idempotent via `_marin_iid_patched` sentinel.

**5. Eval cache-hit when SFT config changes but the changed fields aren't `versioned()`.**

The executor only includes `versioned(...)` values in step hashes. Plain `SimpleSFTConfig` fields (`num_train_steps`, `train_batch_size`, `learning_rate`, `pack` via the format, etc.) are NOT versioned. So changing `num_train_steps=360 → 117` does NOT change the SFT step's hash. Combined with `output_path_of(sft_step)` returning an `InputName` whose hash depends on the SFT step's identity (not its `override_output_path`), the eval step's hash is also stable across SFT config changes. Result: eval cache-hits the previous grades, even though the new SFT writes different weights to a different `with_output_path(...)` location.

Workaround: derive the eval-step name from `SFT_OUTPUT_PREFIX` so changing the SFT prefix forces a new eval namespace. Done in `run_one_sft_gsm8k_qa.py:build_steps`:

```python
sft_variant = SFT_OUTPUT_PREFIX.rstrip("/").split("/")[-1]
eval_step = make_eval_step(
    name=f"downstream_scaling/evals/delphi_sft/{sft_variant}/iid/{slug}",
    ...
)
```

Bookkeeping convention: bump `SFT_OUTPUT_PREFIX` suffix whenever you change SFT config in a way that should re-run eval. The 19-epoch run lives under `gsm8k_qa/`; the 1-epoch run lives under `gsm8k_qa_nopack_1ep/`. Both preserved.

#### Code state (uncommitted, working tree on `aa_fork_rk_downstream`)

Files **modified** today (deltas not yet `git add`-ed):

| File | Change |
|---|---|
| `experiments/downstream_scaling/sft/tokenize.py` | `pack=True` → `pack=1` to bypass the Levanter bool/int bug. Long inline comment explaining why. |
| `experiments/downstream_scaling/sft/train.py` | `DEFAULT_SFT_CONFIG`: `num_train_steps=360 → 117` (1 true epoch w/ pack=1), `steps_per_hf_export=360 → 117`, `steps_per_eval=90 → 30`. `SFT_OUTPUT_PREFIX` bumped to `gsm8k_qa_nopack_1ep`. Added `_to_mirror_uri` helper + `_resolve_latest_hf_checkpoint` accepts `://` URIs. |
| `experiments/downstream_scaling/evals/algorithms/iid.py` | Rebased RPA-kernel patch to `get_default_block_sizes` API; added `hf_overrides={"architectures": ["Qwen3ForCausalLM"]}` to `LLM(...)`. |
| `experiments/downstream_scaling/models/delphi_extra.py` | `1e20_iso` path = absolute `gs://marin-us-central2/.../adamh_scaling_v5/hf` URI (not the `_v6` I'd guessed). |
| `experiments/downstream_scaling/run_one_sft_gsm8k_qa.py` | Derive eval namespace from `SFT_OUTPUT_PREFIX` so eval doesn't cache-hit when SFT config changes. |
| `.agents/logbook/downstream_eval.md` | This file. |

Files **added** today:

| File | Purpose |
|---|---|
| `experiments/downstream_scaling/run_one_sft_gsm8k_qa.py` | Focused single-slug launcher; takes `--slug`. |
| `experiments/downstream_scaling/sft/train.py` | `default_sft` wrapper with config + path-resolution helpers. |
| `experiments/downstream_scaling/sft/tokenize.py` | `GSM8K_QA_CHAT_FORMAT` + `SFT_RESOURCES` tier map. |
| `experiments/downstream_scaling/sft/data/gsm8k_qa.py` | GSM8K train → `messages` JSONL transform step. |
| `experiments/downstream_scaling/sft/__init__.py` + `sft/data/__init__.py` | Package markers. |
| `experiments/downstream_scaling/evals/tasks/gsm8k_qa.py` | Zero-shot Q+A eval task; reuses `GSM8KTask.make_grade_step`. |
| `experiments/downstream_scaling/models/delphi_extra.py` | `1e20_iso` base entry. |
| `experiments/downstream_scaling/models/midtrain.py` | Registry of 6 midtrain variants (p33m67 + p67m33 at 1e20/1e21/1e22, best LR each). |
| `experiments/downstream_scaling/run_delphi_sft_gsm8k_qa.py` | Full 16-cell sweep launcher (not yet used — single-cell variant is `run_one_sft_gsm8k_qa.py`). |
| `experiments/downstream_scaling/smoke_sft_gsm8k_qa.py` | Smoke test. |

#### GCS artifact map for the 1e20 experiment

```
# Base + midtrain bases (read-only)
gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/hf/step-47063/   # 1e20_iso (1.9B), central2 only
gs://marin-us-east5/checkpoints/delphi-1e20-p33m67-4p94b-lr0.67-7c32da/hf/step-9412/                     # midtrain p33m67 lr=0.67

# 19-epoch SFT outputs (pack=True, 360 steps)
gs://marin-us-east5/checkpoints/downstream_scaling/sft/delphi/gsm8k_qa/1e20_iso/hf/step-359/             # SFT'd base, 19 packed epochs
gs://marin-us-east5/checkpoints/downstream_scaling/sft/delphi/gsm8k_qa/1e20_p33m67_lr0.67/hf/step-359/   # SFT'd midtrain, 19 packed epochs

# 19-epoch grades
gs://marin-us-east5/downstream_scaling/evals/delphi_sft/gsm8k_qa/iid/1e20_iso/grade-b0066d/grades.jsonl.gz             # 2.73% / 37.5%
gs://marin-us-east5/downstream_scaling/evals/delphi_sft/gsm8k_qa/iid/1e20_p33m67_lr0.67/grade-94dfe4/grades.jsonl.gz   # 18.82% / 66.4%

# 1-epoch SFT outputs (pack=1, 117 steps)
gs://marin-us-east5/checkpoints/downstream_scaling/sft/delphi/gsm8k_qa_nopack_1ep/1e20_iso/hf/step-116/
gs://marin-us-east5/checkpoints/downstream_scaling/sft/delphi/gsm8k_qa_nopack_1ep/1e20_p33m67_lr0.67/hf/step-116/

# 1-epoch grades
gs://marin-us-east5/downstream_scaling/evals/delphi_sft/gsm8k_qa_nopack_1ep/iid/1e20_iso/grade-17a6eb/grades.jsonl.gz              # 1.86% / 30.5%
gs://marin-us-east5/downstream_scaling/evals/delphi_sft/gsm8k_qa_nopack_1ep/iid/1e20_p33m67_lr0.67/grade-537f41/grades.jsonl.gz    # 13.94% / 65.2%
```

#### W&B runs (search by name prefix)

- 19-epoch runs: `3e18-db520a` (smoke), `1e20_iso-*`, `1e20_p33m67_lr0.67-*` under `marin-community/marin`.
- 1-epoch runs: same project, hashed run IDs starting `1e20_iso-` and `1e20_p33m67_lr0.67-` with later timestamps.

#### Working `iris job run` invocation (the incantation that survives every bug)

```bash
uv run iris --cluster=marin job run --no-wait \
    --job-name aa-sft-<slug>-<tag> \
    --zone us-east5-a \
    --tpu v5p-8 --enable-extra-resources \
    --cpu 32 --memory 128GB --disk 50GB \
    --extra tpu \
    --max-retries 2 \
    -e MARIN_PREFIX gs://marin-us-east5 \
    -- python experiments/downstream_scaling/run_one_sft_gsm8k_qa.py --slug <slug>
```

All five flags matter:

- `--tpu v5p-8 --enable-extra-resources`: coordinator must have a TPU because `_run_iris_job` runs SFT inline (see "Open framework question" above).
- `--cpu 32 --memory 128GB --disk 50GB`: match `SimpleSFTConfig.resources` (`ResourceConfig.with_tpu("v5p-8")` expands to these). Smaller values OOM-kill the container during training.
- `--extra tpu`: installs `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.38`. Without it: `Failed to open libtpu.so` at JAX backend init.
- `--max-retries 2`: caps blast radius if something regresses. Iris auto-retries preemptions separately (default ~1000) — so this only limits hard failure retries.
- `-e MARIN_PREFIX gs://marin-us-east5`: pins artifact region. Midtrain bases and Delphi 1e21/1e22 live in east5; gsm8k_raw gets re-downloaded here if not cached.

#### Operational notes (capacity, wall-clock)

- us-east5-a v5p-8 capacity was thin overnight (multiple jobs competing). Each cell took **2–4 preempts**; Iris auto-resumes from Levanter rolling temp checkpoints so progress isn't fully lost, just wall-clock-stretched.
- Wall-clock per cell:
  - 19-epoch SFT step: **~12 min** (iso-v2 log: `0:11:54.822575`).
  - 1-epoch SFT step: **~3 min** (iso-nopack-v2 log: `0:02:52.519731`).
  - Completions step (vLLM 256 × 32 generations on v5p-8): **~22 min** uninterrupted.
  - Grade step (Zephyr lm-eval graders): **~11 min**.
  - Full pipeline wall-clock with 2–4 preempts: **45–90 min/cell**.
- Plan for the next session: assume east5-a will be similarly contended. Either submit early and walk away, or try a less-popular zone if capacity reports look better.

#### Open questions / handoff items

1. **`_run_iris_job` inline-execution mystery** (unresolved from smoke debugging). Per `step_runner.py:340-346` SFT *should* dispatch as a child Iris job because `step.resources is not None`. Empirically it runs inline in the coordinator. Hypothesis: PR #5279 changed the model. Either ask Russell or grep `step_runner.py` git history at the commit. Resolving this lets the prod 16-cell sweep run as parallel child jobs from a CPU coordinator (~16× wall-clock reduction).

2. **Eval-task mismatch with Rohith's matrix.** Our zero-shot Q+A is different from `mask_00`. Two compatibility runs would give directly-comparable numbers:
   - Re-eval our SFT'd checkpoints with `MaskedGSM8KTask, mask_fraction=0.0, num_fewshot=5` — cheap, ~22 min × N cells (vLLM only, no retrain).
   - Re-eval base Delphi 1.9B with `GSM8KQATask` (zero-shot) to get the "no SFT, no midtrain" baseline. Currently we have no such datapoint.

3. **Code is not committed.** [RESOLVED 2026-05-21 in commit `b7ca6bf5f` "initial downstream eval / sft" — everything described below was bundled into a single commit rather than the suggested 3-way split.] All 6 file modifications + 10 added files are in the dirty working tree on branch `aa_fork_rk_downstream` (pushed to origin earlier as a fork of `rohith-working-agent@a93fcf4cca`). Whether to commit + push is a decision for the user. If committing, suggested split:
   - One commit for the framework fixes (`iid.py`, `train.py`, `tokenize.py`) so they could be PR'd upstream cleanly.
   - One commit for the new experiment skeleton (`sft/`, `run_one_sft_gsm8k_qa.py`, `evals/tasks/gsm8k_qa.py`, model registries).
   - One commit for the smoke launcher + initial logbook.

4. **Next experimental directions (in order of value-for-cost):**
   - **Add `p67m33` (replay-heavy) at 1e20** for a 3-way comparison (base / math-heavy / replay-heavy). 1 additional cell, ~1 TPU-hour, lets us see whether the 7× gap is specifically from math content or from any midtraining.
   - **Scale up: 1e21 (3.4B) and 1e22 (9.7B)** for both base + p33m67 to test whether the 7× gap shrinks or grows with scale. ~6 cells total. Per cell at 1e21: same v5p-8, longer per step; at 1e22: needs v5p-32 (per `SFT_RESOURCES` map in `sft/tokenize.py`).
   - **Compatibility eval pass with `MaskedGSM8KTask, mask_fraction=0.0`** to get a number that lines up with Rohith's 110-cell sweep matrix.
   - **Disentangle packing from epochs**: run `pack=1, num_train_steps=2223` (19 unpacked epochs) so we can attribute the 19→1-epoch performance drop cleanly. Cheaper alternative: `pack=True, num_train_steps=19` (1 packed epoch).

5. **Sanity-check the 1e20 base path before any 1e20 re-launch.** Confirmed today the HF checkpoint lives at `gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/hf/step-47063/`. The `_to_mirror_uri` rewrite handles cross-region read at SFT init time (~1.79 GB one-time copy from central2 → east5 per cell). For 1e21 and 1e22 cells, base lives in east5 already — no mirror copy needed.

#### TL;DR for the next agent

You're picking up `aa_fork_rk_downstream` after a successful 2-cell 1e20 SFT comparison + a 1-epoch follow-up. **Initializing SFT from the math-midtrain checkpoint instead of the raw base pretrain gives a ~7× uplift in post-SFT GSM8K accuracy at the 1.9B scale.** We have no pre-SFT zero-shot eval datapoints — only post-SFT — so this comparison isolates the midtrain-init axis but does not decompose "midtrain effect" from "SFT effect." Code uncommitted at EOD 2026-05-14 (later pushed as `b7ca6bf5f` on 2026-05-21). The working launcher is `experiments/downstream_scaling/run_one_sft_gsm8k_qa.py --slug <slug>`; valid slugs are in `_checkpoint_registry()` (3e18 → 1e22 base ladder + 1e20_iso + 6 midtrain variants). The five framework bugs above are all fixed locally — don't re-discover them. The eval-cache-hit pattern (item 5) is the most subtle: **bump `SFT_OUTPUT_PREFIX` whenever you change SFT hyperparameters that aren't `versioned()`, otherwise the executor will reuse the previous eval's grades.**

### 2026-05-15 — Architectural synopsis of `experiments/downstream_scaling/`

Honest read after a week of working in this subtree: the **eval half is clean**; the **SFT half I bolted on is a side appendage**, not a first-class peer.

```
experiments/downstream_scaling/
  evals/
    GUIDE.md                  ← canonical design doc; read first
    framework/
      core.py                 ← EvalTask + CompletionAlgorithm protocols, make_eval_step()
      schema.py               ← PromptRow / CompletionRow / GradeRow TypedDicts + readers
    tasks/   gsm8k, gsm8k_masked, math500, dummy, gsm8k_qa (mine)
    algorithms/  iid, iid_zephyr, mask, rerank
  models/   delphi (Rohith), delphi_extra + midtrain (mine)
  sft/      tokenize, train, data/gsm8k_qa  (mine, eval-half-shaped but not symmetric)
  run_delphi_masked_gsm8k_iid.py   ← Rohith's launcher
  run_one_sft_gsm8k_qa.py / run_delphi_sft_gsm8k_qa.py / smoke_*  (mine)
```

**What's genuinely clean (Rohith's part):**

- Three-stage contract enforced by TypedDicts: `task → prompts.jsonl.gz → completions.jsonl.gz → grades.jsonl.gz`. Each artifact is gzip JSONL with stable schemas.
- Protocol-based decoupling: any `EvalTask` × any `CompletionAlgorithm`. `make_eval_step()` is the only composer; everything else is plain data.
- Prompt-step identity is stable across model evals → same task config reuses prompt artifacts (nice cache property).
- Crisp separation between **versioned** (semantic — affects identity) and **operational** (workers, chunk_size, grade_workers) config — the executor hash only sees the former.
- Single grader (lm-eval `flexible_extract` / `strict_match`) shared across `gsm8k.py`, `gsm8k_masked.py`, and `gsm8k_qa.py` via `make_grade_step = GSM8KTask.make_grade_step`.

**What's not clean:**

- **No reducer.** `grades.jsonl.gz` is the final artifact — no aggregation step, no plotting. A 110-cell sweep gives you 110 files and a manual loop to compute the matrix.
- **My SFT half is not symmetric.** Tasks/algorithms are clean composables; SFT is a one-off `build_sft_step(slug, rel_path, tokenize_step)` that returns a single step. No protocol, no equivalent of the three-stage contract for training. If we add more recipes (DPO, RLHF), the current shape won't generalize cleanly.
- **Framework bleed in algorithms.** `iid.py` carries vLLM env vars and a `ragged_paged_attention` kernel monkey-patch inside the worker — those are device-specific facts leaking into what should be a transport-agnostic completion algorithm.
- **`iid.py` vs `iid_zephyr.py` divergence.** Two parallel implementations of "IID sampling" with different sharding strategies; only one has a launcher. Drift risk.
- **`mask.py` and `rerank.py` are unlaunched.** Real code, no launcher, no tests — framework-shaped scaffolding that nothing exercises end-to-end.
- **My modifications to `iid.py` are not opt-in.** The Qwen3 `hf_overrides` and RPA kernel rebase silently change behavior for *every* eval, including Rohith's masked-GSM8K relaunches. Should be flag-gated.

**One-line summary:** Rohith built a tight three-stage protocol for eval-as-data; my SFT addition uses the protocol's outputs but doesn't extend it, so the package has a clean half and an attached half.

### 2026-05-26 — 1 packed epoch SFT pass (pack=True, 19 steps)

Re-ran the same 2-cell 1e20 comparison (SFT from base pretrain vs SFT from math-midtrain), now with **sequence packing on AND only 1 epoch of data coverage**. Both cells were SFT'd; the only thing varied between rows is the SFT-init checkpoint, identical to the 2026-05-14 setup. The new axis swept across the three regimes is **SFT budget × packing**: today's run is ~6× fewer optimizer steps than the 2026-05-14 1-unpacked-epoch run, with the same ~7.5k Q+A pair-views of data coverage; and ~19× fewer optimizer steps than the original 19-packed-epoch run.

Question: does the math-midtrain SFT-init uplift survive when the optimizer trajectory is squeezed to just 19 gradient updates?

#### Config delta vs prior runs

| | 19 packed epochs (2026-05-14) | 1 unpacked epoch (2026-05-14) | 1 packed epoch (today) |
|---|---|---|---|
| `pack` | `True` | `1` (no packing) | `True` |
| `num_train_steps` (= optimizer steps) | 360 | 117 | **19** |
| Q+A pair-views per step | ~6.3 (packed) | 1 (unpacked) | ~6.3 (packed) |
| Total Q+A pair-views | ~145k (≈19 epochs of 7.5k) | ~7.5k (≈1 epoch) | ~7.5k (≈1 epoch) |
| `SFT_OUTPUT_PREFIX` | `gsm8k_qa` | `gsm8k_qa_nopack_1ep` | `gsm8k_qa_pack_1ep` |
| Eval cell hash | `grade-b0066d` (base), `grade-94dfe4` (math) | `grade-17a6eb` (base), `grade-537f41` (math) | `grade-4d9231` (base), `grade-aa2a87` (math) |

At 19 steps, `warmup=0.03 × 19 ≈ 0.6 steps` — effectively no warmup. The linear schedule covers essentially the entire run.

#### Headline results (this run only)

**Both rows are post-SFT.** Same eval recipe as before (256 problems × 32 IID samples, T=0.6, top_p=1.0, top_k=1000, max_tokens=512, seed=42, zero-shot Q+A prompt, `flexible_extract` grader).

| SFT-init checkpoint | Mean accuracy | Pass@32 |
|---|---:|---:|
| base 1.9B pretrain (slug `1e20_iso`) | **1.33%** (109/8192) | **20.70%** (53/256) |
| math-midtrain 1.9B (slug `1e20_p33m67_lr0.67`) | **7.54%** (618/8192) | **59.77%** (153/256) |

Uplift from math-midtrain SFT-init vs base-pretrain SFT-init: **5.7× mean accuracy, 2.9× pass@32**. Same direction as prior regimes, smaller multiplier.

#### Pass@32 across all three SFT regimes at 1e20

|  | pack=False, 117 steps (1 unpacked epoch) | pack=True, 19 steps (1 packed epoch) | pack=True, 360 steps (19 packed epochs) |
|---|---:|---:|---:|
| base-init (`1e20_iso`) | 30.47% | 20.70% | 37.50% |
| math-init (`1e20_p33m67_lr0.67`) | 65.23% | 59.77% | 66.41% |

#### Reading

- **Absolute accuracy tracks optimizer-step count, not pair-view count.** The 1-packed-epoch run (19 steps) and the 1-unpacked-epoch run (117 steps) see the same ~7.5k Q+A pair-views but differ by ~6× in gradient updates. Both mean accuracy and pass@32 drop meaningfully when steps go down: packing is data-efficient per FLOP but undertrained per optimizer step at fixed epoch count.
- **The math-midtrain SFT-init uplift is robust across all three regimes** (5.7–7.5× on mean, 2.0–2.9× on pass@32). The midtrain advantage is not a function of SFT budget — it shows up consistently whether the optimizer takes 19, 117, or 360 SFT steps.
- **Math-midtrain init is more pack-robust than base-pretrain init.** At 1 epoch, base-init loses ~10pp pass@32 when you turn packing on (30.47% → 20.70%); math-init loses only ~5.5pp (65.23% → 59.77%). Reasonable hypothesis: the math-midtrained checkpoint already encodes GSM8K-shaped reasoning patterns, so it needs fewer optimizer steps to specialize.
- **Pass@32 generally tracks mean accuracy but with much smaller dynamic range.** The math-init row spans 59.77 → 66.41 (Δ ≈ 6.6pp) across 19→360 optimizer steps; the same range covers 7.54 → 18.82 (Δ ≈ 11.3pp) on mean. Pass@32 is closer to a "ceiling-of-reachable-problems" metric at this sampling budget.

#### Operational notes

- **Job IDs**:
  - `/ahmedah/aa-sft-1e20-iso-pack-1ep` — 2 preempts, 3 attempts, ~60 min total wall-clock.
  - `/ahmedah/aa-sft-1e20-p33m67-pack-1ep` — 0 preempts, 40:23 end-to-end.
- **`WANDB_API_KEY` gotcha when launching from skampere3.** The Mac launches in the 2026-05-14 entry had the key baked into local env / config somewhere; skampere3 does not. First two job submissions today failed at config validation (`ValueError: WANDB_API_KEY must be set in the environment ...` raised inside `marin.training.training._check_for_wandb_key`). Fix: pass `-e WANDB_API_KEY "$WANDB_API_KEY"` explicitly to `iris job run`, extracting the value via Python one-liner from `/lfs/skampere3/0/ahmedah/code/marin/.marin.yaml` (memory `reference_marin_yaml_secrets.md` — never `Read`/`cat` that file). Worth fixing in the launcher or the iris config so future skampere3 launches don't need this step.
- **Iris from skampere3 still needs the IAP-tunnel + pyenv 3.12.0 workaround** in memory `iris_cluster_remote_access.md`. The controller IP has rotated since that memory was written (was `34.171.120.166`, now `34.27.183.11`); the connection method is unchanged. Tunnel listens on `localhost:10000`; iris CLI invocation must include `--controller-url=http://localhost:10000`.
- **Interactive priority is the iris default.** Passing `--priority interactive` is redundant but harmless. Preemptions today were from another interactive job (`/romain/served-qwen3-humaneval-full-4`), not from production-band displacement.

#### Code state

Two uncommitted edits on `aa_fork_rk_downstream` after these runs (both shipped via the iris workspace bundle at submission time, ~6.4 MB):

| File | Change |
|---|---|
| `experiments/downstream_scaling/sft/tokenize.py` | `pack=1` → `pack=True`. Comment block rewritten to reflect new math (19 steps × 64 batch × ~6.3 pair-views per packed sequence ≈ 1 epoch of GSM8K's 7,473-problem train split). |
| `experiments/downstream_scaling/sft/train.py` | `num_train_steps` 117 → 19; `steps_per_hf_export` 117 → 19; `steps_per_eval` 30 → 10 (one eval mid-training); `SFT_OUTPUT_PREFIX` bumped from `gsm8k_qa_nopack_1ep` to `gsm8k_qa_pack_1ep` so eval doesn't cache-hit prior grades. |

#### GCS artifact map for the 1-packed-epoch run

```
# SFT'd HF checkpoints (initialization checkpoints unchanged from prior runs)
gs://marin-us-east5/checkpoints/downstream_scaling/sft/delphi/gsm8k_qa_pack_1ep/1e20_iso/hf/step-18/
gs://marin-us-east5/checkpoints/downstream_scaling/sft/delphi/gsm8k_qa_pack_1ep/1e20_p33m67_lr0.67/hf/step-18/

# Grades
gs://marin-us-east5/downstream_scaling/evals/delphi_sft/gsm8k_qa_pack_1ep/iid/1e20_iso/grade-4d9231/grades.jsonl.gz             # 1.33% mean / 20.70% pass@32
gs://marin-us-east5/downstream_scaling/evals/delphi_sft/gsm8k_qa_pack_1ep/iid/1e20_p33m67_lr0.67/grade-aa2a87/grades.jsonl.gz   # 7.54% mean / 59.77% pass@32
```

#### Eval methodology — non-standard, deliberate

Worth being explicit, since the screenshot of an earlier session's results table re-surfaced today and the wording confused things. **Our GSM8K eval is not the standard lm-eval-harness GSM8K** in several ways:

| Aspect | This pipeline | Standard lm-eval-harness GSM8K |
|---|---|---|
| Prompt | Zero-shot: `Question: {q}\nAnswer:` | 5-shot CoT |
| Decoding | T=0.6, top_p=1.0, top_k=1000, n=32 samples per problem | Greedy (T=0), n=1 sample |
| Max gen tokens | 512 | 256–512 |
| Stop tokens | `("Question:", "</s>", "<|im_end|>", "<|end_of_text|>")` | typically `"\n\n"` |
| Scored filter | `flexible_extract` (last number in response) | `strict_match` (literal `#### N`) |
| Problems | first 256 of GSM8K test split | full 1,319 |
| Reported | mean accuracy + pass@32 | pass@1 |

The setup is internally consistent and well-suited to comparing two SFT-init conditions at fixed eval recipe — it amplifies the signal by drawing 32 samples per problem. **It is not comparable to GSM8K leaderboard numbers**, which use the greedy 5-shot strict-match recipe and report pass@1. `TEMPERATURE=0.6` is inherited from Rohith's `run_delphi_masked_gsm8k_iid.py:30` without a recorded rationale; most likely Llama-3-Instruct's official inference default (Meta's model card pins T=0.6) or pass@k-paper convention.

#### Resolved-by-now from the 2026-05-14 open-questions list

- Item 3 ("Code is not committed") — pushed as `b7ca6bf5f` on 2026-05-21.
- Item 4 sub-bullet ("Disentangle packing from epochs") — partially addressed by today's 1-packed-epoch run. Cleanly attributing the 19→1-epoch drop to packing vs to fewer optimizer steps would require running `pack=1, num_train_steps=2223` (19 unpacked epochs) — not done.

#### Still open after today

1. **No pre-SFT zero-shot eval datapoints.** Same gap flagged in the 2026-05-14 "Missing baselines" section. Cell A: bare base 1.9B → `GSM8KQATask`. Cell B: bare math-midtrain 1.9B → `GSM8KQATask`. Each is one eval-only step (vLLM only, no retrain), ~22 min on v5p-8 in parallel.
2. **No `p67m33` (replay-heavy) datapoint at 1e20.** The 3-way comparison (base / math-midtrain / replay-midtrain) at 1e20 is still 1 cell missing. 1 additional SFT cell, ~1 TPU-hour. Disambiguates "math content drives uplift" from "any midtraining drives uplift."
3. **No 1e21 or 1e22 datapoints.** Scale axis untested. 6 cells total (base + 2 midtrains × 2 scales); 1e22 needs v5p-32 per `SFT_RESOURCES` in `sft/tokenize.py`.
4. **Compatibility eval pass against Rohith's `MaskedGSM8KTask, mask_fraction=0.0, num_fewshot=5`.** Re-eval the existing 4 post-SFT checkpoints (gsm8k_qa and gsm8k_qa_nopack_1ep prefixes, both slugs) for apples-to-apples vs Rohith's 110-cell matrix. ~22 min × 4 cells.
5. **`_run_iris_job` inline-execution mystery** (unresolved from 2026-05-13 smoke debugging). SFT runs inline in the coordinator instead of dispatching as a child Iris job, contrary to `step_runner.py:340-346`. Means each cell is its own `iris job run` from a TPU-equipped coordinator; the 16-cell sweep can't run parallel cells from one coordinator. PR #5279 (Russell) is the leading hypothesis.
6. **No reducer / no plotting.** Computing the 2026-05-26 3-regime pass@32 matrix above required a one-off Python script that pulls each `grades.jsonl.gz` via `gcloud storage cat`. Worth a small read-only summary step in the framework.

#### TL;DR for the next agent (current as of EOD 2026-05-26)

You're picking up `aa_fork_rk_downstream` after a 3rd 1e20 SFT regime — sequence packing turned on with `num_train_steps=19` (1 packed epoch). **Three regimes done, two slugs each. The math-midtrain SFT-init uplift is robust across all three** (5.7–7.5× on mean accuracy, 2.0–2.9× on pass@32). Absolute accuracy tracks optimizer-step count, not pair-view count.

Working launcher: `experiments/downstream_scaling/run_one_sft_gsm8k_qa.py --slug <slug>`. Working iris invocation (logbook section above, plus the new `-e WANDB_API_KEY "$WANDB_API_KEY"` requirement when launching from skampere3). The 5 framework bugs are still fixed; the new edits to `tokenize.py` + `train.py` for today's run are uncommitted on the branch (tree dirty).

**Highest-value next moves**, in order of value-for-cost:
1. Fire the two pre-SFT zero-shot eval cells (base + math-midtrain) to fill the "Missing baselines" gap. ~22 min × 2 cells, no retrain. Lets you separate "what midtrain alone did" from "what the SFT pass added."
2. Add `1e20_p67m33_lr0.33` (replay-heavy) — completes the 1e20 3-way comparison. 1 SFT cell, ~1 TPU-hour.
3. Compatibility eval pass against `MaskedGSM8KTask, mask_fraction=0.0` for apples-to-apples vs Rohith. ~22 min × 4 cells.
4. Scale to 1e21 + 1e22 for both treatments — the paper-grade scaling-law test. 6 cells; 1e22 needs v5p-32.
5. Drop midtrain checkpoints into Rohith's mask_00 eval across the full ladder — see the plan in the next section. 108 vLLM-only cells, no SFT.

### 2026-05-26 (continued) — Plan: midtrain matrix via Rohith's mask_00 (drafted, not yet fired)

#### Goal

Extend Rohith's existing mask_00 baseline (10 base-pretrain cells) with a midtrain dimension: **9 scales × 3 mixes × 4 LRs = 108 eval-only cells.** Each cell runs the corresponding midtrain HF checkpoint through `MaskedGSM8KTask(mask_fraction=0.0, num_fewshot=5)` with the byte-identical sampling config from `evals/run_delphi_masked_gsm8k_iid.py` (n=32, T=0.6, top_p=1.0, top_k=1000, max_tokens=512, seed=42). No SFT, no retraining. Output: a `(scale, mix, lr) → pass@32` matrix directly comparable to Rohith's 10 base-pretrain mask_00 numbers and to itself across axes.

The matrix lets us see: (a) the scaling-law shape of the midtrain effect along the FLOPs axis, (b) the LR sensitivity per (scale, mix), (c) which mix dominates at which scale, and (d) the gap between each midtrain cell and Rohith's matched base-pretrain mask_00.

#### Methodology used to pick canonical checkpoints

GCS-naïve enumeration found extra hashes and replicates per (scale, mix, lr) for many cells. To pick exactly one canonical checkpoint per cell:

1. Pulled all 189 runs from the `marin-community/delphi-midtraining` W&B project.
2. Parsed run names into `(scale, mix, lr)` tuples; grouped duplicates.
3. For each duplicate group, picked the W&B `state=finished` variant and discarded `state=crashed`/`failed` variants.
4. Cross-checked GCS file counts: `finished` runs have ≥9 HF step subdirs; `failed`/stub runs have 0 HF step subdirs. Every match held.
5. Verified all 108 chosen rel-paths exist in `gs://marin-us-east5/checkpoints/` with non-empty `hf/` subdirs. Pass: 108/108.

#### Resolutions of the four open questions

1. **3e18 replicates** — **use `a003`**. W&B has 12 `a003` runs all `finished` for 3e18 (3 mixes × 4 LRs). No `a001`/`a002` 3e18 runs were logged in either `delphi-midtraining` or `marin` projects. GCS confirms: `a001` dirs have 10 HF subdirs each but no W&B record (orphans, possibly from a pre-logging attempt); `a002` dirs have 0 HF subdirs (stub/failed early); `a003` dirs have 10 HF subdirs and match the W&B record. User's instinct that "if we have `a003` something went wrong" is correct — at 3e18 the first two attempts had to be redone.

2. **1e21 / 1e22 hash duplicates** — clean: every duplicate has exactly one `finished` variant and one or more `crashed`/`failed` variants. GCS file counts confirm:
   - 1e22 `p33m67_lr0.33`: `abdeba` (0 HF subdirs, W&B failed) vs **`e9132105` (11 HF subdirs, finished)** ← use long hash
   - 1e22 `p33m67_lr0.5`: `91bcb9` failed vs **`0eeca70d` finished**
   - 1e22 `p33m67_lr0.67`: `089468` failed vs **`54770ae7` finished** (already in registry — correct)
   - 1e22 `p67m33_lr0.33`: `fe948f` failed vs **`4e8cc7a7` finished** (already in registry — correct)
   - 1e22 `p67m33_lr0.5`: `8e526a` failed vs **`f60cb12a` finished**
   - 1e22 `p67m33_lr0.67`: `285e1b` failed vs **`3c17740e` finished**
   - 1e21 `p67m33_lr0.5`: **`114e49` finished (9 HF subdirs)** vs `fdc4ebf1` crashed (7 HF subdirs) ← use short hash
   - 1e21 `p67m33_lr0.67`: **`ecbd27` finished** vs `99752407` crashed
   
   Interesting pattern: 1e22 the long-hash variants are canonical; 1e21 the short-hash variants are canonical. The W&B/GCS check resolves it deterministically — don't pick by hash length. **No identical-hash collisions were found anywhere.**

3. **`1e21_p67m33_lr0.33` region** — the full HF checkpoint is in **`gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.33-ab4e64/`** (10 HF step subdirs). The `gs://marin-us-central2/...` directory with the same name has only a `.executor_info` stub and no `hf/` subdir — it's metadata residue, not a real checkpoint. **No cross-region handling needed**, contrary to my earlier draft. User's correction was right.

4. **Off-ladder bases** — out of scope per user direction. Skipping all `delphi-1e20-iso-d2048-L21-*` (4p94b / 10b / 20b token-budget variants at the 1e20-iso stand-in) and skipping pure-math midtrains `delphi-1e20-iso-d2048-L21-math-*` and `delphi-1e21-v5-math-*`. The plan uses Rohith's 9 canonical ladder points (3e18 → 1e22, excluding 1e23) with the canonical token budgets per scale (k0p20 for 3e18→3e20, 9p25b for 1e21, 32p07b for 1e22).

#### Canonical checkpoint inventory (all 108 cells, verified)

Naming + replicate conventions:

| Scale | Token-budget family | Replicate / hash policy |
|---|---|---|
| 3e18 | `k0p20-a003` | a003 (W&B-verified canonical) |
| 9e18 | `k0p20-a002` | a002 |
| 2e19 | `k0p20-a002` | a002 |
| 3e19 | `k0p20-a002` | a002 |
| 9e19 | `k0p20-a002` | a002 |
| 2e20 | `k0p20-a001` | a001 |
| 3e20 | `k0p20-a001` | a001 |
| 1e21 | `9p25b-{hash}` | per-cell hash (12 distinct, all `finished` in W&B) |
| 1e22 | `32p07b-{hash}` | per-cell hash (12 distinct, all `finished` in W&B) |

Full slug → relative-path mapping (rel-path is appended to `gs://marin-us-east5/`, all paths end in `hf/`):

##### 3e18 — d1024-L11, `k0p20-a003`
| | lr0.33 | lr0.5 | lr0.67 | lr0.83 |
|---|---|---|---|---|
| p33m67 | `checkpoints/delphi-3e18-p33m67-k0p20-lr33-a003/hf` | `...-lr50-a003/hf` | `...-lr67-a003/hf` | `...-lr83-a003/hf` |
| p50m50 | `checkpoints/delphi-3e18-p50m50-k0p20-lr33-a003/hf` | `...-lr50-a003/hf` | `...-lr67-a003/hf` | `...-lr83-a003/hf` |
| p67m33 | `checkpoints/delphi-3e18-p67m33-k0p20-lr33-a003/hf` | `...-lr50-a003/hf` | `...-lr67-a003/hf` | `...-lr83-a003/hf` |

##### 9e18 — d1152-L12, `k0p20-a002`
| | lr0.33 | lr0.5 | lr0.67 | lr0.83 |
|---|---|---|---|---|
| p33m67 | `checkpoints/delphi-9e18-p33m67-k0p20-lr33-a002/hf` | `...-lr50-a002/hf` | `...-lr67-a002/hf` | `...-lr83-a002/hf` |
| p50m50 | `checkpoints/delphi-9e18-p50m50-k0p20-lr33-a002/hf` | `...-lr50-a002/hf` | `...-lr67-a002/hf` | `...-lr83-a002/hf` |
| p67m33 | `checkpoints/delphi-9e18-p67m33-k0p20-lr33-a002/hf` | `...-lr50-a002/hf` | `...-lr67-a002/hf` | `...-lr83-a002/hf` |

##### 2e19 — d1408-L15, `k0p20-a002`
| | lr0.33 | lr0.5 | lr0.67 | lr0.83 |
|---|---|---|---|---|
| p33m67 | `checkpoints/delphi-2e19-p33m67-k0p20-lr33-a002/hf` | `...-lr50-a002/hf` | `...-lr67-a002/hf` | `...-lr83-a002/hf` |
| p50m50 | `checkpoints/delphi-2e19-p50m50-k0p20-lr33-a002/hf` | `...-lr50-a002/hf` | `...-lr67-a002/hf` | `...-lr83-a002/hf` |
| p67m33 | `checkpoints/delphi-2e19-p67m33-k0p20-lr33-a002/hf` | `...-lr50-a002/hf` | `...-lr67-a002/hf` | `...-lr83-a002/hf` |

##### 3e19 — d1536-L16, `k0p20-a002`
| | lr0.33 | lr0.5 | lr0.67 | lr0.83 |
|---|---|---|---|---|
| p33m67 | `checkpoints/delphi-3e19-p33m67-k0p20-lr33-a002/hf` | `...-lr50-a002/hf` | `...-lr67-a002/hf` | `...-lr83-a002/hf` |
| p50m50 | `checkpoints/delphi-3e19-p50m50-k0p20-lr33-a002/hf` | `...-lr50-a002/hf` | `...-lr67-a002/hf` | `...-lr83-a002/hf` |
| p67m33 | `checkpoints/delphi-3e19-p67m33-k0p20-lr33-a002/hf` | `...-lr50-a002/hf` | `...-lr67-a002/hf` | `...-lr83-a002/hf` |

##### 9e19 — d1792-L18, `k0p20-a002`
| | lr0.33 | lr0.5 | lr0.67 | lr0.83 |
|---|---|---|---|---|
| p33m67 | `checkpoints/delphi-9e19-p33m67-k0p20-lr33-a002/hf` | `...-lr50-a002/hf` | `...-lr67-a002/hf` | `...-lr83-a002/hf` |
| p50m50 | `checkpoints/delphi-9e19-p50m50-k0p20-lr33-a002/hf` | `...-lr50-a002/hf` | `...-lr67-a002/hf` | `...-lr83-a002/hf` |
| p67m33 | `checkpoints/delphi-9e19-p67m33-k0p20-lr33-a002/hf` | `...-lr50-a002/hf` | `...-lr67-a002/hf` | `...-lr83-a002/hf` |

##### 2e20 — d2048-L21, `k0p20-a001`
| | lr0.33 | lr0.5 | lr0.67 | lr0.83 |
|---|---|---|---|---|
| p33m67 | `checkpoints/delphi-2e20-p33m67-k0p20-lr33-a001/hf` | `...-lr50-a001/hf` | `...-lr67-a001/hf` | `...-lr83-a001/hf` |
| p50m50 | `checkpoints/delphi-2e20-p50m50-k0p20-lr33-a001/hf` | `...-lr50-a001/hf` | `...-lr67-a001/hf` | `...-lr83-a001/hf` |
| p67m33 | `checkpoints/delphi-2e20-p67m33-k0p20-lr33-a001/hf` | `...-lr50-a001/hf` | `...-lr67-a001/hf` | `...-lr83-a001/hf` |

##### 3e20 — d2304-L23, `k0p20-a001`
| | lr0.33 | lr0.5 | lr0.67 | lr0.83 |
|---|---|---|---|---|
| p33m67 | `checkpoints/delphi-3e20-p33m67-k0p20-lr33-a001/hf` | `...-lr50-a001/hf` | `...-lr67-a001/hf` | `...-lr83-a001/hf` |
| p50m50 | `checkpoints/delphi-3e20-p50m50-k0p20-lr33-a001/hf` | `...-lr50-a001/hf` | `...-lr67-a001/hf` | `...-lr83-a001/hf` |
| p67m33 | `checkpoints/delphi-3e20-p67m33-k0p20-lr33-a001/hf` | `...-lr50-a001/hf` | `...-lr67-a001/hf` | `...-lr83-a001/hf` |

##### 1e21 — adamh-scaling-ladder 3.4B, `9p25b-{hash}` (per-cell finished-in-W&B canonical hash)
| | lr0.33 | lr0.5 | lr0.67 | lr0.83 |
|---|---|---|---|---|
| p33m67 | `checkpoints/delphi-1e21-p33m67-9p25b-lr0.33-58ebcb/hf` | `...-lr0.5-efbc63/hf` | `...-lr0.67-9cf8da/hf` | `...-lr0.83-0cb048/hf` |
| p50m50 | `checkpoints/delphi-1e21-p50m50-9p25b-lr0.33-bccff4/hf` | `...-lr0.5-973c46/hf` | `...-lr0.67-7e82b3/hf` | `...-lr0.83-f9edd2/hf` |
| p67m33 | `checkpoints/delphi-1e21-p67m33-9p25b-lr0.33-ab4e64/hf` | `...-lr0.5-114e49/hf` | `...-lr0.67-ecbd27/hf` | `...-lr0.83-a1a261/hf` |

##### 1e22 — adamh-scaling-ladder 9.7B, `32p07b-{hash}` (per-cell finished-in-W&B canonical hash; long-hash variants for duplicates)
| | lr0.33 | lr0.5 | lr0.67 | lr0.83 |
|---|---|---|---|---|
| p33m67 | `checkpoints/delphi-1e22-p33m67-32p07b-lr0.33-e9132105/hf` | `...-lr0.5-0eeca70d/hf` | `...-lr0.67-54770ae7/hf` | `...-lr0.83-78fd44/hf` |
| p50m50 | `checkpoints/delphi-1e22-p50m50-32p07b-lr0.33-c43ada/hf` | `...-lr0.5-ecfa99/hf` | `...-lr0.67-e78260/hf` | `...-lr0.83-3c9f70/hf` |
| p67m33 | `checkpoints/delphi-1e22-p67m33-32p07b-lr0.33-4e8cc7a7/hf` | `...-lr0.5-f60cb12a/hf` | `...-lr0.67-3c17740e/hf` | `...-lr0.83-d35daa/hf` |

#### Sluggification

`{scale}_{mix}_lr{X.YY}` — e.g., `3e18_p33m67_lr0.33`, `1e22_p67m33_lr0.83`. 108 unique slugs.

#### Implementation sketch

1. **Expand `experiments/downstream_scaling/models/midtrain.py`** from 6 → 108 entries. Build programmatically from a small table — three dicts (`REP`, `HASHES_1e21`, `HASHES_1e22`) plus a `_build_registry()` function that emits the 108-entry dict. The 6 existing entries (`1e20_*_lr*` variants) stay as-is since they describe a *different* base architecture (the 1e20-iso d2048-L21 4p94b family) and are still wired into the SFT pipeline.
2. **New launcher**: `experiments/downstream_scaling/evals/run_midtrain_masked_gsm8k_iid.py`. Single-cell launcher mirroring `run_one_sft_gsm8k_qa.py` — takes `--slug` and runs one eval. Internal call: `make_eval_step(name=f"downstream_scaling/evals/delphi_midtrain/masked_gsm8k_mask00/{slug}", model_path=InputName.hardcoded(registry[slug]), task=MaskedGSM8KTask(MaskedGSM8KTaskConfig(tokenizer_path=llama3_tokenizer, mask_fraction=0.0, num_fewshot=5, fewshot_seed=1234, n_problems=256)), alg=make_algorithm("v5p-8"))`.
3. **Reuse existing `iid.py` fixes**: the `hf_overrides={"architectures": ["Qwen3ForCausalLM"]}` and RPA-kernel rebase are already committed and apply to every vLLM load — midtrain HFs need both (same Levanter-export-with-`LlamaForCausalLM`-architectures-key issue as our SFT'd 1e20_iso).
4. **Launch pattern**: same iris invocation as the SFT runs but with `python experiments/downstream_scaling/evals/run_midtrain_masked_gsm8k_iid.py --slug <slug>`. From skampere3: IAP tunnel + pyenv 3.12.0 + explicit `-e WANDB_API_KEY` per memory `iris_cluster_remote_access.md` (and the WANDB gotcha logged earlier today).
5. **Reducer**: small standalone Python script that pulls all 108 `grades.jsonl.gz`, computes pass@32 and mean accuracy per cell, emits a CSV keyed by `(scale, mix, lr)`. Can defer; the per-cell artifacts are useful even without aggregation.

#### Output paths

```
gs://marin-us-east5/downstream_scaling/evals/delphi_midtrain/masked_gsm8k_mask00/{slug}/
  ├── completions-{hash}/
  └── grade-{hash}/grades.jsonl.gz
```

Plus prompt artifact at `gs://marin-us-east5/downstream_scaling/evals/prompts/masked_gsm8k-42fd11/prompts.jsonl.gz` — already exists (Rohith's mask_00 prompts), will be cache-hit by all 108 cells since `MaskedGSM8KTaskConfig` is identical across them.

#### Cost / wall-clock estimate

- Per cell on v5p-8: ~22 min completions + ~11 min grade = **~33 min uninterrupted**.
- 108 cells × 33 min = **~59 TPU-hours** if serial.
- Parallel across us-east5-a + us-central1-a (per memory `feedback_multi_region_tpu_launch.md`) with 3 jobs per region in flight: **~10–12 wall-clock hours** total, more if east5-a stays preempty.
- Preempt risk: capacity in east5-a has been moderate. Allocate `--max-retries 2` per cell; Iris resumes vLLM jobs cleanly from chunk SUCCESS files. Plan for ~1.5× expected wall-clock to account for preempts.

#### Launch strategy

1. **Smoke first**: pick one small cell, run end-to-end, verify the grade artifact lands. Recommend **`3e18_p33m67_lr0.5`** — smallest model, common LR, ~25 min uninterrupted.
2. After smoke passes, fire **wave 1** in parallel across both zones: the 12 small-scale cells (3e18 row). 4 cells per region, 3 in flight per region = ~3 cell-rounds per region.
3. Wave 2: 9e18, 2e19, 3e19, 9e19 = 48 cells. Same parallelism.
4. Wave 3: 2e20, 3e20 = 24 cells.
5. Wave 4: 1e21, 1e22 = 24 cells. 1e22 (9.7B params) still fits v5p-8 for vLLM inference — no resource bump needed.
6. After wave 1 lands (~12 cells), run the reducer to surface partial results. Don't wait for all 108 before looking.

#### Validation before launch

- Confirm `MaskedGSM8KTaskConfig(mask_fraction=0.0)` resolves to a `versioned()` hash that matches Rohith's existing mask_00 prompts artifact at `prompts/masked_gsm8k-42fd11/`. If not, the new launch will write a fresh prompt artifact (still correct, just doesn't reuse the existing one).
- Confirm tokenizer path resolves: `llama3_tokenizer` should be the same `meta-llama/Meta-Llama-3.1-8B` used by Rohith. If the midtrain checkpoints were trained with a different tokenizer (e.g., Qwen3's), the mask function in `gsm8k_masked.py:77-86` might tokenize the mask token differently, which could change identity. **Check before firing the 108-cell wave** — verify on the smoke cell that the masked solution looks sensible.

#### Open decisions still pending

- **Reducer scope**: just pass@32 + mean_acc per cell? Or also mean ± stderr across the 32 samples per problem? My recommendation: the simple `(scale, mix, lr) → (pass@32, mean_acc, mean_correctness_per_sample)` CSV, with the option to add error bars later by reading the same `grades.jsonl.gz`.
- **Comparison baseline**: do we want to side-by-side compare each midtrain cell against Rohith's matched base mask_00 cell at the same scale? That's a delta-per-cell view. The base mask_00 numbers live at `gs://marin-us-east5/downstream_scaling/evals/delphi/masked_gsm8k/iid/mask_00/{slug}/grade-*/grades.jsonl.gz` already. Worth folding into the reducer.
- **Plotting**: out of scope for this plan unless you want it; pass@32-vs-FLOPs lines per mix is the obvious chart. Can be a follow-up.
