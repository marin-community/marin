# Perplexity Gap Analysis Workflow

This page documents the end-to-end workflow, as of 2026-05, for adding a
raw or supervised perplexity dataset, scoring Marin against another model, and
publishing the updated dashboard and heatmaps.

Use this when the goal is to answer "where is Marin better or worse in bits per
byte?" on raw text slices or supervised target-only text slices. For standard
task accuracy and generation evals, use
[Running Evaluations with Marin](../tutorials/run-lm-evals.md).

## Overview

The workflow has two repositories:

- `marin`: dataset registration, ETL, model scoring, and gap report jobs.
- `marin-community.github.io`: static dashboard data, copied report artifacts,
  sampled document views, and per-character heatmap payloads.

The scoring path is tokenizer-independent. Each model tokenizes the same UTF-8
documents with its own tokenizer, then Levanter projects token losses back onto
document bytes. Positive `gap_bpb` means Marin has higher loss than the
comparison model. Negative `gap_bpb` means Marin has lower loss.

The main implementation points are:

- `lib/marin/src/marin/evaluation/perplexity_gap.py`
- `lib/levanter/src/levanter/main/perplexity_gap.py`
- `lib/levanter/src/levanter/analysis/model_perplexity.py`
- `lib/levanter/src/levanter/analysis/perplexity_gap.py`
- `experiments/evals/model_perplexity_gap_suite.py`
- `experiments/evals/perplexity_gap_registry.py`
- `experiments/exp_model_perplexity_gap_coverage_matrix.py`
- `analysis/perplexity-gap/` in `marin-community.github.io`

## 1. Register the dataset provider

Prefer adding a provider module under `experiments/evals/` unless the data
belongs to an existing provider. The provider should expose a function named
like `<area>_raw_validation_sets()` and return a mapping from stable dataset keys
to `RawTextEvaluationDataset` values.

For raw language modeling text, use `raw_text_dataset`:

```python
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset


def example_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    return {
        "long_tail_ppl/example/source_a": raw_text_dataset(
            "gs://marin-us-central1/raw/example/source_a/heldout.jsonl.gz",
            tags=("long_tail_ppl", "issue:NNNN", "source:example"),
        ),
    }
```

For Hugging Face-backed raw text, pass `HfDatasetSpec` to `raw_text_dataset`.
The committed `HfDatasetSpec` API supports dataset id and config name only. Do
not pass a revision argument here; if immutable pinning matters, materialize the
heldout split to a versioned GCS path or add revision support in the same PR.

```python
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.processing.tokenize import HfDatasetSpec


def example_hf_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    source = HfDatasetSpec(
        id="marin-community/example-ppl",
        name="format_variant",
    )
    return {
        "synthetic_example_ppl/format_variant": raw_text_dataset(
            source,
            tags=("synthetic_example_ppl", "issue:NNNN"),
        ),
    }
```

For supervised target-only scoring, use `supervised_text_dataset` and write rows
with separate input and target fields. This scores only the target bytes while
still allowing the input bytes to condition the model.

```python
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, supervised_text_dataset
from marin.processing.tokenize import HfDatasetSpec


def example_supervised_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    source = HfDatasetSpec(
        id="marin-community/example-ppl",
        name="format_variant",
    )
    return {
        "synthetic_example_ppl/format_variant": supervised_text_dataset(
            source,
            input_key="input",
            target_key="target",
            tags=("synthetic_example_ppl", "issue:NNNN", "loss:target_only"),
        ),
    }
```

Dataset keys are part of the report and dashboard API. Keep them stable,
hierarchical, and explicit. Good keys look like `structured_text/totto`,
`synthetic_delimiter_format_ppl/table_rows/pipe_rows`, or
`long_tail_ppl/code_ecosystem/stack_v2_json`.

Tags drive dashboard rollups. Include at least the family tag and issue tag.
Examples: `("long_tail_ppl", "issue:5254", "code_ecosystem")` or
`("synthetic_numeric_format_prompt_ablation_ppl", "issue:5614")`.

## 2. Materialize or pin the data

Use one of two source patterns.

If the dataset already lives on Hugging Face and can be read directly, use
`HfDatasetSpec`. If the dataset must be pinned to an exact revision, stage the
heldout rows to a versioned GCS path or extend the HF source API first.

If the source needs ETL, create an `ExecutorStep` that writes a small heldout
artifact to GCS. Raw-text scoring expects JSONL or JSONL gzip rows with a `text`
field:

```json
{"text": "..."}
```

Supervised target-only scoring expects rows with input and target fields:

```json
{"input": "...", "target": "..."}
```

Recommended heldout sizing:

- Keep broad probe slices around 1 to 2 MB compressed per subset unless there is
  a specific reason to go larger.
- The existing dashboard runs usually cap scoring at 256 documents per dataset.
- The existing dashboard runs usually cap each scored document at 32 KiB.
- Put raw artifacts in the same region as the scoring run when possible. Avoid
  cross-region GCS reads for large sources.

For ETL outputs, also write a small metadata file when practical. Include source
dataset id, source revision, split or row range, record count, byte count, and
any filtering limits. The Stack v2 v0 cut uses this pattern with
`heldout.jsonl.gz` plus `heldout_metadata.json`.

Add focused tests for the provider when it has non-trivial logic. Prefer tests
under `tests/evals/` that validate dataset keys, tags, source wiring, row
schema, and target-only rendering behavior when applicable. For an ETL step,
include at least a small local fixture test for filtering and output schema.

## 3. Add the provider to a gap experiment

Use one of two patterns.

For the dashboard suite, wire committed providers into
`experiments/evals/model_perplexity_gap_suite.py`. Import the provider, add it
to `suite_raw_validation_sets()`, and keep any temporary broken slices in
`SKIPPED_DATASETS_FOR_THIS_RUN` with a specific comment. The suite defines the
run key, dataset composition, Marin/Qwen score steps, report resource config,
and final pairwise gap step.

For reusable model and bundle coverage, register a bundle in
`experiments/evals/perplexity_gap_registry.py` and include it from
`registered_perplexity_gap_bundles()`. The coverage matrix entrypoint
`experiments/exp_model_perplexity_gap_coverage_matrix.py` expands the registered
bundles and models into score and pairwise gap steps.

For a one-off exploratory run that should not enter the dashboard suite, define
`DATASETS`, model configs, score steps, gap steps, and an `executor_main(...)`
block in a dedicated experiment file. Keep `MAX_DOCS_PER_DATASET` and
`MAX_DOC_BYTES` explicit. Pick descriptive step names so output paths can be
traced back to the run.

The one-off pattern builds two score steps and one report step:

- `model_perplexity_scores(...)` for Marin
- `model_perplexity_scores(...)` for the comparison model
- `model_perplexity_gap_from_scores(...)` for the pairwise report

The committed scorer writes score artifacts under:

```text
gs://marin-us-central1/analysis/model_perplexity_scores/
```

If you add per-dataset score caching in a follow-up branch, include source paths,
dataset keys, scoring limits, and model/tokenizer identifiers in the cache key.
If a dataset key is reused with changed content or changed format, make sure the
cache key changes.

## 4. Run a construction check

Before submitting to Iris, import the experiment and inspect the dataset count.
This catches duplicate keys, missing imports, and obvious source construction
errors without launching TPU scoring.

```bash
cd "$MARIN_REPO"
uv run python - <<'PY'
import experiments.evals.model_perplexity_gap_suite as exp

print("run key:", exp.RUN_KEY)
print("datasets:", len(exp.DATASETS))
print("marin step:", exp.MARIN_SCORES.name)
print("qwen step:", exp.QWEN3_SCORES.name)
print("gap step:", exp.GAP.name)
print("skipped:", sorted(exp.SKIPPED_DATASETS_FOR_THIS_RUN))
PY
```

For the registry-backed coverage matrix, inspect the expanded plan separately:

```bash
cd "$MARIN_REPO"
uv run python - <<'PY'
import experiments.exp_model_perplexity_gap_coverage_matrix as exp

print("score steps:", len(exp.PLAN.score_steps))
print("gap steps:", len(exp.PLAN.pairwise_gap_steps))
for key, step in sorted(exp.PLAN.score_steps.items()):
    print(key, step.name)
PY
```

Confirm that dataset count, model count, and step names match what you expect.

## 5. Submit the gap run

Submit the experiment as a CPU parent job. The parent uses `executor_main`; the
model score children request TPU resources through `RESOURCE_CONFIG`. The
committed gap step currently runs inside the dependency chain after the score
artifacts are available.

```bash
cd "$MARIN_REPO"
uv run iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --job-name model-perplexity-gap-suite \
  --priority production \
  --region us-central1 \
  --cpu 1 \
  --memory 2GB \
  --disk 20GB \
  --enable-extra-resources \
  --extra marin:tpu \
  --no-preemptible \
  -- python experiments/evals/model_perplexity_gap_suite.py
```

Use production priority only for runs that are important enough to preempt
lower-priority work. The parent should stay CPU-only; do not request a TPU on
the parent unless the parent itself needs one. Score steps launch TPU child jobs
from their `RESOURCE_CONFIG`; the report step should use CPU resources through
`model_perplexity_gap_from_scores(..., resource_config=...)`.

## 6. Monitor and recover

Use the job prefix and the run key to monitor the parent and child jobs.

```bash
uv run iris --config lib/iris/examples/marin.yaml job list \
  --json \
  --prefix /<user>/model-perplexity-gap-<short-run-name>

uv run iris --config lib/iris/examples/marin.yaml job summary /<user>/<job-name>
uv run iris --config lib/iris/examples/marin.yaml job logs /<user>/<job-name> -f
```

Score outputs appear under paths derived from each score step's name:

```text
gs://marin-us-central1/analysis/model_perplexity_scores/<score-step-name>-*/
```

Gap outputs appear under paths derived from the gap step's name:

```text
gs://marin-us-central1/analysis/perplexity_gap/<gap-step-name>-*/
```

Each completed model score directory should contain at least:

- `.artifact`
- `summary.json`
- `scored_documents.parquet`
- `token_counts.parquet`
- `token_counts_summary.json`

Each completed gap directory should contain at least:

- `summary.json`
- `report.md`
- `worst_documents.jsonl`

If the report step OOMs after both model score children have succeeded, do not
rerun scoring. Create `experiments/exp_model_perplexity_gap_<run>_report_retry.py`
that calls `model_perplexity_gap_from_scores` directly with the completed score
artifact paths, a larger CPU `resource_config`, and a fresh `retry_key`. If
memory still fails, improve the report comparison path to stream more
aggressively before raising memory again.

If a data source fails, fix or skip only the broken component. Relaunch with a
fresh run key and document the skipped slice in the experiment.

## 7. Copy the report artifact into the website repo

Clone `marin-community.github.io` separately; the dashboard lives under
`analysis/perplexity-gap/` in that repo. Only copy the report files needed by
the static dashboard. Do not copy score parquet files or other large
intermediates into the website repo.

```bash
cd "$WEBSITE_REPO"
GAP_STEP_NAME=model_perplexity_gap_suite_v1/marin_32b-vs-qwen3_32b
GAP_ID=${GAP_STEP_NAME}-...
REMOTE=gs://marin-us-central1/analysis/perplexity_gap/${GAP_ID}
LOCAL=analysis/perplexity-gap/artifacts/${GAP_ID}

mkdir -p "${LOCAL}"
gcloud storage cp "${REMOTE}/summary.json" "${LOCAL}/summary.json"
gcloud storage cp "${REMOTE}/report.md" "${LOCAL}/report.md"
gcloud storage cp "${REMOTE}/worst_documents.jsonl" "${LOCAL}/worst_documents.jsonl"
```

Then update `analysis/perplexity-gap/data.json`:

- point `summary_path`, `report_path`, and `worst_documents_path` at the new GCS
  paths
- point `local_artifact_path` at the copied local artifact directory
- update the run subtitle
- update headline groups, spotlight datasets, and any dashboard notes that are
  specific to the new run

Keep old artifact directories only when the dashboard still references them.
Remove stale local snapshots before committing if they are no longer linked.

## 8. Generate heatmaps

Heatmaps are checked in as static JSON payloads. They are generated from the
gap summary and the two model `scored_documents.parquet` files.

Edit `analysis/perplexity-gap/scripts/generate_span_heatmaps.py` in the website
repo:

- set `SUMMARY_PATH` to the copied local `summary.json`
- set `RUN_KEY` to the new run key
- set `MODEL_A_SCORE_PATH` and `MODEL_B_SCORE_PATH` to the completed score
  parquet paths
- add or update entries in `SETS` for the datasets that need heatmaps

Run the generator with the Marin virtualenv so GCS, fsspec, and pyarrow are
available:

```bash
cd "$WEBSITE_REPO"
"$MARIN_REPO/.venv/bin/python" analysis/perplexity-gap/scripts/generate_span_heatmaps.py
```

The generator rewrites `analysis/perplexity-gap/span-heatmaps/manifest.json` and
the selected set directories. Each set manifest points to a small number of
example payloads, with one or more high positive-gap and negative-gap rows per
dataset.

Validate the generated JSON:

```bash
python3 - <<'PY'
import json
from pathlib import Path

for path in Path("analysis/perplexity-gap/span-heatmaps").rglob("*.json"):
    json.loads(path.read_text())
print("all heatmap json valid")
PY
```

## 9. Serve and inspect locally

Serve the website from the repo root:

```bash
cd "$WEBSITE_REPO"
python3 -m http.server 8765 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8765/analysis/perplexity-gap/
```

Validate the dashboard data and heatmap manifests through the local server:

```bash
python3 - <<'PY'
import json
import urllib.request

base = "http://127.0.0.1:8765/analysis/perplexity-gap"
for path in ["data.json", "span-heatmaps/manifest.json"]:
    with urllib.request.urlopen(f"{base}/{path}", timeout=5) as response:
        json.loads(response.read())
        print(response.status, path)
PY
```

Inspect the page in the browser:

- dataset rows appear individually, not only as an aggregate
- the sign convention is correct
- scored byte spans are visually distinguishable when the run includes partial
  scoring
- heatmap links open the intended tab and set through the URL hash
- tiny near-zero gaps render neutral rather than saturated red or blue
- copied URLs restore the same tab and heatmap

Hard-refresh if a local browser tab still shows stale heatmap sets.

## 10. Open the website PR

Create a website branch and commit only the dashboard files needed for the
update:

```bash
cd "$WEBSITE_REPO"
git checkout -b codex/perplexity-gap-dashboard-<run>
git add analysis/perplexity-gap/data.json
git add analysis/perplexity-gap/artifacts/<GAP_ID>
git add analysis/perplexity-gap/span-heatmaps
git add analysis/perplexity-gap/scripts/generate_span_heatmaps.py
git diff --cached --check
git commit -m "Refresh perplexity gap dashboard with <run>"
git push origin codex/perplexity-gap-dashboard-<run>
```

Do not stage unrelated old artifact directories. Check `git status --short`
before committing.

Open the PR with a short plain-text body. If the run came from a Marin issue,
reference it with `Part of marin-community/marin#NNNN`. Add the
`agent-generated` label for agent-authored PRs.

```bash
gh pr create \
  --title "Refresh perplexity gap dashboard with <run>" \
  --body "$(cat <<'EOF'
Refresh the 32B perplexity-gap dashboard with <run>. Adds the copied report artifact and focused heatmaps for the new slices.

Part of marin-community/marin#NNNN
EOF
)" \
  --label agent-generated
```

After pushing, check the PR and CI:

```bash
gh pr view --json url,statusCheckRollup
```

If the local Ruby/Bundler stack cannot build the Jekyll site, say that in the
PR comment and include the static JSON validation you did run.

## 11. Keep the Marin registration PR separate when possible

The Marin repo change and the website dashboard refresh usually belong in
separate PRs:

- Marin PR: provider code, ETL, tests, experiment branch, and issue comments.
- Website PR: copied report artifacts, `data.json`, dashboard code changes, and
  heatmaps.

This keeps dataset generation and result publication reviewable independently.
For synthetic or archival datasets that should not enter the regular eval
suite, keep their generation code on a long-lived research branch and link it
from the relevant issue. Only add them to the model-perplexity gap suite
dashboard when they are useful for diagnosis.

## Common failure modes

- **Mutable HF sources:** The committed `HfDatasetSpec` does not carry a
  revision. If the exact snapshot matters, materialize to a versioned GCS path
  or add revision support before using the source in a long-lived comparison.
- **Target leakage in target-only rows:** The final prompt should end
  immediately before the target. The target bytes should live in the `target`
  field, not be duplicated in `input`.
- **Chat-framed rows:** Base models can be highly sensitive to `User:` /
  `Assistant:` style framing. Prefer neutral base-model formats such as newline,
  `=>`, or `=` unless the task is explicitly chat-formatted.
- **Oversized local artifacts:** Copy only `summary.json`, `report.md`, and
  `worst_documents.jsonl` into the website. Keep score parquet files in GCS.
- **Report OOM:** Reuse completed score artifacts and retry only the report
  step. Do not spend TPU time rescoring unless the score artifacts are wrong.
- **Contamination:** Code and web slices can contain examples that a model saw
  during training. Treat row-level surprises as hypotheses, then inspect
  provenance before using them as clean capability evidence.
