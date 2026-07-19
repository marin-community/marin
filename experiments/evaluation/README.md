# Evaluation launcher

A one-command path from "model + eval suite" to recorded results. Pick a model and an eval selection
from the registries here; the launcher sizes a serving slice and submits one CPU orchestrator job for
the whole launch. The orchestrator serves the model once, runs every selected eval against that
endpoint in order, and writes one durable `record.json` per eval as it finishes -- so a suite fills
in progressively, each eval independently inspectable (own record, own eval-child job and logs, own
parquet), all sharing a `group_id`. A Postgres index mirrors the records for querying.

The engine is `experiments/evals/evalchemy/serve_and_eval.py` (`run_eval_units`): one marin-serve
child exposes an OpenAI-compatible endpoint (vLLM on TPU or GPU), and one evalchemy child per eval
hits it. One eval failing doesn't stop the rest; if the served endpoint itself dies, the remaining
evals are recorded as serve failures without running.

## Commands

Run through the module:

```bash
uv run python -m experiments.evaluation.cli launch --model qwen3-8b --evals smoke
```

`launch` submits one run per suite and, unless `--no-wait`, waits for them and mirrors results to
Postgres:

```bash
# See the resolved plan without submitting anything.
uv run python -m experiments.evaluation.cli launch --model qwen3-8b --evals smoke --dry-run

# One suite, a specific slice, capped instances, no waiting.
uv run python -m experiments.evaluation.cli launch --model llama3.1-8b-instruct \
  --evals gsm8k --accelerator v6e-8 --limit 128 --no-wait

# GPU-only model routes to its CoreWeave peer automatically.
uv run python -m experiments.evaluation.cli launch --model snowball --evals gsm8k-smoke
```

Key options: `--evals` takes a suite name (`smoke`, `core`) or comma-separated eval keys
(`gsm8k,mmlu-smoke`); `--platform tpu|gpu` overrides the model's default; `--accelerator` overrides the
sizing heuristic with an exact slice (`v6e-8` or `H100x8`); `--limit` caps eval instances;
`--records-prefix` and `--cluster` override where records land and which iris cluster to submit to.

Suites: `smoke` is a fast cluster check (capped mmlu cut + capped gsm8k). `core` is the comprehensive
per-model benchmark set (`CORE_EVALS` in `evals.py`: mmlu, gsm8k, arc-challenge, hellaswag,
winogrande, truthfulqa, boolq, piqa, openbookqa at OpenLLM-v1 shot counts, plus humaneval): one
model boot, ten evals against the shared endpoint, ten records — the dashboard shows the full
model x task grid of runs.

`ingest` rebuilds the Postgres index from object storage (useful after runs written while the DB was
unavailable):

```bash
uv run python -m experiments.evaluation.cli ingest --prefix gs://marin-eval-metadata/runs
```

`runs` queries the index:

```bash
uv run python -m experiments.evaluation.cli runs --model qwen3-8b --status succeeded --limit 20
```

## Records and the database

Every eval writes `{records_prefix}/{run_id}/record.json` (`marin.evaluation.records`). That record
is the source of truth: model, hardware, status (`succeeded` / `failed` / `infra_failed`), the
per-task metrics, provenance, the `group_id` shared by every eval from the same serve, and the iris
job paths of every job behind the run (`jobs`: orchestrator, the shared serve child, this eval's
child). The orchestrator writes it on success and on failure, so a failed run is still accounted
for -- and a failure carries the failed child's last 100 log lines (`log_tails`), so most failures
are diagnosable straight from the record (or the dashboard) without cluster access.

Alongside the results tree, each task's individually-scored questions are exported as parquet:
lm-eval runs with `--log_samples`, and the orchestrator converts every `samples_*.jsonl` into a
parquet sibling (`marin.evaluation.sample_export`) with the document, prompt arguments, model
responses, and per-sample metric columns -- load them with pandas/duckdb to zoom into any run.

`marin.evaluation.results_db` is a rebuildable Postgres mirror (`eval_runs` + `eval_metrics`) keyed by
`run_id`, with the full record kept in a `jsonb` column. `launch` upserts as runs finish; `ingest`
re-derives the whole index from `list_records`. The connection is resolved from `EVAL_DB_*` environment
variables, with the password read from Secret Manager when `EVAL_DB_PASSWORD` is unset; when no password
is available the launcher degrades to writing records only.

## Evals in pipelines

`pipeline.py` exposes the same run as an `ArtifactStep`: `eval_step("qwen3-1.7b", "smoke",
version="2026.07.19")` is a lazy, versioned handle whose records land at the step's artifact path —
compose it into any `StepRunner` pipeline (e.g. right after a checkpoint-export step, or fanned out
over a model sweep) and an identical config re-run is a cache hit. The step process acts as the
orchestrator, so the pipeline must itself run as an Iris job. The slice override is a runtime arg:
changing it never forks the artifact's identity.

## Adding a model or eval

Add a model by adding an `EvalModelConfig` to `MODELS` in `models.py`. Set `hbm_gb` honestly (bf16
weights are `params_billions * 2 GB`, times roughly 1.3 for runtime overhead); the sizing heuristic
picks the smallest slice that fits. Set `tokenizer` when `location` is an object-store export (the eval
client loads its tokenizer through HF and cannot read a `gs://`/`s3://` path). Use `fixed_gpu` and
`target_cluster` to pin an exact GPU shape and CoreWeave peer. Set `serve_memory` for large
object-store exports: weight streaming stages shards through host buffers, so the serve pod's memory
limit must cover the full weight volume or the kernel OOM-kills the server mid-load.

Add an eval by adding an `EvalSuiteConfig` to `EVALS` in `evals.py` (its `tasks` are `EvalTaskConfig`
entries, the same task menu the in-loop suites use). Add it to a group in `SUITES` to make it selectable
by name. Task flags that matter for served evals: `generation` routes the task through the chat API for
chat-template models (MCQ tasks always use completions, which alone can echo prompt logprobs);
`unsafe_code` passes lm-eval's `--confirm_run_unsafe_code` for code-execution scoring; and
`completion_only` pins a generation task to the completions API for every model (humaneval's infill
prompt breaks under chat formatting -- chat models reply with prose and markdown fences).
