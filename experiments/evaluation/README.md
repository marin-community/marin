# Evaluation launcher

A one-command path from "model + eval suite" to recorded results. Pick a model and an eval selection
from the registries here; the launcher sizes a serving slice and submits one CPU orchestrator job for
the whole launch. The orchestrator serves the model once, runs every selected eval against that
endpoint in order, and writes one durable `record.json` per eval as it finishes -- so a suite fills
in progressively, each eval independently inspectable (own record, own eval-child job and logs, own
parquet), all sharing a `group_id`. Evaldash scans those records into its Postgres query index.

`marin.evaluation.runner` opens one `remote_inference` session and passes its Iris endpoint URL to
each executor. An evaluation failure is recorded and later evaluations continue. If inference fails,
the current and remaining evaluations are recorded as infrastructure failures. This directory holds
the model and suite catalogs, Marin fleet policy, and CLI choices.

The user-facing [evaluation guide](../../docs/tutorials/run-lm-evals.md) contains model-specific
commands, suite constraints, launch controls, and result locations.

## Commands

Run through the module:

```bash
uv run python -m experiments.evaluation.cli launch --model qwen3-8b --evals smoke
```

`launch` submits one run per resolved eval key; a suite expands to its member evals. Unless
`--no-wait`, it waits for each object-store record and prints its metrics:

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
winogrande, truthfulqa, boolq, piqa, openbookqa at OpenLLM-v1 shot counts, plus humaneval and
math500): one model boot, eleven evals against the shared endpoint, eleven records — the dashboard
shows the full model x task grid of runs.

`backfill-samples` rewrites every run's per-sample parquets from its kept `samples_*.jsonl` sources --
useful after a change to the contract in `marin.evaluation.samples` (the parquet files are
regenerated in place; the source jsonl is untouched):

```bash
uv run python -m experiments.evaluation.cli backfill-samples --prefix gs://marin-eval-metadata/runs
```

## Records and the dashboard index

Every eval writes `{records_prefix}/{run_id}/record.json` (`marin.evaluation.records`). That record
is the source of truth: model, hardware, status (`succeeded` / `failed` / `infra_failed`), the
per-task metrics, provenance, the `group_id` shared by every eval from the same serve, and the iris
job paths of every job behind the run (`jobs`: orchestrator, the shared inference child, this eval's
child). The orchestrator writes it on success and on failure, so a failed run is still accounted
for -- and a failure carries the failed child's last 100 log lines (`log_tails`), so most failures
are diagnosable straight from the record (or the dashboard) without cluster access.

Alongside the results tree, each task's individually-scored questions are exported as parquet:
lm-eval runs with `--log_samples`, and the orchestrator converts every `samples_*.jsonl` into a
parquet sibling (`marin.evaluation.samples`, the per-sample contract -- `EvalSample`, normalized from
lm-eval's native row shape, with the parquet schema *being* the Pydantic model) -- load them with
pandas/duckdb, or read them back with `EvalSample.model_validate`, to zoom into any run.

Evaldash treats these records as the source of truth. Its background ingestor scans every configured
object-store prefix and upserts the `eval_runs` and `eval_metrics` tables implemented in
`infra/evaldash/src/results_db.py`. Evaluation launchers do not read DB config or connect to Postgres.

## Evals in pipelines

`pipeline.py` exposes the same run as an `ArtifactStep`: `eval_step("qwen3-1.7b", "smoke",
version="2026.07.19")` is a lazy, versioned handle whose records land at the step's artifact path.
The step submits the same CPU orchestrator used by the CLI and waits for it. The slice override is a
runtime arg, so changing it does not change the artifact identity.

## Agentic benchmarks (Harbor)

The `agentic` suite (`tb2`, `swebench`, `gaia`, `bfcl`, `aider`, `medagentbench`, `financeagent`) runs
in-sandbox agentic benchmarks through the same launcher. Each preset names an `hf://` repository whose
root contains Harbor task directories. The runner materializes that repository at its configured
revision, the launcher serves the model once and mints a capability URL for the served endpoint, and
an in-sandbox terminal agent (Daytona) reaches the model through that URL. Harbor's verifier scores
each trial, which normalizes into one agentic `EvalSample` (reward ->
`Grading(method="harbor:verifier")`, trajectory -> `trajectory_uri`) plus a record, so agentic runs
land in evaldash like every other eval.

```bash
# A capped agentic validation run (2 tasks).
uv run python -m experiments.evaluation.cli launch --model qwen3-8b --evals tb2-lite
```

Use `--harbor-config` to launch a Harbor `JobConfig` without adding it to `EVALS`:

```bash
# Serve Qwen3-8B and run the checked-in two-task AIME Harbor policy.
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-8b \
  --evals gsm8k-smoke \
  --harbor-config experiments/evaluation/configs/aime-smoke.yaml
```

`--harbor-config` is repeatable and additive with `--evals`, so one served model can run registry
entries and file-backed Harbor policies in the same launch. When neither option is supplied, the
launcher uses the `smoke` suite; a file-only launch does not add that default. The launcher validates
YAML and JSON files against Marin's pinned Harbor `JobConfig` before opening an Iris client.
File-backed launches support one agent and one dataset; multiple agents, multiple datasets, and
explicit `tasks` are rejected.

Registry and file-backed Harbor policies use the same normalized config and executor path. Marin
replaces each config's `job_name`, `jobs_dir`, agent `model_name`, `api_base`, and OpenCode provider
URL with values for the served endpoint. Model-catalog agent kwargs are merged underneath the
config's agent kwargs. `--limit` overrides the dataset's `n_tasks`; other normalized Harbor fields
remain unchanged. Every Harbor evaluation record stores the policy's SHA-256 digest in
`eval.harbor.config_digest`.

Daytona-backed definitions declare one experiment-owned credential specification. A launch first
uses `DAYTONA_API_KEY` from its environment, then falls back to the `DAYTONA_EVAL_API_KEY` secret in
the `hai-gcp-models` Google Secret Manager project. `DAYTONA_API_KEY` is the only supported
environment override; the old `DAYTONA_EVAL_API_KEY` environment alias is not read. The generic
launcher resolves the declaration immediately before Iris submission, and the isolated Harbor
subprocess receives that key without inheriting the orchestrator's other credentials.

The Grug OpenCode profile keeps its model and Harbor policy on the unified path:

```bash
# One OpenCode trial with the step-1903 Grug SFT on H100x8.
uv run python -m experiments.evaluation.cli launch \
  --model grug-agentic-s3-step1903 --evals grug-opencode-id --limit 1
```

The profile materializes `DCAgent/dev_set_v2` from a pinned Hugging Face commit before passing its
task directories to Harbor.

Mechanism code lives under `marin.evaluation.evalchemy` and `marin.evaluation.harbor`; the common
runner depends only on the callable executor protocol and the shared record types.

## Adding a model or eval

A model is a `ModelConfig` (`marin.evaluation.model_config`): its `location` (HF id or `gs://`/`s3://`
export), a `resource_hint: ResourceHint` (placement compatibility), a `serve: ServeConfig` (server
behavior), a `generation: GenerationConfig`
(`--gen_kwargs`), and an `agent: AgentConfig` (Harbor agent kwargs). Two population paths feed the one
cached `models()` registry in `models.py`:

- **YAML catalog** under `serve/models/<org>/<model>.yaml` -- one file per model, decoded by draccus
  against `ModelConfig` (an unknown or mistyped field fails at load). This is the bulk catalog; see
  `serve/models/README.md` for the schema. Just add a file.
- **Python factory** in `models.py` for the parametric entries whose serve options are computed
  (`_snowball`, `_base_hf`) or the curated hand-tuned ones.

Set `resource_hint.hbm_gb` to a portable serving footprint, or set
`resource_hint.gpu` to an accepted exact GPU shape such as `{"H100": 8}`. The experiment fleet maps
that requirement to a cluster. Set `resource_hint.memory` when serving needs more than the default
host memory. Set `tokenizer` when `location` is an object-store export because the eval client loads
its tokenizer through Hugging Face. vLLM streams object-store weights through the RunAI loader.
Every explicit `serve` value wins over what `auto_serve_overrides` derives from the model's
`config.json`; `generation.extra_gen_kwargs` (e.g. `skip_special_tokens=false` for a thinking model)
rides on `--gen_kwargs`.

Add an `EvalchemyDefinition` or `HarborDefinition` to `EVALS` in `evals.py`, then add its key to
`SUITES` when it belongs in a named group. Task flags that matter for served evals:
`generation` routes the task through the chat API for chat-template models (MCQ tasks always use
completions, which alone can echo prompt logprobs); `unsafe_code` passes lm-eval's
`--confirm_run_unsafe_code`; and `completion_only` pins a generation task to the completions API.

Use `_chat_eval` for a benchmark under Evalchemy's `eval/chat_benchmarks` tree. It normalizes the
task directory into the matching Evalchemy extra, so adding a benchmark installs its endpoint and
grading dependencies without rebuilding an image. The isolated client also installs CPU-only
PyTorch as a compatibility floor; inference remains in the separately served model process.
