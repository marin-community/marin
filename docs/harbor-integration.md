# Harbor evaluation

Marin runs Harbor benchmarks through the shared evaluation launcher. The launcher starts one model
server, gives Harbor its Iris capability URL, normalizes completed trials into v2
`EvalRunRecord`/`EvalSample` artifacts, and tears inference down after the selected evaluations
finish.

Harbor provides containerized agent benchmarks such as Terminal-Bench, SWE-bench Verified, AIME,
GAIA, BFCL, and Aider. Trials can run in Daytona or another Harbor-supported sandbox environment.
See [Running Evaluations with Marin](tutorials/run-lm-evals.md) for the model and suite command
matrix. This page documents Harbor-specific configuration and output behavior.

## Run a benchmark

Use a model and evaluation name from `experiments/evaluation/models.py` and
`experiments/evaluation/evals.py`:

```bash
# Inspect placement, task limit, and record destination.
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-8b \
  --evals tb2-lite \
  --dry-run

# Serve once and run two Terminal-Bench tasks in Daytona.
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-8b \
  --evals tb2-lite
```

The `agentic` suite expands to the standard Harbor presets:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-8b \
  --evals agentic
```

Use `--limit N` to cap the number of trials and `--no-wait` to return after submission.

Launch a Harbor `JobConfig` without adding it to the catalog:

```bash
# Validate policy, model overlay, and placement without opening Iris.
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-8b \
  --harbor-config experiments/evaluation/configs/harbor/aime-smoke.yaml \
  --limit 2 \
  --dry-run

# Serve the model and run the checked-in policy with a two-task cap.
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-8b \
  --harbor-config experiments/evaluation/configs/harbor/aime-smoke.yaml \
  --limit 2
```

`--harbor-config` is repeatable and additive with `--evals`. All selected built-in and supplied
Harbor policies pass pinned-schema and placeholder effective-job validation before the launcher
opens an Iris client.

## Credentials

Daytona-backed definitions in `experiments/evaluation/evals.py` own their credential references.
They use `DAYTONA_API_KEY` from the launch environment when present, then fall back to the approved
Google Secret Manager version. `DAYTONA_API_KEY` is the only supported environment override.

The generic launcher resolves declared references immediately before Iris submission. The isolated
Harbor subprocess receives the Daytona key without inheriting the orchestrator's other credentials.
Do not put resolved credentials in model YAMLs, runner configs, or evaluation artifacts.

## Endpoint lifecycle

Harbor receives a `RunningModel` whose base URL is an Iris link endpoint. The inference runner
chooses the opaque endpoint name, registers either the direct server or broker proxy with Iris, and
mints the capability URL. Daytona never receives a worker address.

Each inference session chooses an opaque endpoint name. Iris endpoint leases remove abandoned
registrations, and a retried task attempt atomically replaces its own same-name registration.

## Policies and datasets

Harbor policy belongs in YAML. This example uses a Hugging Face repository whose root contains
Harbor task directories:

```yaml
n_concurrent_trials: 4
environment:
  type: daytona
agents:
  - name: terminus-2
datasets:
  - name: hf://DCAgent2/terminal_bench_2
    ref: main
```

Use `datasets[].name`, not `datasets[].path`, for `hf://org/repository`. The evaluator downloads the
snapshot on the submitted worker and gives Harbor a local path. A Harbor registry source uses its
native selector, such as `name: aime` plus `version: "1.0"`. A local source uses a relative path:

```yaml
environment:
  type: daytona
agents:
  - name: terminus-2
datasets:
  - path: tasks
```

Local paths resolve against the directory containing the policy. The resulting directory must remain
inside the Marin workspace and must be included in the Iris workspace bundle. The launcher stores its
workspace-relative path so the submitted worker resolves it under the unpacked workspace. Absolute,
outside-workspace, and missing local directories fail before Iris submission. Hugging Face selector
syntax is checked before submission; repository availability is checked when the worker downloads the
snapshot.

Every catalog policy lives under `experiments/evaluation/configs/harbor/` and shares its filename
with its `EVALS` key. Keep suite membership, runtime task caps, model and hardware selection, and
secret source declarations in `experiments/evaluation/evals.py`.

## Ownership boundary

Marin does not install or import Harbor, `harbor_config`, Daytona, or Harbor's path runtime. The
root workspace lock contains none of those packages. `marin.external_dependencies.HARBOR` identifies
the exact Git revision used by two isolated calls:

1. Preflight parses YAML or JSON with Harbor's Pydantic models, rejects unsupported launch shapes,
   validates a placeholder model/endpoint overlay, and emits opaque deterministic policy JSON plus
   Marin-owned metadata.
2. Execution reparses the opaque policy, applies the real endpoint, served model, output directory,
   materialized dataset path, model kwargs, and task limit, then validates the complete typed job
   before calling Harbor.

Runtime values do not change the source-policy digest. Policy kwargs override model-catalog kwargs;
the served endpoint/model, output paths, materialized source, and explicit `--limit` override both.
Temporary policy and overlay files are owner-readable and removed after each isolated call.

## Results

Each Harbor evaluation writes:

- `{records_prefix}/{run_id}/record.json`
- `{records_prefix}/{run_id}/results/samples_harbor.parquet`
- durable Harbor trial directories and trajectory references

Every completed trial becomes an agentic `EvalSample`. The verifier reward is stored as
`Grading(method="harbor:verifier")`, and the trajectory is referenced by `trajectory_uri`. Evaldash
ingests the record and sample parquet in the same way as Evalchemy runs. `record.json` stores the
deterministic source-policy digest and any Marin runtime task cap. A source policy's own `n_tasks`
remains part of the policy digest.

## Synthetic data and SFT export

Evaluation and synthetic-data generation use the same Harbor execution path: model serving, dataset
materialization, sandbox trials, resume, and durable trial upload. They differ after a trial
finishes. Evaluation produces grading records and treats trial errors as an evaluation failure.
Synthetic-data generation applies its own acceptance policy and exports training records. Do not
introduce a separate launcher for the synthetic-data case.

`marin.datakit.download.harbor_sft` converts durable trace datasets into structured SFT parquet.
Use a pinned manifest with the standalone Datakit entrypoint:

```bash
uv run python -m experiments.datakit.harbor_sft \
  --manifest experiments/datakit/manifests/grug_67b_a2b_agentic_sft.json \
  --only exp_rpt_curriculum-hard
```

The converter scans the trace dataset's `agent` column and requires one uniform, recognized
harness. It then selects one of two adapters:

- `terminus-2` preserves the recorded `conversations` as literal SFT ground truth.
- `opencode` reconstructs an installed OpenCode harness interaction from its recorded
  `prompt_token_ids` and `completion_token_ids`, using the exact revision-pinned teacher tokenizer.

The manifest may omit `harness` for automatic detection or set `terminus-2`/`opencode` as an
assertion. Unknown agents, mixed-harness inputs, and manifest/provenance mismatches fail before
conversion.

Installed harnesses run in the sandbox and call the inference endpoint remotely. Harbor therefore
cannot recover the served system prompt, tool schemas, or structured calls from the displayed
`conversations` field. The literal adapter fails closed when those token columns are absent or
misaligned; it never emits the lossy conversation projection as SFT data.
Literal conversion also fails closed unless tokenizer provenance pins both the
served model and its immutable revision (or the manifest supplies both
explicitly).

The output columns are `messages`, `tools`, `task`, `num_turns`, and `num_tool_calls`. Tool calls
remain structured and tool observations use `role: tool`. The trainee's tools-aware chat template
performs final rendering at tokenization time.

The checked-in Grug manifest pins the 29 sources and accepted row counts used by the historical
`grug-67b-a2b-agentic-sft` run. Its compact 22-row source is also a data-integration golden test,
which compares the canonical output hash with the archived training parquet.
