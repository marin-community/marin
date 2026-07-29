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
  --dry-run

# Serve the model and run the checked-in two-task policy.
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-8b \
  --harbor-config experiments/evaluation/configs/harbor/aime-smoke.yaml
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

Local paths resolve against the directory containing the policy. Absolute paths and missing local
directories fail before Iris submission. Hugging Face selector syntax is checked before submission;
repository availability is checked when the worker downloads the snapshot.

Checked-in policies live under `experiments/evaluation/configs/harbor/`. Keep suite membership,
model and hardware selection, and secret source declarations in `experiments/evaluation/evals.py`.

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
deterministic source-policy digest and the effective runtime task limit.
