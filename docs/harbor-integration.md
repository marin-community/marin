# Harbor evaluation

Marin runs Harbor benchmarks through the shared evaluation launcher. The launcher starts one model
server, gives Harbor its Iris capability URL, normalizes completed trials into v2
`EvalRunRecord`/`EvalSample` artifacts, and tears inference down after the selected evaluations
finish.

Harbor provides containerized agent benchmarks such as Terminal-Bench, SWE-bench Verified, AIME,
GAIA, BFCL, and Aider. Trials can run in Daytona or another Harbor-supported sandbox environment.

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

## Credentials

Daytona runs require `DAYTONA_API_KEY` in the launch environment.

The sandbox receives only the credentials needed by the selected agent. Depending on the benchmark
and agent, this can include:

- `HF_TOKEN`
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `E2B_API_KEY`
- `MODAL_API_KEY`

Do not put credentials in model YAMLs or runner definitions.

## Endpoint lifecycle

Harbor receives a `RunningModel` whose base URL is an Iris link endpoint. The inference runner
chooses the opaque endpoint name, registers either the direct server or broker proxy with Iris, and
mints the capability URL. Daytona never receives a worker address.

Each inference session chooses an opaque endpoint name. Iris endpoint leases remove abandoned
registrations, and a retried task attempt atomically replaces its own same-name registration.

## Datasets and presets

A `HarborExecutor` contains a `HarborRunConfig`:

```python
from marin.evaluation.harbor_runner import HarborRunConfig
from marin.evaluation.runner import HarborExecutor

executor = HarborExecutor(
    config=HarborRunConfig(
        dataset="hf://DCAgent2/terminal_bench_2",
        version="main",
        agent="terminus-2",
        env="daytona",
        n_concurrent=4,
        task_limit=2,
    ),
)
```

`hf://org/repository` identifies a Hugging Face dataset repository whose root contains Harbor task
directories. A registry dataset uses its Harbor name, such as `aime` with version `1.0`.

Add project-specific presets to `experiments/evaluation/evals.py`; keep Harbor execution and result
normalization in `lib/marin/src/marin/evaluation`.

## Results

Each Harbor evaluation writes:

- `{records_prefix}/{run_id}/record.json`
- `{records_prefix}/{run_id}/results/samples_harbor.parquet`
- durable Harbor trial directories and trajectory references

Every completed trial becomes an agentic `EvalSample`. The verifier reward is stored as
`Grading(method="harbor:verifier")`, and the trajectory is referenced by `trajectory_uri`. Evaldash
ingests the record and sample parquet in the same way as Evalchemy runs.
