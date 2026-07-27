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

## Datasets and presets

A `HarborExecutor` contains a `HarborRunConfig`:

```python
from experiments.evaluation.evals import HarborDefinition
from experiments.evaluation.harbor_datasets import HuggingFaceHarborDataset
from marin.evaluation.harbor.driver_config import (
    HarborAgentConfig,
    HarborEnvironmentConfig,
    HarborRunConfig,
)

dataset = HuggingFaceHarborDataset(
    repository="DCAgent2/terminal_bench_2",
    commit="693231ec029249e7c91ed2e414bcc9c45d7cd879",
)
definition = HarborDefinition(
    name="tb2-lite",
    config=HarborRunConfig(
        dataset=dataset.repository,
        revision=dataset.commit,
        agent=HarborAgentConfig(name="terminus-2"),
        environment=HarborEnvironmentConfig(environment_type="daytona"),
        n_concurrent=4,
    ),
    dataset_artifact=dataset,
    max_eval_instances=2,
)
```

Every Hugging Face source uses a full commit hash. The evaluator resolves a lazy `download_hf`
artifact under its normal artifact prefix: GCS on GCP or S3 on CoreWeave. A successful artifact is
reused by later model sweeps, so a cache hit does not contact Hugging Face. The evaluation record
stores the repository, commit, and resolved artifact URI.

At the Harbor boundary, Marin lists valid task directories in the mirror and stages only the
selected directories under evaluator-local `/tmp`. `--limit 2`, for example, copies two complete
task trees rather than the full repository. A registry dataset does not need an artifact; it uses
its Harbor name, such as `aime` with version `1.0`.

The commits in `experiments/evaluation/evals.py` come from
`HfApi().dataset_info(repository).sha`. Refresh a dataset only as an intentional benchmark version
change: update its commit and bump `_CATALOG_VERSION` in `harbor_datasets.py`. Normal evaluation
runs never follow the mutable `main` revision.

Add project-specific presets to `experiments/evaluation/evals.py`; keep Harbor execution and result
normalization in `lib/marin/src/marin/evaluation/harbor`.

## Results

Each Harbor evaluation writes:

- `{records_prefix}/{run_id}/record.json`
- `{records_prefix}/{run_id}/results/samples_harbor.parquet`
- durable Harbor trial directories and trajectory references

Every completed trial becomes an agentic `EvalSample`. The verifier reward is stored as
`Grading(method="harbor:verifier")`, and the trajectory is referenced by `trajectory_uri`. Evaldash
ingests the record and sample parquet in the same way as Evalchemy runs.
