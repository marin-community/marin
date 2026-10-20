# Evaluation launcher

Run Evalchemy and Harbor evaluations on Iris. Evaluations in one launch share a
model server and, when configured, a hosted judge. Each evaluation writes its own
results and `record.json`. Serving workers stop when the launch ends.

Run commands from the repository root. You need Iris access and permission to use
the selected compute and storage. Daytona runs also need a credential; see
[Harbor credentials](../../docs/tutorials/run-lm-evals.md#harbor-credentials).

## Launch

Check the plan before submitting an unfamiliar configuration:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-0.6b --evals smoke --dry-run
```

Remove `--dry-run` to submit. The command waits for the orchestrator and prints
each evaluation's metrics. Add `--no-wait` to return after submission.

Choose exactly one model source:

- `--model <key>`: an entry in [the model catalog](serve/models/README.md).
- `--model-config <path>`: a YAML or JSON file with the same schema.

Pass a suite name or comma-separated evaluation keys to `--evals`. Suites are
`smoke`, `core`, `nlp`, `chat`, `math`, `code`, and `agentic`.
[evals.py](evals.py) lists their members and the available evaluation keys.

To use evaluation files, repeat `--evalchemy-config` or `--harbor-config`.
These add to `--evals`; each file produces one evaluation record. With no
evaluation selection, the launcher uses `smoke`. File-only launches do not add
that default.

For example, run an Evalchemy task and a Harbor benchmark on shared GPU serving:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-32b \
  --evals gsm8k-smoke \
  --harbor-config experiments/evaluation/configs/harbor/aime-smoke.yaml \
  --federated_cluster cw-rno2a --limit 2
```

### Launch options

| Option | Effect |
| --- | --- |
| `--platform tpu\|gpu` | Override the model's platform. |
| `--accelerator H100x8` | Select an exact accelerator shape; TPU shapes such as `v6e-8` also work. |
| `--federated_cluster cw-rno2a` | Select the GPU target cluster. Submission uses the `marin` controller. |
| `--priority interactive` | Set the orchestrator and serving-job priority. |
| `--limit N` | Cap items or tasks per evaluation. |
| `--instances N` | Start N candidate serving instances, each with the selected accelerator shape. Default: 1. |
| `--max-concurrent N` | Run at most N evaluators at once on shared serving. Default: 1. |
| `--seed N` | Override the Evalchemy seed. |
| `--version <label>` | Set the cohort label. Labels do not change evaluator settings. |
| `--records-prefix <URI>` | Choose the root for records and default result paths. |
| `--run-id <id>` | Set a launch ID that you can reuse to retry failed evaluations. |

Use `launch --help` for all options. See the
[evaluation guide](../../docs/tutorials/run-lm-evals.md) for more command examples.

### Judges

For Harbor, use `--judge-model` or `--judge-model-config` to serve a separate
judge. The judge must run in the same cluster or region as the candidate.
`--judge-accelerator` selects its accelerator shape.

```bash
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-32b \
  --judge-model qwen3.5-122b-a10b-fp8 --judge-accelerator H100x8 \
  --harbor-config experiments/evaluation/configs/harbor/simpleqa-hosted-judge.yaml \
  --federated_cluster cw-rno2a --limit 2
```

Evalchemy tasks that need an external judge declare a `judge` block in their
configuration. It supplies the endpoint, model, and secret reference. Records
omit the secret reference and resolved credential.

## Evaluation files and runtimes

Evalchemy and Harbor run from [vendor/evalchemy](../../vendor/evalchemy/README.md)
and [vendor/harbor](../../vendor/harbor/README.md). Each has a separate environment
and frozen lockfile. Neither is installed in the main workspace environment.

The current verified cohort is `eval-policy-2026-10-20-verified`. Its benchmark
settings and vendored runtime descriptors are frozen in
[`verified.json`](../../lib/marin/src/marin/evaluation/policies/verified.json).
Pass that label with `--version` to check a launch against the profile. Subsets
are allowed. Other `eval-policy-*` labels cannot launch new runs. Historical
records and cohort labels remain readable in EvalDash, without a verified marker.

### Evalchemy

Put portable YAML or JSON files under [configs/evalchemy](configs/evalchemy).
A file selects `tasks`, per-task options, generation settings, and any required
`runtime_extras`. See [the runtime reference](../../vendor/evalchemy/README.md)
for the retained tasks and dependencies.

A Marin dry run checks the file shape and launch plan. It does not check the
installed task catalog. Validate task names separately before submission:

```bash
uv run --project vendor/evalchemy --frozen --no-default-groups evalchemy validate-config \
  experiments/evaluation/configs/evalchemy/gsm8k-smoke.yaml
```

An explicit `--limit` or `--seed` overrides the file value.
`apply_chat_template` uses the model setting unless the file sets it.
GPQA Diamond's seeded requests require GPU vLLM; they do not work with TPU vLLM.

### Harbor

Files under [configs/harbor](configs/harbor) use Harbor's `JobConfig` schema.
Marin validates them with the vendored runtime before Iris submission.
Each file must select one agent and one dataset; explicit `tasks` are rejected.

Use `datasets[].name: hf://org/repository` for Hugging Face sources, with an
optional `ref`. Local `datasets[].path` values must be relative to the config
file and stay inside the checkout. Iris bundles non-ignored checkout files;
for git-ignored task files, add include globs under `[tool.iris.bundle]` in the root
`pyproject.toml`.

Agent arguments in the Harbor file override the model's agent arguments. Marin supplies the
endpoint, served model name, output path, local dataset path, and explicit
`--limit`. Those values cannot be overridden in the Harbor file.

Daytona runs first use `DAYTONA_API_KEY` from the launch environment, then the
`DAYTONA_EVAL_API_KEY` secret in the `hai-gcp-models` GCP project.

## Results

Each evaluation has its own generated `run_id` and writes:

```text
{records_prefix}/{run_id}/record.json
```

The record contains the resolved model and evaluator settings, runtime
provenance, hardware, status, metrics, coverage, result path, and Iris job IDs.
Evaluations in one launch share a `group_id`, which is the launch ID set by
`--run-id` when supplied. Failure records include available child log tails.

The result path contains a FineStore archive:

- `samples`: normalized per-item scores. Repeated answers have distinct trial IDs.
- `evalchemy_results`: Evalchemy aggregate result documents.
- `steps`: Harbor trajectory steps.
- `rollouts_v1`: normalized conversations derived by Marin.

Harbor also keeps its native trial results and trajectories. Evaldash ingests
the records and archives in the background. For sample schemas and storage
details, see the
[evaluation overview](../../docs/explanations/evaluation.md).

Sequential vLLM evaluations record token and throughput measurements.
Concurrent evaluations omit server-wide measurements because the counts cannot
be assigned to one evaluator.

Evalchemy coverage is scored items divided by attempted items. Infrastructure
failures do not count as scored items. A run with infrastructure errors and
coverage below 90% is marked `infra_failed` and is not cached as successful.

## Recovery

Completed evaluations are reused; failed evaluations remain eligible for retry.
If serving fails at startup, every
pending evaluation gets an infrastructure-failure record.

During inference preemption, Marin stops affected evaluators and waits for the
candidate and any hosted judge to become ready. It then restarts the evaluators
at the same result paths. Harbor keeps completed trials. Evalchemy reuses
committed successful responses and retries failed requests. Uncommitted work can
run again. If inference ends permanently, unfinished evaluations fail.

To repeat a CLI launch while retaining completed steps, set `--run-id` on the
first launch and reuse it with the same settings. Without it, each CLI launch
gets new run directories. Changed model, evaluation, judge, or cohort settings
create separate evaluation run IDs.

To recover an existing Harbor archive in a new launch, use
`--resume-results-path <URI>` with exactly one Harbor evaluation. The saved
model, judge, evaluation configuration, runtime, dataset, and task limit must match. Add
`--retry-unscored-harbor-trials` to remove and retry known unscored trials.
Scored agent failures and unknown errors are preserved.

Wait for all earlier drivers using that result path to stop before resuming.
Two live Harbor drivers must not share one results root.

## Pipelines and catalog changes

[pipeline.py](pipeline.py) exposes the launcher as an `ArtifactStep`:

```python
from experiments.evaluation.models import models
from experiments.evaluation.pipeline import eval_step

evaluation = eval_step(models()["qwen3-0.6b"], "smoke", version="2026.10.20")
```

For produced models, pass `deps` and a `resolve_model(ctx)` callback to
[`eval_step`](pipeline.py). The callback resolves producer locations and returns
a `ModelConfig`. The returned `EvaluationResult.results_paths` lists
the FineStore archives. Downstream steps should consume that result instead of
rebuilding paths.

Pipeline retries reuse completed evaluations when their resolved settings match.
Changes to the model, evaluator, judge, or evaluation configuration create new run directories;
mutable `dev` versions always create new runs. Evalchemy caches include evaluator
source, data, configuration, lockfile, and selected extras. Changes to local
storage libraries or Marin adapters are recorded as provenance but do not
invalidate completed work. Bump the evaluation version or `cache_key` when such
a change affects results.

To extend the catalog:

- Add model YAML under `serve/models/<org>/<model>.yaml`; see the
  [model schema](serve/models/README.md).
- Add Evalchemy files under `configs/evalchemy/` and register their names in
  `_STANDARD_EVALCHEMY_EVALS` in [evals.py](evals.py).
- Add Harbor files under `configs/harbor/` and register them with
  `harbor_definition()`. Add keys to `SUITES` when they belong in a suite.

## Archive maintenance

Archive upgrades and sample backfills use a separate administrative CLI:

```bash
uv run python -m experiments.evaluation.migrations.cli --help
```

Before upgrading production archives, run `smoke-upgrade` on the archive's
platform. It checks a temporary copy and leaves the source unchanged.
`smoke-upgrade-fleet` rehearses the upgrade across the fleet. Use
`upgrade-format` only after the smoke checks pass. These commands upgrade the
FineStore format; `backfill-samples` rewrites normalized samples from retained
source files. Read each command's `--help` before writing to production storage.
