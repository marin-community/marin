# Run evaluations with Marin

The shared launcher evaluates served models against curated TaskSpec corpora and
Harbor datasets. It prepares complete evaluation corpora before starting inference,
shares one candidate server across the selected evals, and records each eval separately.
Evaldash computes scores from the raw results.

## Prerequisites

Use the repository environment and configure Iris access for the selected cluster.
Hugging Face credentials must permit any gated source datasets. Executable graders
run inside a privileged Iris CPU worker, isolated with Shellbox bubblewrap.

Keep source corpora, grader environments and result storage near the evaluation
cluster. The launcher uses that cluster's artifact prefix. See
[task curation](../references/task-curation.md) for source publication and
[Harbor integration](../harbor-integration.md) for agentic benchmarks.

## Launch registered models and evaluations

Inspect a bounded launch without submitting jobs:

```bash
uv run --package marin-core --group test python -m experiments.evaluation.cli launch \
  --model qwen3-8b --evals smoke --platform gpu --accelerator H100x1 \
  --federated_cluster cw-us-east-02a --limit 2 --dry-run
```

Remove `--dry-run` to submit. The launcher resolves model and tokenizer revisions,
selects the named corpus variants, builds missing curation dependencies, starts
inference, and submits evaluation workers. It cancels an evaluator before resuming
its archive when inference is interrupted.

The catalog fixes prompts, graders, shots, sampling, trials and context policies.
Evalchemy protocols inherit the served model's temperature and seed unless the
run explicitly overrides them, preserving the legacy launcher's request behavior.
Native likelihood tasks use completion logprobs. Generation protocols specify chat
or raw completion transport. Instruct-model menus choose separately published chat
variants where the original entrypoint used a chat template. HumanEval's standard
completion protocol stays raw. Checkpoint menus retain their explicit shot variants.

## Named suites

| Suite | Selection |
| --- | --- |
| `smoke` | Bounded MMLU and GSM8K |
| `standard` | Standard curated evaluations, including NUPA |
| `standard-local` | Standard evaluations whose protocol requires no external judge |
| `core`, `nlp`, `chat`, `math`, `code` | The corresponding task menus |
| `agentic` | Harbor benchmarks |

Use comma-separated keys for a custom selection, such as
`--evals math500,piqa,humaneval`. A source limit applies to source rows before trial
expansion. For MMLU it applies separately to each subject corpus. AIME24 retains its
ten trials per selected row. Context policies can exclude rows; a corpus with no
eligible rows fails rather than producing an empty score. GraphWalks requires a
reference-dependent output allowance of at least 8,192 tokens, so a blanket
512-token smoke budget cannot exercise it.

## Launch file-backed evaluations

A selection document names benchmark protocols and experiment overrides:

```yaml
evaluations:
  bounded-math:
    benchmark: math500
    run:
      limit: 2
      max_tokens: 512
      seed: 0
  gsm8k-zero:
    benchmark: gsm8k
    num_fewshot: 0
```

Submit both with `--evaluation-config evaluations.yaml`. Add `--evals bounded-math`
to select one entry from that document. Repeat `--harbor-config` to add Harbor jobs.
Ordinary benchmark defaults are available through `--evals math500,gsm8k` without
a selection file. The repository's curated suite choices and run overrides live in
`experiments/evaluation/configs/evalchemy.yaml`.

Exactly one of `--model <registry-key>` or `--model-config <path>` is required.
The latter accepts a model catalog YAML for a new export without adding it to the
registry. `record.json` stores the normalized model schema under `model.config`
for both forms, so the run record is self-contained. A dry run validates the
selection and model without submitting an Iris job:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model-config /path/to/fresh-rl-checkpoint.yaml \
  --evals smoke \
  --dry-run
```

Shot count configures the benchmark's protocol builder. Curation compiles the
selected demonstrations before inference. `prompt_format: chat` or `completion`
selects a supported request format; otherwise planning resolves it for the model
and records the explicit choice. `protocol` selects a named benchmark-owned preset,
such as GSM8K's `cot` protocol. Unsupported combinations fail before launch.
MMLU owns its subject partitions; `partitions: [abstract_algebra]` selects one
without naming a separate smoke dataset. Likelihood evaluations score supplied
continuations and have no generation output budget.

Judge deployments belong under `run`: `judge` records a `JudgeIdentity`,
`judge_endpoint` is a credential-free endpoint, and `judge_api_key` contains ordered
secret references. The standard
FinanceBench and OlympiadBench definitions use Together and record its model alias
as provider-managed. A provider alias does not establish the exact weights served.
Supply explicit judge deployment identities for reproducible private deployments.
The [September 25 publication campaign](https://github.com/marin-community/marin/tree/main/experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled)
supplies its separately pinned hosted judge through `--policy-finalization-config`.

Harbor files retain Harbor's `JobConfig` schema and `--harbor-config` option.
Harbor execution is independent of curated TaskSpec evaluation.

## Launch controls

- `--model-config`: a model catalog YAML instead of a registry model.
- `--limit`: override the source-row cap for each selected corpus.
- `--seed`: override the protocol's trial seed base.
- `--max-concurrent`: simultaneous evaluations sharing the candidate server.
- `--instances`: candidate serving instances behind the inference broker.
- `--run-id`: reuse an explicitly named launch when its resolved identities match.
- `--records-prefix`: override the run-record destination.
- `--no-wait`: return after submitting the orchestrator.

Model `generation.max_gen_toks` and each selection's `run.max_tokens` cap protocol
budgets. HumanEvalPlus, MBPPPlus, and IFBench have no fixed protocol output cap;
without an override, their allowance comes from the served context minus the
prompt. GSM8K retains both extraction metrics and uses `flexible-extract` for its
headline accuracy. Protocol context policies still determine eligibility. Model chat-template
arguments provide defaults; benchmark arguments take precedence. A benchmark that
turns thinking off can use the model's `thinking_off_template_kwargs` translation.

Historical verified Evalchemy cohort labels require their recorded historical
checkout. Curated runs need a new cohort label; they do not attest equivalence to a
historical evaluator.

## Harbor credentials

Configure Daytona and verifier credentials through the secret references declared
by the selected Harbor policy. `--judge-model` and `--judge-accelerator` configure
Harbor's hosted verifier. See [Harbor integration](../harbor-integration.md).

## Results

Each evaluation gets a `run_id` and writes `{records_prefix}/{run_id}/record.json`.
The selected evaluations share a `group_id`; `--run-id` sets that launch ID.
Curated producer records
contain launch identity, status, jobs and the raw FineStore location. Their metric
maps stay empty. Evaldash imports the raw archive and stores its derived scores in
its own catalog; it does not rewrite the producer's record.

Raw stores contain protocol and manifest snapshots in `evaluation/run.json`, a
selection ledger, append-only start/finish attempts, samples, rollouts and grading
evidence. Evaldash selects the highest finished retry for each dataset, source row
and trial. It retains zero grades, reports infrastructure failures as missing
coverage, and exposes the same logical trials in the sample browser.

Means, instruction weights, classification rates, perplexity and grouped scores come
from protocol metadata. Dataset snapshot identities and the summary revision keep
different protocols distinct in dashboard comparisons. See
[evaluation tasks](../references/evaluation-tasks.md) for the raw contract.

## Use the shared launcher in a pipeline

```python
from marin.evaluation.records import EvaluationSelection
from experiments.evaluation.models import models
from experiments.evaluation.pipeline import eval_step

step = eval_step(models()["qwen3-8b"], (EvaluationSelection(benchmark="gsm8k", num_fewshot=0),),
                 version="dev", limit=2,
                 accelerator="H100x1", federated_cluster="cw-us-east-02a")
```

Run this graph where it can reach Iris. Its artifact contains run IDs and raw result
paths; the canonical run records remain under the evaluation records prefix.

## Compose evaluation steps from Levanter checkpoints

```python
from dataclasses import replace

from experiments.evals.evals import core_evals
from experiments.evals.evaluation import eval_report, eval_steps

groups = [replace(group, tokenizer="Qwen/Qwen3-8B", tokenizer_revision="<commit-sha>")
          for group in core_evals()]
results = eval_steps(checkpoint, groups, version="dev")
report = eval_report(results, name="checkpoint-core", version="dev")
```

Set the group's tokenizer and tokenizer revision when the checkpoint does not
contain tokenizer files. An object-store checkpoint can supply its own tokenizer.
Reports contain durable run references; Evaldash owns score computation.

## Audit a curated run before its record appears

Use Iris to inspect the orchestrator and its children. Inspect the corpus
dependencies and their recorded status in the artifact graph before investigating
serving: no GPU is acquired until those dependencies finish. Catalog publication
remains a separate curation operation. Once evaluation starts, inspect the raw
selection and attempt tables at the run's FineStore root. A started row without a
finished row is incomplete work.
Repeat the standalone [task driver](../references/evaluation-tasks.md#curate-and-run)
with the identical output and run configuration to resume raw attempts.

## Run the repository example scripts

`experiments.evals.run_key_evals` and `experiments.evals.run_base_model_evals` use
curated checkpoint menus. The reference training pipeline uses the same entrypoint.
Historical Evalchemy archive readers remain available for saved results. Reproducing
an old vendor-dependent campaign replay requires its recorded checkout.
