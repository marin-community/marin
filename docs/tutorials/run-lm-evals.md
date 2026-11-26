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
[Harbor integration](../harbor-integration.md) for Harbor policies. Catalog agentic
benchmarks run as curated corpora; see [evaluation tasks](../references/evaluation-tasks.md).

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
Evalchemy protocols send no temperature, so the served model's generation settings
apply; set them per model in its serving config. Runs use seed 0 unless a seed is
given. TPU vLLM rejects seeded sampling, so runs there are unseeded by default.
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

[PDBThink](../explanations/pdbthink.md) uses the published 88-question coordinate
panel with three completions per question. `--evals pdbthink` preserves each
prompt and fails before inference if any selected question exceeds the context.
Use the model's full context and output allowance; headline accuracy gives each
question family equal weight. It needs no LLM judge.

[LongBench v2](https://huggingface.co/datasets/zai-org/LongBench-v2) asks 503
multiple-choice questions over contexts of 8,000 to 2 million words, with the
upstream [THUDM/LongBench](https://github.com/THUDM/LongBench/tree/2e00731f8d0bff23dc4325161044d0ed8af94c1e)
prompt and answer parser. Each item is published in full and middle-truncated to
token budgets from 8K to 894K, and a run sends the largest copy that fits the
served context. Serve a long context, such as `qwen3.5-35b-a3b-262k` or
`qwen3-32b-yarn-128k`. `longbench-v2` decodes greedily with thinking off and 4,096
output tokens; `longbench-v2-thinking` samples with thinking on and 32,768 output
tokens, which leaves less room for the prompt. A limit takes short, medium and
long items in turn.

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
to select one entry from that document.
Ordinary benchmark defaults are available through `--evals math500,gsm8k` without
a selection file. The repository's curated suite choices and run overrides live in
`experiments/evaluation/configs/evaluations.yaml`.

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
FinanceBench, OlympiadBench, EconEvals and IMO-Bench definitions use Together and record its model alias
as provider-managed. IMO-Bench's published scores used Gemini 2.5 Pro as the judge, so IMO-AnswerBench and
IMO-ProofBench scores are comparable only between runs that share a judge. A provider alias does not establish the exact weights served.
Supply explicit judge deployment identities for reproducible private deployments.
The [September 25 publication campaign](https://github.com/marin-community/marin/tree/main/experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled)
supplies its separately pinned hosted judge through `--policy-finalization-config`.

EconEvals scores are not directly comparable with the
[EconEvals paper](https://arxiv.org/abs/2607.19375), which judged with gpt-4.1-mini. The `econevals`
suite uses the paper's o3-mini baseline; strong models beat it on most items, so compare them with
the `econevals-o3` suite, which judges against o3's released answers. The `econevals-sampled` and
`econevals-sampled-o3` suites keep six questions from each category, about a tenth of the items,
for development runs. The judge often prefers whichever answer it reads first, and such split items
score 0.5, so treat score differences under about 0.1 as noise.

### Curated agent benchmarks

A curated benchmark whose protocol is an agent episode runs its declared harness,
such as `terminus-2`, on a task machine for each attempt. Its task machines run on
Daytona unless `run.task_machine` or `--task-machine-backend` selects `docker` or
`local`. `run.task_machine` also sets default `cpus`, `memory_mb` and `storage_mb`
for tasks that declare none, and `startup_timeout`, which covers online image builds:

```yaml
evaluations:
  terminal-bench-docker:
    benchmark: terminal-bench-2.0
    run:
      task_machine:
        backend: docker
        memory_mb: 8192
```

On Daytona, `image_build` selects how each machine gets its image. The default,
`snapshot`, builds each distinct environment once into a named snapshot that counts
against the eval account's snapshot quota. `declarative` builds the image for every
machine and creates no snapshot. Catalog entries for datasets with more distinct
environments than the quota, such as `tb2`, `swebench` and `aider`, select
`declarative`:

```yaml
    run:
      task_machine:
        image_build: declarative
```

Each attempt holds a task machine, and the agent's time budget includes the model's latency,
which grows with load. `run.concurrency` bounds the attempts that run at once; the model's
`evaluation_num_concurrent` also bounds them, and the default is 64. The catalog's agent
benchmarks run as many attempts at once as their Harbor jobs did:

```yaml
    run:
      concurrency: 32
```

The model must set `serve.max_model_len`, the agent's context window. Its
`generation.max_gen_toks` and `run.max_tokens` cap each reply, as in Harbor.
The protocol can cap the context window further and set its own reply limit. The dry run
prints the harness and its revision, the backend, and the resolved
`max_input_tokens` and `max_output_tokens`. `record.json` keeps them under
`eval.curated.agent` and `eval.curated.settings.task_machine`. The backend, task
machine resources, harness revision and budgets are part of the run identity;
request concurrency is not.

## Launch controls

- `--model-config`: a model catalog YAML instead of a registry model.
- `--limit`: override the source-row cap for each selected corpus.
- `--seed`: override the run seed; trial seeds advance from it.
- `--task-machine-backend`: run curated agent benchmarks on `daytona`, `docker` or `local` task machines.
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

## Daytona and verifier credentials

Curated agent benchmarks on Daytona use the eval-org key:
`DAYTONA_EVAL_API_KEY` from the launch environment, then the `DAYTONA_EVAL_API_KEY`
secret in the `hai-gcp-models` GCP project. The evaluation worker receives it as its
`DAYTONA_API_KEY` secret and may reach `app.daytona.io`. When the task machines are on
Daytona, package tests that grade in a separate verifier machine run on Daytona too, so
the agent's changes execute only on Daytona. Graders run on the task machines' backend
in general, and in the worker's local sandbox when a run has no task machines. Judged agent graders use the
curated `judge`, `judge_endpoint` and `judge_api_key` run settings; task machines
receive no judge credentials.

Task machines receive only the host variables named in `run.task_secrets`, each
resolved from ordered secret references. A benchmark whose tasks read an undeclared
variable fails before its first attempt. The agent can read every value it receives,
so supply keys scoped for that use. FinanceAgent's tools read four variables, and its
answers go to a GPT-5 judge. No catalog entry runs FinanceAgent, because these keys are
not provisioned; a run that supplies them uses:

```yaml
evaluations:
  financeagent-curated:
    benchmark: financeagent
    run:
      judge:
        deployment: openai
        models:
          gpt-5-2025-08-07: provider-managed:gpt-5-2025-08-07
        default_model: gpt-5-2025-08-07
      judge_endpoint: https://api.openai.com/v1
      judge_api_key: [env:OPENAI_API_KEY]
      task_secrets:
        SEC_EDGAR_API_KEY: [env:SEC_EDGAR_API_KEY]
        SERPAPI_API_KEY: [env:SERPAPI_API_KEY]
        MODEL_FOR_TOOLS: [env:MODEL_FOR_TOOLS]
        OPENAI_API_KEY: [env:FINANCEAGENT_TOOLS_OPENAI_API_KEY]
```

The package's image and `retrieve_information` tool use LiteLLM; Marin's FinanceAgent
dataset edits them to install and call the OpenAI SDK instead. The tool also reads
`MODEL_API_KEY`, which the package never passes to the agent's container, so that tool
fails under Harbor as well.

## Results

Each evaluation gets a `run_id` and writes `{records_prefix}/{run_id}/record.json`.
The selected evaluations share a `group_id`; `--run-id` sets that launch ID.
Curated producer records
contain launch identity, status, jobs and the raw FineStore location. Their metric
maps stay empty. Evaldash imports the raw archive and stores its derived scores in
its own catalog; it does not rewrite the producer's record. The launcher prints the
same derived coverage and scores when it waits for a launch. For a finished run:

```bash
uv run python -m experiments.evaluation.cli scores gs://marin-eval-metadata/evals/<run_id>
```

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
from evalrecords.records import EvaluationSelection
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
