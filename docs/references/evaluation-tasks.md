# Evaluation tasks

Evaluation corpora pair canonical `TaskSpec` rows with a benchmark protocol and
per-row execution inputs. RolloutEngine executes the corpus and writes raw
attempts to FineStore. Evaldash summary computation and raw-archive ingestion
remain separate work; this path produces no headline score or legacy
`record.json`.

## Ownership and data contracts

| Owner | Responsibility |
| --- | --- |
| `experiments/post_training/task_curation/` | Pinned acquisition, environment declarations, artifact graphs, and the `EvalDataset` binding |
| `taskcompendium.importers.evalchemy` | Source parsing, prompts, conversion, item builders, rubrics, and source grading programs |
| `taskcompendium.evaluation` | Typed metadata, corpus readers, response policy, and pure cohort selection |
| `taskcompendium.pipeline.evaluation` | Population-preserving corpus publication |
| `verifyit` | Shared comparisons, mathematical equivalence, judge transport, and structured verdicts |
| `rolloutengine.evaluation` | Generation and likelihood execution, grading, attempt bookkeeping, and FineStore writes |
| `experiments.evaluation.task_driver` | CLI, endpoint and credential binding, retry/backoff, Shellbox factories, and Iris submission |
| Shellbox | Machine isolation, resource access, and outer execution deadlines |

`EvalDataset` in `experiments.post_training.task_curation.evaluation` binds a
`CurationRecipe`, an `EvaluationProtocol`, and an item builder. Leaf declarations
live in `datasets/evalchemy/`; the shared catalog is
`experiments.post_training.task_curation.sources.all_sources()`.

Import `EvaluationProtocol`, `EvaluationItem`, `OperationKind`, and `SummaryKind`
from `taskcompendium.evaluation.spec`. The protocol's `schema_version` is `2`,
independent of TaskSpec and corpus-manifest schema versions. It records sampling,
trial seeds, metric meanings, summary semantics, context policy, and the expected
full source population. `EvaluationItem` joins each task to its original source
position and stores operation inputs, grouping, and selection metadata.

Operations distinguish chat generation, raw-text generation, choice likelihood,
and continuation likelihood. Chat preserves the source conversation without an
added submission instruction. Likelihood preserves context/continuation token
boundaries and source normalization. Row eligibility precedes the run's source
limit and trial expansion. Context selection records exclusions and token
budgets; it never silently truncates an oversized chat prompt.

Use CalVer for authored recipe and protocol revisions. Keep upstream commit
pins, content hashes, and schema versions in their own fields. A changed
converter, grader asset, or protocol can change corpus identity. Publish a fresh
corpus and use a new result root when identity changes.

## Curate and run

Select evaluation sources explicitly; default training curation excludes them.
QUICK publishes the complete declared evaluation population, skips RL admission
and review, and refuses missing, rejected, or rewritten rows.

```bash
uv run --package marin-core --group test python \
  -m experiments.post_training.task_curation.driver \
  --mode quick --source evalchemy-math500 \
  --cache-dir ~/.cache/marin/eval-curation \
  --output-root scratch/eval-curation --max-workers 2 --run
```

The command reports the corpus root and records it in `campaign.json`. Local
source overrides use the curation CLI's `--input-file` and `--input` options;
their byte hashes enter artifact identity.

Create `run-config.json` using `EvaluationRunConfig` from
`rolloutengine.evaluation.runner`. Replace model, tokenizer, and serving fields
with the identities of the actual endpoint:

```json
{
  "model": {
    "name": "YOUR_MODEL",
    "revision": "YOUR_MODEL_REVISION",
    "tokenizer": "YOUR_TOKENIZER",
    "tokenizer_revision": "YOUR_TOKENIZER_REVISION",
    "context_window": 32768,
    "serving": {"backend": "vllm", "version": "YOUR_VERSION", "hardware": "YOUR_HARDWARE"}
  },
  "max_tokens": 4096,
  "source_limit": 8,
  "limits": {
    "max_turns": 1, "max_total_tokens": null,
    "model_turn_timeout": 600, "tool_turn_timeout": 60,
    "total_turn_timeout": 600, "attempt_timeout": 900,
    "verifier_timeout": 240, "cleanup_timeout": 15
  }
}
```

```bash
uv run --package marin-core --group test python \
  -m experiments.evaluation.task_driver --local \
  --dataset scratch/eval-curation/evalchemy-math500 \
  --config run-config.json --endpoint-file endpoint.txt \
  --output scratch/eval-run
```

`endpoint.txt` contains the OpenAI-compatible base URL. Supply its credential in
`OPENAI_API_KEY`, or select another variable with `--api-key-env`. Repeat
`--dataset` for mixed-corpus runs; `source_limit` applies separately to each
corpus before trial expansion. Standalone library callers install
`marin-rolloutengine[evaluation]` and supply clients and machine factories to
`run_evaluation`.

Concurrency defaults to 64 attempts across generation and grading. The explicit
run `max_tokens` is further bounded by the protocol and available context. A null
run seed preserves source trial seeds; an explicit seed uses base-plus-trial.
Omitted source sampling options stay omitted: for example, `top_p: null` lets the
served generation configuration apply. Record those serving defaults with the
model identity.

Omit `--local` to submit a disposable privileged Iris CPU worker. Shellbox's
bubblewrap backend needs permission to create namespaces; local execution needs
a host with the same support. Grader machines deny network access unless a
scripted judge declares that capability. Choose a cluster near the corpora,
results, and inference endpoint, and use shared storage roots for remote jobs.
Use `python -m experiments.evaluation.task_driver --help` for worker resources,
cluster selection, and pass limits.

Judge tasks also require `--judge-endpoint`, a credential selected by
`--judge-api-key-env`, and a `judge` object in the run configuration:

```json
{
  "judge": {
    "deployment": "YOUR_JUDGE_DEPLOYMENT",
    "models": {"gpt-4o-mini": "YOUR_IMMUTABLE_JUDGE_REVISION"},
    "default_model": null
  }
}
```

Read requested aliases from `taskcompendium.models.judge_spec(case.task.grader)`
while iterating `evaluation_cases(root)`; tasks without a judge return `None`.
`deployment` identifies the service; `models` maps every requested alias to an
immutable revision. `default_model` supplies the alias for source graders that
omit one. These fields record the operator's pin; they cannot verify the
provider's deployment. Endpoint URLs and credentials stay outside the persisted
run descriptor. Changing judge identity requires a new run.

Source grading programs and private tests are packaged with their pinned grader
environment. Source program builders prepare tests and apply the benchmark
execution timer. A completion receipt distinguishes finished tests from an early
process exit. Shellbox provides the sandbox and bounds the whole grader command. Verifyit remains the shared scoring and verdict layer.

## Published corpus and raw results

| Corpus file | Contract |
| --- | --- |
| `normalize/*.parquet` | Canonical tasks in the conversion `task_json` column |
| `evaluation/*.parquet` | `EvaluationItem` rows keyed by task ID and original source position |
| `protocol.json` | Versioned benchmark execution and interpretation policy |
| `manifest.json` | Conversion provenance and row accounting |
| `evaluation_manifest.json` | Completion marker binding protocol, population, provenance, and shard hashes |

`evaluation_cases(root)` in `taskcompendium.evaluation.corpus` verifies these
bindings and streams joined tasks. `evaluation_items(root)` reads sidecars
without decoding large TaskSpecs. Duplicate rows at different source positions
remain distinct. Publication checks the full expected population independently
of run-time sampling; MRCR's pinned source population is 2,400 rows.

FineStore stores `samples`, `rollouts_v1`, `evaluation_attempts`, and
`evaluation_selection`, with `evaluation/run.json` and per-attempt evidence
blobs. The run descriptor binds model and driver identities to corpus/protocol
snapshots and execution settings. Raw evidence retains named metrics, weights,
extractions, token likelihoods, and eligibility decisions. Infrastructure
failures remain unscored; they do not become synthetic zero scores.

Repeat the same command and output root to resume. Completed logical attempts,
including zero grades, are skipped. Infrastructure failures retry with the same
row, trial, and seed and a new physical retry index. The CLI uses standard
backoff between passes; each pass processes all supplied corpora before
reporting retryable failures. Solver SDK retries are disabled. Verifyit handles
judge HTTP retries separately under the source's transport policy.

Resume rejects changed model, driver, corpus, protocol, or run settings. Consumers
must select the latest terminal physical attempt for each dataset snapshot,
source row, and trial to avoid counting retries twice. Terminal rows have
`phase="finished"`; choose the highest `retry` index. Rows left at
`phase="started"` have no terminal result. Exhausted infrastructure
failures preserve evidence and raise `EvaluationInfrastructureError`.

## Summary semantics

`EvaluationProtocol.summary` describes the consumer's reduction of raw
observations. The runtime does not compute these summaries. Evaldash must also
use the selection ledger and distinguish unscored failures from graded answers.

| `SummaryKind` | Reduction |
| --- | --- |
| `MEAN` (`mean`) | Arithmetic mean of each metric over its scored population |
| `WEIGHTED_MEAN` (`weighted_mean`) | Sum of weighted observations divided by the sum of explicit per-metric weights in grading detail |
| `CLASSIFICATION` (`classification`) | Correct, incorrect, and abstention counts; accuracy and class rates divide by their total, attempted accuracy divides correct by correct + incorrect |
| `PERPLEXITY` (`perplexity`) | `exp(-mean(log_likelihood))` over per-row continuation log-likelihood sums; accuracy remains an arithmetic mean |

Classification excludes judge failures from the judged total. Its F1 is the
harmonic mean of accuracy and attempted accuracy; attempted accuracy and F1 are
zero when their denominators are zero. LAMBADA likelihoods sum token log
probabilities for each continuation before the row mean; the perplexity summary
does not divide by the total number of continuation tokens.

Groups use the declared reduction independently. MRCR's headline is a row mean;
its cell means are separate summaries, with no equal weighting of cells in the
headline. NUPA uses ordinary means; task/bucket stratification constructs the
population and supplies group labels. FinanceBench uses means. SimpleQA and
SimpleQAMini use classification summaries.

See [task curation](task-curation.md) for acquisition and publication and
[task rollouts](task-rollouts.md) for single-task execution. Legacy Harbor
execution migration and Evaldash's raw-archive importer remain separate from
this evaluation path.
