# Evaluation tasks

Evaluation corpora pair canonical `TaskSpec` rows with a benchmark protocol and
per-row execution inputs. Marin's evaluation application executes the corpus and
writes raw attempts to FineStore. Evaldash summary computation and raw-archive ingestion
remain separate work; this path produces no headline score or legacy
`record.json`.

## Ownership and data contracts

| Owner | Responsibility |
| --- | --- |
| `experiments/post_training/task_curation/` | Pinned acquisition, environments, artifact graphs, and dataset-specific cohort and budget declarations |
| `taskcompendium.importers.evalchemy` | Source parsing, prompts, conversion, item builders, rubrics, and source grading programs |
| `taskcompendium.evaluation` | Typed metadata, corpus readers, response policy, and pure cohort selection |
| `taskcompendium.pipeline.evaluation` | Population-preserving corpus publication |
| `verifyit` | Shared comparisons, mathematical equivalence, judge transport, and structured verdicts |
| `marin.rollouts.evaluation` | Corpus scheduling, attempt bookkeeping, FineStore writes, and resume |
| `rolloutengine` | Single-attempt execution, harnesses, and chat, completion, and likelihood transport |
| `experiments.evaluation.task_driver` | CLI, endpoint and credential binding, retry/backoff, Shellbox factories, and Iris submission |
| Shellbox | Machine isolation, resource access, and outer execution deadlines |

`EvaluationCorpusRecipe` in `experiments.evaluation.preparation` binds source
acquisition, prompt compilation, an `EvaluationProtocol`, and an item builder.
Benchmark data and evaluation choices live under `experiments/evaluation/benchmarks/`;
`experiments.evaluation.catalog` assembles their supported configurations.
Shot count and request format are evaluation settings. Preparation compiles their
final prompts before publication, preserving source-order sampling and immutable
TaskSpec identities. The shared curation inventory registers default evaluation
recipes alongside training sources.

Import `EvaluationProtocol`, `EvaluationItem`, `OperationKind`, and `SummaryKind`
from `taskcompendium.evaluation.spec`. The protocol's `schema_version` is `3`,
independent of TaskSpec and corpus-manifest schema versions. It records sampling,
trial seeds, metric meanings, summary semantics, context policy, and the expected
full source population. `EvaluationItem` joins each task to its original source
position and stores additional operation inputs, summary groups, and private
reference text when needed. Raw-generation prompts come from TaskSpec's context.

TaskSpec tags and string-valued labels are projected into normalized Parquet.
`task_rows(root, tags=("eval",), labels={...})` queries those columns without
decoding conversations or grader resources. A protocol's `CohortPolicy` matches
ordered `TaskGroup` declarations against those annotations, in source or
round-robin order. Groups can require a minimum model context. Missing active
groups and overlapping group matches fail rather than changing the cohort.

Dataset-specific declarations live beside their recipes. MRCR declares its
bin/needle label groups there; neither the generic schema nor the selector knows
MRCR. GraphWalks declares a reference-token output budget and prefix probe using
generic `ContextPolicy` fields. Reference counts use the run's pinned tokenizer.
LongBench v2 publishes each prompt in full and middle-truncated to a fixed ladder
of `o200k_base` token budgets, as ranked chat prompt variants. A `fit_context` run
sends the largest variant that fits the served context and output budget, so the
truncation is part of the published corpus and every model receives every item.

Operations distinguish chat generation, raw-text generation, choice likelihood,
continuation likelihood, and agent episodes. Chat preserves the source conversation without an
added submission instruction. Likelihood preserves context/continuation token
boundaries and source normalization. Ordinary protocols limit candidate rows
before context preflight. Protocols with `LimitStage.ELIGIBLE`, including MRCR
and GraphWalks, refill after context exclusions until the eligible-row limit is
met or candidates are exhausted. Both apply limits before trial expansion.
Context selection records exclusions and token
budgets; it never silently truncates an oversized chat prompt.

An `AgentEpisode` operation runs a multi-turn harness on a task machine. Its
`harness` declares the agent: `ShellAgent`, `TerminusAgent`, `MiniSweAgent`, or `PiAgent`
with Pi's version pin, thinking level, compaction reserve and setup timeout. It also declares
`EpisodeLimits`: turn and token limits, the model-turn timeout, the agent's
`total_turn_timeout`, the verifier timeout, and the `attempt_timeout` backstop. `None`
sets no limit. Agent protocols declare generation sampling, which applies to every
shell and Terminus request, followed by the model's `agent.agent_kwargs.extra_body`,
as Harbor's agents send it; Pi sends its own sampling, output limit and thinking
switch, and the protocol's `max_tokens` becomes Pi's per-reply limit. mini-swe-agent
likewise sends its own requests, with the protocol's `max_tokens` as its largest reply.
A Pi run also needs the model's `agent.agent_kwargs.thinking_format`, which says how Pi
switches the served model's thinking. The model's `agent.agent_kwargs.model_format`,
`openai` by default or `k2_horizon`, is Harbor's mini-swe-agent history format. Agent protocols declare no context policy: the harness renders its own prompts, so selection admits rows without
counting prompt tokens. Each agent item also carries `EpisodeInputs` from the source
task: an optional `agent_timeout` and `verifier_timeout`, which replace the
protocol's values, a network policy, and optional CPU, memory, storage, and GPU
requests, which replace the run's task-machine resources. Publication rejects an
agent item without episode inputs, inputs on any other item, and resolved timeouts
whose agent and verifier budgets reach the attempt backstop.

`experiments.evaluation.benchmarks.agentic` declares the Harbor datasets. Each pins
the packages its Harbor policy ran and stages one archive per package, in Harbor
Hub's layout, from one of three sources: a Hugging Face dataset commit
(`HarborRepoSource`), a Harbor Hub release by content digest (`HarborHubSource`), or,
for a Harbor registry dataset, the GitHub archive of the registry's commit, checked
against its SHA-256 (`HarborArchiveSource`). Each package lowers with
`taskcompendium.importers.harbor`, and the dataset pins its container images through
a checked-in lock in `experiments/evaluation/benchmarks/harbor_images/`. The package's
agent and verifier timeouts, resources and network policy become the item's
`EpisodeInputs`. The attempt backstop is four hours unless a dataset declares a longer
one for packages whose agent and verifier timeouts sum past it. Each reply is limited
to the policy's output limit, Harbor's default of 8,192 tokens when the policy sets
none, or the run's smaller budget. A package the importer cannot lower is a rejection,
and a corpus with a rejection does not publish. A dataset's grading rule runs each
package's `tests/test.sh` by default.
Packages whose tests call an LLM judge with host credentials instead use a rule from
`experiments.evaluation.benchmarks.harbor_judges`, which captures the answer file and
grades it on the host with a judge that sends the script's request; such a dataset
sets `requires_judge`, and its runs name the rule's judge model in their judge
identity. FinanceAgent's tools also read `SEC_EDGAR_API_KEY`, `SERPAPI_API_KEY`,
`MODEL_FOR_TOOLS` and `OPENAI_API_KEY` on the task machine, which its runs supply
through `task_secrets`. A dataset's `edits` change its packages before lowering;
FinanceAgent's replace LiteLLM in its image and `retrieve_information` tool with the
OpenAI SDK. Refresh a lock after changing a dataset's source:

```bash
uv run python -m experiments.evaluation.benchmarks.agentic lock <lock>
```

An agent-episode `EvaluationCorpusRecipe` in `experiments.evaluation` declares its
`AgentEpisode` as `agent`, and publication rejects items that run another operation.
The launcher plans and records that harness before the corpus is built. Each
registered harness has a revision in
`marin.evaluation.task_worker.EPISODE_HARNESS_REVISIONS`; bump it when the harness's
prompts, parsing, tools or options change.

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

If a grader environment has not been built, run the build command printed by
the missing-artifact error, then retry curation. The command preserves the
selected artifact root. For remote evaluation, use a shared `--artifact-root`
for curation and the same `MARIN_PREFIX` for environment builds. Locks stored
under a local `--cache-dir` belong to local runs.

Create `run-config.json` using `EvaluationRunConfig` from
`marin.rollouts.evaluation`. Replace model, tokenizer, and serving fields
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
  --dataset "$CORPUS_ROOT" \
  --config run-config.json --endpoint-file endpoint.txt \
  --output scratch/eval-run
```

Set `CORPUS_ROOT` to the completed corpus root reported by curation, rather than
assuming a path beneath its output directory. `endpoint.txt` contains the
OpenAI-compatible base URL. Supply its credential in
`OPENAI_API_KEY`, or select another variable with `--api-key-env`. Repeat
`--dataset` for mixed-corpus runs; `source_limit` applies separately to each
corpus before trial expansion. Library callers install `marin-core` and supply
clients and machine factories to
`run_evaluation`.

Concurrency defaults to 64 attempts across generation and grading. The explicit
run `max_tokens` is further bounded by the protocol and available context. A null
run seed preserves source trial seeds; an explicit seed uses base-plus-trial.
The `experiments.evaluation` launcher sets run seed 0 when none is given and the
model is served where per-request seeds are accepted. TPU vLLM rejects a seed
on sampled requests, so those runs stay unseeded unless a seed is given.
Omitted source sampling options stay omitted: for example, `top_p: null` lets the
served generation configuration apply. Record those serving defaults with the
model identity.

Set `generation_extra_body` in the run config for endpoint-specific request
options. For Grug or Snowball served without a reasoning parser, use
`"generation_extra_body": {"skip_special_tokens": false}` to keep
`<|end_think|>` in the response. The NUPA scorer needs that boundary to exclude
reasoning. The option is recorded in the run descriptor; changing it requires a
new result root.

Agent episodes require a `task_machine` in the run configuration. Its `backend`
selects the Shellbox factory, and its resources are defaults for items that set none.
The episode's limits replace the run's turn, token, model, agent, verifier, and attempt
limits; the run still supplies `tool_turn_timeout` and `cleanup_timeout`. Machine
settings are part of the run identity, so changing the backend requires a new result
root. The worker builds factories for the `local`, `docker`, and `daytona` backends that
the task and verifier machines select, and rejects others. Daytona uses the eval
account: the CLI reads its key from `DAYTONA_EVAL_API_KEY` (or `--daytona-api-key-env`)
and `DAYTONA_TARGET`, then passes the key to the Iris worker as `DAYTONA_API_KEY`
and installs `marin-core`'s `daytona` extra. Harness names resolve through
`marin.evaluation.task_worker.episode_harnesses`; library callers pass their own
`harnesses` mapping to `run_evaluation`.

A task machine's variables may read host values through whole-value `${NAME}` or
`${NAME:-default}` templates. The worker resolves them from its own environment when it creates the machine, so the
run declares each name it supplies in `task_secrets`; `run_evaluation` rejects a task
that reads any other name before its first attempt. A curated launch sets
`run.task_secrets` to ordered secret references for each name, as `judge_api_key`
does for the judge, and the worker receives the values as Iris secrets.
`task_driver` reads each declared name with `require_secret` for an Iris worker;
with `--local`, the names must be set in its environment. The run descriptor records
the names, never the values.

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
| `normalize/*.parquet` | Canonical tasks in `task_json`, with projected identities, tags and labels |
| `evaluation/*.parquet` | `EvaluationItem` rows keyed by task ID and original source position |
| `protocol.json` | Versioned benchmark execution and interpretation policy |
| `manifest.json` | Conversion provenance and row accounting |
| `evaluation_manifest.json` | Completion marker binding protocol, population, provenance, and shard hashes |

`EvaluationCorpus` in `taskcompendium.evaluation.corpus` verifies hashes and the
complete projected identity join once per read session. It decodes requested
tasks in bounded batches and skips unrelated Parquet row groups.
`evaluation_cases(root)` streams joined tasks; `evaluation_items(root)` streams
sidecars without decoding TaskSpecs. Duplicate rows at different source positions
remain distinct. Publication checks the full expected population independently
of run-time sampling; MRCR's pinned source population is 2,400 rows.

FineStore stores `samples`, `rollouts_v1`, `evaluation_attempts`, and
`evaluation_selection`, with `evaluation/run.json` and per-attempt evidence
blobs. The run descriptor binds model and driver identities to corpus/protocol
snapshots and execution settings. Raw evidence retains named metrics, weights,
extractions, token likelihoods, eligibility decisions, and each rollout's resolved limits.
Infrastructure failures remain unscored; they do not become synthetic zero scores.

An episode that reaches its turn, token, length, or agent deadline is graded on what it
completed. An attempt that fails before the harness starts, such as a failed machine
build, provider call, or setup command, is an infrastructure failure, as are an expired
attempt backstop, a terminated machine, and a deadline reached before the first model
response. Other unexpected errors stop the run after saving their evidence.

Repeat the same command and output root to resume. Request concurrency may change
without invalidating completed observations; model, source, selection and scoring
settings must remain identical. Completed logical attempts,
including zero grades, are skipped. Infrastructure failures retry with the same
row, trial, and seed and a new physical retry index. A completed model response
is reused if only grading failed. Model failures make a fresh request. The CLI
uses standard backoff between passes; each pass processes all supplied corpora before
reporting retryable failures. Solver SDK retries are disabled. Verifyit handles
judge HTTP retries separately under the source's transport policy.

Resume rejects changed model, execution revision, corpus, protocol, or run settings.
The application-owned execution revision prevents terminal grades from different
execution semantics being mixed; it is independent of library release versions.
Consumers
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
population and supplies group labels. FinanceBench uses means. EconEvals uses a
weighted mean of `win_rate` over judged games, so 0.5 is parity with the
baseline: the paper's o3-mini, or o3 for the `-o3` variants. SimpleQA and
SimpleQAMini use classification summaries.

The shared evaluation CLI and checkpoint menus now select these curated corpora.
Corpus artifacts are explicit dependencies that finish before inference starts;
executors receive their resolved roots. Catalog publication remains in the curation
workflow. Native generation menus publish
raw and chat variants separately; no runtime task registry is loaded.

`score_curated_record` in `evalrecords.rollout_results` selects one latest
finished retry per logical trial and applies these reductions. Evaldash uses it to
import raw FineStore stores; `experiments.evaluation.cli scores` prints the same
result for a run. Its sample browser uses the same trial selection.
The PostgreSQL reconciler checks both the producer record and the raw store version.
Curation identities and the summary revision participate in dashboard protocol
comparison when both records carry identities. The canonical curation graph records
the source, compiler, grader build, protocol and input overrides as `curation_identity`
in the manifest. It excludes regional output paths, lock URLs and conversion timing.
Physical manifest hashes still identify raw attempts and protect resume. Callers of
`publish_evaluation_corpus` can supply the same semantic identity explicitly.
Historical results without corpus
identities can share a column with curated scores when canonical metric names,
statistical kinds and known shot counts agree. Saved historical scores and their
coverage remain unchanged; this compatibility rule requires no model rerun or
archive rewrite. The newest identified declaration governs known corpus differences;
a newer unidentified record cannot erase them. Producer records keep their metric maps empty.

Dashboard measurements use shared normalization rules for historical and native records:

| Source metric | Canonical metric | Scope |
| --- | --- | --- |
| `acc`, `em`, `exact_match`, `exact-match`, `accuracy_avg` | `accuracy` | All benchmarks |
| `acc_norm`, `acc_norm_nospace` | `normalized_accuracy` | All benchmarks |
| `pass@1`, `pass_at_1` | `accuracy` | HumanEval, HumanEvalPlus, MBPPPlus and CRUXEval |
| `pass@1` | `pass_at_1` | Other benchmarks |
| `mrcr_accuracy` | `accuracy` | MRCR |
| `mean_reward` | `reward` | All benchmarks |

The rules strip extraction-filter suffixes such as `,none` from the selected metric.
Selection still uses the recorded primary metric and existing filter priority.
Raw metric maps retain all source spellings and filter results. Unknown metric names
remain unchanged; F1, normalized accuracy and pass rates for larger values of k are
not converted to ordinary accuracy. Normalization changes names, never score values,
coverage or recorded uncertainty.

See [task curation](task-curation.md) for acquisition and publication,
[running evaluations](../tutorials/run-lm-evals.md) for the shared launcher, and
[task rollouts](task-rollouts.md) for single-task execution. Harbor policies passed with
`--harbor-config` run through Harbor; see [Harbor integration](../harbor-integration.md). Historical Evalchemy archives remain readable; reproducing their runs
requires the recorded checkout.
