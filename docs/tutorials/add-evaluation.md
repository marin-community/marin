# Add an evaluation benchmark

This guide adds a non-agentic benchmark to Marin's curated evaluation catalog
and validates it end to end: unit tests, a local corpus build, a comparison with
a reference, and a smoke run on a small served model. Agentic benchmarks run
through Harbor instead; see [Harbor integration](../harbor-integration.md).

A curated benchmark is an immutable corpus of `TaskSpec` rows plus an
`EvaluationProtocol`. The protocol fixes the request type, sampling, trials,
metrics and summary. The launcher builds the corpus before starting inference,
the evaluation driver records one observation per attempt, and the shared
reducer computes the score. The [evaluation overview](../explanations/evaluation.md)
describes the full flow.

## Choose a pattern

Most benchmarks fit an existing pattern. Start from the closest example.

| Benchmark shape | Operation | Build with | Example |
| --- | --- | --- | --- |
| Pick one of N continuations by log-likelihood | `CHOICE_LIKELIHOOD` | `ChoiceTemplate` and `multiple_choice_evaluation` | `benchmarks/commonsense.py` |
| Few-shot raw completion graded by a source scorer | `TEXT_GENERATION` | `GenerationTemplate` and `text_generation_evaluation` | `gsm8k_cot` in `benchmarks/gsm8k.py` |
| Chat question with a programmatic answer check | `CHAT_GENERATION` | Converter, `source_package` scorer and `evaluation_source` | `gpqa-diamond` in `benchmarks/reasoning.py`, `benchmarks/jee.py` |
| Chat question graded by an LLM judge | `CHAT_GENERATION` | Converter with a judge rubric, `requires_judge=True` | `benchmarks/simpleqa.py` |

Paths are relative to `experiments/evaluation/`. Likelihood benchmarks need a
completion endpoint with prompt log-probabilities and no grader environment.
Generation benchmarks grade each response in an isolated Shellbox machine with
a pinned Python environment.

## Where the pieces go

| Piece | Location |
| --- | --- |
| Prompt rendering, row conversion, item builder | `lib/taskcompendium/src/taskcompendium/importers/evalchemy/<family>.py` or `taskcompendium/convert/` |
| New answer scorer | `lib/taskcompendium/src/taskcompendium/importers/evalchemy/scorers/`, registered in `scorers/main.py` and `importers/evalchemy/grader.py` |
| Source pin, population, protocol | `experiments/evaluation/benchmarks/<family>.py`, exposing `benchmarks()` |
| Catalog registration | `experiments/evaluation/catalog.py` |
| Non-default selections, judge bindings, suites | `experiments/evaluation/configs/evalchemy.yaml` |
| Tests | `lib/taskcompendium/tests/` for prompts and scorers; `experiments/evaluation/test_*.py` for publication |

Reuse an existing scorer when its answer format fits. `choices.py` extracts
multiple-choice letters, `math.py` and `amc.py` compare mathematical answers,
and `harness_generation.py` reproduces lm-eval-harness extraction filters.

## 1. Pin the source

Every source is a Hugging Face dataset at a 40-character commit, or a URL with a
SHA-256 digest. Mutable branches fail preparation.

```python
from huggingface_hub import HfApi

info = HfApi().dataset_info("qiaojin/PubMedQA", files_metadata=True)
print(info.sha, [file.rfilename for file in info.siblings])
```

Choose the exact files and split the reference implementation uses. A dataset
that has only a loading script usually has Parquet exports on the
`refs/convert/parquet` revision; pin that revision's commit. Count the rows you
will evaluate: `expected_population` must equal the published population, and
publication fails when it differs.

Record the license and any use restrictions in the benchmark module docstring
or the PR. Do not commit benchmark questions or answers to the repository; tests
use synthetic rows.

## 2. Render prompts

Match the reference implementation's prompt byte for byte, including
whitespace, answer delimiters and few-shot separators. Small prompt differences
move small-model scores by several points.

For a likelihood benchmark, write a render function that returns the prompt,
the candidate continuations and the gold index:

```python
from taskcompendium.importers.fewshot import (
    ChoiceDocument,
    ChoiceTemplate,
    Document,
    FewshotPolicy,
    FewshotSampling,
)


def pubmedqa_document(doc: Document) -> ChoiceDocument:
    abstract = " ".join(doc["context"]["contexts"])
    prompt = f"Abstract: {abstract}\nQuestion: {doc['question']}\nAnswer:"
    choices = ("yes", "no", "maybe")
    return ChoiceDocument(prompt, choices, choices.index(doc["final_decision"]))


def pubmedqa_template() -> ChoiceTemplate:
    return ChoiceTemplate("pubmedqa", pubmedqa_document, FewshotPolicy(0, FewshotSampling.FIRST))
```

The compiler adds the leading space before each choice, matching
lm-eval-harness's default target delimiter. `FewshotPolicy` reproduces
lm-eval-harness demonstration sampling when shots are nonzero.

For a chat benchmark, write a converter from a source row to a `TaskSpec`. The
converter builds the user message, packages the scorer with the row's private
answer, and keeps the source row as evidence:

```python
def convert_example(row: RawRow, context: ConversionContext) -> TaskSpec:
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=PROMPT.format(question=row.data["question"])),),
        package=source_package("gpqa", {"expected": row.data["answer"]}, environment=required_grader_environment(context)),
        evidence=dict(row.data),
    )
    return task.model_copy(update={"tags": ("eval", "benchmark:Example")})
```

An item builder maps each task to an `EvaluationItem`. Use `chat_item` unless
the benchmark reports subgroup scores; then return `groups` such as
`{"subject": ...}` and the reducer adds one score per group.

## 3. Grade responses

A scorer receives the private config from `source_package` and the response
text, and returns a `verifyit` `Reward`. To add one:

1. Add `scorers/<name>.py` with a pure grading function.
2. Dispatch to it in `scorers/main.py` `score()`.
3. List its module dependencies under its name in `grader.py` `DEPENDENCIES`.
4. Add its third-party requirements to an `Environment` in
   `experiments/evaluation/benchmarks/environments.py`, or reuse `BASIC_GRADER`.

`scorer_revision(name)` hashes the scorer files, so a scorer change publishes a
new corpus automatically. Grade the final answer only; a response that the
scorer cannot parse is a scored zero, not an infrastructure failure. Use the
`audit-grader` skill before relying on a new LLM judge.

## 4. Declare the benchmark

Create `experiments/evaluation/benchmarks/<family>.py` with the pin, the
protocol and a `benchmarks()` function. Use the helper for your pattern; each
returns a source whose protocol and recipe fields are explicit.

```python
from taskcompendium.evaluation.spec import LikelihoodMetric, LikelihoodNormalization
from taskcompendium.importers.evalchemy.biomedical import pubmedqa_template
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.prompt_compilation import multiple_choice_evaluation
from experiments.post_training.task_curation.pipeline import HfSource

PUBMEDQA_REVISION = "<40-character dataset commit>"


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks(
        [
            multiple_choice_evaluation(
                name="pubmedqa",
                benchmark="pubmedqa",
                source=HfSource("<org>/<dataset>", PUBMEDQA_REVISION, ("<file>.parquet",), SourceFormat.PARQUET),
                population=500,
                template=pubmedqa_template(),
                metrics=(LikelihoodMetric(name="acc,none", normalization=LikelihoodNormalization.RAW),),
            )
        ],
        "likelihood",
    )
```

`fixed_benchmarks` exposes each protocol under the lowercased protocol
`benchmark` name, which becomes the `--evals` key. Name the protocol benchmark
with the key you want, such as `pubmedqa`, rather than mapping names afterward.

For a converter-based benchmark, `evaluation_source` binds the converter, the
grader environment and an explicit protocol:

```python
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import EvaluationProtocol, Metrics, OperationKind, Sampling, SummaryKind, TrialPolicy
from taskcompendium.importers.evalchemy.grader import scorer_revision

from experiments.evaluation.benchmarks.environments import BASIC_GRADER
from experiments.evaluation.prompt_compilation import evaluation_source


def benchmarks() -> list[EvaluationBenchmark]:
    protocol = EvaluationProtocol(
        benchmark="example",
        version="2026.11.16",
        operation=OperationKind.CHAT_GENERATION,
        sampling=Sampling(
            temperature=None, top_p=None, top_k=None, max_tokens=None, stop=(),
            chat_template_kwargs={"enable_thinking": False},
        ),
        trials=TrialPolicy(count=1, seed=None),
        metrics=(Metrics.ACCURACY,),
        primary_metric="accuracy",
        summary=SummaryKind.MEAN,
        expected_population=1000,
    )
    source = evaluation_source(
        name="example",
        source=HfSource("<org>/<dataset>", EXAMPLE_REVISION, ("<file>.parquet",), SourceFormat.PARQUET),
        convert=convert_example,
        version="2026.11.16-" + scorer_revision("gpqa"),
        protocol=protocol,
        item_builder=chat_item,
        grader=BASIC_GRADER,
    )
    return fixed_benchmarks([source], "chat")
```

Protocol fields that need a deliberate choice:

| Field | Choice |
| --- | --- |
| `version` | CalVer such as `2026.11.16`. Corpus identity records converter and render functions by name, not by code, so bump the recipe and protocol versions whenever prompt or conversion code changes. |
| `sampling` | `None` for likelihood. For generation, `temperature=None` and `top_p=None` keep the served model's defaults; set them only when the reference fixes them. `max_tokens=None` uses the run's model budget. |
| `chat_template_kwargs` | `{"enable_thinking": False}` for a non-thinking protocol. Publish thinking and non-thinking variants as separate benchmarks. |
| `trials` | `TrialPolicy(count=1, seed=None)` unless the reference averages repeated samples. |
| `metrics`, `primary_metric` | The reference's headline metric first. Use `Metrics.ACCURACY` for binary correctness. |
| `summary` | `MEAN` for most benchmarks; `CLASSIFICATION` for judged correct/incorrect/abstain; `WEIGHTED_MEAN` when graders return weights. |
| `expected_population` | The exact number of evaluated rows. |

Register the module in `experiments/evaluation/catalog.py` by adding it to the
imported families and the tuple in `evaluation_benchmarks()`. The benchmark is
then available as `--evals <key>`. Add an entry to `configs/evalchemy.yaml`
only for a non-default selection, such as a judge binding or a bounded smoke
variant. Add it to a suite only when the suite's owners agree.

## 5. Test

Write behavior tests with synthetic rows; see the root `TESTING.md`.

- Prompt rendering: a synthetic source row produces the exact reference prompt,
  choices and gold index. Copy the expected prompt from the reference
  implementation's output rather than re-deriving it from your code.
- Scorer: correct, incorrect, unparseable and ambiguous responses produce the
  expected reward. Existing examples are in `lib/taskcompendium/tests/test_evalchemy_*.py`.
- Publication: `experiments/evaluation/test_prompt_publication.py` publishes a
  synthetic Parquet file through the catalog and checks the compiled prompt.
  Add a case when your benchmark adds a new request layout.

```bash
uv run pytest lib/taskcompendium/tests/test_<your_test>.py experiments/evaluation/test_prompt_publication.py
```

## 6. Build and inspect the corpus

Build the full corpus locally. The source name is the recipe name:
`multiple_choice_evaluation` prefixes it with `checkpoint-` and
`evaluation_source` with `eval-`.

```bash
uv run --package marin-core --group test python -m experiments.post_training.task_curation.driver \
  --mode quick --source checkpoint-pubmedqa \
  --cache-dir ~/.cache/marin/eval-curation --output-root /tmp/eval-curation --max-workers 2 --run
```

A generation benchmark with an unbuilt grader environment fails with the build
command to run; run it and retry. The last line reports the corpus `path`.
Print a few compiled requests and compare them with the reference:

```python
from itertools import islice

from taskcompendium.evaluation.corpus import evaluation_cases, evaluation_protocol

root = "/tmp/eval-curation/checkpoint-pubmedqa"
print(evaluation_protocol(root)[0].model_dump_json(indent=2))
for case in islice(evaluation_cases(root), 3):
    print(case.task.context.events)
    print(case.item.operation)
```

## 7. Smoke test on a served model

Check the launch plan, then run a bounded evaluation on one GPU. Qwen3-0.6B is
small enough for a quick, cheap run:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-0.6b --evals pubmedqa \
  --platform gpu --accelerator H100x1 --federated_cluster cw-us-east-02a \
  --limit 50 --version dev --dry-run
```

Remove `--dry-run` to submit. The launcher builds the corpus in the serving
region, serves the model, runs the evaluation, and prints each run's coverage
and scores when it finishes. Print them again later, or for a `--no-wait`
launch, with:

```bash
uv run python -m experiments.evaluation.cli scores gs://marin-eval-metadata/evals/<run_id>
```

A complete smoke has every selected item scored and no infrastructure errors.
Remove `--limit` for the full population before comparing with a reference.

## 8. Validate against a reference

A smoke shows that the benchmark runs; it does not show that the score is right.
Validate with the strongest reference available:

1. **Item-level agreement with the reference implementation.** Run the original
   harness on the same model and items and compare per-item outcomes. For
   likelihood benchmarks, lm-eval-harness with the Hugging Face backend runs a
   0.6B model on a CPU. Expect agreement on nearly every item; investigate each
   disagreement, since numerical differences between vLLM and Hugging Face only
   flip near-ties.
2. **Published scores.** Compare with the model card or paper under the same
   protocol. Agreement within about two standard errors is consistent;
   differences in prompt, shots or thinking mode explain larger gaps only when
   you can name them.
3. **Grader spot check.** Read at least 20 graded responses, including every
   unparseable answer, and confirm each grade by hand.

Report the population, score, standard error, coverage, reference value and
the source of the reference in the PR.
