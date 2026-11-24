# Add an evaluation benchmark

This guide adds a non-agentic benchmark to Marin's curated evaluation catalog
and validates it end to end: unit tests, a local corpus build, a smoke run on a
small served model, and a comparison with a reference. Agentic benchmarks are
Harbor task packages declared in `experiments.evaluation.benchmarks.agentic`; see
[evaluation tasks](../references/evaluation-tasks.md).

A curated benchmark is an immutable corpus of `TaskSpec` rows plus an
`EvaluationProtocol`. The protocol fixes the request type, sampling, trials,
metrics and summary. The launcher builds the corpus before starting inference,
the evaluation driver records one observation per attempt, and a shared reducer
computes the score. The [evaluation overview](../explanations/evaluation.md)
describes the full flow.

## Choose a pattern

Most benchmarks fit an existing pattern. Read the example for your pattern end
to end before writing code; paths are relative to `experiments/evaluation/`.

| Benchmark shape | Operation | Build with | Example |
| --- | --- | --- | --- |
| Pick one of N continuations by log-likelihood | `CHOICE_LIKELIHOOD` | `ChoiceTemplate` and `multiple_choice_evaluation` | PubMedQA in `benchmarks/biomedical.py` |
| Chat question with a programmatic answer check | `CHAT_GENERATION` | Converter, packaged scorer and `evaluation_source` | MMLU-Redux in `benchmarks/mmlu_redux.py` |
| Few-shot raw completion graded by a source scorer | `TEXT_GENERATION` | `GenerationTemplate` and `text_generation_evaluation` | `gsm8k_cot` in `benchmarks/gsm8k.py` |
| Chat question graded by an LLM judge | `CHAT_GENERATION` | Converter with a judge rubric, `requires_judge=True` | `benchmarks/simpleqa.py` |

Likelihood benchmarks score prompt log-probabilities and need no grader
environment. Generation benchmarks grade each response in an isolated Shellbox
machine with a pinned Python environment.

## Where the pieces go

| Piece | Location |
| --- | --- |
| Prompt rendering, row conversion, item builder | `lib/taskcompendium/src/taskcompendium/importers/evalchemy/<family>.py` |
| New answer scorer | `lib/taskcompendium/src/taskcompendium/importers/evalchemy/scorers/`, registered in `scorers/main.py` and `importers/evalchemy/grader.py` |
| Source pin, population, protocol | `experiments/evaluation/benchmarks/<family>.py`, exposing `benchmarks()` |
| Catalog registration | `experiments/evaluation/catalog.py` |
| Non-default selections, judge bindings, suites | `experiments/evaluation/configs/evaluations.yaml` |
| Prompt and scorer tests | `lib/taskcompendium/tests/`: `test_checkpoint_harness.py` for likelihood templates, `test_evalchemy_<family>.py` for converters and scorers |

Reuse an existing scorer when its answer format fits. `choices.py` extracts
multiple-choice letters, `zeroeval.py` reads ZeroEval's JSON answers, `math.py`
and `amc.py` compare mathematical answers, and `harness_generation.py`
reproduces lm-eval-harness extraction filters.

## 1. Choose the reference

The reference implementation defines the prompt, answer extraction, shots,
sampling and evaluated rows. Pin it to a commit or release: an
lm-eval-harness task, the benchmark authors' runner, or the evaluation code
behind a model card. When a paper reports scores without publishing its
harness, use a public runner whose published scores reproduce the paper's
numbers for several models, and record that evidence. If the runner's code
cannot be executed locally, reproduce its prompt from the template source and
test against those bytes.

Also choose the score you will validate against: per-item results from running
the reference implementation on a model Marin can serve, or a published score
for such a model. Per-item results are stronger.

## 2. Pin the source

Every source is a Hugging Face dataset at a 40-character commit, or a URL with a
SHA-256 digest. Mutable branches fail preparation.

```python
from huggingface_hub import HfApi

info = HfApi().dataset_info("bigbio/pubmed_qa", revision="refs/convert/parquet", files_metadata=True)
print(info.sha, [file.rfilename for file in info.siblings])
```

Pin the exact files and split the reference evaluates. A dataset that ships
only a loading script usually has Parquet exports on the `refs/convert/parquet`
revision; pin that revision's commit. Set `expected_population` to the row
count of the pinned files, not a count from a paper; publication fails when the
two differ.

Record the license in the benchmark module docstring. Do not commit benchmark
questions or answers to the repository; tests use synthetic rows.

## 3. Render prompts

Match the reference prompt byte for byte, including whitespace, answer
delimiters and few-shot separators. Small prompt differences move small-model
scores by several points. Write templates with explicit `\n` escapes rather
than triple-quoted strings: editors and tools strip trailing spaces, which some
reference templates contain.

For a likelihood benchmark, a render function returns the prompt, the candidate
continuations and the gold index. PubMedQA's, from
`importers/evalchemy/biomedical.py`:

```python
PUBMEDQA_CHOICES = ("yes", "no", "maybe")


def pubmedqa_document(doc: Document) -> ChoiceDocument:
    abstract = "\n".join(doc["CONTEXTS"])
    prompt = f"Abstract: {abstract}\nQuestion: {doc['QUESTION']}\nAnswer:"
    return ChoiceDocument(prompt, PUBMEDQA_CHOICES, PUBMEDQA_CHOICES.index(doc["final_decision"]))


def pubmedqa_template() -> ChoiceTemplate:
    return ChoiceTemplate("pubmedqa", pubmedqa_document, FewshotPolicy(0, FewshotSampling.FIRST))
```

The compiler adds a space before each choice, matching lm-eval-harness's
default target delimiter. A nonzero `FewshotPolicy` reproduces lm-eval-harness
demonstration sampling.

For a chat benchmark, a converter turns a source row into a `TaskSpec`: the user
message, the scorer packaged with the row's private answer, and the source row
as evidence. MMLU-Redux's, from `importers/evalchemy/mmlu_redux.py`:

```python
def convert_mmlu_redux(row: RawRow, context: ConversionContext) -> TaskSpec:
    choices = list(row.data["choices"])
    expected = string.ascii_uppercase[choices.index(row.data["correct_answer"])]
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=zeroeval_mcqa_prompt(row.data["question"], choices)),),
        package=source_package(
            "zeroeval_mcqa", {"expected": expected}, environment=required_grader_environment(context)
        ),
        evidence={**row.data, "choices": choices},
    )
    return task.model_copy(update={"tags": ("eval", "source:zeroeval", "benchmark:MMLU-Redux")})


def mmlu_redux_item(task: TaskSpec) -> EvaluationItem:
    return chat_item(task, response_selection="final_only").model_copy(
        update={"groups": {"subject": source_evidence(task)["config"]}}
    )
```

The item builder maps each task to an `EvaluationItem`. Use `chat_item` alone
unless the benchmark reports subgroup scores; with `groups`, the reducer adds
one score per group, such as `subject=anatomy/accuracy`.

Every chat item states which part of the reply is graded. Generation evals
grade the final answer (`final_only`), so graders never see reasoning and a
reply cut off mid-reasoning scores as empty. Choose a content-first selection
only when the benchmark's reference harness requires it, as MATH500, AIME and
MMLU-Pro do.

## 4. Grade responses

A scorer receives the private config from `source_package` and the response
text, and returns a `verifyit` `Reward`. To add one:

1. Add `scorers/<name>.py` with a pure grading function.
2. Dispatch to it in `scorers/main.py` `score()`.
3. List its module dependencies under its name in `grader.py` `DEPENDENCIES`.
4. Add its third-party requirements to an `Environment` in
   `experiments/evaluation/benchmarks/environments.py`, or reuse `BASIC_GRADER`.

`scorer_revision(name)` hashes the scorer files, so a scorer change publishes a
new corpus automatically. Grade the final answer only; a response the scorer
cannot parse is a scored zero, not an infrastructure failure. Use the
`audit-grader` skill before relying on a new LLM judge.

## 5. Declare the benchmark

Create `experiments/evaluation/benchmarks/<family>.py` with the pin, the
protocol and a `benchmarks()` function. A likelihood benchmark passes its
template to `multiple_choice_evaluation`:

```python
def sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    return [
        multiple_choice_evaluation(
            name="pubmedqa",
            benchmark="pubmedqa",
            source=HfSource(
                "bigbio/pubmed_qa",
                PUBMEDQA_PARQUET_REVISION,
                ("pubmed_qa_labeled_fold0_source/test/0000.parquet",),
                SourceFormat.PARQUET,
            ),
            population=500,
            template=pubmedqa_template(),
            metrics=(LikelihoodMetric(name="acc,none", normalization=LikelihoodNormalization.RAW),),
        )
    ]


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks(sources(), "likelihood")
```

A converter-based benchmark builds its protocol explicitly and binds it with
`evaluation_source`:

```python
def benchmarks() -> list[EvaluationBenchmark]:
    protocol = EvaluationProtocol(
        benchmark="mmlu-redux",
        version=VERSION,
        operation=OperationKind.CHAT_GENERATION,
        sampling=Sampling(
            temperature=0.0,
            top_p=None,
            top_k=None,
            max_tokens=None,
            stop=(),
            chat_template_kwargs={"enable_thinking": False},
        ),
        trials=TrialPolicy(count=1, seed=None),
        metrics=(Metrics.ACCURACY,),
        primary_metric="accuracy",
        summary=SummaryKind.MEAN,
        expected_population=MMLU_REDUX_POPULATION,
    )
    source = evaluation_source(
        name="mmlu-redux",
        source=HfSource("yuchenlin/zero-eval", ZEROEVAL_DATA_REVISION, (MMLU_REDUX_FILE,), SourceFormat.PARQUET),
        convert=convert_mmlu_redux,
        version=VERSION + "-" + scorer_revision("zeroeval_mcqa"),
        protocol=protocol,
        item_builder=mmlu_redux_item,
        grader=BASIC_GRADER,
    )
    return fixed_benchmarks([source], "chat")
```

`fixed_benchmarks` exposes each protocol under its `benchmark` name, lowercased
with `_` replaced by `-`; that is the `--evals` key. Name the protocol
benchmark with the key you want rather than mapping names afterward.

Protocol fields that need a deliberate choice:

| Field | Choice |
| --- | --- |
| `version` | CalVer such as `2026.11.16`, in both the protocol and the `evaluation_source` recipe. Corpus identity records converter and render functions by name, not by code: bump the version whenever prompt or conversion code changes, or the launcher reuses the stale corpus. `multiple_choice_evaluation` shares `EXPLICIT_TEMPLATE_VERSION`; a new template needs no bump, but changing an existing one requires bumping it, which republishes every benchmark built by that helper. |
| `sampling` | `None` for likelihood. For generation, `temperature=None`, `top_p=None` and `top_k=None` keep the served model's defaults. Fix them only when the reference does, and note that fixed values apply to every model evaluated. `max_tokens=None` uses the run's model budget. |
| `chat_template_kwargs` | `{"enable_thinking": False}` for a non-thinking protocol. Publish thinking and non-thinking variants as separate benchmarks. |
| `trials` | `TrialPolicy(count=1, seed=None)` unless the reference averages repeated samples. |
| `metrics`, `primary_metric` | The reference's headline metric first. Use `Metrics.ACCURACY` for binary correctness. |
| `summary` | `MEAN` for most benchmarks; `CLASSIFICATION` for judged correct, incorrect or abstained answers; `WEIGHTED_MEAN` when graders return weights. |
| `expected_population` | The row count of the pinned files. |

Register the module in `experiments/evaluation/catalog.py` by adding it to the
imported families and the tuple in `evaluation_benchmarks()`. The benchmark is
then available as `--evals <key>`. Add an entry to `configs/evaluations.yaml`
only for a non-default selection, such as a judge binding or a bounded smoke
variant. Add it to a suite only when the suite's owners agree.

## 6. Test

Write behavior tests with synthetic rows; see the root `TESTING.md`.

- Prompt rendering: a synthetic source row produces the exact reference prompt,
  choices and gold index. Take the expected prompt from the reference
  implementation's own rendering rather than re-deriving it from your code.
- Scorer: correct, incorrect, unparseable and ambiguous responses produce the
  expected reward.
- Publication, for a new converter: publish a synthetic Parquet file through
  the catalog and check the compiled request, as
  `experiments/evaluation/test_mmlu_redux_publication.py` does.

```bash
uv run --package marin-core --group test pytest \
  lib/taskcompendium/tests/test_<your_test>.py experiments/evaluation/test_<your_test>.py
```

## 7. Build and inspect the corpus

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
Compare every compiled request with the reference's rendering, not a sample:

```python
from taskcompendium.evaluation.corpus import evaluation_cases, evaluation_protocol

root = "/tmp/eval-curation/checkpoint-pubmedqa"
print(evaluation_protocol(root)[0].model_dump_json(indent=2))
for case in evaluation_cases(root):
    ...  # Compare case.task.context.events or case.item.operation with the reference.
```

For likelihood benchmarks, `case.item.operation.requests` holds each choice's
`(context, continuation)` pair, and lm-eval-harness logs the same pairs as
`arguments` in its `samples_<task>.jsonl` (run with `--log_samples`). Its
`target` field is the gold answer string, not an index.

## 8. Smoke test on a served model

Check the launch plan, then run a bounded evaluation. Qwen3-0.6B is small
enough for a quick, cheap run. GPU launches run on CoreWeave and require your
object-storage keys exported as `CW_KEY_ID` and `CW_KEY_SECRET` (see
[secret delivery](https://github.com/marin-community/marin/blob/main/lib/iris/docs/secrets.md));
the launcher refuses a CoreWeave launch without them. Without the keys, omit
the GPU options to run on the model's default GCP TPU.

```bash
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-0.6b --evals pubmedqa \
  --platform gpu --accelerator H100x1 --federated_cluster cw-us-east-02a \
  --limit 50 --version dev --dry-run
```

Remove `--dry-run` to submit. The launcher prints the Iris job and the records
prefix, builds the corpus in the serving region, serves the model, runs the
evaluation, and prints each run's coverage and scores when it finishes. It
exits non-zero, with the job's error, when an evaluation writes no record.
Print the scores again later, or after a `--no-wait` launch, from the run
directory under the printed records prefix:

```bash
uv run python -m experiments.evaluation.cli scores <records-prefix>/<run_id>
```

GPU runs write records under `s3://marin-us-east-02a/marin/evals` and TPU runs
under `gs://marin-eval-metadata/evals`. A complete smoke has every selected
item scored and no infrastructure errors. Remove `--limit` for the full
population before comparing with a reference.

## 9. Validate against a reference

A smoke shows that the benchmark runs; it does not show that the score is right.
Validate with the strongest reference available:

1. **Item-level agreement with the reference implementation.** Run the original
   harness on the same model and items and compare per-item outcomes. For
   likelihood benchmarks, lm-eval-harness with the Hugging Face backend runs a
   0.6B model on a CPU in about 15 minutes for 500 items:

   ```bash
   uv run --no-project --with 'lm-eval[hf]==0.4.12' --with 'datasets==3.6.0' --with torch \
     lm_eval --model hf \
     --model_args pretrained=Qwen/Qwen3-0.6B,revision=c1899de289a04d12100db370d81485cdf75e47ca,dtype=float32 \
     --device cpu --tasks pubmedqa --log_samples --output_path /tmp/lm-eval-pubmedqa --trust_remote_code
   ```

   `datasets` 3.6 still runs dataset loading scripts, which some lm-eval tasks
   need. Expect agreement on nearly every item; numerical differences between
   vLLM and Hugging Face flip only near-ties, so examine every disagreement.
   Read Marin's per-item results with `evalrecords.rollout_results.logical_samples`
   over a `finestore.reader.ReadView` of the run's results path.
2. **Published scores.** Compare with the model card or paper under the same
   protocol. Agreement within about two standard errors is consistent. A larger
   gap needs named causes, such as a different prompt, sampling setting or row
   filter.
3. **Grader spot check.** For generation benchmarks, read at least 20 graded
   responses, including every unparseable answer, and confirm each grade by
   hand.

Report the population, score, standard error, coverage, reference value and
the source of the reference in the PR.
