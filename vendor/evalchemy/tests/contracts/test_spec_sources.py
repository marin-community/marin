"""Normalized source scores and samples agree with the pinned task definitions."""

import asyncio
import json
from pathlib import Path

import pytest
import yaml
from lm_eval.api.model import LM
from lm_eval import evaluator
from lm_eval.tasks import TaskManager

from eval.contracts.eval_spec import RequestKind
from eval.execution.engine import CollectSamples, Evaluator
from eval.execution.grading import Graders
from eval.sources.lm_eval import LMEvalMetrics, LMEvalSource
from eval.sources.common import SourceOptions
from eval.contracts.eval_spec import SourceRef
from eval.graders.specs import GRADERS
from eval.execution.metrics import SourceMetrics
from eval.chat_benchmarks.MultiPLE.eval_instruct import DATA_DIR, LANUGUAGES, LANGUAGE_MAP, MultipleBenchmark
from eval.chat_benchmarks.MBPP.eval_instruct import MBPPBenchmark
from eval.graders.program_suites import program_suite


class _FixtureLM(LM):
    def generate_until(self, requests):
        return ["2" if index % 3 < 2 else "wrong" for index, _ in enumerate(requests)]

    def loglikelihood(self, requests):
        return [(-1.0 if request.args[1].strip() == "2" else -10.0, True) for request in requests]

    def loglikelihood_rolling(self, requests):
        return [-float(len(request.args[0])) for request in requests]


class _FixtureBackend:
    def __init__(self):
        self.generations = 0

    async def execute(self, request):
        if request.kind is RequestKind.GENERATE:
            self.generations += 1
            return "2" if (self.generations - 1) % 3 < 2 else "wrong"
        if request.kind is RequestKind.LIKELIHOOD:
            return (-1.0 if request.continuation.strip() == "2" else -10.0, True)
        return -float(len(request.prompt))


@pytest.mark.parametrize("answer, expected", [("signal.getitimer(signal.ITIMER_REAL)[0] > 5", 1.0), ("False", 0.0)])
def test_mbpp_source_preserves_original_native_execution_budget(tmp_path, answer, expected):
    # Observe the real sandbox timer without waiting: the original ten-second
    # budget admits this program, whereas the regressed three-second budget fails.
    code = f"def has_original_budget():\n    import signal\n    return {answer}\n"
    examples = [{"task_id": index, "text": "Check the native execution budget", "test_list": ["assert has_original_budget()"],
                 "code": code} for index in range(4)]
    (tmp_path / "mbpp.jsonl").write_text("\n".join(json.dumps(row) for row in examples))
    (tmp_path / "mbpp_test.jsonl").write_text(json.dumps({
        "task_id": 0, "prompt": "Check the native execution budget", "test": ["assert has_original_budget()"], "code": code,
    }) + "\n")
    source = MBPPBenchmark(data_dir=str(tmp_path), start_idx=0, end_idx=1)
    [spec] = source.iter_specs(SourceOptions("MBPP", source.SOURCE))

    grade = program_suite(spec, [f"```python\n{code}```"])

    assert grade.metrics == {"accuracy": expected}


@pytest.mark.parametrize("language", LANUGUAGES)
def test_multiple_bundled_rows_preserve_prompt_and_grader_reference(language):
    row = json.loads((Path(DATA_DIR) / f"multipl-e-{language}.json").read_text())[0]
    source = MultipleBenchmark(languages=[language], max_tokens=768, timeout=3)
    options = SourceOptions("MultiPLE", source.SOURCE, limit=1)

    specs = list(source.iter_specs(options))

    assert len(specs) == 1
    spec = specs[0]
    assert spec.source_id == row["task_id"]
    assert spec.namespace == language
    assert spec.doc == row
    assert spec.target == row["test"]
    assert spec.requests[0].prompt == [
        {"role": "system", "content": "You are a helpful programming assistant designed to complete code snippets."},
        {"role": "user", "content": (
            "Please generate code to complete the following problem:\n"
            f"        ```{LANGUAGE_MAP[language]}\n        {row['prompt']}\n        ```\n        "
        )},
    ]
    assert spec.requests[0].generation == {"temperature": 0, "max_gen_toks": 768}
    assert spec.grader.name == "program_suite"
    assert spec.grader.reference == {"suite": "MultiPLE", "language": language, "problem": row, "timeout": 3}
    assert spec.tags == {"language": language}


@pytest.mark.parametrize("kind", ["generate_until", "multiple_choice", "loglikelihood_rolling"])
def test_lm_eval_iterator_matches_native_scoring_with_filters_and_choices(tmp_path, kind):
    (tmp_path / "data.jsonl").write_text(json.dumps({"question": "1+1?", "answer": "2", "choices": ["1", "2"]}) + "\n")
    config = {
        "task": "spec_probe", "dataset_path": "json",
        "dataset_kwargs": {"data_files": {"test": str(tmp_path / "data.jsonl")}},
        "test_split": "test", "output_type": kind,
        "doc_to_text": "{{question}}", "doc_to_target": "{{answer}}",
        "num_fewshot": 0,
    }
    if kind == "multiple_choice":
        config.update(doc_to_choice="choices", doc_to_target=1, metric_list=[{"metric": "acc"}, {"metric": "acc_norm"}])
    elif kind == "generate_until":
        config.update(
            repeats=3, metric_list=[{"metric": "exact_match"}],
            filter_list=[
                {"name": "first", "filter": [{"function": "take_first"}]},
                {"name": "majority", "filter": [{"function": "majority_vote"}, {"function": "take_first"}]},
            ],
        )
    else:
        config.update(metric_list=[{"metric": name} for name in ("word_perplexity", "byte_perplexity", "bits_per_byte")])
    (tmp_path / "spec_probe.yaml").write_text(yaml.safe_dump(config))
    baseline = evaluator.simple_evaluate(
        model=_FixtureLM(), tasks=["spec_probe"],
        task_manager=TaskManager(include_path=str(tmp_path)), limit=1, bootstrap_iters=0,
    )
    loaded = TaskManager(include_path=str(tmp_path)).load(["spec_probe"])
    source = LMEvalSource("spec_probe", loaded["tasks"]["spec_probe"], 1, 0, {}, 1234)
    reducer = LMEvalMetrics(loaded["tasks"], loaded["groups"], bootstrap_iters=0)
    samples = CollectSamples()
    engine = Evaluator(_FixtureBackend(), Graders({source.grader_name: source.grader()}), samples, workers=2)

    async def run():
        async for result in engine.run(source):
            assert result.error is None
            reducer.add(result.result)

    asyncio.run(run())
    actual = reducer.finish("spec_probe")
    assert actual["results"] == baseline["results"]
    assert actual["n-samples"] == baseline["n-samples"]
    record = samples.samples["spec_probe"][0]
    original = baseline["samples"]["spec_probe"][0]
    assert record["resps"] == [[response] for cohort in original["resps"] for response in cohort]
    assert record["arguments"] == [
        arguments for arguments, cohort in zip(original["arguments"], original["resps"], strict=True) for _response in cohort
    ]
    for name in ("filtered_resps", "target", "doc_hash", "prompt_hash", "target_hash"):
        assert record[name] == original[name]
    if kind == "generate_until":
        assert {variant["filter"] for variant in record["filter_variants"]} == {"first", "majority"}


@pytest.mark.parametrize("program, expected", [("print(input())", 1.0), ("print(float(input()))", 0.0)])
def test_codeelo_source_uses_common_dispatch_and_preserves_exact_output(monkeypatch, program, expected):
    from eval.chat_benchmarks.CodeElo import eval_instruct as codeelo

    problem = {
        "id": "example", "title": "Echo", "description": "Echo an integer",
        "input": "One integer", "output": "The same integer", "rating": 800,
        "time_limit_ms": 1000, "interaction": "", "note": "", "examples": [("2", "2-\n")],
    }
    monkeypatch.setattr(codeelo, "load_dataset", lambda *_args, **_kwargs: {"test": [problem]})
    source = codeelo.CodeEloBenchmark()
    options = SourceOptions("CodeElo", SourceRef("Qwen/CodeElo"))

    class Model:
        async def execute(self, request):
            return f"```python\n{program}\n```"

    samples = CollectSamples()
    metrics = SourceMetrics(repeated=True, policy=source.REDUCER)
    grading = Graders(GRADERS)
    engine = Evaluator(Model(), grading, samples, workers=2)

    async def run():
        async for result in engine.run(source.iter_specs(options)):
            assert result.error is None
            metrics.add(result.result)

    try:
        asyncio.run(run())
    finally:
        grading.close()
    assert metrics.finish("CodeElo")["results"]["CodeElo"]["accuracy_avg"] == expected
    assert sorted(record["sample_repeat"] for record in samples.samples["CodeElo"]) == [0, 1, 2]
    assert len(samples.samples["CodeElo"]) == 3


@pytest.mark.parametrize("rating", [800, None, 0])
def test_codeforces_source_normalizes_stdout_newlines(monkeypatch, rating):
    from eval.chat_benchmarks.CodeForces import eval_instruct as codeforces

    problem = {
        "id": "example", "title": "Two lines", "description": "Print two values",
        "input_format": "", "output_format": "", "rating": rating, "time_limit": 1,
        "interaction_format": "", "note": "", "editorial": "", "examples": [],
        "official_tests": [{"input": "", "output": "1 2"}],
    }
    monkeypatch.setattr(codeforces, "load_dataset", lambda *_args, **_kwargs: {"test": [problem]})
    source = codeforces.CodeForcesBenchmark()
    options = SourceOptions("CodeForces", SourceRef("open-r1/codeforces"))

    class Model:
        async def execute(self, request):
            return "```python\nprint(1)\nprint(2)\n```"

    samples = CollectSamples()
    metrics = SourceMetrics(repeated=True, policy=source.REDUCER)
    grading = Graders(GRADERS)
    engine = Evaluator(Model(), grading, samples, workers=2)

    async def run():
        async for result in engine.run(source.iter_specs(options)):
            assert result.error is None
            metrics.add(result.result)

    try:
        asyncio.run(run())
    finally:
        grading.close()
    assert metrics.finish("CodeForces")["results"]["CodeForces"]["accuracy_avg"] == 1.0
