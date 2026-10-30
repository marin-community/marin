"""Parity at the source/runtime boundary for judge calls and instruction weights."""

import asyncio
import json

import pytest

from eval.completion_response import FailedGeneration
from eval.chat_benchmarks.IFEval.eval_instruct import IFEvalBenchmark
from eval.chat_benchmarks.SimpleQA.eval_instruct import SimpleQABenchmark
from eval.chat_benchmarks.GSM8KPerturbed.eval_instruct import GSM8KPerturbedBenchmark
from eval.contracts.eval_spec import SourceRef
from eval.execution.engine import ItemResult
from eval.execution.metrics import SourceMetrics
from eval.graders.structured import gsm8k_flexible
from eval.sources.common import SourceOptions
from tests.contracts.source_harness import FixedEndpoint, MemoryCheckpoints, evaluate_source


def test_simpleqa_restores_candidate_and_judge_operations_without_repaying():
    source = SimpleQABenchmark(debug=True)
    options = SourceOptions("SimpleQA", source.SOURCE, limit=1)
    candidate = FixedEndpoint("An answer")
    judge = FixedEndpoint("A")
    checkpoints = MemoryCheckpoints()
    metrics, samples = evaluate_source(source, options, candidate, judges={"SimpleQA": judge},
                                       checkpoints={"SimpleQA": checkpoints})
    assert metrics["accuracy"] == 1.0
    assert samples[0]["judge_label"] == "correct"
    assert "An answer" in judge.requests[0].prompt
    candidate_again, judge_again = FixedEndpoint("wrong"), FixedEndpoint("B")
    resumed_metrics, resumed_samples = evaluate_source(
        source, options, candidate_again, judges={"SimpleQA": judge_again}, checkpoints={"SimpleQA": checkpoints},
    )
    assert resumed_metrics == metrics
    assert resumed_samples == samples
    assert candidate_again.requests == judge_again.requests == []


def test_failed_judge_retries_on_resume_without_regenerating_candidate():
    source = SimpleQABenchmark(debug=True)
    options = SourceOptions("SimpleQA", source.SOURCE, limit=1)
    store = MemoryCheckpoints()
    first, _ = evaluate_source(
        source, options, FixedEndpoint("candidate"),
        judges={"SimpleQA": FixedEndpoint(FailedGeneration("model_transport"))}, checkpoints={"SimpleQA": store},
    )
    assert first["accuracy"] is None
    candidate, judge = FixedEndpoint("new candidate"), FixedEndpoint("B")
    resumed, _ = evaluate_source(source, options, candidate, judges={"SimpleQA": judge}, checkpoints={"SimpleQA": store})
    assert resumed["accuracy"] == 0.0
    assert candidate.requests == []
    assert len(judge.requests) == 1


def test_simpleqa_excludes_unavailable_judges_from_classification_metrics(tmp_path):
    data_file = tmp_path / "simpleqa.csv"
    data_file.write_text("problem,answer,metadata\nCapital of France?,Paris,{}\nCapital of Spain?,Madrid,{}\n")
    judge = FixedEndpoint(lambda request: "A" if len(judge.requests) == 1 else FailedGeneration("model_transport"))

    metrics, _samples = evaluate_source(
        SimpleQABenchmark(data_file=str(data_file)),
        SourceOptions("SimpleQA", SourceRef(str(data_file))),
        FixedEndpoint("candidate"),
        judges={"SimpleQA": judge},
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["num_judged"] == 1
    assert metrics["num_judge_failed"] == 1
    assert metrics["judge_coverage"] == pytest.approx(0.5)


def test_instruction_reducer_weights_by_constraints_instead_of_averaging_prompt_rates(tmp_path):
    records = []
    for key, keywords in enumerate((["apple"], ["apple", "banana", "orange"])):
        records.append({"key": key, "prompt": "Say " + ", ".join(keywords),
                        "instruction_id_list": ["keywords:existence"] * len(keywords),
                        "kwargs": [{"keywords": [word]} for word in keywords]})
    (tmp_path / "input_data.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    source = IFEvalBenchmark(data_dir=str(tmp_path), start_idx=0, end_idx=2)
    metrics, samples = evaluate_source(source, SourceOptions("IFEval", SourceRef("fixture")), FixedEndpoint("apple"))
    assert metrics["prompt_level_strict"] == 0.5
    assert metrics["instruction_level_strict"] == 0.5
    assert metrics["instruction-level"] == 0.5
    assert sorted(sample["instruction_level_strict"] for sample in samples) == [1 / 3, 1.0]


def test_explicit_trials_share_binary_pass_at_k_reduction():
    from eval.contracts.eval_spec import ReducerSpec
    from eval.sources.common import text_specs

    class Source:
        REDUCER = ReducerSpec()

        def iter_specs(self, options):
            return text_specs(
                [{"id": "a"}, {"id": "b"}], options,
                prompt=lambda row: row["id"], target=lambda row: "yes", grader="exact_match",
                generation_kwargs={"do_sample": True},
            )

    class Backend:
        async def execute(self, request):
            return "yes" if request.generation["seed"] < 2 else "no"

    from eval.execution.engine import CollectSamples, Evaluator
    from eval.execution.grading import Graders
    from eval.execution.metrics import SourceMetrics
    from eval.graders.specs import GRADERS

    source = Source()
    options = SourceOptions("trials", SourceRef("fixture"), limit=1, num_samples=4, seed=0)
    metrics = SourceMetrics(repeated=False, pass_at_k=(1, 2))
    sink = CollectSamples()
    grading = Graders(GRADERS)
    async def run():
        engine = Evaluator(Backend(), grading, sink, workers=3)
        async for result in engine.run(source.iter_specs(options)):
            assert result.error is None
            metrics.add(result.result)
    try:
        asyncio.run(run())
    finally:
        grading.close()
    assert metrics.finish("trials")["results"]["trials"] == {"pass@1": 0.5, "pass@2": pytest.approx(5 / 6)}
    assert len(sink.samples["trials"]) == 4
    assert {record["source_id"] for record in sink.samples["trials"]} == {"a"}


@pytest.mark.parametrize("reverse_clean", [False, True])
def test_gsm_perturbation_pairs_matching_trials_and_keeps_pass_at_k(reverse_clean):
    source = GSM8KPerturbedBenchmark()
    items = list(source.iter_specs(SourceOptions("GSM8KPerturbed", source.SOURCE, limit=1, num_samples=2)))
    completed = []
    for item in items:
        correct = (item.namespace == "gsm8k-clean") == (item.trial == 0)
        grade = gsm8k_flexible(item, [str(item.target) if correct else "no answer"])
        completed.append(ItemResult(item, (), grade, {"sample_id": item.namespace + str(item.source_id)}))
    if reverse_clean:
        completed[:2] = reversed(completed[:2])
    metrics = SourceMetrics(repeated=True, pass_at_k=(1, 2), policy=source.REDUCER)
    for item in completed:
        metrics.add(item)

    reduced = metrics.finish("GSM8KPerturbed")["results"]["GSM8KPerturbed"]

    assert reduced["gsm8k-clean"] == reduced["gsm8k-perturbed"] == 50.0
    assert reduced["gsm8k-perturbed:typos_clean_subset"] == 50.0
    assert reduced["gsm8k-perturbed:typos_paired_delta"] == 0.0
    assert reduced["gsm8k-perturbed:typos_harmed"] == reduced["gsm8k-perturbed:typos_helped"] == 1
    assert reduced["pass@1"] == 0.5
    assert reduced["pass@2"] == 1.0
