"""Offline source-to-judge fixtures for judge-backed benchmarks."""

import pytest

from eval.chat_benchmarks.MixEval.eval_instruct import MixEvalBenchmark
from eval.chat_benchmarks.MTBench.eval_instruct import MTBenchBenchmark
from eval.chat_benchmarks.OlympiadBench.eval_instruct import OlympiadBenchBenchmark
from eval.chat_benchmarks.SimpleQA.eval_instruct import SimpleQABenchmark
from eval.chat_benchmarks.WildBench import eval_instruct as wildbench_source
from eval.completion_response import FailedGeneration
from eval.contracts.eval_spec import EvalSpec, Grade, GraderSpec, ReducerSpec, SourceRef
from eval.execution.engine import ItemResult
from eval.execution.metrics import SourceMetrics
from eval.graders.conversations import turn_ratings
from eval.sources.common import SourceOptions
from tests.contracts.source_harness import FixedEndpoint, evaluate_source


def test_mixeval_preserves_both_split_identities_and_judge_parsers():
    source = MixEvalBenchmark(debug=True)
    candidate = FixedEndpoint("candidate")
    judge = FixedEndpoint(
        lambda request: "[[1.0]]" if "correctness" in request.prompt[-1]["content"].lower() else "[[C]]"
    )

    metrics, samples = evaluate_source(
        source,
        SourceOptions("MixEval", SourceRef("fixture"), limit=1),
        candidate,
        judges={"MixEval": judge},
    )

    assert metrics["overall"] == 1.0
    assert {sample["sample_namespace"] for sample in samples} == {"freeform", "multichoice"}
    assert len(judge.requests) == 2


def test_mixeval_pinned_jsonl_normalizes_docs_and_keeps_dates_distinct():
    june = MixEvalBenchmark(debug=True)
    specs = list(june.iter_specs(SourceOptions("MixEval", SourceRef("fixture"))))
    freeform = specs[0]
    multichoice = next(spec for spec in specs if spec.namespace == "multichoice")

    assert freeform.doc["context"] is None
    assert "options" not in freeform.doc
    assert freeform.source_id == multichoice.source_id == "0"
    assert freeform.target == ["JAMES ELROY"]
    assert multichoice.target == [2]
    assert freeform.requests[0].prompt == [{"role": "user", "content": "Question: Who wrote the book 'LA Confidential'?\nAnswer the question shortly."}]
    assert multichoice.grader.reference == {"kind": "multichoice"}
    assert freeform.source.dataset == "MixEval/MixEval"
    assert freeform.source.revision == "66a05ec3821a2b5089ed1e32a5d65ab6ecdbf906"

    august = next(MixEvalBenchmark(version="2024-08-11", debug=True).iter_specs(SourceOptions("MixEval", june.SOURCE)))
    assert august.source.revision == "2024-08-11"
    assert august.doc != freeform.doc


def test_wildbench_preserves_published_truncation_and_weighted_tags(monkeypatch):
    row = {
        "session_id": "session-1",
        "conversation_input": [{"role": "user", "content": "prior"}, {"role": "assistant", "content": "reply"}, {"role": "user", "content": "question"}],
        "checklist": ["be right"],
        "primary_tag": "Math",
        "secondary_tags": [],
    }
    monkeypatch.setattr(wildbench_source, "load_dataset", lambda *_args, **_kwargs: [row])
    source = wildbench_source.WildBenchBenchmark(config=wildbench_source.WildBenchConfig(max_words_to_eval=1))
    candidate = FixedEndpoint("one two")
    judge = FixedEndpoint('{"score": "7"}')

    metrics, samples = evaluate_source(
        source,
        SourceOptions("WildBench", SourceRef("fixture")),
        candidate,
        judges={"WildBench": judge},
    )

    assert metrics["score"] == 7.0
    assert metrics["avg_output_length"] == 0.0
    assert metrics["task_categorized_scores"] == {"Math & Data Analysis": 4.0}
    assert "one... (truncated)" in judge.requests[0].prompt
    assert judge.requests[0].generation["max_gen_toks"] == 4096
    assert samples[0]["adjusted_score"] == 4.0


def _scored_item(task, source_id, grade, tags):
    spec = EvalSpec(
        task=task, source=SourceRef("fixture"), source_id=source_id, ordinal=0, requests=(),
        grader=GraderSpec("fixture"), tags=tags,
    )
    return ItemResult(spec, (), grade, {"sample_id": str(source_id)})


def test_judge_coverage_preserves_a_renamed_accuracy_metric():
    metrics = SourceMetrics(repeated=False, policy=ReducerSpec(accuracy="accuracy_avg"))
    metrics.add(_scored_item("fixture", "judged", Grade({"accuracy": 1.0, "judge_failed": 0.0}), {}))
    metrics.add(_scored_item("fixture", "unavailable", Grade({"judge_failed": 1.0}), {}))

    reduced = metrics.finish("fixture")["results"]["fixture"]

    assert reduced["accuracy_avg"] == 1.0
    assert reduced["num_judged"] == 1
    assert reduced["num_judge_failed"] == 1
    assert reduced["judge_coverage"] == 0.5


def test_mixeval_reducer_preserves_split_rounding_and_public_keys():
    metrics = SourceMetrics(repeated=False, policy=ReducerSpec(name="rounded_split_mean"))
    for source_id, split, score in [("freeform-1", "freeform", 0.3333), ("freeform-2", "freeform", 1.0), ("multi", "multichoice", 0.0)]:
        metrics.add(
            _scored_item(
                "MixEval", source_id, Grade({"accuracy": score}),
                {"split": split, "benchmark_split": f"BBH|{split}"},
            )
        )

    assert metrics.finish("MixEval")["results"]["MixEval"] == {
        "overall": pytest.approx(0.44466666666666665),
        "BBH": pytest.approx(0.44466666666666665),
    }


def test_mtbench_excludes_invalid_turns_and_weights_partial_averages():
    metrics = SourceMetrics(repeated=False, policy=ReducerSpec(trial_metric=None))
    first = turn_ratings(None, ["[[10]]", "[[-1]]"])
    second = turn_ratings(None, ["[[2]]", "[[2]]"])
    metrics.add(_scored_item("MTBench", "first", first, {}))
    metrics.add(_scored_item("MTBench", "second", second, {}))

    reduced = metrics.finish("MTBench")["results"]["MTBench"]
    assert reduced["Turn 1"] == 6.0
    assert reduced["Turn 2"] == 2.0
    assert reduced["Average"] == pytest.approx(14 / 3)


def test_mtbench_parser_preserves_fastchat_rating_selection():
    grade = turn_ratings(None, ["[[-1]] then [[2]]", "[11]"])
    assert grade.metrics == {"Turn 1": 2.0, "Turn 2": 11.0, "Average": 6.5, "judge_failed": 0.0}


@pytest.mark.parametrize(
    "ratings, expected, failures",
    [(["[[-1]]", "invalid", "[[2]]", "[[2]]"], 2.0, 1), (["invalid"] * 4, None, 2)],
)
def test_mtbench_runtime_preserves_unavailable_ratings(ratings, expected, failures):
    source = MTBenchBenchmark(debug=True)
    saved_ratings = iter(ratings)

    metrics, samples = evaluate_source(
        source, SourceOptions("MTBench", source.SOURCE), FixedEndpoint("candidate"),
        judges={"MTBench": FixedEndpoint(lambda request: next(saved_ratings))}, workers=1,
    )

    assert metrics["Average"] == expected
    assert "accuracy" not in metrics
    assert metrics["num_judge_failed"] == failures
    assert metrics["num_judged"] == 2 - failures
    assert len(samples) == 2
    assert samples[0]["judge_failed"] == 1.0
    assert "Average" not in samples[0]["metrics"]


@pytest.mark.parametrize(
    "repeats,minerva_solved,judge_available,judge_label",
    [
        (10, 0, 4, "incorrect"), (10, 0, 4, "correct"), (10, 0, 0, "correct"),
        (10, 0, 10, "incorrect"), (10, 2, 2, "correct"), (1, 0, 0, "correct"),
    ],
)
def test_olympiadbench_coverage_counts_judge_observations_and_keeps_fallback_scores(
    tmp_path, repeats, minerva_solved, judge_available, judge_label,
):
    data = tmp_path / "olympiad.jsonl"
    data.write_text('{"id": "sum", "problem": "Compute 2+2.", "answer": ["4"]}\n')
    source = OlympiadBenchBenchmark(data_file=str(data), n_repeat=repeats)
    candidate = FixedEndpoint(
        lambda request: r"\boxed{4}" if (request.metadata["trial"] or 0) < minerva_solved else r"\boxed{3}"
    )
    judge = FixedEndpoint(
        lambda _request: judge_label if len(judge.requests) <= judge_available else FailedGeneration("model_transport")
    )

    metrics, samples = evaluate_source(
        source, SourceOptions("OlympiadBench", SourceRef(str(data))), candidate,
        judges={"OlympiadBench": judge},
    )

    solved = minerva_solved + (judge_available if judge_label == "correct" else 0)
    accuracy = solved / repeats
    accuracy_name = "accuracy_avg" if repeats > 1 else "accuracy"
    assert metrics[accuracy_name] == pytest.approx(accuracy)
    assert metrics["num_total"] == 1
    assert metrics["num_judged"] == judge_available
    assert metrics["num_judge_failed"] == repeats - minerva_solved - judge_available
    assert metrics["judge_coverage"] == pytest.approx(judge_available / (repeats - minerva_solved))
    assert len(samples) == repeats
    assert sum(sample["accuracy"] for sample in samples) == solved
    failed_samples = [sample for sample in samples if sample.get("judge_failed")]
    assert len(failed_samples) == repeats - minerva_solved - judge_available
    assert all(sample["accuracy"] == 0.0 for sample in failed_samples)
    assert all(sample["judge_error"]["category"] == "grader_infrastructure" for sample in failed_samples)
    if repeats > 1:
        assert "accuracy" not in metrics
        assert metrics["num_repeat"] == repeats
        assert metrics["solved_avg"] == pytest.approx(accuracy)
        assert [run["accuracy"] for run in metrics["run_stats"]] == [sample["accuracy"] for sample in samples]
        assert metrics["accuracy_std_err"] == pytest.approx((accuracy * (1 - accuracy) / repeats) ** 0.5)


def test_simpleqa_repeated_coverage_uses_judge_observations(tmp_path):
    data = tmp_path / "simpleqa.csv"
    data.write_text("problem,answer,metadata\nCapital of France?,Paris,{}\nCapital of Spain?,Madrid,{}\n")
    judge = FixedEndpoint(
        lambda _request: "A" if len(judge.requests) % 2 else FailedGeneration("model_transport")
    )

    metrics, _samples = evaluate_source(
        SimpleQABenchmark(data_file=str(data)),
        SourceOptions("SimpleQA", SourceRef(str(data)), num_samples=3),
        FixedEndpoint("candidate"), judges={"SimpleQA": judge},
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["num_total"] == 2
    assert metrics["num_judged"] == 3
    assert metrics["num_judge_failed"] == 3
    assert metrics["judge_coverage"] == pytest.approx(0.5)
