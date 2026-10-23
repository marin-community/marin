"""LAB-Bench prompts, selective accuracy and deterministic parsing."""

from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from eval.chat_benchmarks.LABBench import eval_instruct as lab
from eval.completion_response import CompletionContentPolicy, CompletionText, completion_response_from_chat_choice
from eval.execution.engine import ItemResult
from eval.execution.metrics import SourceMetrics
from eval.graders.labbench import labbench
from eval.limits import ContextWindowExceededError
from eval.sources.common import SourceOptions
from eval.task import TaskManager


@pytest.fixture
def source_data(tmp_path, monkeypatch):
    def fetch(repo_id, filename, *, revision, cache_prefix):
        return str(tmp_path / filename)

    monkeypatch.setattr(lab, "fetch_hf", fetch)
    for component in lab.COMPONENTS.values():
        rows = [
            {
                "id": f"item-{index}", "question": "Which option?", "protocol": "Protocol context.\n",
                "ideal": "correct answer", "distractors": ["wrong one", "wrong two", "wrong three"],
                "subtask": "first" if index < 2 else "second",
                "canary": "hidden-canary", "source": "hidden-source",
            }
            for index in range(component.count)
        ]
        path = tmp_path / component.subset / "train-00000-of-00001.parquet"
        path.parent.mkdir()
        pq.write_table(pa.Table.from_pylist(rows), path)


def source_specs(task="LABBenchSeqQA", limit=4):
    source = TaskManager([task]).get_benchmark(task)
    options = SourceOptions(
        task, source.SOURCE, limit=limit, seed=42, context_length=32768, max_tokens=8192,
        prompt_token_count=lambda prompt: len(str(prompt).split()),
    )
    return source, options, list(source.iter_specs(options))


@pytest.mark.parametrize("task", list(lab.COMPONENTS))
def test_component_prompts_preserve_inputs_and_shuffle_identity(source_data, task):
    source, options, specs = source_specs(task)
    capped = list(source.iter_specs(replace(options, limit=1)))
    assert capped[0] == specs[0]
    assert len({spec.source_id for spec in specs}) == 4
    for spec in specs:
        prompt = spec.requests[0].prompt[0]["content"]
        assert f"({spec.target}) correct answer" in prompt
        assert f"({spec.grader.reference['abstention']}) {lab.ABSTENTION}" in prompt
        assert ("Protocol context." in prompt) == (task == "LABBenchProtocolQA")
        assert "hidden-canary" not in prompt and "hidden-source" not in prompt
        assert "wrong one" in prompt and "wrong two" in prompt and "wrong three" in prompt
    changed = list(source.iter_specs(replace(options, seed=43)))
    assert [s.requests for s in changed] != [s.requests for s in specs]


def test_selective_accuracy_and_subtask_denominators(source_data):
    source, options, specs = source_specs()
    wrong = next(letter for letter in specs[3].grader.reference["letters"] if letter not in (
        specs[3].target, specs[3].grader.reference["abstention"],
    ))
    responses = [
        f"[ANSWER]{specs[0].target}[/ANSWER]",
        f"[ANSWER]{specs[1].grader.reference['abstention']}[/ANSWER]",
        "I choose an answer without a tag",
        f"[ANSWER]{wrong}[/ANSWER]",
    ]
    reducer = SourceMetrics(repeated=False, policy=source.REDUCER)
    for spec, response in zip(specs, responses, strict=True):
        grade = labbench(spec, [response])
        reducer.add(ItemResult(spec, (response,), grade, {"sample_id": spec.source_id}))
    metrics = reducer.finish(options.task)["results"][options.task]
    # Independent reference: one correct, two incorrect (including malformed), one abstained.
    assert metrics["accuracy"] == 1 / 4
    assert metrics["precision"] == 1 / 3
    assert metrics["attempted_answer_coverage"] == 3 / 4
    assert metrics["num_parse_failures"] == 1
    assert metrics["num_abstained"] == 1
    assert metrics["num_attempted_answers"] == 3
    assert metrics["accuracy_first"] == 1 / 2
    assert metrics["subtask:first/precision"] == 1
    assert metrics["accuracy_second"] == 0
    assert metrics["count_first"] == metrics["count_second"] == 2


def test_all_abstentions_have_zero_precision_and_no_parse_failures(source_data):
    source, options, specs = source_specs()
    reducer = SourceMetrics(repeated=False, policy=source.REDUCER)
    for spec in specs:
        response = f"[ANSWER]{spec.grader.reference['abstention']}[/ANSWER]"
        reducer.add(ItemResult(spec, (response,), labbench(spec, [response]), {"sample_id": spec.source_id}))
    metrics = reducer.finish(options.task)["results"][options.task]
    assert metrics["accuracy"] == metrics["precision"] == metrics["attempted_answer_coverage"] == 0
    assert metrics["num_abstained"] == 4
    assert metrics["num_parse_failures"] == 0


@pytest.mark.parametrize("response", [
    "", "A", "[ANSWER]Z[/ANSWER]", "[ANSWER]A or B[/ANSWER]",
    "[ANSWER]A[/ANSWER] [ANSWER]B[/ANSWER]", "[ANSWER]a[/ANSWER]",
    "[ANSWER]A[/ANSWER] [ANSWER]", "[ANSWER]<answer>A[/ANSWER]",
])
def test_malformed_responses_are_incorrect_attempts(source_data, response):
    _, _, specs = source_specs(limit=1)
    grade = labbench(specs[0], [response])
    assert grade.metrics["accuracy"] == 0
    assert grade.metrics["parse_failure_rate"] == grade.metrics["attempted_answer_coverage"] == 1
    assert grade.metrics["abstention_rate"] == 0


def test_structured_reasoning_does_not_supply_the_final_answer(source_data):
    _, _, specs = source_specs(limit=1)
    spec = specs[0]
    choice = {"message": {"content": "", "reasoning_content": f"[ANSWER]{spec.target}[/ANSWER]"}, "finish_reason": "length"}
    completion = completion_response_from_chat_choice({}, choice)
    response = CompletionText(completion.normalized_content(), completion, CompletionContentPolicy.COMBINE)
    assert labbench(spec, [response]).metrics["parse_failure_rate"] == 1
    choice["message"]["content"] = f"[ANSWER] {spec.target} [/ANSWER]"
    completion = completion_response_from_chat_choice({}, choice)
    response = CompletionText(completion.normalized_content(), completion, CompletionContentPolicy.COMBINE)
    assert labbench(spec, [response]).metrics["accuracy"] == 1


def test_context_overflow_fails_before_yielding_requests(source_data):
    source, options, _ = source_specs()
    too_small = replace(options, context_length=8192)
    with pytest.raises(ContextWindowExceededError):
        next(source.iter_specs(too_small))
