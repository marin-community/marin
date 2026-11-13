# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replies that ignore the task are graded by each task's own grader and reduced per corpus."""

import json

import pytest
from fray.local_backend import LocalClient
from shellbox.machine import UnsupportedMachineSpec
from verifyit.spec import Constraint, ExactSpec, JudgeSpec, Spec, StructuredExactSpec
from zephyr.context import ZephyrContext

from taskcompendium.convert.answers import answer_task, exact_answer_task, ifeval_task, numeric_answer_task
from taskcompendium.evaluation.corpus import EvaluationCorpus, chat_item
from taskcompendium.evaluation.spec import EvaluationProtocol, Metric, MetricKind, Sampling, SummaryKind, TrialPolicy
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, JsonValueAnswer, NoGrader, Source, TaskSpec
from taskcompendium.pipeline import trivial_replies
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from taskcompendium.pipeline.inputs import ConversionContext, SourceFiles, SourceFormat
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, RawRow, SourceRecipe
from taskcompendium.pipeline.trivial_replies import (
    FLOORS_FILENAME,
    GRADES_FILENAME,
    SKIPS_FILENAME,
    GradingPlan,
    SkipReason,
    TrivialReply,
    grade_reply,
    grading_plan,
    main,
    reply_text,
)

from .pipeline_stages import FixtureGradingMachines, script_graded

ALL_REPLIES = tuple(TrivialReply)
LOWERCASE = (Constraint("change_case:english_lowercase", {}),)


def row(index: int, data: dict | None = None) -> RawRow:
    return RawRow(
        f"task-{index}", Source(dataset="fixture", revision="1", row=str(index), importer_revision="1"), data or {}
    )


def checked(task: TaskSpec | ImportRejection) -> TaskSpec:
    assert isinstance(task, TaskSpec)
    return task


def capital_task(index: int) -> TaskSpec:
    return checked(
        exact_answer_task(row(index), prompt="What is the capital of France?", answers=("Paris",), ignore_case=True)
    )


def lowercase_task(index: int) -> TaskSpec:
    return checked(ifeval_task(row(index), prompt="Write a story in all lowercase letters.", constraints=LOWERCASE))


def zero_task(index: int) -> TaskSpec:
    return checked(
        numeric_answer_task(row(index), prompt="What is 5 minus 5?", answer="0", tolerance_abs=0, tolerance_rel=0)
    )


def json_value_task(index: int, verifier: Spec) -> TaskSpec:
    task = answer_task(row(index), prompt="Answer with one JSON value.", spec=verifier)
    return TaskSpec.model_validate({**dict(task), "answer_type": AnswerType.JSON, "answer_format": JsonValueAnswer()})


def scores(task: TaskSpec, machines=None) -> dict[TrivialReply, float | None]:
    plan = grading_plan(task, machines)
    assert isinstance(plan, GradingPlan)
    return {reply: grade_reply(plan, reply_text(task, reply)).reward for reply in ALL_REPLIES}


class UnsupportedMachines:
    """Grading machines that support no environment, as a campaign without the grader's image."""

    def identity(self) -> dict:
        return {"backend": "none"}

    def machine(self, environment, memory_mb):
        raise UnsupportedMachineSpec("No factory runs this environment")


def test_in_process_graders_credit_only_the_replies_they_accept():
    # One lowercase word satisfies a constraint that only restricts the reply; 0 matches a numeric answer of zero.
    assert scores(capital_task(0)) == dict.fromkeys(ALL_REPLIES, 0.0)
    assert scores(lowercase_task(1)) == {**dict.fromkeys(ALL_REPLIES, 0.0), TrivialReply.WORD: 1.0}
    assert scores(zero_task(2)) == {**dict.fromkeys(ALL_REPLIES, 0.0), TrivialReply.ZERO: 1.0}


def test_machine_graders_grade_replies_in_the_grading_machine():
    # The grader passes any nonempty answer file; the empty reply is rejected before the grader runs.
    task = script_graded(capital_task(0), b"test -s answer.txt\n")
    assert scores(task, FixtureGradingMachines()) == {**dict.fromkeys(ALL_REPLIES, 1.0), TrivialReply.EMPTY: 0.0}


def test_json_value_tasks_receive_each_reply_as_a_json_value():
    # The format parses the whole reply as JSON: 0 is already a JSON value, and every other reply becomes a string.
    zero = json_value_task(0, StructuredExactSpec(expected=0))
    plan = grading_plan(zero, None)
    assert isinstance(plan, GradingPlan)
    grades = {reply: grade_reply(plan, reply_text(zero, reply)) for reply in ALL_REPLIES}
    assert {reply: (grade.status, grade.reward) for reply, grade in grades.items()} == {
        **dict.fromkeys(ALL_REPLIES, (Outcome.GRADED, 0.0)),
        TrivialReply.ZERO: (Outcome.GRADED, 1.0),
    }
    abstention = json_value_task(1, ExactSpec(expected=("I don't know.",)))
    assert scores(abstention) == {**dict.fromkeys(ALL_REPLIES, 0.0), TrivialReply.ABSTAIN: 1.0}


def test_tasks_this_audit_cannot_grade_are_skipped_with_their_reason():
    machine_graded = script_graded(capital_task(0), b'test "$(cat answer.txt)" = Paris\n')
    file_answer = capital_task(1).model_copy(update={"answer_type": AnswerType.FILE, "output_paths": ("/app/a.txt",)})
    cases = {
        SkipReason.NO_GRADER: (capital_task(2).model_copy(update={"grader": NoGrader(reason="fixture")}), None),
        SkipReason.JUDGE: (
            capital_task(3).model_copy(update={"grader": verifyit_package(JudgeSpec(references=("Paris",))).grader}),
            None,
        ),
        SkipReason.WORKSPACE_ANSWER: (script_graded(file_answer, b"true\n", answer_path=None), FixtureGradingMachines()),
        SkipReason.NO_MACHINES: (machine_graded, None),
        SkipReason.UNSUPPORTED_MACHINE: (machine_graded, UnsupportedMachines()),
    }
    assert {reason: grading_plan(task, machines) for reason, (task, machines) in cases.items()} == {
        reason: reason for reason in cases
    }


def convert_lowercase(source_row: RawRow, _context: ConversionContext) -> TaskSpec:
    task = checked(ifeval_task(source_row, prompt=source_row.data["prompt"], constraints=LOWERCASE))
    return task.model_copy(update={"tags": ("eval",)})


def publish_lowercase_corpus(tmp_path, prompts: list[str]) -> str:
    source = tmp_path / "source"
    source.mkdir()
    (source / "rows.jsonl").write_text("".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts))
    recipe = SourceRecipe(
        name="lowercase",
        version="1",
        source=SourceFiles("lowercase", "pinned", ("rows.jsonl",), SourceFormat.JSONL),
        convert=convert_lowercase,
        rubric=None,
        controls=None,
        intended_use=IntendedUse.EVAL,
    )
    protocol = EvaluationProtocol(
        benchmark="lowercase",
        version="1",
        operation="chat_generation",
        sampling=Sampling(temperature=0, top_p=1, top_k=None, max_tokens=20, stop=(), chat_template_kwargs={}),
        trials=TrialPolicy(count=1, seed=None),
        metrics=(Metric(name="accuracy", kind=MetricKind.BINARY, higher_is_better=True),),
        primary_metric="accuracy",
        summary=SummaryKind.MEAN,
        expected_population=len(prompts),
    )
    corpus = tmp_path / "corpus"
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            publish_evaluation_corpus(recipe, protocol, chat_item, context, str(source), str(corpus))
    finally:
        client.shutdown()
    return str(corpus)


def test_cli_writes_grades_skips_and_the_floor_of_each_corpus(tmp_path):
    corpus = publish_lowercase_corpus(tmp_path, ["Write a poem in lowercase.", "Describe the sea in lowercase."])
    report = tmp_path / "report"
    main([corpus, "--output", str(report), "--grading-machines", "none", "--workers", "2"])

    grades = [json.loads(line) for line in (report / GRADES_FILENAME).read_text().splitlines()]
    assert len(grades) == 2 * len(ALL_REPLIES)
    assert {(grade["source_row"], grade["reply"]) for grade in grades if grade["score"]} == {
        ("rows.jsonl:0", "word"),
        ("rows.jsonl:1", "word"),
    }
    assert (report / SKIPS_FILENAME).read_text() == ""
    (floor,) = json.loads((report / FLOORS_FILENAME).read_text())
    assert (floor["benchmark"], floor["corpus"], floor["graded_tasks"], floor["skipped"]) == ("lowercase", corpus, 2, {})
    assert floor["dataset"] == grades[0]["dataset"]
    assert {item["reply"]: (item["mean_score"], item["credited"], item["no_verdict"]) for item in floor["replies"]} == {
        "empty": (0.0, 0, 0),
        "word": (1.0, 2, 0),
        "zero": (0.0, 0, 0),
        "abstain": (0.0, 0, 0),
        "echo": (0.0, 0, 0),
    }


def test_cli_writes_each_batch_of_grades_before_reading_later_tasks(tmp_path, monkeypatch):
    # Long-context corpora need not fit in memory; batching leaves every output file unchanged.
    corpus = publish_lowercase_corpus(tmp_path, [f"Write poem {index} in lowercase." for index in range(3)])
    single = tmp_path / "single"
    main([corpus, "--output", str(single), "--grading-machines", "none"])

    batched = tmp_path / "batched"
    written_before_each_task = []
    read_cases = EvaluationCorpus.cases

    def cases(self, **kwargs):
        for case in read_cases(self, **kwargs):
            written_before_each_task.append(len((batched / GRADES_FILENAME).read_text().splitlines()))
            yield case

    monkeypatch.setattr(EvaluationCorpus, "cases", cases)
    monkeypatch.setattr(trivial_replies, "BATCH_TASKS", 1, raising=False)
    main([corpus, "--output", str(batched), "--grading-machines", "none", "--workers", "2"])

    assert written_before_each_task == [0, len(ALL_REPLIES), 2 * len(ALL_REPLIES)]
    for name in (GRADES_FILENAME, SKIPS_FILENAME, FLOORS_FILENAME):
        assert (batched / name).read_text() == (single / name).read_text()


def test_cli_rejects_the_same_corpus_snapshot_twice(tmp_path):
    # Grades of one snapshot listed twice would be counted twice in its floor.
    corpus = publish_lowercase_corpus(tmp_path, ["Write a poem in lowercase."])
    with pytest.raises(ValueError, match="same corpus snapshot twice"):
        main([corpus, corpus, "--output", str(tmp_path / "report"), "--grading-machines", "none"])
