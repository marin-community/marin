# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
from taskcompendium.environment import EnvironmentKind, ExitCodeReward, RewardFileFormat
from taskcompendium.execution import StageExecution, TaskExecution
from taskcompendium.grading import verifier_descriptor
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, Source, StageRewardStrategy, TaskSpec
from verifyit.spec import ExactSpec

from taskforge.spec.controls import (
    Control,
    ControlCategory,
    ControlKind,
    Expectation,
    Transcript,
    Workspace,
    controls_json,
    parse_controls,
    reply,
    shell_turn,
    validate_controls,
)
from taskforge.spec.draft import assemble, environment, file, reward_file, shell_verifier, stage, staged

SOURCE = Source(dataset="taskforge-test", revision="r1", row="0", importer_revision="test")
NO_EXECUTION = TaskExecution()
GRADED_ONE = Expectation(Outcome.GRADED, reward_min=1.0)
GRADED_ZERO = Expectation(Outcome.GRADED, reward_max=0.0)


def file_check(path: str = "/workspace/answer"):
    return shell_verifier(("sh", "-c", f'[ "$(cat {path})" = 12 ]'), ExitCodeReward(), timeout=5)


def file_task() -> TaskSpec:
    return assemble(
        "file",
        "Write 12 to /workspace/answer.",
        AnswerType.FILE,
        environment(EnvironmentKind.SHELLSIM),
        file_check(),
        SOURCE,
        execution=NO_EXECUTION,
    )


def rubric_task() -> TaskSpec:
    """A file task whose grader writes per-criterion components to a JSON reward file."""
    script = (
        "format=0; value=0; [ -f /workspace/answer ] && format=1; "
        '[ "$(cat /workspace/answer)" = 12 ] && value=1; mkdir -p /logs; '
        'echo "{\\"reward\\": $(( (format + value) * 50 ))e-2, \\"format\\": $format, \\"value\\": $value}"'
        " > /logs/reward.json"
    )
    return assemble(
        "rubric",
        "Write 12 to /workspace/answer.",
        AnswerType.FILE,
        environment(EnvironmentKind.SHELLSIM),
        shell_verifier(("sh", "-c", script), reward_file("/logs/reward.json", RewardFileFormat.JSON), timeout=5),
        SOURCE,
        execution=NO_EXECUTION,
    )


def answer_task() -> TaskSpec:
    return assemble(
        "answer",
        "What is six plus six?",
        AnswerType.TEXT,
        environment(EnvironmentKind.NULL),
        verifier_descriptor(ExactSpec(expected=("12",))),
        SOURCE,
        execution=NO_EXECUTION,
    )


def control(
    identifier: str, kind: ControlKind, category: ControlCategory, payload, expect: Expectation, **fields
) -> Control:
    return Control(identifier, kind, category, "builder", payload, expect, **fields)


def file_controls(stage_index: int = 0) -> list[Control]:
    def write(value: str) -> Transcript:
        return Transcript((shell_turn((f"s{stage_index}", f"echo {value} > /workspace/answer")), reply("Done.")))

    prefix = f"s{stage_index}-"
    return [
        control(
            prefix + "gold",
            ControlKind.POSITIVE,
            ControlCategory.KNOWN_CORRECT,
            write("12"),
            GRADED_ONE,
            stage=stage_index,
        ),
        control(
            prefix + "empty",
            ControlKind.MALFORMED,
            ControlCategory.EMPTY_OR_MALFORMED,
            Workspace(()),
            GRADED_ZERO,
            stage=stage_index,
        ),
        control(
            prefix + "off-by-one",
            ControlKind.NEGATIVE,
            ControlCategory.PLAUSIBLE_WRONG,
            write("13"),
            GRADED_ZERO,
            stage=stage_index,
        ),
        control(
            prefix + "fake-file",
            ControlKind.NEGATIVE,
            ControlCategory.REWARD_HACK,
            Workspace((file("/workspace/answer", "12 \n"),)),
            GRADED_ZERO,
            stage=stage_index,
        ),
    ]


HALF = control(
    "half",
    ControlKind.PARTIAL,
    ControlCategory.CRITERION_MUTATION,
    Workspace((file("/workspace/answer", "1"),)),
    Expectation(Outcome.GRADED, reward_min=0.4, reward_max=0.6, components={"format": 1.0, "value": 0.0}),
    partial_credit_reason="Right file, wrong value: the rubric awards the format criterion only.",
)


def test_complete_control_set_validates_and_round_trips_through_json():
    controls = [*file_controls(), HALF]
    validate_controls(rubric_task(), controls)

    restored = parse_controls(controls_json(controls))

    assert restored == tuple(controls)
    validate_controls(rubric_task(), restored)


def test_partial_control_needs_a_grader_that_reports_components():
    with pytest.raises(ValueError, match="only a JSON reward file reports"):
        validate_controls(file_task(), [*file_controls(), HALF])


def test_answer_task_controls_with_extraction_error_validate():
    controls = [
        control("gold", ControlKind.POSITIVE, ControlCategory.KNOWN_CORRECT, Transcript((reply("12"),)), GRADED_ONE),
        control(
            "empty",
            ControlKind.MALFORMED,
            ControlCategory.EMPTY_OR_MALFORMED,
            Transcript((reply(""),)),
            Expectation(Outcome.EXTRACTION_ERROR),
        ),
        control("wrong", ControlKind.NEGATIVE, ControlCategory.PLAUSIBLE_WRONG, Transcript((reply("13"),)), GRADED_ZERO),
        control(
            "unit",
            ControlKind.NEGATIVE,
            ControlCategory.TASK_SPECIFIC_SHORTCUT,
            Transcript((reply("12 apples"),)),
            GRADED_ZERO,
        ),
    ]
    validate_controls(answer_task(), controls)


@pytest.mark.parametrize(
    "dropped,message",
    [
        ("s0-gold", "known_correct"),
        ("s0-fake-file", "task_specific_shortcut or reward_hack"),
        ("s0-empty", "empty_or_malformed"),
    ],
)
def test_each_required_category_is_enforced(dropped, message):
    controls = [item for item in file_controls() if item.id != dropped]
    with pytest.raises(ValueError, match=message):
        validate_controls(file_task(), controls)


def test_every_stage_needs_its_own_controls():
    task = assemble(
        "staged",
        "Write 12 to /workspace/answer.",
        AnswerType.FILE,
        environment(EnvironmentKind.SHELLSIM),
        staged(StageRewardStrategy.FINAL),
        SOURCE,
        execution=TaskExecution(stages={"one": StageExecution(), "two": StageExecution()}),
        stages=(stage("one", file_check()), stage("two", file_check("/workspace/b"), instruction="Again.")),
    )
    with pytest.raises(ValueError, match="Stage 1 lacks"):
        validate_controls(task, file_controls(0))
    validate_controls(task, [*file_controls(0), *file_controls(1)])
    with pytest.raises(ValueError, match="names stage 2"):
        validate_controls(task, [*file_controls(0), *file_controls(1), *file_controls(2)])


@pytest.mark.parametrize(
    "kind,category,expect,message",
    [
        (ControlKind.POSITIVE, ControlCategory.PLAUSIBLE_WRONG, GRADED_ONE, "not a positive category"),
        (ControlKind.NEGATIVE, ControlCategory.PLAUSIBLE_WRONG, Expectation(Outcome.GRADED, reward_max=0.5), "<= 0.2"),
        (ControlKind.NEGATIVE, ControlCategory.REWARD_HACK, Expectation(Outcome.GRADED), "<= 0.2"),
        (ControlKind.MALFORMED, ControlCategory.EMPTY_OR_MALFORMED, Expectation(Outcome.GRADED), "<= 0.2"),
        (ControlKind.POSITIVE, ControlCategory.KNOWN_CORRECT, Expectation(Outcome.GRADED, reward_max=1.0), "minimum"),
        (ControlKind.POSITIVE, ControlCategory.KNOWN_CORRECT, Expectation(Outcome.GRADED, reward_min=0.0), "above 0.2"),
        (
            ControlKind.PARTIAL,
            ControlCategory.CRITERION_MUTATION,
            Expectation(Outcome.GRADED, reward_min=0.4, reward_max=0.6),
            "reward components",
        ),
    ],
)
def test_control_labels_must_match_their_expected_grade(kind, category, expect, message):
    with pytest.raises(ValueError, match=message):
        control("c", kind, category, Transcript((reply("x"),)), expect)


@pytest.mark.parametrize(
    "status,bounds,message",
    [
        (Outcome.EXTRACTION_ERROR, {"reward_max": 0.0}, "no reward bounds"),
        (Outcome.INFRA_ERROR, {}, "graded result or an extraction error"),
        (Outcome.GRADED, {"reward_min": 0.8, "reward_max": 0.2}, "reversed"),
        (Outcome.GRADED, {"reward_max": 1.5}, "outside"),
    ],
)
def test_expectation_rejects_ungradeable_bounds(status, bounds, message):
    with pytest.raises(ValueError, match=message):
        Expectation(status, **bounds)


def test_transcript_ends_with_its_only_text_reply():
    with pytest.raises(ValueError, match="final transcript turn"):
        Transcript((reply("thinking"), reply("12")))
    with pytest.raises(ValueError, match="unique"):
        Transcript((shell_turn(("a", "ls")), shell_turn(("a", "pwd")), reply("12")))


def test_replay_mismatches_with_the_task_are_rejected():
    gold, empty, wrong, hack = file_controls()
    shell_on_answer_task = replace(gold, payload=Transcript((shell_turn(("c", "ls")), reply("12"))))
    answer_controls = [
        control("gold", ControlKind.POSITIVE, ControlCategory.KNOWN_CORRECT, Transcript((reply("12"),)), GRADED_ONE),
        control("wrong", ControlKind.NEGATIVE, ControlCategory.PLAUSIBLE_WRONG, Transcript((reply("13"),)), GRADED_ZERO),
        control(
            "short", ControlKind.NEGATIVE, ControlCategory.TASK_SPECIFIC_SHORTCUT, Transcript((reply("1"),)), GRADED_ZERO
        ),
    ]
    extraction_on_shell = replace(empty, payload=Transcript((reply(""),)), expect=Expectation(Outcome.EXTRACTION_ERROR))
    unfinished = replace(wrong, payload=Transcript((shell_turn(("x", "echo 13 > /workspace/answer")),)))

    with pytest.raises(ValueError, match="without a machine"):
        validate_controls(answer_task(), [shell_on_answer_task, *answer_controls])
    with pytest.raises(ValueError, match="requires an executable"):
        validate_controls(answer_task(), [empty, *answer_controls])
    with pytest.raises(ValueError, match="extraction error from a shell verifier"):
        validate_controls(file_task(), [gold, extraction_on_shell, wrong, hack])
    with pytest.raises(ValueError, match="does not end with a reply"):
        validate_controls(file_task(), [gold, empty, unfinished, hack])
