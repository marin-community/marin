# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron Ultra grade scripts score replies with the vendored NeMo Gym scorers in the grader image.

See ``local_grader`` for building the image these tests run.
"""

import pytest
from taskcompendium.convert.code import CODE_GRADER_MEMORY_MB
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AssistantToolCalls, ConversationToolCall, TaskSpec, TextMessage
from taskcompendium.pipeline.controls import run_controls
from taskcompendium.pipeline.models import CheckStatus, Reply
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.datasets.nemotron_ultra.components import pipeline_name
from experiments.post_training.task_curation.tests.conversion import converted_task
from experiments.post_training.task_curation.tests.local_grader import (
    LocalGraderMachines,
    grade,
    local_grader_machines,
    with_verifier_file,
)
from experiments.post_training.task_curation.tests.test_nemotron_ultra import COMPONENT_ROWS, PIPELINES

pytestmark = [pytest.mark.docker, pytest.mark.timeout(600)]

STRUCTURED_TOOL = "ultra_v3_agentic_rl_step73_structured_outputs_v2"
STDIN_BUFFER_SUM = "```python\nimport sys\na, b = map(int, sys.stdin.buffer.read().split())\nprint(a + b)\n```"
"""A correct program that reads its input as bytes, which the SkyRL LiveCodeBench evaluator supports."""
SCHEDULE = '[{"event_id": 0, "start_time": "10:30", "duration": 30}]'
EARLY_SCHEDULE = '[{"event_id": 0, "start_time": "09:00", "duration": 30}]'


def reply(content: str) -> Reply:
    return Reply(TextMessage(role="assistant", content=content))


def report_call(count: object) -> Reply:
    return Reply(
        AssistantToolCalls(calls=(ConversationToolCall(call_id="1", name="report", arguments={"count": count}),))
    )


def task(path: str) -> TaskSpec:
    blend = "mopd" if path == STRUCTURED_TOOL else "rlvr2"
    return converted_task(PIPELINES[pipeline_name(blend, path)], COMPONENT_ROWS[path])


@pytest.fixture(scope="module")
def machines() -> LocalGraderMachines:
    return local_grader_machines()


@pytest.mark.parametrize(
    ("path", "submission", "reward"),
    [
        ("ultra_sft_step3200_calendar_v2", reply(f"<think>Plan.</think>{SCHEDULE}"), 1.0),
        ("ultra_sft_step3200_calendar_v2", reply(EARLY_SCHEDULE), 0.0),
        ("ultra_sft_step3200_ds2_freeform", reply("- apple\n- pear\n- plum"), 1.0),
        ("ultra_sft_step3200_ds2_freeform", reply("apple, pear and plum"), 0.0),
        ("ultra_sft_step3200_instruction_following", reply("Tea is a drink brewed from leaves."), 1.0),
        ("ultra_sft_step3200_instruction_following", reply("Tea, a drink, is brewed from leaves."), 0.0),
        ("ultra_sft_step3200_stem_mcqa", reply("\\boxed{B}"), 1.0),
        ("ultra_sft_step3200_stem_mcqa", reply("\\boxed{A}"), 0.0),
        ("ultra_sft_step3200_comp_coding", reply(STDIN_BUFFER_SUM), 1.0),
        ("ultra_sft_step3200_structured_outputs_v2", reply('{"count": 3}'), 1.0),
        ("ultra_sft_step3200_structured_outputs_v2", reply('{"count": "three"}'), 0.0),
        (STRUCTURED_TOOL, report_call(3), 1.0),
        (STRUCTURED_TOOL, report_call("three"), 0.0),
        ("ultra_sft_step3200_reasoning_gym", reply("x = 50 - 8, so <answer>42</answer>"), 1.0),
        ("ultra_sft_step3200_reasoning_gym", reply("<answer>58</answer>"), 0.0),
    ],
)
def test_grade_script_scores_the_reply(path, submission, reward, machines):
    result = grade(task(path), submission, machines, CODE_GRADER_MEMORY_MB)
    assert (result.status, result.reward) == (Outcome.GRADED, reward), result


@pytest.mark.parametrize(
    "path",
    ["ultra_sft_step3200_comp_coding", "ultra_sft_step3200_rdkit", "ultra_sft_step3200_toolcall_schema"],
)
def test_controls_score_the_reference_one_and_wrong_replies_zero(path, machines):
    controls = PIPELINES[pipeline_name("rlvr2", path)].controls
    assert controls is not None
    report = run_controls(task(path), controls=controls, machines=machines)
    statuses = {check.check: check.status for check in report.checks}
    assert statuses == {"empty": CheckStatus.PASS, "golden": CheckStatus.PASS, "negative": CheckStatus.PASS}, report


@pytest.mark.parametrize(
    ("path", "module"),
    [
        ("ultra_sft_step3200_calendar_v2", "calendar"),
        ("ultra_sft_step3200_ds2_freeform", "format_verification"),
        ("ultra_sft_step3200_instruction_following", "instruction_following"),
        ("ultra_sft_step3200_stem_mcqa", "mcqa"),
        ("ultra_sft_step3200_comp_coding", "code_gen"),
        ("ultra_sft_step3200_structured_outputs_v2", "structured_outputs"),
        ("ultra_sft_step3200_rdkit", "rdkit_chemistry"),
        ("ultra_sft_step3200_toolcall_schema", "tool_call"),
        ("ultra_sft_step3200_reasoning_gym", "answer_extraction"),
    ],
)
def test_grade_script_fails_rather_than_scoring_when_its_scorer_cannot_import(path, module, machines):
    broken = inline_resource(
        f"skyrl_gym/envs/nemotron_ultra/{module}.py", b"import package_missing_from_the_grader_image\n"
    )
    result = grade(with_verifier_file(task(path), broken), reply("\\boxed{1}"), machines, CODE_GRADER_MEMORY_MB)
    assert result.status == Outcome.INFRA_ERROR
    assert result.diagnostics is not None and result.diagnostics["exit_code"] != 0
