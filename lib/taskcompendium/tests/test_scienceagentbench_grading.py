# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from shellbox.machine import DockerImage, MachineSpec

from taskcompendium.grading_result import Outcome
from taskcompendium.importers.scienceagentbench_assets.evaluate import evaluate_output
from taskcompendium.importers.scienceagentbench_grading import papyrus_grader
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    GradingAttempt,
    PlainText,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.task_grading import grade_task

from .test_replay_grading import GRADER_IMAGE, REPLAY_IMAGE, ReplayMachines
from .test_runtime import finished

GOLD_ROWS = b'["1", "1.235", "a"]\n["2", "2.346", "b"]\n'


def grade_papyrus_program(tmp_path, program, gold_rows=GOLD_ROWS):
    replay = EnvironmentRequirements(
        docker_image=REPLAY_IMAGE,
        command_semantics=CommandSemantics.LINUX_PROCESS,
        working_directory="/testbed",
    )
    package = papyrus_grader(
        environment=EnvironmentRequirements(
            docker_image=GRADER_IMAGE,
            command_semantics=CommandSemantics.LINUX_PROCESS,
            working_directory="/testbed",
        ),
        replay_environment=replay,
        gold_rows=gold_rows,
        replay_timeout=10,
        verifier_timeout=30,
    )
    task = TaskSpec(
        id="papyrus-fixture",
        source=Source(dataset="test/papyrus", revision="1", row="0", importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content="Write the filtered table."),)),
        environment_requirements=replay,
        answer_type=AnswerType.FILE,
        answer_format=PlainText(),
        output_paths=("/testbed/solution.py",),
        grader=package.grader,
        resources=ResourceGroups(verifier=package.resources),
    )
    machines = ReplayMachines(tmp_path / "machines")
    result = grade_task(
        TaskSpec.model_validate_json(task.model_dump_json()),
        GradingAttempt(finished(task), {"/testbed/solution.py": program}),
        machine_factory=machines,
        machine_spec=MachineSpec(DockerImage(GRADER_IMAGE)),
    )
    return result, machines.machines


@pytest.mark.parametrize(
    "rows,expected",
    [
        ([[2, 2.3456, "b"], [1, 1.2346, "a"]], 1.0),
        ([[2, 2.3456, "wrong"], [1, 1.2346, "a"]], 0.0),
        ([[1, 1.2346, "a"]], 0.0),
        ([[2, 2.3456, "b"], [1, 1.2346, "a"], [99, 9.0, "extra"]], 1.0),
    ],
)
def test_pickle_replay_preserves_official_order_rounding_and_gold_prefix(tmp_path, rows, expected):
    program = f"""
from pathlib import Path
import pandas as pd
assert not Path('../tests/gold_rows.jsonl').exists()
Path('pred_results').mkdir()
pd.DataFrame({rows!r}, columns=['Activity_ID', 'value', 'label']).to_pickle('pred_results/papyrus_filtered.pkl')
""".encode()
    result, machines = grade_papyrus_program(tmp_path, program)
    assert (result.status, result.reward) == (Outcome.GRADED, expected), result
    assert len(machines) == 2 and all(machine.closed for machine in machines)
    replay, grader = machines
    assert replay.local("/replay/decode_papyrus.py").is_file()
    assert not grader.local("/replay/decode_papyrus.py").exists()
    assert not grader.local("/testbed/pred_results/papyrus_filtered.pkl").exists()
    assert not grader.local("/testbed/solution.py").exists()


def test_pickle_cannot_read_gold_during_decoding(tmp_path):
    program = b"""
from pathlib import Path
import pickle

class ReadGold:
    def __reduce__(self):
        expression = (
            "(__import__('pathlib').Path('../tests/gold_rows.jsonl').read_text(), "
            "__import__('pandas').DataFrame({'Activity_ID': [1], 'value': [42]}))[1]"
        )
        return (eval, (expression,))

Path('pred_results').mkdir()
Path('pred_results/papyrus_filtered.pkl').write_bytes(pickle.dumps(ReadGold()))
"""
    result, machines = grade_papyrus_program(tmp_path, program, b'["1", "42"]\n')
    assert (result.status, result.reward) == (Outcome.GRADED, 0.0), result
    assert len(machines) == 1 and machines[0].closed


def test_malformed_trusted_gold_is_infrastructure_failure(tmp_path):
    program = b"""
from pathlib import Path
import pandas as pd
Path('pred_results').mkdir()
pd.DataFrame({'Activity_ID': [1], 'value': [42]}).to_pickle('pred_results/papyrus_filtered.pkl')
"""
    result, machines = grade_papyrus_program(tmp_path, program, b"broken gold\n")
    assert (result.status, result.reward) == (Outcome.INFRA_ERROR, None), result
    assert len(machines) == 2 and all(machine.closed for machine in machines)


@pytest.mark.parametrize(
    "body,expected",
    [
        ("return 1, 'passed'", 1.0),
        ("return 0.8, 'not task success'", 0.0),
        ("return False", 0.0),
        ("raise ValueError('invalid candidate shape')", 0.0),
    ],
)
def test_source_evaluator_reports_binary_task_success(tmp_path, body, expected):
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text(f"def eval():\n    {body}\n")
    primary = tmp_path / "candidate.csv"
    primary.write_text("candidate output")
    verdict = evaluate_output(evaluator, primary, tmp_path)
    assert (verdict["status"], verdict["reward"]) == ("scored", expected)


def test_missing_secondary_output_is_scored_but_missing_gold_is_ungraded(tmp_path):
    candidate_directory = tmp_path / "pred_results"
    candidate_directory.mkdir()
    primary = candidate_directory / "primary.csv"
    primary.write_text("present")
    evaluator = tmp_path / "evaluate.py"
    secondary = candidate_directory / "secondary.csv"
    evaluator.write_text(f"from pathlib import Path\ndef eval():\n    return Path({str(secondary)!r}).read_text()\n")
    verdict = evaluate_output(evaluator, primary, candidate_directory)
    assert (verdict["status"], verdict["reward"]) == ("scored", 0.0)
    gold = tmp_path / "gold.csv"
    evaluator.write_text(f"from pathlib import Path\ndef eval():\n    return Path({str(gold)!r}).read_text()\n")
    with pytest.raises(FileNotFoundError):
        evaluate_output(evaluator, primary, candidate_directory)


def test_missing_evaluator_dependency_is_ungraded(tmp_path):
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text("import nonexistent_scienceagentbench_test_dependency\ndef eval():\n    return 1, 'ok'\n")
    primary = tmp_path / "candidate.csv"
    primary.write_text("present")
    with pytest.raises(ModuleNotFoundError):
        evaluate_output(evaluator, primary, tmp_path)
