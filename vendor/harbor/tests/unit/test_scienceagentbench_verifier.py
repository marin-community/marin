# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass

import pytest
from shellbox.machine import DockerImage, MachineSpec
from taskcompendium.models import (
    AnswerType,
    ArtifactKind,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ReplayStage,
    ResourceGroups,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
    VerifierArtifact,
)
from taskcompendium.runtime.resources import inline_resource

from harbor.environments.daytona import DaytonaEnvironment
from harbor.models.task.config import EnvironmentConfig, TaskConfig
from harbor.models.task.task import Task
from harbor.models.trial.config import VerifierConfig
from harbor.models.trial.paths import TrialPaths
from harbor.verifier.factory import VerifierFactory
from harbor.verifier.verifier import VerifierRuntimeError
from lib.taskcompendium.tests.grading_machines import GRADER_IMAGE, REPLAY_IMAGE, ReplayMachine, ReplayMachines


@dataclass(kw_only=True)
class ActorMachine(ReplayMachine):
    spec: MachineSpec


@pytest.mark.parametrize("candidate,gold,expected", [("42", b"42", 1.0), ("wrong", b"42", 0.0), ("42", None, None)])
async def test_harbor_bridge_records_replayed_grade_without_exposing_private_assets(tmp_path, candidate, gold, expected):
    replay = EnvironmentRequirements(
        docker_image=REPLAY_IMAGE,
        command_semantics=CommandSemantics.LINUX_PROCESS,
        working_directory="/testbed",
    )
    grader = ScriptGrader(
        environment=replay.model_copy(update={"docker_image": GRADER_IMAGE}),
        argv=("python3", "/tests/grade.py"),
        cwd="/testbed",
        answer_path=None,
        timeout=30,
        replay=ReplayStage(
            environment=replay,
            argv=("python3", "/testbed/solution.py"),
            cwd="/testbed",
            timeout=10,
            artifacts=(
                VerifierArtifact(
                    source="/testbed/pred_results", target="/testbed/pred_results", kind=ArtifactKind.DIRECTORY
                ),
            ),
        ),
    )
    grader_program = b"""
from pathlib import Path
import os
assert 'ACTOR_TOKEN' not in os.environ
assert not Path('solution.py').exists()
assert not Path('../controls/reference.py').exists()
assert not Path('pred_results/actor-only.txt').exists()
print(float(Path('pred_results/result.txt').read_text() == Path('../tests/gold.txt').read_text()))
"""
    resources = (inline_resource("grade.py", grader_program),)
    if gold is not None:
        resources += (inline_resource("gold.txt", gold),)
    task = TaskSpec(
        id="scienceagentbench-fixture",
        source=Source(dataset="test/bridge", revision="1", row="1", importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content="Write solution.py."),)),
        environment_requirements=replay,
        answer_type=AnswerType.FILE,
        answer_format=PlainText(),
        output_paths=("/testbed/solution.py",),
        grader=grader,
        resources=ResourceGroups(
            verifier=resources,
            oracle=(inline_resource("controls/reference.py", b"private reference"),),
        ),
    )
    task_dir = tmp_path / "task"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "environment").mkdir()
    (task_dir / "instruction.md").write_text("Write solution.py.")
    (task_dir / "task.toml").write_text(
        TaskConfig(environment=EnvironmentConfig(docker_image=REPLAY_IMAGE, workdir="/testbed")).model_dump_toml()
    )
    (task_dir / "tests" / "task.json").write_text(task.model_dump_json())
    (task_dir / "tests" / "test.sh").write_text("exit 1\n")
    harbor_task = Task(task_dir)
    paths = TrialPaths(tmp_path / "trial")
    paths.mkdir()
    environment = DaytonaEnvironment(
        environment_dir=task_dir / "environment",
        environment_name="fixture",
        session_id="fixture",
        trial_paths=paths,
        task_env_config=harbor_task.config.environment,
    )
    actor_root = tmp_path / "actor"
    (actor_root / "testbed" / "pred_results").mkdir(parents=True)
    program = f"""
from pathlib import Path
import os
assert 'ACTOR_TOKEN' not in os.environ
assert not Path('../tests/gold.txt').exists()
assert not Path('../controls/reference.py').exists()
assert not Path('pred_results/actor-only.txt').exists()
Path('pred_results').mkdir()
Path('pred_results/result.txt').write_text({candidate!r})
""".encode()
    (actor_root / "testbed" / "solution.py").write_bytes(program)
    (actor_root / "testbed" / "pred_results" / "actor-only.txt").write_text("actor output must be discarded")
    environment.machine = ActorMachine(
        root=actor_root,
        workdir="/testbed",
        spec=MachineSpec(DockerImage(REPLAY_IMAGE), env={"ACTOR_TOKEN": "fixture"}),
    )
    machines = ReplayMachines(tmp_path / "grading-machines")
    environment.factory = machines
    verifier = VerifierFactory.create_verifier_from_config(
        VerifierConfig(import_path="experiments.evaluation.scienceagentbench_verifier:ScienceAgentBenchVerifier"),
        task=harbor_task,
        trial_paths=paths,
        environment=environment,
    )
    if expected is None:
        with pytest.raises(VerifierRuntimeError):
            await verifier.verify()
        assert not paths.reward_json_path.exists()
    else:
        result = await verifier.verify()
        assert result.rewards == {"reward": expected}
        assert json.loads(paths.reward_json_path.read_text()) == result.rewards
    recorded = json.loads((paths.verifier_dir / "taskcompendium-result.json").read_text())
    assert (recorded["status"], recorded["reward"]) == ("infra_error" if expected is None else "graded", expected)
    assert (paths.artifacts_dir / "solution.py").read_bytes() == program
    assert not (actor_root / "tests").exists()
    assert len(machines.machines) == 2 and all(machine.closed for machine in machines.machines)
    assert not machines.machines[0].local("/tests/gold.txt").exists()
