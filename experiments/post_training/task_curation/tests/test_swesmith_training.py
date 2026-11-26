# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.strata import MembershipState, SelectionDefinition, TaskStrata, read_selection_manifest
from marin.execution.lazy import ArtifactStep, StepContext
from taskcompendium.identity import canonical_sha256
from taskcompendium.importers.harbor.config import UnsupportedHarborFeature
from taskcompendium.models import (
    AgentMachine,
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.post_training.task_curation.selection import (
    SELECTION_PREFIX,
    SUMMARY_FILENAME,
    TASK_MANIFEST_FILENAME,
    TASK_MANIFEST_SCHEMA,
    TASK_SCHEMA,
    TASK_SPEC_COLUMN,
    TRAIN_FILENAME,
    GraderKind,
    _selection_row,
    task_records,
)
from experiments.post_training.task_curation.swesmith_training import (
    SWESMITH_DATASET,
    SWESMITH_MACHINE,
    lower_swesmith_task,
    native_swesmith_selection,
)

VERSION = "2026.11.19.2"
BOOTSTRAP = "touch /testbed/setup"
INSTRUCTION = "Fix the repository.\n"
VERIFIER = b"test -f /testbed/setup && printf 1 > /logs/verifier/reward.txt\n"
PYTHON_IMAGE = "python@sha256:" + "a" * 64


@pytest.fixture
def task() -> TaskSpec:
    package = {
        "task.toml": (
            f"""
[environment.setup]
command = "{BOOTSTRAP}"
timeout_sec = 1800
[agent]
user = "longloop-agent"
[agent.env]
HOME = "/home/longloop-agent"
[verifier]
user = "root"
timeout_sec = 600
""".encode()
        ),
        "environment/Dockerfile": f"FROM {PYTHON_IMAGE}\nWORKDIR /testbed\n".encode(),
        "environment/long-loop-pi/package-lock.json": (
            b"""{
    "packages": {"node_modules/@earendil-works/pi-coding-agent": {"version": "0.87.0"}}
}"""
        ),
        "instruction.md": INSTRUCTION.encode(),
        "tests/test.sh": VERIFIER,
    }
    return TaskSpec(
        id="swesmith:fixture",
        source=Source(dataset=SWESMITH_DATASET, revision="release", row="fixture", importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content=INSTRUCTION),)),
        environment_requirements=EnvironmentRequirements(setup_commands=(BOOTSTRAP,)),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=NoGrader(reason="Harbor grades the completed trial"),
        resources=ResourceGroups(verifier=tuple(inline_resource(name, contents) for name, contents in package.items())),
        tags=("harbor", "agentic"),
    )


def test_native_selection_keeps_membership_and_binds_the_package_verifier(task, tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    strata = TaskStrata(family="agentic", source=SWESMITH_DATASET, agentic=True)
    row = _selection_row(task, strata, state=MembershipState.ACTIVE)
    pq.write_table(
        pa.Table.from_pylist(list(task_records(iter([task]), {task.id: strata})), TASK_SCHEMA), parent / TRAIN_FILENAME
    )
    pq.write_table(
        pa.Table.from_pylist([row.model_dump(mode="json")], TASK_MANIFEST_SCHEMA), parent / TASK_MANIFEST_FILENAME
    )
    definition = SelectionDefinition(
        name="fixture", version=VERSION, sampling="fixed", task_files=(TRAIN_FILENAME,), active_files=(TRAIN_FILENAME,)
    )
    (parent / SUMMARY_FILENAME).write_text(
        json.dumps(
            {
                "selection_strata": definition.model_dump(mode="json"),
                "grader_kind_counts": {"train": {GraderKind.HARNESS: 1, GraderKind.IMAGE: 0}},
                "harness_graded_datasets": [SWESMITH_DATASET],
            }
        )
    )
    source = ArtifactStep.adopt(f"{SELECTION_PREFIX}/{definition.name}", VERSION, str(parent))
    stage = native_swesmith_selection(source)
    output = tmp_path / "output"
    context = StepContext.for_run(str(output), str(tmp_path), deps=stage.deps)
    stage.run(stage.build_config(context))

    manifest = read_selection_manifest(str(output))
    assert manifest is not None
    assert manifest.definition.parent.root == str(parent)
    assert {key: value.strata for key, value in manifest.tasks.items()} == {task.id: strata}
    (native_row,) = pq.read_table(output / TRAIN_FILENAME).to_pylist()
    native = TaskSpec.model_validate_json(native_row[TASK_SPEC_COLUMN])
    assert native.id == task.id
    assert native.source == task.source
    assert native.context == task.context
    assert native.environment_requirements.setup_commands[1:] == (BOOTSTRAP,)
    assert native.environment_requirements.environment_variables["HOME"] == "/home/longloop-agent"
    assert native.environment_requirements.working_directory == "/testbed"
    assert isinstance(native.grader, ScriptGrader)
    assert isinstance(native.grader.placement, AgentMachine)
    assert native.grader.timeout == 900
    assert resource_bytes(native.resources.verifier[0]) == VERIFIER
    assert manifest.tasks[task.id].task_digest == canonical_sha256(native.model_dump(mode="json"))
    assert manifest.tasks[task.id].grader_kind == GraderKind.IMAGE


def test_native_selection_rejects_changed_bootstrap_or_execution_identity(task):
    modified = task.model_copy(update={"environment_requirements": EnvironmentRequirements(setup_commands=("true",))})
    with pytest.raises(ValueError, match="does not match the admitted SWESmith setup"):
        lower_swesmith_task(modified, SWESMITH_MACHINE, {})
    with pytest.raises(UnsupportedHarborFeature, match="deployment expects learner"):
        lower_swesmith_task(task, SWESMITH_MACHINE.model_copy(update={"user": "learner"}), {})
