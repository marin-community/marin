# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validated Recursive Task Synthesis tasks with original-root verifier delivery."""

import hashlib
import tomllib

from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.harbor.adapter import ARCHIVED_SOURCE_CONFIG
from taskcompendium.harbor.root_delivery import REWARD_PATH, DeliveredVerifier, delivery_commands, shell_quiesce_command
from taskcompendium.integrity import DeliveredVerifierFile, RootDeliveryContract, SharedDependencyState
from taskcompendium.models import (
    ArtifactKind,
    FileReward,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    TaskSpec,
    VerifierArtifact,
)
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, IntendedUse, RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.native_harbor import native_task, unpack_native
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.shell_heredoc_repair import repair_task
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

DATASET = "laion/rts-1k-rootless-apptainer"
REVISION = "6dbb24eedc0d7ffe5431e1b052b0de8e19eaf6fe"
VERSION = "2026.11.17.1"
VERIFIER_NAMES = frozenset({"test.sh", "test.sh.uvx_bak", "test_state.py"})
VERIFIER_OUTPUT_PATH = "/logs/verifier/test-output.txt"
VERIFIER_EXIT_PATH = "/logs/verifier/exit-code.txt"
COPIED_REWARD_PATH = "/source-reward.txt"
RUBRIC = """Compare the public terminal instruction with the original native verifier and solution.
Check whether required behavior is specified and whether the verifier measures it faithfully.
The original environment files and compatibility patches are part of the task contract.
Source validation on aarch64 Apptainer does not establish execution on the current Harbor backend.
"""


def convert_rst(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    """Retain the source actor and deliver its unchanged verifier only after it finishes."""
    task = native_task(row, context)
    if isinstance(task, ImportRejection):
        return task
    task = repair_task(task)
    resources = {resource.path: resource for resource in task.resources.verifier}
    configuration = tomllib.loads(resource_bytes(resources["task.toml"]).decode())
    private = {
        name.removeprefix("tests/"): resource for name, resource in resources.items() if name.startswith("tests/")
    }
    if private.keys() != VERIFIER_NAMES or configuration.get("agent", {}).get("user") is not None:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="rst_source_contract_changed",
            detail="RST root delivery requires the surveyed verifier layout and original default root actor",
        )
    for name, resource in private.items():
        expected = 0o755 if name == "test.sh" else 0o644
        if resource.mode is None or int(resource.mode, 8) != expected:
            raise ValueError(f"RST original verifier mode changed: {name}")
    deadline = configuration.get("verifier", {}).get("timeout_sec")
    verifier = DeliveredVerifier({name: resource_bytes(resource) for name, resource in private.items()})
    transport_environment = task.environment_requirements.model_copy(
        update={"working_directory": "/", "setup_commands": (), "environment_variables": {}}
    )
    archived = tuple(
        resource.model_copy(update={"path": "source_archive/" + resource.path}) for resource in task.resources.verifier
    )
    assert any(resource.path == ARCHIVED_SOURCE_CONFIG for resource in archived)
    return task.model_copy(
        update={
            "environment_requirements": task.environment_requirements.model_copy(
                update={"capabilities": tuple(dict.fromkeys((*task.environment_requirements.capabilities, "shell")))}
            ),
            "grader": ScriptGrader(
                argv=("cp", COPIED_REWARD_PATH, REWARD_PATH),
                cwd="/",
                environment=transport_environment,
                answer_path=None,
                collect=(
                    shell_quiesce_command(),
                    *delivery_commands(verifier, output_path=VERIFIER_OUTPUT_PATH, exit_code_path=VERIFIER_EXIT_PATH),
                ),
                artifacts=(
                    VerifierArtifact(
                        source=VERIFIER_OUTPUT_PATH, target="/source-test-output.txt", kind=ArtifactKind.FILE
                    ),
                    VerifierArtifact(source=VERIFIER_EXIT_PATH, target="/source-exit-code.txt", kind=ArtifactKind.FILE),
                    VerifierArtifact(source=REWARD_PATH, target=COPIED_REWARD_PATH, kind=ArtifactKind.FILE),
                ),
                reward=FileReward(files=(RewardFile(path=REWARD_PATH, format=RewardFileFormat.NUMBER),)),
                timeout=deadline,
            ),
            "resources": ResourceGroups(
                all=task.resources.all, worker=task.resources.worker, oracle=task.resources.oracle, verifier=archived
            ),
            "integrity_contract": RootDeliveryContract(
                dependency_state=SharedDependencyState.AGENT_MUTABLE,
                agent_user=None,
                undeclared_deadline=deadline is None,
                verifier_files=tuple(
                    DeliveredVerifierFile(
                        name=name,
                        mode=int(resource.mode, 8),
                        sha256=hashlib.sha256(resource_bytes(resource)).hexdigest(),
                    )
                    for name, resource in sorted(private.items())
                ),
            ),
            "tags": ("root-delivery", "recursive-task-synthesis"),
            "labels": {**task.labels, "rst_delivery_version": VERSION},
        }
    )


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                origin=SourceOrigin.RST,
                upstream_dataset=DATASET,
                upstream_name="Recursive Task Synthesis validated RTS-1K",
                build_path="Original-root TaskSpec with grading-time verifier delivery",
                family="terminal",
                tags=("license:cc-by-4.0", "recursive-task-synthesis"),
                task_format=TaskFormat.TASKSPEC,
                notes=(
                    "984 source-validated packages retain original root actors, instructions, builds and solutions. "
                    "The original verifier is host-delivered at grading time and its reward copied for numeric parsing. "
                    "Mutable Dockerfile bases and build downloads use explicit Daytona EVAL declarative provisioning. "
                    "Pending current-pin execution qualification and admission."
                ),
            ),
            config=CurationRecipe(
                name="rst-validated",
                version=VERSION,
                source=HfSource(DATASET, REVISION, ("tasks.parquet",), SourceFormat.PARQUET, decode=unpack_native),
                convert=convert_rst,
                intended_use=IntendedUse.TRAIN,
                rubric=RUBRIC,
                oracle_solutions=True,
            ),
            pipeline=process_rows,
        )
    ]
