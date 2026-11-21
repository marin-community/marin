# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-SWE's original actor and host-side report-scoring boundary."""

import base64
import hashlib
import io
import json
import shlex
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pyarrow.parquet as pq
import tomlkit
from finestore.rl.status_taxonomy import DataRestriction
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.answers import source_defect
from taskcompendium.harbor.root_delivery import home_restore_commands, multiswe_run_command, shell_quiesce_command
from taskcompendium.integrity import MultiSweDeliveryContract, SharedDependencyState
from taskcompendium.models import (
    AnswerType,
    ArtifactKind,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    FileReward,
    PlainText,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    TaskSpec,
    TextMessage,
    VerifierArtifact,
    VerifierCommand,
)
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, RawRow
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.datasets.prime_intellect.capture_artifacts import archive_members
from experiments.post_training.task_curation.datasets.prime_intellect.capture_models import ImageBinding
from experiments.post_training.task_curation.datasets.prime_intellect.multiswe.data import (
    build_problem_statement,
    build_workdir,
    restore_row,
    source_image_reference,
)
from experiments.post_training.task_curation.datasets.prime_intellect.multiswe.environment import report_environment
from experiments.post_training.task_curation.datasets.prime_intellect.multiswe.source_commands import (
    ENV,
    REMOVE_GRADING_ARTIFACTS,
)
from experiments.post_training.task_curation.datasets.prime_intellect.native_scripts import SOLVER_SYSTEM_PROMPT
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

HERE = Path(__file__).parent
DATASET = "PrimeIntellect/Multi-SWE-RL-Verified"
REVISION = "64b3ebf62750b1373e26106cfc658a1861809d0f"
TASKSET_REVISION = "b41d8759070715dc102000e06fba0d73ebce9955"
VERSION = "2026.11.16.2"
CAPTURE_INPUT = "grading_artifacts"
CAPTURES_SHA256 = "89d79cfefaf4ac785e1f0d4c48ce21d3a45a391fb14ddb78741f00b951b44bf0"
RESOURCE_BUDGET_BYTES = 64 * 1024 * 1024
REWARD_PATH = "/logs/verifier/reward.txt"
TRANSPORT = {
    "test-output": "/tmp/multiswe-test-output.txt",
    "read-exit-code": "/tmp/multiswe-read-exit-code.txt",
    "harness-stdout": "/tmp/multiswe-harness-stdout.txt",
    "harness-stderr": "/tmp/multiswe-harness-stderr.txt",
    "harness-exit-code": "/tmp/multiswe-harness-exit-code.txt",
    "cat-exit-code": "/tmp/multiswe-cat-exit-code.txt",
}
PARSER_FILES = ("__init__.py", "data.py", "score.py", "host_grade.py", "_qiskit/__init__.py", "_qiskit/qiskit.py")
PUBLIC_SETUP = (
    "test -d .",
    "command -v patch || (apt-get -o Acquire::Retries=3 update && apt-get -o Acquire::Retries=3 install -y patch)",
    "rm -f /home/fix.patch /home/test_output.txt /home/create_fix_patch.sh /home/extract_multiswe_fix_patch.sh",
)
RUBRIC = """Compare the public repository issue with the original golden patch and hidden tests.
The source image, original /home harness and upstream multi-swe-bench report parser are part of the contract.
Check whether the issue specifies the required behavior and whether the tests measure that behavior.
"""


@dataclass(frozen=True)
class CapturedImage:
    binding: ImageBinding
    archive: bytes
    archive_sha256: str
    receipt_json: str


@cache
def captured_images(path: StoragePath) -> dict[str, CapturedImage]:
    """Read the complete pinned auxiliary input, verifying every original image archive."""
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != CAPTURES_SHA256:
        raise ValueError("Multi-SWE capture parquet differs from its frozen private input")
    records = {}
    with io.BytesIO(data) as stream:
        rows = pq.read_table(stream).to_pylist()
    for row in rows:
        receipt = json.loads(row["capture_json"])
        binding = ImageBinding.model_validate_json(row["binding_json"])
        archive = row["archive"]
        digest = hashlib.sha256(archive).hexdigest()
        if (
            digest != row["archive_sha256"]
            or digest != receipt["archive_sha256"]
            or len(archive) != receipt["archive_bytes"]
            or archive_members(archive) != receipt["members"]
            or receipt["taskset_revision"] != TASKSET_REVISION
            or receipt["binding"] != binding.model_dump(mode="json")
            or row["pinned_reference"] != binding.pinned_reference
        ):
            raise ValueError("Multi-SWE capture bytes or original image binding changed")
        capture = CapturedImage(binding, archive, digest, row["capture_json"])
        for task_id in binding.task_ids:
            if task_id in records:
                raise ValueError(f"Repeated Multi-SWE image binding: {task_id}")
            records[task_id] = capture
    if len(rows) != 2231 or len(records) != 2232:
        raise ValueError("Multi-SWE private input omits source images or tasks")
    return records


def task_from_capture(row: RawRow, capture: CapturedImage) -> TaskSpec | ImportRejection:
    """Convert one original source row and its frozen image-resident grading files."""
    data = restore_row(row.data)
    if (row.source.dataset, row.source.revision) != (DATASET, REVISION):
        raise ValueError("Multi-SWE conversion requires the declared source pin")
    if (
        data["instance_id"] not in capture.binding.task_ids
        or source_image_reference(data) not in capture.binding.original_references
    ):
        raise ValueError("Multi-SWE row is not bound to this original image capture")
    patch = data.get("fix_patch") or ""
    if not patch.strip():
        return source_defect("missing_gold_patch", "The source row has no original golden fix patch")
    patch = patch if patch.endswith("\n") else patch + "\n"
    workdir = build_workdir(data)
    setup = (
        *("sh -c " + shlex.quote(command) for command in PUBLIC_SETUP),
        "bash -c " + shlex.quote(REMOVE_GRADING_ARTIFACTS),
    )
    actor = EnvironmentRequirements(
        capabilities=("shell",),
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_image=capture.binding.pinned_reference,
        working_directory=workdir,
        setup_commands=setup,
        environment_variables=ENV,
    )
    extract = (HERE / "extract_fix_patch.sh").read_bytes()
    deliver_extract = VerifierCommand(
        argv=(
            "bash",
            "-c",
            "printf '%s' "
            + shlex.quote(base64.b64encode(extract).decode())
            + " | base64 -d > /home/extract_multiswe_fix_patch.sh",
        ),
        cwd="/",
    )
    base = data.get("base")
    base_commit = (base.get("sha") if isinstance(base, dict) else None) or "HEAD"
    source_config = tomlkit.dumps({"environment": {"cpus": 4, "memory_mb": 4096, "storage_mb": 10240}}).encode()
    private = (
        *(inline_resource("multiswe/" + name, (HERE / name).read_bytes()) for name in PARSER_FILES),
        inline_resource("row.json", json.dumps(row.data, sort_keys=True, ensure_ascii=False).encode()),
        inline_resource("home-grading.tar.gz", capture.archive),
        inline_resource("extract_fix_patch.sh", extract),
        inline_resource("capture.json", capture.receipt_json.encode()),
        inline_resource("task.toml", source_config),
        inline_resource("multiswe/LICENSE", (HERE.parent / "LICENSE").read_bytes()),
    )
    oracle = (
        "#!/bin/bash\nset -eu\ncd "
        + shlex.quote(workdir)
        + "\ngit apply --whitespace=fix /solution/gold.patch || patch --fuzz=5 -p1 -i /solution/gold.patch\n"
    )
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(
            events=(TextMessage(role="user", content=SOLVER_SYSTEM_PROMPT + "\n\n" + build_problem_statement(data)),)
        ),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        environment_requirements=actor,
        grader=ScriptGrader(
            argv=(
                "python3",
                "-m",
                "multiswe.host_grade",
                "--row",
                "/tests/row.json",
                "--test-output",
                "/source-test-output.txt",
                "--read-exit-code",
                "/source-read-exit-code.txt",
                "--harness-stdout",
                "/source-harness-stdout.txt",
                "--harness-stderr",
                "/source-harness-stderr.txt",
                "--reward",
                REWARD_PATH,
            ),
            cwd="/tests",
            environment=report_environment(),
            answer_path=None,
            timeout=None,
            collect=(
                shell_quiesce_command(),
                *home_restore_commands(capture.archive),
                deliver_extract,
                multiswe_run_command(
                    repo=data["repo"],
                    base=base_commit,
                    output_path=TRANSPORT["test-output"],
                    read_exit_code_path=TRANSPORT["read-exit-code"],
                    pipeline_stdout_path=TRANSPORT["harness-stdout"],
                    pipeline_stderr_path=TRANSPORT["harness-stderr"],
                    pipeline_rc_path=TRANSPORT["harness-exit-code"],
                    cat_rc_path=TRANSPORT["cat-exit-code"],
                ),
            ),
            artifacts=tuple(
                VerifierArtifact(source=path, target="/source-" + name + ".txt", kind=ArtifactKind.FILE)
                for name, path in TRANSPORT.items()
            ),
            reward=FileReward(files=(RewardFile(path=REWARD_PATH, format=RewardFileFormat.NUMBER),)),
        ),
        resources=ResourceGroups(
            verifier=private,
            oracle=(
                inline_resource("solution/solve.sh", oracle.encode()).model_copy(update={"mode": "0755"}),
                inline_resource("solution/gold.patch", patch.encode()),
            ),
        ),
        integrity_contract=MultiSweDeliveryContract(
            dependency_state=SharedDependencyState.AGENT_MUTABLE,
            agent_user=None,
            archive_sha256=capture.archive_sha256,
            undeclared_deadline=True,
        ),
        tags=("multi-swe-delivery", "swe-repo"),
        labels={
            "multiswe_conversion_version": VERSION,
            "source_image_reference": source_image_reference(data),
            "taskset_revision": TASKSET_REVISION,
        },
    )


def convert_multiswe(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    capture = captured_images(context.inputs[CAPTURE_INPUT] / "captures.parquet")[row.data["instance_id"]]
    return task_from_capture(row, capture)


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                origin=SourceOrigin.PRIME_INTELLECT,
                upstream_dataset=DATASET,
                upstream_name="Multi-SWE-RL-Verified",
                build_path="Original-root TaskSpec with late /home delivery and upstream host report parsing",
                family="swe-repo",
                tags=("license:cc0-1.0",),
                task_format=TaskFormat.TASKSPEC,
                notes=(
                    "Original source images and captured /home grading archives are pinned. "
                    "Original root actor and host-side multi-swe-bench1.1.2 report checks retained. "
                    "Pending current-pin execution qualification and admission."
                ),
            ),
            config=CurationRecipe(
                name="multi-swe-rl-verified",
                version=VERSION,
                source=HfSource(
                    DATASET,
                    REVISION,
                    tuple(f"data/train-{i:05d}-of-00003.parquet" for i in range(3)),
                    SourceFormat.PARQUET,
                ),
                inputs={
                    CAPTURE_INPUT: HfSource(
                        "open-athena/atlas-multiswe-grading-artifacts-20261115-v1",
                        "acc1f291ad6ad293c095bd8cde868ab2715717a2",
                        ("captures.parquet",),
                        SourceFormat.PARQUET,
                    )
                },
                convert=convert_multiswe,
                intended_use=IntendedUse.TRAIN,
                rubric=RUBRIC,
                oracle_solutions=True,
                resource_budget_bytes=RESOURCE_BUDGET_BYTES,
            ),
            pipeline=process_rows,
        )
    ]
