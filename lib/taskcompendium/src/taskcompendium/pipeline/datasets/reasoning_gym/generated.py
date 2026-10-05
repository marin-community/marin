# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct generated Reasoning Gym entries with their pinned native scorer contract."""

import json
import os
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Iterator
from tempfile import TemporaryDirectory, TemporaryFile
from typing import Any

from pydantic import BaseModel, ValidationError
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.grader import grader_config
from taskcompendium.models import (
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.datasets.direct_contracts import contract_task
from taskcompendium.pipeline.inputs import RecipeInputs, SourceFiles, SourceFormat, UrlDownload
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    DatasetRecipe,
    HFSource,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
    VerificationReport,
)
from taskcompendium.pipeline.verification import verify_task

DATASET = "open-thought/reasoning-gym"
REVISION = "49b07130b3fcd12f2d064bba7c43869543a0e7e7"
VERIFIER_REVISION = "bb6494e678ffaa1e6bd3967e221d1a67b038757e"
RUBRIC = ReviewRubric(
    id="direct-reasoning-gym-answerability",
    version="1",
    criteria=(
        (
            "Read the complete generated question and verify that every grid, rule, sequence, or example "
            "required to solve it is present."
        ),
        (
            "Independently check the private generated answer against the public problem where feasible; "
            "passing native scoring does not establish correctness."
        ),
        (
            "Preserve the task-native partial reward and answer parsing; a label or approximate substring "
            "match is not a substitute for that scorer."
        ),
        (
            "Judge default generated difficulty and unusual puzzle formats on their actual content, without "
            "assuming they are defects."
        ),
        (
            "Check non-discrimination when a scorer accepts incorrect alternatives, and record concrete "
            "semantic mismatches rather than rejecting an unbound runtime alone."
        ),
    ),
)


def generated_rows(archive_path: StoragePath) -> Iterator[dict[str, Any]]:
    """Run the pinned generator checkout without importing it into the audit worker."""
    with TemporaryDirectory() as directory:
        local_archive = os.path.join(directory, "generator.tar.gz")
        with archive_path.open("rb") as source, open(local_archive, "wb") as destination:
            shutil.copyfileobj(source, destination)
        with tarfile.open(local_archive, mode="r:gz") as archive:
            roots = {member.name.split("/", 1)[0] for member in archive if member.name}
            if len(roots) != 1:
                raise ValueError("Pinned generator archive has multiple roots")
            archive.extractall(directory, filter="data")
        root = os.path.join(directory, roots.pop())
        environment = {
            **os.environ,
            "PYTHONPATH": root + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "MPLCONFIGDIR": directory,
        }
        with TemporaryFile(mode="w+t") as errors:
            process = subprocess.Popen(
                [sys.executable, "-m", "taskcompendium.pipeline.datasets.reasoning_gym.source"],
                stdout=subprocess.PIPE,
                stderr=errors,
                text=True,
                env=environment,
            )
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    yield json.loads(line)
                if process.wait() != 0:
                    errors.seek(0)
                    raise RuntimeError(f"Pinned reasoning-gym generator failed: {errors.read()}")
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()


class RecordedReward(BaseModel):
    candidate: str
    reward: float


class RecordedControls(BaseModel):
    generator_revision: str
    positive: RecordedReward
    negative: RecordedReward
    execution: str


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    try:
        entry = row.data["entry"]
        generation = row.data["generation"]
        task_name = entry["metadata"]["source_dataset"]
        if task_name != generation["task"] or not isinstance(entry["answer"], str):
            raise ValueError("Generated entry and generation provenance must identify the same native task")
        recorded = row.data.get("recorded_pinned_generator_controls")
        controls = RecordedControls.model_validate(recorded) if recorded is not None else None
        contract = {
            "task": task_name,
            "entry": entry,
            "generation": generation,
            "generator_revision": REVISION,
            "answer_extraction": "Text after the last Answer: marker, stripped; otherwise stripped whole response",
            "reward": "float(reasoning_gym.get_score_answer_fn(task)(answer, entry))",
            "recorded_pinned_generator_controls": controls.model_dump(mode="json") if controls is not None else None,
        }
    except (ValidationError, ValueError, KeyError, TypeError) as error:
        return ImportRejection(reason="invalid_generated_reasoning_entry", detail=str(error))
    return contract_task(
        row,
        (TextMessage(role="user", content=entry["question"]),),
        "MarinSkyRL:skyrl_gym.envs.reasoning_gym.scoring.score_response",
        contract,
        (f"Native reasoning-gym scorer at generator revision {REVISION}",),
    )


def verification_report(task: TaskSpec) -> VerificationReport:
    controls = grader_config(task)["contract"].get("recorded_pinned_generator_controls")
    checks = verify_task(task)
    if controls is None:
        return VerificationReport(checks=checks)
    controls = RecordedControls.model_validate(controls)
    if controls.generator_revision != REVISION:
        checks.append(
            CheckResult(
                check="recorded_pinned_generator_controls",
                status=CheckStatus.FAIL,
                detail="Recorded native controls do not identify the pinned generator revision",
            )
        )
        return VerificationReport(checks=checks)
    passed = controls.positive.reward == 1.0 and controls.negative.reward < controls.positive.reward
    checks.append(
        CheckResult(
            check="recorded_pinned_generator_controls",
            status=CheckStatus.PASS if passed else CheckStatus.FAIL,
            detail="Recorded acquisition-time native controls, not current bound runtime: " + controls.model_dump_json(),
        )
    )
    return VerificationReport(checks=checks)


def recipe() -> DatasetRecipe:
    return DatasetRecipe(
        name="reasoning_gym_generated",
        version="reasoning-gym-direct-v1",
        source=HFSource(DATASET, REVISION, "generated", "generated"),
        inputs=RecipeInputs(
            files=SourceFiles(("generator.tar.gz",), SourceFormat.GENERATED, reader=generated_rows),
            downloads=(UrlDownload(f"https://api.github.com/repos/{DATASET}/tarball/{REVISION}", "generator.tar.gz"),),
        ),
        normalize=normalize,
        rubric=RUBRIC,
        intended_use=IntendedUse.TRAIN,
        check_suite=CheckSuite(
            id="recorded-pinned-generator-controls", revision="1", parameters={}, run=verification_report
        ),
    )
