# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow.parquet as pq
import pytest
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PackageSpec,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)

from experiments.post_training.task_curation import forge_release
from experiments.post_training.task_curation.forge_release import Choice

PASS = {"status": "graded", "reward": 1.0, "passed": None}
FAIL = {"status": "graded", "reward": 0.0, "passed": None}


SHELL_SIMULATOR = EnvironmentRequirements(
    capabilities=("shell",), command_semantics=CommandSemantics.SHELL_SIMULATOR, working_directory="/app"
)
GRADER_BASE = EnvironmentRequirements(
    capabilities=("shell",),
    command_semantics=CommandSemantics.LINUX_PROCESS,
    docker_image=forge_release.GRADER_BASE_IMAGE,
    working_directory="/app",
)
# How forge's local sandbox records a grader-base machine: no image, an empty package spec.
HOSTED = GRADER_BASE.model_copy(update={"docker_image": None, "package_spec": PackageSpec(pypi_packages=())})


def _task(identity: str, requirements: EnvironmentRequirements = SHELL_SIMULATOR) -> dict:
    return TaskSpec(
        id=identity,
        context=ConversationInput(events=(TextMessage(role="user", content="Answer the question."),)),
        environment_requirements=requirements,
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="fixture"),
        source=Source(dataset="capability", revision="r", row=identity, importer_revision="1"),
    ).model_dump(mode="json")


def _record(
    identity: str,
    passes: int,
    *,
    band: str = "in_band",
    notes: tuple = (),
    revert: bool = False,
    authored: EnvironmentRequirements = SHELL_SIMULATOR,
    hosted: EnvironmentRequirements = SHELL_SIMULATOR,
) -> dict:
    capability = identity.rsplit("/", 1)[0]
    trials = [{"trial": i, "grade": PASS if i < passes else FAIL, "cause": None} for i in range(8)]
    decisions = [{"kind": "accept", "band": band, "revert": revert}]
    return {
        "schema": forge_release.RECORD_SCHEMA,
        "id": identity,
        "capability": capability,
        "arm": "b4",
        "workpiece": {
            "terminal": "accepted",
            "notes": list(notes),
            "missing_evidence": [],
            "proposal": {"header": {"verification": "code", "environment": "shellsim"}},
            "submission": {"controls": [{}] * 4, "verification": None, "task": _task(identity, authored)},
            "evidence": [{"confirmation": True, "solver": trials}],
            "decisions": decisions,
        },
        "rollout_spec": {
            "task": _task(identity, hosted),
            "task_machine": {"cpus": None, "memory_mb": 8192, "storage_mb": None, "gpus": 0},
            "verifier_machine": None,
        },
    }


def test_selection_covers_every_capability_before_a_second_task():
    records = [_record("d01.a/01", 4), _record("d01.a/02", 4), _record("d02.b/01", 0, band="too_hard")]
    selection = forge_release.select([forge_release.task_summary(r) for r in records], 2)
    chosen = sorted(item.summary.capability for item in selection if item.choice is Choice.SELECTED)
    # The too-hard task is its capability's only one, so it outranks the first capability's second in-band task.
    assert chosen == ["d01.a", "d02.b"]


def test_tasks_with_notes_are_never_selected():
    records = [_record("d01.a/01", 4, notes=({"part": "grader"},)), _record("d01.a/02", 8, band="too_easy")]
    selection = forge_release.select([forge_release.task_summary(r) for r in records], 1)
    by_id = {item.summary.id: item.choice for item in selection}
    assert by_id == {"d01.a/01": Choice.NOT_CLEAN, "d01.a/02": Choice.SELECTED}


def test_selection_prefers_in_band_tasks_within_a_capability():
    records = [_record("d01.a/01", 4), _record("d01.a/02", 8, band="too_easy")]
    selection = forge_release.select([forge_release.task_summary(r) for r in records], 1)
    assert {item.summary.id for item in selection if item.choice is Choice.SELECTED} == {"d01.a/01"}


def test_release_writes_selected_rows_and_selection_evidence(tmp_path):
    records = [_record("d01.a/01", 4), _record("d02.b/01", 3), _record("d02.b/02", 2)]
    manifest = forge_release.write_release(records, 2, tmp_path, "memory://records")
    rows = pq.read_table(tmp_path / forge_release.LOWERED_FILENAME).to_pylist()
    assert manifest["selected"] == len(rows) == 2
    assert {row["path"] for row in rows} == {"d01.a/01", "d02.b/01"}
    lines = (tmp_path / forge_release.SELECTION_FILENAME).read_text().splitlines()
    assert {json.loads(line)["choice"] for line in lines} == {"selected", "below_coverage_tier"}


def test_a_locally_sandboxed_machine_is_restored_to_the_image_its_author_named():
    record = _record("d01.a/01", 4, authored=GRADER_BASE, hosted=HOSTED)
    lowered = forge_release.lowered_spec(record)
    requirements = lowered.task.environment_requirements
    assert (requirements.docker_image, requirements.package_spec) == (forge_release.GRADER_BASE_IMAGE, None)
    assert forge_release.task_summary(record).environment == "linux"


def test_a_restored_machine_that_disagrees_with_its_author_is_refused():
    record = _record("d01.a/01", 4, authored=SHELL_SIMULATOR, hosted=HOSTED)
    with pytest.raises(ValueError, match="restored environment_requirements"):
        forge_release.lowered_spec(record)
