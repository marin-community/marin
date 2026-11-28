# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The carried-release admission: grading evidence rides verified content identity.

Coordinator ruling 2026-11-25 ~22:20 EST (derived from the user's 2026-11-25 ~20:00
principle that a previously admitted source is re-admitted unless its code is broken):
a re-pinned release whose content is byte-identical to the prior admitted pin carries
the prior pin's retained grading. The carry admits only with exact content identity,
unchanged grader identity and a format-only converter change; a single changed task or
a changed grader refuses.
"""

import json

import pytest
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from pydantic import ValidationError

from experiments.post_training.task_curation.content_identity import (
    SourceTextPopulation,
    artifact_spec_digest,
    carry_task_digests,
    population_task_digests,
    verify_carry_content,
)
from experiments.post_training.task_curation.lexical_contamination import write_text_population
from experiments.post_training.task_curation.remediation_binding import RetainedGradingCarry

pytest.importorskip("taskcompendium")

from taskcompendium.convert.answers import exact_answer_task
from taskcompendium.identity import task_spec_json
from taskcompendium.pipeline.models import RawRow, Source


def task_document(row: int, *, answers: tuple[str, ...] = ("red",)) -> dict:
    source = Source(dataset="fixture", revision="pinned", row=f"fixture/tasks.parquet:{row}", importer_revision="1")
    task = exact_answer_task(
        RawRow(f"fixture-task-{row}", source, {}), prompt=f"Name color {row}", answers=answers, ignore_case=False
    )
    return json.loads(task_spec_json(task))


def stage_population(
    tmp_path, name: str, documents: list[dict], digest_override: str | None = None
) -> SourceTextPopulation:
    output = tmp_path / name
    files = write_text_population(((document, {}) for document in documents), output_path=str(output))
    population = SourceTextPopulation(
        path=str(output),
        source_id="TaskTrove:test__carry-cohort",
        dataset_id="open-athena/carry-fixture",
        dataset_revision="b" * 40,
        source_version="harbor-cohort-taskspec-v1",
        converter_revision="c" * 40,
        text_renderer_version="taskcompendium-format-conversation-v1",
        task_count=files.task_count,
        text_sha256=files.text_sha256,
        task_specs_sha256=digest_override or files.task_specs_sha256,
        main_output_dir=files.main_output_dir,
        specs_output_dir=files.specs_output_dir,
        converter_rejections={},
        admission_rejections={},
        within_source_dedup={
            "raw_count": files.raw_task_count,
            "unique_count": files.task_count,
            "within_source_duplicates": files.within_source_duplicates,
            "drop_manifest_uri": files.drop_manifest_uri,
        },
    )
    payload_fields = {field for field in SourceTextPopulation.model_fields if field != "path"}
    write_record(
        ArtifactRecord(
            output_path=str(output),
            result_type=result_type_name(SourceTextPopulation),
            result=population.model_dump(mode="json", include=payload_fields),
        )
    )
    return population


def test_carry_admits_identical_content(tmp_path):
    """Byte-identical content carries: the identity digest is exact and both populations
    stay self-consistent with their declared digests. The prior pin's population and the
    re-pinned one hold the same task bytes; only the artifact layout differs."""
    documents = [task_document(row) for row in range(3)]
    prior = stage_population(tmp_path, "prior", documents)
    reissued = stage_population(tmp_path, "reissued", documents)
    identity = verify_carry_content(prior, reissued)
    assert len(identity) == 64
    assert sorted(carry_task_digests(prior).values()) == sorted(carry_task_digests(reissued).values())


def test_carry_refuses_a_single_changed_task(tmp_path):
    documents = [task_document(row) for row in range(3)]
    prior = stage_population(tmp_path, "prior", documents)
    changed = [task_document(row) for row in range(2)] + [task_document(7)]
    reissued = stage_population(tmp_path, "reissued", changed)
    with pytest.raises(ValueError, match=r"Carry content identity refused"):
        verify_carry_content(prior, reissued)


def test_carry_refuses_a_changed_grader(tmp_path):
    """A grader change (different accepted answers) changes the task bytes; the grader
    digests expose it independently of the byte check, and the carry refuses."""
    documents = [task_document(row) for row in range(3)]
    prior = stage_population(tmp_path, "prior", documents)
    changed = [task_document(row) for row in range(2)] + [task_document(2, answers=("blue",))]
    reissued = stage_population(tmp_path, "reissued", changed)
    assert sorted(carry_task_digests(prior).values()) != sorted(carry_task_digests(reissued).values())
    with pytest.raises(ValueError, match=r"Carry grader identity refused|Carry content identity refused"):
        verify_carry_content(prior, reissued)


def test_carry_detects_a_changed_grader_through_its_own_digests(tmp_path):
    """The grader check runs on the parsed structure: the grader digests move when the
    grader changes, independently of the task-byte digests."""
    prior = stage_population(tmp_path, "prior", [task_document(0)])
    reissued = stage_population(tmp_path, "reissued", [task_document(0, answers=("blue",))])
    prior_bytes = sorted(population_task_digests(prior).values())
    reissued_bytes = sorted(population_task_digests(reissued).values())
    assert prior_bytes != reissued_bytes
    assert sorted(carry_task_digests(prior).values()) != sorted(carry_task_digests(reissued).values())


def test_carry_refuses_a_lying_declared_digest(tmp_path):
    """A population whose record declares a digest its files do not hash to refuses."""
    prior = stage_population(tmp_path, "prior", [task_document(0)])
    reissued = stage_population(tmp_path, "reissued", [task_document(0)], digest_override="a" * 64)
    assert artifact_spec_digest(reissued) != reissued.task_specs_sha256
    with pytest.raises(ValueError, match=r"declares"):
        verify_carry_content(prior, reissued)


def test_carry_model_requires_the_admission_conditions():
    """The receipt model fails closed: a carry claiming a changed grader or a semantic
    converter change never validates."""
    with pytest.raises(ValidationError, match=r"format-only converter change"):
        RetainedGradingCarry(
            carried_from_repo="open-athena/prior",
            carried_from_revision="d" * 40,
            identity_proof_sha256="a" * 64,
            grader_identity_unchanged=False,
            converter_format_only=True,
            tasks=3,
        )
    with pytest.raises(ValidationError, match=r"format-only converter change"):
        RetainedGradingCarry(
            carried_from_repo="open-athena/prior",
            carried_from_revision="d" * 40,
            identity_proof_sha256="a" * 64,
            grader_identity_unchanged=True,
            converter_format_only=False,
            tasks=3,
        )
