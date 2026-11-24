# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Agentic sources are admitted only in an accepted task format."""

from dataclasses import replace
from pathlib import Path

import pytest
from finestore.rl.curation import DatasetRelease, OversightEvidence, OversightLevel
from finestore.rl.status_taxonomy import (
    REASON_EXPLANATIONS,
    DataRestriction,
    ExclusionEvidenceKind,
    ExclusionReason,
    SourceStatus,
)
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation import sources as source_catalog
from experiments.post_training.task_curation.datasets import native_releases
from experiments.post_training.task_curation.datasets.litecoder import tasks as litecoder
from experiments.post_training.task_curation.datasets.mcp_atlas_easy import tasks as mcp_atlas_easy
from experiments.post_training.task_curation.datasets.mimo import tasks as mimo
from experiments.post_training.task_curation.datasets.terminal_lego import tasks as terminal_lego
from experiments.post_training.task_curation.prebuilt import PrebuiltCohort
from experiments.post_training.task_curation.retained_release import RetainedCohort
from experiments.post_training.task_curation.source import (
    RlDataSource,
    SourceInfo,
    SourceOrigin,
    SourceReference,
    TaskFormat,
)
from experiments.post_training.task_curation.sources import all_sources, require_accepted_task_format, runnable_sources
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source

FIXTURE_COHORT_REPO = "open-athena/fixture-tasks"
FIXTURE_COHORT_REVISION = "a" * 40


def _inject_offline_prebuilt_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve consistent fixture qualification documents for the prebuilt catalog reads.

    ``configured_sources`` validates each prebuilt dataset's pinned qualification
    document against its declared release. Offline unit tests cannot read the
    private registry archive, so each dataset module's registry read returns a
    consistent fixture document; the download boundary refuses any network read.
    """
    release = DatasetRelease(
        repo=FIXTURE_COHORT_REPO,
        revision=FIXTURE_COHORT_REVISION,
        url=f"https://huggingface.co/datasets/{FIXTURE_COHORT_REPO}/tree/{FIXTURE_COHORT_REVISION}",
    )

    def fixture_info(prefix: str) -> SourceInfo:
        return SourceInfo(
            origin=SourceOrigin.MARIN,
            upstream_dataset=f"fixture/{prefix}",
            upstream_name=f"{prefix} fixture",
            build_path="fixture prebuilt qualification",
            current_release=release,
        )

    def admitted_document(prefix: str) -> dict:
        info = fixture_info(prefix)
        config = PrebuiltCohort(
            name="fixture",
            version="fixture-1",
            repo=release.repo,
            revision=release.revision,
            expected_rows=1,
            intended_use=IntendedUse.TRAIN,
        )
        return {
            "source_id": info.id,
            "original_info": info,
            "config": config,
            "admission": {"filtration": {"original_count": config.expected_rows}},
        }

    def eval_document(prefix: str) -> dict:
        info = fixture_info(prefix)
        config = PrebuiltCohort(
            name="fixture",
            version="fixture-1",
            repo=release.repo,
            revision=release.revision,
            expected_rows=1,
            intended_use=IntendedUse.EVAL,
        )
        return {"source_id": info.id, "info": info, "cohort": config}

    mimo_info = fixture_info("mimo")
    mimo_cohort = RetainedCohort(
        name="fixture",
        version="fixture-1",
        release=release,
        original_dataset=SourceReference(release.repo, release.revision, release.url),
        expected_rows=1,
        tasks_parquet_sha256="a" * 64,
        intended_use=IntendedUse.TRAIN,
        evidence_url=f"{release.url}/evidence",
    )
    monkeypatch.setattr(litecoder, "read_registry_json", lambda path: admitted_document("litecoder"))
    monkeypatch.setattr(terminal_lego, "read_registry_json", lambda path: admitted_document("terminal-lego"))
    monkeypatch.setattr(mcp_atlas_easy, "read_registry_json", lambda path: eval_document("mcp-atlas-easy"))
    monkeypatch.setattr(
        mimo,
        "read_registry_json",
        lambda path: {"source_id": mimo_info.id, "info": mimo_info, "cohort": mimo_cohort},
    )


@pytest.mark.parametrize(
    "source_id,dataset",
    [
        ("marcusm117:ProofNet-Verified", "marcusm117/ProofNet-Verified"),
        ("project-numina:NuminaMath-LEAN", "project-numina/NuminaMath-LEAN"),
    ],
)
def test_new_lean_sources_wait_for_measurement_before_entering_the_atlas_catalog(monkeypatch, source_id, dataset):
    monkeypatch.setattr(native_releases, "read_registry_json", lambda path: {"entries": []})
    _inject_offline_prebuilt_registries(monkeypatch)
    pending = {source.info.id: source for source in source_catalog.PENDING_MEASUREMENT_SOURCES}
    source = pending[source_id]
    assert source.dataset.name == dataset
    assert source.info.actor_census_input is None
    assert source.info.within_source_dedup is None
    assert source.info.lexical_contamination is None
    assert source.info.id not in {source.info.id for source in source_catalog.configured_sources()}


def _agentic(task_format: TaskFormat | None, source_id: str = "Marin:agentic") -> RlDataSource:
    return RlDataSource(
        info=SourceInfo(
            data_restriction=DataRestriction.OPEN,
            id=source_id,
            upstream_dataset=(source_id).partition(":")[2],
            upstream_name=(source_id).partition(":")[2],
            build_path="test",
            origin=SourceOrigin.MARIN,
            tags=("agentic",),
            task_format=task_format,
        )
    )


@pytest.mark.parametrize(
    "task_format, source_id, error",
    [(None, "Marin:agentic", "declares no task format")],
)
def test_unaccepted_agentic_formats_are_rejected(task_format, source_id, error):
    with pytest.raises(ValueError, match=error):
        require_accepted_task_format(_agentic(task_format, source_id))


def _admitted_fixture(tmp_path: Path) -> RlDataSource:
    from experiments.post_training.task_curation.tests.admission_fixtures import with_admission_census  # noqa: PLC0415
    from experiments.post_training.task_curation.tests.numbers_pipeline import number_source  # noqa: PLC0415

    source = number_source(tmp_path / "input")
    source = replace(
        source,
        info=replace(
            source.info,
            oversight=OversightEvidence(
                level=OversightLevel.SAMPLE,
                campaign="fixture",
                verification="Fixture ingestion and grader controls executed.",
            ),
        ),
    )
    return with_admission_census(source)


def test_harbor_source_is_refused_without_a_taskspec_conversion_receipt(tmp_path):
    """TaskSpec-only admission: a Harbor population fails closed, never admits.

    marin #12167 removes the Harbor runtime from SkyRL; RL serves only
    executable TaskSpecs through RolloutEngine. A source whose declared shape
    is not TaskSpec has no verified TaskSpec conversion receipt for its
    admitted revision, so admission refuses it with NOT_TASKSPEC_CONVERTED
    even when every other admission receipt is complete. Declaring the
    TaskSpec shape (the conversion census then binds the admitted pin) is what
    clears it.
    """
    source = _admitted_fixture(tmp_path)
    assert source.status is SourceStatus.AVAILABLE
    harbor = replace(source, info=replace(source.info, task_format=TaskFormat.HARBOR))
    blocker = "admission requires a verified TaskSpec conversion receipt for the admitted revision"
    assert blocker in harbor.admission_blockers
    claims = {claim.reason for claim in harbor.admission_claims}
    assert ExclusionReason.NOT_TASKSPEC_CONVERTED in claims
    claim = next(c for c in harbor.admission_claims if c.reason is ExclusionReason.NOT_TASKSPEC_CONVERTED)
    assert claim.evidence.kind is ExclusionEvidenceKind.MISSING_ARTIFACT
    assert "harbor" in claim.evidence.expected
    assert harbor.status is SourceStatus.EXCLUDED


def test_not_taskspec_converted_explanation_requires_a_verified_conversion_receipt():
    explanation = REASON_EXPLANATIONS[ExclusionReason.NOT_TASKSPEC_CONVERTED]
    assert "verified TaskSpec conversion receipt" in explanation.meaning
    assert "admitted revision" in explanation.clears


@pytest.mark.data_integration
def test_every_catalog_declaration_resolves_to_exactly_one_task_format():
    """The served catalog only carries declarations whose task layout resolves to one enum value."""
    for source in all_sources().values():
        declaration = source.catalog_declaration
        if declaration is None:
            continue
        assert isinstance(declaration.task_format, TaskFormat)


def test_pipeline_less_pointers_are_rejected():
    with pytest.raises(ValueError, match="no registered conversion pipeline"):
        _ = _agentic(TaskFormat.TASKSPEC).declaration


@pytest.mark.parametrize("missing", ["count", "basis", "oversight", "verification"])
def test_unqualified_agentic_pipeline_stays_visible_but_cannot_be_selected(monkeypatch, missing):
    info = replace(
        _agentic(TaskFormat.TASKSPEC).info,
        within_source_dedup=number_source(Path("unused")).info.within_source_dedup,
        daytona_snapshot_count=1,
        daytona_machine_slots=1,
        daytona_execution_failures=0,
        daytona_count_basis="production snapshot identities over pinned tasks",
        oversight=OversightEvidence(
            level=OversightLevel.SAMPLE, campaign="fixture", verification="20/20 negative controls"
        ),
    )
    if missing == "count":
        info = replace(info, daytona_snapshot_count=None)
    elif missing == "basis":
        info = replace(info, daytona_count_basis=None)
    elif missing == "oversight":
        info = replace(info, oversight=None)
    else:
        info = replace(
            info, oversight=OversightEvidence(level=OversightLevel.SAMPLE, campaign="fixture", verification="")
        )
    source = replace(number_source(Path("unused")), info=info)
    monkeypatch.setattr("experiments.post_training.task_curation.sources.all_sources", lambda: {info.id: source})
    if missing in {"count", "basis"}:
        with pytest.raises(ValueError, match="actor safety measurement"):
            _ = source.declaration
        return
    assert source.name not in runnable_sources()
    assert source.declaration.status == "Excluded"


@pytest.mark.data_integration
def test_every_registered_source_declares_catalog_metadata():
    """Missing census evidence remains pending when catalog metadata is declared."""
    for source in all_sources().values():
        declaration = source.declaration
        assert declaration.id == source.info.id


@pytest.mark.data_integration
def test_nemotron_gym_math_v5_is_deprecated_in_favor_of_the_qa_repair():
    """The user ruled math-v5 Deprecated in favor of the repaired math_gym-qa pin.

    Deprecated stays trainable on explicit request: the tag carries no exclusion
    claim, so the row resolves to SourceStatus.DEPRECATED and the default
    training selections skip it.
    """
    source = all_sources()["TaskTrove:laion__nemotron-gym-math-v5"]
    assert source.status is SourceStatus.DEPRECATED
    assert "deprecated" in source.info.tags
    assert source.exclusion_claims == ()
    assert "TaskTrove:tasktrove-math_gym-qa" in source.info.notes
