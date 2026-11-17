# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Current-release admission checks shared by curation and the packaged Atlas app.

The packaged app imports this module in Marina's base environment, which has no
``taskcompendium``: keep runtime imports to the bundled models and treat the
backend survey shape by its string labels.
"""

from collections.abc import Mapping

from finestore.rl.curation import (
    ActorCensusInput,
    CurationSummary,
    DeconReceiptStatus,
    GradingEnvironmentEvidence,
    LexicalContaminationEvidence,
    LexicalUse,
    OversightLevel,
    SourceDeclaration,
    SourceUse,
    require_decontamination,
)
from finestore.rl.licenses import require_license

from .licensing import require_unbanned, source_license_ban

MISSING_GRADER_REASON = "No executable grader is declared"

# The survey's aggregate support labels; "supported" and "partial" mean some
# backend can run the source at all. The labels are matched as strings because
# the packaged runtime has no taskcompendium.
RUNNABLE_FIT = frozenset({"supported", "partial"})

# Deployment classes mirrored from shellbox.machine.BACKEND_DEPLOYMENT (marina
# does not depend on shellbox at runtime). Surveys carry their own
# deployment_classes map when the producing code defines one; unknown backends
# fall back to these defaults.
PRODUCTION_BACKENDS_DEFAULT = frozenset({"iris", "daytona", "local"})
DEV_ONLY_BACKENDS_DEFAULT = frozenset({"docker", "docker-gvisor", "qemu", "shellsim"})


def population_admission_blocker(
    binding: ActorCensusInput | None, retained_tasks: int, grading: GradingEnvironmentEvidence | None
) -> str | None:
    if retained_tasks <= 0:
        return "admission requires a nonempty retained population"
    if binding is None or binding.measured_tasks != retained_tasks:
        return "actor census does not cover the retained population"
    if grading is None or grading.assessed_tasks != retained_tasks:
        return "grading census does not cover the complete retained population"
    if any(
        sum(getattr(grading, field).values()) != retained_tasks
        for field in ("required_counts", "declared_counts", "status_counts", "reason_counts")
    ):
        return "grading census counts do not cover the complete retained population"
    if grading.reason_counts.get(MISSING_GRADER_REASON, 0):
        return "admitted source has tasks without runnable graders"
    if missing := grading.missing_artifact_counts:
        return "declared environment artifacts exist in no storage region: " + ", ".join(sorted(missing))
    return None


def backend_support_blocker(
    verdicts: Mapping[str, str],
    status: str | None,
    deployment_classes: Mapping[str, str] | None = None,
) -> str | None:
    """Require one surveyed PRODUCTION backend that can run the source before it is Available.

    ``verdicts`` maps backend name to the survey's aggregate support label for
    one source. ``deployment_classes`` is the survey's per-backend production /
    dev-only map; when absent or partial, the mirrored defaults apply. Dev-only
    backends (local Docker, Docker+runsc) are dropped before the runnable check:
    a DEV_ONLY SUPPORTED cell can never satisfy admission. ``None`` for every
    non-Available status: exclusion and hold statuses never need a runnable
    backend, and a survey row measured before this check may keep its historical
    verdict. ``role_absent`` marks a population that declares no environment for
    a role at all, so a source whose every production cell is ``role_absent``
    needs no backend: that applicability outcome keeps its N/A label instead of
    becoming a failure. ``not_applicable`` and ``unsupported`` are real verdicts
    on declared roles and still block.
    """
    if status != "Available":
        return None
    classes = dict.fromkeys(
        (*PRODUCTION_BACKENDS_DEFAULT, *DEV_ONLY_BACKENDS_DEFAULT),
        "dev_only",
    )
    classes.update({name: "production" for name in PRODUCTION_BACKENDS_DEFAULT})
    if deployment_classes:
        classes.update(deployment_classes)
    production_verdicts = {
        name: support for name, support in verdicts.items() if classes.get(name, "dev_only") == "production"
    }
    if any(support in RUNNABLE_FIT for support in production_verdicts.values()):
        return None
    if production_verdicts and all(support == "role_absent" for support in production_verdicts.values()):
        return None
    return "no surveyed production backend can run the source: " + ", ".join(
        f"{name}={support}" for name, support in sorted(verdicts.items())
    )


def _require_declaration_license(declaration: SourceDeclaration) -> None:
    require_unbanned(declaration.id, declaration.dataset_id, declaration.license_receipt)
    require_license(
        declaration.license_receipt,
        declaration.id,
        declaration.dataset_id,
        declaration.dataset_revision,
        declaration.intended_use.value if declaration.intended_use else "unknown",
    )


def require_declaration_admission(declaration: SourceDeclaration) -> None:
    """Require license permission before admitting a newly published source revision."""
    _require_declaration_license(declaration)
    require_population_admission(declaration)


def require_population_admission(declaration: SourceDeclaration) -> None:
    """Reject incomplete or differently pinned evidence before publication or loading."""
    binding, pipeline = declaration.actor_census_input, declaration.pipeline
    if (
        binding is None
        or pipeline is None
        or (binding.dataset_id, binding.dataset_revision, binding.converter_version)
        != (declaration.dataset_id, declaration.dataset_revision, pipeline.version)
    ):
        raise ValueError(f"Admission census belongs to another current population: {declaration.id}")
    if blocker := population_admission_blocker(
        binding, declaration.within_source_dedup.unique_count, declaration.grading_environment
    ):
        raise ValueError(f"Admission census blocked: {declaration.id}: {blocker}")


def require_content_result(
    declaration: SourceDeclaration, curation: CurationSummary | None, assessment_key: str | None
) -> None:
    """Require a computed Result for reviewed sources before selecting them."""
    if assessment_key is not None or declaration.quality_result is not None:
        return
    controls = curation.verification if curation is not None else None
    campaign_reviewed = (
        curation is not None
        and curation.mode in {"sample", "full"}
        and controls is not None
        and controls.status in {"passed", "validated"}
    )
    oversight_reviewed = declaration.oversight in {OversightLevel.SAMPLE, OversightLevel.FULL}
    # A campaign that reviewed its own sample owes a computed Result even when the source
    # is Excluded or Not selectable; a declared oversight owes one only for Available sources.
    if campaign_reviewed or (declaration.status == "Available" and oversight_reviewed):
        raise ValueError(f"SAMPLE/FULL quality requires a computed Result: {declaration.id}")


def require_private_release(
    evidence: LexicalContaminationEvidence | None, dataset_id: str | None, dataset_revision: str | None
) -> None:
    require_decontamination(evidence, dataset_id, dataset_revision)
    assert evidence is not None and evidence.receipt is not None
    receipt = evidence.receipt
    release = receipt.release
    assert release is not None
    if not release.private or release.format != "taskcompendium" or not release.dataset_id.startswith("open-athena/"):
        raise ValueError("Atlas admission requires a private open-athena TaskCompendium release")
    if receipt.status is DeconReceiptStatus.REMEDIATED:
        assert release.original_private_dataset_id is not None
        if not release.original_private_dataset_id.startswith("open-athena/"):
            raise ValueError("Atlas remediation requires the original private open-athena snapshot")


def require_lexical_qualification(
    evidence: LexicalContaminationEvidence | None,
    dataset_id: str | None,
    dataset_revision: str | None,
    intended_use: SourceUse,
) -> None:
    """Require training remediation or an unchanged evaluation measurement at the current pin."""
    if intended_use is SourceUse.TRAIN:
        require_private_release(evidence, dataset_id, dataset_revision)
        return
    if (
        evidence is None
        or evidence.receipt is None
        or evidence.receipt.use not in {LexicalUse.EVAL, LexicalUse.BENCHMARK}
        or evidence.receipt.status is not DeconReceiptStatus.BENCHMARK_MEASURED
        or (evidence.input_dataset_id, evidence.input_dataset_revision) != (dataset_id, dataset_revision)
    ):
        raise ValueError("Evaluation requires a measured lexical receipt for the current dataset pin")


def require_catalog_declaration(declaration: SourceDeclaration) -> None:
    """Require licenses for metadata and population evidence for Available sources."""
    if (
        source_license_ban(declaration.id, declaration.dataset_id, declaration.license_receipt) is not None
        and declaration.status == "Excluded"
    ):
        return
    _require_declaration_license(declaration)
    if declaration.status == "Available":
        require_population_admission(declaration)
