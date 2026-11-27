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
    SamplePanelCurrency,
    SourceUse,
    consumer_blockers,
    consumer_decision_status,
    require_decontamination,
)
from finestore.rl.curation import (
    QualifiedSourceDeclaration as SourceDeclaration,
)
from finestore.rl.licenses import require_license

from .consumer_evidence import CONSUMER_RUNTIME_COMMIT
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


def has_runnable_production_support(
    verdicts: Mapping[str, str], deployment_classes: Mapping[str, str] | None = None
) -> bool:
    """Whether any production backend's verdict satisfies admission.

    Dev-only backends (local Docker, Docker+runsc) are dropped before the
    check: a DEV_ONLY SUPPORTED cell can never satisfy admission.
    """
    classes = dict.fromkeys(
        (*PRODUCTION_BACKENDS_DEFAULT, *DEV_ONLY_BACKENDS_DEFAULT),
        "dev_only",
    )
    classes.update({name: "production" for name in PRODUCTION_BACKENDS_DEFAULT})
    if deployment_classes:
        classes.update(deployment_classes)
    return any(
        support in RUNNABLE_FIT for name, support in verdicts.items() if classes.get(name, "dev_only") == "production"
    )


def backend_admission_review(
    verdicts: Mapping[str, str],
    status: str | None,
    deployment_classes: Mapping[str, str] | None = None,
) -> tuple[str | None, list[str]]:
    """Review one source's backend fit for admission.

    Returns ``(blocker, verification_pending)``. A production ``verify`` cell is
    pending live qualification, not a fit failure: it never blocks, and the
    caller decides the status — an already-admitted source keeps its status with
    the visible marker, a new admission materializes as Pending. Dev-only
    backends (local Docker, Docker+runsc) are dropped before every check: a
    DEV_ONLY SUPPORTED cell can never satisfy admission. ``None`` for every
    non-Available status: exclusion and hold statuses never need a runnable
    backend. ``role_absent`` marks a population that declares no environment
    for a role at all, so a source whose every production cell is
    ``role_absent`` needs no backend. ``not_applicable`` cells are applicability
    outcomes on declared roles and cannot negate a pending qualification, but
    when every production backend is ``not_applicable``/``unsupported`` the
    blocker names them all.
    """
    if status != "Available":
        return None, []
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
    pending = sorted(name for name, support in production_verdicts.items() if support == "verify")
    if any(support in RUNNABLE_FIT for support in production_verdicts.values()):
        return None, pending
    if pending:
        # One production VERIFY cell is enough: not_applicable cells on other
        # production backends are applicability outcomes, not fit failures, and
        # cannot negate a pending live qualification.
        return None, pending
    if production_verdicts and all(support == "role_absent" for support in production_verdicts.values()):
        return None, []
    return (
        "no surveyed production backend can run the source: "
        + ", ".join(f"{name}={support}" for name, support in sorted(verdicts.items())),
        pending,
    )


def _require_declaration_license(declaration: SourceDeclaration) -> None:
    require_unbanned(
        declaration.id,
        declaration.dataset_id,
        declaration.license_receipt,
        declaration.intended_use.value if declaration.intended_use else None,
    )
    if declaration.intended_use is None:
        raise ValueError(f"Admission requires an explicit intended_use: {declaration.id}")
    require_license(
        declaration.license_receipt,
        declaration.id,
        declaration.intended_use.value,
        ban=None,
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
    reasons = consumer_blockers(
        declaration.consumer_receipt,
        declaration.id,
        binding,
        declaration.within_source_dedup.unique_count,
        declaration.consumer_requirements,
        runtime_commit=declaration.consumer_runtime_commit or CONSUMER_RUNTIME_COMMIT,
    )
    if (
        consumer_decision_status(
            declaration.consumer_decision,
            declaration.id,
            binding,
            reasons,
            runtime_commit=declaration.consumer_runtime_commit or CONSUMER_RUNTIME_COMMIT,
        )
        is not None
    ):
        assert declaration.consumer_decision is not None
        raise ValueError(
            f"User-approved consumer decision: {declaration.id}: {declaration.consumer_decision.reason}: "
            f"{declaration.consumer_decision.user_ruling}"
        )
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


def require_available_lexical_clearance(declaration: SourceDeclaration) -> None:
    """Require lexical decontamination evidence before a NEW Available admission.

    Training or unassigned use requires the decontaminated private-release
    receipt bound to the selected pin; evaluation use requires a measured
    benchmark receipt at the same pin — the contract the oracle-filtration
    path already enforces, applied to every newly admitted Available source.
    Callers grandfather sources that were already Available in the previously
    selected catalog.
    """
    if declaration.intended_use is None:
        raise ValueError(f"Admission requires an explicit intended_use: {declaration.id}")
    if declaration.intended_use is SourceUse.TRAIN:
        require_private_release(declaration.lexical_contamination, declaration.dataset_id, declaration.dataset_revision)
        return
    require_lexical_qualification(
        declaration.lexical_contamination,
        declaration.dataset_id,
        declaration.dataset_revision,
        declaration.intended_use,
    )


def require_sample_pin_currency(declaration: SourceDeclaration) -> None:
    """Require the admission SAMPLE panel on the current final pin.

    Filtered private releases need their OWN SAMPLE on the final pin: a panel
    bound through an archived lineage was drawn against an older population
    and cannot qualify a new admission (coordinator ruling, SAMPLE
    pin-currency). Sources already Available in the previously selected
    catalog are grandfathered by the caller with a visible marker.
    """
    qc = declaration.independent_qc
    if qc is not None and qc.panel_currency is SamplePanelCurrency.ARCHIVED_LINEAGE:
        raise ValueError(f"Admission SAMPLE panel belongs to an older population pin: {declaration.id}")


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
