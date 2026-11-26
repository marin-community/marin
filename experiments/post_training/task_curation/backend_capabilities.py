# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Survey backend validation and join a complete converted-spec census for Atlas.

This command never creates machines, snapshots, or provider clients. Inputs are
an explicit local JSON manifest; stage object-store shards and rewrite their
paths with stage_survey_manifest on a worker near their storage region.
"""

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from finestore.rl.curation import CurationCatalog
from finestore.rl.status_taxonomy import BackendBlockerReason, backend_blocker_reason_slugs
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rolloutengine.machines import _machine_spec
from rolloutengine.models import MachineRuntimeSpec
from shellbox.backends.local.disk import LocalDiskPool
from shellbox.image import DockerfileSource
from shellbox.machine import (
    BACKEND_DEPLOYMENT,
    DeploymentClass,
    HostImage,
    PackageSpec,
    RegistryImage,
    UnsupportedMachineSpec,
)
from shellbox.requirements import Requirement, Support
from shellbox.survey import CapabilityCell, Checker, capability_matrix, production_checkers
from taskcompendium.backend_requirements import (
    BackendFit,
    BackendSourceMap,
    BackendSurvey,
    FitSupport,
    RuntimeProfile,
    backend_schema_sha256,
    execution_resource_shape,
    task_demands,
)
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import EnvironmentRequirements
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.catalog_identity import canonical_backend_survey, canonical_snapshot_source
from experiments.post_training.task_curation.toolchain_binding import resolved_toolchain_image
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id, canonical_source_map

SUPPORT_ORDER = (Support.SUPPORTED, Support.VERIFY, Support.PARTIAL, Support.UNSUPPORTED)
APPLICABLE_ORDER = (FitSupport.SUPPORTED, FitSupport.VERIFY, FitSupport.PARTIAL, FitSupport.UNSUPPORTED)
RESOURCE_FIELDS = ("cpus", "memory_mb", "storage_mb", "gpus")


def environment_kind(source) -> Requirement:
    """The environment kind a converted role resolves to, named by its machine source.

    The matrix cell keyed by this requirement records each validator's acceptance of a
    minimal probe of the kind, so backend applicability is read from the same measured
    cells and cannot drift from the validators.
    """
    if isinstance(source, HostImage):
        return Requirement.HOST_IMAGE
    if isinstance(source, PackageSpec):
        return Requirement.PACKAGE_SPEC
    if isinstance(source, RegistryImage):
        return Requirement.DOCKER_IMAGE
    if isinstance(source, DockerfileSource):
        return Requirement.DOCKER_BUILD
    raise ValueError(
        f"Converted environments resolve to RegistryImage, HostImage or PackageSpec, not {type(source).__name__}"
    )


def production_backends(survey: BackendSurvey) -> dict[str, BackendFit]:
    """The per-source fit restricted to production deployment targets.

    Admission reads this view: a DEV_ONLY backend (local Docker, Docker+runsc)
    can never satisfy admission no matter what its cells say.
    """
    classes = survey.deployment_classes or {
        backend.value: deployment_class.value for backend, deployment_class in BACKEND_DEPLOYMENT.items()
    }
    production = {}
    for source in survey.sources:
        for backend, fit in source.backends.items():
            if classes.get(backend) == DeploymentClass.PRODUCTION.value:
                production[backend] = fit
    return production


def combined_fit(values: Iterable[FitSupport]) -> FitSupport:
    """The worst applicable outcome; applicability outcomes never dominate a real verdict."""
    applicable = [value for value in values if value in APPLICABLE_ORDER]
    if applicable:
        return max(applicable, key=APPLICABLE_ORDER.index)
    return FitSupport.NOT_APPLICABLE if FitSupport.NOT_APPLICABLE in values else FitSupport.ROLE_ABSENT


LOCAL_ORCHESTRATED_REQUIREMENTS = frozenset(
    {
        Requirement.SETUP,
        Requirement.FILE_UPLOAD,
        Requirement.COMMAND_TIMEOUT,
        Requirement.FILE_MODE,
        Requirement.FILE_MTIME,
    }
)
LOCAL_RESOURCE_REQUIREMENTS = frozenset({Requirement.CPUS, Requirement.MEMORY, Requirement.STORAGE})


def production_matrix(checkers: dict[str, Checker]) -> list[CapabilityCell]:
    """Combine factory evidence with production setup, startup and cleanup orchestration."""
    orchestrated = {
        Requirement.CLEANUP_TIMEOUT: (
            "lib/rolloutengine/src/rolloutengine/cleanup.py",
            "Rolloutengine bounds cleanup independently of factory creation.",
        ),
        Requirement.STARTUP_TIMEOUT: (
            "lib/rolloutengine/src/rolloutengine/machines.py:_prepare_machine",
            "Rolloutengine bounds machine startup with the runtime startup deadline; "
            "Daytona also enforces it at creation.",
        ),
    }
    local_evidence = getattr(checkers.get("local"), "keywords", {}).get("resource_evidence")
    cells = []
    for cell in capability_matrix(checkers):
        upgrade = orchestrated.get(cell.requirement)
        if upgrade is not None and cell.support != Support.UNSUPPORTED:
            evidence, note = upgrade
            cells.append(
                cell.model_copy(
                    update={"support": Support.SUPPORTED, "evidence": (*cell.evidence, evidence), "note": note}
                )
            )
        elif cell.backend == "local" and cell.requirement in LOCAL_ORCHESTRATED_REQUIREMENTS:
            cells.append(
                cell.model_copy(
                    update={
                        "support": Support.SUPPORTED,
                        "evidence": (*cell.evidence, "lib/rolloutengine/src/rolloutengine/machines.py:_prepare_machine"),
                        "note": (
                            "Production LOCAL grading trials run each declared setup command via sh -c inside "
                            "the startup deadline (failing the task on failure), upload grader inputs, enforce "
                            "command deadlines, and the pax tar transfer restores permission bits and "
                            "sub-second modification times."
                        ),
                    }
                )
            )
        elif cell.requirement == Requirement.SETUP:
            cells.append(
                cell.model_copy(
                    update={
                        "evidence": (
                            *cell.evidence,
                            "lib/rolloutengine/src/rolloutengine/machines.py:_prepare_machine",
                        ),
                        "note": (
                            "Rolloutengine runs setup as root through sh; "
                            "the image or simulator must support each command."
                        ),
                    }
                )
            )
        elif cell.backend == "local" and cell.requirement in LOCAL_RESOURCE_REQUIREMENTS:
            verified = (
                cell.requirement == Requirement.STORAGE
                and local_evidence is not None
                and local_evidence.storage_enforced
            ) or (
                cell.requirement in {Requirement.CPUS, Requirement.MEMORY}
                and local_evidence is not None
                and local_evidence.cgroup_verified
            )
            if not verified:
                cells.append(cell)
                continue
            if cell.requirement == Requirement.STORAGE:
                note = (
                    "LOCAL storage enforcement verified on this worker: the disk pool's construction probe "
                    "returned ENOSPC on a fixed-size ext4 loop filesystem and the pool holds free capacity "
                    "for the probe size; production allocations provision exactly the declared storage_mb."
                )
            else:
                note = (
                    f"LOCAL {'cpu' if cell.requirement == Requirement.CPUS else 'memory'} enforcement verified on "
                    "this worker: a probe machine cgroup applied and read back the limit in the delegated "
                    "cgroup v2 hierarchy production machine creation uses."
                )
            cells.append(
                cell.model_copy(
                    update={
                        "support": Support.SUPPORTED,
                        "evidence": (*cell.evidence, "lib/shellbox/src/shellbox/survey.py:local_resource_evidence"),
                        "note": note,
                    }
                )
            )
        else:
            cells.append(cell)
    return cells


def combined_support(values: Iterable[Support]) -> Support:
    return max(values, key=SUPPORT_ORDER.index, default=Support.SUPPORTED)


_DELIVERY_CONTRACT_KINDS = frozenset({"root-delivery", "multi-swe-delivery"})
_SHIPPED_CONTEXT_CACHE: dict[str, DockerfileSource] = {}


def _contract_covered(contract: dict | None) -> bool:
    """Whether the task carries a matched delivery contract (ruled kinds only)."""
    return isinstance(contract, dict) and contract.get("id") in _DELIVERY_CONTRACT_KINDS


def _shipped_build_source(requirements: EnvironmentRequirements) -> DockerfileSource:
    """Materialize the task's shipped Docker build context for the survey.

    A delivery contract ships its Docker build context declaratively (the
    adapter accepts DockerBuildExecution.DECLARATIVE under the contract): the
    shared machine builds it from the shipped files at runtime, so the survey
    measures the backend's build capability against the shipped recipe.
    """
    context = requirements.docker_build
    assert context is not None
    key = canonical_sha256([resource.model_dump(mode="json") for resource in context.files])
    source = _SHIPPED_CONTEXT_CACHE.get(key)
    if source is not None:
        return source
    root = Path(tempfile.mkdtemp(prefix="atlas-contract-build-"))
    for resource in context.files:
        target = root / resource.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resource_bytes(resource))
    dockerfile = root / "Dockerfile"
    if not dockerfile.is_file():
        raise UnsupportedMachineSpec("The shipped Docker build context carries no Dockerfile")
    source = DockerfileSource(context=root, dockerfile=dockerfile)
    _SHIPPED_CONTEXT_CACHE[key] = source
    return source


def source_map(
    source_id: str,
    documents: Iterable[Mapping],
    matrix: list[CapabilityCell],
    checkers: dict[str, Checker],
    inputs: dict[str, str],
    runtime_defaults: dict[str, RuntimeProfile],
    grading_environment: dict | None = None,
    integrity_contracts: dict | None = None,
) -> BackendSourceMap:
    """Count every converted task, validating combined demands in each machine slot.

    A role is measured only where it is a candidate: roles the population does
    not declare are role_absent, and roles whose environment kind the backend
    cannot host are not_applicable with their exclusion counted. Neither is a
    failure; genuine demand failures stay typed on candidate backends.
    """
    cells = {(cell.backend, cell.requirement): cell for cell in matrix}
    demand_counts = defaultdict(Counter)
    ranges = defaultdict(dict)
    task_counts = defaultdict(Counter)
    role_counts = {backend: defaultdict(Counter) for backend in checkers}
    blockers = defaultdict(Counter)
    blocker_reasons = defaultdict(dict)
    exclusions = defaultdict(Counter)
    errors = defaultdict(Counter)
    assignments = defaultdict(Counter)
    # Role-scoped delivery-contract handling (coordinator ruling): a task
    # whose role carries a complete matched delivery contract does not need a
    # resolved docker_build for that role — the contract ships the build
    # context declaratively and the shared agent machine builds it at
    # runtime. Coverage reads the source's contract census (zero extra
    # streaming): absent_contract_tasks == 0 with a named matched group means
    # every task is contracted; a partial or unnamed contract fails closed.
    census = integrity_contracts or {}
    contracts_complete = census.get("absent_contract_tasks") == 0 and any(
        group.get("id") in _DELIVERY_CONTRACT_KINDS and group.get("present") for group in census.get("groups") or []
    )
    count = 0
    for document in documents:
        count += 1
        task = document.get("task", document)
        published_runtime = document.get("runtime") or {}
        runtime = {
            slot: {
                **runtime_defaults[role].model_dump(mode="json"),
                **execution_resource_shape(task, published_runtime.get(slot), role),
            }
            for role, slot in (("agent", "task_machine"), ("grader", "verifier_machine"))
        }
        demands, failures = task_demands({"task": task, "runtime": runtime})
        for role, needs in demands.items():
            declared = task.get("environment_requirements") if role == "agent" else task["grader"]["environment"]
            assigned = declared.get("required_backend") or (
                "local" if role == "grader" and declared.get("packages_lock") else "daytona"
            )
            assignments[role][assigned] += 1
            demand_counts[role].update(needs)
            shape = runtime.get("task_machine" if role == "agent" else "verifier_machine")
            if shape:
                for name in RESOURCE_FIELDS:
                    value = shape.get(name)
                    if value is not None:
                        low, high = ranges[role].get(name, (value, value))
                        ranges[role][name] = (min(low, value), max(high, value))
        # A lock-only role binds its composed toolchain image exactly as the serving wave
        # does (lock_satisfied_by_image, 047bc68189): the stored HostImage source is the
        # freeze-time fallback, and the binding makes image backends (Daytona) candidates
        # while the role's disk quota rides the machine spec's own storage sizing. The
        # task bytes, requirements and grader semantics never change; only the machine
        # source the role runs from does. The reported demands stay the declaration's.
        bound_images: dict[str, str] = {}
        for role in demands:
            declared = task.get("environment_requirements") if role == "agent" else task["grader"]["environment"]
            requirements = EnvironmentRequirements.model_validate(declared)
            if (
                requirements.packages_lock is not None
                and requirements.docker_image is None
                and requirements.docker_build is None
            ):
                image = resolved_toolchain_image(requirements.packages_lock)
                if image is not None:
                    bound_images[role] = image
        for backend, check in checkers.items():
            outcomes: list[FitSupport] = []
            for role in ("agent", "grader"):
                if role not in demands:
                    role_counts[backend][role][FitSupport.ROLE_ABSENT] += 1
                    outcomes.append(FitSupport.ROLE_ABSENT)
                    continue
                if role in failures:
                    error = failures[role]
                    # An unclassified capability is a survey-vocabulary gap,
                    # not a demonstrated backend gap: the cells stay VERIFY
                    # until the vocabulary classifies the capability or a live
                    # probe resolves the role (VERIFY ruling).
                    role_counts[backend][role][FitSupport.VERIFY] += 1
                    for capability in error.capabilities:
                        blockers[backend][f"{role}:{capability}"] += 1
                        blocker_reasons[backend][
                            f"{role}:{capability}"
                        ] = BackendBlockerReason.UNCLASSIFIED_CAPABILITY.value
                    errors[backend][str(error)] += 1
                    outcomes.append(FitSupport.VERIFY)
                    continue
                declared = task.get("environment_requirements") if role == "agent" else task["grader"]["environment"]
                shape = runtime.get("task_machine" if role == "agent" else "verifier_machine")
                requirements = EnvironmentRequirements.model_validate(declared)
                if requirements.required_backend is not None and backend != requirements.required_backend.value:
                    role_counts[backend][role][FitSupport.UNSUPPORTED] += 1
                    blockers[backend][f"{role}:required_backend"] += 1
                    blocker_reasons[backend][f"{role}:required_backend"] = BackendBlockerReason.REQUIRED_BACKEND.value
                    errors[backend][f"{role}: task requires {requirements.required_backend.value}"] += 1
                    outcomes.append(FitSupport.UNSUPPORTED)
                    continue
                settings = MachineRuntimeSpec.model_validate({**shape, "backend": backend})
                try:
                    contract = task.get("integrity_contract")
                    if contracts_complete and _contract_covered(contract) and requirements.docker_build is not None:
                        # the contract ships the build context declaratively: the
                        # shared agent machine builds it at runtime, so the cell
                        # measures the backend's build capability
                        spec = _machine_spec(
                            requirements, settings, docker_build_source=_shipped_build_source(requirements)
                        )
                    else:
                        spec = _machine_spec(requirements, settings)
                except UnsupportedMachineSpec as error:
                    role_counts[backend][role][FitSupport.UNSUPPORTED] += 1
                    errors[backend][f"{role}: {error}"] += 1
                    outcomes.append(FitSupport.UNSUPPORTED)
                    continue
                bound = bound_images.get(role)
                if bound is not None and isinstance(spec.source, HostImage):
                    # The binding replaces the freeze-time host fallback with the composed
                    # digest-pinned toolchain image; the machine-source demands become the
                    # registry image the role now runs from — the image bakes the lock, so
                    # both the host-image and packages-lock demands resolve into it.
                    spec = replace(spec, source=RegistryImage(bound), lock_satisfied_by_image=True)
                role_needs = demands[role]
                if bound is not None:
                    role_needs = {Requirement.DOCKER_IMAGE} | (
                        set(role_needs) - {Requirement.HOST_IMAGE, Requirement.PACKAGES_LOCK}
                    )
                kind = environment_kind(spec.source)
                if cells[backend, kind].support == Support.UNSUPPORTED:
                    role_counts[backend][role][FitSupport.NOT_APPLICABLE] += 1
                    exclusions[backend][f"{role}:{kind.value}"] += 1
                    blocker_reasons[backend][
                        f"{role}:{kind.value}"
                    ] = BackendBlockerReason.UNSUPPORTED_ENVIRONMENT_KIND.value
                    outcomes.append(FitSupport.NOT_APPLICABLE)
                    continue
                if backend == "local":
                    spec = replace(spec, memory_mb=None)
                status = combined_support(cells[backend, need].support for need in role_needs)
                for need in role_needs:
                    # VERIFY demands are pending live evidence, not failures; the
                    # role's fit carries the VERIFY verdict and admission gates
                    # wait on the live qualification instead of blocking here.
                    if cells[backend, need].support not in {Support.SUPPORTED, Support.VERIFY}:
                        blockers[backend][f"{role}:{need.value}"] += 1
                        # A typed demand renders as its own value; the served
                        # backend-reason table carries every requirement value.
                        blocker_reasons[backend][f"{role}:{need.value}"] = need.value
                try:
                    check(requirements, spec)
                except UnsupportedMachineSpec as error:
                    status = Support.UNSUPPORTED
                    errors[backend][f"{role}: {error}"] += 1
                role_counts[backend][role][FitSupport(status.value)] += 1
                outcomes.append(FitSupport(status.value))
            task_counts[backend][combined_fit(outcomes)] += 1
    if not count:
        raise ValueError(f"Converted-spec census is empty: {source_id}")
    reason_slugs = backend_blocker_reason_slugs([requirement.value for requirement in Requirement])
    for backend in checkers:
        stored = set(blockers[backend]) | set(exclusions[backend])
        unmapped = sorted(key for key in stored if key not in blocker_reasons[backend])
        if unmapped:
            raise ValueError(f"Backend blockers with no mapped reason for {source_id}/{backend}: {unmapped}")
        unknown = sorted(set(blocker_reasons[backend].values()) - reason_slugs)
        if unknown:
            raise ValueError(f"Backend blocker reasons outside the served taxonomy for {source_id}/{backend}: {unknown}")
    return BackendSourceMap(
        source_id=source_id,
        basis="converted_spec_census",
        task_count=count,
        inputs=inputs,
        demands=dict(demand_counts),
        resource_ranges=dict(ranges),
        role_backend_assignments=dict(assignments),
        grading_environment=grading_environment,
        integrity_contracts=integrity_contracts,
        backends={
            backend: BackendFit(
                support=combined_fit(task_counts[backend]),
                task_counts={**task_counts[backend]},
                roles=dict(role_counts[backend]),
                blockers=dict(blockers[backend]),
                not_applicable=dict(exclusions[backend]),
                blocker_reasons=dict(blocker_reasons[backend]),
                validation_errors=dict(errors[backend]),
            )
            for backend in checkers
        },
    )


def verify_backend_blocker_reasons(survey: BackendSurvey) -> None:
    """Fail closed on any backend blocker the served taxonomy cannot label.

    The Backend evidence page derives a blocker for every stored ``blockers`` and
    ``not_applicable`` key (plus the unmeasured-role and incomplete-support keys it
    synthesizes) and refuses to render one without a typed reason. The derivation
    labels every blocker it stores; the catalog load re-checks the sealed survey
    here, so a release whose survey drifted from the taxonomy refuses to load
    instead of crashing the page (R40's crash class, found only at the final
    browser sweep).
    """
    reason_slugs = backend_blocker_reason_slugs([requirement.value for requirement in Requirement])
    for source in survey.sources:
        if source.basis != "converted_spec_census":
            continue
        for backend, fit in source.backends.items():
            stored = {**fit.blockers, **fit.not_applicable}
            unmapped = sorted(set(stored) - set(fit.blocker_reasons))
            if unmapped:
                raise ValueError(f"Backend blockers with no mapped reason for {source.source_id}/{backend}: {unmapped}")
            unknown = sorted(set(fit.blocker_reasons.values()) - reason_slugs)
            if unknown:
                raise ValueError(
                    f"Backend blocker reasons outside the served taxonomy for {source.source_id}/{backend}: {unknown}"
                )
            for role in ("agent", "grader"):
                if sum(fit.roles.get(role, {}).values()) != source.task_count:
                    raise ValueError(
                        f"Backend fit for {source.source_id}/{backend} leaves {role} outcomes unmeasured;"
                        " the evidence page would derive an unlabelled not-measured blocker"
                    )
            if fit.support == FitSupport.PARTIAL and not stored:
                raise ValueError(
                    f"Backend fit for {source.source_id}/{backend} is partial with no stored blocker;"
                    " the evidence page would derive an unlabelled incomplete-support blocker"
                )


def converted_specs(paths: list[Path], column: str, runtime_column: str | None = None) -> Iterable[dict]:
    """Stream converted spec JSON without retaining private task payloads."""
    columns = [column, runtime_column] if runtime_column is not None else [column]
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=columns):
            for row in batch.to_pylist():
                encoded = row[column]
                if encoded is None:
                    raise ValueError(f"Null converted spec in {path}")
                task = json.loads(encoded)
                if runtime_column is None:
                    yield task
                    continue
                yield {"task": task, "runtime": json.loads(row[runtime_column])}


def verified_disk_pool(checkers: dict[str, Checker]) -> LocalDiskPool | None:
    """Read the LOCAL disk pool the checkers were actually built with."""
    pool = getattr(checkers.get("local"), "keywords", {}).get("disk_pool")
    return pool if isinstance(pool, LocalDiskPool) else None


def build_survey(manifest: dict, checkers: dict[str, Checker], revision: str) -> BackendSurvey:
    """Bind a census to its catalog inventory; unavailable sources remain unassessed.

    The artifact records the LOCAL disk pool the local checker was actually
    constructed with, never a manifest claim about one.
    """
    matrix = production_matrix(checkers)
    disk_pool = verified_disk_pool(checkers)
    runtime_defaults = {
        role: RuntimeProfile.model_validate(manifest["runtime_defaults"][role]) for role in ("agent", "grader")
    }
    source_ids = [canonical_source_id(value) for value in manifest["source_ids"]]
    manifest = {
        **manifest,
        "grading_environments": canonical_source_map(manifest.get("grading_environments", {})),
        "integrity_contracts": canonical_source_map(manifest.get("integrity_contracts", {})),
    }
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Duplicate inventory source IDs")
    selected = {}
    for entry in manifest["inputs"]:
        source_id = canonical_source_id(entry["source_id"])
        if source_id not in source_ids or source_id in selected:
            raise ValueError(f"Unknown or duplicate input source: {source_id}")
        paths = [Path(path) for path in entry["paths"]]
        digests = {}
        for index, path in enumerate(paths):
            with path.open("rb") as stream:
                digests[f"{index}:{path.name}"] = hashlib.file_digest(stream, "sha256").hexdigest()
        measured = source_map(
            source_id,
            converted_specs(paths, entry["column"], entry.get("runtime_column")),
            matrix,
            checkers,
            digests,
            runtime_defaults,
            manifest.get("grading_environments", {}).get(source_id),
            manifest.get("integrity_contracts", {}).get(source_id),
        )
        if measured.task_count != entry["expected_tasks"]:
            raise ValueError(f"Converted population differs from manifest: {source_id}")
        selected[source_id] = measured
    unassessed = [source_id for source_id in source_ids if source_id not in selected]
    declared_reasons = canonical_source_map(manifest.get("unassessed_reasons", {}))
    missing_reasons = [source_id for source_id in unassessed if source_id not in declared_reasons]
    if missing_reasons:
        raise ValueError(
            "Sources without converted inputs require their own unassessed_reasons entry so an "
            f"accidentally omitted input cannot hide behind a shared reason: {missing_reasons}"
        )
    unused_reasons = [source_id for source_id in declared_reasons if source_id not in unassessed]
    if unused_reasons:
        raise ValueError(f"unassessed_reasons declares sources that have converted inputs: {unused_reasons}")
    limits = {
        backend: {
            "max_concurrent_sandboxes": None,
            "snapshot_cap": None,
            "evidence": "Not measured by offline factory validation.",
        }
        for backend in checkers
    }
    for backend, observation in manifest.get("deployment_limits", {}).items():
        if backend not in checkers or not observation.get("evidence") or not observation.get("observed_at"):
            raise ValueError("Deployment limits require a backend, timestamp and evidence")
        limits[backend] = observation
    return BackendSurvey(
        catalog_root=manifest["catalog_root"],
        snapshot_id=manifest["snapshot_id"],
        generated_at=datetime.now(UTC).isoformat(),
        git_revision=revision,
        schema_sha256=backend_schema_sha256(),
        matrix=matrix,
        deployment_limits=limits,
        deployment_classes={
            backend.value: deployment_class.value for backend, deployment_class in BACKEND_DEPLOYMENT.items()
        },
        local_disk_pool=None if disk_pool is None else str(disk_pool.root),
        runtime_defaults=runtime_defaults,
        sources=[
            selected.get(source_id)
            or BackendSourceMap(
                source_id=source_id,
                basis="unassessed",
                reason=declared_reasons[source_id],
                task_count=0,
                inputs={},
                demands={},
                resource_ranges={},
                backends={},
                role_backend_assignments={},
                grading_environment=manifest.get("grading_environments", {}).get(source_id),
                integrity_contracts=manifest.get("integrity_contracts", {}).get(source_id),
            )
            for source_id in source_ids
        ],
    )


def merge_surveys(partials: list[BackendSurvey], *, catalog_root: str, snapshot_id: str) -> BackendSurvey:
    """Combine shard surveys into one artifact bound to the selected snapshot.

    Projects every partial through the canonical source-ID boundary before
    deduplication, so historical and current spellings of one source cannot
    evade the overlap check. Rejects shards that disagree on the matrix,
    runtime profile, deployment limits, schema, revision or verified disk
    pool, and sources measured in more than one shard. A source that is
    unassessed in every shard keeps its declared reason in the merged
    artifact; a measured row overrides shard placeholders. The selected
    snapshot's coverage is checked again at load time.
    """
    projected = [canonical_backend_survey(partial.model_dump(mode="json")) for partial in partials]
    first = projected[0]
    identity = (
        first.schema_sha256,
        first.git_revision,
        first.local_disk_pool,
        [cell.model_dump(mode="json") for cell in first.matrix],
        {role: profile.model_dump(mode="json") for role, profile in first.runtime_defaults.items()},
        first.deployment_limits,
        first.deployment_classes,
    )
    for partial in projected[1:]:
        other = (
            partial.schema_sha256,
            partial.git_revision,
            partial.local_disk_pool,
            [cell.model_dump(mode="json") for cell in partial.matrix],
            {role: profile.model_dump(mode="json") for role, profile in partial.runtime_defaults.items()},
            partial.deployment_limits,
            partial.deployment_classes,
        )
        if other != identity:
            raise ValueError(f"Shard {partial.snapshot_id} does not share the survey identity")
    rows: dict[str, BackendSourceMap] = {}
    unassessed: dict[str, BackendSourceMap] = {}
    for partial in projected:
        for source in partial.sources:
            if source.basis == "unassessed":
                if source.source_id not in rows:
                    unassessed.setdefault(source.source_id, source)
                continue
            if source.source_id in rows:
                raise ValueError(f"Source measured in more than one shard: {source.source_id}")
            unassessed.pop(source.source_id, None)
            rows[source.source_id] = source
    if not rows and not unassessed:
        raise ValueError("The shards measure no sources")
    return BackendSurvey(
        catalog_root=catalog_root,
        snapshot_id=snapshot_id,
        generated_at=datetime.now(UTC).isoformat(),
        git_revision=first.git_revision,
        schema_sha256=first.schema_sha256,
        matrix=first.matrix,
        deployment_limits=first.deployment_limits,
        local_disk_pool=first.local_disk_pool,
        runtime_defaults=first.runtime_defaults,
        deployment_classes=first.deployment_classes,
        sources=[*rows.values(), *unassessed.values()],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Survey manifest for a census run.")
    parser.add_argument("--merge", type=Path, nargs="+", help="Combine these shard surveys into one artifact instead.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--local-disk-root", type=Path, help="Probe ext4 loop/OverlayFS enforcement on this privileged worker."
    )
    parser.add_argument(
        "--catalog-root",
        help="Live catalog root to bind a census to, or the literal binding root for --merge.",
    )
    parser.add_argument("--snapshot-id", help="Selected snapshot ID bound by --merge.")
    parser.add_argument("--revision", help="Survey code revision; the workspace carries no git metadata on workers.")
    args = parser.parse_args()
    if args.merge is not None:
        if args.snapshot_id is None or args.catalog_root is None:
            raise SystemExit("--merge requires the binding --catalog-root and --snapshot-id")
        partials = [BackendSurvey.model_validate_json(path.read_text()) for path in args.merge]
        survey = merge_surveys(partials, catalog_root=args.catalog_root, snapshot_id=args.snapshot_id)
        args.output.write_text(survey.model_dump_json(indent=2) + "\n")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "shards": len(partials),
                    "sources": len(survey.sources),
                    "tasks": sum(source.task_count for source in survey.sources),
                }
            )
        )
        return
    if args.manifest is None:
        raise SystemExit("A census run requires --manifest")
    manifest = json.loads(args.manifest.read_text())
    if args.catalog_root is not None:
        configure_coreweave_s3()
        catalog = CurationCatalog(args.catalog_root)
        snapshot = catalog.current()
        sources = [canonical_snapshot_source(source) for source in catalog.sources(snapshot.snapshot_id)]
        manifest = {
            **manifest,
            "catalog_root": catalog.root,
            "snapshot_id": snapshot.snapshot_id,
            "source_ids": [source.id for source in sources],
            "grading_environments": {
                source.id: source.declaration.grading_environment.model_dump(mode="json")
                for source in sources
                if source.declaration.grading_environment is not None
            },
        }
    if args.revision is not None:
        revision = args.revision
    else:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    disk_pool = LocalDiskPool(args.local_disk_root) if args.local_disk_root is not None else None
    survey = build_survey(manifest, production_checkers(local_disk_pool=disk_pool), revision)
    args.output.write_text(survey.model_dump_json(indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "matrix_cells": len(survey.matrix),
                "sources": len(survey.sources),
                "tasks": sum(source.task_count for source in survey.sources),
            }
        )
    )


if __name__ == "__main__":
    main()
