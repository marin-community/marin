# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preflight checks + cooldown checkpoint staging.

GCS access is injected as two callables (``exists``, ``list_``) so tests
substitute lambdas over a frozenset. Production code uses the
``default_gcs_exists`` / ``default_gcs_list`` helpers below.
"""

import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from marin.midtraining.identity import RunIdentity, output_region
from marin.midtraining.modes import CheckpointSourceKind, CooldownMode, CptMode
from marin.midtraining.spec import MidtrainSpec, ResolvedMidtrainSpec

logger = logging.getLogger(__name__)

_CHECKPOINT_STEP_PATTERN = re.compile(r"step-(\d+)$")

GcsExists = Callable[[str], bool]
GcsList = Callable[[str], tuple[str, ...]]


def default_gcs_exists(uri: str) -> bool:
    import fsspec

    fs, path = fsspec.core.url_to_fs(uri)
    return bool(fs.exists(path))


def default_gcs_list(uri: str) -> tuple[str, ...]:
    import fsspec

    fs, path = fsspec.core.url_to_fs(uri)
    return tuple(fs.ls(path, detail=False))


@dataclass(frozen=True)
class PreflightReport:
    """Snapshot of what preflight inspected and concluded."""

    cell_id: str
    mode_kind: str
    permanent_checkpoints_uri: str
    temp_checkpoints_uri: str
    init_checkpoint_uri: str | None
    staged_checkpoint_uri: str | None
    data_manifest_uri: str | None
    data_section_provenance: str | None
    failures: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class CrossRegionCopyPolicy:
    """Per-cell guard around large GCS-to-GCS copies during cooldown stage."""

    allowed: bool = False
    budget_gb: int = 0
    reason: str = ""

    def __post_init__(self) -> None:
        if self.allowed:
            if self.budget_gb <= 0:
                raise ValueError("CrossRegionCopyPolicy(allowed=True) requires budget_gb > 0")
            if not self.reason or len(self.reason) < 10:
                raise ValueError(
                    f"CrossRegionCopyPolicy(allowed=True) requires a meaningful reason, got {self.reason!r}"
                )


def preflight(
    resolved: ResolvedMidtrainSpec,
    *,
    cross_region_copy: CrossRegionCopyPolicy = CrossRegionCopyPolicy(),
    exists: GcsExists = default_gcs_exists,
    list_: GcsList = default_gcs_list,
) -> PreflightReport:
    """Run all guards required before a launch."""
    spec = resolved.spec
    run = spec.run
    failures: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    permanent_uri = run.permanent_checkpoints_uri
    temp_uri = _temp_checkpoint_uri(run)
    notes.append(f"permanent checkpoints search path: {permanent_uri}")
    notes.append(f"temporary checkpoints search path: {temp_uri}")

    init_uri: str | None = None
    staged_uri: str | None = None

    if spec.data_manifest_uri is not None:
        if not exists(spec.data_manifest_uri):
            failures.append(f"Data manifest {spec.data_manifest_uri} not reachable")
        if spec.run.output_region != output_region(spec.data_manifest_uri):
            failures.append(
                f"Run region {spec.run.output_region!r} != data manifest region "
                f"{output_region(spec.data_manifest_uri)!r}"
            )
    else:
        # data_section_override path: verify each declared component cache exists in the run region.
        assert spec.data_section_override is not None
        notes.append(f"data section provenance: {spec.data_section_provenance!r}")
        components = spec.data_section_override.get("components") or {}
        for name, component in components.items():
            cache_dir = component.get("cache_dir") if isinstance(component, dict) else None
            if cache_dir and cache_dir.startswith("gs://"):
                if output_region(cache_dir) != spec.run.output_region:
                    failures.append(
                        f"Component {name!r} cache_dir region {output_region(cache_dir)!r} "
                        f"!= run region {spec.run.output_region!r}"
                    )
                if not exists(cache_dir):
                    failures.append(f"Component {name!r} cache_dir not reachable: {cache_dir}")

    if isinstance(spec.mode, CptMode):
        init_uri = _check_cpt_init(spec.mode, exists, failures, warnings)
    else:
        assert isinstance(spec.mode, CooldownMode)
        staged_uri = _check_cooldown_stage(spec, spec.mode, exists, failures, cross_region_copy)

    _check_run_namespace(resolved, exists, list_, failures)

    return PreflightReport(
        cell_id=run.logical_cell_id,
        mode_kind=spec.mode.kind,
        permanent_checkpoints_uri=permanent_uri,
        temp_checkpoints_uri=temp_uri,
        init_checkpoint_uri=init_uri,
        staged_checkpoint_uri=staged_uri,
        data_manifest_uri=spec.data_manifest_uri,
        data_section_provenance=spec.data_section_provenance,
        failures=tuple(failures),
        warnings=tuple(warnings),
        notes=tuple(notes),
    )


def _check_cpt_init(
    mode: CptMode,
    exists: GcsExists,
    failures: list[str],
    warnings: list[str],
) -> str | None:
    if mode.init.source_kind == CheckpointSourceKind.NATIVE_LEVANTER:
        path = mode.init.resolved_checkpoint_path()
        if path is None:
            failures.append("CPT NATIVE_LEVANTER source did not produce a checkpoint path")
            return None
        if path.startswith("mirror://"):
            warnings.append(f"CPT init source is logical {path!r}; resolve at stage time")
            return path
        if not exists(path):
            failures.append(f"CPT init checkpoint not found at {path}")
        return path
    return mode.init.hf_repo


def _check_cooldown_stage(
    spec: MidtrainSpec,
    mode: CooldownMode,
    exists: GcsExists,
    failures: list[str],
    cross_region_copy: CrossRegionCopyPolicy,
) -> str:
    staged_path = mode.resume.staged_checkpoint_path
    if not exists(staged_path):
        failures.append(
            f"Cooldown staged checkpoint not found at {staged_path}; "
            "run stage_cooldown_checkpoint(spec, ...) before launch."
        )
        return staged_path

    source = mode.resume.pretrain_checkpoint_path
    if source.startswith("gs://"):
        source_region = output_region(source)
        dest_region = spec.run.output_region
        if source_region != dest_region and not cross_region_copy.allowed:
            failures.append(
                f"Cooldown source region {source_region!r} != destination region {dest_region!r}; "
                "pass CrossRegionCopyPolicy(allowed=True, budget_gb=..., reason=...) to authorize."
            )

    for required in ("manifest.ocdbt", "metadata.json"):
        if not exists(f"{staged_path}/{required}"):
            failures.append(f"Cooldown staged checkpoint missing {required} at {staged_path}/{required}")
    if not exists(f"{staged_path}/d"):
        failures.append(f"Cooldown staged checkpoint missing 'd/' subtree at {staged_path}/d")
    return staged_path


def _check_run_namespace(
    resolved: ResolvedMidtrainSpec,
    exists: GcsExists,
    list_: GcsList,
    failures: list[str],
) -> None:
    spec = resolved.spec
    run = spec.run
    permanent_root = run.permanent_checkpoints_uri
    has_permanent_steps = bool(_latest_step_in(exists, list_, permanent_root))
    has_existing_manifest = exists(run.manifest_uri)

    if isinstance(spec.mode, CptMode):
        if not spec.is_resume:
            if has_permanent_steps:
                failures.append(
                    f"Fresh CPT launch refused: permanent checkpoints already exist under {permanent_root}; "
                    "bump RunIdentity.attempt for a fresh restart, or set expected_min_step to resume."
                )
            if has_existing_manifest:
                failures.append(
                    f"Fresh CPT launch refused: manifest already at {run.manifest_uri}; "
                    "bump RunIdentity.attempt for a fresh restart, or set expected_min_step to resume."
                )
        else:
            latest = _latest_step_in(exists, list_, permanent_root) or _latest_step_in(
                exists, list_, _temp_checkpoint_uri(run)
            )
            assert spec.expected_min_step is not None
            if latest is None and not spec.allow_empty_resume:
                failures.append(
                    f"Resume requires a checkpoint at or above step {spec.expected_min_step}; "
                    f"none found under {permanent_root} or {_temp_checkpoint_uri(run)}."
                )
            elif latest is not None and latest < spec.expected_min_step:
                failures.append(
                    f"Latest checkpoint step {latest} below expected_min_step "
                    f"{spec.expected_min_step} under {permanent_root}"
                )
    else:
        assert isinstance(spec.mode, CooldownMode)
        latest = _latest_step_in(exists, list_, permanent_root)
        if spec.expected_min_step is not None and latest is not None and latest < spec.expected_min_step:
            failures.append(
                f"Cooldown resume: latest checkpoint step {latest} below expected_min_step " f"{spec.expected_min_step}"
            )


def _latest_step_in(exists: GcsExists, list_: GcsList, base_uri: str) -> int | None:
    if not exists(base_uri):
        return None
    steps: list[int] = []
    for entry in list_(base_uri):
        match = _CHECKPOINT_STEP_PATTERN.search(entry.rstrip("/"))
        if match:
            steps.append(int(match.group(1)))
    if not steps:
        return None
    return max(steps)


def _temp_checkpoint_uri(run: RunIdentity) -> str:
    sanitized_output = run.output_path.removeprefix("gs://").strip("/").replace("/", "_")
    return f"gs://marin-{run.output_region}/tmp/ttl=14d/checkpoints-temp/{sanitized_output}/checkpoints"


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def fake_gcs(*paths: str) -> tuple[GcsExists, GcsList]:
    """Return ``(exists, list_)`` callables over an in-memory path set."""
    universe = frozenset(paths)

    def exists(uri: str) -> bool:
        normalized = uri.rstrip("/")
        return any(p == normalized or p.startswith(normalized + "/") for p in universe)

    def list_(uri: str) -> tuple[str, ...]:
        normalized = uri.rstrip("/") + "/"
        children: set[str] = set()
        for p in universe:
            if p.startswith(normalized):
                tail = p[len(normalized) :]
                first = tail.split("/", 1)[0]
                children.add(normalized + first)
        return tuple(sorted(children))

    return exists, list_


# ---------------------------------------------------------------------------
# Cooldown staging
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CooldownStageRecord:
    source: str
    destination: str
    cross_region_copy: bool
    bytes_copied: int
    budget_gb: int
    reason: str


def stage_cooldown_checkpoint(
    spec: MidtrainSpec,
    *,
    cross_region_copy: CrossRegionCopyPolicy = CrossRegionCopyPolicy(),
    exists: GcsExists = default_gcs_exists,
    dry_run: bool = False,
) -> CooldownStageRecord:
    """Materialize the cooldown staged checkpoint under the run namespace."""
    if not isinstance(spec.mode, CooldownMode):
        raise ValueError("stage_cooldown_checkpoint requires a cooldown spec")
    source = spec.mode.resume.pretrain_checkpoint_path
    destination = spec.mode.resume.staged_checkpoint_path

    source_region = output_region(source) if source.startswith("gs://") else "(logical)"
    dest_region = spec.run.output_region
    is_cross_region = source_region != "(logical)" and source_region != dest_region

    if is_cross_region and not cross_region_copy.allowed:
        raise ValueError(
            f"Refusing cross-region cooldown stage from {source_region!r} to {dest_region!r}. "
            "Pass CrossRegionCopyPolicy(allowed=True, budget_gb=..., reason=...) to authorize."
        )

    if exists(destination):
        logger.info("Cooldown checkpoint already staged at %s; reusing.", destination)
        return _stage_record(source, destination, is_cross_region, 0, cross_region_copy)

    if dry_run:
        logger.info("[dry-run] would stage %s -> %s", source, destination)
        return _stage_record(source, destination, is_cross_region, 0, cross_region_copy)

    bytes_copied = _copy_gcs_tree(source, destination)
    return _stage_record(source, destination, is_cross_region, bytes_copied, cross_region_copy)


def _stage_record(
    source: str,
    destination: str,
    is_cross_region: bool,
    bytes_copied: int,
    policy: CrossRegionCopyPolicy,
) -> CooldownStageRecord:
    return CooldownStageRecord(
        source=source,
        destination=destination,
        cross_region_copy=is_cross_region,
        bytes_copied=bytes_copied,
        budget_gb=policy.budget_gb,
        reason=policy.reason,
    )


def _copy_gcs_tree(source: str, destination: str) -> int:
    """Recursive ``gs://`` copy via fsspec."""
    import fsspec

    src_fs, src_path = fsspec.core.url_to_fs(source)
    dst_fs, dst_path = fsspec.core.url_to_fs(destination)
    total = 0
    files = src_fs.find(src_path, detail=True)
    for src_entry in files:
        rel = os.path.relpath(src_entry, src_path)
        target = os.path.join(dst_path, rel)
        with src_fs.open(src_entry, "rb") as src_file, dst_fs.open(target, "wb") as dst_file:
            while True:
                chunk = src_file.read(8 * 1024 * 1024)
                if not chunk:
                    break
                dst_file.write(chunk)
                total += len(chunk)
    return total
