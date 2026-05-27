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

from marin.midtraining.checkpoint_schema import (
    ListCheckpointKeys,
    assert_checkpoint_complete_for_model_type,
    default_list_checkpoint_keys,
)
from marin.midtraining.identity import RunIdentity, output_region
from marin.midtraining.modes import CheckpointSourceKind, CooldownMode, CptMode
from marin.midtraining.schema import SCHEMA_VERSION, RunManifestRow, read_run_manifest
from marin.midtraining.spec import MidtrainSpec, ResolvedMidtrainSpec

logger = logging.getLogger(__name__)

_CHECKPOINT_STEP_PATTERN = re.compile(r"step-(\d+)$")

GcsExists = Callable[[str], bool]
GcsList = Callable[[str], tuple[str, ...]]
ReadManifest = Callable[[str], RunManifestRow]


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
    allow_existing_matching_manifest: bool = False,
    exists: GcsExists = default_gcs_exists,
    list_: GcsList = default_gcs_list,
    read_manifest: ReadManifest = read_run_manifest,
    list_ocdbt_keys: ListCheckpointKeys = default_list_checkpoint_keys,
) -> PreflightReport:
    """Run all guards required before a launch.

    ``allow_existing_matching_manifest`` is only for an Iris retry of the same
    coordinator task attempt. A normal fresh launch should leave it false so an
    existing run namespace remains a hard collision.

    ``list_ocdbt_keys`` is the injection point for the staged-checkpoint
    schema check (see ``marin.midtraining.checkpoint_schema``). Tests pass
    a fake; production uses ``default_list_checkpoint_keys`` which goes
    through ``tensorstore``.
    """
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
        staged_uri = _check_cooldown_stage(
            spec,
            spec.mode,
            exists,
            failures,
            cross_region_copy,
            list_ocdbt_keys=list_ocdbt_keys,
        )

    _check_run_namespace(
        resolved,
        exists,
        list_,
        read_manifest,
        allow_existing_matching_manifest,
        failures,
        notes,
    )

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
    *,
    list_ocdbt_keys: ListCheckpointKeys,
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

    artifacts_ok = True
    for required in ("manifest.ocdbt", "metadata.json"):
        if not exists(f"{staged_path}/{required}"):
            failures.append(f"Cooldown staged checkpoint missing {required} at {staged_path}/{required}")
            artifacts_ok = False
    if not exists(f"{staged_path}/d"):
        failures.append(f"Cooldown staged checkpoint missing 'd/' subtree at {staged_path}/d")
        artifacts_ok = False

    # Schema-level check: the on-disk OCDBT must contain every array the
    # declared model class expects. This catches the silent-type-degradation
    # bug class (see marin.midtraining.checkpoint_schema) BEFORE we burn TPU
    # time on a staged checkpoint that's missing q_norm/k_norm arrays.
    # Skip if file-level artifacts are already failing — listing OCDBT keys
    # would just produce a confusing secondary error.
    if artifacts_ok:
        model_type = spec.model_config.get("type") if isinstance(spec.model_config, dict) else None
        if not model_type:
            failures.append(
                f"Cooldown spec.model_config is missing the 'type' discriminator field; "
                f"got model_config keys={sorted(spec.model_config) if hasattr(spec.model_config, 'keys') else '?'}. "
                "An untyped model config silently defaults at decode time and is the root "
                "cause of the 2026-05-27 silent-type-degradation bug. Set "
                "model_config['type'] explicitly (e.g. 'qwen3' for Delphi)."
            )
        else:
            try:
                assert_checkpoint_complete_for_model_type(
                    staged_path,
                    model_type=model_type,
                    num_layers=spec.base.num_layers,
                    list_keys=list_ocdbt_keys,
                )
            except ValueError as exc:
                failures.append(str(exc))
            except RuntimeError as exc:
                # tensorstore could not open the kvstore — surface as a
                # preflight failure rather than swallowing it.
                failures.append(f"Cooldown staged checkpoint schema check failed: {exc}")
    return staged_path


def _check_run_namespace(
    resolved: ResolvedMidtrainSpec,
    exists: GcsExists,
    list_: GcsList,
    read_manifest: ReadManifest,
    allow_existing_matching_manifest: bool,
    failures: list[str],
    notes: list[str],
) -> None:
    spec = resolved.spec
    run = spec.run
    permanent_root = run.permanent_checkpoints_uri
    temp_root = _temp_checkpoint_uri(run)
    has_permanent_steps = bool(_latest_step_in(exists, list_, permanent_root))
    has_temp_steps = bool(_latest_step_in(exists, list_, temp_root))
    has_existing_manifest = exists(run.manifest_uri)

    if isinstance(spec.mode, CptMode):
        if not spec.is_resume:
            if has_existing_manifest and allow_existing_matching_manifest:
                if _existing_manifest_matches_current_run(resolved, read_manifest, failures):
                    notes.append(
                        "existing manifest matches this run identity; accepting namespace as same-attempt "
                        "coordinator retry"
                    )
                    return
                return
            if has_permanent_steps:
                failures.append(
                    f"Fresh CPT launch refused: permanent checkpoints already exist under {permanent_root}; "
                    "bump RunIdentity.attempt for a fresh restart, or set expected_min_step to resume."
                )
            if has_temp_steps:
                failures.append(
                    f"Fresh CPT launch refused: temporary checkpoints already exist under {temp_root}; "
                    "bump RunIdentity.attempt for a fresh restart, or set expected_min_step to resume."
                )
            if has_existing_manifest:
                failures.append(
                    f"Fresh CPT launch refused: manifest already at {run.manifest_uri}; "
                    "bump RunIdentity.attempt for a fresh restart, or set expected_min_step to resume."
                )
        else:
            latest = _latest_step_across(exists, list_, permanent_root, temp_root)
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
        latest = _latest_step_across(exists, list_, permanent_root, temp_root)
        if spec.expected_min_step is not None and latest is not None and latest < spec.expected_min_step:
            failures.append(
                f"Cooldown resume: latest checkpoint step {latest} below expected_min_step " f"{spec.expected_min_step}"
            )


def _existing_manifest_matches_current_run(
    resolved: ResolvedMidtrainSpec,
    read_manifest: ReadManifest,
    failures: list[str],
) -> bool:
    run = resolved.spec.run
    try:
        row = read_manifest(run.manifest_uri)
    except Exception as exc:
        failures.append(
            f"Fresh CPT launch found manifest at {run.manifest_uri}, but could not read it "
            f"to verify same-attempt retry safety: {exc!r}"
        )
        return False

    expected = _manifest_retry_fields(resolved)
    mismatches = []
    for key, value in expected.items():
        actual = row.get(key)
        if actual != value:
            mismatches.append(f"{key}: manifest={actual!r} current={value!r}")
    if mismatches:
        detail = "; ".join(mismatches[:8])
        if len(mismatches) > 8:
            detail += f"; ... {len(mismatches) - 8} more"
        failures.append(
            f"Fresh CPT launch refused: manifest already at {run.manifest_uri} but does not match "
            f"the current resolved run identity/config ({detail}). Bump RunIdentity.attempt for "
            "a fresh restart, or set expected_min_step for an intentional resume."
        )
        return False
    return True


def _manifest_retry_fields(resolved: ResolvedMidtrainSpec) -> dict[str, object]:
    spec = resolved.spec
    run = spec.run
    data_manifest_uri = spec.data_manifest_uri or f"legacy:{spec.data_section_provenance or 'unknown'}"
    data_manifest_fingerprint = (
        resolved.data_manifest.fingerprint()
        if resolved.data_manifest is not None
        else f"legacy:{spec.data_section_provenance or 'unknown'}"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "logical_cell_id": run.logical_cell_id,
        "attempt": run.attempt,
        "run_id": run.run_id,
        "mode": spec.mode.kind,
        "output_path": run.output_path,
        "wandb_project": run.wandb_project,
        "wandb_entity": run.wandb_entity,
        "base_flops_key": spec.base.flops_key,
        "tpu_type": spec.compute.tpu_type,
        "train_batch_size": spec.compute.batch_size,
        "per_device_parallelism": spec.compute.per_device_parallelism,
        "max_retries_failure": spec.compute.max_retries_failure,
        "max_task_failures": spec.compute.max_task_failures,
        "data_manifest_uri": data_manifest_uri,
        "data_manifest_fingerprint": data_manifest_fingerprint,
        "seq_len": spec.seq_len,
        "num_train_steps": resolved.num_train_steps,
        "actual_tokens": resolved.actual_tokens,
        "train_config_uri": run.train_config_uri,
        "permanent_checkpoints_uri": run.permanent_checkpoints_uri,
        "temp_checkpoints_uri": _temp_checkpoint_uri(run),
    }


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


def _latest_step_across(exists: GcsExists, list_: GcsList, *base_uris: str) -> int | None:
    steps = [_latest_step_in(exists, list_, uri) for uri in base_uris]
    present = [step for step in steps if step is not None]
    if not present:
        return None
    return max(present)


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
