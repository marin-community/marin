# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed spec for one midtraining launch.

The spec is intentionally flat. Mode-specific behavior lives in
:mod:`marin.midtraining.modes` (``CptMode``, ``CooldownMode``); the
artifact contract lives in :mod:`marin.midtraining.schema`. Eval cadence,
checkpoint cadence, safety knobs, and resume settings sit directly on
:class:`MidtrainSpec` — nested policy objects added navigation without
saving call-site characters.

Sweep expansion is the cell author's responsibility (write a ``for`` loop
in your launcher script). The redesign doc explicitly calls for
"multi-cell sweeps are a shell/driver loop over one-cell launches".
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

from marin.midtraining.budget import ResolvedBudget
from marin.midtraining.data_manifest import DataCacheManifest, load_data_manifest
from marin.midtraining.identity import RunIdentity, output_region
from marin.midtraining.modes import (
    CooldownMode,
    CptMode,
    TrainingMode,
)
from marin.midtraining.tokenizers import TokenizerRef, assert_tokenizer_compatible


@runtime_checkable
class BaseModelRef(Protocol):
    """Structural subset of a base-model registry entry used by the launcher.

    ``experiments.delphi_models.DelphiModel`` satisfies this protocol.
    """

    flops_key: str
    params: int
    hidden_dim: int
    num_layers: int
    batch_size: int
    num_train_steps: int
    hf_repo: str
    hf_revision: str
    gcs_run_root: str

    @property
    def tokens(self) -> int: ...
    @property
    def verified_checkpoint_path(self) -> str: ...
    def levanter_checkpoint_path(self, step: int | None = None) -> str: ...


# ---------------------------------------------------------------------------
# Compute profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComputeProfile:
    """TPU resources for one attempt.

    ``regions`` may list multiple regions only at planning time. A real
    launch collapses to ``(RunIdentity.output_region,)``.
    """

    tpu_type: str
    batch_size: int
    ram: str = "128g"
    per_device_parallelism: int = -1
    regions: tuple[str, ...] = ()
    preemptible: bool = True
    max_retries_failure: int = 3
    max_task_failures: int = 100

    def __post_init__(self) -> None:
        if not self.tpu_type:
            raise ValueError("ComputeProfile.tpu_type must be non-empty")
        if self.batch_size <= 0:
            raise ValueError(f"ComputeProfile.batch_size must be positive, got {self.batch_size!r}")
        if not self.ram:
            raise ValueError("ComputeProfile.ram must be non-empty")
        if self.per_device_parallelism == 0:
            raise ValueError("ComputeProfile.per_device_parallelism must be -1 or positive, not zero")
        if self.max_retries_failure < 0:
            raise ValueError(f"max_retries_failure must be >= 0, got {self.max_retries_failure!r}")
        if self.max_task_failures < 0:
            raise ValueError(f"max_task_failures must be >= 0, got {self.max_task_failures!r}")
        for region in self.regions:
            if not region:
                raise ValueError(f"ComputeProfile.regions must not contain empty strings, got {self.regions!r}")


# ---------------------------------------------------------------------------
# MidtrainSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MidtrainSpec:
    """One explicit cell specification.

    Composed of identity (:class:`RunIdentity`), what to train (``base`` +
    ``mode``), data (:class:`DataCacheManifest` URI + tokenizer), compute,
    and rendered Levanter config blocks (``model_config`` / ``optimizer_config``).
    Cadence and safety knobs sit directly on the spec with sensible defaults.
    """

    base: BaseModelRef
    run: RunIdentity
    compute: ComputeProfile
    mode: TrainingMode
    tokenizer: TokenizerRef
    model_config: Mapping[str, Any]
    optimizer_config: Mapping[str, Any]
    # Exactly one of these must be set:
    # - ``data_manifest_uri`` points at a content-addressed JSON manifest under
    #   ``gs://marin-<region>/midtrain-manifests/data/...``.
    # - ``data_section_override`` is a raw Levanter ``data:`` dict, used when
    #   we need bit-identical val partitioning to a known-good prior run
    #   (the legacy 1e21/1e22 K=0.20 references; see
    #   ``experiments/midtrain_specs/data_sections/<mix>.json``).
    data_manifest_uri: str | None = None
    data_section_override: Mapping[str, Any] | None = None
    data_section_provenance: str | None = None
    seq_len: int = 4096
    # Eval cadence.
    eval_target_points: int = 40
    eval_min_steps: int = 25
    eval_max_steps: int = 200
    # Permanent-checkpoint cadence.
    permanent_fraction: float = 0.10
    min_permanent_steps: int = 50
    temp_save_interval: str = "10m"
    # Safety.
    banned_substrings: frozenset[str] = frozenset()
    require_bos_sample: bool = True
    allow_unsafe_no_val_split: bool = False
    # Resume (only set when intentionally resuming an existing attempt).
    expected_min_step: int | None = None
    allow_empty_resume: bool = False
    # Extra W&B tags rendered into the Levanter config.
    extra_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len!r}")
        data_sources_set = (self.data_manifest_uri is not None, self.data_section_override is not None)
        if sum(data_sources_set) != 1:
            raise ValueError(
                "MidtrainSpec requires exactly one of {data_manifest_uri, data_section_override}; "
                f"got data_manifest_uri={self.data_manifest_uri!r}, "
                f"data_section_override={'<set>' if self.data_section_override else None!r}"
            )
        if self.data_manifest_uri is not None and not self.data_manifest_uri.startswith("gs://"):
            raise ValueError(f"data_manifest_uri must be a gs:// URI, got {self.data_manifest_uri!r}")
        if self.data_section_override is not None and not self.data_section_provenance:
            raise ValueError(
                "data_section_override requires data_section_provenance "
                "(e.g. 'legacy:delphi-1e21-p33m67-9p25b-lr0.5-efbc63') so launches stay auditable."
            )
        if not self.model_config:
            raise ValueError("model_config must be non-empty")
        if not self.optimizer_config:
            raise ValueError("optimizer_config must be non-empty")
        if self.eval_target_points <= 0:
            raise ValueError(f"eval_target_points must be positive, got {self.eval_target_points!r}")
        if self.eval_min_steps <= 0 or self.eval_max_steps <= 0:
            raise ValueError(f"eval step bounds must be positive, got ({self.eval_min_steps}, {self.eval_max_steps})")
        if self.eval_min_steps > self.eval_max_steps:
            raise ValueError(f"eval_min_steps > eval_max_steps: {self.eval_min_steps} > {self.eval_max_steps}")
        if not (0 < self.permanent_fraction <= 1):
            raise ValueError(f"permanent_fraction must be in (0, 1], got {self.permanent_fraction!r}")
        if self.min_permanent_steps <= 0:
            raise ValueError(f"min_permanent_steps must be positive, got {self.min_permanent_steps!r}")
        if not self.temp_save_interval:
            raise ValueError("temp_save_interval must be non-empty")
        if self.expected_min_step is not None and self.expected_min_step < 0:
            raise ValueError(f"expected_min_step must be >= 0, got {self.expected_min_step!r}")
        if self.allow_empty_resume and (self.expected_min_step or 0) > 0:
            raise ValueError("allow_empty_resume=True is incompatible with expected_min_step > 0")
        if isinstance(self.mode, CooldownMode):
            if self.mode.resume.staged_output_path != self.run.output_path:
                raise ValueError(
                    "COOLDOWN run.output_path must equal cooldown_resume.staged_output_path; "
                    f"got run={self.run.output_path!r} vs staged={self.mode.resume.staged_output_path!r}"
                )

    @property
    def mode_kind(self) -> str:
        return self.mode.kind

    @property
    def is_resume(self) -> bool:
        return self.expected_min_step is not None


@dataclass(frozen=True)
class ResolvedMidtrainSpec:
    """A :class:`MidtrainSpec` with the data manifest loaded and budget resolved."""

    spec: MidtrainSpec
    data_manifest: DataCacheManifest | None
    resolved_budget: ResolvedBudget | None
    num_train_steps: int
    actual_tokens: int

    @property
    def mode(self) -> TrainingMode:
        return self.spec.mode

    @property
    def run(self) -> RunIdentity:
        return self.spec.run

    @property
    def uses_legacy_data_section(self) -> bool:
        return self.spec.data_section_override is not None


# ---------------------------------------------------------------------------
# Resolve + validate
# ---------------------------------------------------------------------------


def resolve_midtrain_spec(spec: MidtrainSpec) -> ResolvedMidtrainSpec:
    """Materialize the data manifest and resolve the budget (CPT) for one spec."""
    manifest = load_data_manifest(spec.data_manifest_uri) if spec.data_manifest_uri is not None else None
    mode = spec.mode
    if isinstance(mode, CptMode):
        resolved_budget = mode.resolve_budget(base=spec.base, batch_size=spec.compute.batch_size, seq_len=spec.seq_len)
        num_train_steps = resolved_budget.num_train_steps
        actual_tokens = resolved_budget.actual_tokens
    else:
        resolved_budget = None
        num_train_steps = mode.num_train_steps(base=spec.base, batch_size=spec.compute.batch_size, seq_len=spec.seq_len)
        actual_tokens = mode.actual_tokens(base=spec.base, batch_size=spec.compute.batch_size, seq_len=spec.seq_len)
    return ResolvedMidtrainSpec(
        spec=spec,
        data_manifest=manifest,
        resolved_budget=resolved_budget,
        num_train_steps=num_train_steps,
        actual_tokens=actual_tokens,
    )


def validate_midtrain_spec(resolved: ResolvedMidtrainSpec) -> None:
    """Run all cross-cutting + mode-specific guards."""
    spec = resolved.spec
    _assert_no_banned_substrings(spec)
    _assert_run_region_alignment(spec)
    _assert_compute_regions_collapse(spec)
    if resolved.data_manifest is not None:
        _assert_tokenizers_agree(spec, resolved.data_manifest)
        _assert_seq_len_matches_data(spec, resolved.data_manifest)
        _assert_safety(spec, resolved.data_manifest)
    else:
        _assert_legacy_data_section_consistent(spec)
    _assert_model_config_matches_base(spec)
    spec.mode.validate(base=spec.base, seq_len=spec.seq_len)
    if resolved.num_train_steps <= 0:
        raise ValueError(f"Resolved num_train_steps must be positive, got {resolved.num_train_steps!r}")


def _assert_no_banned_substrings(spec: MidtrainSpec) -> None:
    if not spec.banned_substrings:
        return
    candidates: list[str] = [spec.base.gcs_run_root, spec.base.verified_checkpoint_path, spec.base.hf_repo]
    candidates.extend(spec.mode.all_checkpoint_paths())
    for candidate in candidates:
        for banned in spec.banned_substrings:
            if banned and banned in candidate:
                raise ValueError(
                    f"Refusing to use banned substring {banned!r} in {candidate!r}. "
                    "See experiments/delphi_models.py for the canonical banned set."
                )


def _assert_run_region_alignment(spec: MidtrainSpec) -> None:
    run_region = spec.run.output_region
    if spec.data_manifest_uri is not None:
        data_uri_region = output_region(spec.data_manifest_uri)
        if data_uri_region != run_region:
            raise ValueError(
                f"Data manifest URI region {data_uri_region!r} does not match run output region {run_region!r}; "
                "training launch must consume a region-local manifest."
            )
        return
    # data_section_override: derive region(s) from each component's cache_dir.
    assert spec.data_section_override is not None
    components = spec.data_section_override.get("components") or {}
    cache_regions: set[str] = set()
    for comp in components.values():
        cache_dir = comp.get("cache_dir") if isinstance(comp, dict) else None
        if cache_dir and cache_dir.startswith("gs://marin-"):
            cache_regions.add(output_region(cache_dir))
    if cache_regions and cache_regions != {run_region}:
        raise ValueError(
            f"data_section_override component caches live in {sorted(cache_regions)!r}; "
            f"run output region is {run_region!r}. Stage caches in the run region first."
        )


def _assert_legacy_data_section_consistent(spec: MidtrainSpec) -> None:
    """Sanity-check a passthrough data section (tokenizer, val carve-out present)."""
    section = spec.data_section_override
    assert section is not None
    rendered_tokenizer = section.get("tokenizer")
    if rendered_tokenizer != spec.tokenizer.hf_repo:
        raise ValueError(
            f"data_section_override.tokenizer={rendered_tokenizer!r} "
            f"!= spec.tokenizer.hf_repo={spec.tokenizer.hf_repo!r}"
        )
    if not section.get("num_validation_sequences"):
        raise ValueError(
            "data_section_override has no num_validation_sequences — "
            "refusing to launch without an explicit val carve-out."
        )
    if section.get("shuffle_before_trainval_split") is not True:
        raise ValueError(
            "data_section_override.shuffle_before_trainval_split must be True for deterministic val partitioning."
        )


def _assert_compute_regions_collapse(spec: MidtrainSpec) -> None:
    if not spec.compute.regions:
        return
    if spec.compute.regions != (spec.run.output_region,):
        raise ValueError(
            f"ComputeProfile.regions={spec.compute.regions!r} must equal ({spec.run.output_region!r},) "
            "for a real launch; use a fresh attempt in a different region for compute fallback."
        )


def _assert_tokenizers_agree(spec: MidtrainSpec, manifest: DataCacheManifest) -> None:
    assert_tokenizer_compatible(spec.tokenizer, manifest.tokenizer)
    spec.mode.cross_check_tokenizer(spec.tokenizer)


def _assert_seq_len_matches_data(spec: MidtrainSpec, manifest: DataCacheManifest) -> None:
    if manifest.seq_len != spec.seq_len:
        raise ValueError(f"Data manifest seq_len={manifest.seq_len} does not match spec.seq_len={spec.seq_len}")


def _assert_safety(spec: MidtrainSpec, manifest: DataCacheManifest) -> None:
    if spec.require_bos_sample:
        for component in manifest.components:
            if not component.bos_sample:
                raise ValueError(
                    f"require_bos_sample=True but component {component.logical_name!r} "
                    "has no BOS sample in the manifest."
                )
    if not manifest.shuffle_before_trainval_split and not spec.allow_unsafe_no_val_split:
        raise ValueError(
            f"Data manifest {manifest.mix_name!r} was built with shuffle_before_trainval_split=False; "
            "set allow_unsafe_no_val_split=True with a documented reason to launch anyway."
        )


def _assert_model_config_matches_base(spec: MidtrainSpec) -> None:
    expected = {
        "hidden_dim": (spec.base.hidden_dim, "base.hidden_dim"),
        "num_layers": (spec.base.num_layers, "base.num_layers"),
        "max_seq_len": (spec.seq_len, "spec.seq_len"),
    }
    for key, value in expected.items():
        expected_value, expected_name = value
        rendered = spec.model_config.get(key)
        if rendered is not None and rendered != expected_value:
            raise ValueError(
                f"MidtrainSpec.model_config[{key!r}]={rendered!r} " f"does not match {expected_name}={expected_value!r}"
            )


def replace_run_identity(spec: MidtrainSpec, identity: RunIdentity) -> MidtrainSpec:
    """Return a copy of ``spec`` with a different :class:`RunIdentity`.

    Cooldown specs also need ``mode.resume.staged_output_path`` rewritten;
    this helper handles both atomically.
    """
    if isinstance(spec.mode, CooldownMode):
        new_resume = replace(spec.mode.resume, staged_output_path=identity.output_path)
        new_mode = replace(spec.mode, resume=new_resume)
        return replace(spec, run=identity, mode=new_mode)
    return replace(spec, run=identity)
