# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training-mode classes.

Each mode owns the parts of the launch that differ between CPT and true
midtraining: which checkpoint to consume at init, how to size the run,
which startup-log lines prove the job started correctly, and whether a
staging step has to run before launch.

This replaces the original ``MidtrainMode = StrEnum`` plus parallel
optional ``cpt_init`` / ``cooldown_resume`` / ``budget`` fields on
``MidtrainSpec`` with a closed union of two concrete classes. Consumers
dispatch via duck-typed method calls (or ``isinstance`` when needed).

``TrainingMode`` is the type alias ``CptMode | CooldownMode``. There is
no abstract base class; both implementations expose the same method set
by convention. The redesign doc (``midtraining_redesign.md``) names the
two modes explicitly and forbids ambiguous "hybrid" launches.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeAlias

from marin.midtraining.budget import (
    BudgetPolicy,
    ResolvedBudget,
    resolve_cpt_budget,
)
from marin.midtraining.tokenizers import TokenizerRef, assert_tokenizer_compatible

CHECKPOINT_INIT_MODE_MODEL_ONLY = "model_only"
CHECKPOINT_INIT_MODE_FULL_STATE = "full_state"

# ---------------------------------------------------------------------------
# Init-source value objects (shared between modes via composition)
# ---------------------------------------------------------------------------


class CheckpointSourceKind(StrEnum):
    """How CPT obtains its initial model weights."""

    NATIVE_LEVANTER = "native_levanter"
    HF_WEIGHTS = "hf_weights"


@dataclass(frozen=True)
class CheckpointOverride:
    """CPT-only escape hatch for using a non-registry init checkpoint."""

    checkpoint_path: str
    reason: str
    run_name_suffix: str
    expected_hidden_dim: int
    expected_num_layers: int
    expected_seq_len: int
    expected_tokenizer: TokenizerRef
    allow_hf_weights: bool = False

    def __post_init__(self) -> None:
        if not self.checkpoint_path:
            raise ValueError("CheckpointOverride.checkpoint_path must be non-empty")
        if not self.reason or len(self.reason) < 10:
            raise ValueError(
                f"CheckpointOverride.reason must be a meaningful explanation (>=10 chars), got {self.reason!r}"
            )
        if not self.run_name_suffix or not self.run_name_suffix.startswith("-"):
            raise ValueError(
                "CheckpointOverride.run_name_suffix must start with '-' so the override is visible in the run name"
            )
        for field_name in ("expected_hidden_dim", "expected_num_layers", "expected_seq_len"):
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"CheckpointOverride.{field_name} must be positive, got {value!r}")


@dataclass(frozen=True)
class CptInit:
    """CPT initialization source. Exactly one of registry / override / HF set."""

    source_kind: CheckpointSourceKind
    registry_model: Any | None = None  # BaseModelRef satisfier
    checkpoint_override: CheckpointOverride | None = None
    hf_repo: str | None = None
    hf_revision: str | None = None
    reset_optimizer: bool = True
    reset_data_loader: bool = True

    def __post_init__(self) -> None:
        sources = (
            self.registry_model is not None,
            self.checkpoint_override is not None,
            self.hf_repo is not None or self.hf_revision is not None,
        )
        if sum(sources) != 1:
            raise ValueError(
                "CptInit must set exactly one of {registry_model, checkpoint_override, (hf_repo, hf_revision)}; "
                f"got registry={sources[0]}, override={sources[1]}, hf={sources[2]}"
            )
        if self.hf_repo is not None and (not self.hf_revision or self.hf_revision == "main"):
            raise ValueError(
                f"CptInit HF source requires a pinned hf_revision (got {self.hf_revision!r}); "
                "'main' or empty is rejected to keep model identity deterministic."
            )
        if (self.hf_repo is None) != (self.hf_revision is None):
            raise ValueError("CptInit hf_repo and hf_revision must be set together")
        if self.source_kind == CheckpointSourceKind.NATIVE_LEVANTER:
            if self.hf_repo is not None:
                raise ValueError("CheckpointSourceKind.NATIVE_LEVANTER cannot pair with hf_repo")
            if self.checkpoint_override is not None and self.checkpoint_override.allow_hf_weights:
                raise ValueError("NATIVE_LEVANTER source kind rejects CheckpointOverride.allow_hf_weights=True")
        elif self.source_kind == CheckpointSourceKind.HF_WEIGHTS:
            if self.registry_model is not None:
                raise ValueError("CheckpointSourceKind.HF_WEIGHTS cannot pair with a registry_model")
            if self.checkpoint_override is not None and not self.checkpoint_override.allow_hf_weights:
                raise ValueError(
                    "CheckpointSourceKind.HF_WEIGHTS with a CheckpointOverride requires allow_hf_weights=True"
                )
        if not self.reset_optimizer:
            raise ValueError(
                "CPT must reset the optimizer (reset_optimizer=True); use cooldown mode to preserve optimizer state."
            )

    def resolved_checkpoint_path(self) -> str | None:
        """Native checkpoint path for NATIVE_LEVANTER sources; ``None`` for HF."""
        if self.checkpoint_override is not None:
            if self.checkpoint_override.allow_hf_weights:
                return None
            return self.checkpoint_override.checkpoint_path
        if self.registry_model is not None:
            return self.registry_model.verified_checkpoint_path
        return None


@dataclass(frozen=True)
class CooldownResume:
    """Pre-stage a pretrain checkpoint and resume naturally."""

    pretrain_checkpoint_path: str
    resume_step: int
    staged_output_path: str
    preserve_optimizer: bool = True
    preserve_scheduler_count: bool = True
    preserve_state_step: bool = True

    def __post_init__(self) -> None:
        if not self.pretrain_checkpoint_path.startswith("gs://") and not self.pretrain_checkpoint_path.startswith(
            "mirror://"
        ):
            raise ValueError(
                "CooldownResume.pretrain_checkpoint_path must be a gs:// or mirror:// URI, got "
                f"{self.pretrain_checkpoint_path!r}"
            )
        if self.resume_step <= 0:
            raise ValueError(f"CooldownResume.resume_step must be positive, got {self.resume_step!r}")
        if not self.staged_output_path.startswith("gs://"):
            raise ValueError(
                f"CooldownResume.staged_output_path must be a concrete gs:// URI, got {self.staged_output_path!r}"
            )
        if "/checkpoints/step-" in self.staged_output_path:
            raise ValueError(
                f"staged_output_path must be a run root, not a concrete checkpoint: {self.staged_output_path!r}"
            )
        if not (self.preserve_optimizer and self.preserve_scheduler_count and self.preserve_state_step):
            raise ValueError(
                "Cooldown must preserve optimizer, scheduler count, and state step "
                "(set preserve_* to True or use CPT mode)."
            )

    @property
    def staged_checkpoint_path(self) -> str:
        """Where ``stage-cooldown`` materializes the resume checkpoint."""
        return f"{self.staged_output_path}/checkpoints/step-{self.resume_step}"


# ---------------------------------------------------------------------------
# CPT mode
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CptMode:
    """Continued pretraining: load model weights only, fresh optimizer state."""

    init: CptInit
    budget: BudgetPolicy
    lr_factor: float | None = None
    lr_multiplier: float = 1.0
    min_lr_ratio: float = 0.0
    warmup_tokens: int | None = None
    warmup_fraction: float | None = None

    kind = "cpt"

    def __post_init__(self) -> None:
        if self.lr_factor is not None and self.lr_factor <= 0:
            raise ValueError(f"CptMode.lr_factor must be positive, got {self.lr_factor!r}")
        if self.lr_multiplier <= 0:
            raise ValueError(f"CptMode.lr_multiplier must be positive, got {self.lr_multiplier!r}")
        if not (0 <= self.min_lr_ratio <= 1):
            raise ValueError(f"CptMode.min_lr_ratio must be in [0, 1], got {self.min_lr_ratio!r}")
        if self.warmup_tokens is not None and self.warmup_tokens <= 0:
            raise ValueError(f"warmup_tokens must be positive, got {self.warmup_tokens!r}")
        if self.warmup_fraction is not None and not (0 < self.warmup_fraction < 1):
            raise ValueError(f"warmup_fraction must be in (0, 1), got {self.warmup_fraction!r}")
        if self.warmup_tokens is not None and self.warmup_fraction is not None:
            raise ValueError("CptMode: set at most one of warmup_tokens or warmup_fraction")

    # --- mode protocol -----------------------------------------------------

    def validate(self, *, base: Any, seq_len: int) -> None:
        """Cross-check CPT-specific fields against the base registry entry."""
        if self.init.registry_model is not None and self.init.registry_model.flops_key != base.flops_key:
            raise ValueError(
                f"CptInit.registry_model.flops_key={self.init.registry_model.flops_key!r} "
                f"does not match spec.base.flops_key={base.flops_key!r}"
            )
        override = self.init.checkpoint_override
        if override is not None:
            if override.expected_hidden_dim != base.hidden_dim:
                raise ValueError(
                    f"CheckpointOverride expected_hidden_dim={override.expected_hidden_dim} "
                    f"does not match base.hidden_dim={base.hidden_dim}"
                )
            if override.expected_num_layers != base.num_layers:
                raise ValueError(
                    f"CheckpointOverride expected_num_layers={override.expected_num_layers} "
                    f"does not match base.num_layers={base.num_layers}"
                )
            if override.expected_seq_len != seq_len:
                raise ValueError(
                    f"CheckpointOverride expected_seq_len={override.expected_seq_len} "
                    f"does not match spec.seq_len={seq_len}"
                )

    def cross_check_tokenizer(self, tokenizer: TokenizerRef) -> None:
        if self.init.checkpoint_override is not None:
            assert_tokenizer_compatible(tokenizer, self.init.checkpoint_override.expected_tokenizer)

    def resolve_budget(self, *, base: Any, batch_size: int, seq_len: int) -> ResolvedBudget:
        return resolve_cpt_budget(
            self.budget,
            base_flops_key=base.flops_key,
            base_pretrain_tokens=base.tokens,
            batch_size=batch_size,
            seq_len=seq_len,
        )

    def num_train_steps(self, *, base: Any, batch_size: int, seq_len: int) -> int:
        return self.resolve_budget(base=base, batch_size=batch_size, seq_len=seq_len).num_train_steps

    def actual_tokens(self, *, base: Any, batch_size: int, seq_len: int) -> int:
        return self.resolve_budget(base=base, batch_size=batch_size, seq_len=seq_len).actual_tokens

    def requires_staging(self) -> bool:
        return False

    def expected_min_step(self) -> int | None:
        return None

    def init_artifact_uri(self) -> str | None:
        """URI preflight should probe for the CPT init checkpoint."""
        if self.init.source_kind == CheckpointSourceKind.NATIVE_LEVANTER:
            return self.init.resolved_checkpoint_path()
        return self.init.hf_repo

    def render_init_section(self) -> Mapping[str, Any]:
        """``TrainLmConfig`` YAML keys that pin the CPT initialization.

        HF sources render as ``<repo>@<revision>`` so Levanter's
        :class:`RepoRef` parses the pinned commit sha and forwards it to
        ``AutoConfig.from_pretrained(..., revision=)`` and
        ``hf_hub_download(..., revision=)``. No local download needed —
        Levanter streams safetensors over ``hf://`` URLs.
        """
        init_section: dict[str, Any] = {"checkpoint_init_mode": CHECKPOINT_INIT_MODE_MODEL_ONLY}
        resolved_ckpt = self.init.resolved_checkpoint_path()
        if resolved_ckpt is not None:
            init_section["initialize_from_checkpoint_path"] = resolved_ckpt
        elif self.init.hf_repo is not None:
            assert self.init.hf_revision is not None
            init_section["initialize_from_hf"] = f"{self.init.hf_repo}@{self.init.hf_revision}"
        else:
            raise ValueError("CptInit has no resolvable source; mode validation should have caught this")
        return init_section

    @staticmethod
    def expected_startup_lines() -> tuple[str, ...]:
        return (
            "Using output path",
            "Using run ID",
            "No checkpoints found",
            "Loading checkpoint from",
            "checkpoint_init_mode=model_only",
        )

    @staticmethod
    def forbidden_startup_lines() -> tuple[str, ...]:
        return ("Starting from scratch",)

    def all_checkpoint_paths(self) -> tuple[str, ...]:
        """Paths to check against banned-substring rules."""
        paths: list[str] = []
        if self.init.checkpoint_override is not None:
            paths.append(self.init.checkpoint_override.checkpoint_path)
        resolved = self.init.resolved_checkpoint_path()
        if resolved is not None:
            paths.append(resolved)
        return tuple(paths)


# ---------------------------------------------------------------------------
# Cooldown mode
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CooldownMode:
    """True midtraining: resume a pretrain checkpoint with full optimizer state."""

    resume: CooldownResume
    stop_step_override: int | None = None

    kind = "cooldown"

    def __post_init__(self) -> None:
        if self.stop_step_override is not None and self.stop_step_override <= self.resume.resume_step:
            raise ValueError(
                f"stop_step_override {self.stop_step_override!r} must exceed resume_step {self.resume.resume_step!r}"
            )

    # --- mode protocol -----------------------------------------------------

    def validate(self, *, base: Any, seq_len: int) -> None:
        if self.resume.resume_step >= base.num_train_steps:
            raise ValueError(
                f"CooldownResume.resume_step={self.resume.resume_step} must be < "
                f"base.num_train_steps={base.num_train_steps}"
            )
        if self.stop_step_override is not None and self.stop_step_override > base.num_train_steps:
            raise ValueError(
                f"stop_step_override={self.stop_step_override} exceeds " f"base.num_train_steps={base.num_train_steps}"
            )

    def cross_check_tokenizer(self, tokenizer: TokenizerRef) -> None:
        return None

    def num_train_steps(self, *, base: Any, batch_size: int, seq_len: int) -> int:
        return self.stop_step_override or base.num_train_steps

    def actual_tokens(self, *, base: Any, batch_size: int, seq_len: int) -> int:
        return self.num_train_steps(base=base, batch_size=batch_size, seq_len=seq_len) * batch_size * seq_len

    def requires_staging(self) -> bool:
        return True

    def expected_min_step(self) -> int | None:
        return self.resume.resume_step

    def init_artifact_uri(self) -> str | None:
        return self.resume.pretrain_checkpoint_path

    def render_init_section(self) -> Mapping[str, Any]:
        return {
            "initialize_from_checkpoint_path": None,
            "checkpoint_init_mode": CHECKPOINT_INIT_MODE_FULL_STATE,
        }

    @staticmethod
    def expected_startup_lines() -> tuple[str, ...]:
        return (
            "Using output path",
            "Using run ID",
            "Discovered latest checkpoint at",
            "Resuming training from step",
        )

    @staticmethod
    def forbidden_startup_lines() -> tuple[str, ...]:
        return (
            "Starting from scratch",
            "No checkpoints found",
            "Loading checkpoint from",
        )

    def all_checkpoint_paths(self) -> tuple[str, ...]:
        return (self.resume.pretrain_checkpoint_path,)


# ---------------------------------------------------------------------------
# Public union
# ---------------------------------------------------------------------------


TrainingMode: TypeAlias = CptMode | CooldownMode
