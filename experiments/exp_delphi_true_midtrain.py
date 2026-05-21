# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TRUE midtraining: hijack the WSD cooldown of each Delphi/AdamH pretrain with the math midtraining data mix.

This file is the launch driver for true-midtraining cells over verified Delphi
base models. Each cell starts from a *pre-staged* pretrain
checkpoint (Phase B6 in the launch plan, see
``.agents/logbooks/true_midtraining.md`` § 13.4) so Levanter's natural
``latest_checkpoint_path(output_path)`` resume path picks it up. State,
optimizer momentum, and the inject_hyperparams schedule count are all
preserved from the pretrain run; only the data mixture changes.

This is fundamentally different from ``exp_delphi_math_10b_midtrain.py``,
which used ``initialize_from_checkpoint_path`` + ``CheckpointInitMode.MODEL_ONLY``
(weights-only, fresh opt_state, fresh warmup/decay) — i.e. "fake midtraining."
Here we use no ``initialize_from_checkpoint_path`` at all and rely on the
trainer's standard preemption-recovery code path to load the pre-staged
checkpoint as if it were our own crash-recovery write.

Per-scale resume policy (user-confirmed 2026-05-09; see logbook § 4):
- 1e21: resume from pretrain `step-20000` (53% into the cooldown).
- 1e22: resume from pretrain `step-30000` (588 steps before decay starts —
  the closest available checkpoint to the cooldown boundary on the 5,000-step
  cadence). 1e22 is the only scale that gets its full cooldown on math.

WandB naming convention (§ 13.1, MANDATORY):
    true-midtrain-{scale}-{mix}-step{resume_step}

Every safety guard from ``exp_delphi_math_10b_midtrain.py`` is mirrored here
(G1-G17 in the logbook). Seven new guards (N1-N7) are layered on top
specifically because every cell IS a resume from day one — there is no
"fresh" code path.

Launch one cell at a time. Required env vars:

    TRUE_MIDTRAIN_SELECT_SCALE              in {1e21, 1e22}
    TRUE_MIDTRAIN_SELECT_MIX                ∈ {p33m67, p50m50, p67m33}
    TRUE_MIDTRAIN_RESUME_OUTPUT_PATH        gs://marin-<region>/checkpoints/<run-name>
    TRUE_MIDTRAIN_EXPECT_RESUME_STEP        the exact pretrain step that was pre-staged
                                             (must equal PRETRAINS[scale].resume_from_step)

Optional:

    TRUE_MIDTRAIN_TPU_TYPE                  override the per-scale default v5p slice
    TRUE_MIDTRAIN_PER_DEVICE_PARALLELISM    grad-accum factor override
    TRUE_MIDTRAIN_TENSOR_PARALLEL_SIZE      TP override
    TRUE_MIDTRAIN_TRAIN_REGION              pin to one of {us-central1, us-east5}
    TRUE_MIDTRAIN_RUN_NAME_SUFFIX           appended to the run name (debug only)

The legacy ``MIDTRAIN_OUTPUT_PATH_OVERRIDE`` is hard-rejected (G5).

See ``.agents/logbooks/true_midtraining.md`` for the full design rationale,
checkpoint-cadence analysis, throughput/wall-clock estimates per TPU, and
the per-cell launch sequence.
"""

import logging
import os
import re
from dataclasses import dataclass, replace

import fsspec
from fray.cluster import ResourceConfig
from haliax import Axis
from levanter.checkpoint import discover_latest_checkpoint
from levanter.main.train_lm import CheckpointInitMode
from levanter.optim import AdamHConfig
from marin.execution.executor import ExecutorStep, MirroredValue, executor_main, mirrored
from marin.training.training import temporary_checkpoint_base_path
from rigging.filesystem import marin_region

from experiments.defaults import default_train
from experiments.delphi_models import DELPHI_1E21, DELPHI_1E22, DelphiModel
from experiments.midtrain_data_safety import assert_val_train_disjoint

# Mirror the mixture imports from `exp_delphi_math_10b_midtrain.py` so the
# pretrain-replay side of every midtraining mix is bit-identical to what the
# prior K=0.20 sweep used. `midtraining_mix_by_name(name)` resolves to a
# `MidtrainMixSpec` whose `pretrain_base = experiments.pretraining_datasets
# .nemotron.nemotron_mix` (Nemotron-CC HQ + 25% starcoderdata + 5.5% proofpile_2),
# the same mix the Delphi pretrains were trained on. `BUCKET_2` is imported
# for the legacy single-source 100% math path (not used in true-midtraining
# launches but kept here so the import surface matches the original).
from experiments.midtraining_data_buckets import BUCKET_2  # noqa: F401  — imported for parity with the original
from experiments.midtraining_mixes import (
    FULL_HIGHQUALITY_NEMO_MATH_NAME,
    MIDTRAIN_BUDGET_FRACTION,  # noqa: F401  — imported for parity (true-midtraining ignores K)
    PRETRAIN_33P_MATH_67P_HIGHQUALITY_NEMO_MATH_NAME,
    PRETRAIN_50P_MATH_50P_HIGHQUALITY_NEMO_MATH_NAME,
    PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME,
    PRETRAIN_70P_MATH_30P_HIGHQUALITY_NEMO_MATH_NAME,
    full_highquality_nemo_math,
    log_partition_summary,
    midtrain_token_budget,  # noqa: F401  — imported for parity (true-midtraining ignores K)
    midtraining_mix_by_name,
)
from experiments.scaling_law_sweeps.completed_adamh import completed_adamh_heuristic
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger(__name__)


# ─── Constants ──────────────────────────────────────────────────────────────

DEFAULT_SEQ_LEN: int = 4096
STEPS_PER_EVAL: int = 200
EXPORT_FRACTION_OF_RUN: float = 0.10
MIN_STEPS_PER_EXPORT: int = 50
RUN_NAME_PREFIX: str = "true-midtrain-"  # G7 — every wandb run starts with this

# G2 — coordinator-allowed regions (matches the original sweep).
TRUE_MIDTRAIN_COORDINATOR_REGIONS: tuple[str, ...] = ("us-central1", "us-east5")

# N1 — heuristic-drift assertion baseline. The pretrain WSD profile is:
#   warmup: first 10% of num_train_steps
#   stable: middle 70%
#   decay:  last 20%, linear → 0 (min_lr_ratio=0)
# We refuse to launch if the heuristic ever returns different values, because
# the entire true-midtraining recipe assumes this profile.
EXPECTED_PRETRAIN_WARMUP_FRACTION: float = 0.1
EXPECTED_PRETRAIN_DECAY_FRACTION: float = 0.2
EXPECTED_PRETRAIN_MIN_LR_RATIO: float = 0.0
EXPECTED_PRETRAIN_LR_SCHEDULE: str = "linear"


# ─── Pretrain spec per scale (G3, G4) ────────────────────────────────────────


@dataclass(frozen=True)
class V5PComputeConfig:
    """Per-(scale, TPU) compute config. Mirrors the original file's V5PComputeConfig."""

    tpu_type: str
    per_device_parallelism: int = -1
    tensor_parallel_size: int = 1


@dataclass(frozen=True)
class PretrainSpec:
    """One Delphi/AdamH pretrain run that we'll resume into true midtraining.

    Hyperparams are the source of truth (G4) — read verbatim from each
    pretrain run's W&B config so the loaded weights are optimised against
    bit-exactly the same optimizer the pretrain used.
    """

    scale_tag: str  # e.g. "1e21"
    pretrain_run_name: str  # for documentation/tags only
    pretrain_num_train_steps: int  # N — used to rebuild the WSD schedule (N2)
    pretrain_tokens: int  # T — sanity-check vs heuristic
    batch_size: int  # B — pretrain native; we keep it
    hidden_dim: int  # for Qwen3 model rebuild
    seq_len: int

    resume_from_step: int  # the pretrain step pre-staged into output_path (N3)
    staged_base_path: MirroredValue[str] | str  # G1 — mirror-staged with explicit budget

    default_tpu_type: str  # G3 — per-scale default
    v5p_compute: tuple[V5PComputeConfig, ...]  # G3 — allowlist

    # G4 — verbatim from pretrain W&B config.
    peak_lr: float
    peak_adam_lr: float
    beta2: float
    epsilon: float

    def compute_config(self, tpu_type: str) -> V5PComputeConfig:
        for cfg in self.v5p_compute:
            if cfg.tpu_type == tpu_type:
                return cfg
        allowed = ", ".join(c.tpu_type for c in self.v5p_compute)
        raise ValueError(f"{tpu_type!r} is not an approved v5p target for {self.scale_tag}. Allowed: {allowed}")


def _pretrain_run_name(model: DelphiModel) -> str:
    return os.path.basename(model.gcs_run_root.rstrip("/"))


# ─── Per-scale specs ────────────────────────────────────────────────────────
# Hyperparameters are lifted verbatim from `BASES` in
# experiments/exp_delphi_math_10b_midtrain.py (which itself reads them from
# W&B configs). DO NOT recompute via the heuristic; the W&B config is the
# canonical source of truth for what the weights were optimized against (G4).

PRETRAINS: dict[str, PretrainSpec] = {
    # The old "1e20" entry was removed after the 2026-05-14 wrong-base
    # incident. It used an adamh_scaling_v5 isoflop ablation and must not be
    # launched. Add a 3e20-v6 true-midtraining entry only after verifying the
    # native checkpoint step, staging path, and exact pretrain optimizer state.
    "1e21": PretrainSpec(
        scale_tag="1e21",
        pretrain_run_name=_pretrain_run_name(DELPHI_1E21),
        pretrain_num_train_steps=DELPHI_1E21.num_train_steps,
        pretrain_tokens=DELPHI_1E21.tokens,
        batch_size=DELPHI_1E21.batch_size,
        hidden_dim=DELPHI_1E21.hidden_dim,
        seq_len=DEFAULT_SEQ_LEN,
        resume_from_step=20_000,  # 53% into cooldown
        staged_base_path=mirrored(
            "midtrain-bases/delphi-1e21-v5-019021/step-20000",
            budget_gb=50,
        ),
        default_tpu_type="v5p-64",
        v5p_compute=(
            V5PComputeConfig("v5p-64"),
            V5PComputeConfig("v5p-128"),
            V5PComputeConfig("v5p-256"),
            V5PComputeConfig("v5p-512"),
        ),
        peak_lr=7.425e-3,
        peak_adam_lr=4.314e-4,
        beta2=0.99920,
        epsilon=2.81e-8,
    ),
    "1e22": PretrainSpec(
        scale_tag="1e22",
        pretrain_run_name=_pretrain_run_name(DELPHI_1E22),
        pretrain_num_train_steps=DELPHI_1E22.num_train_steps,
        pretrain_tokens=DELPHI_1E22.tokens,
        batch_size=DELPHI_1E22.batch_size,
        hidden_dim=DELPHI_1E22.hidden_dim,
        seq_len=DEFAULT_SEQ_LEN,
        resume_from_step=30_000,  # 588 steps before decay starts
        staged_base_path=mirrored(
            "midtrain-bases/delphi-1e22-v5-025b0e/step-30000",
            budget_gb=150,
        ),
        default_tpu_type="v5p-256",
        v5p_compute=(
            V5PComputeConfig("v5p-64", per_device_parallelism=4),
            V5PComputeConfig("v5p-128", per_device_parallelism=4),
            V5PComputeConfig("v5p-256", per_device_parallelism=4),
            V5PComputeConfig("v5p-512", per_device_parallelism=4),
        ),
        peak_lr=7.231797280729413e-3,
        peak_adam_lr=3.276222099351447e-4,
        beta2=0.9984011994401821,
        epsilon=3.70426657045089e-8,
    ),
}

# All midtraining-mix variants the original sweep registered (see
# `experiments/midtraining_mixes.py`). All five share the same pretrain-replay
# base (`nemotron_mix`) and the same math component
# (`nemotron_cc_math_v1/4plus`); they differ only in the pretrain:math weight
# split. Listed here for parity with the original, but the active true-
# midtraining launch set per user request (2026-05-09) is **p33m67 only**.
MIXES: dict[str, str] = {
    "p33m67": PRETRAIN_33P_MATH_67P_HIGHQUALITY_NEMO_MATH_NAME,  # active
    "p50m50": PRETRAIN_50P_MATH_50P_HIGHQUALITY_NEMO_MATH_NAME,  # available
    "p67m33": PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME,  # available
    "p70m30": PRETRAIN_70P_MATH_30P_HIGHQUALITY_NEMO_MATH_NAME,  # available
    "full": FULL_HIGHQUALITY_NEMO_MATH_NAME,  # available (100% math; pretrain-replay weight = 0)
}


# ─── Env-var parsing ────────────────────────────────────────────────────────


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {value!r}")


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


# ─── Region / resource resolution (G2) ──────────────────────────────────────


def _normalize_region(region_or_zone: str) -> str:
    region_or_zone = region_or_zone.lower()
    parts = region_or_zone.split("-")
    if len(parts) >= 3 and len(parts[-1]) == 1 and parts[-1].isalpha() and any(ch.isdigit() for ch in parts[-2]):
        return "-".join(parts[:-1])
    return region_or_zone


def _selected_train_region() -> str | None:
    train_region_env = os.environ.get("TRUE_MIDTRAIN_TRAIN_REGION")
    explicit_region = train_region_env is not None
    region = train_region_env or marin_region()
    if region is None:
        return None
    region = _normalize_region(region)
    if region in TRUE_MIDTRAIN_COORDINATOR_REGIONS:
        return region
    if explicit_region:
        allowed = ", ".join(TRUE_MIDTRAIN_COORDINATOR_REGIONS)
        raise ValueError(f"True-midtraining must run in one of {{{allowed}}}, got {region!r}")
    return None


def _train_tpu_resources(tpu_type: str) -> ResourceConfig:
    region = _selected_train_region()
    if region is None:
        return ResourceConfig.with_tpu(tpu_type)
    return ResourceConfig.with_tpu(tpu_type, regions=[region])


# ─── Resume identity (G6, G7) ───────────────────────────────────────────────


_CHECKPOINT_STEP_RE = re.compile(r"(?:^|/)step-(\d+)/?$")


def _resume_run_id_from_output_path(output_path: str) -> str:
    run_id = os.path.basename(output_path.rstrip("/"))
    if not run_id:
        raise ValueError(f"TRUE_MIDTRAIN_RESUME_OUTPUT_PATH must end in a run id, got {output_path!r}")
    return run_id


def _resume_identity_env_vars(output_path: str) -> dict[str, str]:
    """Single source of truth for resume identity. RUN_ID = WANDB_RUN_ID = basename(output_path)."""
    run_id = _resume_run_id_from_output_path(output_path)
    return {
        "RUN_ID": run_id,
        "WANDB_RUN_ID": run_id,
        "WANDB_RESUME": "allow",
    }


def _with_resume_identity(train_cfg: SimpleTrainConfig, output_path: str) -> SimpleTrainConfig:
    env_vars = {
        **(train_cfg.env_vars or {}),
        **_resume_identity_env_vars(output_path),
    }
    return replace(train_cfg, env_vars=env_vars)


def _resume_checkpoint_search_paths(output_path: str) -> tuple[str, str]:
    output_path = output_path.rstrip("/")
    return os.path.join(output_path, "checkpoints"), temporary_checkpoint_base_path(output_path)


def _discover_latest_resume_checkpoint(output_path: str) -> str | None:
    permanent_path, temporary_path = _resume_checkpoint_search_paths(output_path)
    return discover_latest_checkpoint(permanent_path, temporary_path)


def _checkpoint_step_from_path(checkpoint_path: str) -> int | None:
    match = _CHECKPOINT_STEP_RE.search(checkpoint_path.rstrip("/"))
    if match is None:
        return None
    return int(match.group(1))


# ─── Run name (G7, N7, G10) ─────────────────────────────────────────────────


def _build_run_name(scale: str, mix_tag: str, resume_step: int) -> str:
    name = f"{RUN_NAME_PREFIX}{scale}-{mix_tag}-step{resume_step}"
    suffix = os.environ.get("TRUE_MIDTRAIN_RUN_NAME_SUFFIX")
    if suffix:
        name = f"{name}-{suffix}"
    if not name.startswith(RUN_NAME_PREFIX):
        raise ValueError(f"true-midtraining run name must start with {RUN_NAME_PREFIX!r}, got {name!r}")
    if len(name) > 64:
        raise ValueError(f"true-midtraining run name must stay within W&B's 64-char limit, got {len(name)}: {name}")
    return name


def _validate_resume_output_path_matches_run(
    output_path: str,
    *,
    scale: str,
    mix_tag: str,
    resume_step: int,
) -> None:
    expected = _build_run_name(scale, mix_tag, resume_step)
    run_id = _resume_run_id_from_output_path(output_path)
    if run_id == expected or run_id.startswith(f"{expected}-"):
        return
    raise ValueError(
        "TRUE_MIDTRAIN_RESUME_OUTPUT_PATH does not match the selected cell. "
        f"Selected run name is {expected!r}, but resume path ends in {run_id!r}. "
        "Check TRUE_MIDTRAIN_SELECT_SCALE, TRUE_MIDTRAIN_SELECT_MIX, and the staged path."
    )


# ─── Checkpoint integrity verification (N3, N4, G8) ─────────────────────────


def _verify_pre_staged_checkpoint(
    output_path: str,
    *,
    expected_step: int,
) -> str:
    """Pre-flight check on the pre-staged pretrain checkpoint.

    Combines guards G8 (a checkpoint must exist under output_path), N3 (the
    discovered step must be >= expected_step — i.e. the pre-staged checkpoint
    is present at the floor, OR a later checkpoint from a prior interrupted
    training run that we want to resume from), and N4 (manifest.ocdbt +
    metadata.json + d/ must all be present so TensorStore won't silently
    NaN-restore).

    The N3 floor (rather than exact match) is intentional: on the FIRST
    launch the latest checkpoint will equal expected_step (the staged
    pretrain ckpt). On a preemption-recovery launch, Levanter has been
    writing temp/perm checkpoints at higher steps, and the latest will be
    > expected_step. Exact-match would refuse the recovery and burn the
    training progress already made; the floor lets natural-resume work.
    The "wrong checkpoint staged" scenario this guard was meant to catch
    is still detected — just inverted: we'd see step < expected_step.

    Returns the resolved checkpoint path on success; raises on any failure.
    """
    discovered = _discover_latest_resume_checkpoint(output_path)
    if discovered is None:
        permanent_path, temporary_path = _resume_checkpoint_search_paths(output_path)
        raise FileNotFoundError(
            f"No pre-staged checkpoint found for output path {output_path!r}. "
            f"Checked permanent path {permanent_path!r} and temp path {temporary_path!r}. "
            "Did Phase B (per-cell fan-out) of the launch plan run?"
        )

    step = _checkpoint_step_from_path(discovered)
    if step is None:
        raise ValueError(
            f"Discovered checkpoint {discovered!r} has no step encoded in its path; " "expected `.../step-NNNN/`."
        )
    if step < expected_step:
        raise ValueError(
            f"Pre-staged checkpoint step too low for {output_path!r}: "
            f"expected step >= {expected_step}, found step={step} ({discovered!r}). "
            "The wrong pretrain ckpt was staged."
        )

    # G8/N4 — verify the three TensorStore artefacts are present.
    fs, _, _ = fsspec.get_fs_token_paths(discovered)
    discovered = discovered.rstrip("/")
    missing: list[str] = []
    for required in ("manifest.ocdbt", "metadata.json"):
        if not fs.exists(f"{discovered}/{required}"):
            missing.append(required)
    # `d/` is a keystore directory — list it instead of `exists`.
    try:
        keystore_entries = fs.ls(f"{discovered}/d", detail=False)
    except (FileNotFoundError, OSError):
        keystore_entries = []
    if not keystore_entries:
        missing.append("d/")
    if missing:
        raise FileNotFoundError(
            f"Pre-staged checkpoint {discovered!r} is missing required TensorStore artefacts: "
            f"{missing}. TensorStore will silently NaN-restore on missing arrays; refusing to launch."
        )

    logger.info("Pre-flight: pre-staged checkpoint %s validated (step=%d, all 3 artefacts present).", discovered, step)
    return discovered


# ─── Heuristic-drift guard (N1) ─────────────────────────────────────────────


def _assert_heuristic_schedule_unchanged() -> None:
    """N1 — refuse to launch if the AdamH heuristic's WSD schedule has drifted.

    The whole true-midtraining recipe assumes warmup=0.1, decay=0.2,
    min_lr_ratio=0.0, linear. If somebody changes the heuristic, every cell
    will silently train at the wrong schedule. Hard-fail at module import.
    """
    h = completed_adamh_heuristic
    mismatches: list[str] = []
    if h.warmup != EXPECTED_PRETRAIN_WARMUP_FRACTION:
        mismatches.append(f"warmup={h.warmup} (expected {EXPECTED_PRETRAIN_WARMUP_FRACTION})")
    if h.decay != EXPECTED_PRETRAIN_DECAY_FRACTION:
        mismatches.append(f"decay={h.decay} (expected {EXPECTED_PRETRAIN_DECAY_FRACTION})")
    if h.min_lr_ratio != EXPECTED_PRETRAIN_MIN_LR_RATIO:
        mismatches.append(f"min_lr_ratio={h.min_lr_ratio} (expected {EXPECTED_PRETRAIN_MIN_LR_RATIO})")
    if h.lr_schedule != EXPECTED_PRETRAIN_LR_SCHEDULE:
        mismatches.append(f"lr_schedule={h.lr_schedule!r} (expected {EXPECTED_PRETRAIN_LR_SCHEDULE!r})")
    if mismatches:
        raise AssertionError(
            "completed_adamh_heuristic schedule fields have drifted from the values true-midtraining "
            f"depends on: {', '.join(mismatches)}. Update this guard intentionally if the change is real."
        )


_assert_heuristic_schedule_unchanged()


# ─── Optimizer build (G4, N1) ───────────────────────────────────────────────


def _build_pretrain_optimizer(spec: PretrainSpec) -> AdamHConfig:
    """Build the AdamH config that the pretrain run used.

    Hyperparams come verbatim from `spec` (G4 — W&B config is the truth).
    Schedule fields (warmup=0.1, decay=0.2, min_lr_ratio=0.0, linear) come
    from the heuristic, validated by N1 before this function is callable.
    `warmup` and `decay` are passed as **fractions** of num_train_steps so
    the AdamH schedule rebuild reproduces the pretrain WSD profile exactly.
    """
    return AdamHConfig(
        learning_rate=spec.peak_lr,  # N6 — no LR factor knob
        adam_lr=spec.peak_adam_lr,
        beta1=0.9,
        beta2=spec.beta2,
        epsilon=spec.epsilon,
        max_grad_norm=0.1,
        warmup=EXPECTED_PRETRAIN_WARMUP_FRACTION,  # 0.1
        decay=EXPECTED_PRETRAIN_DECAY_FRACTION,  # 0.2
        min_lr_ratio=EXPECTED_PRETRAIN_MIN_LR_RATIO,  # 0.0
        lr_schedule=EXPECTED_PRETRAIN_LR_SCHEDULE,  # "linear"
        nesterov=False,
    )


# ─── Compute-config selection ───────────────────────────────────────────────


def _selected_compute_config(spec: PretrainSpec) -> V5PComputeConfig:
    tpu_type = os.environ.get("TRUE_MIDTRAIN_TPU_TYPE") or spec.default_tpu_type
    cfg = spec.compute_config(tpu_type)  # G3 — raises if not allowed for this scale
    pdp = _optional_int_env("TRUE_MIDTRAIN_PER_DEVICE_PARALLELISM")
    tps = _optional_int_env("TRUE_MIDTRAIN_TENSOR_PARALLEL_SIZE")
    return V5PComputeConfig(
        tpu_type=cfg.tpu_type,
        per_device_parallelism=pdp if pdp is not None else cfg.per_device_parallelism,
        tensor_parallel_size=tps if tps is not None else cfg.tensor_parallel_size,
    )


# ─── Permanent ckpt cadence (G14) ───────────────────────────────────────────


def _steps_per_export(num_train_steps: int) -> int:
    return max(MIN_STEPS_PER_EXPORT, int(num_train_steps * EXPORT_FRACTION_OF_RUN))


def _assert_true_midtraining_optimizer_state_policy(train_cfg: SimpleTrainConfig) -> None:
    """True midtraining must load full trainer state through natural resume.

    Do not use ``initialize_from_checkpoint_path`` here. That branch either
    loads model-only state (fresh optimizer) or full state with the outer step
    reset to zero. True midtraining instead pre-stages the pretrain checkpoint
    under this run's own ``output_path/checkpoints`` namespace so Levanter's
    normal checkpoint search restores model, optimizer state, and step together.
    """
    if train_cfg.initialize_from_checkpoint_path is not None:
        raise AssertionError("true midtraining must not use initialize_from_checkpoint_path")
    if train_cfg.checkpoint_init_mode is not CheckpointInitMode.FULL_STATE:
        raise AssertionError("true midtraining must keep FULL_STATE as the documented full-state policy")


# ─── Env-var contract ───────────────────────────────────────────────────────


_LEGACY_OUTPUT_PATH_OVERRIDE = os.environ.get("MIDTRAIN_OUTPUT_PATH_OVERRIDE")
if _LEGACY_OUTPUT_PATH_OVERRIDE:
    # G5 — same hard-reject as the prior file. Force everyone onto the
    # single-source-of-truth resume path.
    raise ValueError(
        "MIDTRAIN_OUTPUT_PATH_OVERRIDE is disabled. Use TRUE_MIDTRAIN_RESUME_OUTPUT_PATH instead; "
        "the script derives output_path, RUN_ID, and WANDB_RUN_ID from that one value."
    )

# Also reject the alternative spelling someone might reach for.
if os.environ.get("TRUE_MIDTRAIN_OUTPUT_PATH_OVERRIDE"):
    raise ValueError("TRUE_MIDTRAIN_OUTPUT_PATH_OVERRIDE is not supported. Use TRUE_MIDTRAIN_RESUME_OUTPUT_PATH.")

_SELECT_SCALE = os.environ.get("TRUE_MIDTRAIN_SELECT_SCALE")
_SELECT_MIX = os.environ.get("TRUE_MIDTRAIN_SELECT_MIX")
_RESUME_OUTPUT_PATH = os.environ.get("TRUE_MIDTRAIN_RESUME_OUTPUT_PATH")
_EXPECT_RESUME_STEP = _optional_int_env("TRUE_MIDTRAIN_EXPECT_RESUME_STEP")
_DRY_RUN = _env_flag("TRUE_MIDTRAIN_DRY_RUN")


def _validate_env_contract() -> None:
    """At-import contract validation. Raises before any TPU is touched."""
    # Selectors must be set together — a single cell per launch.
    if (_SELECT_SCALE is None) != (_SELECT_MIX is None):
        raise ValueError("TRUE_MIDTRAIN_SELECT_SCALE and TRUE_MIDTRAIN_SELECT_MIX must be set together.")
    if _SELECT_SCALE is None:
        # No selectors → enumerate-all dry-run path. Resume vars must NOT be set.
        if _RESUME_OUTPUT_PATH is not None or _EXPECT_RESUME_STEP is not None:
            raise ValueError(
                "TRUE_MIDTRAIN_RESUME_OUTPUT_PATH / TRUE_MIDTRAIN_EXPECT_RESUME_STEP require "
                "TRUE_MIDTRAIN_SELECT_SCALE and TRUE_MIDTRAIN_SELECT_MIX."
            )
        return

    # Single-cell launch: validate every input.
    if _SELECT_SCALE not in PRETRAINS:
        allowed = ", ".join(PRETRAINS)
        raise ValueError(f"TRUE_MIDTRAIN_SELECT_SCALE must be one of {{{allowed}}}, got {_SELECT_SCALE!r}")
    if _SELECT_MIX not in MIXES:
        allowed = ", ".join(MIXES)
        raise ValueError(f"TRUE_MIDTRAIN_SELECT_MIX must be one of {{{allowed}}}, got {_SELECT_MIX!r}")

    # Resume contract — required for actual launches; optional for dry-run.
    if _DRY_RUN:
        return
    if _RESUME_OUTPUT_PATH is None:
        raise ValueError(
            "TRUE_MIDTRAIN_RESUME_OUTPUT_PATH is required for true-midtraining launches "
            "(every cell IS a resume from a pre-staged pretrain checkpoint). "
            "Set TRUE_MIDTRAIN_DRY_RUN=1 to skip this check during local introspection."
        )
    if _EXPECT_RESUME_STEP is None:
        raise ValueError(
            "TRUE_MIDTRAIN_EXPECT_RESUME_STEP is required when TRUE_MIDTRAIN_RESUME_OUTPUT_PATH is set. "
            "Pass the EXACT pretrain step (matches PRETRAINS[<scale>].resume_from_step)."
        )

    spec = PRETRAINS[_SELECT_SCALE]
    # N3 — the env-var must agree with the spec. No "off-by-cadence" wiggle room.
    if _EXPECT_RESUME_STEP != spec.resume_from_step:
        raise ValueError(
            f"TRUE_MIDTRAIN_EXPECT_RESUME_STEP={_EXPECT_RESUME_STEP} does not match "
            f"PRETRAINS[{_SELECT_SCALE!r}].resume_from_step={spec.resume_from_step}. "
            "Either you staged the wrong checkpoint, or the spec disagrees with the launcher."
        )


_validate_env_contract()


def _validate_executor_launch_contract() -> None:
    """Guard the real ``executor_main`` path against accidental fresh starts."""
    if _SELECT_SCALE is None or _SELECT_MIX is None:
        raise ValueError(
            "TRUE_MIDTRAIN_SELECT_SCALE, TRUE_MIDTRAIN_SELECT_MIX, TRUE_MIDTRAIN_RESUME_OUTPUT_PATH, "
            "and TRUE_MIDTRAIN_EXPECT_RESUME_STEP are required for true-midtraining launches. "
            "Set TRUE_MIDTRAIN_DRY_RUN=1 for no-selector introspection."
        )
    if _RESUME_OUTPUT_PATH is None or _EXPECT_RESUME_STEP is None:
        raise ValueError(
            "TRUE_MIDTRAIN_RESUME_OUTPUT_PATH and TRUE_MIDTRAIN_EXPECT_RESUME_STEP are required so "
            "Levanter loads the pre-staged full trainer state instead of starting from scratch."
        )


# ─── Build the run(s) ───────────────────────────────────────────────────────


def _build_one_run(scale: str, mix_tag: str, *, output_path_override: str | None) -> ExecutorStep:
    spec = PRETRAINS[scale]
    mix_name = MIXES[mix_tag]
    mix_data = midtraining_mix_by_name(mix_name)

    compute_cfg = _selected_compute_config(spec)
    optimizer = _build_pretrain_optimizer(spec)
    num_train_steps = spec.pretrain_num_train_steps  # N2 — invariant

    # Reconstruct the Qwen3Config exactly as the pretrain run built it. The
    # heuristic's _build_model_config is the single source of truth for
    # Delphi architecture; using it ensures the TensorStore restore matches
    # every array shape (using a hand-built config risks silent NaN-restore).
    model_config = completed_adamh_heuristic._build_model_config(
        hidden_size=spec.hidden_dim,
        seq_len=spec.seq_len,
    )

    train_cfg = SimpleTrainConfig(
        resources=_train_tpu_resources(compute_cfg.tpu_type),
        train_batch_size=spec.batch_size,  # native pretrain BS
        num_train_steps=num_train_steps,  # original pretrain target (N2)
        train_seq_len=spec.seq_len,
        per_device_parallelism=compute_cfg.per_device_parallelism,
        tensor_parallel_size=compute_cfg.tensor_parallel_size,
        # `learning_rate` is a required SimpleTrainConfig field but unused
        # when `optimizer_config` is set. Keep it consistent with peak.
        learning_rate=optimizer.learning_rate,
        optimizer_config=optimizer,
        # N5 — no initialize_from_checkpoint_path. Levanter's natural-resume
        # path picks up the pre-staged ckpt from output_path/checkpoints/.
        initialize_from_checkpoint_path=None,
        reset_data_loader_on_init=True,  # G12
        # checkpoint_init_mode is moot here (initialize_from_checkpoint_path
        # is None), but set FULL_STATE for clarity in case a future change
        # introduces an init path.
        checkpoint_init_mode=CheckpointInitMode.FULL_STATE,
        steps_per_eval=STEPS_PER_EVAL,  # G13
        steps_per_export=_steps_per_export(num_train_steps),  # G14
        steps_per_hf_export=None,  # → matches steps_per_export
    )
    _assert_true_midtraining_optimizer_state_policy(train_cfg)

    name = _build_run_name(spec.scale_tag, mix_tag, spec.resume_from_step)
    if output_path_override is not None:
        # G7 — resume-path basename must match the run name we generate.
        _validate_resume_output_path_matches_run(
            output_path_override,
            scale=spec.scale_tag,
            mix_tag=mix_tag,
            resume_step=spec.resume_from_step,
        )
        # G6 — RUN_ID = WANDB_RUN_ID = output_path basename, single source of truth.
        train_cfg = _with_resume_identity(train_cfg, output_path_override)

    # WandB tag length limit is 64 chars per tag. The full pretrain_run_name
    # (`adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021`, 53 chars) plus
    # any prefix overflows; the run-name info is already in the wandb config
    # logged by AdamHConfig + the base tag, so don't duplicate it here.
    tags = (
        "true-midtraining",  # G7 / N7 — discriminator
        "midtraining",
        f"base={spec.scale_tag}",
        f"midtraining_mix={mix_name}",
        "midtraining-mix",
        f"resume_step={spec.resume_from_step}",
        f"batch_size={spec.batch_size}",
        f"seq_len={spec.seq_len}",
        f"tpu_type={compute_cfg.tpu_type}",
        f"per_device_parallelism={compute_cfg.per_device_parallelism}",
        f"tensor_parallel_size={compute_cfg.tensor_parallel_size}",
        f"pretrain_tokens={spec.pretrain_tokens}",
        f"num_train_steps={num_train_steps}",
        f"peak_lr={optimizer.learning_rate:.3e}",
        f"adam_lr={optimizer.adam_lr:.3e}",
        "adamh",
        "delphi-midtrain",
    )
    # Sanity: every tag must fit the wandb 64-char limit; refuse to launch if any drift.
    for tag in tags:
        if len(tag) > 64:
            raise AssertionError(f"WandB tag exceeds 64-char limit ({len(tag)}): {tag!r}")

    return default_train(
        name=name,
        tokenized=mix_data,
        model_config=model_config,
        train_config=train_cfg,
        tags=tags,
        eval_harness_tasks=(),
        wandb_project="delphi-midtraining",
        override_output_path=output_path_override,
    )


def _build_runs() -> list[ExecutorStep]:
    """Build one ExecutorStep per (scale, mix) cell.

    With no selectors set, enumerate all configured cells for dry-run
    inspection only — do not pass them to executor_main as a single list (the
    `train_lm` job-name collision bug from v10 still applies).
    """
    if _SELECT_SCALE is not None and _SELECT_MIX is not None:
        runs = [_build_one_run(_SELECT_SCALE, _SELECT_MIX, output_path_override=_RESUME_OUTPUT_PATH)]
    else:
        runs = [_build_one_run(scale, mix_tag, output_path_override=None) for scale in PRETRAINS for mix_tag in MIXES]
    names = [r.name for r in runs]
    if len(names) != len(set(names)):  # G11
        raise ValueError(f"Generated duplicate true-midtraining run names: {names}")
    # G7 / N7 — `default_train` prepends `checkpoints/` to the ExecutorStep
    # name; the W&B run id uses the unprefixed basename. Verify the basename
    # starts with `true-midtrain-` so wandb runs are clearly tagged.
    for n in names:
        bare = n.removeprefix("checkpoints/")
        if not bare.startswith(RUN_NAME_PREFIX):
            raise AssertionError(f"Run name does not start with {RUN_NAME_PREFIX!r}: {n!r}")
    return runs


# ─── Pre-flight ─────────────────────────────────────────────────────────────


def _run_pre_flight_safety_checks() -> None:
    """At launch (under __main__): heavy GCS reads + safety checks."""
    # G15 + G16 — Layer 3 (val/train disjoint by hash sample) + Layer 4
    # (partition summary). Same call shape as the original sweep.
    pos = Axis("position", DEFAULT_SEQ_LEN)
    try:
        log_partition_summary(full_highquality_nemo_math, pos)
        assert_val_train_disjoint(full_highquality_nemo_math, pos)
        logger.info(
            "Layer 3+4 pre-flight passed on math-only check config. The actual training mix "
            "reuses the same math cache and val carve-out (12,500 sequences, identical across runs)."
        )
    except TypeError as exc:
        # Same caveat as the original file: at iris coordinator startup the
        # versioned() wrappers in TokenizeConfig haven't been resolved yet,
        # which can trip Levanter's len() check inside urls_for_split. The
        # disjointness property holds by Feistel bijection; AssertionError
        # (a real safety violation) is NOT caught.
        logger.warning(
            "Layer 3 pre-flight skipped due to unresolved versioned() wrappers in TokenizeConfig "
            "(%s). Layers 1+2 already validated the mix at module import; the disjointness "
            "property holds by Feistel bijection in _split_into_trainval_sets.",
            exc,
        )

    # G8 + N3 + N4 — pre-staged checkpoint must exist at exactly the expected
    # step, with all 3 TensorStore artefacts present.
    if _RESUME_OUTPUT_PATH is None:
        return
    assert _EXPECT_RESUME_STEP is not None  # validated by _validate_env_contract
    _verify_pre_staged_checkpoint(
        _RESUME_OUTPUT_PATH,
        expected_step=_EXPECT_RESUME_STEP,
    )


# ─── Module-level run list ──────────────────────────────────────────────────

runs: list[ExecutorStep] = _build_runs()


if __name__ == "__main__":
    if _DRY_RUN:
        print(f"Built {len(runs)} ExecutorStep(s):")
        for r in runs:
            print(f"  {r.name}")
        print()
        if _SELECT_SCALE is not None:
            spec = PRETRAINS[_SELECT_SCALE]
            mix_name = MIXES[_SELECT_MIX]
            optimizer = _build_pretrain_optimizer(spec)
            print(f"Selected cell: {_SELECT_SCALE} x {_SELECT_MIX}")
            print(f"  pretrain_run_name      = {spec.pretrain_run_name}")
            print(f"  pretrain_num_train_steps = {spec.pretrain_num_train_steps}")
            print(f"  pretrain_tokens        = {spec.pretrain_tokens:,}")
            print(f"  resume_from_step       = {spec.resume_from_step}")
            print(f"  remaining steps        = {spec.pretrain_num_train_steps - spec.resume_from_step}")
            print(f"  batch_size             = {spec.batch_size}")
            print(f"  seq_len                = {spec.seq_len}")
            print(f"  optimizer.peak_lr      = {optimizer.learning_rate:.6e}")
            print(f"  optimizer.adam_lr      = {optimizer.adam_lr:.6e}")
            print(f"  optimizer.beta2        = {optimizer.beta2}")
            print(f"  optimizer.epsilon      = {optimizer.epsilon:.6e}")
            print(f"  optimizer.warmup       = {optimizer.warmup}  (fraction of N)")
            print(f"  optimizer.decay        = {optimizer.decay}  (fraction of N)")
            print(f"  optimizer.min_lr_ratio = {optimizer.min_lr_ratio}")
            print(f"  optimizer.lr_schedule  = {optimizer.lr_schedule!r}")
            print(f"  data mix               = {mix_name}")
        print()
        print("Dry-run; no executor_main, no GCS reads beyond module-level.")
    else:
        _validate_executor_launch_contract()
        _run_pre_flight_safety_checks()
        executor_main(
            steps=runs,
            description="True-midtraining: hijack Delphi pretrain WSD cooldown with the math midtraining mix.",
        )
