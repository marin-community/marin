# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared scale recipes and launch metadata for mixture-scaling studies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from levanter.main.train_lm import LmConfig
from levanter.optim import MuonHConfig

from experiments.domain_phase_mix.proxy_sweep import (
    REGMIX_130M_CHINCHILLA_BUDGET,
    REGMIX_1_2B_CHINCHILLA_BUDGET,
    REGMIX_300M_CHINCHILLA_BUDGET,
    REGMIX_520M_CHINCHILLA_BUDGET,
    get_num_train_steps,
    regmix_130m_muonh_base,
    regmix_130m_proxy,
    regmix_1_2b_muonh_base,
    regmix_1_2b_proxy,
    regmix_300m_muonh_base,
    regmix_300m_proxy,
    regmix_520m_muonh_base,
    regmix_520m_proxy,
    regmix_60m_proxy,
)
from experiments.domain_phase_mix.qsplit240_replay import (
    BASELINES3_PANEL,
    DEFAULT_REGION_AGNOSTIC_TPU_REGIONS,
    REPRESENTATIVE12_PANEL,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    EXPERIMENT_BUDGET as REGMIX_60M_1P2B_BUDGET,
    TARGET_BUDGET as BASE_TARGET_BUDGET,
)

STRONG_TIER_TARGET_BUDGET_MULTIPLIERS = (0.5, 1.0, 2.0)
STRONG_TIER_QSPLIT_PANEL = REPRESENTATIVE12_PANEL
STRONG_TIER_HOLDOUT_PANEL = BASELINES3_PANEL


class ScalingStudyScale(StrEnum):
    """Supported model scales for the explicit mixture-scaling study."""

    REGMIX_60M_1P2B = "60m_1p2b"
    REGMIX_130M_2P6B = "130m_2p6b"
    REGMIX_300M_6B = "300m_6b"
    REGMIX_520M_10P4B = "520m_10p4b"
    REGMIX_1_2B_24B = "1_2b_24b"


class ScalingStudyPath(StrEnum):
    """Logical launch track within the scaling study."""

    QSPLIT_REPRESENTATIVE12 = "qsplit_representative12"
    STRATIFIED = "stratified"
    QSPLIT_BASELINES3_HOLDOUT = "qsplit_baselines3_holdout"
    STRATIFIED_HOLDOUT = "stratified_holdout"


class ScalingStudyCellStatus(StrEnum):
    """Whether a cell is newly launched, reused, or holdout-only."""

    NEW = "new"
    REUSED = "reused"
    HOLDOUT_ONLY = "holdout_only"


@dataclass(frozen=True)
class ScalingStudyScaleSpec:
    """Canonical recipe for one model-size rung."""

    scale: ScalingStudyScale
    scale_slug: str
    model_family: str
    stratified_name_prefix: str
    base_experiment_budget: int
    batch_size: int
    seq_len: int
    model_config: LmConfig
    optimizer_config: MuonHConfig | None
    tpu_type: str
    tpu_regions: tuple[str, ...]
    tpu_zone: str | None

    @property
    def experiment_budget(self) -> int:
        """Return the 1x actual proxy token budget for this rung."""
        return self.base_experiment_budget

    @property
    def name_prefix(self) -> str:
        """Return the historical 1x stratified name prefix for this rung."""
        return self.stratified_name_prefix

    def experiment_budget_for_multiplier(self, multiplier: float) -> int:
        """Return the actual proxy token budget for a token-scale multiplier."""
        return scaled_budget(self.base_experiment_budget, multiplier)

    def target_budget_for_multiplier(self, multiplier: float) -> int:
        """Return the simulated-epoch target budget for a token-scale multiplier."""
        return scaled_budget(BASE_TARGET_BUDGET, multiplier)

    def num_train_steps_for_multiplier(self, multiplier: float) -> int:
        """Return the train-step count implied by the scaled actual budget."""
        return get_num_train_steps(
            self.experiment_budget_for_multiplier(multiplier),
            self.batch_size,
            self.seq_len,
        )


@dataclass(frozen=True)
class ScalingStudyCell:
    """One logical cell in the strong-tier scaling study matrix."""

    path: ScalingStudyPath
    scale: ScalingStudyScale
    status: ScalingStudyCellStatus
    panel: str | None
    cohort: str
    name_prefix: str
    model_family: str
    experiment_budget: int
    target_budget: int
    target_budget_multiplier: float
    num_train_steps: int
    batch_size: int
    seq_len: int
    tpu_type: str
    tpu_regions: tuple[str, ...]
    tpu_zone: str | None
    run_count: int
    source_name_prefix: str | None = None

    def to_manifest_dict(self) -> dict[str, object]:
        """Return a JSON-serializable manifest row."""
        return {
            "path": self.path.value,
            "scale": self.scale.value,
            "status": self.status.value,
            "panel": self.panel,
            "cohort": self.cohort,
            "name_prefix": self.name_prefix,
            "source_name_prefix": self.source_name_prefix,
            "model_family": self.model_family,
            "experiment_budget": self.experiment_budget,
            "target_budget": self.target_budget,
            "target_budget_multiplier": self.target_budget_multiplier,
            "num_train_steps": self.num_train_steps,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "tpu_type": self.tpu_type,
            "tpu_regions": list(self.tpu_regions),
            "tpu_zone": self.tpu_zone,
            "run_count": self.run_count,
        }


SCALE_SPECS = {
    ScalingStudyScale.REGMIX_60M_1P2B: ScalingStudyScaleSpec(
        scale=ScalingStudyScale.REGMIX_60M_1P2B,
        scale_slug="60m",
        model_family="regmix_60m_proxy",
        stratified_name_prefix="pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_60m_1p2b",
        base_experiment_budget=REGMIX_60M_1P2B_BUDGET,
        batch_size=128,
        seq_len=2048,
        model_config=regmix_60m_proxy,
        optimizer_config=None,
        tpu_type="v5p-8",
        tpu_regions=DEFAULT_REGION_AGNOSTIC_TPU_REGIONS,
        tpu_zone=None,
    ),
    ScalingStudyScale.REGMIX_130M_2P6B: ScalingStudyScaleSpec(
        scale=ScalingStudyScale.REGMIX_130M_2P6B,
        scale_slug="130m",
        model_family="regmix_130m_proxy",
        stratified_name_prefix="pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_130m_2p6b",
        base_experiment_budget=REGMIX_130M_CHINCHILLA_BUDGET,
        batch_size=128,
        seq_len=2048,
        model_config=regmix_130m_proxy,
        optimizer_config=regmix_130m_muonh_base,
        tpu_type="v5p-8",
        tpu_regions=DEFAULT_REGION_AGNOSTIC_TPU_REGIONS,
        tpu_zone=None,
    ),
    ScalingStudyScale.REGMIX_300M_6B: ScalingStudyScaleSpec(
        scale=ScalingStudyScale.REGMIX_300M_6B,
        scale_slug="300m",
        model_family="regmix_300m_proxy",
        stratified_name_prefix="pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_300m_6b",
        base_experiment_budget=REGMIX_300M_CHINCHILLA_BUDGET,
        batch_size=128,
        seq_len=2048,
        model_config=regmix_300m_proxy,
        optimizer_config=regmix_300m_muonh_base,
        tpu_type="v5p-8",
        tpu_regions=DEFAULT_REGION_AGNOSTIC_TPU_REGIONS,
        tpu_zone=None,
    ),
    ScalingStudyScale.REGMIX_520M_10P4B: ScalingStudyScaleSpec(
        scale=ScalingStudyScale.REGMIX_520M_10P4B,
        scale_slug="520m",
        model_family="regmix_520m_proxy",
        stratified_name_prefix="pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_520m_10p4b",
        base_experiment_budget=REGMIX_520M_CHINCHILLA_BUDGET,
        batch_size=256,
        seq_len=2048,
        model_config=regmix_520m_proxy,
        optimizer_config=regmix_520m_muonh_base,
        tpu_type="v5p-32",
        tpu_regions=DEFAULT_REGION_AGNOSTIC_TPU_REGIONS,
        tpu_zone=None,
    ),
    ScalingStudyScale.REGMIX_1_2B_24B: ScalingStudyScaleSpec(
        scale=ScalingStudyScale.REGMIX_1_2B_24B,
        scale_slug="1_2b",
        model_family="regmix_1_2b_proxy",
        stratified_name_prefix="pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_1_2b_24b",
        base_experiment_budget=REGMIX_1_2B_CHINCHILLA_BUDGET,
        batch_size=256,
        seq_len=2048,
        model_config=regmix_1_2b_proxy,
        optimizer_config=regmix_1_2b_muonh_base,
        tpu_type="v5p-64",
        tpu_regions=("us-east5",),
        tpu_zone="us-east5-a",
    ),
}


def resolve_scale_spec(scale: ScalingStudyScale) -> ScalingStudyScaleSpec:
    """Return the canonical recipe for one scaling-study rung."""
    return SCALE_SPECS[scale]


def scaled_budget(base_budget: int, multiplier: float) -> int:
    """Scale a token budget by a multiplier.

    The shared simulated target budget is odd, so the 0.5x case rounds down by
    one token. That does not materially affect epoch semantics, but keeping the
    budget integral avoids hidden float truncation later in the pipeline.
    """
    if multiplier <= 0:
        raise ValueError(f"multiplier must be positive, got {multiplier}")
    scaled = base_budget * multiplier
    return math.floor(scaled)


def budget_slug(token_budget: int) -> str:
    """Return a compact human-readable token-budget slug."""
    if token_budget % 1_000_000_000 == 0:
        return f"{token_budget // 1_000_000_000}b"
    if token_budget % 100_000_000 == 0:
        billions = token_budget / 1_000_000_000
        return f"{billions:.1f}".replace(".", "p") + "b"
    raise ValueError(f"Unexpected token budget for slug formatting: {token_budget}")


def multiplier_slug(multiplier: float) -> str:
    """Return a stable slug for a target-budget multiplier."""
    if multiplier == 1.0:
        return "1x"
    return f"{multiplier:.1f}".replace(".", "p") + "x"


def qsplit_name_prefix(*, scale: ScalingStudyScale, multiplier: float) -> str:
    """Return the canonical qsplit launch prefix for one strong-tier cell."""
    spec = resolve_scale_spec(scale)
    return (
        "pinlin_calvin_xu/data_mixture/"
        f"ngd3dm2_qsplit240_representative12_{spec.scale_slug}_{budget_slug(spec.experiment_budget_for_multiplier(multiplier))}"
        f"_target{multiplier_slug(multiplier)}"
    )


def stratified_name_prefix(*, scale: ScalingStudyScale, multiplier: float) -> str:
    """Return the canonical stratified launch prefix for one strong-tier cell."""
    spec = resolve_scale_spec(scale)
    return (
        "pinlin_calvin_xu/data_mixture/"
        f"ngd3dm2_stratified_{spec.scale_slug}_{budget_slug(spec.experiment_budget_for_multiplier(multiplier))}"
        f"_target{multiplier_slug(multiplier)}"
    )


def _qsplit_status(scale: ScalingStudyScale, multiplier: float) -> ScalingStudyCellStatus:
    if scale in {ScalingStudyScale.REGMIX_300M_6B, ScalingStudyScale.REGMIX_520M_10P4B} and multiplier == 1.0:
        return ScalingStudyCellStatus.REUSED
    return ScalingStudyCellStatus.NEW


def _stratified_status(scale: ScalingStudyScale, multiplier: float) -> ScalingStudyCellStatus:
    if scale in {ScalingStudyScale.REGMIX_300M_6B, ScalingStudyScale.REGMIX_520M_10P4B} and multiplier == 1.0:
        return ScalingStudyCellStatus.REUSED
    return ScalingStudyCellStatus.NEW


def build_strong_tier_cells() -> list[ScalingStudyCell]:
    """Build the full strong-tier scaling-study matrix, including reused cells."""
    cells: list[ScalingStudyCell] = []
    fit_scales = (
        ScalingStudyScale.REGMIX_130M_2P6B,
        ScalingStudyScale.REGMIX_300M_6B,
        ScalingStudyScale.REGMIX_520M_10P4B,
    )
    for scale in fit_scales:
        spec = resolve_scale_spec(scale)
        for multiplier in STRONG_TIER_TARGET_BUDGET_MULTIPLIERS:
            experiment_budget = spec.experiment_budget_for_multiplier(multiplier)
            target_budget = spec.target_budget_for_multiplier(multiplier)
            num_train_steps = spec.num_train_steps_for_multiplier(multiplier)
            cells.append(
                ScalingStudyCell(
                    path=ScalingStudyPath.QSPLIT_REPRESENTATIVE12,
                    scale=scale,
                    status=_qsplit_status(scale, multiplier),
                    panel=STRONG_TIER_QSPLIT_PANEL,
                    cohort=("strong_tier_qsplit_representative12_" f"{spec.scale_slug}_{multiplier_slug(multiplier)}"),
                    name_prefix=qsplit_name_prefix(scale=scale, multiplier=multiplier),
                    source_name_prefix=(
                        "pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b"
                        if scale == ScalingStudyScale.REGMIX_300M_6B and multiplier == 1.0
                        else (
                            "pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_520m_chinchilla"
                            if scale == ScalingStudyScale.REGMIX_520M_10P4B and multiplier == 1.0
                            else None
                        )
                    ),
                    model_family=spec.model_family,
                    experiment_budget=experiment_budget,
                    target_budget=target_budget,
                    target_budget_multiplier=multiplier,
                    num_train_steps=num_train_steps,
                    batch_size=spec.batch_size,
                    seq_len=spec.seq_len,
                    tpu_type=spec.tpu_type,
                    tpu_regions=spec.tpu_regions,
                    tpu_zone=spec.tpu_zone,
                    run_count=12,
                )
            )
            cells.append(
                ScalingStudyCell(
                    path=ScalingStudyPath.STRATIFIED,
                    scale=scale,
                    status=_stratified_status(scale, multiplier),
                    panel=None,
                    cohort=("strong_tier_stratified_" f"{spec.scale_slug}_{multiplier_slug(multiplier)}"),
                    name_prefix=stratified_name_prefix(scale=scale, multiplier=multiplier),
                    source_name_prefix=(
                        "pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_300m_6b"
                        if scale == ScalingStudyScale.REGMIX_300M_6B and multiplier == 1.0
                        else (
                            "pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_520m_10p4b"
                            if scale == ScalingStudyScale.REGMIX_520M_10P4B and multiplier == 1.0
                            else None
                        )
                    ),
                    model_family=spec.model_family,
                    experiment_budget=experiment_budget,
                    target_budget=target_budget,
                    target_budget_multiplier=multiplier,
                    num_train_steps=num_train_steps,
                    batch_size=spec.batch_size,
                    seq_len=spec.seq_len,
                    tpu_type=spec.tpu_type,
                    tpu_regions=spec.tpu_regions,
                    tpu_zone=spec.tpu_zone,
                    run_count=1,
                )
            )

    holdout_spec = resolve_scale_spec(ScalingStudyScale.REGMIX_1_2B_24B)
    holdout_budget = holdout_spec.experiment_budget_for_multiplier(1.0)
    holdout_target_budget = holdout_spec.target_budget_for_multiplier(1.0)
    holdout_steps = holdout_spec.num_train_steps_for_multiplier(1.0)
    cells.append(
        ScalingStudyCell(
            path=ScalingStudyPath.QSPLIT_BASELINES3_HOLDOUT,
            scale=ScalingStudyScale.REGMIX_1_2B_24B,
            status=ScalingStudyCellStatus.HOLDOUT_ONLY,
            panel=STRONG_TIER_HOLDOUT_PANEL,
            cohort="holdout_qsplit_baselines3_1_2b_1x",
            name_prefix="pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_1_2b_chinchilla",
            source_name_prefix="pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_1_2b_chinchilla",
            model_family=holdout_spec.model_family,
            experiment_budget=holdout_budget,
            target_budget=holdout_target_budget,
            target_budget_multiplier=1.0,
            num_train_steps=holdout_steps,
            batch_size=holdout_spec.batch_size,
            seq_len=holdout_spec.seq_len,
            tpu_type=holdout_spec.tpu_type,
            tpu_regions=holdout_spec.tpu_regions,
            tpu_zone=holdout_spec.tpu_zone,
            run_count=3,
        )
    )
    cells.append(
        ScalingStudyCell(
            path=ScalingStudyPath.STRATIFIED_HOLDOUT,
            scale=ScalingStudyScale.REGMIX_1_2B_24B,
            status=ScalingStudyCellStatus.HOLDOUT_ONLY,
            panel=None,
            cohort="holdout_stratified_1_2b_1x",
            name_prefix="pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_1_2b_24b",
            source_name_prefix="pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_1_2b_24b",
            model_family=holdout_spec.model_family,
            experiment_budget=holdout_budget,
            target_budget=holdout_target_budget,
            target_budget_multiplier=1.0,
            num_train_steps=holdout_steps,
            batch_size=holdout_spec.batch_size,
            seq_len=holdout_spec.seq_len,
            tpu_type=holdout_spec.tpu_type,
            tpu_regions=holdout_spec.tpu_regions,
            tpu_zone=holdout_spec.tpu_zone,
            run_count=1,
        )
    )
    return cells


def new_submission_cells(cells: list[ScalingStudyCell] | None = None) -> list[ScalingStudyCell]:
    """Return only cells that require new launches."""
    resolved_cells = build_strong_tier_cells() if cells is None else cells
    return [cell for cell in resolved_cells if cell.status == ScalingStudyCellStatus.NEW]


def count_new_submission_runs(cells: list[ScalingStudyCell] | None = None) -> int:
    """Return the number of new training runs implied by the strong-tier matrix."""
    return sum(cell.run_count for cell in new_submission_cells(cells))


def external_holdout_references() -> list[dict[str, object]]:
    """Return extra holdout references that are not launched in the strong tier."""
    return [
        {
            "scale": ScalingStudyScale.REGMIX_60M_1P2B.value,
            "path": "qsplit240_full_swarm",
            "status": "external_holdout_only",
            "source_name_prefix": "pinlin_calvin_xu/data_mixture/two_phase_dolma3_dolmino_top_level",
            "notes": "Existing 60M full swarm remains a downward extrapolation holdout.",
        }
    ]
