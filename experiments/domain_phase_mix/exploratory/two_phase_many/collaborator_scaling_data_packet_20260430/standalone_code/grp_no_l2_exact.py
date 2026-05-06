# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3

"""Self-contained exact GRP power-family-penalty no-L2 retune.

This is a standalone port of the repository's GRP no-L2 path. It intentionally
does not import Marin modules. The default command fits the included best
nonlinear parameters and emits predictions/optimum diagnostics. The full retune
procedure is also included:

    python standalone_code/grp_no_l2_exact.py --mode retune --method Powell --coarse-top-k 3

The retune follows the repo procedure:
1. Load the original 60M/1.2B `two_phase_many.csv` panel.
2. Append the known `baseline_stratified` row for `eval/uncheatable_eval/bpb`.
3. Use epoch multipliers from `two_phase_many_epoch_metadata.csv`.
4. Build the `power_family_penalty` GRP surrogate.
5. Fix `reg = 0.0`.
6. Build the same 9-start deterministic start bank from the regularized
   `power_family_penalty` best row.
7. Score starts with the same 5-fold calibration objective.
8. Refine the top 3 starts with Powell, maxiter 30.
9. Fit the final NNLS linear head and optimize the raw simplex optimum.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import nnls
from sklearn.model_selection import KFold

PACKET_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PACKET_ROOT / "data" / "grp_no_l2"
DEFAULT_TARGET = "eval/uncheatable_eval/bpb"
GENERIC_FAMILY_NAMES = ("broad_text", "tech_code", "reasoning")
VARIANT_NAME = "power_family_penalty"
NO_L2_VARIANT_NAME = "power_family_penalty_no_l2"
CV_SEED = 0
REG_FIXED = 0.0
REG_PACK_PLACEHOLDER_LOG = -18.0

CALIBRATION_CV_WEIGHT = 1.0
CALIBRATION_FOLDMEAN_WEIGHT = 0.05
CALIBRATION_TAIL_WEIGHT = 0.5
CALIBRATION_DEPOPT_WEIGHT = 0.1
CALIBRATION_SUPPORT_WEIGHT = 0.01
LOWER_TAIL_FRAC = 0.15
TRUSTBLEND_TOPK_ACTUAL = 8


@dataclass(frozen=True)
class PacketData:
    """Feature-ready GRP data."""

    frame: pd.DataFrame
    name_col: str
    y: np.ndarray
    w: np.ndarray
    m: int
    c0: np.ndarray
    c1: np.ndarray
    domain_names: list[str]


@dataclass(frozen=True)
class GenericFamilyPacket:
    """GRP packet augmented with CC high/low pairs and source families."""

    base: PacketData
    pairs: list[tuple[int, int]]
    pair_topics: list[str]
    singletons: list[int]
    family_map: dict[str, list[int]]


@dataclass(frozen=True)
class VariantSpec:
    """Structural configuration for a calibration-oriented GRP variant."""

    signal_kind: str = "power"
    family_signal_kind: str = "power"
    family_curvature: bool = True
    global_group_penalty: bool = False
    family_group_penalty: bool = True
    family_total_penalty: bool = False
    domain_curvature: bool = False
    pair_aggregator: str = "linear"


def softplus(x: np.ndarray | float) -> np.ndarray:
    """Stable softplus matching the repo helper."""
    arr = np.asarray(x, dtype=float)
    return np.where(arr > 20.0, arr, np.log1p(np.exp(np.minimum(arr, 20.0))))


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    """Stable logistic sigmoid matching the repo helper."""
    arr = np.asarray(x, dtype=float)
    positive = arr >= 0.0
    out = np.empty_like(arr, dtype=float)
    out[positive] = 1.0 / (1.0 + np.exp(-arr[positive]))
    exp_x = np.exp(arr[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def sigmoid_scalar_clipped(x: float) -> float:
    """Scalar sigmoid with the same clipping used in the tuner."""
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0))))


def average_phase_tv_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Average total-variation distance across phases."""
    return np.abs(left - right).sum(axis=(1, 2)) / (2.0 * left.shape[1])


def phase_domains(columns: list[str]) -> list[str]:
    """Return domains in the same order as the first phase columns."""
    return [
        column.removeprefix("phase_0_")
        for column in columns
        if column.startswith("phase_0_") and not column.endswith("_epochs")
    ]


def normalize_rows(weights: np.ndarray) -> np.ndarray:
    """Normalize phase rows and fail on invalid mass."""
    row_sums = weights.sum(axis=-1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Encountered a phase row with non-positive mass")
    return weights / row_sums


def append_stratified_baseline(frame: pd.DataFrame, *, target: str) -> pd.DataFrame:
    """Append the known 60M stratified baseline used by the repo loader."""
    if target != DEFAULT_TARGET or frame["run_name"].astype(str).eq("baseline_stratified").any():
        return frame
    domains = phase_domains(list(frame.columns))
    uniform_weight = 1.0 / len(domains)
    row: dict[str, float | int | str] = {
        "run_id": 3,
        "run_name": "baseline_stratified",
        "source_experiment": "pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_60m_1p2b",
        "status": "completed",
        target: 1.078909158706665,
    }
    for phase_name in ("phase_0", "phase_1"):
        for domain in domains:
            row[f"{phase_name}_{domain}"] = uniform_weight
    augmented = frame.copy()
    for column in row:
        if column not in augmented:
            augmented[column] = np.nan
    augmented.loc[len(augmented)] = {column: row.get(column, np.nan) for column in augmented.columns}
    return augmented


def load_packet(data_dir: Path, *, target: str) -> GenericFamilyPacket:
    """Load the original GRP fit panel and epoch metadata from packet-local CSVs."""
    frame = pd.read_csv(data_dir / "two_phase_many.csv")
    if "status" in frame:
        frame = frame[frame["status"] == "completed"].reset_index(drop=True)
    frame = append_stratified_baseline(frame, target=target)
    frame = frame[frame[target].notna()].copy().reset_index(drop=True)
    domains = phase_domains(list(frame.columns))
    weights = np.zeros((len(frame), 2, len(domains)), dtype=float)
    for phase_idx, phase_name in enumerate(("phase_0", "phase_1")):
        for domain_idx, domain in enumerate(domains):
            weights[:, phase_idx, domain_idx] = frame[f"{phase_name}_{domain}"].to_numpy(dtype=float)
    weights = normalize_rows(weights)

    metadata = pd.read_csv(data_dir / "two_phase_many_epoch_metadata.csv").set_index("domain_name")
    c0 = metadata.loc[domains, "phase_0_epoch_multiplier"].to_numpy(dtype=float)
    c1 = metadata.loc[domains, "phase_1_epoch_multiplier"].to_numpy(dtype=float)
    base = PacketData(
        frame=frame,
        name_col="candidate_run_name" if "candidate_run_name" in frame else "run_name",
        y=frame[target].to_numpy(dtype=float),
        w=weights,
        m=len(domains),
        c0=c0,
        c1=c1,
        domain_names=domains,
    )

    pairs: list[tuple[int, int]] = []
    pair_topics: list[str] = []
    paired: set[int] = set()
    for idx, domain in enumerate(domains):
        if idx in paired:
            continue
        if domain.startswith("dolma3_cc/") and domain.endswith("_high"):
            low_name = domain[:-5] + "_low"
            if low_name in domains:
                low_idx = domains.index(low_name)
                pairs.append((idx, low_idx))
                pair_topics.append(domain[len("dolma3_cc/") : -5])
                paired.add(idx)
                paired.add(low_idx)
    singletons = [idx for idx in range(base.m) if idx not in paired]
    family_map = {family_name: [] for family_name in GENERIC_FAMILY_NAMES}
    for idx, domain in enumerate(domains):
        is_broad = (
            domain.startswith("dolma3_cc/")
            or domain
            in {
                "dolma3_wikipedia",
                "dolmino_common_crawl_hq",
                "dolmino_olmocr_pdfs_hq",
                "dolmino_stem_heavy_crawl",
            }
            or domain.endswith("synth_qa")
        )
        is_tech = any(token in domain for token in ("stack_edu", "synth_code", "synth_math")) or (
            domain in {"dolma3_arxiv", "dolma3_finemath_3plus"}
        )
        is_reasoning = domain in {"dolmino_synth_instruction", "dolmino_synth_thinking"}
        if is_broad:
            family_map["broad_text"].append(idx)
        if is_tech:
            family_map["tech_code"].append(idx)
        if is_reasoning:
            family_map["reasoning"].append(idx)

    return GenericFamilyPacket(
        base=base,
        pairs=pairs,
        pair_topics=pair_topics,
        singletons=singletons,
        family_map=family_map,
    )


def domain_exponent_key(domain_idx: int) -> str:
    """Return the parameter key for a domain-specific exponent."""
    return f"a_domain_{domain_idx:02d}"


def resolve_exponent_value(params: dict[str, float], key: str) -> float:
    """Resolve a curvature parameter with repo-compatible fallbacks."""
    if key in params:
        return float(params[key])
    if "a" in params:
        return float(params["a"])
    raise KeyError(f"Missing curvature parameter {key!r}")


def resolve_curvature(
    params: dict[str, float],
    signal_kind: str,
    family_name: str | None = None,
    other_family_name: str | None = None,
    domain_indices: tuple[int, ...] | None = None,
) -> float:
    """Resolve family/domain curvature exactly as in the repo."""
    if signal_kind not in {"power", "boxcox"}:
        raise ValueError(f"Curvature is only defined for power/boxcox, got {signal_kind!r}")
    if domain_indices:
        values = [resolve_exponent_value(params, domain_exponent_key(domain_idx)) for domain_idx in domain_indices]
        return float(np.mean(np.asarray(values, dtype=float)))
    if family_name is None:
        return float(params["a"])
    first = resolve_exponent_value(params, f"a_{family_name}")
    if other_family_name is None:
        return first
    second = resolve_exponent_value(params, f"a_{other_family_name}")
    return 0.5 * (first + second)


def signal_transform(
    values: np.ndarray,
    params: dict[str, float],
    signal_kind: str,
    family_name: str | None = None,
    other_family_name: str | None = None,
    domain_indices: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Signal transform from the repo's flexible GRP implementation."""
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    if signal_kind == "log":
        alpha = float(params["alpha"])
        return np.log1p(alpha * values)
    if signal_kind == "power":
        a = resolve_curvature(params, signal_kind, family_name, other_family_name, domain_indices)
        return np.power(np.maximum(values, 1e-12), a)
    if signal_kind == "boxcox":
        alpha = float(params["alpha"])
        a = resolve_curvature(params, signal_kind, family_name, other_family_name, domain_indices)
        u = 1.0 + alpha * values
        if abs(a) < 1e-8:
            return np.log(u)
        return (np.power(u, a) - 1.0) / a
    raise ValueError(f"Unsupported signal kind: {signal_kind}")


def signal_derivative(
    values: np.ndarray,
    params: dict[str, float],
    signal_kind: str,
    family_name: str | None = None,
    other_family_name: str | None = None,
    domain_indices: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Signal derivative from the repo's flexible GRP implementation."""
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    if signal_kind == "log":
        alpha = float(params["alpha"])
        return alpha / (1.0 + alpha * values)
    if signal_kind == "power":
        a = resolve_curvature(params, signal_kind, family_name, other_family_name, domain_indices)
        safe = np.maximum(values, 1e-12)
        return a * np.power(safe, a - 1.0)
    if signal_kind == "boxcox":
        alpha = float(params["alpha"])
        a = resolve_curvature(params, signal_kind, family_name, other_family_name, domain_indices)
        u = 1.0 + alpha * values
        if abs(a) < 1e-8:
            return alpha / u
        return alpha * np.power(u, a - 1.0)
    raise ValueError(f"Unsupported signal kind: {signal_kind}")


def family_tau(params: dict[str, float], family_name: str) -> float:
    """Return family penalty threshold."""
    return float(params.get(f"tau_{family_name}", params.get("tau", 3.0)))


class GenericFamilyPenaltyCalibrationSurrogate:
    """Repo-equivalent power-family-penalty GRP surrogate."""

    def __init__(self, packet: GenericFamilyPacket, *, params: dict[str, float], spec: VariantSpec):
        self.packet = packet
        self.params = dict(params)
        self.spec = spec
        self.family_totals = GENERIC_FAMILY_NAMES
        self.quality_discount = True
        self.pair_cc_domains = True
        self.include_singletons = True
        self.include_pairs = True
        self.include_family_totals = True
        self.include_global_group_penalty = False
        self.include_family_group_penalty = True
        self.include_family_total_penalty = False
        domain_to_family: list[str] = []
        for domain_idx in range(packet.base.m):
            assigned = [family_name for family_name, members in packet.family_map.items() if domain_idx in members]
            if len(assigned) != 1:
                raise ValueError(f"Expected exactly one family for domain index {domain_idx}, got {assigned}")
            domain_to_family.append(assigned[0])
        self.domain_to_family = tuple(domain_to_family)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def retained_x(self, weights: np.ndarray) -> np.ndarray:
        """Retained exposure x used by GRP."""
        p0 = weights[:, 0, :]
        p1 = weights[:, 1, :]
        e0 = p0 * self.packet.base.c0[None, :]
        e1 = p1 * self.packet.base.c1[None, :]
        lam = float(self.params["lam"])
        eta = float(self.params["eta"])
        return np.exp(-lam * (1.0 - p1)) * e0 + eta * e1

    def feature_transform(
        self,
        values: np.ndarray,
        *,
        signal_kind: str,
        family_name: str | None = None,
        other_family_name: str | None = None,
        domain_indices: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Apply the configured signal transform."""
        return signal_transform(
            values,
            self.params,
            signal_kind,
            family_name if self.spec.family_curvature else None,
            other_family_name if self.spec.family_curvature else None,
            domain_indices if self.spec.domain_curvature else None,
        )

    def feature_derivative(
        self,
        values: np.ndarray,
        *,
        signal_kind: str,
        family_name: str | None = None,
        other_family_name: str | None = None,
        domain_indices: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Apply the configured signal derivative."""
        return signal_derivative(
            values,
            self.params,
            signal_kind,
            family_name if self.spec.family_curvature else None,
            other_family_name if self.spec.family_curvature else None,
            domain_indices if self.spec.domain_curvature else None,
        )

    def pair_signal_total(self, x_hi: np.ndarray, x_lo: np.ndarray) -> np.ndarray:
        """Aggregate high/low CC pair signal."""
        lo_scale = float(self.params["beta"]) if self.quality_discount else 1.0
        return np.asarray(x_hi, dtype=float) + lo_scale * np.asarray(x_lo, dtype=float)

    def pair_signal_partials(self, _x_hi: float, _x_lo: float) -> tuple[float, float]:
        """Partial derivatives of the high/low pair signal."""
        lo_scale = float(self.params["beta"]) if self.quality_discount else 1.0
        return 1.0, lo_scale

    def build_design(self, weights: np.ndarray) -> np.ndarray:
        """Build the exact repo design matrix for power_family_penalty."""
        x = self.retained_x(weights)
        features: list[np.ndarray] = []
        group_totals: list[np.ndarray] = []
        family_group_totals: dict[str, list[np.ndarray]] = {family_name: [] for family_name in self.family_totals}

        for idx in self.packet.singletons:
            family_name = self.domain_to_family[idx]
            features.append(
                self.feature_transform(
                    x[:, idx],
                    signal_kind=self.spec.signal_kind,
                    family_name=family_name,
                    domain_indices=(idx,),
                )[:, None]
            )
            group_totals.append(x[:, idx])
            family_group_totals[family_name].append(x[:, idx])

        for hi, lo in self.packet.pairs:
            hi_family = self.domain_to_family[hi]
            lo_family = self.domain_to_family[lo]
            signal_total = self.pair_signal_total(x[:, hi], x[:, lo])
            features.append(
                self.feature_transform(
                    signal_total,
                    signal_kind=self.spec.signal_kind,
                    family_name=hi_family,
                    other_family_name=lo_family,
                    domain_indices=(hi, lo),
                )[:, None]
            )
            total = x[:, hi] + x[:, lo]
            group_totals.append(total)
            family_group_totals[hi_family].append(total)

        for family_name in self.family_totals:
            members = self.packet.family_map[family_name]
            family_total = np.sum(x[:, members], axis=1)
            features.append(
                self.feature_transform(
                    family_total,
                    signal_kind=self.spec.family_signal_kind,
                    family_name=family_name,
                    domain_indices=tuple(int(member) for member in members),
                )[:, None]
            )

        penalties: list[np.ndarray] = []
        for family_name in self.family_totals:
            tau_f = family_tau(self.params, family_name)
            penalty_inputs = np.stack(family_group_totals[family_name], axis=1)
            penalties.append(np.sum(softplus(np.log1p(penalty_inputs) - tau_f) ** 2, axis=1, keepdims=True))

        design = np.hstack(features + penalties)
        design[:, : len(features)] *= -1.0
        return design

    def fit(self, weights: np.ndarray, targets: np.ndarray) -> GenericFamilyPenaltyCalibrationSurrogate:
        """Fit the nonnegative linear head with optional ridge rows."""
        design = self.build_design(weights)
        design_mean = design.mean(axis=0, keepdims=True)
        target_mean = float(targets.mean())
        centered_design = design - design_mean
        centered_targets = targets - target_mean
        reg = float(self.params["reg"])
        if reg > 0.0:
            centered_design = np.vstack([centered_design, np.sqrt(reg) * np.eye(centered_design.shape[1])])
            centered_targets = np.concatenate([centered_targets, np.zeros(centered_design.shape[1], dtype=float)])
        coef, _ = nnls(centered_design, centered_targets)
        self.coef_ = coef
        self.intercept_ = float(target_mean - (design_mean @ coef).item())
        return self

    def predict(self, weights: np.ndarray) -> np.ndarray:
        """Predict BPB for phase weights."""
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model must be fit before prediction")
        return np.asarray(self.intercept_ + self.build_design(weights) @ self.coef_, dtype=float)

    def components(self) -> dict[str, Any]:
        """Return linear-head component blocks for raw optimization."""
        if self.coef_ is None:
            raise RuntimeError("Model must be fit")
        n_singletons = len(self.packet.singletons)
        n_pairs = len(self.packet.pairs)
        n_families = len(self.family_totals)
        offset = 0
        singleton_coef = np.asarray(self.coef_[offset : offset + n_singletons], dtype=float)
        offset += n_singletons
        pair_coef = np.asarray(self.coef_[offset : offset + n_pairs], dtype=float)
        offset += n_pairs
        family_coef = {
            family_name: float(coef)
            for family_name, coef in zip(self.family_totals, self.coef_[offset : offset + n_families], strict=True)
        }
        offset += n_families
        family_group_penalty_coef = {
            family_name: float(coef)
            for family_name, coef in zip(self.family_totals, self.coef_[offset : offset + n_families], strict=True)
        }
        return {
            "singleton_coef": singleton_coef,
            "pair_coef": pair_coef,
            "family_coef": family_coef,
            "global_penalty_coef": 0.0,
            "family_group_penalty_coef": family_group_penalty_coef,
            "family_total_penalty_coef": {family_name: 0.0 for family_name in self.family_totals},
        }


def build_model(packet: GenericFamilyPacket, params: dict[str, float]) -> GenericFamilyPenaltyCalibrationSurrogate:
    """Build the exact no-L2 surrogate target variant."""
    return GenericFamilyPenaltyCalibrationSurrogate(packet, params=params, spec=VariantSpec())


def param_keys() -> tuple[str, ...]:
    """Return nonlinear parameter keys for power_family_penalty."""
    return (
        "eta",
        "lam",
        "reg",
        "beta",
        "a_broad_text",
        "a_tech_code",
        "a_reasoning",
        "tau_broad_text",
        "tau_tech_code",
        "tau_reasoning",
    )


def pack_params(params: dict[str, float]) -> np.ndarray:
    """Pack full power_family_penalty params into unconstrained coordinates."""
    beta = float(np.clip(params["beta"], 1e-8, 1.0 - 1.0e-8))
    return np.asarray(
        [
            np.log(float(params["eta"])),
            np.log(float(params["lam"])),
            np.log(float(params["reg"])),
            np.log(beta / (1.0 - beta)),
            np.log(float(params["a_broad_text"])),
            np.log(float(params["a_tech_code"])),
            np.log(float(params["a_reasoning"])),
            float(params["tau_broad_text"]),
            float(params["tau_tech_code"]),
            float(params["tau_reasoning"]),
        ],
        dtype=float,
    )


def unpack_params(z: np.ndarray) -> dict[str, float]:
    """Decode full power_family_penalty params from unconstrained coordinates."""
    return {
        "eta": float(np.exp(np.clip(z[0], -8.0, 8.0))),
        "lam": float(np.exp(np.clip(z[1], -12.0, 4.0))),
        "reg": float(np.exp(np.clip(z[2], -18.0, 0.0))),
        "beta": float(np.clip(sigmoid_scalar_clipped(float(z[3])), 1e-6, 1.0 - 1e-6)),
        "a_broad_text": float(np.exp(np.clip(z[4], np.log(0.02), np.log(2.0)))),
        "a_tech_code": float(np.exp(np.clip(z[5], np.log(0.02), np.log(2.0)))),
        "a_reasoning": float(np.exp(np.clip(z[6], np.log(0.02), np.log(2.0)))),
        "tau_broad_text": float(np.clip(z[7], -2.0, 8.0)),
        "tau_tech_code": float(np.clip(z[8], -2.0, 8.0)),
        "tau_reasoning": float(np.clip(z[9], -2.0, 8.0)),
    }


def pack_no_l2_params(params: dict[str, float]) -> np.ndarray:
    """Pack no-L2 params while dropping reg from optimizer coordinates."""
    full = dict(params)
    full["reg"] = float(np.exp(REG_PACK_PLACEHOLDER_LOG))
    packed = pack_params(full)
    return np.concatenate([packed[:2], packed[3:]])


def unpack_no_l2_params(z: np.ndarray) -> dict[str, float]:
    """Decode no-L2 optimizer coordinates and force reg exactly to zero."""
    full_z = np.insert(np.asarray(z, dtype=float), 2, REG_PACK_PLACEHOLDER_LOG)
    params = unpack_params(full_z)
    params["reg"] = REG_FIXED
    return params


def extract_params(row: pd.Series | dict[str, Any]) -> dict[str, float]:
    """Extract GRP nonlinear params from a CSV row."""
    return {key: float(row[key]) for key in param_keys() if key in row and pd.notna(row[key])}


def base_best_params(data_dir: Path) -> dict[str, float]:
    """Load the regularized power_family_penalty best row used to seed no-L2 tuning."""
    frame = pd.read_csv(data_dir / "grp_penalty_calibration_variants_best.csv")
    matches = frame.loc[(frame["variant"] == VARIANT_NAME) & (frame["stage"] == "refine")]
    if matches.empty:
        raise ValueError("Missing refined power_family_penalty row")
    params = extract_params(matches.iloc[0])
    params["reg"] = REG_FIXED
    return params


def included_no_l2_best_params(data_dir: Path) -> dict[str, float]:
    """Load the included no-L2 best params produced by the full repo retune."""
    row = included_no_l2_best_row(data_dir)
    params = extract_params(row)
    params["reg"] = REG_FIXED
    return params


def included_no_l2_best_row(data_dir: Path) -> dict[str, Any]:
    """Load the included no-L2 best row produced by the full repo retune."""
    frame = pd.read_csv(data_dir / "grp_power_family_penalty_no_l2_retune_best.csv")
    matches = frame.loc[frame["variant"] == NO_L2_VARIANT_NAME]
    if matches.empty:
        raise ValueError("Missing no-L2 best row")
    return matches.iloc[0].to_dict()


def with_updates(base: dict[str, float], **updates: float) -> dict[str, float]:
    """Return base params with float updates and reg fixed to zero."""
    row = dict(base)
    row.update({key: float(value) for key, value in updates.items()})
    row["reg"] = REG_FIXED
    return row


def start_bank(data_dir: Path) -> tuple[dict[str, float], ...]:
    """Build the exact deterministic 9-start no-L2 start bank."""
    base = base_best_params(data_dir)
    starts = [
        dict(base),
        with_updates(base, eta=base["eta"] * 0.8, lam=max(base["lam"] * 0.5, 1e-8)),
        with_updates(base, eta=base["eta"] * 1.2, lam=min(base["lam"] * 2.0 + 1e-8, 1.0)),
        with_updates(base, beta=max(base["beta"] - 0.08, 0.05), tau_broad_text=base["tau_broad_text"] - 0.4),
        with_updates(base, beta=min(base["beta"] + 0.08, 0.95), tau_tech_code=base["tau_tech_code"] + 0.5),
        with_updates(base, a_broad_text=np.clip(base["a_broad_text"] * 0.75, 0.02, 2.0)),
        with_updates(base, a_tech_code=np.clip(base["a_tech_code"] * 1.5, 0.02, 2.0)),
        with_updates(base, a_reasoning=np.clip(base["a_reasoning"] * 1.35, 0.02, 2.0)),
        with_updates(
            base,
            tau_broad_text=base["tau_broad_text"] + 0.5,
            tau_tech_code=base["tau_tech_code"] - 0.6,
            tau_reasoning=base["tau_reasoning"] + 0.4,
        ),
    ]
    seen: set[tuple[tuple[str, float], ...]] = set()
    deduped: list[dict[str, float]] = []
    for row in starts:
        key = tuple(sorted((param, round(float(value), 8)) for param, value in row.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append({param: float(value) for param, value in row.items()})
    return tuple(deduped)


def parameter_counts(packet: GenericFamilyPacket) -> dict[str, int]:
    """Return repo-equivalent no-L2 parameter counts."""
    signal_feature_count = len(packet.singletons) + len(packet.pairs) + len(GENERIC_FAMILY_NAMES)
    penalty_feature_count = len(GENERIC_FAMILY_NAMES)
    linear_coefficient_count = signal_feature_count + penalty_feature_count
    intercept_count = 1
    nonlinear_param_count = len(param_keys()) - 1
    return {
        "signal_feature_count": signal_feature_count,
        "penalty_feature_count": penalty_feature_count,
        "linear_coefficient_count": linear_coefficient_count,
        "intercept_count": intercept_count,
        "linear_head_param_count": linear_coefficient_count + intercept_count,
        "nonlinear_param_count": nonlinear_param_count,
        "total_param_count": nonlinear_param_count + linear_coefficient_count + intercept_count,
    }


def optimize_model(
    packet: GenericFamilyPacket,
    model: GenericFamilyPenaltyCalibrationSurrogate,
    *,
    n_random: int = 1,
    seed: int = 0,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Optimize the continuous mixture implied by a fitted surrogate."""
    if model.coef_ is None or model.intercept_ is None:
        raise RuntimeError("Model must be fit before optimization")

    parts = model.components()
    singleton_coef = parts["singleton_coef"]
    pair_coef = parts["pair_coef"]
    family_coef = parts["family_coef"]
    family_group_penalty_coef = {
        family_name: float(value) for family_name, value in parts["family_group_penalty_coef"].items()
    }
    n_domains = packet.base.m
    c0 = packet.base.c0
    c1 = packet.base.c1
    lam = float(model.params["lam"])
    eta = float(model.params["eta"])
    family_indices = {
        family_name: np.asarray(packet.family_map[family_name], dtype=int) for family_name in model.family_totals
    }
    rng = np.random.default_rng(seed)

    def value_grad_logits(z: np.ndarray) -> tuple[float, np.ndarray]:
        logits0 = z[:n_domains]
        logits1 = z[n_domains:]
        p0 = np.exp(logits0 - np.max(logits0))
        p0 /= np.sum(p0)
        p1 = np.exp(logits1 - np.max(logits1))
        p1 /= np.sum(p1)

        e0 = c0 * p0
        retained = np.exp(-lam * (1.0 - p1))
        x = retained * e0 + eta * c1 * p1
        dx_dp0 = retained * c0
        dx_dp1 = lam * retained * e0 + eta * c1

        value = float(model.intercept_)
        grad_x = np.zeros(n_domains, dtype=float)
        group_info: list[tuple[tuple[int, ...], float, str]] = []

        for local_idx, domain_idx in enumerate(packet.singletons):
            family_name = model.domain_to_family[domain_idx]
            x_value = float(x[domain_idx])
            coef = float(singleton_coef[local_idx])
            value -= (
                coef
                * model.feature_transform(
                    np.asarray([x_value]),
                    signal_kind=model.spec.signal_kind,
                    family_name=family_name,
                    domain_indices=(domain_idx,),
                )[0]
            )
            grad_x[domain_idx] -= (
                coef
                * model.feature_derivative(
                    np.asarray([x_value]),
                    signal_kind=model.spec.signal_kind,
                    family_name=family_name,
                    domain_indices=(domain_idx,),
                )[0]
            )
            group_info.append(((domain_idx,), x_value, family_name))

        for local_idx, (hi, lo) in enumerate(packet.pairs):
            hi_family = model.domain_to_family[hi]
            lo_family = model.domain_to_family[lo]
            coef = float(pair_coef[local_idx])
            signal_total = float(model.pair_signal_total(np.asarray([x[hi]]), np.asarray([x[lo]]))[0])
            value -= (
                coef
                * model.feature_transform(
                    np.asarray([signal_total]),
                    signal_kind=model.spec.signal_kind,
                    family_name=hi_family,
                    other_family_name=lo_family,
                    domain_indices=(hi, lo),
                )[0]
            )
            d_hi, d_lo = model.pair_signal_partials(float(x[hi]), float(x[lo]))
            chain = (
                coef
                * model.feature_derivative(
                    np.asarray([signal_total]),
                    signal_kind=model.spec.signal_kind,
                    family_name=hi_family,
                    other_family_name=lo_family,
                    domain_indices=(hi, lo),
                )[0]
            )
            grad_x[hi] -= chain * d_hi
            grad_x[lo] -= chain * d_lo
            group_info.append(((hi, lo), float(x[hi] + x[lo]), hi_family))

        for family_name in model.family_totals:
            members = family_indices[family_name]
            family_total = float(np.sum(x[members]))
            coef = float(family_coef[family_name])
            value -= (
                coef
                * model.feature_transform(
                    np.asarray([family_total]),
                    signal_kind=model.spec.family_signal_kind,
                    family_name=family_name,
                    domain_indices=tuple(int(member) for member in members),
                )[0]
            )
            grad_x[members] -= (
                coef
                * model.feature_derivative(
                    np.asarray([family_total]),
                    signal_kind=model.spec.family_signal_kind,
                    family_name=family_name,
                    domain_indices=tuple(int(member) for member in members),
                )[0]
            )

        for members, total, family_name in group_info:
            coef = float(family_group_penalty_coef[family_name])
            if coef == 0.0:
                continue
            tau_f = family_tau(model.params, family_name)
            inside = np.log1p(total) - tau_f
            sp = float(softplus(inside))
            value += coef * sp * sp
            if sp == 0.0:
                continue
            common = coef * 2.0 * sp * float(sigmoid(inside)) / (1.0 + total)
            for idx in members:
                grad_x[idx] += common

        grad_p0 = grad_x * dx_dp0
        grad_p1 = grad_x * dx_dp1
        grad_logits0 = p0 * (grad_p0 - np.dot(grad_p0, p0))
        grad_logits1 = p1 * (grad_p1 - np.dot(grad_p1, p1))
        return value, np.concatenate([grad_logits0, grad_logits1])

    starts = [np.zeros(2 * n_domains, dtype=float)]
    starts.extend(
        np.concatenate([rng.normal(scale=0.2, size=n_domains), rng.normal(scale=0.2, size=n_domains)])
        for _ in range(n_random)
    )
    best = None
    for start in starts:
        result = minimize(
            lambda z: value_grad_logits(z)[0],
            start,
            jac=lambda z: value_grad_logits(z)[1],
            method="L-BFGS-B",
            options={"maxiter": 400},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    if best is None:
        raise RuntimeError("GRP raw optimization failed")
    z = np.asarray(best.x, dtype=float)
    logits0 = z[:n_domains]
    logits1 = z[n_domains:]
    phase0 = np.exp(logits0 - np.max(logits0))
    phase0 /= np.sum(phase0)
    phase1 = np.exp(logits1 - np.max(logits1))
    phase1 /= np.sum(phase1)
    return best, phase0, phase1


def oof_metrics(
    packet: GenericFamilyPacket,
    params: dict[str, float],
    *,
    seed: int = CV_SEED,
    lower_tail_frac: float = LOWER_TAIL_FRAC,
    support_top_k: int = TRUSTBLEND_TOPK_ACTUAL,
) -> dict[str, float]:
    """Compute repo-equivalent out-of-fold calibration metrics."""
    y = packet.base.y
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros_like(y)
    fold_regrets: list[float] = []
    depopt_scores: list[float] = []
    rawopt_support_scores: list[float] = []

    for tr, te in kf.split(packet.base.w):
        model = build_model(packet, params).fit(packet.base.w[tr], y[tr])
        pred = model.predict(packet.base.w[te])
        oof[te] = pred
        fold_regrets.append(float(y[te][int(np.argmin(pred))] - np.min(y[te])))

        raw_result, phase0, phase1 = optimize_model(packet, model, n_random=1, seed=seed)
        raw_weights = np.stack([phase0, phase1], axis=0)
        distances = average_phase_tv_distance(packet.base.w[te], raw_weights[None, :, :])
        nearest_count = min(int(support_top_k), len(te))
        nearest_idx = np.argsort(distances)[:nearest_count]
        depopt_scores.append(max(float(np.min(y[te][nearest_idx])) - float(raw_result.fun), 0.0))
        rawopt_support_scores.append(float(distances[nearest_idx[0]]))

    residuals = oof - y
    cv_rmse = float(np.sqrt(np.mean(residuals**2)))
    tail_count = max(5, int(np.ceil(float(lower_tail_frac) * float(len(y)))))
    tail_idx = np.argsort(oof)[:tail_count]
    lower_tail_optimism = float(np.mean(np.maximum(y[tail_idx] - oof[tail_idx], 0.0)))
    mean_regret = float(np.mean(fold_regrets))
    mean_depopt = float(np.mean(depopt_scores))
    mean_support = float(np.mean(rawopt_support_scores))
    objective = (
        CALIBRATION_CV_WEIGHT * cv_rmse
        + CALIBRATION_FOLDMEAN_WEIGHT * mean_regret
        + CALIBRATION_TAIL_WEIGHT * lower_tail_optimism
        + CALIBRATION_DEPOPT_WEIGHT * mean_depopt
        + CALIBRATION_SUPPORT_WEIGHT * mean_support
    )
    return {
        "cv_rmse": cv_rmse,
        "cv_regret_at_1": float(y[int(np.argmin(oof))] - np.min(y)),
        "cv_foldmean_regret_at_1": mean_regret,
        "lower_tail_optimism": lower_tail_optimism,
        "cv_depopt_best8": mean_depopt,
        "cv_rawopt_nearest_tv": mean_support,
        "objective": objective,
    }


def full_metrics(packet: GenericFamilyPacket, model: GenericFamilyPenaltyCalibrationSurrogate) -> dict[str, Any]:
    """Compute full repo-style metrics for a fitted no-L2 model."""
    metrics = oof_metrics(packet, model.params, seed=CV_SEED)
    train_pred = model.predict(packet.base.w)
    raw_result, phase0, phase1 = optimize_model(packet, model, seed=CV_SEED)
    raw_weights = np.stack([phase0, phase1], axis=0)
    raw_distances = average_phase_tv_distance(packet.base.w, raw_weights[None, :, :])
    nearest_idx = int(np.argmin(raw_distances))
    return {
        **metrics,
        "train_rmse": float(np.sqrt(np.mean((train_pred - packet.base.y) ** 2))),
        "raw_predicted_optimum_value": float(raw_result.fun),
        "raw_nearest_observed_tv": float(raw_distances[nearest_idx]),
        "raw_nearest_observed_run_name": str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
        "raw_nearest_observed_value": float(packet.base.y[nearest_idx]),
        "raw_phase0_lt_1e4": int(np.sum(phase0 < 1e-4)),
        "raw_phase1_lt_1e4": int(np.sum(phase1 < 1e-4)),
    }


def coarse_rows(packet: GenericFamilyPacket, starts: tuple[dict[str, float], ...]) -> pd.DataFrame:
    """Score the deterministic coarse start bank."""
    rows: list[dict[str, float | bool | str]] = []
    for start_id, params in enumerate(starts):
        rows.append(
            {
                "variant": NO_L2_VARIANT_NAME,
                "surrogate_variant": VARIANT_NAME,
                "stage": "coarse",
                "start_id": int(start_id),
                **params,
                **oof_metrics(packet, params, seed=CV_SEED),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["objective", "cv_rmse", "cv_depopt_best8"],
        ascending=[True, True, True],
    )


def refine_rows(
    packet: GenericFamilyPacket,
    starts: tuple[dict[str, float], ...],
    *,
    coarse_top_k: int,
    method: str,
) -> tuple[pd.DataFrame, dict[str, float | bool | str], pd.DataFrame]:
    """Run the exact Powell refinement loop."""
    coarse_frame = coarse_rows(packet, starts)
    chosen_ids = coarse_frame["start_id"].head(int(coarse_top_k)).tolist()
    best_metrics: dict[str, float | bool | str] | None = None
    best_objective = float("inf")
    refine_records: list[dict[str, float | bool | str]] = []

    for chosen_rank, start_id in enumerate(chosen_ids):
        start = pack_no_l2_params(starts[start_id])
        cache: dict[tuple[float, ...], float] = {}

        def objective(z: np.ndarray, _cache: dict[tuple[float, ...], float] = cache) -> float:
            key = tuple(np.round(np.asarray(z, dtype=float), 8))
            if key not in _cache:
                metrics = oof_metrics(packet, unpack_no_l2_params(z), seed=CV_SEED)
                _cache[key] = float(metrics["objective"])
            return _cache[key]

        options = {
            "L-BFGS-B": {"maxiter": 80, "ftol": 1e-6},
            "Nelder-Mead": {"maxiter": 400, "xatol": 1e-4, "fatol": 1e-6},
            "Powell": {"maxiter": 30, "xtol": 1e-4, "ftol": 1e-6},
        }.get(method, {"maxiter": 120})
        result = minimize(objective, start, method=method, options=options)
        params = unpack_no_l2_params(np.asarray(result.x, dtype=float))
        metrics = oof_metrics(packet, params, seed=CV_SEED)
        row = {
            "variant": NO_L2_VARIANT_NAME,
            "surrogate_variant": VARIANT_NAME,
            "stage": "refine",
            "chosen_rank": int(chosen_rank),
            "start_id": int(start_id),
            "success": bool(result.success),
            "message": str(result.message),
            **params,
            **metrics,
        }
        refine_records.append(row)
        if float(row["objective"]) < best_objective:
            best_objective = float(row["objective"])
            best_metrics = row
    if best_metrics is None:
        raise RuntimeError("No-L2 retune failed")
    return coarse_frame, best_metrics, pd.DataFrame.from_records(refine_records)


def best_row(packet: GenericFamilyPacket, params: dict[str, float], best_metrics: dict[str, Any]) -> dict[str, Any]:
    """Fit final model and return repo-style best row."""
    model = build_model(packet, params).fit(packet.base.w, packet.base.y)
    return {
        "variant": NO_L2_VARIANT_NAME,
        "surrogate_variant": VARIANT_NAME,
        "stage": "refine",
        "success": bool(best_metrics.get("success", True)),
        "message": str(best_metrics.get("message", "included_best_params")),
        **params,
        **full_metrics(packet, model),
        **parameter_counts(packet),
        "retuned": True,
        "objective_metric": DEFAULT_TARGET,
        "notes": "Full nonlinear retune of power_family_penalty with reg fixed exactly to 0.0.",
    }


def write_model_outputs(
    out: Path,
    packet: GenericFamilyPacket,
    params: dict[str, float],
    best_metrics: dict[str, Any],
) -> None:
    """Write final predictions, coefficients, optimum, and summary outputs."""
    out.mkdir(parents=True, exist_ok=True)
    model = build_model(packet, params).fit(packet.base.w, packet.base.y)
    predictions = model.predict(packet.base.w)
    raw_result, phase0, phase1 = optimize_model(packet, model, seed=CV_SEED)
    weights = np.stack([phase0, phase1], axis=0)
    distances = average_phase_tv_distance(packet.base.w, weights[None, :, :])
    nearest_idx = int(np.argmin(distances))
    row = best_row(packet, params, best_metrics)

    pred_frame = packet.base.frame[
        [column for column in ["run_id", "run_name", "source_experiment", "status"] if column in packet.base.frame]
    ].copy()
    pred_frame["actual"] = packet.base.y
    pred_frame["predicted"] = predictions
    pred_frame["residual"] = predictions - packet.base.y
    pred_frame.to_csv(out / "predictions.csv", index=False)

    coef_frame = pd.DataFrame({"coefficient": model.coef_})
    coef_frame.to_csv(out / "linear_head_coefficients.csv", index=False)

    opt_rows = []
    for phase_idx, phase in enumerate(("phase_0", "phase_1")):
        for domain, value in zip(packet.base.domain_names, weights[phase_idx], strict=True):
            opt_rows.append({"phase": phase, "domain": domain, "weight": float(value)})
    pd.DataFrame(opt_rows).to_csv(out / "raw_optimum_weights.csv", index=False)
    pd.DataFrame([row]).to_csv(out / "grp_power_family_penalty_no_l2_retune_best.csv", index=False)
    summary = {
        "best_row": row,
        "raw_optimizer_success": bool(raw_result.success),
        "raw_optimizer_message": str(raw_result.message),
        "nearest_observed_run_name": str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
        "nearest_observed_tv": float(distances[nearest_idx]),
        "params": params,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PACKET_ROOT / "outputs" / "grp_no_l2_exact")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--mode", choices=("fit-best", "score-best", "retune"), default="fit-best")
    parser.add_argument("--method", default="Powell")
    parser.add_argument("--coarse-top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Run exact GRP no-L2 reproduction or retune."""
    args = parse_args()
    packet = load_packet(args.data_dir, target=args.target)
    if args.mode == "retune":
        starts = start_bank(args.data_dir)
        coarse_frame, best_metrics, refine_frame = refine_rows(
            packet,
            starts,
            coarse_top_k=args.coarse_top_k,
            method=args.method,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        coarse_frame.to_csv(args.output_dir / "grp_power_family_penalty_no_l2_retune_coarse.csv", index=False)
        refine_frame.sort_values("objective").to_csv(
            args.output_dir / "grp_power_family_penalty_no_l2_retune_refine.csv",
            index=False,
        )
        params = {key: float(best_metrics[key]) for key in param_keys() if key != "reg"}
        params["reg"] = REG_FIXED
        write_model_outputs(args.output_dir, packet, params, best_metrics)
        return

    params = included_no_l2_best_params(args.data_dir)
    best_metrics: dict[str, Any] = included_no_l2_best_row(args.data_dir)
    if args.mode == "score-best":
        # Recompute the full expensive CV objective for direct verification.
        best_metrics.update(oof_metrics(packet, params, seed=CV_SEED))
    write_model_outputs(args.output_dir, packet, params, best_metrics)


if __name__ == "__main__":
    main()
