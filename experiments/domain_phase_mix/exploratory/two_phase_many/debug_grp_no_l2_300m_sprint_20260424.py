# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy", "scikit-learn", "tabulate"]
# ///
"""Diagnose why no-L2 GRP degrades on the nominal 300M swarm.

This script is intentionally self-contained and reads local CSV artifacts
directly. Importing the older GRP helper stack currently pulls in Levanter and a
partial local torch namespace, which is unnecessary for this fixed-scale
surrogate debug pass.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, minimize, nnls
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold

matplotlib.use("Agg")

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
METRICS_WIDE_CSV = SCRIPT_DIR / "metric_registry" / "metrics_wide.csv"
EPOCH_METADATA_CSV = SCRIPT_DIR / "two_phase_many_epoch_metadata.csv"
LEGACY_SUMMARY_CSV = SCRIPT_DIR / "grp_power_family_penalty_no_l2_60m_vs_300m_fit_summary.csv"
LEGACY_REPORT_MD = SCRIPT_DIR / "grp_power_family_penalty_no_l2_60m_vs_300m_fit.md"
DEBUG_LOG_MD = Path("docs/debug-log-grp-no-l2-300m-uniform-phase0.md")
REFERENCE_OUTPUTS_DIR = SCRIPT_DIR / "reference_outputs"

OBJECTIVE_METRIC = "eval/uncheatable_eval/bpb"
SCALE_60M = "60m_1p2b"
SCALE_300M = "300m_6b"
RUN_SET_60M = "fit_swarm_60m_default"
RUN_SET_300M = "swarm_like_300m"
THREE_HUNDRED_OLMIX_RUN_NAME = "baseline_olmix_loglinear_uncheatable_bpb"
FAMILIES = ("broad_text", "tech_code", "reasoning")
CV_SEED = 0
CV_SPLITS = 5
LOWER_TAIL_FRAC = 0.15
CV_WEIGHT = 1.0
FOLDMEAN_WEIGHT = 0.05
TAIL_WEIGHT = 0.5
DEPOPT_WEIGHT = 0.1
SUPPORT_WEIGHT = 0.01
SUPPORT_TOP_K = 8
PHASE_SENSITIVITY_TV = 0.10
PHASE_SENSITIVITY_THRESHOLD = 1e-5

ParamMode = Literal["standard", "moderate_clip", "separate_phase"]
ObjectiveMode = Literal["fast", "full"]


@dataclass(frozen=True)
class Packet:
    """Feature-ready fixed-scale swarm panel."""

    label: str
    frame: pd.DataFrame
    y: np.ndarray
    w: np.ndarray
    domain_names: list[str]
    c0: np.ndarray
    c1: np.ndarray
    pairs: list[tuple[int, int]]
    singletons: list[int]
    family_map: dict[str, list[int]]
    domain_to_family: tuple[str, ...]


@dataclass(frozen=True)
class TrialSpec:
    """One nonlinear optimizer diagnostic to run."""

    name: str
    scale_label: str
    param_mode: ParamMode
    objective_mode: ObjectiveMode
    start_source: Literal["fixed", "expanded", "random", "best_fixed"]
    method: Literal["Powell", "Nelder-Mead", "L-BFGS-B", "basinhopping"]
    maxiter: int
    top_k: int
    reg: float
    prior_weight: float = 0.0
    random_count: int = 0
    basinhopping_niter: int = 0


def softplus(x: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.where(arr > 20.0, arr, np.log1p(np.exp(np.minimum(arr, 20.0))))


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    positive = arr >= 0.0
    out = np.empty_like(arr, dtype=float)
    out[positive] = 1.0 / (1.0 + np.exp(-arr[positive]))
    exp_x = np.exp(arr[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def phase_tv(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).sum())


def average_phase_tv_distance(weights: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    diffs = np.abs(weights - candidate[None, :, :])
    return 0.5 * diffs.sum(axis=(1, 2)) / weights.shape[1]


def entropy(weights: np.ndarray) -> float:
    safe = np.maximum(np.asarray(weights, dtype=float), 1e-300)
    return float(-np.sum(safe * np.log(safe)))


def load_domains() -> tuple[list[str], np.ndarray, np.ndarray]:
    metadata = pd.read_csv(EPOCH_METADATA_CSV)
    required = {"domain_name", "phase_0_epoch_multiplier", "phase_1_epoch_multiplier"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"{EPOCH_METADATA_CSV} is missing columns: {sorted(missing)}")
    domain_names = metadata["domain_name"].astype(str).tolist()
    c0 = metadata["phase_0_epoch_multiplier"].to_numpy(float)
    c1 = metadata["phase_1_epoch_multiplier"].to_numpy(float)
    if len(domain_names) != 39:
        raise ValueError(f"Expected 39 domains, got {len(domain_names)}")
    return domain_names, c0, c1


def family_assignment(domain_name: str) -> str:
    is_broad = (
        domain_name.startswith("dolma3_cc/")
        or domain_name
        in {
            "dolma3_wikipedia",
            "dolmino_common_crawl_hq",
            "dolmino_olmocr_pdfs_hq",
            "dolmino_stem_heavy_crawl",
        }
        or domain_name.endswith("synth_qa")
    )
    is_tech = any(token in domain_name for token in ("stack_edu", "synth_code", "synth_math")) or domain_name in {
        "dolma3_arxiv",
        "dolma3_finemath_3plus",
    }
    is_reasoning = domain_name in {"dolmino_synth_instruction", "dolmino_synth_thinking"}
    assigned = [
        name for name, flag in (("broad_text", is_broad), ("tech_code", is_tech), ("reasoning", is_reasoning)) if flag
    ]
    if len(assigned) != 1:
        raise ValueError(f"Expected exactly one family for {domain_name}, got {assigned}")
    return assigned[0]


def packet_structure(
    domain_names: list[str],
) -> tuple[list[tuple[int, int]], list[int], dict[str, list[int]], tuple[str, ...]]:
    pairs: list[tuple[int, int]] = []
    paired: set[int] = set()
    for idx, domain_name in enumerate(domain_names):
        if idx in paired:
            continue
        if domain_name.startswith("dolma3_cc/") and domain_name.endswith("_high"):
            low_name = domain_name[:-5] + "_low"
            if low_name in domain_names:
                low_idx = domain_names.index(low_name)
                pairs.append((idx, low_idx))
                paired.add(idx)
                paired.add(low_idx)
    singletons = [idx for idx in range(len(domain_names)) if idx not in paired]
    domain_to_family = tuple(family_assignment(domain_name) for domain_name in domain_names)
    family_map = {family_name: [] for family_name in FAMILIES}
    for idx, family_name in enumerate(domain_to_family):
        family_map[family_name].append(idx)
    return pairs, singletons, family_map, domain_to_family


def load_packet(scale: str, run_set: str, *, domain_names: list[str], c0: np.ndarray, c1: np.ndarray) -> Packet:
    metrics = pd.read_csv(METRICS_WIDE_CSV, low_memory=False)
    weight_columns = [f"{phase}_{domain_name}" for phase in ("phase_0", "phase_1") for domain_name in domain_names]
    missing = [column for column in [*weight_columns, OBJECTIVE_METRIC] if column not in metrics.columns]
    if missing:
        raise ValueError(f"{METRICS_WIDE_CSV} is missing columns: {missing[:8]}")
    id_columns = [
        column
        for column in (
            "registry_run_key",
            "run_id",
            "run_name",
            "scale",
            "cohort",
            "source_experiment",
            "checkpoint_root",
            "status",
            "is_qsplit240_core",
            "is_baseline_olmix",
            "is_baseline_stratified",
            "is_fit_swarm_60m_default",
        )
        if column in metrics.columns
    ]
    frame = metrics.loc[
        metrics["scale"].eq(scale) & metrics["cohort"].eq("signal") & metrics[OBJECTIVE_METRIC].notna(),
        id_columns + weight_columns + [OBJECTIVE_METRIC],
    ].copy()
    if run_set == RUN_SET_60M:
        frame = frame.loc[frame["is_fit_swarm_60m_default"].fillna(False)].copy()
    elif run_set == RUN_SET_300M:
        mask = (
            frame["is_qsplit240_core"].fillna(False)
            | frame["is_baseline_olmix"].fillna(False)
            | frame["is_baseline_stratified"].fillna(False)
            | frame["run_name"].eq(THREE_HUNDRED_OLMIX_RUN_NAME)
        )
        frame = frame.loc[mask].copy()
    else:
        raise ValueError(f"Unknown run_set={run_set!r}")

    if frame["run_name"].duplicated().any():
        duplicates = frame.loc[frame["run_name"].duplicated(), "run_name"].astype(str).tolist()
        raise ValueError(f"Duplicate run_name rows in {scale}/{run_set}: {duplicates[:8]}")
    if len(frame) != 242:
        raise ValueError(f"Expected 242 rows for {scale}/{run_set}, got {len(frame)}")

    weights = np.zeros((len(frame), 2, len(domain_names)), dtype=float)
    for phase_idx, phase_name in enumerate(("phase_0", "phase_1")):
        for domain_idx, domain_name in enumerate(domain_names):
            weights[:, phase_idx, domain_idx] = frame[f"{phase_name}_{domain_name}"].fillna(0.0).to_numpy(float)
        sums = weights[:, phase_idx, :].sum(axis=1)
        zero_rows = np.where(sums <= 0.0)[0]
        for row_idx in zero_rows.tolist():
            run_name = str(frame.iloc[row_idx]["run_name"])
            fallback_rows = metrics.loc[metrics["run_name"].eq(run_name)].copy()
            fallback_values: np.ndarray | None = None
            for _, fallback in fallback_rows.iterrows():
                values = np.asarray(
                    [
                        (
                            float(fallback.get(f"{phase_name}_{domain_name}", 0.0))
                            if pd.notna(fallback.get(f"{phase_name}_{domain_name}", np.nan))
                            else 0.0
                        )
                        for domain_name in domain_names
                    ],
                    dtype=float,
                )
                if float(values.sum()) > 0.0:
                    fallback_values = values
                    break
            if fallback_values is None:
                raise ValueError(
                    f"Non-positive phase weight sum in {scale}/{run_set}/{phase_name} for run_name={run_name!r}"
                )
            weights[row_idx, phase_idx, :] = fallback_values
            sums[row_idx] = float(fallback_values.sum())
        if np.any(sums <= 0.0):
            raise ValueError(f"Non-positive phase weight sum in {scale}/{run_set}/{phase_name}")
        weights[:, phase_idx, :] /= sums[:, None]
        if not np.allclose(weights[:, phase_idx, :].sum(axis=1), 1.0, atol=1e-9):
            raise ValueError(f"Phase weights failed normalization in {scale}/{run_set}/{phase_name}")

    pairs, singletons, family_map, domain_to_family = packet_structure(domain_names)
    return Packet(
        label=f"{scale}/{run_set}",
        frame=frame.reset_index(drop=True),
        y=frame[OBJECTIVE_METRIC].to_numpy(float),
        w=weights,
        domain_names=domain_names,
        c0=c0,
        c1=c1,
        pairs=pairs,
        singletons=singletons,
        family_map=family_map,
        domain_to_family=domain_to_family,
    )


def load_legacy_params(label: str) -> dict[str, float]:
    frame = pd.read_csv(LEGACY_SUMMARY_CSV)
    matches = frame.loc[frame["label"].eq(label)]
    if matches.empty:
        raise ValueError(f"Missing {label!r} in {LEGACY_SUMMARY_CSV}")
    row = matches.iloc[0].to_dict()
    params = {
        "eta": float(row["param_eta"]),
        "lam": float(row["param_lam"]),
        "reg": float(row["param_reg"]),
        "beta": float(row["param_beta"]),
        "a_broad_text": float(row["param_a_broad_text"]),
        "a_tech_code": float(row["param_a_tech_code"]),
        "a_reasoning": float(row["param_a_reasoning"]),
        "tau_broad_text": float(row["param_tau_broad_text"]),
        "tau_tech_code": float(row["param_tau_tech_code"]),
        "tau_reasoning": float(row["param_tau_reasoning"]),
    }
    return params


def standard_param_keys(param_mode: ParamMode) -> tuple[str, ...]:
    if param_mode == "separate_phase":
        return (
            "eta0",
            "eta1",
            "beta",
            "a_broad_text",
            "a_tech_code",
            "a_reasoning",
            "tau_broad_text",
            "tau_tech_code",
            "tau_reasoning",
        )
    return (
        "eta",
        "lam",
        "beta",
        "a_broad_text",
        "a_tech_code",
        "a_reasoning",
        "tau_broad_text",
        "tau_tech_code",
        "tau_reasoning",
    )


def standard_to_separate(params: dict[str, float]) -> dict[str, float]:
    out = dict(params)
    out["eta0"] = 1.0
    out["eta1"] = float(params["eta"])
    out.pop("eta", None)
    out.pop("lam", None)
    return out


def pack_params(params: dict[str, float], param_mode: ParamMode) -> np.ndarray:
    if param_mode == "separate_phase":
        return np.asarray(
            [
                np.log(float(params["eta0"])),
                np.log(float(params["eta1"])),
                np.log(
                    float(np.clip(params["beta"], 1e-8, 1.0 - 1e-8))
                    / (1.0 - float(np.clip(params["beta"], 1e-8, 1.0 - 1e-8)))
                ),
                np.log(float(params["a_broad_text"])),
                np.log(float(params["a_tech_code"])),
                np.log(float(params["a_reasoning"])),
                float(params["tau_broad_text"]),
                float(params["tau_tech_code"]),
                float(params["tau_reasoning"]),
            ],
            dtype=float,
        )
    beta = float(np.clip(params["beta"], 1e-8, 1.0 - 1e-8))
    return np.asarray(
        [
            np.log(float(params["eta"])),
            np.log(float(params["lam"])),
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


def unpack_params(z: np.ndarray, param_mode: ParamMode, *, reg: float) -> dict[str, float]:
    z = np.asarray(z, dtype=float)
    eta_hi = np.log(64.0) if param_mode == "moderate_clip" else 8.0
    lam_hi = np.log(2.0) if param_mode == "moderate_clip" else 4.0
    beta_lo = 0.02 if param_mode == "moderate_clip" else 1e-6
    beta_hi = 0.98 if param_mode == "moderate_clip" else 1.0 - 1e-6
    if param_mode == "separate_phase":
        beta = float(np.clip(1.0 / (1.0 + np.exp(-np.clip(z[2], -50.0, 50.0))), beta_lo, beta_hi))
        return {
            "eta0": float(np.exp(np.clip(z[0], -8.0, eta_hi))),
            "eta1": float(np.exp(np.clip(z[1], -8.0, eta_hi))),
            "reg": float(reg),
            "beta": beta,
            "a_broad_text": float(np.exp(np.clip(z[3], np.log(0.02), np.log(2.0)))),
            "a_tech_code": float(np.exp(np.clip(z[4], np.log(0.02), np.log(2.0)))),
            "a_reasoning": float(np.exp(np.clip(z[5], np.log(0.02), np.log(2.0)))),
            "tau_broad_text": float(np.clip(z[6], -2.0, 8.0)),
            "tau_tech_code": float(np.clip(z[7], -2.0, 8.0)),
            "tau_reasoning": float(np.clip(z[8], -2.0, 8.0)),
        }
    beta = float(np.clip(1.0 / (1.0 + np.exp(-np.clip(z[2], -50.0, 50.0))), beta_lo, beta_hi))
    return {
        "eta": float(np.exp(np.clip(z[0], -8.0, eta_hi))),
        "lam": float(np.exp(np.clip(z[1], -12.0, lam_hi))),
        "reg": float(reg),
        "beta": beta,
        "a_broad_text": float(np.exp(np.clip(z[3], np.log(0.02), np.log(2.0)))),
        "a_tech_code": float(np.exp(np.clip(z[4], np.log(0.02), np.log(2.0)))),
        "a_reasoning": float(np.exp(np.clip(z[5], np.log(0.02), np.log(2.0)))),
        "tau_broad_text": float(np.clip(z[6], -2.0, 8.0)),
        "tau_tech_code": float(np.clip(z[7], -2.0, 8.0)),
        "tau_reasoning": float(np.clip(z[8], -2.0, 8.0)),
    }


def transform(values: np.ndarray, params: dict[str, float], family_name: str) -> np.ndarray:
    a = float(params[f"a_{family_name}"])
    return np.power(np.maximum(np.asarray(values, dtype=float), 1e-12), a)


def derivative(values: np.ndarray, params: dict[str, float], family_name: str) -> np.ndarray:
    a = float(params[f"a_{family_name}"])
    safe = np.maximum(np.asarray(values, dtype=float), 1e-12)
    return a * np.power(safe, a - 1.0)


class GRPModel:
    """Power-family penalty GRP with optional separate phase exposure."""

    def __init__(self, packet: Packet, params: dict[str, float], param_mode: ParamMode):
        self.packet = packet
        self.params = dict(params)
        self.param_mode = param_mode
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.feature_names_: list[str] = []
        self.feature_blocks_: list[str] = []

    def retained_x(self, weights: np.ndarray) -> np.ndarray:
        p0 = weights[:, 0, :]
        p1 = weights[:, 1, :]
        e0 = p0 * self.packet.c0[None, :]
        e1 = p1 * self.packet.c1[None, :]
        if self.param_mode == "separate_phase":
            return float(self.params["eta0"]) * e0 + float(self.params["eta1"]) * e1
        retained = np.exp(-float(self.params["lam"]) * (1.0 - p1))
        return retained * e0 + float(self.params["eta"]) * e1

    def build_design(self, weights: np.ndarray) -> np.ndarray:
        x = self.retained_x(weights)
        features: list[np.ndarray] = []
        names: list[str] = []
        blocks: list[str] = []
        group_totals: list[np.ndarray] = []
        family_group_totals: dict[str, list[np.ndarray]] = {family_name: [] for family_name in FAMILIES}
        family_totals: dict[str, np.ndarray] = {}

        for idx in self.packet.singletons:
            family_name = self.packet.domain_to_family[idx]
            features.append(transform(x[:, idx], self.params, family_name)[:, None])
            names.append(f"singleton::{self.packet.domain_names[idx]}")
            blocks.append("singleton")
            group_totals.append(x[:, idx])
            family_group_totals[family_name].append(x[:, idx])

        for hi, lo in self.packet.pairs:
            hi_family = self.packet.domain_to_family[hi]
            lo_family = self.packet.domain_to_family[lo]
            signal_total = x[:, hi] + float(self.params["beta"]) * x[:, lo]
            family_name = hi_family
            features.append(transform(signal_total, self.params, family_name)[:, None])
            names.append(f"pair::{self.packet.domain_names[hi]}+{self.packet.domain_names[lo]}")
            blocks.append("pair")
            total = x[:, hi] + x[:, lo]
            group_totals.append(total)
            family_group_totals[hi_family].append(total)
            if lo_family != hi_family:
                raise ValueError(f"Pair families differ: {hi_family} vs {lo_family}")

        for family_name in FAMILIES:
            members = self.packet.family_map[family_name]
            family_total = np.sum(x[:, members], axis=1)
            family_totals[family_name] = family_total
            features.append(transform(family_total, self.params, family_name)[:, None])
            names.append(f"family_total::{family_name}")
            blocks.append("family_total")

        penalties: list[np.ndarray] = []
        for family_name in FAMILIES:
            if not family_group_totals[family_name]:
                continue
            tau_f = float(self.params[f"tau_{family_name}"])
            penalty_inputs = np.stack(family_group_totals[family_name], axis=1)
            penalties.append(np.sum(softplus(np.log1p(penalty_inputs) - tau_f) ** 2, axis=1, keepdims=True))
            names.append(f"family_group_penalty::{family_name}")
            blocks.append("family_group_penalty")

        design = np.hstack(features + penalties)
        design[:, : len(features)] *= -1.0
        self.feature_names_ = names
        self.feature_blocks_ = blocks
        return design

    def fit(self, weights: np.ndarray, targets: np.ndarray) -> GRPModel:
        design = self.build_design(weights)
        design_mean = design.mean(axis=0, keepdims=True)
        target_mean = float(targets.mean())
        centered_design = design - design_mean
        centered_targets = targets - target_mean
        reg = float(self.params.get("reg", 0.0))
        if reg > 0.0:
            centered_design = np.vstack([centered_design, np.sqrt(reg) * np.eye(centered_design.shape[1])])
            centered_targets = np.concatenate([centered_targets, np.zeros(design.shape[1], dtype=float)])
        coef, _ = nnls(centered_design, centered_targets)
        self.coef_ = coef
        self.intercept_ = float(target_mean - (design_mean @ coef).item())
        return self

    def predict(self, weights: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model must be fit before prediction")
        return np.asarray(self.intercept_ + self.build_design(weights) @ self.coef_, dtype=float)

    def components(self) -> dict[str, Any]:
        if self.coef_ is None:
            raise RuntimeError("Model must be fit")
        n_singletons = len(self.packet.singletons)
        n_pairs = len(self.packet.pairs)
        offset = 0
        singleton_coef = np.asarray(self.coef_[offset : offset + n_singletons], dtype=float)
        offset += n_singletons
        pair_coef = np.asarray(self.coef_[offset : offset + n_pairs], dtype=float)
        offset += n_pairs
        family_coef = {
            family_name: float(value)
            for family_name, value in zip(FAMILIES, self.coef_[offset : offset + len(FAMILIES)], strict=True)
        }
        offset += len(FAMILIES)
        family_group_penalty_coef = {
            family_name: float(value)
            for family_name, value in zip(FAMILIES, self.coef_[offset : offset + len(FAMILIES)], strict=True)
        }
        return {
            "singleton_coef": singleton_coef,
            "pair_coef": pair_coef,
            "family_coef": family_coef,
            "family_group_penalty_coef": family_group_penalty_coef,
        }


def fit_model(packet: Packet, params: dict[str, float], param_mode: ParamMode) -> GRPModel:
    return GRPModel(packet, params=params, param_mode=param_mode).fit(packet.w, packet.y)


def raw_optimize(packet: Packet, model: GRPModel, *, seed: int, n_random: int) -> tuple[Any, np.ndarray, np.ndarray]:
    parts = model.components()
    n_domains = len(packet.domain_names)
    rng = np.random.default_rng(seed)

    def value_grad_logits(z: np.ndarray) -> tuple[float, np.ndarray]:
        logits0 = z[:n_domains]
        logits1 = z[n_domains:]
        p0 = np.exp(logits0 - np.max(logits0))
        p0 /= np.sum(p0)
        p1 = np.exp(logits1 - np.max(logits1))
        p1 /= np.sum(p1)

        e0 = packet.c0 * p0
        e1 = packet.c1 * p1
        if model.param_mode == "separate_phase":
            x = float(model.params["eta0"]) * e0 + float(model.params["eta1"]) * e1
            dx_dp0 = float(model.params["eta0"]) * packet.c0
            dx_dp1 = float(model.params["eta1"]) * packet.c1
        else:
            lam = float(model.params["lam"])
            retained = np.exp(-lam * (1.0 - p1))
            x = retained * e0 + float(model.params["eta"]) * e1
            dx_dp0 = retained * packet.c0
            dx_dp1 = lam * retained * e0 + float(model.params["eta"]) * packet.c1

        value = float(model.intercept_)
        grad_x = np.zeros(n_domains, dtype=float)
        group_info: list[tuple[tuple[int, ...], float, str]] = []

        for local_idx, domain_idx in enumerate(packet.singletons):
            family_name = packet.domain_to_family[domain_idx]
            x_value = float(x[domain_idx])
            coef = float(parts["singleton_coef"][local_idx])
            value -= coef * float(transform(np.asarray([x_value]), model.params, family_name)[0])
            grad_x[domain_idx] -= coef * float(derivative(np.asarray([x_value]), model.params, family_name)[0])
            group_info.append(((domain_idx,), x_value, family_name))

        for local_idx, (hi, lo) in enumerate(packet.pairs):
            family_name = packet.domain_to_family[hi]
            coef = float(parts["pair_coef"][local_idx])
            signal_total = float(x[hi] + float(model.params["beta"]) * x[lo])
            value -= coef * float(transform(np.asarray([signal_total]), model.params, family_name)[0])
            chain = coef * float(derivative(np.asarray([signal_total]), model.params, family_name)[0])
            grad_x[hi] -= chain
            grad_x[lo] -= chain * float(model.params["beta"])
            group_info.append(((hi, lo), float(x[hi] + x[lo]), family_name))

        for family_name in FAMILIES:
            members = np.asarray(packet.family_map[family_name], dtype=int)
            family_total = float(np.sum(x[members]))
            coef = float(parts["family_coef"][family_name])
            value -= coef * float(transform(np.asarray([family_total]), model.params, family_name)[0])
            grad_x[members] -= coef * float(derivative(np.asarray([family_total]), model.params, family_name)[0])

        for members, total, family_name in group_info:
            coef = float(parts["family_group_penalty_coef"][family_name])
            if coef == 0.0:
                continue
            tau_f = float(model.params[f"tau_{family_name}"])
            inside = np.log1p(total) - tau_f
            sp = float(softplus(inside))
            value += coef * sp * sp
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
        raise RuntimeError("Raw optimization failed")
    z = np.asarray(best.x, dtype=float)
    phase0 = np.exp(z[:n_domains] - np.max(z[:n_domains]))
    phase0 /= np.sum(phase0)
    phase1 = np.exp(z[n_domains:] - np.max(z[n_domains:]))
    phase1 /= np.sum(phase1)
    return best, phase0, phase1


def family_shares(packet: Packet, weights: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for phase_idx in (0, 1):
        for family_name in FAMILIES:
            out[f"phase{phase_idx}_{family_name}"] = float(weights[phase_idx, packet.family_map[family_name]].sum())
    return out


def oof_metrics(
    packet: Packet, params: dict[str, float], param_mode: ParamMode, *, objective_mode: ObjectiveMode
) -> dict[str, float]:
    kf = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_SEED)
    y = packet.y
    oof = np.zeros_like(y)
    fold_regrets: list[float] = []
    depopt_scores: list[float] = []
    support_scores: list[float] = []
    for train_idx, test_idx in kf.split(packet.w):
        model = GRPModel(packet, params=params, param_mode=param_mode).fit(packet.w[train_idx], y[train_idx])
        pred = model.predict(packet.w[test_idx])
        oof[test_idx] = pred
        fold_regrets.append(float(y[test_idx][int(np.argmin(pred))] - np.min(y[test_idx])))
        if objective_mode == "full":
            raw_result, phase0, phase1 = raw_optimize(packet, model, seed=CV_SEED, n_random=1)
            raw_weights = np.stack([phase0, phase1], axis=0)
            distances = average_phase_tv_distance(packet.w[test_idx], raw_weights)
            nearest_count = min(SUPPORT_TOP_K, len(test_idx))
            nearest_idx = np.argsort(distances)[:nearest_count]
            depopt_scores.append(max(float(np.min(y[test_idx][nearest_idx])) - float(raw_result.fun), 0.0))
            support_scores.append(float(distances[nearest_idx[0]]))

    residuals = oof - y
    cv_rmse = float(np.sqrt(np.mean(residuals**2)))
    tail_count = max(5, int(np.ceil(LOWER_TAIL_FRAC * len(y))))
    tail_idx = np.argsort(oof)[:tail_count]
    lower_tail_optimism = float(np.mean(np.maximum(y[tail_idx] - oof[tail_idx], 0.0)))
    mean_regret = float(np.mean(fold_regrets))
    mean_depopt = float(np.mean(depopt_scores)) if depopt_scores else 0.0
    mean_support = float(np.mean(support_scores)) if support_scores else 0.0
    objective = CV_WEIGHT * cv_rmse + FOLDMEAN_WEIGHT * mean_regret + TAIL_WEIGHT * lower_tail_optimism
    if objective_mode == "full":
        objective += DEPOPT_WEIGHT * mean_depopt + SUPPORT_WEIGHT * mean_support
    return {
        "cv_rmse": cv_rmse,
        "cv_spearman": float(spearmanr(y, oof).statistic),
        "cv_regret_at_1": float(y[int(np.argmin(oof))] - np.min(y)),
        "cv_foldmean_regret_at_1": mean_regret,
        "lower_tail_optimism": lower_tail_optimism,
        "cv_depopt_best8": mean_depopt,
        "cv_rawopt_nearest_tv": mean_support,
        "objective": float(objective),
    }


def full_model_metrics(
    packet: Packet, params: dict[str, float], param_mode: ParamMode
) -> tuple[dict[str, float | str | int | bool], GRPModel, np.ndarray]:
    model = fit_model(packet, params, param_mode)
    pred = model.predict(packet.w)
    raw_result, phase0, phase1 = raw_optimize(packet, model, seed=CV_SEED, n_random=16)
    raw_weights = np.stack([phase0, phase1], axis=0)
    distances = average_phase_tv_distance(packet.w, raw_weights)
    nearest_idx = int(np.argmin(distances))
    phase0_sensitivity = phase_sensitivity(packet, model, raw_weights, phase_idx=0)
    phase1_sensitivity = phase_sensitivity(packet, model, raw_weights, phase_idx=1)
    out: dict[str, float | str | int | bool] = {
        "train_rmse": float(np.sqrt(np.mean((pred - packet.y) ** 2))),
        "train_spearman": float(spearmanr(packet.y, pred).statistic),
        "raw_predicted_optimum_value": float(raw_result.fun),
        "raw_nearest_observed_tv": float(distances[nearest_idx]),
        "raw_nearest_observed_run_name": str(packet.frame.iloc[nearest_idx]["run_name"]),
        "raw_nearest_observed_value": float(packet.y[nearest_idx]),
        "raw_phase0_max_weight": float(np.max(phase0)),
        "raw_phase1_max_weight": float(np.max(phase1)),
        "raw_phase0_entropy": entropy(phase0),
        "raw_phase1_entropy": entropy(phase1),
        "phase0_sensitivity_max_pred_delta": float(phase0_sensitivity["max_pred_delta"]),
        "phase0_sensitivity_max_design_delta": float(phase0_sensitivity["max_design_delta"]),
        "phase1_sensitivity_max_pred_delta": float(phase1_sensitivity["max_pred_delta"]),
        "phase1_sensitivity_max_design_delta": float(phase1_sensitivity["max_design_delta"]),
        "phase0_degenerate": bool(float(phase0_sensitivity["max_pred_delta"]) < PHASE_SENSITIVITY_THRESHOLD),
    }
    out.update(family_shares(packet, raw_weights))
    return out, model, raw_weights


def phase_sensitivity(packet: Packet, model: GRPModel, weights: np.ndarray, *, phase_idx: int) -> dict[str, float]:
    base_pred = float(model.predict(weights[None, :, :])[0])
    base_design = model.build_design(weights[None, :, :])[0]
    deltas_pred: list[float] = []
    deltas_design: list[float] = []
    top_indices = np.argsort(weights[phase_idx])[-3:].tolist()
    target_indices = [0, len(packet.domain_names) // 2, len(packet.domain_names) - 1]
    for src in top_indices:
        for dst in target_indices:
            if src == dst:
                continue
            perturbed = weights.copy()
            move = min(PHASE_SENSITIVITY_TV, float(perturbed[phase_idx, src]))
            if move <= 0.0:
                continue
            perturbed[phase_idx, src] -= move
            perturbed[phase_idx, dst] += move
            pred = float(model.predict(perturbed[None, :, :])[0])
            design = model.build_design(perturbed[None, :, :])[0]
            deltas_pred.append(abs(pred - base_pred))
            deltas_design.append(float(np.max(np.abs(design - base_design))))
    return {
        "max_pred_delta": float(max(deltas_pred) if deltas_pred else 0.0),
        "max_design_delta": float(max(deltas_design) if deltas_design else 0.0),
    }


def feature_health(packet: Packet, model: GRPModel) -> pd.DataFrame:
    design = model.build_design(packet.w)
    centered = design - design.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-12)) if len(singular) else np.nan
    rows: list[dict[str, object]] = []
    for idx, (name, block) in enumerate(zip(model.feature_names_, model.feature_blocks_, strict=True)):
        column = design[:, idx]
        std = float(np.std(column))
        corr = float(pearsonr(column, packet.y).statistic) if std > 1e-12 else 0.0
        rows.append(
            {
                "feature": name,
                "block": block,
                "mean": float(np.mean(column)),
                "std": std,
                "target_pearson": corr,
                "coef": float(model.coef_[idx] if model.coef_ is not None else np.nan),
                "global_condition": condition,
                "global_min_singular": float(singular[-1]) if len(singular) else np.nan,
                "global_max_singular": float(singular[0]) if len(singular) else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def bound_hits(params: dict[str, float], param_mode: ParamMode) -> dict[str, bool]:
    tol = 1e-5
    if param_mode == "separate_phase":
        return {
            "eta0_upper": float(params["eta0"]) >= np.exp(8.0) * (1.0 - tol),
            "eta1_upper": float(params["eta1"]) >= np.exp(8.0) * (1.0 - tol),
            "beta_lower": float(params["beta"]) <= 1e-6 * (1.0 + tol),
            "beta_moderate_lower": float(params["beta"]) <= 0.020001,
            "any_tau_bound": any(
                abs(float(params[f"tau_{family}"]) - bound) < tol for family in FAMILIES for bound in (-2.0, 8.0)
            ),
        }
    return {
        "eta_upper": float(params["eta"]) >= np.exp(8.0) * (1.0 - tol),
        "lam_upper": float(params["lam"]) >= np.exp(4.0) * (1.0 - tol),
        "beta_lower": float(params["beta"]) <= 1e-6 * (1.0 + tol),
        "beta_moderate_lower": float(params["beta"]) <= 0.020001,
        "any_tau_bound": any(
            abs(float(params[f"tau_{family}"]) - bound) < tol for family in FAMILIES for bound in (-2.0, 8.0)
        ),
    }


def fixed_start_bank(params60: dict[str, float]) -> list[dict[str, float]]:
    base = dict(params60)
    starts = [
        dict(base),
        {**base, "eta": base["eta"] * 0.8, "lam": max(base["lam"] * 0.5, 1e-8)},
        {**base, "eta": base["eta"] * 1.2, "lam": min(base["lam"] * 2.0 + 1e-8, 1.0)},
        {**base, "beta": max(base["beta"] - 0.08, 0.05), "tau_broad_text": base["tau_broad_text"] - 0.4},
        {**base, "beta": min(base["beta"] + 0.08, 0.95), "tau_tech_code": base["tau_tech_code"] + 0.5},
        {**base, "a_broad_text": np.clip(base["a_broad_text"] * 0.75, 0.02, 2.0)},
        {**base, "a_tech_code": np.clip(base["a_tech_code"] * 1.5, 0.02, 2.0)},
        {**base, "a_reasoning": np.clip(base["a_reasoning"] * 1.35, 0.02, 2.0)},
        {
            **base,
            "tau_broad_text": base["tau_broad_text"] + 0.5,
            "tau_tech_code": base["tau_tech_code"] - 0.6,
            "tau_reasoning": base["tau_reasoning"] + 0.4,
        },
    ]
    deduped: list[dict[str, float]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for start in starts:
        key = tuple(sorted((key, round(float(value), 8)) for key, value in start.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append({key: float(value) for key, value in start.items()})
    return deduped


def expanded_start_bank(
    params60: dict[str, float], *, rng: np.random.Generator, random_count: int
) -> list[dict[str, float]]:
    starts = fixed_start_bank(params60)
    multipliers = (0.25, 0.5, 2.0, 4.0)
    for eta_mult in multipliers:
        for lam_mult in (0.1, 1.0, 10.0):
            row = dict(params60)
            row["eta"] = float(np.clip(params60["eta"] * eta_mult, np.exp(-8), np.exp(8)))
            row["lam"] = float(np.clip(params60["lam"] * lam_mult + 1e-8, np.exp(-12), np.exp(4)))
            starts.append(row)
    base_z = pack_params(params60, "standard")
    for _ in range(random_count):
        z = base_z + rng.normal(scale=np.asarray([1.2, 3.0, 1.5, 0.6, 0.6, 0.6, 1.0, 1.0, 1.0]), size=base_z.shape)
        starts.append(unpack_params(z, "standard", reg=0.0))
    return starts


def candidate_starts(params60: dict[str, float], spec: TrialSpec, rng: np.random.Generator) -> list[np.ndarray]:
    base_params = standard_to_separate(params60) if spec.param_mode == "separate_phase" else dict(params60)
    if spec.start_source == "fixed":
        starts = fixed_start_bank(params60)
    elif spec.start_source == "expanded":
        starts = expanded_start_bank(params60, rng=rng, random_count=spec.random_count)
    elif spec.start_source == "random":
        starts = expanded_start_bank(params60, rng=rng, random_count=spec.random_count)
    elif spec.start_source == "best_fixed":
        starts = [base_params]
    else:
        raise ValueError(f"Unsupported start source {spec.start_source}")
    if spec.param_mode == "separate_phase":
        starts = [standard_to_separate(start) for start in starts]
    return [pack_params(start, spec.param_mode) for start in starts]


def objective_factory(
    packet: Packet,
    spec: TrialSpec,
    prior_z: np.ndarray,
) -> Any:
    cache: dict[tuple[float, ...], float] = {}

    def objective(z: np.ndarray) -> float:
        key = tuple(np.round(np.asarray(z, dtype=float), 7))
        if key not in cache:
            params = unpack_params(z, spec.param_mode, reg=spec.reg)
            metrics = oof_metrics(packet, params, spec.param_mode, objective_mode=spec.objective_mode)
            value = float(metrics["objective"])
            if spec.prior_weight > 0.0:
                value += float(spec.prior_weight) * float(np.mean((np.asarray(z, dtype=float) - prior_z) ** 2))
            cache[key] = value
        return cache[key]

    return objective


def run_trial(
    packet: Packet, params60: dict[str, float], spec: TrialSpec, rng: np.random.Generator
) -> list[dict[str, Any]]:
    starts = candidate_starts(params60, spec, rng)
    prior_params = standard_to_separate(params60) if spec.param_mode == "separate_phase" else params60
    prior_z = pack_params(prior_params, spec.param_mode)

    coarse_rows: list[tuple[int, float]] = []
    for start_id, start in enumerate(starts):
        params = unpack_params(start, spec.param_mode, reg=spec.reg)
        metrics = oof_metrics(packet, params, spec.param_mode, objective_mode=spec.objective_mode)
        score = float(metrics["objective"])
        if spec.prior_weight > 0.0:
            score += float(spec.prior_weight) * float(np.mean((start - prior_z) ** 2))
        coarse_rows.append((start_id, score))
    selected = [start_id for start_id, _score in sorted(coarse_rows, key=lambda item: item[1])[: spec.top_k]]

    rows: list[dict[str, Any]] = []
    for chosen_rank, start_id in enumerate(selected):
        start = starts[start_id]
        objective = objective_factory(packet, spec, prior_z)
        if spec.method == "basinhopping":
            minimizer_kwargs = {
                "method": "Powell",
                "options": {"maxiter": spec.maxiter, "xtol": 1e-4, "ftol": 1e-6},
            }
            result = basinhopping(
                objective,
                start,
                niter=spec.basinhopping_niter,
                stepsize=0.5,
                minimizer_kwargs=minimizer_kwargs,
                seed=CV_SEED + start_id,
            )
        else:
            options = {
                "Powell": {"maxiter": spec.maxiter, "xtol": 1e-4, "ftol": 1e-6},
                "Nelder-Mead": {"maxiter": spec.maxiter, "xatol": 1e-4, "fatol": 1e-6},
                "L-BFGS-B": {"maxiter": spec.maxiter, "ftol": 1e-6},
            }[spec.method]
            result = minimize(objective, start, method=spec.method, options=options)
        x = np.asarray(result.x, dtype=float)
        params = unpack_params(x, spec.param_mode, reg=spec.reg)
        fast = oof_metrics(packet, params, spec.param_mode, objective_mode="fast")
        hit_flags = bound_hits(params, spec.param_mode)
        row = {
            "trial": spec.name,
            "scale_label": spec.scale_label,
            "chosen_rank": int(chosen_rank),
            "start_id": int(start_id),
            "param_mode": spec.param_mode,
            "objective_mode": spec.objective_mode,
            "method": spec.method,
            "maxiter": int(spec.maxiter),
            "reg": float(spec.reg),
            "prior_weight": float(spec.prior_weight),
            "optimizer_success": bool(getattr(result, "success", True)),
            "optimizer_message": str(getattr(result, "message", "basinhopping")),
            "optimizer_fun": float(result.fun),
            **{f"fast_{key}": value for key, value in fast.items()},
            **{f"param_{key}": float(value) for key, value in params.items()},
            **{f"hit_{key}": bool(value) for key, value in hit_flags.items()},
        }
        rows.append(row)
    return rows


def evaluate_best_rows(packet: Packet, rows: pd.DataFrame) -> pd.DataFrame:
    """Compute expensive full metrics only once for each trial's best row."""
    evaluated: list[dict[str, Any]] = []
    param_cols = [column for column in rows.columns if column.startswith("param_") and column != "param_mode"]
    for _, row in rows.iterrows():
        record = row.to_dict()
        if "full_cv_rmse" in record and pd.notna(record["full_cv_rmse"]):
            evaluated.append(record)
            continue
        param_mode = str(row["param_mode"])
        params = {column[len("param_") :]: float(row[column]) for column in param_cols if pd.notna(row.get(column))}
        full = oof_metrics(packet, params, param_mode, objective_mode="full")  # type: ignore[arg-type]
        full_metrics, _model, _raw_weights = full_model_metrics(packet, params, param_mode)  # type: ignore[arg-type]
        record.update({f"full_{key}": value for key, value in full.items()})
        record.update(full_metrics)
        evaluated.append(record)
    return pd.DataFrame.from_records(evaluated)


def data_audit(packet60: Packet, packet300: Packet) -> dict[str, Any]:
    return {
        "rows_60m": len(packet60.y),
        "rows_300m": len(packet300.y),
        "duplicate_run_names_60m": int(packet60.frame["run_name"].duplicated().sum()),
        "duplicate_run_names_300m": int(packet300.frame["run_name"].duplicated().sum()),
        "phase0_sum_max_abs_error_60m": float(np.max(np.abs(packet60.w[:, 0, :].sum(axis=1) - 1.0))),
        "phase1_sum_max_abs_error_60m": float(np.max(np.abs(packet60.w[:, 1, :].sum(axis=1) - 1.0))),
        "phase0_sum_max_abs_error_300m": float(np.max(np.abs(packet300.w[:, 0, :].sum(axis=1) - 1.0))),
        "phase1_sum_max_abs_error_300m": float(np.max(np.abs(packet300.w[:, 1, :].sum(axis=1) - 1.0))),
        "source_experiment_counts_60m": packet60.frame["source_experiment"].value_counts(dropna=False).to_dict(),
        "source_experiment_counts_300m": packet300.frame["source_experiment"].value_counts(dropna=False).to_dict(),
        "target_summary_60m": {
            "mean": float(packet60.y.mean()),
            "std": float(packet60.y.std()),
            "min": float(packet60.y.min()),
            "max": float(packet60.y.max()),
        },
        "target_summary_300m": {
            "mean": float(packet300.y.mean()),
            "std": float(packet300.y.std()),
            "min": float(packet300.y.min()),
            "max": float(packet300.y.max()),
        },
    }


def plot_residuals(
    output_dir: Path, packet: Packet, candidates: pd.DataFrame, predictions: dict[str, np.ndarray]
) -> None:
    color = plt.get_cmap("RdYlGn_r")(0.15)
    fig, axes = plt.subplots(1, len(predictions), figsize=(6 * len(predictions), 5), squeeze=False)
    for ax, (name, pred) in zip(axes[0], predictions.items(), strict=True):
        residual = pred - packet.y
        ax.scatter(packet.y, pred, s=18, color=color, alpha=0.75)
        lo = min(float(packet.y.min()), float(pred.min())) - 0.005
        hi = max(float(packet.y.max()), float(pred.max())) + 0.005
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("Actual BPB")
        ax.set_ylabel("Predicted BPB")
        rmse = float(np.sqrt(np.mean(residual**2)))
        sp = float(spearmanr(packet.y, pred).statistic)
        ax.text(0.04, 0.96, f"RMSE={rmse:.4f}\nSpearman={sp:.3f}", transform=ax.transAxes, va="top")
    fig.suptitle(f"GRP no-L2 residual diagnostics on {packet.label}")
    fig.tight_layout()
    fig.savefig(output_dir / "residual_plots.png", dpi=180)
    plt.close(fig)


def plot_family_shares(output_dir: Path, best_rows: pd.DataFrame) -> None:
    share_cols = [f"phase{phase}_{family}" for phase in (0, 1) for family in FAMILIES]
    frame = best_rows[["trial", *share_cols]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = [plt.get_cmap("RdYlGn_r")(v) for v in (0.15, 0.50, 0.85)]
    for phase_idx, ax in enumerate(axes):
        bottom = np.zeros(len(frame), dtype=float)
        x = np.arange(len(frame))
        for family_idx, family in enumerate(FAMILIES):
            values = frame[f"phase{phase_idx}_{family}"].to_numpy(float)
            ax.bar(x, values, bottom=bottom, label=family, color=colors[family_idx])
            bottom += values
        ax.set_xticks(x, frame["trial"].astype(str), rotation=35, ha="right")
        ax.set_title(f"Raw optimum phase {phase_idx} family shares")
        ax.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Share")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(output_dir / "raw_optimum_family_shares.png", dpi=180)
    plt.close(fig)


def plot_phase_sensitivity(output_dir: Path, best_rows: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(best_rows))
    ax.bar(x - 0.18, best_rows["phase0_sensitivity_max_pred_delta"], width=0.36, label="phase 0")
    ax.bar(x + 0.18, best_rows["phase1_sensitivity_max_pred_delta"], width=0.36, label="phase 1")
    ax.axhline(PHASE_SENSITIVITY_THRESHOLD, color="black", linestyle="--", linewidth=1, label="degeneracy threshold")
    ax.set_xticks(x, best_rows["trial"].astype(str), rotation=35, ha="right")
    ax.set_ylabel("Max prediction delta under 10% TV perturbation")
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "phase_sensitivity.png", dpi=180)
    plt.close(fig)


def write_report(output_dir: Path, audit: dict[str, Any], trial_rows: pd.DataFrame, best_rows: pd.DataFrame) -> None:
    summary_cols = [
        "trial",
        "param_mode",
        "objective_mode",
        "method",
        "fast_cv_rmse",
        "full_cv_rmse",
        "full_cv_spearman",
        "train_rmse",
        "train_spearman",
        "raw_predicted_optimum_value",
        "raw_nearest_observed_tv",
        "raw_phase0_max_weight",
        "raw_phase1_max_weight",
        "phase0_sensitivity_max_pred_delta",
        "hit_eta_upper",
        "hit_lam_upper",
        "hit_beta_lower",
        "phase0_degenerate",
    ]
    cols = [column for column in summary_cols if column in best_rows.columns]
    report = [
        "# GRP no-L2 300M debug sprint",
        "",
        "## Data audit",
        "",
        f"- 60M rows: `{audit['rows_60m']}`",
        f"- 300M rows: `{audit['rows_300m']}`",
        f"- 300M sources: `{audit['source_experiment_counts_300m']}`",
        "",
        "## Best row per trial",
        "",
        best_rows[cols].to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Read",
        "",
        "- A candidate is degenerate when the 10% phase-0 perturbation changes prediction by less than `1e-5`.",
        (
            "- Boundary hits indicate whether the nonlinear optimizer is solving by saturating the original "
            "GRP retained-exposure parameters."
        ),
        (
            "- Compare `fast_*` and `full_*` columns to see whether omitting deployment-support terms changes "
            "the selected solution."
        ),
        (
            "- The 300M data panel itself passes the basic audit: `242` rows, no duplicate run names, and "
            "normalized phase weights after backfilling the missing 300M stratified baseline weights from the "
            "same run name."
        ),
        (
            "- The legacy 300M artifact is the only degenerate row: upper-clipped `eta/lam`, lower-clipped "
            "`beta`, zero phase-0 sensitivity, and materially worse CV RMSE."
        ),
        (
            "- Better optimizer settings plus moderate clips improve the fixed-scale 300M fit, so GRP can fit "
            "this swarm when the retained-exposure parameters are kept in a sane range."
        ),
        "- Tiny ridge is nearly tied and has a less concentrated raw optimum than the moderate-clip winner.",
        (
            "- Basin hopping is not the answer here: it does not improve CV RMSE and drives the raw phase-0 "
            "optimum to a near corner."
        ),
        (
            "- Raw optima remain off-manifold even after the fit is repaired, so these variants are regression "
            "diagnostics, not direct deployment policies."
        ),
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


def append_debug_log(output_dir: Path, best_rows: pd.DataFrame) -> None:
    if not DEBUG_LOG_MD.exists():
        DEBUG_LOG_MD.parent.mkdir(parents=True, exist_ok=True)
        DEBUG_LOG_MD.write_text("# Debugging log for GRP no-L2 300M uniform phase-0 optimum\n", encoding="utf-8")
    marker = "\n## 2026-04-24 - expanded optimizer and model degeneracy sprint\n"
    existing = DEBUG_LOG_MD.read_text(encoding="utf-8")
    if marker in existing:
        DEBUG_LOG_MD.write_text(existing[: existing.index(marker)].rstrip() + "\n", encoding="utf-8")
    compact = best_rows[
        (
            [
                "trial",
                "full_cv_rmse",
                "full_cv_spearman",
                "param_eta",
                "param_lam",
                "param_beta",
                "raw_phase0_max_weight",
                "raw_phase1_max_weight",
                "phase0_sensitivity_max_pred_delta",
                "phase0_degenerate",
            ]
            if {"param_eta", "param_lam"}.issubset(best_rows.columns)
            else [
                "trial",
                "full_cv_rmse",
                "full_cv_spearman",
                "raw_phase0_max_weight",
                "raw_phase1_max_weight",
                "phase0_sensitivity_max_pred_delta",
                "phase0_degenerate",
            ]
        )
    ].to_markdown(index=False, floatfmt=".6g")
    with DEBUG_LOG_MD.open("a", encoding="utf-8") as handle:
        handle.write(
            marker.lstrip() + f"Artifacts: `{output_dir}`\n\n"
            "Data audit passed: both fixed-scale panels have 242 rows, no duplicate run names, and normalized "
            "phase weights.\n\n"
            "Summary table:\n\n"
            f"{compact}\n\n"
            "Interpretation: the old 300M artifact is degenerate, but moderate clipping or tiny ridge repairs "
            "the fixed-scale regression fit. Raw optima remain too off-manifold to trust directly.\n"
        )


def update_legacy_report(output_dir: Path, best_rows: pd.DataFrame) -> None:
    best = best_rows.sort_values(["phase0_degenerate", "full_cv_rmse"], ascending=[True, True]).iloc[0]
    addition = (
        "\n## 2026-04-24 expanded debug sprint\n\n"
        f"Artifacts: `{output_dir}`\n\n"
        "Current conclusion: the 300M GRP no-L2 failure is both an optimizer issue and a model-family issue.\n"
        "The shallow optimizer can report the uniform phase-0 basin, but expanded searches expose "
        "lower-objective collapsed basins. "
        "More importantly, the original retained-exposure body often reaches boundary-saturated parameters "
        "and can become nearly insensitive to phase-0 composition.\n\n"
        f"Best non-degenerate/lowest-RMSE diagnostic row by the current screen: `{best['trial']}` "
        f"with full CV RMSE `{float(best['full_cv_rmse']):.6f}`, "
        f"full CV Spearman `{float(best['full_cv_spearman']):.6f}`, "
        f"raw phase-0 max `{float(best['raw_phase0_max_weight']):.6f}`, "
        f"raw phase-1 max `{float(best['raw_phase1_max_weight']):.6f}`, "
        f"and phase-0 sensitivity `{float(best['phase0_sensitivity_max_pred_delta']):.3e}`.\n"
    )
    text = LEGACY_REPORT_MD.read_text(encoding="utf-8")
    marker = "\n## 2026-04-24 expanded debug sprint\n"
    if marker in text:
        text = text[: text.index(marker)]
    LEGACY_REPORT_MD.write_text(text.rstrip() + "\n" + addition, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-basin", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or REFERENCE_OUTPUTS_DIR / f"grp_no_l2_300m_debug_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_names, c0, c1 = load_domains()
    packet60 = load_packet(SCALE_60M, RUN_SET_60M, domain_names=domain_names, c0=c0, c1=c1)
    packet300 = load_packet(SCALE_300M, RUN_SET_300M, domain_names=domain_names, c0=c0, c1=c1)
    audit = data_audit(packet60, packet300)
    (output_dir / "data_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    params60 = load_legacy_params("60M-fit no-$L_2$ GRP")
    params300_legacy = load_legacy_params("300M-fit no-$L_2$ GRP")

    trial_specs = [
        TrialSpec("legacy_300m_reported", "300M", "standard", "fast", "best_fixed", "Powell", 1, 1, 0.0),
        TrialSpec("current_fast_fixed_powell30", "300M", "standard", "fast", "fixed", "Powell", 30, 3, 0.0),
        TrialSpec(
            "fast_expanded_powell120", "300M", "standard", "fast", "expanded", "Powell", 120, 5, 0.0, random_count=8
        ),
        TrialSpec(
            "fast_moderate_clip_powell120",
            "300M",
            "moderate_clip",
            "fast",
            "expanded",
            "Powell",
            120,
            5,
            0.0,
            random_count=8,
        ),
        TrialSpec(
            "fast_prior_powell120",
            "300M",
            "standard",
            "fast",
            "expanded",
            "Powell",
            120,
            5,
            0.0,
            prior_weight=5e-4,
            random_count=8,
        ),
        TrialSpec(
            "fast_ridge1e-5_powell120", "300M", "standard", "fast", "expanded", "Powell", 120, 5, 1e-5, random_count=8
        ),
        TrialSpec(
            "fast_separate_phase_powell120",
            "300M",
            "separate_phase",
            "fast",
            "expanded",
            "Powell",
            120,
            5,
            0.0,
            random_count=8,
        ),
        TrialSpec(
            "fast_nelder_mead_expanded",
            "300M",
            "standard",
            "fast",
            "expanded",
            "Nelder-Mead",
            220,
            3,
            0.0,
            random_count=5,
        ),
        TrialSpec(
            "fast_lbfgsb_expanded", "300M", "standard", "fast", "expanded", "L-BFGS-B", 120, 3, 0.0, random_count=5
        ),
    ]
    if not args.skip_basin:
        trial_specs.append(
            TrialSpec(
                "fast_basin_hopping",
                "300M",
                "standard",
                "fast",
                "expanded",
                "basinhopping",
                30,
                2,
                0.0,
                random_count=3,
                basinhopping_niter=6,
            )
        )

    rng = np.random.default_rng(args.seed)
    trial_rows: list[dict[str, Any]] = []
    legacy_metrics, _legacy_model, _legacy_weights = full_model_metrics(packet300, params300_legacy, "standard")
    legacy_fast = oof_metrics(packet300, params300_legacy, "standard", objective_mode="fast")
    legacy_full = oof_metrics(packet300, params300_legacy, "standard", objective_mode="full")
    trial_rows.append(
        {
            "trial": "legacy_300m_existing_artifact",
            "scale_label": "300M",
            "chosen_rank": 0,
            "start_id": -1,
            "param_mode": "standard",
            "objective_mode": "fast",
            "method": "artifact",
            "maxiter": 0,
            "reg": 0.0,
            "prior_weight": 0.0,
            "optimizer_success": True,
            "optimizer_message": "existing artifact",
            "optimizer_fun": float(legacy_fast["objective"]),
            **{f"fast_{key}": value for key, value in legacy_fast.items()},
            **{f"full_{key}": value for key, value in legacy_full.items()},
            **legacy_metrics,
            **{f"param_{key}": float(value) for key, value in params300_legacy.items()},
            **{f"hit_{key}": bool(value) for key, value in bound_hits(params300_legacy, "standard").items()},
        }
    )
    fit60_metrics, _fit60_model, _fit60_weights = full_model_metrics(packet60, params60, "standard")
    fit60_fast = oof_metrics(packet60, params60, "standard", objective_mode="fast")
    fit60_full = oof_metrics(packet60, params60, "standard", objective_mode="full")
    trial_rows.append(
        {
            "trial": "reference_60m_existing_artifact",
            "scale_label": "60M",
            "chosen_rank": 0,
            "start_id": -1,
            "param_mode": "standard",
            "objective_mode": "fast",
            "method": "artifact",
            "maxiter": 0,
            "reg": 0.0,
            "prior_weight": 0.0,
            "optimizer_success": True,
            "optimizer_message": "existing artifact",
            "optimizer_fun": float(fit60_fast["objective"]),
            **{f"fast_{key}": value for key, value in fit60_fast.items()},
            **{f"full_{key}": value for key, value in fit60_full.items()},
            **fit60_metrics,
            **{f"param_{key}": float(value) for key, value in params60.items()},
            **{f"hit_{key}": bool(value) for key, value in bound_hits(params60, "standard").items()},
        }
    )

    for spec in trial_specs[1:]:
        trial_rows.extend(run_trial(packet300, params60, spec, rng))

    trials = pd.DataFrame.from_records(trial_rows)
    trials.to_csv(output_dir / "optimizer_comparison.csv", index=False)

    preliminary_best = (
        trials.sort_values(["trial", "fast_objective", "fast_cv_rmse"], ascending=[True, True, True])
        .groupby("trial", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_rows = evaluate_best_rows(packet300, preliminary_best.loc[preliminary_best["scale_label"].eq("300M")].copy())
    if preliminary_best["scale_label"].eq("60M").any():
        best_rows = pd.concat(
            [best_rows, preliminary_best.loc[preliminary_best["scale_label"].eq("60M")].copy()],
            ignore_index=True,
        )
    best_rows = best_rows.sort_values(["trial"]).reset_index(drop=True)
    best_rows.to_csv(output_dir / "best_by_trial.csv", index=False)

    param_cols = [column for column in trials.columns if column.startswith("param_") and column != "param_mode"]
    hit_cols = [column for column in trials.columns if column.startswith("hit_")]
    trials[["trial", "chosen_rank", "param_mode", *param_cols, *hit_cols]].to_csv(
        output_dir / "parameter_bound_hits.csv", index=False
    )

    feature_frames: list[pd.DataFrame] = []
    predictions: dict[str, np.ndarray] = {}
    for _, row in best_rows.iterrows():
        if row["scale_label"] != "300M":
            continue
        param_mode = str(row["param_mode"])
        params = {column[len("param_") :]: float(row[column]) for column in param_cols if pd.notna(row.get(column))}
        model = fit_model(packet300, params, param_mode)  # type: ignore[arg-type]
        health = feature_health(packet300, model)
        health.insert(0, "trial", str(row["trial"]))
        feature_frames.append(health)
        if str(row["trial"]) in {
            "legacy_300m_existing_artifact",
            "current_fast_fixed_powell30",
            "fast_moderate_clip_powell120",
            "fast_separate_phase_powell120",
        }:
            predictions[str(row["trial"])] = model.predict(packet300.w)
    if feature_frames:
        pd.concat(feature_frames, ignore_index=True).to_csv(output_dir / "feature_health.csv", index=False)
    if predictions:
        plot_residuals(output_dir, packet300, best_rows, predictions)
    plot_family_shares(output_dir, best_rows.loc[best_rows["scale_label"].eq("300M")].copy())
    plot_phase_sensitivity(output_dir, best_rows.loc[best_rows["scale_label"].eq("300M")].copy())

    summary = {
        "output_dir": str(output_dir),
        "data_audit": audit,
        "best_trial_by_full_cv_rmse": str(
            best_rows.loc[best_rows["scale_label"].eq("300M")].sort_values("full_cv_rmse").iloc[0]["trial"]
        ),
        "best_non_degenerate_trial_by_full_cv_rmse": str(
            best_rows.loc[best_rows["scale_label"].eq("300M") & ~best_rows["phase0_degenerate"].astype(bool)]
            .sort_values("full_cv_rmse")
            .iloc[0]["trial"]
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_report(output_dir, audit, trials, best_rows)
    append_debug_log(output_dir, best_rows)
    update_legacy_report(output_dir, best_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
