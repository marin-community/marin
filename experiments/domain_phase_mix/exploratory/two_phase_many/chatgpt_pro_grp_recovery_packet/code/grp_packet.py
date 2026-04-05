# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Self-contained GRP packet utilities for ChatGPT Pro handoff."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

GENERIC_FAMILY_NAMES = ("broad_text", "tech_code", "reasoning")
CURRENT_TUNED_GENERIC_PARAMS = {
    "alpha": 12.94088092035213,
    "eta": 13.229384772843037,
    "lam": 0.035627177458741076,
    "tau": 3.2740751832677875,
    "reg": 0.0010114720923828182,
    "beta": 0.6634021668256815,
}
CURRENT_BROAD_BETA_START_PARAMS = {
    "alpha": 11.533461482593735,
    "eta": 10.859113730214359,
    "lam": 0.3422735488822989,
    "tau": 2.843180828656475,
    "reg": 0.0001896587113845684,
    "beta": 0.9324427249160729,
}
CV_WEIGHT = 1.0
ANCHOR_WEIGHT = 1.0
REGRET_WEIGHT = 0.02


@dataclass(frozen=True)
class PacketData:
    """Base packet arrays."""

    name_col: str
    objective_metric: str
    run_names: list[str]
    domain_names: list[str]
    y: np.ndarray
    w: np.ndarray
    c0: np.ndarray
    c1: np.ndarray

    @property
    def m(self) -> int:
        return int(self.w.shape[2])

    @property
    def n(self) -> int:
        return int(self.w.shape[0])


@dataclass(frozen=True)
class GenericFamilyPacket:
    """Packet plus CC-pair and family metadata."""

    base: PacketData
    pairs: list[tuple[int, int]]
    pair_topics: list[str]
    singletons: list[int]
    family_map: dict[str, list[int]]


@dataclass(frozen=True)
class AnchorState:
    """Current reference anchors and deployed GRP state."""

    validated_global_bpb: float
    validated_pair_bpb: float
    validated_global_weights: np.ndarray
    validated_pair_weights: np.ndarray
    best_observed_weights: np.ndarray
    best_observed_run_name: str
    best_observed_bpb: float
    proportional_weights: np.ndarray
    proportional_run_name: str
    proportional_bpb: float
    deployed_grp: dict[str, Any]
    current_tuned_params: dict[str, float]
    current_broad_beta_start_params: dict[str, float]


def packet_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return packet_root() / "data"


def reference_dir() -> Path:
    return packet_root() / "reference_outputs"


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


def mean_phase_tv_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return 0.5 * float(np.mean(np.sum(np.abs(lhs - rhs), axis=1)))


def average_phase_tv_distance(batch: np.ndarray, target: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(batch - target).sum(axis=2).mean(axis=1)


def load_packet(root: Path | None = None) -> GenericFamilyPacket:
    root = packet_root() if root is None else root
    with np.load(root / "data" / "many_domain_packet.npz", allow_pickle=False) as payload:
        base = PacketData(
            name_col=str(payload["name_col"][0]),
            objective_metric=str(payload["objective_metric"][0]),
            run_names=[str(x) for x in payload["run_names"].tolist()],
            domain_names=[str(x) for x in payload["domain_names"].tolist()],
            y=np.asarray(payload["y"], dtype=float),
            w=np.asarray(payload["w"], dtype=float),
            c0=np.asarray(payload["c0"], dtype=float),
            c1=np.asarray(payload["c1"], dtype=float),
        )

    pairs: list[tuple[int, int]] = []
    pair_topics: list[str] = []
    paired: set[int] = set()
    for idx, domain_name in enumerate(base.domain_names):
        if idx in paired:
            continue
        if domain_name.startswith("dolma3_cc/") and domain_name.endswith("_high"):
            low_name = domain_name[:-5] + "_low"
            if low_name in base.domain_names:
                low_idx = base.domain_names.index(low_name)
                pairs.append((idx, low_idx))
                pair_topics.append(domain_name[len("dolma3_cc/") : -5])
                paired.add(idx)
                paired.add(low_idx)

    singletons = [idx for idx in range(base.m) if idx not in paired]
    family_map = {family_name: [] for family_name in GENERIC_FAMILY_NAMES}
    for idx, domain_name in enumerate(base.domain_names):
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
        is_tech = any(token in domain_name for token in ("stack_edu", "synth_code", "synth_math")) or (
            domain_name in {"dolma3_arxiv", "dolma3_finemath_3plus"}
        )
        is_reasoning = domain_name in {"dolmino_synth_instruction", "dolmino_synth_thinking"}

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


def load_subset_indices(root: Path | None = None) -> dict[int, list[int]]:
    root = packet_root() if root is None else root
    payload = json.loads((root / "data" / "subset_indices_feature_bayes_linear.json").read_text())
    return {int(k): [int(v) for v in values] for k, values in payload["subset_indices"].items()}


def _phase_weights_to_array(phase_weights: dict[str, dict[str, float]], domain_names: list[str]) -> np.ndarray:
    phase0 = np.asarray([float(phase_weights["phase_0"][name]) for name in domain_names], dtype=float)
    phase1 = np.asarray([float(phase_weights["phase_1"][name]) for name in domain_names], dtype=float)
    return np.stack([phase0, phase1], axis=0)


def load_reference_state(root: Path | None = None) -> AnchorState:
    root = packet_root() if root is None else root
    payload = json.loads((root / "data" / "current_reference_state.json").read_text())
    return AnchorState(
        validated_global_bpb=float(payload["validated_global_bpb"]),
        validated_pair_bpb=float(payload["validated_pair_bpb"]),
        validated_global_weights=np.asarray(payload["validated_global"]["phase_weights"], dtype=float),
        validated_pair_weights=np.asarray(payload["validated_pair"]["phase_weights"], dtype=float),
        best_observed_weights=np.asarray(payload["best_observed"]["phase_weights"], dtype=float),
        best_observed_run_name=str(payload["best_observed"]["run_name"]),
        best_observed_bpb=float(payload["best_observed"]["actual_bpb"]),
        proportional_weights=np.asarray(payload["baseline_proportional"]["phase_weights"], dtype=float),
        proportional_run_name=str(payload["baseline_proportional"]["run_name"]),
        proportional_bpb=float(payload["baseline_proportional"]["actual_bpb"]),
        deployed_grp=payload["current_deployed_grp"],
        current_tuned_params={k: float(v) for k, v in payload["current_tuned_generic_params"].items()},
        current_broad_beta_start_params={k: float(v) for k, v in payload["current_broad_beta_start_params"].items()},
    )


def subset_packet(packet: GenericFamilyPacket, indices: np.ndarray) -> GenericFamilyPacket:
    indices = np.asarray(indices, dtype=int)
    return GenericFamilyPacket(
        base=replace(
            packet.base,
            run_names=[packet.base.run_names[idx] for idx in indices.tolist()],
            y=packet.base.y[indices],
            w=packet.base.w[indices],
        ),
        pairs=packet.pairs,
        pair_topics=packet.pair_topics,
        singletons=packet.singletons,
        family_map=packet.family_map,
    )


def family_shares(packet: GenericFamilyPacket, weights: np.ndarray) -> dict[str, float]:
    shares: dict[str, float] = {}
    for phase_idx in (0, 1):
        for family_name in GENERIC_FAMILY_NAMES:
            shares[f"phase{phase_idx}_{family_name}"] = float(weights[phase_idx, packet.family_map[family_name]].sum())
    return shares


class GenericFamilyRetainedTotalSurrogate:
    """Generic-family retained-total surrogate."""

    def __init__(
        self,
        packet: GenericFamilyPacket,
        *,
        params: dict[str, float] | None = None,
        family_totals: tuple[str, ...] = GENERIC_FAMILY_NAMES,
        quality_discount: bool = True,
        pair_cc_domains: bool = True,
    ):
        self.packet = packet
        self.params = dict(CURRENT_TUNED_GENERIC_PARAMS if params is None else params)
        self.family_totals = tuple(family_totals)
        self.quality_discount = bool(quality_discount)
        self.pair_cc_domains = bool(pair_cc_domains)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def _retained_x(self, weights: np.ndarray) -> np.ndarray:
        p0 = weights[:, 0, :]
        p1 = weights[:, 1, :]
        e0 = p0 * self.packet.base.c0[None, :]
        e1 = p1 * self.packet.base.c1[None, :]
        lam = float(self.params["lam"])
        eta = float(self.params["eta"])
        return np.exp(-lam * (1.0 - p1)) * e0 + eta * e1

    def build_design(self, weights: np.ndarray) -> np.ndarray:
        alpha = float(self.params["alpha"])
        tau = float(self.params["tau"])
        beta = float(self.params["beta"])
        x = self._retained_x(weights)

        features: list[np.ndarray] = []
        group_totals: list[np.ndarray] = []

        singleton_indices = self.packet.singletons if self.pair_cc_domains else list(range(self.packet.base.m))
        pair_map = self.packet.pairs if self.pair_cc_domains else []

        for idx in singleton_indices:
            features.append(np.log1p(alpha * x[:, idx : idx + 1]))
            group_totals.append(x[:, idx])

        for hi, lo in pair_map:
            pair_signal_total = x[:, hi] + (beta * x[:, lo] if self.quality_discount else x[:, lo])
            features.append(np.log1p(alpha * pair_signal_total)[:, None])
            group_totals.append(x[:, hi] + x[:, lo])

        for family_name in self.family_totals:
            family_indices = self.packet.family_map[family_name]
            family_total = np.sum(x[:, family_indices], axis=1)
            features.append(np.log1p(alpha * family_total)[:, None])

        penalty_inputs = np.stack(group_totals, axis=1)
        penalty = np.sum(softplus(np.log1p(penalty_inputs) - tau) ** 2, axis=1, keepdims=True)
        features.append(penalty)

        design = np.hstack(features)
        num_signal = design.shape[1] - 1
        design[:, :num_signal] *= -1.0
        return design

    def fit(self, weights: np.ndarray, targets: np.ndarray) -> GenericFamilyRetainedTotalSurrogate:
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
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model must be fit before prediction")
        design = self.build_design(weights)
        return np.asarray(self.intercept_ + design @ self.coef_, dtype=float)


def fitted_generic_family_components(
    packet: GenericFamilyPacket,
    model: GenericFamilyRetainedTotalSurrogate,
) -> dict[str, Any]:
    if model.coef_ is None:
        raise RuntimeError("Model must be fit")

    n_singletons = len(packet.singletons) if model.pair_cc_domains else packet.base.m
    n_pairs = len(packet.pairs) if model.pair_cc_domains else 0
    n_families = len(model.family_totals)

    offset = 0
    singleton_coef = np.asarray(model.coef_[offset : offset + n_singletons], dtype=float)
    offset += n_singletons
    pair_coef = np.asarray(model.coef_[offset : offset + n_pairs], dtype=float)
    offset += n_pairs
    family_coef = {
        family_name: float(coef)
        for family_name, coef in zip(
            model.family_totals,
            model.coef_[offset : offset + n_families],
            strict=True,
        )
    }
    offset += n_families
    penalty_coef = float(model.coef_[offset])
    return {
        "singleton_coef": singleton_coef,
        "pair_coef": pair_coef,
        "family_coef": family_coef,
        "penalty_coef": penalty_coef,
    }


def optimize_generic_family_model(
    packet: GenericFamilyPacket,
    model: GenericFamilyRetainedTotalSurrogate,
    *,
    n_random: int = 20,
    seed: int = 0,
) -> tuple[Any, np.ndarray, np.ndarray]:
    if model.coef_ is None or model.intercept_ is None:
        raise RuntimeError("Model must be fit")

    parts = fitted_generic_family_components(packet, model)
    singleton_coef = parts["singleton_coef"]
    pair_coef = parts["pair_coef"]
    family_coef = parts["family_coef"]
    penalty_coef = float(parts["penalty_coef"])

    n_domains = packet.base.m
    c0 = packet.base.c0
    c1 = packet.base.c1
    alpha = float(model.params["alpha"])
    eta = float(model.params["eta"])
    lam = float(model.params["lam"])
    tau = float(model.params["tau"])
    beta = float(model.params["beta"]) if model.quality_discount else 1.0
    rng = np.random.default_rng(seed)

    pair_map = packet.pairs if model.pair_cc_domains else []
    singleton_indices = packet.singletons if model.pair_cc_domains else list(range(packet.base.m))
    family_indices = {
        family_name: np.asarray(packet.family_map[family_name], dtype=int) for family_name in model.family_totals
    }

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

        for local_idx, domain_idx in enumerate(singleton_indices):
            coef = float(singleton_coef[local_idx])
            value -= coef * np.log1p(alpha * x[domain_idx])
            grad_x[domain_idx] -= coef * alpha / (1.0 + alpha * x[domain_idx])

        for local_idx, (hi, lo) in enumerate(pair_map):
            coef = float(pair_coef[local_idx])
            total = x[hi] + beta * x[lo]
            value -= coef * np.log1p(alpha * total)
            common = coef * alpha / (1.0 + alpha * total)
            grad_x[hi] -= common
            grad_x[lo] -= common * beta

        for family_name in model.family_totals:
            coef = float(family_coef[family_name])
            members = family_indices[family_name]
            total = float(np.sum(x[members]))
            value -= coef * np.log1p(alpha * total)
            grad_x[members] -= coef * alpha / (1.0 + alpha * total)

        if penalty_coef != 0.0:
            penalty_grad = np.zeros(n_domains, dtype=float)
            for domain_idx in singleton_indices:
                inside = np.log1p(x[domain_idx]) - tau
                sp = float(softplus(inside))
                if sp != 0.0:
                    grad_inside = 2.0 * sp * float(sigmoid(inside))
                    penalty_grad[domain_idx] += grad_inside / (1.0 + x[domain_idx])
            for hi, lo in pair_map:
                total = x[hi] + x[lo]
                inside = np.log1p(total) - tau
                sp = float(softplus(inside))
                if sp != 0.0:
                    grad_inside = 2.0 * sp * float(sigmoid(inside))
                    common = grad_inside / (1.0 + total)
                    penalty_grad[hi] += common
                    penalty_grad[lo] += common
            value += penalty_coef * np.sum(
                softplus(np.log1p(np.asarray([x[idx] for idx in singleton_indices])) - tau) ** 2
            )
            if pair_map:
                pair_totals = np.asarray([x[hi] + x[lo] for hi, lo in pair_map], dtype=float)
                value += penalty_coef * np.sum(softplus(np.log1p(pair_totals) - tau) ** 2)
            grad_x += penalty_coef * penalty_grad

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
        raise RuntimeError("Failed to optimize generic-family model")

    z = np.asarray(best.x, dtype=float)
    logits0 = z[:n_domains]
    logits1 = z[n_domains:]
    phase0 = np.exp(logits0 - np.max(logits0))
    phase0 /= np.sum(phase0)
    phase1 = np.exp(logits1 - np.max(logits1))
    phase1 /= np.sum(phase1)
    return best, phase0, phase1


def optimize_generic_family_convex_hull(
    model: GenericFamilyRetainedTotalSurrogate,
    anchors: np.ndarray,
    *,
    maxiter: int = 100,
    start_indices: np.ndarray | None = None,
    linear_penalty: np.ndarray | None = None,
    pairwise_penalty: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    if model.coef_ is None or model.intercept_ is None:
        raise RuntimeError("Model must be fit before convex-hull optimization")

    num_anchors = anchors.shape[0]
    if linear_penalty is not None and linear_penalty.shape != (num_anchors,):
        raise ValueError(f"linear_penalty shape {linear_penalty.shape} != ({num_anchors},)")
    if pairwise_penalty is not None and pairwise_penalty.shape != (num_anchors, num_anchors):
        raise ValueError(f"pairwise_penalty shape {pairwise_penalty.shape} != ({num_anchors}, {num_anchors})")

    def objective(z: np.ndarray) -> float:
        shifted = z - np.max(z)
        coeffs = np.exp(shifted)
        coeffs /= np.sum(coeffs)
        weights = np.tensordot(coeffs, anchors, axes=1)[None, :, :]
        value = float(model.predict(weights)[0])
        if linear_penalty is not None:
            value += float(linear_penalty @ coeffs)
        if pairwise_penalty is not None:
            value += float(coeffs @ pairwise_penalty @ coeffs)
        return value

    vertex_indices = range(num_anchors) if start_indices is None else np.asarray(start_indices, dtype=int).tolist()
    starts = [np.zeros(num_anchors, dtype=float)] + [
        np.eye(num_anchors, dtype=float)[idx] * 4.0 for idx in vertex_indices
    ]

    best_result = None
    best_value = float("inf")
    for start in starts:
        result = minimize(objective, start, method="L-BFGS-B", options={"maxiter": maxiter})
        if float(result.fun) < best_value:
            best_value = float(result.fun)
            best_result = result

    if best_result is None:
        raise RuntimeError("Convex-hull optimization failed")

    shifted = np.asarray(best_result.x, dtype=float) - np.max(best_result.x)
    coeffs = np.exp(shifted)
    coeffs /= np.sum(coeffs)
    best_weights = np.tensordot(coeffs, anchors, axes=1)
    predicted_value = float(model.predict(best_weights[None, :, :])[0])
    return predicted_value, coeffs, best_weights


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0))))


def pack_params(params: dict[str, float]) -> np.ndarray:
    beta = float(np.clip(params["beta"], 1e-8, 1.0 - 1e-8))
    return np.asarray(
        [
            np.log(float(params["alpha"])),
            np.log(float(params["eta"])),
            np.log(float(params["lam"])),
            float(params["tau"]),
            np.log(float(params["reg"])),
            np.log(beta / (1.0 - beta)),
        ],
        dtype=float,
    )


def unpack_params(z: np.ndarray) -> dict[str, float]:
    return {
        "alpha": float(np.exp(np.clip(z[0], -8.0, 8.0))),
        "eta": float(np.exp(np.clip(z[1], -8.0, 8.0))),
        "lam": float(np.exp(np.clip(z[2], -12.0, 4.0))),
        "tau": float(np.clip(z[3], -2.0, 8.0)),
        "reg": float(np.exp(np.clip(z[4], -18.0, -2.0))),
        "beta": float(np.clip(_sigmoid(float(z[5])), 1e-6, 1.0 - 1e-6)),
    }


def evaluate_params(
    z: np.ndarray,
    packet: GenericFamilyPacket,
    valid_weights: np.ndarray,
    valid_y: np.ndarray,
    *,
    seed: int = 0,
) -> dict[str, float | bool]:
    params = unpack_params(z)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros_like(packet.base.y)
    fold_regrets: list[float] = []

    for tr, te in kf.split(packet.base.w):
        model = GenericFamilyRetainedTotalSurrogate(packet, params=params).fit(packet.base.w[tr], packet.base.y[tr])
        pred = model.predict(packet.base.w[te])
        oof[te] = pred
        fold_regrets.append(float(packet.base.y[te][int(np.argmin(pred))] - np.min(packet.base.y[te])))

    full_model = GenericFamilyRetainedTotalSurrogate(packet, params=params).fit(packet.base.w, packet.base.y)
    train_pred = full_model.predict(packet.base.w)
    anchor_pred = full_model.predict(valid_weights)
    anchor_err = anchor_pred - valid_y

    train_res = train_pred - packet.base.y
    cv_res = oof - packet.base.y
    sst = float(np.sum((packet.base.y - np.mean(packet.base.y)) ** 2))
    cv_rmse = float(np.sqrt(np.mean(cv_res**2)))
    anchor_mae = float(np.mean(np.abs(anchor_err)))
    foldmean_regret = float(np.mean(fold_regrets))
    objective = CV_WEIGHT * cv_rmse + ANCHOR_WEIGHT * anchor_mae + REGRET_WEIGHT * foldmean_regret

    return {
        **params,
        "objective": objective,
        "train_rmse": float(np.sqrt(np.mean(train_res**2))),
        "train_r2": float(1.0 - float(np.sum(train_res**2)) / sst),
        "train_spearman": float(spearmanr(packet.base.y, train_pred).statistic),
        "cv_rmse": cv_rmse,
        "cv_r2": float(1.0 - float(np.sum(cv_res**2)) / sst),
        "cv_spearman": float(spearmanr(packet.base.y, oof).statistic),
        "cv_regret_at_1": float(packet.base.y[int(np.argmin(oof))] - np.min(packet.base.y)),
        "cv_foldmean_regret_at_1": foldmean_regret,
        "anchor_mae": anchor_mae,
        "anchor_rmse": float(np.sqrt(np.mean(anchor_err**2))),
        "anchor_rank_correct": bool(int(np.argmin(anchor_pred)) == int(np.argmin(valid_y))),
        "pred_validated_global": float(anchor_pred[0]),
        "pred_validated_pair": float(anchor_pred[1]),
    }


def objective_value_from_metrics(metrics: dict[str, float | bool], objective_name: str) -> float:
    if objective_name == "single_foldmean":
        return float(metrics["objective"])
    if objective_name == "single_cvregret":
        return float(metrics["cv_rmse"]) + float(metrics["anchor_mae"]) + 0.2 * float(metrics["cv_regret_at_1"])
    if objective_name == "single_both":
        return (
            float(metrics["cv_rmse"])
            + float(metrics["anchor_mae"])
            + 0.2 * float(metrics["cv_regret_at_1"])
            + 0.02 * float(metrics["cv_foldmean_regret_at_1"])
        )
    raise ValueError(f"Unsupported tuning objective: {objective_name}")


def tune_genericfamily_params(
    packet: GenericFamilyPacket,
    valid_weights: np.ndarray,
    valid_y: np.ndarray,
    *,
    method: str = "L-BFGS-B",
    objective_name: str = "single_foldmean",
    start_params: dict[str, float] | None = None,
    seed: int = 0,
) -> tuple[dict[str, float | bool], Any]:
    start = pack_params(CURRENT_TUNED_GENERIC_PARAMS if start_params is None else start_params)

    def objective(z: np.ndarray) -> float:
        metrics = evaluate_params(z, packet, valid_weights, valid_y, seed=seed)
        return objective_value_from_metrics(metrics, objective_name)

    options = {
        "L-BFGS-B": {"maxiter": 250, "ftol": 1e-6},
        "Nelder-Mead": {"maxiter": 900, "xatol": 1e-4, "fatol": 1e-6},
        "Powell": {"maxiter": 400, "xtol": 1e-4, "ftol": 1e-6},
    }.get(method, {"maxiter": 250})
    result = minimize(objective, start, method=method, options=options)
    metrics = evaluate_params(np.asarray(result.x, dtype=float), packet, valid_weights, valid_y, seed=seed)
    metrics = {
        "success": bool(result.success),
        "message": str(result.message),
        "method": method,
        "objective_name": objective_name,
        **metrics,
        "objective": objective_value_from_metrics(metrics, objective_name),
    }
    return metrics, result
