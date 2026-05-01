# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "plotly", "scipy", "scikit-learn", "kaleido"]
# ///
"""Analyze the 60M single-phase ablation and fit one-phase GRP no-L2 variants."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from scipy.optimize import minimize, nnls

from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import (
    TARGET_BUDGET_DOLMA3_COMMON_CRAWL,
    TOP_LEVEL_DOMAIN_TOKEN_COUNTS,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = SCRIPT_DIR / "reference_outputs" / "single_phase_exposure_average_60m_1p2b"
MANIFEST_CSV = ARTIFACT_DIR / "single_phase_exposure_average_manifest.csv"
OUTPUT_DIR = ARTIFACT_DIR / "analysis"
SINGLE_PHASE_FIT_DATASET_CSV = OUTPUT_DIR / "single_phase_fit_dataset.csv"
PAIRED_CSV = OUTPUT_DIR / "paired_single_vs_two_phase.csv"
SUMMARY_JSON = OUTPUT_DIR / "single_phase_summary.json"
ABLATION_SUMMARY_CSV = OUTPUT_DIR / "single_phase_grp_no_l2_ablation_summary.csv"
ABLATION_PARAMS_CSV = OUTPUT_DIR / "single_phase_grp_no_l2_ablation_params.csv"
REPORT_MD = OUTPUT_DIR / "single_phase_grp_no_l2_ablation_report.md"
GCS_METRICS_PATTERN = (
    "gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_single_phase_exposure_average_60m_1p2b/*/checkpoints/eval_metrics.jsonl"
)
OBJECTIVE_METRIC = "eval/uncheatable_eval/bpb"
CV_SEED = 0
LOWER_TAIL_FRAC = 0.15
PHASE_FRACTIONS = (0.8, 0.2)
GENERIC_FAMILY_NAMES = ("broad_text", "tech_code", "reasoning")
REG_FIXED = 0.0
NO_L2_PARAM_KEYS = (
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
BASE_NO_L2_PARAMS = {
    "eta": 5.222440513840459,
    "lam": 7.04928339546768e-06,
    "reg": REG_FIXED,
    "beta": 0.1967681464478872,
    "a_broad_text": 0.48485414608456984,
    "a_tech_code": 0.04843166940506106,
    "a_reasoning": 1.0344800333570379,
    "tau_broad_text": 3.0915710553505598,
    "tau_tech_code": 8.0,
    "tau_reasoning": 4.860956465592155,
}


def _kfold_indices(n_rows: int, *, n_splits: int = 5, seed: int = CV_SEED):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_rows)
    fold_sizes = np.full(n_splits, n_rows // n_splits, dtype=int)
    fold_sizes[: n_rows % n_splits] += 1
    start = 0
    for fold_size in fold_sizes:
        stop = start + int(fold_size)
        test_idx = np.sort(indices[start:stop])
        train_idx = np.sort(np.concatenate([indices[:start], indices[stop:]]))
        yield train_idx, test_idx
        start = stop


@dataclass(frozen=True)
class SinglePhasePacket:
    """Feature-ready packet for the one-phase GRP ablation."""

    frame: pd.DataFrame
    y: np.ndarray
    w: np.ndarray
    c0: np.ndarray
    c1: np.ndarray
    domain_names: list[str]
    pairs: list[tuple[int, int]]
    singletons: list[int]
    family_map: dict[str, list[int]]


@dataclass(frozen=True)
class AblationSpec:
    """A single one-phase GRP ablation."""

    slug: str
    display_name: str
    fixed_params: dict[str, float]
    quality_discount: bool = True
    include_singletons: bool = True
    include_pairs: bool = True
    include_family_totals: bool = True
    include_family_group_penalty: bool = True


ABLATIONS = (
    AblationSpec(
        slug="full_one_phase",
        display_name="Full one-phase GRP no-L2",
        fixed_params={},
    ),
    AblationSpec(
        slug="no_retention_lam0",
        display_name="No retention: lambda=0",
        fixed_params={"lam": 0.0},
    ),
    AblationSpec(
        slug="no_mu_eta1",
        display_name="No phase multiplier: eta=1",
        fixed_params={"eta": 1.0},
    ),
    AblationSpec(
        slug="exposure_only_lam0_eta1",
        display_name="Exposure only: lambda=0, eta=1",
        fixed_params={"lam": 0.0, "eta": 1.0},
    ),
    AblationSpec(
        slug="exposure_only_no_quality",
        display_name="Exposure only, no quality discount",
        fixed_params={"lam": 0.0, "eta": 1.0, "beta": 1.0},
        quality_discount=False,
    ),
    AblationSpec(
        slug="full_no_family_penalty",
        display_name="Full signal, no family saturation penalty",
        fixed_params={},
        include_family_group_penalty=False,
    ),
    AblationSpec(
        slug="exposure_only_no_family_penalty",
        display_name="Exposure only, no family saturation penalty",
        fixed_params={"lam": 0.0, "eta": 1.0},
        include_family_group_penalty=False,
    ),
    AblationSpec(
        slug="family_totals_only_exposure",
        display_name="Family totals only, exposure only",
        fixed_params={"lam": 0.0, "eta": 1.0},
        include_singletons=False,
        include_pairs=False,
        include_family_group_penalty=False,
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--method", default="Powell")
    parser.add_argument("--coarse-top-k", type=int, default=4)
    parser.add_argument("--random-starts", type=int, default=12)
    parser.add_argument("--skip-gcs-refresh", action="store_true")
    return parser.parse_args()


def _list_gcs_paths(pattern: str) -> list[str]:
    output = subprocess.check_output(["gcloud", "storage", "ls", pattern], text=True)
    return [line.strip() for line in output.splitlines() if line.strip()]


def _read_last_jsonl_record(uri: str) -> dict[str, Any]:
    output = subprocess.check_output(["gcloud", "storage", "cat", uri], text=True)
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty metrics file: {uri}")
    return json.loads(lines[-1])


def _run_name_from_metric_uri(uri: str, expected_run_names: tuple[str, ...]) -> str:
    checkpoint_name = uri.rstrip("/").split("/")[-3]
    for run_name in expected_run_names:
        if checkpoint_name.startswith(f"{run_name}-"):
            return run_name
    raise ValueError(f"Could not map checkpoint name {checkpoint_name!r} to a manifest run")


def collect_single_phase_fit_dataset(manifest: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Collect target-step single-phase BPB labels from checkpoint metrics."""
    expected_run_names = tuple(sorted(manifest["run_name"].astype(str).unique(), key=len, reverse=True))
    rows: list[dict[str, Any]] = []
    for uri in _list_gcs_paths(GCS_METRICS_PATTERN):
        metrics = _read_last_jsonl_record(uri)
        run_name = _run_name_from_metric_uri(uri, expected_run_names)
        if OBJECTIVE_METRIC not in metrics:
            raise ValueError(f"{OBJECTIVE_METRIC} missing in {uri}")
        rows.append(
            {
                "run_name": run_name,
                "step": int(metrics.get("step", -1)),
                OBJECTIVE_METRIC: float(metrics[OBJECTIVE_METRIC]),
                "metrics_uri": uri,
                "checkpoint_root": uri.removesuffix("/checkpoints/eval_metrics.jsonl"),
            }
        )
    frame = pd.DataFrame(rows).sort_values("run_name")
    if frame["run_name"].duplicated().any():
        duplicated = sorted(frame.loc[frame["run_name"].duplicated(), "run_name"].unique())
        raise ValueError(f"Duplicate single-phase metrics rows: {duplicated[:8]}")
    missing = sorted(set(expected_run_names) - set(frame["run_name"]))
    if missing:
        raise ValueError(f"Missing single-phase metrics for {len(missing)} runs: {missing[:8]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def _paired_frame(manifest: pd.DataFrame, single: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    slim_single = single[["run_name", OBJECTIVE_METRIC, "checkpoint_root"]].rename(
        columns={OBJECTIVE_METRIC: "single_phase_bpb", "checkpoint_root": "single_phase_checkpoint_root"}
    )
    paired = manifest.merge(slim_single, on="run_name", how="left")
    paired["two_phase_bpb"] = paired["source_60m_bpb"]
    paired["delta_bpb"] = paired["single_phase_bpb"] - paired["two_phase_bpb"]
    paired.to_csv(output_dir / "paired_single_vs_two_phase.csv", index=False)
    return paired


def _write_plot(fig: go.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path.with_suffix(".html"))
    try:
        fig.write_image(output_path.with_suffix(".png"), scale=2)
    except Exception as exc:
        print(f"warning: failed to write {output_path.with_suffix('.png')}: {exc}", flush=True)


def plot_paired_single_vs_two(paired: pd.DataFrame, output_dir: Path) -> None:
    """Plot paired single-phase versus original two-phase BPB."""
    observed = paired.loc[paired["single_phase_bpb"].notna()].copy()
    low = float(min(observed["two_phase_bpb"].min(), observed["single_phase_bpb"].min()))
    high = float(max(observed["two_phase_bpb"].max(), observed["single_phase_bpb"].max()))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=observed["two_phase_bpb"],
            y=observed["single_phase_bpb"],
            mode="markers",
            marker={
                "size": 8,
                "color": observed["delta_bpb"],
                "colorscale": "RdYlGn_r",
                "reversescale": True,
                "colorbar": {"title": "single - two BPB"},
            },
            text=observed["run_name"],
            customdata=np.stack(
                [
                    observed["source_run_name"].astype(str),
                    observed["delta_bpb"].to_numpy(dtype=float),
                    observed["phase_tv"].to_numpy(dtype=float),
                    observed["priority_rank"].to_numpy(dtype=float),
                ],
                axis=1,
            ),
            hovertemplate=(
                "run=%{text}<br>source=%{customdata[0]}<br>"
                "two-phase=%{x:.6f}<br>single-phase=%{y:.6f}<br>"
                "delta=%{customdata[1]:+.6f}<br>phase TV=%{customdata[2]:.3f}<br>"
                "priority=%{customdata[3]:.0f}<extra></extra>"
            ),
            name="paired rows",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            line={"color": "black", "dash": "dash"},
            name="parity",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title="Single-phase exposure-average vs original two-phase BPB",
        xaxis_title="Original two-phase BPB",
        yaxis_title="Single-phase exposure-average BPB",
        width=1100,
        height=800,
    )
    _write_plot(fig, output_dir / "single_vs_two_phase_bpb")

    hist = go.Figure()
    hist.add_trace(go.Histogram(x=observed["delta_bpb"], nbinsx=40, name="delta BPB"))
    hist.add_vline(x=0.0, line_color="black", line_dash="dash")
    hist.update_layout(
        title="Delta BPB distribution: single-phase - two-phase",
        xaxis_title="Delta BPB",
        yaxis_title="Count",
        width=1000,
        height=600,
    )
    _write_plot(hist, output_dir / "delta_bpb_distribution")


def _summary_stats(paired: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    observed = paired.loc[paired["single_phase_bpb"].notna()].copy()
    summary = {
        "n": len(observed),
        "mean_single_phase_bpb": float(observed["single_phase_bpb"].mean()),
        "mean_two_phase_bpb": float(observed["two_phase_bpb"].mean()),
        "mean_delta_bpb": float(observed["delta_bpb"].mean()),
        "median_delta_bpb": float(observed["delta_bpb"].median()),
        "single_phase_better_count": int((observed["delta_bpb"] < 0.0).sum()),
        "two_phase_better_count": int((observed["delta_bpb"] > 0.0).sum()),
        "tie_count": int((observed["delta_bpb"] == 0.0).sum()),
        "single_phase_better_fraction": float((observed["delta_bpb"] < 0.0).mean()),
    }
    (output_dir / "single_phase_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _phase_names(frame: pd.DataFrame) -> list[str]:
    return sorted(
        {
            column.split("_", 2)[0] + "_" + column.split("_", 2)[1]
            for column in frame.columns
            if column.startswith("phase_") and not column.endswith("_epochs") and column.count("_") >= 2
        }
    )


def _domain_names(frame: pd.DataFrame, first_phase: str) -> list[str]:
    return [
        column.replace(f"{first_phase}_", "")
        for column in frame.columns
        if column.startswith(f"{first_phase}_") and not column.endswith("_epochs")
    ]


def _default_family(domain_name: str) -> str:
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
        family
        for family, is_member in (("broad_text", is_broad), ("tech_code", is_tech), ("reasoning", is_reasoning))
        if is_member
    ]
    if len(assigned) != 1:
        raise ValueError(f"Expected exactly one family for {domain_name!r}, got {assigned}")
    return assigned[0]


def _packet_from_single_phase_frame(frame: pd.DataFrame) -> SinglePhasePacket:
    model_frame = frame.loc[frame[OBJECTIVE_METRIC].notna()].reset_index(drop=True)
    phase_names = _phase_names(model_frame)
    if phase_names != ["phase_0", "phase_1"]:
        raise ValueError(f"Expected phase_0/phase_1 columns, got {phase_names}")
    domain_names = _domain_names(model_frame, phase_names[0])
    weights = np.zeros((len(model_frame), len(phase_names), len(domain_names)), dtype=float)
    for phase_idx, phase_name in enumerate(phase_names):
        for domain_idx, domain_name in enumerate(domain_names):
            weights[:, phase_idx, domain_idx] = model_frame[f"{phase_name}_{domain_name}"].to_numpy(dtype=float)
    token_counts = np.asarray([TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain_name] for domain_name in domain_names], dtype=float)
    epoch_multipliers = np.asarray(
        [
            [fraction * TARGET_BUDGET_DOLMA3_COMMON_CRAWL / token_count for token_count in token_counts]
            for fraction in PHASE_FRACTIONS
        ],
        dtype=float,
    )

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
    family_map = {family_name: [] for family_name in GENERIC_FAMILY_NAMES}
    for idx, domain_name in enumerate(domain_names):
        family_map[_default_family(domain_name)].append(idx)
    return SinglePhasePacket(
        frame=model_frame,
        y=model_frame[OBJECTIVE_METRIC].to_numpy(dtype=float),
        w=weights,
        c0=epoch_multipliers[0],
        c1=epoch_multipliers[1],
        domain_names=domain_names,
        pairs=pairs,
        singletons=[idx for idx in range(len(domain_names)) if idx not in paired],
        family_map=family_map,
    )


def _softplus(x: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(arr))) + np.maximum(arr, 0.0)


def _pack_no_l2_params(params: dict[str, float]) -> np.ndarray:
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


def _unpack_no_l2_params(z: np.ndarray) -> dict[str, float]:
    return {
        "eta": float(np.exp(np.clip(z[0], -8.0, 8.0))),
        "lam": float(np.exp(np.clip(z[1], -12.0, 4.0))),
        "beta": float(np.clip(1.0 / (1.0 + np.exp(-np.clip(z[2], -50.0, 50.0))), 1e-6, 1.0 - 1e-6)),
        "a_broad_text": float(np.exp(np.clip(z[3], np.log(0.02), np.log(2.0)))),
        "a_tech_code": float(np.exp(np.clip(z[4], np.log(0.02), np.log(2.0)))),
        "a_reasoning": float(np.exp(np.clip(z[5], np.log(0.02), np.log(2.0)))),
        "tau_broad_text": float(np.clip(z[6], -2.0, 8.0)),
        "tau_tech_code": float(np.clip(z[7], -2.0, 8.0)),
        "tau_reasoning": float(np.clip(z[8], -2.0, 8.0)),
        "reg": REG_FIXED,
    }


class SinglePhaseGrpSurrogate:
    """Minimal GRP power-family-penalty model for one-phase ablations."""

    def __init__(self, packet: SinglePhasePacket, *, params: dict[str, float], spec: AblationSpec):
        self.packet = packet
        self.params = dict(params)
        self.spec = spec
        domain_to_family: list[str] = []
        for domain_idx in range(len(packet.domain_names)):
            assigned = [family_name for family_name, members in packet.family_map.items() if domain_idx in members]
            if len(assigned) != 1:
                raise ValueError(f"Expected one family for domain {domain_idx}, got {assigned}")
            domain_to_family.append(assigned[0])
        self.domain_to_family = tuple(domain_to_family)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def _retained_x(self, weights: np.ndarray) -> np.ndarray:
        p0 = weights[:, 0, :]
        p1 = weights[:, 1, :]
        e0 = p0 * self.packet.c0[None, :]
        e1 = p1 * self.packet.c1[None, :]
        lam = float(self.params["lam"])
        eta = float(self.params["eta"])
        return np.exp(-lam * (1.0 - p1)) * e0 + eta * e1

    def _curvature(self, family_name: str, other_family_name: str | None = None) -> float:
        first = float(self.params[f"a_{family_name}"])
        if other_family_name is None:
            return first
        return 0.5 * (first + float(self.params[f"a_{other_family_name}"]))

    def _feature_transform(
        self, values: np.ndarray, family_name: str, other_family_name: str | None = None
    ) -> np.ndarray:
        safe = np.maximum(np.asarray(values, dtype=float), 1e-12)
        return np.power(safe, self._curvature(family_name, other_family_name))

    def _pair_signal_total(self, x_hi: np.ndarray, x_lo: np.ndarray) -> np.ndarray:
        lo_scale = float(self.params["beta"]) if self.spec.quality_discount else 1.0
        return np.asarray(x_hi, dtype=float) + lo_scale * np.asarray(x_lo, dtype=float)

    def build_design(self, weights: np.ndarray) -> np.ndarray:
        x = self._retained_x(weights)
        features: list[np.ndarray] = []
        family_group_totals: dict[str, list[np.ndarray]] = {family_name: [] for family_name in GENERIC_FAMILY_NAMES}

        if self.spec.include_singletons:
            for idx in self.packet.singletons:
                family_name = self.domain_to_family[idx]
                features.append(self._feature_transform(x[:, idx], family_name)[:, None])
                family_group_totals[family_name].append(x[:, idx])

        if self.spec.include_pairs:
            for hi, lo in self.packet.pairs:
                hi_family = self.domain_to_family[hi]
                lo_family = self.domain_to_family[lo]
                signal_total = self._pair_signal_total(x[:, hi], x[:, lo])
                features.append(self._feature_transform(signal_total, hi_family, lo_family)[:, None])
                family_group_totals[hi_family].append(x[:, hi] + x[:, lo])

        family_totals: dict[str, np.ndarray] = {}
        for family_name in GENERIC_FAMILY_NAMES:
            members = self.packet.family_map[family_name]
            family_total = np.sum(x[:, members], axis=1)
            family_totals[family_name] = family_total
            if self.spec.include_family_totals:
                features.append(self._feature_transform(family_total, family_name)[:, None])

        penalties: list[np.ndarray] = []
        if self.spec.include_family_group_penalty:
            for family_name in GENERIC_FAMILY_NAMES:
                if not family_group_totals[family_name]:
                    continue
                tau_f = float(self.params[f"tau_{family_name}"])
                penalty_inputs = np.stack(family_group_totals[family_name], axis=1)
                penalties.append(np.sum(_softplus(np.log1p(penalty_inputs) - tau_f) ** 2, axis=1, keepdims=True))

        design = np.hstack(features + penalties)
        design[:, : len(features)] *= -1.0
        return design

    def fit(self, weights: np.ndarray, targets: np.ndarray) -> SinglePhaseGrpSurrogate:
        design = self.build_design(weights)
        design_mean = design.mean(axis=0, keepdims=True)
        target_mean = float(targets.mean())
        coef, _ = nnls(design - design_mean, targets - target_mean)
        self.coef_ = coef
        self.intercept_ = float(target_mean - (design_mean @ coef).item())
        return self

    def predict(self, weights: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model must be fit before prediction")
        return np.asarray(self.intercept_ + self.build_design(weights) @ self.coef_, dtype=float)


def _start_bank() -> tuple[dict[str, float], ...]:
    base = dict(BASE_NO_L2_PARAMS)
    rows = [
        base,
        {**base, "eta": base["eta"] * 0.7},
        {**base, "eta": base["eta"] * 1.4},
        {**base, "lam": max(base["lam"] * 0.1, 1e-8)},
        {**base, "lam": min(base["lam"] * 20.0 + 1e-8, 1.0)},
        {**base, "beta": 0.05},
        {**base, "beta": 0.5},
        {**base, "a_broad_text": max(base["a_broad_text"] * 0.7, 0.02)},
        {**base, "a_tech_code": min(base["a_tech_code"] * 2.0, 2.0)},
        {**base, "a_reasoning": min(base["a_reasoning"] * 1.4, 2.0)},
    ]
    return tuple({key: float(value) for key, value in row.items()} for row in rows)


def _expanded_start_bank(random_starts: int) -> tuple[dict[str, float], ...]:
    starts = list(_start_bank())
    if random_starts <= 0:
        return tuple(starts)
    rng = np.random.default_rng(0)
    seed_pack = _pack_no_l2_params(starts[0])
    for _ in range(random_starts):
        starts.append(_unpack_no_l2_params(seed_pack + rng.normal(0.0, 0.85, size=seed_pack.shape)))
    seen: set[tuple[tuple[str, float], ...]] = set()
    deduped: list[dict[str, float]] = []
    for row in starts:
        row = dict(row)
        row["reg"] = REG_FIXED
        key = tuple(sorted((name, round(float(value), 8)) for name, value in row.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append({name: float(value) for name, value in row.items()})
    return tuple(deduped)


def _free_keys_for_ablation(spec: AblationSpec) -> tuple[str, ...]:
    fixed = set(spec.fixed_params)
    keys = [key for key in NO_L2_PARAM_KEYS if key not in fixed]
    if not spec.quality_discount and "beta" in keys:
        keys.remove("beta")
    if not spec.include_family_group_penalty:
        keys = [key for key in keys if not key.startswith("tau_")]
    return tuple(keys)


def _unpack_one_param(key: str, value: float) -> float:
    if key in {"eta", "lam"}:
        return float(np.exp(np.clip(value, -12.0 if key == "lam" else -8.0, 4.0 if key == "lam" else 8.0)))
    if key == "beta":
        return float(np.clip(1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0))), 1e-6, 1.0 - 1e-6))
    if key.startswith("a_"):
        return float(np.exp(np.clip(value, np.log(0.02), np.log(2.0))))
    if key.startswith("tau_"):
        return float(np.clip(value, -2.0, 8.0))
    raise ValueError(f"Unsupported parameter key: {key}")


def _pack_one_param(key: str, value: float) -> float:
    if key in {"eta", "lam"}:
        return float(np.log(max(float(value), 1e-12)))
    if key == "beta":
        beta = float(np.clip(value, 1e-8, 1.0 - 1.0e-8))
        return float(np.log(beta / (1.0 - beta)))
    if key.startswith("a_"):
        return float(np.log(max(float(value), 1e-12)))
    if key.startswith("tau_"):
        return float(value)
    raise ValueError(f"Unsupported parameter key: {key}")


def _params_from_free_vector(
    z: np.ndarray, base_params: dict[str, float], free_keys: tuple[str, ...]
) -> dict[str, float]:
    params = dict(base_params)
    for value, key in zip(z, free_keys, strict=True):
        params[key] = _unpack_one_param(key, float(value))
    params["reg"] = REG_FIXED
    return params


def _pack_free_params(params: dict[str, float], free_keys: tuple[str, ...]) -> np.ndarray:
    return np.asarray([_pack_one_param(key, params[key]) for key in free_keys], dtype=float)


def _fit_model(packet, params: dict[str, float], spec: AblationSpec):
    return SinglePhaseGrpSurrogate(packet, params=params, spec=spec).fit(packet.w, packet.y)


def _oof_metrics(packet, params: dict[str, float], spec: AblationSpec) -> dict[str, float]:
    y = packet.y
    oof = np.zeros_like(y)
    fold_regrets: list[float] = []
    for train_idx, test_idx in _kfold_indices(len(y), n_splits=5, seed=CV_SEED):
        model = SinglePhaseGrpSurrogate(packet, params=params, spec=spec).fit(packet.w[train_idx], y[train_idx])
        pred = model.predict(packet.w[test_idx])
        oof[test_idx] = pred
        fold_regrets.append(float(y[test_idx][int(np.argmin(pred))] - np.min(y[test_idx])))
    residual = oof - y
    tail_count = max(5, int(np.ceil(LOWER_TAIL_FRAC * len(y))))
    tail_idx = np.argsort(oof)[:tail_count]
    lower_tail_optimism = float(np.mean(np.maximum(y[tail_idx] - oof[tail_idx], 0.0)))
    cv_rmse = float(np.sqrt(np.mean(residual**2)))
    objective = cv_rmse + 0.05 * float(np.mean(fold_regrets)) + 0.5 * lower_tail_optimism
    return {
        "cv_rmse": cv_rmse,
        "cv_mae": float(np.mean(np.abs(residual))),
        "cv_spearman": float(stats.spearmanr(y, oof).statistic),
        "cv_regret_at_1": float(y[int(np.argmin(oof))] - np.min(y)),
        "cv_foldmean_regret_at_1": float(np.mean(fold_regrets)),
        "lower_tail_optimism": lower_tail_optimism,
        "objective": objective,
    }


def _coarse_rows(packet, start_bank: tuple[dict[str, float], ...], spec: AblationSpec) -> pd.DataFrame:
    rows = []
    for start_id, start in enumerate(start_bank):
        params = dict(start)
        params.update(spec.fixed_params)
        params["reg"] = REG_FIXED
        rows.append({"start_id": start_id, "stage": "coarse", **params, **_oof_metrics(packet, params, spec)})
    return pd.DataFrame(rows).sort_values(["objective", "cv_rmse"], ascending=[True, True])


def _refine_ablation(
    packet,
    start_bank: tuple[dict[str, float], ...],
    spec: AblationSpec,
    *,
    coarse_top_k: int,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    coarse = _coarse_rows(packet, start_bank, spec)
    free_keys = _free_keys_for_ablation(spec)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for chosen_rank, start_id in enumerate(coarse["start_id"].head(coarse_top_k).tolist()):
        base_params = dict(start_bank[int(start_id)])
        base_params.update(spec.fixed_params)
        base_params["reg"] = REG_FIXED
        if not free_keys:
            params = dict(base_params)
            metrics = _oof_metrics(packet, params, spec)
            row = {
                "chosen_rank": chosen_rank,
                "start_id": int(start_id),
                "stage": "refine",
                "success": True,
                "message": "no_free_params",
                **params,
                **metrics,
            }
            rows.append(row)
            best = row if best is None or row["objective"] < best["objective"] else best
            continue

        start = _pack_free_params(base_params, free_keys)
        cache: dict[tuple[float, ...], float] = {}

        def objective(
            z: np.ndarray,
            _cache: dict[tuple[float, ...], float] = cache,
            _base_params: dict[str, float] = base_params,
        ) -> float:
            key = tuple(np.round(np.asarray(z, dtype=float), 8))
            if key not in _cache:
                params = _params_from_free_vector(z, _base_params, free_keys)
                params.update(spec.fixed_params)
                params["reg"] = REG_FIXED
                _cache[key] = _oof_metrics(packet, params, spec)["objective"]
            return _cache[key]

        options = {
            "L-BFGS-B": {"maxiter": 160, "ftol": 1e-8},
            "Nelder-Mead": {"maxiter": 900, "xatol": 1e-4, "fatol": 1e-8},
            "Powell": {"maxiter": 120, "xtol": 1e-4, "ftol": 1e-8},
        }.get(method, {"maxiter": 240})
        result = minimize(objective, start, method=method, options=options)
        params = _params_from_free_vector(np.asarray(result.x, dtype=float), base_params, free_keys)
        params.update(spec.fixed_params)
        params["reg"] = REG_FIXED
        row = {
            "chosen_rank": chosen_rank,
            "start_id": int(start_id),
            "stage": "refine",
            "success": bool(result.success),
            "message": str(result.message),
            **params,
            **_oof_metrics(packet, params, spec),
        }
        rows.append(row)
        best = row if best is None or row["objective"] < best["objective"] else best
    if best is None:
        raise RuntimeError(f"No best row for {spec.slug}")
    return coarse, pd.DataFrame(rows).sort_values("objective"), best


def _optimize_single_phase_model(packet, model, *, seed: int = 0, n_random: int = 32) -> tuple[Any, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_domains = len(packet.domain_names)
    starts = [np.zeros(n_domains, dtype=float)]
    starts.extend(rng.normal(scale=0.5, size=n_domains) for _ in range(n_random))
    best = None

    def weights_from_logits(z: np.ndarray) -> np.ndarray:
        w = np.exp(z - np.max(z))
        return w / np.sum(w)

    def objective(z: np.ndarray) -> float:
        w = weights_from_logits(z)
        weights = np.stack([w, w], axis=0)[None, :, :]
        return float(model.predict(weights)[0])

    for start in starts:
        result = minimize(objective, start, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-10})
        if best is None or float(result.fun) < float(best.fun):
            best = result
    if best is None:
        raise RuntimeError("Single-phase optimization failed")
    return best, weights_from_logits(np.asarray(best.x, dtype=float))


def _single_phase_family_shares(packet, weights: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for family_name, indices in packet.family_map.items():
        out[f"raw_{family_name}_share"] = float(np.sum(weights[list(indices)]))
    out["raw_support_gt_1e4"] = int(np.sum(weights > 1e-4))
    positive = weights[weights > 0.0]
    out["raw_entropy"] = -float(np.sum(positive * np.log(positive)))
    return out


def _plot_optimum(packet, weights: np.ndarray, output_path: Path, title: str) -> None:
    rows = pd.DataFrame({"domain_name": packet.domain_names, "weight": weights})
    rows = rows.sort_values("weight", ascending=False).head(24)
    fig = go.Figure(
        go.Bar(
            x=rows["weight"],
            y=rows["domain_name"],
            orientation="h",
            marker={"color": rows["weight"], "colorscale": "RdYlGn_r", "reversescale": True},
            hovertemplate="%{y}<br>weight=%{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Single-phase mixture weight",
        yaxis_title="Domain",
        yaxis={"autorange": "reversed"},
        width=1100,
        height=800,
    )
    _write_plot(fig, output_path)


def _parameter_counts(packet: SinglePhasePacket, spec: AblationSpec) -> dict[str, int]:
    signal_feature_count = (
        int(spec.include_singletons) * len(packet.singletons)
        + int(spec.include_pairs) * len(packet.pairs)
        + int(spec.include_family_totals) * len(GENERIC_FAMILY_NAMES)
    )
    penalty_feature_count = int(spec.include_family_group_penalty) * len(GENERIC_FAMILY_NAMES)
    linear_coefficient_count = signal_feature_count + penalty_feature_count
    nonlinear_param_count = len(_free_keys_for_ablation(spec))
    return {
        "signal_feature_count": signal_feature_count,
        "penalty_feature_count": penalty_feature_count,
        "linear_coefficient_count": linear_coefficient_count,
        "intercept_count": 1,
        "linear_head_param_count": linear_coefficient_count + 1,
        "nonlinear_param_count": nonlinear_param_count,
        "total_param_count": nonlinear_param_count + linear_coefficient_count + 1,
    }


def fit_ablations(paired: pd.DataFrame, output_dir: Path, *, method: str, coarse_top_k: int, random_starts: int) -> None:
    fit_frame = paired.copy()
    fit_frame[OBJECTIVE_METRIC] = fit_frame["single_phase_bpb"]
    packet = _packet_from_single_phase_frame(fit_frame)
    start_bank = _expanded_start_bank(random_starts)
    summary_rows: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []
    for spec in ABLATIONS:
        print(f"fitting {spec.slug}", flush=True)
        spec_dir = output_dir / "grp_no_l2_one_phase_ablations" / spec.slug
        spec_dir.mkdir(parents=True, exist_ok=True)
        coarse, refine, best = _refine_ablation(
            packet,
            start_bank,
            spec,
            coarse_top_k=coarse_top_k,
            method=method,
        )
        coarse.to_csv(spec_dir / "coarse.csv", index=False)
        refine.to_csv(spec_dir / "refine.csv", index=False)
        params = {key: float(best[key]) for key in NO_L2_PARAM_KEYS if key in best}
        params.update(spec.fixed_params)
        params["reg"] = REG_FIXED
        model = _fit_model(packet, params, spec)
        train_pred = model.predict(packet.w)
        single_result, single_weights = _optimize_single_phase_model(packet, model, seed=CV_SEED)
        single_as_two = np.stack([single_weights, single_weights], axis=0)
        single_distances = 0.5 * np.mean(np.sum(np.abs(packet.w - single_as_two[None, :, :]), axis=2), axis=1)
        single_nearest_idx = int(np.argmin(single_distances))
        pd.DataFrame(
            {
                "domain_name": packet.domain_names,
                "single_phase_weight": single_weights,
                "single_phase_epochs_total": single_weights * (packet.c0 + packet.c1),
            }
        ).to_csv(spec_dir / "raw_single_phase_optimum_weights.csv", index=False)
        _plot_optimum(packet, single_weights, spec_dir / "raw_single_phase_optimum_top_domains", spec.display_name)
        summary = {
            "slug": spec.slug,
            "display_name": spec.display_name,
            "quality_discount": spec.quality_discount,
            "include_singletons": spec.include_singletons,
            "include_pairs": spec.include_pairs,
            "include_family_totals": spec.include_family_totals,
            "include_family_group_penalty": spec.include_family_group_penalty,
            "fixed_params": json.dumps(spec.fixed_params, sort_keys=True),
            "method": method,
            "coarse_top_k": int(coarse_top_k),
            "start_bank_size": len(start_bank),
            "train_rmse": float(np.sqrt(np.mean((train_pred - packet.y) ** 2))),
            "train_mae": float(np.mean(np.abs(train_pred - packet.y))),
            "train_spearman": float(stats.spearmanr(packet.y, train_pred).statistic),
            "raw_single_phase_predicted_bpb": float(single_result.fun),
            "raw_single_phase_nearest_observed_tv": float(single_distances[single_nearest_idx]),
            "raw_single_phase_nearest_observed_run_name": str(packet.frame.iloc[single_nearest_idx]["run_name"]),
            "raw_single_phase_nearest_observed_bpb": float(packet.y[single_nearest_idx]),
            **{
                key: float(value)
                for key, value in best.items()
                if isinstance(value, int | float | np.integer | np.floating)
            },
            **_single_phase_family_shares(packet, single_weights),
            **_parameter_counts(packet, spec),
        }
        summary_rows.append(summary)
        param_rows.append({"slug": spec.slug, **params})
        print(
            f"{spec.slug}: cv_rmse={summary['cv_rmse']:.6f} "
            f"spearman={summary['cv_spearman']:.3f} "
            f"single_opt={summary['raw_single_phase_predicted_bpb']:.6f}",
            flush=True,
        )
    pd.DataFrame(summary_rows).sort_values(["objective", "cv_rmse"]).to_csv(ABLATION_SUMMARY_CSV, index=False)
    pd.DataFrame(param_rows).to_csv(ABLATION_PARAMS_CSV, index=False)
    _write_report(pd.DataFrame(summary_rows).sort_values(["objective", "cv_rmse"]), output_dir)


def _write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    columns = [
        "slug",
        "total_param_count",
        "cv_rmse",
        "cv_mae",
        "cv_spearman",
        "cv_regret_at_1",
        "cv_foldmean_regret_at_1",
        "lower_tail_optimism",
        "train_rmse",
        "raw_single_phase_predicted_bpb",
        "raw_single_phase_nearest_observed_run_name",
        "raw_single_phase_nearest_observed_bpb",
        "raw_single_phase_nearest_observed_tv",
        "raw_broad_text_share",
        "raw_tech_code_share",
        "raw_reasoning_share",
        "raw_support_gt_1e4",
        "raw_entropy",
    ]
    body = [
        "# Single-Phase GRP No-L2 Ablations",
        "",
        "Objective: `eval/uncheatable_eval/bpb` on the 242-row 60M/1.2B exposure-average single-phase swarm.",
        "",
        "Lower BPB and lower regret are better. The raw optimum is constrained to one shared mixture "
        "for both phases (`phase_0 == phase_1`).",
        "",
        summary[columns].to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (output_dir / "single_phase_grp_no_l2_ablation_report.md").write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    if args.skip_gcs_refresh and SINGLE_PHASE_FIT_DATASET_CSV.exists():
        single = pd.read_csv(SINGLE_PHASE_FIT_DATASET_CSV)
    else:
        single = collect_single_phase_fit_dataset(manifest, SINGLE_PHASE_FIT_DATASET_CSV)
    paired = _paired_frame(manifest, single, args.output_dir)
    plot_paired_single_vs_two(paired, args.output_dir)
    summary = _summary_stats(paired, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    fit_ablations(
        paired,
        args.output_dir,
        method=args.method,
        coarse_top_k=args.coarse_top_k,
        random_starts=args.random_starts,
    )
    print(f"Wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
