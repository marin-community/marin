# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# ///
"""Build a lean collaborator packet for mixture/scale modeling data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PACKET_NAME = "collaborator_scaling_data_packet_20260430"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / DEFAULT_PACKET_NAME
DEFAULT_ARCHIVE_PATH = SCRIPT_DIR / f"{DEFAULT_PACKET_NAME}.zip"


@dataclass(frozen=True)
class CopySpec:
    """One file to copy into the packet."""

    source: Path
    destination: Path
    required: bool = True


STANDALONE_README = """# Standalone Code

These scripts are intentionally small and self-contained. They do not import Marin, Levanter, Iris,
or any branch-local experiment modules. They only expect the CSV/NPZ files shipped in this packet.

Suggested environment:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r standalone_code/requirements.txt
```

Smoke commands from the packet root:

```bash
python standalone_code/load_packet.py
python standalone_code/fit_mct_lrq.py --output-dir outputs/mct_lrq_demo
python standalone_code/grp_no_l2_exact.py --mode fit-best --output-dir outputs/grp_no_l2_exact
```

`fit_mct_lrq.py` is a compact, readable implementation of the MCT-LRQ family:

```text
L(w,N,D) = E_LRQ(w)
         + A ((N/N0)^(-alpha) - 1)
         + B_fam(w) ((D/D0)^(-beta) - 1)
         + C ((N/N0)^(-gamma) (D/D0)^(-delta) - 1)
```

`grp_no_l2_exact.py` is a standalone port of the repo's GRP power-family-penalty no-L2 path. It includes
the exact retained-exposure design, NNLS linear head, calibration objective, raw optimum optimizer, and
full nonlinear retuning procedure:

```bash
python standalone_code/grp_no_l2_exact.py --mode retune --method Powell --coarse-top-k 3
```
"""


STANDALONE_REQUIREMENTS = """numpy>=1.26
pandas>=2.0
scipy>=1.11
scikit-learn>=1.4
"""


STANDALONE_LOAD_PACKET = r'''#!/usr/bin/env python3
"""Load and validate the core tables in the collaborator scaling packet.

This file intentionally has no Marin imports. Run from the packet root:

    python standalone_code/load_packet.py
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


PACKET_ROOT = Path(__file__).resolve().parents[1]


def phase_domains(columns: list[str]) -> list[str]:
    domains = sorted(c.removeprefix("phase_0_") for c in columns if c.startswith("phase_0_"))
    missing = [domain for domain in domains if f"phase_1_{domain}" not in columns]
    if missing:
        raise ValueError(f"Missing phase_1 columns for {len(missing)} domains: {missing[:5]}")
    if not domains:
        raise ValueError("No phase_0_* columns found")
    return domains


def normalized_phase_arrays(df: pd.DataFrame, domains: list[str]) -> tuple[np.ndarray, np.ndarray]:
    w0 = df[[f"phase_0_{domain}" for domain in domains]].to_numpy(dtype=float)
    w1 = df[[f"phase_1_{domain}" for domain in domains]].to_numpy(dtype=float)
    for name, weights in (("phase_0", w0), ("phase_1", w1)):
        if np.any(~np.isfinite(weights)):
            raise ValueError(f"{name} contains non-finite weights")
        if np.any(weights < -1e-12):
            raise ValueError(f"{name} contains negative weights")
        row_sums = weights.sum(axis=1)
        if np.any(row_sums <= 0):
            raise ValueError(f"{name} contains a row with non-positive mass")
        weights /= row_sums[:, None]
    return w0, w1


def summarize_table(path: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    domains = phase_domains(list(df.columns))
    w0, w1 = normalized_phase_arrays(df, domains)
    return {
        "path": str(path.relative_to(PACKET_ROOT)),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "domains": int(len(domains)),
        "phase_0_sum_min": float(w0.sum(axis=1).min()),
        "phase_0_sum_max": float(w0.sum(axis=1).max()),
        "phase_1_sum_min": float(w1.sum(axis=1).min()),
        "phase_1_sum_max": float(w1.sum(axis=1).max()),
        "scale_counts": df["scale_display_label"].value_counts(dropna=False).to_dict()
        if "scale_display_label" in df
        else df.get("scale", pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),
    }


def main() -> None:
    tables = [
        PACKET_ROOT / "data" / "analysis_dataset" / "nd_scale_runs.csv",
        PACKET_ROOT / "data" / "raw_metric_matrix_300m" / "raw_metric_matrix_300m.csv",
        PACKET_ROOT / "data" / "raw_metric_matrix_300m" / "raw_metric_matrix_300m_with_noise.csv",
    ]
    summaries = [summarize_table(path) for path in tables if path.exists()]

    npz_path = PACKET_ROOT / "data" / "analysis_dataset" / "nd_scale_packet.npz"
    if npz_path.exists():
        packet = np.load(npz_path, allow_pickle=True)
        summaries.append(
            {
                "path": str(npz_path.relative_to(PACKET_ROOT)),
                "arrays": {key: list(packet[key].shape) for key in packet.files},
            }
        )

    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
'''


STANDALONE_FIT_MCT_LRQ = r'''#!/usr/bin/env python3
"""Fit a compact MCT-LRQ-style joint mixture/scale law from packet CSVs.

This is a self-contained reference implementation. It is intentionally smaller
than the internal research scripts, but preserves the important structure:

    L(w,N,D) = E_LRQ(w)
             + A ((N/N0)^(-alpha) - 1)
             + B_fam(w) ((D/D0)^(-beta) - 1)
             + C ((N/N0)^(-gamma) (D/D0)^(-delta) - 1)

At N=N0 and D=D0, this reduces exactly to the learned mixture regression
E_LRQ(w). For a fixed mixture w, it reduces to a low-dimensional scaling law.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


PACKET_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = PACKET_ROOT / "data" / "analysis_dataset" / "nd_scale_runs.csv"
DEFAULT_TARGET = "eval/uncheatable_eval/bpb"


@dataclass(frozen=True)
class ScaleConstants:
    n0: float = 102_648_576.0
    d0: float = 5_999_951_872.0
    alpha: float = 0.154791
    beta: float = 0.146425
    gamma: float = 0.014295
    delta: float = 1.063376


def phase_domains(columns: list[str]) -> list[str]:
    domains = sorted(c.removeprefix("phase_0_") for c in columns if c.startswith("phase_0_"))
    if not domains:
        raise ValueError("No phase_0_* columns found")
    missing = [domain for domain in domains if f"phase_1_{domain}" not in columns]
    if missing:
        raise ValueError(f"Missing phase_1 columns for {len(missing)} domains")
    return domains


def normalized_phase_arrays(df: pd.DataFrame, domains: list[str]) -> tuple[np.ndarray, np.ndarray]:
    w0 = df[[f"phase_0_{domain}" for domain in domains]].to_numpy(dtype=float)
    w1 = df[[f"phase_1_{domain}" for domain in domains]].to_numpy(dtype=float)
    for name, weights in (("phase_0", w0), ("phase_1", w1)):
        row_sums = weights.sum(axis=1)
        if np.any(row_sums <= 0):
            raise ValueError(f"{name} contains non-positive row mass")
        weights /= row_sums[:, None]
    return w0, w1


def domain_family(domain: str) -> str:
    low = domain.lower()
    if "synth_math" in low or "finemath" in low:
        return "math"
    if "synth_code" in low or "stack" in low:
        return "code"
    if "synth_thinking" in low or "synth_qa" in low or "instruction" in low:
        return "synthetic_reasoning"
    if "arxiv" in low or "stem" in low or "science_math" in low:
        return "stem"
    if "wikipedia" in low:
        return "wiki"
    if "olmocr" in low:
        return "pdf"
    if "common_crawl" in low or "dolma3_cc/" in low:
        return "web"
    return "other"


def family_matrix(domains: list[str]) -> tuple[list[str], np.ndarray]:
    families = sorted({domain_family(domain) for domain in domains})
    matrix = np.zeros((len(domains), len(families)), dtype=float)
    index = {family: i for i, family in enumerate(families)}
    for row, domain in enumerate(domains):
        matrix[row, index[domain_family(domain)]] = 1.0
    return families, matrix


def lrq_features(w0: np.ndarray, w1: np.ndarray, domains: list[str]) -> tuple[list[str], np.ndarray]:
    exposure = 0.8 * w0 + 0.2 * w1
    shift = w1 - w0
    families, fam_map = family_matrix(domains)
    fam_exposure = exposure @ fam_map
    fam_shift = shift @ fam_map
    entropy0 = -(w0 * np.log(np.clip(w0, 1e-12, None))).sum(axis=1, keepdims=True)
    entropy1 = -(w1 * np.log(np.clip(w1, 1e-12, None))).sum(axis=1, keepdims=True)
    phase_tv = 0.5 * np.abs(w1 - w0).sum(axis=1, keepdims=True)

    blocks = [
        np.ones((w0.shape[0], 1)),
        exposure,
        np.sqrt(np.clip(exposure, 0.0, None)),
        np.log1p(20.0 * exposure),
        shift,
        fam_exposure,
        fam_shift,
        entropy0,
        entropy1,
        phase_tv,
    ]
    names = (
        ["intercept"]
        + [f"exposure:{domain}" for domain in domains]
        + [f"sqrt_exposure:{domain}" for domain in domains]
        + [f"log_exposure:{domain}" for domain in domains]
        + [f"phase_shift:{domain}" for domain in domains]
        + [f"family_exposure:{family}" for family in families]
        + [f"family_shift:{family}" for family in families]
        + ["entropy_phase0", "entropy_phase1", "phase_tv"]
    )
    return names, np.hstack(blocks)


def ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    mean[0] = 0.0
    std[0] = 1.0
    std = np.where(std < 1e-12, 1.0, std)
    return (x - mean) / std, mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = pred - y
    return {
        "rows": float(len(y)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "pearson": float(np.corrcoef(y, pred)[0, 1]) if len(y) > 1 else float("nan"),
        "spearman": rank_corr(y, pred) if len(y) > 1 else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--output-dir", type=Path, default=PACKET_ROOT / "outputs" / "mct_lrq_demo")
    parser.add_argument("--ridge-anchor", type=float, default=1e-4)
    parser.add_argument("--ridge-scale", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.table)
    if args.target not in df:
        raise ValueError(f"Target column not found: {args.target}")
    required = ["non_embedding_params", "target_budget"]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    domains = phase_domains(list(df.columns))
    w0_all, w1_all = normalized_phase_arrays(df, domains)
    y_all = pd.to_numeric(df[args.target], errors="coerce").to_numpy(dtype=float)
    n_all = pd.to_numeric(df["non_embedding_params"], errors="coerce").to_numpy(dtype=float)
    d_all = pd.to_numeric(df["target_budget"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y_all) & np.isfinite(n_all) & np.isfinite(d_all) & (n_all > 0) & (d_all > 0)
    df = df.loc[valid].reset_index(drop=True)
    w0 = w0_all[valid]
    w1 = w1_all[valid]
    y = y_all[valid]
    n = n_all[valid]
    d = d_all[valid]

    constants = ScaleConstants()
    names, x_anchor = lrq_features(w0, w1, domains)
    anchor_mask = (np.abs(n / constants.n0 - 1.0) < 0.03) & (np.abs(d / constants.d0 - 1.0) < 0.03)
    if anchor_mask.sum() < max(20, x_anchor.shape[1] // 4):
        anchor_mask = np.ones(len(y), dtype=bool)

    x_anchor_std, x_mean, x_std = standardize_fit(x_anchor[anchor_mask])
    theta_anchor_std = ridge_fit(x_anchor_std, y[anchor_mask], args.ridge_anchor)
    theta_anchor = theta_anchor_std / x_std
    theta_anchor[0] = theta_anchor_std[0] - np.dot(x_mean / x_std, theta_anchor_std)
    anchor_pred = x_anchor @ theta_anchor

    families, fam_map = family_matrix(domains)
    family_share = (0.8 * w0 + 0.2 * w1) @ fam_map
    n_term = (n / constants.n0) ** (-constants.alpha) - 1.0
    d_term = (d / constants.d0) ** (-constants.beta) - 1.0
    cross_term = (n / constants.n0) ** (-constants.gamma) * (d / constants.d0) ** (-constants.delta) - 1.0
    z_scale = np.column_stack([n_term, d_term[:, None] * family_share, cross_term])
    scale_names = ["A_global_N"] + [f"B_family_D:{family}" for family in families] + ["C_cross_ND"]

    z_std, z_mean, z_scale_std = standardize_fit(np.column_stack([np.ones(len(y)), z_scale]))
    residual = y - anchor_pred
    theta_scale_std = ridge_fit(z_std, residual, args.ridge_scale)
    theta_scale = theta_scale_std[1:] / z_scale_std[1:]
    scale_intercept = theta_scale_std[0] - np.dot(z_mean[1:] / z_scale_std[1:], theta_scale_std[1:])
    pred = anchor_pred + scale_intercept + z_scale @ theta_scale

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pred_df = df[
        [
            column
            for column in [
                "registry_run_key",
                "run_name",
                "scale",
                "scale_display_label",
                "target_budget_multiplier",
                "fit_role",
            ]
            if column in df
        ]
    ].copy()
    pred_df["actual"] = y
    pred_df["predicted"] = pred
    pred_df["residual"] = pred - y
    pred_df.to_csv(out / "predictions.csv", index=False)

    summary: dict[str, object] = {
        "target": args.target,
        "constants": asdict(constants),
        "rows": int(len(y)),
        "domains": int(len(domains)),
        "families": families,
        "anchor_rows": int(anchor_mask.sum()),
        "overall": metrics(y, pred),
        "scale_coefficients": dict(zip(scale_names, map(float, theta_scale), strict=True)),
    }
    if "scale_display_label" in df:
        summary["by_scale"] = {
            str(scale): metrics(y[group.index.to_numpy()], pred[group.index.to_numpy()])
            for scale, group in df.groupby("scale_display_label")
        }
    if "target_budget_multiplier" in df:
        summary["by_target_budget_multiplier"] = {
            str(mult): metrics(y[group.index.to_numpy()], pred[group.index.to_numpy()])
            for mult, group in df.groupby("target_budget_multiplier")
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    pd.DataFrame({"feature": names, "coefficient": theta_anchor}).to_csv(out / "anchor_coefficients.csv", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
'''


STANDALONE_FIT_GRP_FIXED_SCALE = r'''#!/usr/bin/env python3
"""Fit a fixed-scale GRP-style mixture surrogate on the 300M/6B raw metric matrix.

This script is self-contained and does not import Marin. It fits ridge models on
generic retained-exposure/family/phase features, reports cross-validated fit
metrics, and prints the best observed mixture under the fitted surrogate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PACKET_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = PACKET_ROOT / "data" / "raw_metric_matrix_300m" / "raw_metric_matrix_300m.csv"
DEFAULT_TARGET = "eval/uncheatable_eval/bpb"


def phase_domains(columns: list[str]) -> list[str]:
    domains = sorted(c.removeprefix("phase_0_") for c in columns if c.startswith("phase_0_"))
    if not domains:
        raise ValueError("No phase_0_* columns found")
    missing = [domain for domain in domains if f"phase_1_{domain}" not in columns]
    if missing:
        raise ValueError(f"Missing phase_1 columns for {len(missing)} domains")
    return domains


def normalized_phase_arrays(df: pd.DataFrame, domains: list[str]) -> tuple[np.ndarray, np.ndarray]:
    w0 = df[[f"phase_0_{domain}" for domain in domains]].to_numpy(dtype=float)
    w1 = df[[f"phase_1_{domain}" for domain in domains]].to_numpy(dtype=float)
    for weights in (w0, w1):
        row_sums = weights.sum(axis=1)
        if np.any(row_sums <= 0):
            raise ValueError("Phase weights contain non-positive row mass")
        weights /= row_sums[:, None]
    return w0, w1


def domain_family(domain: str) -> str:
    low = domain.lower()
    if "synth_math" in low or "finemath" in low:
        return "math"
    if "synth_code" in low or "stack" in low:
        return "code"
    if "synth_thinking" in low or "synth_qa" in low or "instruction" in low:
        return "synthetic_reasoning"
    if "arxiv" in low or "stem" in low or "science_math" in low:
        return "stem"
    if "wikipedia" in low:
        return "wiki"
    if "olmocr" in low:
        return "pdf"
    if "common_crawl" in low or "dolma3_cc/" in low:
        return "web"
    return "other"


def family_matrix(domains: list[str]) -> tuple[list[str], np.ndarray]:
    families = sorted({domain_family(domain) for domain in domains})
    matrix = np.zeros((len(domains), len(families)), dtype=float)
    index = {family: i for i, family in enumerate(families)}
    for row, domain in enumerate(domains):
        matrix[row, index[domain_family(domain)]] = 1.0
    return families, matrix


def design_matrix(w0: np.ndarray, w1: np.ndarray, domains: list[str]) -> tuple[list[str], np.ndarray]:
    exposure = 0.8 * w0 + 0.2 * w1
    retention = np.minimum(w0, w1)
    phase_shift = w1 - w0
    phase_tv = 0.5 * np.abs(phase_shift).sum(axis=1, keepdims=True)
    entropy0 = -(w0 * np.log(np.clip(w0, 1e-12, None))).sum(axis=1, keepdims=True)
    entropy1 = -(w1 * np.log(np.clip(w1, 1e-12, None))).sum(axis=1, keepdims=True)
    families, fam_map = family_matrix(domains)
    fam_exposure = exposure @ fam_map
    fam_shift = phase_shift @ fam_map
    blocks = [
        np.ones((w0.shape[0], 1)),
        exposure,
        np.sqrt(np.clip(exposure, 0.0, None)),
        np.log1p(20.0 * exposure),
        retention,
        phase_shift,
        fam_exposure,
        fam_shift,
        phase_tv,
        entropy0,
        entropy1,
    ]
    names = (
        ["intercept"]
        + [f"exposure:{domain}" for domain in domains]
        + [f"sqrt_exposure:{domain}" for domain in domains]
        + [f"log_exposure:{domain}" for domain in domains]
        + [f"retention:{domain}" for domain in domains]
        + [f"phase_shift:{domain}" for domain in domains]
        + [f"family_exposure:{family}" for family in families]
        + [f"family_shift:{family}" for family in families]
        + ["phase_tv", "entropy_phase0", "entropy_phase1"]
    )
    return names, np.hstack(blocks)


def standardize_train(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    mean[0] = 0.0
    std[0] = 1.0
    std = np.where(std < 1e-12, 1.0, std)
    return (x - mean) / std, mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def kfold_indices(n: int, folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return [fold for fold in np.array_split(order, folds) if len(fold)]


def fit_oof(x: np.ndarray, y: np.ndarray, ridge: float, folds: int, seed: int) -> np.ndarray:
    pred = np.empty_like(y, dtype=float)
    for test_idx in kfold_indices(len(y), folds, seed):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False
        x_train, mean, std = standardize_train(x[train_mask])
        theta = ridge_fit(x_train, y[train_mask], ridge)
        pred[test_idx] = standardize_apply(x[test_idx], mean, std) @ theta
    return pred


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = pred - y
    order = np.argsort(pred)
    top8_actual = y[order[:8]]
    best_actual = float(np.min(y))
    return {
        "rows": float(len(y)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "pearson": float(np.corrcoef(y, pred)[0, 1]) if len(y) > 1 else float("nan"),
        "spearman": rank_corr(y, pred) if len(y) > 1 else float("nan"),
        "predicted_top1_actual": float(y[order[0]]),
        "observed_best_actual": best_actual,
        "top1_regret": float(y[order[0]] - best_actual),
        "top8_mean_regret": float(top8_actual.mean() - best_actual),
    }


def top_domain_table(
    df: pd.DataFrame,
    domains: list[str],
    w0: np.ndarray,
    w1: np.ndarray,
    pred: np.ndarray,
    output_path: Path,
    top_k: int,
) -> None:
    best = int(np.argmin(pred))
    exposure = 0.8 * w0[best] + 0.2 * w1[best]
    rows = []
    for idx in np.argsort(-exposure)[:top_k]:
        rows.append(
            {
                "run_name": df.iloc[best].get("run_name", ""),
                "predicted_target": float(pred[best]),
                "domain": domains[idx],
                "phase_0_weight": float(w0[best, idx]),
                "phase_1_weight": float(w1[best, idx]),
                "exposure_80_20_weight": float(exposure[idx]),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--output-dir", type=Path, default=PACKET_ROOT / "outputs" / "grp_300m_demo")
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.table)
    if args.target not in df:
        raise ValueError(f"Target column not found: {args.target}")
    domains = phase_domains(list(df.columns))
    w0_all, w1_all = normalized_phase_arrays(df, domains)
    y_all = pd.to_numeric(df[args.target], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y_all)
    df = df.loc[valid].reset_index(drop=True)
    w0 = w0_all[valid]
    w1 = w1_all[valid]
    y = y_all[valid]

    names, x = design_matrix(w0, w1, domains)
    oof = fit_oof(x, y, args.ridge, args.folds, args.seed)
    x_std, mean, std = standardize_train(x)
    theta_std = ridge_fit(x_std, y, args.ridge)
    pred = x_std @ theta_std
    theta = theta_std / std
    theta[0] = theta_std[0] - np.dot(mean / std, theta_std)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pred_df = df[[column for column in ["run_name", "run_id", "scale", "row_kind"] if column in df]].copy()
    pred_df["actual"] = y
    pred_df["predicted_in_sample"] = pred
    pred_df["predicted_oof"] = oof
    pred_df.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame({"feature": names, "coefficient": theta}).to_csv(out / "coefficients.csv", index=False)
    top_domain_table(df, domains, w0, w1, pred, out / "best_predicted_observed_top_domains.csv", top_k=20)
    summary = {
        "target": args.target,
        "rows": int(len(y)),
        "domains": int(len(domains)),
        "features": int(x.shape[1]),
        "ridge": args.ridge,
        "in_sample": metrics(y, pred),
        "oof": metrics(y, oof),
        "best_predicted_observed_run": str(df.iloc[int(np.argmin(pred))].get("run_name", "")),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
'''


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--archive-path", type=Path, default=DEFAULT_ARCHIVE_PATH)
    parser.add_argument("--no-archive", action="store_true")
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(spec: CopySpec, packet_dir: Path) -> dict[str, object] | None:
    if not spec.source.exists():
        if spec.required:
            raise FileNotFoundError(f"Missing required packet source: {spec.source}")
        return None
    destination = packet_dir / spec.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(spec.source, destination)
    try:
        source_label = str(spec.source.relative_to(SCRIPT_DIR))
    except ValueError:
        source_label = str(spec.source)
    return {
        "source": source_label,
        "path": str(spec.destination),
        "bytes": destination.stat().st_size,
        "sha256": _file_sha256(destination),
    }


def _copy_tree_files(source_root: Path, destination_root: Path, packet_dir: Path) -> list[dict[str, object]]:
    if not source_root.exists():
        raise FileNotFoundError(f"Missing required packet source tree: {source_root}")
    copied: list[dict[str, object]] = []
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        if "__pycache__" in source.parts or source.suffix == ".pyc" or source.name == ".DS_Store":
            continue
        relative = source.relative_to(source_root)
        copied_item = _copy_file(CopySpec(source, destination_root / relative), packet_dir)
        if copied_item is not None:
            copied.append(copied_item)
    return copied


def _write_generated_file(packet_dir: Path, relative_path: Path, content: str) -> dict[str, object]:
    destination = packet_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return {
        "source": "generated",
        "path": str(relative_path),
        "bytes": destination.stat().st_size,
        "sha256": _file_sha256(destination),
    }


def _write_standalone_code(packet_dir: Path) -> list[dict[str, object]]:
    return [
        _write_generated_file(packet_dir, Path("standalone_code/README.md"), STANDALONE_README),
        _write_generated_file(packet_dir, Path("standalone_code/requirements.txt"), STANDALONE_REQUIREMENTS),
        _write_generated_file(packet_dir, Path("standalone_code/load_packet.py"), STANDALONE_LOAD_PACKET),
        _write_generated_file(packet_dir, Path("standalone_code/fit_mct_lrq.py"), STANDALONE_FIT_MCT_LRQ),
    ]


def _json_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _packet_copy_specs() -> list[CopySpec]:
    return [
        CopySpec(Path("analysis_dataset/nd_scale_runs.csv"), Path("data/analysis_dataset/nd_scale_runs.csv")),
        CopySpec(Path("analysis_dataset/nd_scale_packet.npz"), Path("data/analysis_dataset/nd_scale_packet.npz")),
        CopySpec(Path("analysis_dataset/summary.json"), Path("data/analysis_dataset/summary.json")),
        CopySpec(Path("run_registry/logical_runs.csv"), Path("data/run_registry/logical_runs.csv")),
        CopySpec(
            Path("run_registry/strong_tier_perplexity_ready.csv"),
            Path("data/run_registry/strong_tier_perplexity_ready.csv"),
        ),
        CopySpec(Path("run_registry/summary.json"), Path("data/run_registry/summary.json")),
        CopySpec(Path("metric_registry/runs.csv"), Path("data/metric_registry/runs.csv")),
        CopySpec(Path("metric_registry/coverage.csv"), Path("data/metric_registry/coverage.csv")),
        CopySpec(Path("metric_registry/metrics_wide.csv"), Path("data/metric_registry/metrics_wide.csv")),
        CopySpec(Path("metric_registry/summary.json"), Path("data/metric_registry/summary.json")),
        CopySpec(
            Path("metric_registry/raw_metric_matrix_300m/raw_metric_matrix_300m.csv"),
            Path("data/raw_metric_matrix_300m/raw_metric_matrix_300m.csv"),
        ),
        CopySpec(
            Path("metric_registry/raw_metric_matrix_300m/raw_metric_matrix_300m_with_noise.csv"),
            Path("data/raw_metric_matrix_300m/raw_metric_matrix_300m_with_noise.csv"),
        ),
        CopySpec(
            Path("metric_registry/raw_metric_matrix_300m/noise_baseline_run00097_fixed_subset_300m.csv"),
            Path("data/raw_metric_matrix_300m/noise_baseline_run00097_fixed_subset_300m.csv"),
        ),
        CopySpec(
            Path("metric_registry/raw_metric_matrix_300m/noise_baseline_run00097_variable_subset_300m.csv"),
            Path("data/raw_metric_matrix_300m/noise_baseline_run00097_variable_subset_300m.csv"),
        ),
        CopySpec(
            Path("metric_registry/raw_metric_matrix_300m/summary.json"),
            Path("data/raw_metric_matrix_300m/summary.json"),
        ),
        CopySpec(
            Path("metric_registry/raw_metric_matrix_300m/README.md"),
            Path("data/raw_metric_matrix_300m/README.md"),
        ),
        CopySpec(Path("two_phase_many.csv"), Path("data/grp_no_l2/two_phase_many.csv")),
        CopySpec(Path("two_phase_many_epoch_metadata.csv"), Path("data/grp_no_l2/two_phase_many_epoch_metadata.csv")),
        CopySpec(
            Path("grp_penalty_calibration_variants_best.csv"),
            Path("data/grp_no_l2/grp_penalty_calibration_variants_best.csv"),
        ),
        CopySpec(
            Path("grp_power_family_penalty_no_l2_retune_best.csv"),
            Path("data/grp_no_l2/grp_power_family_penalty_no_l2_retune_best.csv"),
        ),
        CopySpec(Path("standalone_code/grp_no_l2_exact.py"), Path("standalone_code/grp_no_l2_exact.py")),
    ]


def _write_readme(packet_dir: Path) -> None:
    analysis_summary = _json_summary(SCRIPT_DIR / "analysis_dataset" / "summary.json")
    metric_summary = _json_summary(SCRIPT_DIR / "metric_registry" / "summary.json")
    raw_summary = _json_summary(SCRIPT_DIR / "metric_registry" / "raw_metric_matrix_300m" / "summary.json")
    run_summary = _json_summary(SCRIPT_DIR / "run_registry" / "summary.json")
    readme = f"""# Collaborator Scaling Data Packet

Generated: {datetime.now(UTC).isoformat()}

This packet contains the refreshed data portion for joint data-mixture and scale modeling, plus small
self-contained reference implementations for fixed-scale GRP-style fitting and the MCT-LRQ joint scale
law. The included scripts do not depend on this branch or the broader Marin repository.

## Data Contract

- `data/analysis_dataset/nd_scale_runs.csv` is the canonical row table for joint scale/mix modeling.
- `data/analysis_dataset/nd_scale_packet.npz` is the packet-compatible array form.
- `model_size` / `model_sizes` mean corrected non-embedding parameter count, not historical nominal labels.
- Historical scale strings such as `300m_6b` are stable IDs only.
- Display labels use corrected non-embedding `N` plus realized target-budget `D`:
  `20M/2.6B`, `60M/1.2B`, `100M/6B`, `340M/10.4B`, `900M/24B`.
- Do not make N-only scaling claims from this data without also showing `D`.

## Included Tables

- `data/analysis_dataset/nd_scale_runs.csv`: canonical scaling-law modeling rows.
- `data/analysis_dataset/nd_scale_packet.npz`: arrays for fitting scripts.
- `data/metric_registry/metrics_wide.csv`: refreshed wide metric registry view.
- `data/raw_metric_matrix_300m/raw_metric_matrix_300m.csv`: 242-row 300M/6B qsplit-core metric matrix with
  `phase_0_*`, `phase_1_*`, and `exposure_80_20_*` mixture columns.
- `data/raw_metric_matrix_300m/raw_metric_matrix_300m_with_noise.csv`: the same signal rows plus any
  completed `run_00097` noise rows.
- `data/raw_metric_matrix_300m/noise_baseline_run00097_fixed_subset_300m.csv`: trainer-seed noise with
  the simulated-epoch subset fixed to `run_00097`.
- `data/raw_metric_matrix_300m/noise_baseline_run00097_variable_subset_300m.csv`: trainer-seed noise with
  the simulated-epoch subset left variable, matching the original swarm sampling more closely.
- `data/grp_no_l2/two_phase_many.csv`: original 60M/1.2B GRP fit panel.
- `data/grp_no_l2/two_phase_many_epoch_metadata.csv`: epoch multipliers used by the exact GRP fit.
- `data/grp_no_l2/grp_penalty_calibration_variants_best.csv`: regularized GRP best row used to seed the
  no-L2 retune.
- `data/grp_no_l2/grp_power_family_penalty_no_l2_retune_best.csv`: repo-generated no-L2 best row for
  reproducibility checks.
- `data/run_registry/logical_runs.csv`: refreshed operational provenance table.

## Standalone Code

- `standalone_code/load_packet.py`: validates the shipped CSV/NPZ files and reconstructs phase weights.
- `standalone_code/grp_no_l2_exact.py`: exact standalone port of the repo's GRP power-family-penalty
  no-L2 path, including nonlinear retuning.
- `standalone_code/fit_mct_lrq.py`: compact MCT-LRQ-style joint mixture/scale law using only packet CSVs.
- `standalone_code/requirements.txt`: minimal Python package requirements.

Run from the packet root:

```bash
python standalone_code/load_packet.py
python standalone_code/grp_no_l2_exact.py --mode fit-best --output-dir outputs/grp_no_l2_exact
python standalone_code/grp_no_l2_exact.py --mode retune --method Powell --coarse-top-k 3
python standalone_code/fit_mct_lrq.py --output-dir outputs/mct_lrq_demo
```

The packet still includes `reference_outputs/joint_model_refreshed_20260426/` reports, CSVs, and model
JSONs for comparison, but it intentionally does not copy branch-dependent Marin experiment scripts.

## Refreshed Counts

```json
{json.dumps({
    "analysis_dataset": analysis_summary.get("rows", analysis_summary),
    "metric_registry": {
        "run_count": metric_summary.get("run_count"),
        "canonical_metric_count": metric_summary.get("canonical_metric_count"),
        "metric_fact_count_canonical": metric_summary.get("metric_fact_count_canonical"),
        "source_count": metric_summary.get("source_count"),
        "conflict_count": metric_summary.get("conflict_count"),
    },
    "raw_metric_matrix_300m": raw_summary.get("canonical", raw_summary),
    "run_registry": {
        "logical_run_count": run_summary.get("logical_run_count"),
        "logical_runs_by_status": run_summary.get("logical_runs_by_status"),
        "strong_tier_perplexity_ready_rows": run_summary.get("strong_tier_perplexity_ready_rows"),
    },
}, indent=2, sort_keys=True)}
```

## Notes

The metric registry may report many source conflicts because local historical CSVs, checkpoint metrics,
and collected eval outputs overlap. The `metrics_wide.csv` table uses the registry's canonical source
priority resolution.
"""
    (packet_dir / "README.md").write_text(readme, encoding="utf-8")


def _write_manifest(packet_dir: Path, copied: list[dict[str, object]]) -> None:
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "packet_dir": str(packet_dir),
        "file_count": len(copied),
        "total_bytes": sum(int(item["bytes"]) for item in copied),
        "files": copied,
    }
    (packet_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_archive(packet_dir: Path, archive_path: Path) -> Path:
    for ds_store in packet_dir.rglob(".DS_Store"):
        ds_store.unlink()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()
    archive_base = archive_path.with_suffix("")
    created = Path(shutil.make_archive(str(archive_base), "zip", root_dir=packet_dir.parent, base_dir=packet_dir.name))
    if created != archive_path:
        created.replace(archive_path)
    return archive_path


def main() -> None:
    args = _parse_args()
    packet_dir = args.output_dir.resolve()
    if packet_dir.exists():
        shutil.rmtree(packet_dir)
    packet_dir.mkdir(parents=True)

    copied: list[dict[str, object]] = []
    for spec in _packet_copy_specs():
        copied_item = _copy_file(
            CopySpec(SCRIPT_DIR / spec.source, spec.destination, required=spec.required),
            packet_dir,
        )
        if copied_item is not None:
            copied.append(copied_item)

    copied.extend(_write_standalone_code(packet_dir))
    for output_name in ("mct_lrq_refresh", "mct_lrq_no_barrier_canonical"):
        source_root = SCRIPT_DIR / "reference_outputs" / "joint_model_refreshed_20260426" / output_name
        for child in ("REPORT.md", "summary.json", "MANIFEST.json"):
            copied_item = _copy_file(
                CopySpec(
                    source_root / child,
                    Path("reference_outputs/joint_model_refreshed_20260426") / output_name / child,
                    required=False,
                ),
                packet_dir,
            )
            if copied_item is not None:
                copied.append(copied_item)
        for subdir in ("csv", "models"):
            path = source_root / subdir
            if path.exists():
                copied.extend(
                    _copy_tree_files(
                        path,
                        Path("reference_outputs/joint_model_refreshed_20260426") / output_name / subdir,
                        packet_dir,
                    )
                )

    _write_readme(packet_dir)
    copied.append(
        {
            "source": "generated",
            "path": "README.md",
            "bytes": (packet_dir / "README.md").stat().st_size,
            "sha256": _file_sha256(packet_dir / "README.md"),
        }
    )
    _write_manifest(packet_dir, copied)
    if not args.no_archive:
        archive_path = _make_archive(packet_dir, args.archive_path.resolve())
        print(f"Wrote {packet_dir}")
        print(f"Wrote {archive_path}")
    else:
        print(f"Wrote {packet_dir}")


if __name__ == "__main__":
    main()
