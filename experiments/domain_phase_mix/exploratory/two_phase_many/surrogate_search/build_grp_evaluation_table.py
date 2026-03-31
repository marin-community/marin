# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""Build the GRP evaluation table for the many-domain Uncheatable-BPB slide.

Default behavior is intentionally reproducible and still reasonably cheap:
- GRP is evaluated in-repo from the fixed tuned nonlinear parameters.
- Olmix uses the restored real-epoch packet train row plus the cached
  cross-validation summary used in the follow-up surrogate packet.
- CES gets a stable live in-repo train refit plus the cached cross-validation
  summary from the follow-up surrogate packet.
- DS-RE-CEQ gets a fast live in-repo train refit and 5-fold OOF CV refit
  through the relaxed generic DS-RE-CEQ solver.

Outputs:
  - Full metrics CSV with train/CV metrics, provenance, and notes.
  - Slide-ready CSV with the presentation columns.
  - A LaTeX row fragment matching the presentation table.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.dsre_ceq_tools import fit_dsre_ceq_artifacts
from experiments.domain_phase_mix.exploratory.general_scaling_models import DatasetSpec as GeneralDatasetSpec
from experiments.domain_phase_mix.exploratory.scaling_models import fit_ces as stable_fit_ces
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GENERIC_FAMILY_NAMES,
    TUNED_GENERIC_FAMILY_PARAMS,
    GenericFamilyRetainedTotalSurrogate,
    load_generic_family_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    MANY_DOMAIN_TARGET,
    regression_metrics,
)

SCRIPT_DIR = Path(__file__).resolve().parent
FULL_OUTPUT_CSV = SCRIPT_DIR / "grp_evaluation_table_full.csv"
SLIDE_OUTPUT_CSV = SCRIPT_DIR / "grp_evaluation_table_slide.csv"
LATEX_OUTPUT = SCRIPT_DIR / "grp_evaluation_table_rows.tex"
EXISTING_FULL_OUTPUT_CSV = SCRIPT_DIR / "grp_evaluation_table_full.csv"

DEFAULT_PACKET_ROOT = Path("/Users/calvinxu/Downloads/chatgpt_5_4_packet_real_epochs_uncheatable_bpb")
DEFAULT_CACHED_CV_CSV = Path(
    "/Users/calvinxu/Downloads/CCGlobalPremium-RetainedTotal/many_domain_reference_subset_transfer.csv"
)
DEFAULT_DSRE_CV_SOURCE = SCRIPT_DIR / "power_ridge_single.md"

MODEL_ORDER = (
    "GRP",
    "GRP w/o family signals",
    "GRP w/o retention",
    "GRP w/o retention or family signals",
    "GRP w/o quality splits",
    "GRP w/o quality splits or family signals",
    "Olmix loglinear",
    "CES",
    "DS-RE-CEQ",
)
SHAPE_PARAMETER_COUNT = len(TUNED_GENERIC_FAMILY_PARAMS)


@dataclass(frozen=True)
class SourceBundle:
    """External cached sources used to assemble the table."""

    packet_root: Path
    cached_cv_csv: Path
    dsre_cv_md: Path

    @property
    def packet_train_csv(self) -> Path:
        return self.packet_root / "context" / "current_fit_metrics_real_epochs.csv"

    @property
    def packet_code(self) -> Path:
        return self.packet_root / "code" / "reference_surrogates_real_epochs.py"

    @property
    def packet_swarm_csv(self) -> Path:
        return self.packet_root / "data" / "two_phase_many_original_swarm.csv"

    @property
    def packet_epoch_metadata_csv(self) -> Path:
        return self.packet_root / "data" / "two_phase_many_epoch_metadata.csv"


def _clean_float(token: str) -> float:
    cleaned = token.strip().strip("$").replace("\\mathbf{", "").replace("}", "")
    cleaned = cleaned.replace("{\\sim}", "")
    if "±" in cleaned:
        cleaned = cleaned.split("±", 1)[0].strip()
    if cleaned.startswith("~"):
        cleaned = cleaned[1:]
    return float(cleaned)


def _display_float(value: float | None, fmt: str) -> str:
    if value is None or pd.isna(value):
        return "--"
    return format(float(value), fmt)


def _run_name_column(frame: pd.DataFrame) -> str:
    if "candidate_run_name" in frame.columns:
        return "candidate_run_name"
    if "run_name" in frame.columns:
        return "run_name"
    raise ValueError("No run-name column found in frame")


def _target_variance(y: np.ndarray) -> float:
    return float(np.mean((y - np.mean(y)) ** 2))


def _rmse_from_r2(r2: float, variance: float) -> float:
    return float(math.sqrt(max(0.0, (1.0 - float(r2)) * variance)))


def _grp_param_count(packet, model: GenericFamilyRetainedTotalSurrogate) -> dict[str, int]:
    if model.coef_ is None:
        raise RuntimeError("GRP model must be fit before counting parameters")
    linear = len(model.coef_)
    return {
        "n_params": linear + 1 + SHAPE_PARAMETER_COUNT,
        "linear_coefficients": linear,
        "intercept_params": 1,
        "shape_parameters": SHAPE_PARAMETER_COUNT,
    }


def compute_grp_metrics(
    *,
    cv_seed: int,
    model_name: str = "GRP",
    family_totals: tuple[str, ...] = GENERIC_FAMILY_NAMES,
    params: dict[str, float] | None = None,
    active_shape_parameters: int = SHAPE_PARAMETER_COUNT,
    pair_cc_domains: bool = True,
    notes: str | None = None,
) -> dict[str, Any]:
    """Compute GRP-family train and CV metrics in-repo."""
    packet = load_generic_family_packet(target=MANY_DOMAIN_TARGET)
    weights = packet.base.w
    y = packet.base.y
    frame = packet.base.frame
    name_col = packet.base.name_col

    model_params = dict(TUNED_GENERIC_FAMILY_PARAMS if params is None else params)
    model = GenericFamilyRetainedTotalSurrogate(
        packet,
        params=model_params,
        family_totals=family_totals,
        quality_discount=True,
        pair_cc_domains=pair_cc_domains,
    ).fit(weights, y)
    train_pred = model.predict(weights)
    train = regression_metrics(frame, name_col, y, train_pred)
    counts = _grp_param_count(packet, model)
    counts["n_params"] = counts["linear_coefficients"] + counts["intercept_params"] + active_shape_parameters

    kf = KFold(n_splits=5, shuffle=True, random_state=cv_seed)
    oof = np.zeros_like(y, dtype=float)
    fold_regrets: list[float] = []
    for _fold_idx, (tr, te) in enumerate(kf.split(weights)):
        fold_model = GenericFamilyRetainedTotalSurrogate(
            packet,
            params=model_params,
            family_totals=family_totals,
            quality_discount=True,
            pair_cc_domains=pair_cc_domains,
        ).fit(weights[tr], y[tr])
        pred = fold_model.predict(weights[te])
        oof[te] = pred
        chosen = int(np.argmin(pred))
        fold_regrets.append(float(y[te][chosen] - np.min(y[te])))

    cv = regression_metrics(frame, name_col, y, oof)
    return {
        "model": model_name,
        "status": "ok",
        "n_params": counts["n_params"],
        "train_r2": float(train["r2"]),
        "train_rmse": float(train["rmse"]),
        "train_spearman": float(train["spearman"]),
        "train_regret_at_1": float(train["regret_at_1"]),
        "cv_r2": float(cv["r2"]),
        "cv_rmse": float(cv["rmse"]),
        "cv_spearman": float(cv["spearman"]),
        "cv_regret_at_1": float(cv["regret_at_1"]),
        "cv_foldmean_regret_at_1": float(np.mean(fold_regrets)),
        "source_train": "in_repo_generic_family_followup",
        "source_cv": "in_repo_generic_family_followup",
        "notes": notes or "Computed from fixed tuned GRP parameters with 5-fold CV.",
    }


def _load_train_cache(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "model" not in df.columns:
        raise ValueError(f"Missing model column in {path}")
    return df


def _load_cv_cache(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "model" not in df.columns:
        raise ValueError(f"Missing model column in {path}")
    return df


def _cached_train_row(df: pd.DataFrame, model_name: str) -> pd.Series:
    matches = df.loc[df["model"] == model_name]
    if matches.empty:
        raise ValueError(f"Missing cached train row for {model_name!r}")
    return matches.iloc[0]


def _load_packet_module(sources: SourceBundle):
    spec = importlib.util.spec_from_file_location("grp_packet_reference_surrogates", sources.packet_code)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load packet module from {sources.packet_code}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_packet_frame_and_dataset(sources: SourceBundle):
    packet_module = _load_packet_module(sources)
    frame, dataset = packet_module.load_candidate_summary(
        sources.packet_swarm_csv,
        epoch_metadata_csv=sources.packet_epoch_metadata_csv,
    )
    return packet_module, frame, dataset


def _cached_cv_row(df: pd.DataFrame, model_name: str) -> pd.Series:
    matches = df.loc[df["model"] == model_name]
    if matches.empty:
        raise ValueError(f"Missing cached CV row for {model_name!r}")
    return matches.iloc[0]


def load_cached_baseline_metrics(sources: SourceBundle) -> list[dict[str, Any]]:
    """Load Olmix plus a live-train CES row from canonical sources."""
    train_df = _load_train_cache(sources.packet_train_csv)
    cv_df = _load_cv_cache(sources.cached_cv_csv)
    _, frame, dataset = _load_packet_frame_and_dataset(sources)
    name_col = _run_name_column(frame)

    rows: list[dict[str, Any]] = []
    for display_name, train_name, cv_name in (("Olmix loglinear", "Olmix Loglinear", "Olmix Loglinear"),):
        train_row = _cached_train_row(train_df, train_name)
        cv_row = _cached_cv_row(cv_df, cv_name)
        rows.append(
            {
                "model": display_name,
                "status": str(train_row["status"]),
                "n_params": int(cv_row["n_params"]),
                "train_r2": float(train_row["r2"]),
                "train_rmse": float(train_row["rmse"]),
                "train_spearman": float(train_row["spearman"]),
                "train_regret_at_1": float(train_row["regret_at_1"]),
                "cv_r2": float(cv_row["cv_r2"]),
                "cv_rmse": float(cv_row["cv_rmse"]),
                "cv_spearman": float(cv_row["cv_spearman"]),
                "cv_regret_at_1": float(cv_row["cv_regret_at_1"]),
                "cv_foldmean_regret_at_1": float(cv_row["cv_foldmean_regret_at_1"]),
                "source_train": str(sources.packet_train_csv),
                "source_cv": str(sources.cached_cv_csv),
                "notes": (
                    "Train metrics from the restored real-epoch packet; "
                    "CV metrics from the cached many-domain reference subset "
                    "transfer table."
                ),
            }
        )

    phase_fracs = np.ones(dataset.N, dtype=float) / float(dataset.N)
    ces_design = (dataset.weights * phase_fracs[None, :, None]).reshape(dataset.R, -1)
    ces_predict_fn, _ = stable_fit_ces(ces_design, dataset.y, n_restarts=40, seed=42)
    ces_train_pred = np.asarray(ces_predict_fn(ces_design), dtype=float)
    ces_train = regression_metrics(frame, name_col, dataset.y, ces_train_pred)
    ces_kf = KFold(n_splits=5, shuffle=True, random_state=0)
    ces_oof = np.zeros_like(dataset.y, dtype=float)
    ces_fold_regrets: list[float] = []
    for fold_idx, (tr, te) in enumerate(ces_kf.split(ces_design)):
        fold_predict_fn, _ = stable_fit_ces(ces_design[tr], dataset.y[tr], n_restarts=40, seed=42 + fold_idx)
        fold_pred = np.asarray(fold_predict_fn(ces_design[te]), dtype=float)
        ces_oof[te] = fold_pred
        ces_fold_regrets.append(float(dataset.y[te][np.argmin(fold_pred)] - np.min(dataset.y[te])))
    ces_cv = regression_metrics(frame, name_col, dataset.y, ces_oof)
    rows.append(
        {
            "model": "CES",
            "status": "ok",
            "n_params": 81,
            "train_r2": float(ces_train["r2"]),
            "train_rmse": float(ces_train["rmse"]),
            "train_spearman": float(ces_train["spearman"]),
            "train_regret_at_1": float(ces_train["regret_at_1"]),
            "cv_r2": float(ces_cv["r2"]),
            "cv_rmse": float(ces_cv["rmse"]),
            "cv_spearman": float(ces_cv["spearman"]),
            "cv_regret_at_1": float(ces_cv["regret_at_1"]),
            "cv_foldmean_regret_at_1": float(np.mean(ces_fold_regrets)),
            "source_train": "in_repo_scaling_models.fit_ces",
            "source_cv": "in_repo_scaling_models.fit_ces",
            "notes": (
                "Train and 5-fold OOF CV metrics come from a stable live CES "
                "refit using experiments.domain_phase_mix.exploratory.scaling_models.fit_ces "
                "(n_restarts=40, full-data seed=42, fold seeds=42..46)."
            ),
        }
    )
    return rows


def fit_fast_live_dsre_metrics(
    sources: SourceBundle,
    *,
    cv_seed: int = 0,
) -> dict[str, Any]:
    """Fit fast live DS-RE-CEQ train and 5-fold OOF CV metrics."""
    _module, frame, dataset = _load_packet_frame_and_dataset(sources)
    name_col = _run_name_column(frame)

    def _spec(weights: np.ndarray, y: np.ndarray, name: str) -> GeneralDatasetSpec:
        return GeneralDatasetSpec(
            weights=weights,
            y=y,
            epoch_multipliers=dataset.epoch_multipliers,
            small_domains=list(dataset.small_domains),
            domain_names=list(dataset.domain_names),
            phase_names=list(dataset.phase_names),
            name=name,
        )

    artifacts = fit_dsre_ceq_artifacts(
        _spec(dataset.weights, dataset.y, "two_phase_many_dsre_ceq_live_train"),
        n_restarts=1,
        seed=0,
        maxiter=50,
    )
    pred = np.asarray(artifacts.predict_fn(dataset.weights), dtype=float)
    train = regression_metrics(frame, name_col, dataset.y, pred)

    kf = KFold(n_splits=5, shuffle=True, random_state=cv_seed)
    oof = np.zeros_like(dataset.y, dtype=float)
    fold_regrets: list[float] = []
    for fold_idx, (tr, te) in enumerate(kf.split(dataset.weights)):
        fold_artifacts = fit_dsre_ceq_artifacts(
            _spec(dataset.weights[tr], dataset.y[tr], f"two_phase_many_dsre_ceq_fold_{fold_idx}"),
            n_restarts=1,
            seed=fold_idx,
            maxiter=50,
        )
        fold_pred = np.asarray(fold_artifacts.predict_fn(dataset.weights[te]), dtype=float)
        oof[te] = fold_pred
        fold_regrets.append(float(dataset.y[te][np.argmin(fold_pred)] - np.min(dataset.y[te])))

    cv = regression_metrics(frame, name_col, dataset.y, oof)
    return {
        "status": "ok",
        "n_params": int(artifacts.n_params),
        "train_r2": float(train["r2"]),
        "train_rmse": float(train["rmse"]),
        "train_spearman": float(train["spearman"]),
        "train_regret_at_1": float(train["regret_at_1"]),
        "cv_r2": float(cv["r2"]),
        "cv_rmse": float(cv["rmse"]),
        "cv_spearman": float(cv["spearman"]),
        "cv_regret_at_1": float(cv["regret_at_1"]),
        "cv_foldmean_regret_at_1": float(np.mean(fold_regrets)),
        "source_train": "in_repo_dsre_ceq_tools.fit_dsre_ceq_artifacts",
        "source_cv": "in_repo_dsre_ceq_tools.fit_dsre_ceq_artifacts",
    }


def load_dsre_ceq_metrics(
    *,
    sources: SourceBundle,
    cv_seed: int = 0,
) -> dict[str, Any]:
    """Return a live fast-refit DS-RE-CEQ row."""
    row = fit_fast_live_dsre_metrics(sources, cv_seed=cv_seed)
    row.update(
        {
            "model": "DS-RE-CEQ",
            "notes": (
                "Train and 5-fold OOF CV metrics come from a fast live in-repo "
                "DS-RE-CEQ refit via experiments.domain_phase_mix.exploratory.dsre_ceq_tools "
                "(n_restarts=1, full-data seed=0, fold seeds=0..4, maxiter=50)."
            ),
        }
    )
    return row


def _slide_frame(full: pd.DataFrame) -> pd.DataFrame:
    return full[
        [
            "model",
            "n_params",
            "train_r2",
            "cv_r2",
            "cv_rmse",
            "cv_spearman",
            "cv_regret_at_1",
            "cv_foldmean_regret_at_1",
        ]
    ].copy()


def _existing_row(path: Path, model_name: str) -> dict[str, Any]:
    full = pd.read_csv(path)
    matches = full.loc[full["model"] == model_name]
    if matches.empty:
        raise ValueError(f"Missing existing cached row for {model_name!r} in {path}")
    return dict(matches.iloc[0])


def _latex_rows(slide: pd.DataFrame) -> str:
    lines = []
    for row in slide.itertuples(index=False):
        lines.append(
            f"{row.model} & "
            f"{int(row.n_params)} & "
            f"{_display_float(row.train_r2, '.3f')} & "
            f"{_display_float(row.cv_r2, '.3f')} & "
            f"{_display_float(row.cv_rmse, '.4f')} & "
            f"{_display_float(row.cv_spearman, '.3f')} & "
            f"{_display_float(row.cv_regret_at_1, '.4f')} & "
            f"{_display_float(row.cv_foldmean_regret_at_1, '.4f')} \\\\"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", type=Path, default=DEFAULT_PACKET_ROOT)
    parser.add_argument("--cached-cv-csv", type=Path, default=DEFAULT_CACHED_CV_CSV)
    parser.add_argument("--dsre-cv-md", type=Path, default=DEFAULT_DSRE_CV_SOURCE)
    parser.add_argument("--output-csv", type=Path, default=FULL_OUTPUT_CSV)
    parser.add_argument("--slide-csv", type=Path, default=SLIDE_OUTPUT_CSV)
    parser.add_argument("--latex-output", type=Path, default=LATEX_OUTPUT)
    parser.add_argument("--cv-seed", type=int, default=0)
    args = parser.parse_args()

    sources = SourceBundle(
        packet_root=args.packet_root,
        cached_cv_csv=args.cached_cv_csv,
        dsre_cv_md=args.dsre_cv_md,
    )

    grp_row = compute_grp_metrics(
        cv_seed=args.cv_seed,
        model_name="GRP",
        family_totals=GENERIC_FAMILY_NAMES,
        notes="Computed from fixed tuned GRP parameters with 5-fold CV.",
    )
    grp_ablation_row = compute_grp_metrics(
        cv_seed=args.cv_seed,
        model_name="GRP w/o family signals",
        family_totals=(),
        notes="Computed from the same fixed tuned GRP nonlinear parameters, but with the three family features removed.",
    )
    no_retention_params = dict(TUNED_GENERIC_FAMILY_PARAMS)
    no_retention_params["lam"] = 0.0
    grp_no_retention_row = compute_grp_metrics(
        cv_seed=args.cv_seed,
        model_name="GRP w/o retention",
        family_totals=GENERIC_FAMILY_NAMES,
        params=no_retention_params,
        active_shape_parameters=SHAPE_PARAMETER_COUNT - 1,
        notes=(
            "Computed from the same fixed tuned GRP nonlinear parameters, "
            "but with retention disabled by fixing lambda = 0."
        ),
    )
    grp_no_retention_no_groups_row = compute_grp_metrics(
        cv_seed=args.cv_seed,
        model_name="GRP w/o retention or family signals",
        family_totals=(),
        params=no_retention_params,
        active_shape_parameters=SHAPE_PARAMETER_COUNT - 1,
        notes=(
            "Computed from the same fixed tuned GRP nonlinear parameters, "
            "but with both retention disabled by fixing lambda = 0 and the "
            "three family features removed."
        ),
    )
    grp_no_quality_splits_row = compute_grp_metrics(
        cv_seed=args.cv_seed,
        model_name="GRP w/o quality splits",
        family_totals=GENERIC_FAMILY_NAMES,
        pair_cc_domains=False,
        active_shape_parameters=SHAPE_PARAMETER_COUNT - 1,
        notes=(
            "Computed from the same fixed tuned GRP nonlinear parameters, "
            "but with CC high/low buckets treated as separate domains rather "
            "than paired signals."
        ),
    )
    grp_no_quality_splits_no_groups_row = compute_grp_metrics(
        cv_seed=args.cv_seed,
        model_name="GRP w/o quality splits or family signals",
        family_totals=(),
        pair_cc_domains=False,
        active_shape_parameters=SHAPE_PARAMETER_COUNT - 1,
        notes=(
            "Computed from the same fixed tuned GRP nonlinear parameters, "
            "but with CC high/low buckets treated as separate domains and the "
            "three family features removed."
        ),
    )
    if sources.packet_train_csv.exists():
        cached_rows = load_cached_baseline_metrics(sources)
        dsre_row = load_dsre_ceq_metrics(
            sources=sources,
            cv_seed=args.cv_seed,
        )
    else:
        cached_rows = [
            _existing_row(EXISTING_FULL_OUTPUT_CSV, "Olmix loglinear"),
            _existing_row(EXISTING_FULL_OUTPUT_CSV, "CES"),
        ]
        dsre_row = _existing_row(EXISTING_FULL_OUTPUT_CSV, "DS-RE-CEQ")

    full = pd.DataFrame(
        [
            grp_row,
            grp_ablation_row,
            grp_no_retention_row,
            grp_no_retention_no_groups_row,
            grp_no_quality_splits_row,
            grp_no_quality_splits_no_groups_row,
            *cached_rows,
            dsre_row,
        ]
    )
    full["model"] = pd.Categorical(full["model"], categories=MODEL_ORDER, ordered=True)
    full = full.sort_values("model").reset_index(drop=True)
    slide = _slide_frame(full)

    args.output_csv.write_text(full.to_csv(index=False))
    args.slide_csv.write_text(slide.to_csv(index=False))
    args.latex_output.write_text(_latex_rows(slide))

    print("Full metrics:")
    print(full.to_markdown(index=False))
    print(f"\nWrote {args.output_csv}")
    print(f"Wrote {args.slide_csv}")
    print(f"Wrote {args.latex_output}")


if __name__ == "__main__":
    main()
