# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""Build a GRP evaluation table on alternate many-domain targets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import (
    load_two_phase_many_candidate_summary_spec,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GENERIC_FAMILY_NAMES,
    TUNED_GENERIC_FAMILY_PARAMS,
    GenericFamilyRetainedTotalSurrogate,
    load_generic_family_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    MANY_DOMAIN_TARGET,
)
from experiments.domain_phase_mix.two_phase_many_olmix_loglinear_sl_verb import fit_olmix_loglinear_model

SCRIPT_DIR = Path(__file__).resolve().parent
FULL_OUTPUT_CSV = SCRIPT_DIR / "grp_other_metrics_table_full.csv"
SLIDE_OUTPUT_CSV = SCRIPT_DIR / "grp_other_metrics_table_slide.csv"
LATEX_OUTPUT = SCRIPT_DIR / "grp_other_metrics_table_rows.tex"

SEEDPANEL_SUMMARY_CSV = (
    Path(__file__).resolve().parents[1] / "qsplit240_fixed_subset_seedpanel_n3_mmlu_sl_verb_candidate_summary.csv"
)


@dataclass(frozen=True)
class TargetSpec:
    label: str
    csv_path: Path | None
    target: str
    higher_is_better: bool
    notes: str


TARGET_SPECS = (
    TargetSpec(
        label="Paloma BPB (orig swarm)",
        csv_path=None,
        target="eval/paloma/bpb",
        higher_is_better=False,
        notes="Original 241-run swarm. Lower is better.",
    ),
    TargetSpec(
        label="c4_en BPB (orig swarm)",
        csv_path=None,
        target="eval/paloma/c4_en/bpb",
        higher_is_better=False,
        notes="Original 241-run swarm. Lower is better.",
    ),
    TargetSpec(
        label="Programming-languages BPB (orig swarm)",
        csv_path=None,
        target="eval/paloma/dolma_100_programing_languages/bpb",
        higher_is_better=False,
        notes="Original 241-run swarm. Lower is better.",
    ),
    TargetSpec(
        label="MMLU choice_logprob_norm (orig swarm)",
        csv_path=None,
        target="lm_eval/mmlu_5shot/choice_logprob_norm",
        higher_is_better=True,
        notes="Original 241-run swarm. Higher is better.",
    ),
    TargetSpec(
        label="choice_logprob_norm_mean (seed panel)",
        csv_path=SEEDPANEL_SUMMARY_CSV,
        target="choice_logprob_norm_mean",
        higher_is_better=True,
        notes="Fixed-subset 3-seed panel summary target. Higher is better.",
    ),
)
MODEL_ORDER = ("GRP", "--- no groups", "Olmix loglinear")


def _display_float(value: float | None, fmt: str) -> str:
    if value is None or pd.isna(value):
        return "--"
    return format(float(value), fmt)


def _metrics(
    frame: pd.DataFrame,
    name_col: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    higher_is_better: bool,
) -> dict[str, Any]:
    residuals = y_pred - y_true
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    chosen_idx = int(np.argmax(y_pred) if higher_is_better else np.argmin(y_pred))
    best_idx = int(np.argmax(y_true) if higher_is_better else np.argmin(y_true))
    regret = (
        float(y_true[best_idx] - y_true[chosen_idx])
        if higher_is_better
        else float(y_true[chosen_idx] - y_true[best_idx])
    )
    return {
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "r2": float(1.0 - sse / sst),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "regret_at_1": regret,
        "chosen_candidate": str(frame.iloc[chosen_idx][name_col]),
        "best_candidate": str(frame.iloc[best_idx][name_col]),
    }


def _load_packet(target_spec: TargetSpec):
    if target_spec.csv_path is None:
        packet = load_generic_family_packet(target=target_spec.target)
        return packet.base.frame, packet.base.name_col, packet.base.w, packet.base.y, packet

    frame, spec, _ = load_two_phase_many_candidate_summary_spec(
        target_spec.csv_path,
        objective_metric=target_spec.target,
        name=f"grp_other_metrics_{target_spec.target}",
    )
    packet = load_generic_family_packet(target=MANY_DOMAIN_TARGET)
    packet = type(packet)(
        base=type(packet.base)(
            frame=frame,
            name_col="candidate_run_name",
            y=spec.y,
            w=spec.weights,
            m=spec.M,
            c0=np.asarray(spec.epoch_multipliers[0], dtype=float),
            c1=np.asarray(spec.epoch_multipliers[1], dtype=float),
            domain_names=list(spec.domain_names),
        ),
        pairs=packet.pairs,
        pair_topics=packet.pair_topics,
        singletons=packet.singletons,
        family_map=packet.family_map,
    )
    return frame, "candidate_run_name", spec.weights, spec.y, packet


def _compute_row(
    target_spec: TargetSpec,
    *,
    family_totals: tuple[str, ...],
    model_name: str,
    cv_seed: int,
) -> dict[str, Any]:
    frame, name_col, weights, y, packet = _load_packet(target_spec)
    model = GenericFamilyRetainedTotalSurrogate(
        packet,
        params=TUNED_GENERIC_FAMILY_PARAMS,
        family_totals=family_totals,
        quality_discount=True,
    ).fit(weights, y)
    train_pred = model.predict(weights)
    train = _metrics(frame, name_col, y, train_pred, higher_is_better=target_spec.higher_is_better)

    kf = KFold(n_splits=5, shuffle=True, random_state=cv_seed)
    oof = np.zeros_like(y, dtype=float)
    fold_regrets: list[float] = []
    for _fold_idx, (tr, te) in enumerate(kf.split(weights)):
        fold_model = GenericFamilyRetainedTotalSurrogate(
            packet,
            params=TUNED_GENERIC_FAMILY_PARAMS,
            family_totals=family_totals,
            quality_discount=True,
        ).fit(weights[tr], y[tr])
        pred = fold_model.predict(weights[te])
        oof[te] = pred
        chosen = int(np.argmax(pred) if target_spec.higher_is_better else np.argmin(pred))
        best = int(np.argmax(y[te]) if target_spec.higher_is_better else np.argmin(y[te]))
        if target_spec.higher_is_better:
            fold_regrets.append(float(y[te][best] - y[te][chosen]))
        else:
            fold_regrets.append(float(y[te][chosen] - y[te][best]))

    cv = _metrics(frame, name_col, y, oof, higher_is_better=target_spec.higher_is_better)
    return {
        "target": target_spec.label,
        "model": model_name,
        "n_params": len(model.coef_) + 1 + len(TUNED_GENERIC_FAMILY_PARAMS),
        "train_r2": float(train["r2"]),
        "cv_r2": float(cv["r2"]),
        "cv_rmse": float(cv["rmse"]),
        "cv_spearman": float(cv["spearman"]),
        "cv_regret_at_1": float(cv["regret_at_1"]),
        "cv_foldmean_regret_at_1": float(np.mean(fold_regrets)),
        "notes": target_spec.notes,
    }


def _compute_olmix_row(target_spec: TargetSpec, *, cv_seed: int) -> dict[str, Any]:
    frame, name_col, weights, y, _packet = _load_packet(target_spec)
    fit_targets = -y if target_spec.higher_is_better else y
    fit = fit_olmix_loglinear_model(weights, fit_targets)
    train_fit_pred = fit.predict(weights)
    train_pred = -train_fit_pred if target_spec.higher_is_better else train_fit_pred
    train = _metrics(frame, name_col, y, train_pred, higher_is_better=target_spec.higher_is_better)

    kf = KFold(n_splits=5, shuffle=True, random_state=cv_seed)
    oof = np.zeros_like(y, dtype=float)
    fold_regrets: list[float] = []
    for fold_idx, (tr, te) in enumerate(kf.split(weights)):
        fold_fit = fit_olmix_loglinear_model(weights[tr], fit_targets[tr], seed=fold_idx)
        fold_fit_pred = fold_fit.predict(weights[te])
        fold_pred = -fold_fit_pred if target_spec.higher_is_better else fold_fit_pred
        oof[te] = fold_pred
        chosen = int(np.argmax(fold_pred) if target_spec.higher_is_better else np.argmin(fold_pred))
        best = int(np.argmax(y[te]) if target_spec.higher_is_better else np.argmin(y[te]))
        if target_spec.higher_is_better:
            fold_regrets.append(float(y[te][best] - y[te][chosen]))
        else:
            fold_regrets.append(float(y[te][chosen] - y[te][best]))

    cv = _metrics(frame, name_col, y, oof, higher_is_better=target_spec.higher_is_better)
    return {
        "target": target_spec.label,
        "model": "Olmix loglinear",
        "n_params": len(fit.coefficients) + 1,
        "train_r2": float(train["r2"]),
        "cv_r2": float(cv["r2"]),
        "cv_rmse": float(cv["rmse"]),
        "cv_spearman": float(cv["spearman"]),
        "cv_regret_at_1": float(cv["regret_at_1"]),
        "cv_foldmean_regret_at_1": float(np.mean(fold_regrets)),
        "notes": (
            f"{target_spec.notes} Olmix is fit directly on the target for lower-is-better metrics "
            "and on the negated target for higher-is-better metrics."
        ),
    }


def _slide_frame(full: pd.DataFrame) -> pd.DataFrame:
    return full[
        [
            "target",
            "model",
            "train_r2",
            "cv_r2",
            "cv_rmse",
            "cv_spearman",
            "cv_regret_at_1",
            "cv_foldmean_regret_at_1",
        ]
    ].copy()


def _latex_rows(slide: pd.DataFrame) -> str:
    lines = []
    for row in slide.itertuples(index=False):
        lines.append(
            f"{row.target} & "
            f"{row.model} & "
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
    parser.add_argument("--cv-seed", type=int, default=0)
    parser.add_argument("--output-csv", type=Path, default=FULL_OUTPUT_CSV)
    parser.add_argument("--slide-csv", type=Path, default=SLIDE_OUTPUT_CSV)
    parser.add_argument("--latex-output", type=Path, default=LATEX_OUTPUT)
    args = parser.parse_args()

    rows = []
    for target_spec in TARGET_SPECS:
        rows.append(
            _compute_row(
                target_spec,
                family_totals=GENERIC_FAMILY_NAMES,
                model_name="GRP",
                cv_seed=args.cv_seed,
            )
        )
        rows.append(
            _compute_row(
                target_spec,
                family_totals=(),
                model_name="--- no groups",
                cv_seed=args.cv_seed,
            )
        )
        rows.append(_compute_olmix_row(target_spec, cv_seed=args.cv_seed))

    full = pd.DataFrame(rows)
    full["target"] = pd.Categorical(full["target"], categories=[spec.label for spec in TARGET_SPECS], ordered=True)
    full["model"] = pd.Categorical(full["model"], categories=list(MODEL_ORDER), ordered=True)
    full = full.sort_values(["target", "model"]).reset_index(drop=True)
    slide = _slide_frame(full)

    args.output_csv.write_text(full.to_csv(index=False))
    args.slide_csv.write_text(slide.to_csv(index=False))
    args.latex_output.write_text(_latex_rows(slide))

    print(full.to_markdown(index=False))
    print(f"\nWrote {args.output_csv}")
    print(f"Wrote {args.slide_csv}")
    print(f"Wrote {args.latex_output}")


if __name__ == "__main__":
    main()
