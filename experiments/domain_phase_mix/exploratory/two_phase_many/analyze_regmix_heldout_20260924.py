# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Why tuned RegMix matches MARINER on OlmoBaseEval Easy held-out selection.

Five diagnostics from existing records, no new runs:

1. ``picks.csv``: the candidate each model selects at the full swarm on the frozen and expanded held-out sets, per
   draw, with its measured loss, its source, its total-variation distance to the nearest swarm run and its
   largest exposure.
2. ``bootstrap.csv``: MARINER's regret minus tuned RegMix's under 2,000 resamples of the held-out candidates
   (with replacement), averaged over the ten learning-curve draws, with a 95% percentile interval.
3. ``family_errors.csv``: mean absolute component error by component family, and the signed aggregate error, at
   the ten best frozen-set OlmoBaseEval Easy candidates; plus rank correlation among the 30 best.
4. ``split_gain.csv``: how concentrated the tuned heads' split gains are per objective (share of gain on the
   top three buckets, and which buckets recur in the top three across component heads).
5. ``bank_sources_regmix.csv``: the tuned-RegMix column of the paper's held-out-sources table, from the
   observatory's full-swarm predictions, with MARINER recomputed as a check of the source grouping.

usage: OMP_NUM_THREADS=1 uv run --offline --no-sync --with lightgbm==4.7.0 python -m \\
    experiments.domain_phase_mix.exploratory.two_phase_many.analyze_regmix_heldout_20260924
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    learning_curve_expanded_bank_20260913 as expanded,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    learning_curve_mariner_fits_20260908 as fits,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_models_20260902 as models,
)

REFERENCE = SCRIPT_DIR / "reference_outputs"
OUTPUT_DIR = REFERENCE / "regmix_heldout_analysis_20260924"
OBSERVATORY_PREDICTIONS = (
    REFERENCE / "single_phase_observatory_comparators_20260908" / "external_heldout_predictions.csv"
)
TARGETS = ("uncheatable", "table9")
MODELS = ("mariner", "regmix", "regmix_official")
FULL_K = 279
DRAWS = 10
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20_260_924
TOP_CANDIDATES = 10
RANK_CANDIDATES = 30
MARINER_ID = "weibull_softplus_unscaled@kappa_floor_link_flat15_nocap"
TUNED_ID = "lightgbm_regmix"
CAP_SWEEP_SOURCES = {
    "aggregate_v_epoch_cap",
    "full_canonical_dsp_epoch_cap",
    "shared_shape_dsp_epoch_cap",
    "weibull_softplus_unscaled_epoch_cap",
}
SOURCE_GROUPS = (
    "Epoch dose-response ladders",
    "Epoch-cap sweeps of fitted optima",
    "Olmix KL sweep and scaling",
    "Earlier-surrogate KL sweeps",
    "Earlier proposals, controls, and stress panels",
)


def family(component: str) -> str:
    for key in ("mt_mbpp", "minerva", "basic_skills", "mmlu"):
        if key in component:
            return key
    return "qa_other"


def source_group(sources: str) -> str | None:
    if sources == "conditional_epoch_dose_response":
        return SOURCE_GROUPS[0]
    if sources in CAP_SWEEP_SOURCES:
        return SOURCE_GROUPS[1]
    if "olmix_kl_sweep" in sources or "olmix_scaling" in sources:
        return SOURCE_GROUPS[2]
    if "table9_dsp_kl_sweep" in sources:
        return SOURCE_GROUPS[3]
    if sources.startswith("archive::"):
        return SOURCE_GROUPS[4]
    return None


def record_predictions(target: str, model: str, draw: int, count: int) -> np.ndarray:
    """Per-component frozen-bank predictions stored in the learning-curve records at the full swarm."""
    out = np.full((0, count), np.nan)
    chunks = fits.component_chunks(tuple(range(count)))
    pieces: dict[int, np.ndarray] = {}
    for chunk in range(len(chunks)):
        payload = benchmark.load_shard(fits.job_path(fits.Job(target, model, FULL_K, draw, chunk, (), ())))
        if payload is None:
            raise FileNotFoundError(f"{target}/{model}/k{FULL_K}/draw{draw}/chunk{chunk}")
        for column, index in enumerate(payload["component_indices"]):
            pieces[int(index)] = payload["heldout_prediction"][:, column]
    out = np.stack([pieces[i] for i in range(count)], axis=1)
    return out


def bank_frame(panel: benchmark.BenchPanel, target: str) -> tuple[pd.DataFrame, models.Features, str]:
    bank, query = benchmark.heldout_features(panel, target)
    _count_column, mean_column = benchmark.HELDOUT_TARGET_COLUMNS[target]
    swarm = panel.features.weights
    bank = bank.assign(
        tv_to_swarm=[0.5 * np.abs(swarm - w[None, :]).sum(axis=1).min() for w in query.weights],
        max_epochs=query.exposures.max(axis=1),
    )
    return bank, query, mean_column


def objective_predictions(bank_name: str, panel: benchmark.BenchPanel, target: str, query: models.Features) -> dict:
    """Aggregate predictions per model and draw on the given bank: (DRAWS x coordinates) arrays."""
    group = panel.group(target)
    weights = group.aggregation_weights
    out = {}
    for model in MODELS:
        rows = []
        for draw in range(DRAWS):
            if bank_name == "frozen":
                components = record_predictions(target, model, draw, len(group.components))
            elif model == "regmix":
                components = expanded.regmix_predictions(target, FULL_K, draw, panel, query)[0]
            elif model == "regmix_official":
                components = expanded.regmix_official_predictions(target, FULL_K, draw, panel, query)
            else:
                components = expanded.stored_predictions(target, model, FULL_K, draw, panel, query)[0]
            rows.append(components @ weights)
        out[model] = np.stack(rows)
    return out


def picks_and_bootstrap(
    bank_name: str, target: str, bank: pd.DataFrame, mean_column: str, predictions: dict, generator: np.random.Generator
) -> tuple[list[dict], dict]:
    measured = bank[mean_column].to_numpy(float)
    picks = []
    for model, matrix in predictions.items():
        for draw in range(DRAWS):
            chosen = int(np.argmin(matrix[draw]))
            row = bank.iloc[chosen]
            picks.append(
                {
                    "bank": bank_name,
                    "target": target,
                    "model": model,
                    "draw": draw,
                    "coordinate_id": row["coordinate_id"],
                    "sources": row["sources"],
                    "measured_bpb": float(measured[chosen]),
                    "regret_at_1": float(measured[chosen] - measured.min()),
                    "tv_to_swarm": float(row["tv_to_swarm"]),
                    "max_epochs": float(row["max_epochs"]),
                }
            )
    n = len(measured)
    samples = generator.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
    regrets = {}
    for model, matrix in predictions.items():
        values = np.empty(BOOTSTRAP_DRAWS)
        for b in range(BOOTSTRAP_DRAWS):
            idx = samples[b]
            chosen = idx[np.argmin(matrix[:, idx], axis=1)]  # one pick per draw
            values[b] = float(np.mean(measured[chosen]) - measured[idx].min())
        regrets[model] = values
    difference = regrets["mariner"] - regrets["regmix"]
    summary = {
        "bank": bank_name,
        "target": target,
        "coordinates": n,
        "mariner_regret": float(regrets["mariner"].mean()),
        "regmix_regret": float(regrets["regmix"].mean()),
        "regmix_official_regret": float(regrets["regmix_official"].mean()),
        "mariner_minus_regmix": float(difference.mean()),
        "low": float(np.percentile(difference, 2.5)),
        "high": float(np.percentile(difference, 97.5)),
        "share_mariner_better": float((difference < 0).mean()),
        "share_equal": float((difference == 0).mean()),
    }
    return picks, summary


def family_errors(panel: benchmark.BenchPanel, bank: pd.DataFrame, mean_column: str) -> tuple[pd.DataFrame, dict]:
    """Component errors at the best frozen-set OlmoBaseEval Easy candidates, averaged over the ten draws."""
    _coords, components, _hashes = benchmark.heldout_registry()
    group = panel.group("table9")
    names = list(group.components)
    table = components[components.target.eq("table9") & components.panel.eq(panel.name)]
    measured = table.pivot_table(index="coordinate_id", columns="component", values="bpb_mean")
    measured = measured.reindex(bank["coordinate_id"])[names].to_numpy(float)
    families = np.array([family(name) for name in names])
    order = np.argsort(bank[mean_column].to_numpy(float))
    top = order[:TOP_CANDIDATES]
    rank_set = order[:RANK_CANDIDATES]
    rows, correlations = [], {}
    for model in MODELS:
        per_draw = np.stack([record_predictions("table9", model, draw, len(names)) for draw in range(DRAWS)])
        error = np.abs(per_draw[:, top, :] - measured[None, top, :])
        row = {"model": model}
        for key in ("mt_mbpp", "minerva", "basic_skills", "mmlu", "qa_other"):
            row[key] = float(np.nanmean(error[:, :, families == key]))
        row["aggregate_signed"] = float(
            np.nanmean(per_draw[:, top, :].mean(axis=2) - measured[None, top, :].mean(axis=2))
        )
        rows.append(row)
        aggregate = per_draw[:, rank_set, :].mean(axis=2)
        truth = measured[rank_set].mean(axis=1)
        correlations[model] = float(np.mean([stats.spearmanr(aggregate[d], truth).correlation for d in range(DRAWS)]))
    return pd.DataFrame(rows), correlations


def split_gains(panel: benchmark.BenchPanel) -> tuple[pd.DataFrame, dict]:
    """Refit the tuned heads at the full swarm with the recorded draw-0 shapes and read LightGBM's split gains."""
    design = fits.cached_subset_design(FULL_K, 0)
    train, _inner, _test = design.fit_rows(fits.FULL_FIT)
    matrix = panel.features.weights
    buckets = list(panel.buckets)
    model = models.LightGBMModel()
    rows, summary = [], {}
    for target in TARGETS:
        group = panel.group(target)
        names = list(group.components)
        shapes = {}
        for chunk in range(len(fits.component_chunks(tuple(range(len(names)))))):
            payload = benchmark.load_shard(fits.job_path(fits.Job(target, "regmix", FULL_K, 0, chunk, (), ())))
            for index, shape in zip(payload["component_indices"], payload["shape_json"][fits.FULL_FIT], strict=True):
                shapes[int(index)] = json.loads(str(shape))
        top3_share, counts = [], np.zeros(len(buckets), dtype=int)
        for column in range(len(names)):
            make = model._make(int(shapes[column]["n_estimators"]), int(shapes[column]["num_leaves"]))
            head = models._fit_estimator_head(make, matrix[train], group.outcomes[train, column])
            gain = head.estimator.booster_.feature_importance(importance_type="gain").astype(float)
            gain = gain / gain.sum()
            order = np.argsort(gain)[::-1]
            top3_share.append(float(gain[order[:3]].sum()))
            counts[order[:3]] += 1
        for bucket, count in zip(buckets, counts, strict=True):
            rows.append(
                {"target": target, "bucket": bucket, "heads_with_bucket_in_top3": int(count), "heads": len(names)}
            )
        summary[target] = {"median_top3_gain_share": float(np.median(top3_share)), "heads": len(names)}
    return pd.DataFrame(rows), summary


def bank_sources_table() -> pd.DataFrame:
    """Group-level selection metrics of the held-out-sources table for MARINER (check) and tuned RegMix."""
    frame = pd.read_csv(OBSERVATORY_PREDICTIONS, low_memory=False)
    frame = frame[frame.panel.eq(fits.PANEL) & frame.model.isin((MARINER_ID, TUNED_ID))].copy()
    frame["group"] = frame["sources"].map(source_group)
    frame = frame[frame["group"].notna()]
    rows = []
    for target in TARGETS:
        for group_name in SOURCE_GROUPS:
            block = frame[frame.target.eq(target) & frame.group.eq(group_name)]
            for model in (MARINER_ID, TUNED_ID):
                part = block[block.model.eq(model)].reset_index(drop=True)
                if part.empty:
                    continue
                measured = part["measured_mean_bpb"].to_numpy(float)
                predicted = part["prediction"].to_numpy(float)
                chosen = int(np.argmin(predicted))
                rows.append(
                    {
                        "target": target,
                        "group": group_name,
                        "model": "MARINER" if model == MARINER_ID else "RegMix (tuned)",
                        "coordinates": len(part),
                        "regret_at_1": float(measured[chosen] - measured.min()),
                        "rank": int(stats.rankdata(measured, method="min")[chosen]),
                        "optimism": float(measured[chosen] - predicted[chosen]),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = benchmark.load_panel(fits.PANEL)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    picks, summaries = [], []
    frozen_banks = {}
    for target in TARGETS:
        bank, query, mean_column = bank_frame(panel, target)
        frozen_banks[target] = (bank, mean_column)
        rows, summary = picks_and_bootstrap(
            "frozen", target, bank, mean_column, objective_predictions("frozen", panel, target, query), generator
        )
        picks.extend(rows)
        summaries.append(summary)
    errors, correlations = family_errors(panel, *frozen_banks["table9"])
    gains, gain_summary = split_gains(panel)
    sources = bank_sources_table()
    expanded.use_bank("expanded")
    for target in TARGETS:
        bank, query, mean_column = bank_frame(panel, target)
        rows, summary = picks_and_bootstrap(
            "expanded", target, bank, mean_column, objective_predictions("expanded", panel, target, query), generator
        )
        picks.extend(rows)
        summaries.append(summary)
    pd.DataFrame(picks).to_csv(OUTPUT_DIR / "picks.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUTPUT_DIR / "bootstrap.csv", index=False)
    errors.to_csv(OUTPUT_DIR / "family_errors.csv", index=False)
    gains.to_csv(OUTPUT_DIR / "split_gain.csv", index=False)
    sources.to_csv(OUTPUT_DIR / "bank_sources_regmix.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps({"top30_spearman": correlations, "split_gain": gain_summary, "bootstrap": summaries}, indent=2)
    )
    pd.set_option("display.width", 200)
    print("=== picks (distinct per model)")
    frame = pd.DataFrame(picks)
    print(
        frame.groupby(["bank", "target", "model"])
        .agg(
            picks=("coordinate_id", lambda s: ", ".join(f"{k[-8:]}x{v}" for k, v in s.value_counts().items())),
            bpb=("measured_bpb", "mean"),
            regret=("regret_at_1", "mean"),
            tv=("tv_to_swarm", "mean"),
            sources=("sources", lambda s: s.iloc[0][:40]),
        )
        .round(4)
        .to_string()
    )
    print("=== bootstrap")
    print(pd.DataFrame(summaries).round(4).to_string(index=False))
    print("=== family errors at the ten best frozen OBE candidates")
    print(errors.round(4).to_string(index=False))
    print("top-30 Spearman:", {k: round(v, 3) for k, v in correlations.items()})
    print("=== split gains")
    print(gain_summary)
    print(
        gains.sort_values(["target", "heads_with_bucket_in_top3"], ascending=[True, False])
        .groupby("target")
        .head(6)
        .to_string(index=False)
    )
    print("=== held-out-sources table rows")
    print(sources.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
