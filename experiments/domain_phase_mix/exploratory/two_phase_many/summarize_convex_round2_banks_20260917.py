# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "matplotlib", "joblib", "tabulate"]
# ///
"""Combine frozen finalist bank predictions and a predeclared equal convex average."""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_mariner_policy_banks_20260917 as analysis,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    validate_convex_round2_banks_20260917 as round2,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    validate_mariner_policy_banks_20260917 as prior,
)

ROOT = prior.OUTPUT.parent
OUTPUT = ROOT / "convex_round2_summary"
REFERENCE = "hs17_mariner"
BOOTSTRAPS = 2000
SEED = 20260917
NOISE_SD = {"uncheatable": 0.0008043605, "table9": 0.0045795967}
LABELS = {
    REFERENCE: "MARINER (frozen)",
    "cv17_smooth_d1": "Prior convex smooth hinge",
    "cv17_huber_d8": "Prior convex Huber hinge",
    "cv2_log_benefit": "Log benefit + linear harm",
    "cv2_shifted_power": "Shifted-power benefit (one unstable fit)",
    "cf2_huber_d8__selection": "Convex Huber: selection-aware fit",
    "cf2_huber_d8__bounded_rms": "Convex Huber: bounded-RMS penalty",
    "cf2_mariner__selection": "MARINER: selection-aware control",
    "cf2_mariner__bounded_rms": "MARINER: bounded-RMS control",
    "ce2_majorant": "Convex tangent continuation",
    "ce2_envelope": "Convex lower envelope",
    round2.ENSEMBLE_ID: "Equal average of new convex finalists",
}
PANEL_LABELS = {"delphi_3e18_39bucket": "Qwen3", "300m_39bucket": "Llama 200M", "60m_39bucket": "Llama 160M"}


def load_and_combine(tracks: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    banks, doses, members, identities = [], [], [], {}
    for track in tracks:
        path = ROOT / "convex_round2_banks" / track
        verification = json.loads((path / "verification.json").read_text())
        assert verification["fits"] == verification["expected_fits"]
        manifest = json.loads((path / "manifest.json").read_text())
        identities[track] = prior.sha256(path / "manifest.json")
        models = manifest["models"]
        members.extend(model for model in models if "mariner" not in model)
        bank, dose = pd.read_csv(path / "bank_predictions.csv"), pd.read_csv(path / "dose_predictions.csv")
        banks.append(bank[bank.model.isin(models)])
        doses.append(dose[dose.model.isin(models)])
    assert len(set(members)) == len(members)
    banks = pd.concat(banks, ignore_index=True)
    doses = pd.concat(doses, ignore_index=True)
    banks = pd.concat([banks, round2.equal_average(banks, tuple(members), ["panel", "target", "coordinate_id"])])
    doses = pd.concat([doses, round2.equal_average(doses, tuple(members), ["target", "run_name"])])
    for path, models in [(prior.OUTPUT, [REFERENCE]), (prior.OUTPUT / "convex", ["cv17_smooth_d1", "cv17_huber_d8"])]:
        bank, dose = pd.read_csv(path / "bank_predictions.csv"), pd.read_csv(path / "dose_predictions.csv")
        banks = pd.concat([banks, bank[bank.model.isin(models)]], ignore_index=True)
        doses = pd.concat([doses, dose[dose.model.isin(models)]], ignore_index=True)
        identities[str(path)] = prior.sha256(path / "manifest.json")
    assert not banks.duplicated(["model", "panel", "target", "coordinate_id"]).any()
    assert not doses.duplicated(["model", "target", "run_name"]).any()
    prior.json_write(
        OUTPUT / "bank_combination_manifest.json",
        {
            "input_manifest_hashes": identities,
            "equal_convex_members": members,
            "ensemble_rule": "All advanced new convex models; equal positive weights fixed without bank labels",
            "bootstrap": {"kind": "paired coordinate-menu sensitivity", "draws": BOOTSTRAPS, "seed": SEED},
            "label_noise": {
                "kind": "conditional Gaussian sensitivity, not identified measurement uncertainty",
                "qwen_coordinate_sd": NOISE_SD,
                "pair_correlations": [0.0, 0.5, 0.9],
            },
        },
    )
    banks.to_csv(OUTPUT / "bank_predictions.csv", index=False)
    doses.to_csv(OUTPUT / "dose_predictions.csv", index=False)
    return banks, doses, members


def metrics_and_sensitivity(banks: pd.DataFrame) -> pd.DataFrame:
    metrics, bootstrap, noise = [], [], []
    rng = np.random.default_rng(SEED)
    for (panel, target), all_models in banks.groupby(["panel", "target"]):
        reference = all_models[all_models.model.eq(REFERENCE)].set_index("coordinate_id").sort_index()
        complete = reference.complete_components.to_numpy(bool)
        y = reference.measured.to_numpy()[complete]
        p0 = reference.prediction.to_numpy()[complete]
        ids = reference.index.to_numpy()[complete]
        selected0 = int(np.argmin(p0))
        draws = rng.integers(0, len(y), (BOOTSTRAPS, len(y)))
        picked0 = draws[np.arange(BOOTSTRAPS), np.argmin(p0[draws], axis=1)]
        for model, own in all_models.groupby("model"):
            own = own.set_index("coordinate_id").sort_index()
            assert own.index.equals(reference.index)
            np.testing.assert_allclose(own.measured, reference.measured, rtol=0, atol=0, equal_nan=True)
            for labels in ("current_weights_complete", "historical_aggregate"):
                frame = (
                    own[own.complete_components].reset_index()
                    if labels == "current_weights_complete"
                    else own.assign(measured=own.historical_aggregate).reset_index()
                )
                for stratum, mask in prior.source_masks(frame):
                    metrics.append(
                        {
                            "model": model,
                            "panel": panel,
                            "target": target,
                            "labels": labels,
                            "stratum": stratum,
                            **prior.selection_metrics(frame.iloc[np.flatnonzero(mask)]),
                        }
                    )
            if model == REFERENCE:
                continue
            p = own.prediction.to_numpy()[complete]
            selected = int(np.argmin(p))
            picked = draws[np.arange(BOOTSTRAPS), np.argmin(p[draws], axis=1)]
            delta = y[picked] - y[picked0]
            bootstrap.append(
                {
                    "model": model,
                    "panel": panel,
                    "target": target,
                    "n": len(y),
                    "observed_regret_delta": y[selected] - y[selected0],
                    "menu_delta_mean": delta.mean(),
                    "menu_delta_p025": np.quantile(delta, 0.025),
                    "menu_delta_p975": np.quantile(delta, 0.975),
                    "menu_win_fraction": (delta < -1e-12).mean(),
                    "menu_tie_fraction": (np.abs(delta) <= 1e-12).mean(),
                }
            )
            if panel == prior.DOSE_PANEL:
                gap = y[selected] - y[selected0]
                for correlation in (0.0, 0.5, 0.9):
                    sd = np.sqrt(2 * (1 - correlation)) * NOISE_SD[target]
                    same = ids[selected] == ids[selected0]
                    noise.append(
                        {
                            "model": model,
                            "target": target,
                            "same_selected_policy": same,
                            "assumed_pair_correlation": correlation,
                            "assumed_pair_sd": 0 if same else sd,
                            "observed_and_expected_regret_delta": gap,
                            "conditional_probability_lower_loss": 0.0 if same else stats.norm.cdf(-gap / sd),
                            "conditional_probability_tie": float(same),
                        }
                    )
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OUTPUT / "bank_metrics.csv", index=False)
    pd.DataFrame(bootstrap).to_csv(OUTPUT / "menu_sensitivity.csv", index=False)
    pd.DataFrame(noise).to_csv(OUTPUT / "noise_sensitivity.csv", index=False)
    return metrics


def plot_bank_gaps(metrics: pd.DataFrame) -> None:
    frame = metrics[metrics.labels.eq("current_weights_complete") & metrics.stratum.eq("all")]
    columns = [(panel, target) for panel in PANEL_LABELS for target in ("uncheatable", "table9")]
    order = [model for model in LABELS if model in set(frame.model) and model != REFERENCE]
    pivot = frame.pivot(index="model", columns=["panel", "target"], values="regret_at_1").reindex(columns=columns)
    delta = pivot.loc[order].to_numpy() - pivot.loc[REFERENCE].to_numpy()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.usetex": False, "axes.grid": False})
    fig, axis = plt.subplots(figsize=(12.8, 1.0 + 0.53 * len(order)), layout="constrained")
    limit = max(0.005, float(np.max(np.abs(delta))))
    heat = axis.imshow(delta, cmap="RdYlGn_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_yticks(range(len(order)), [LABELS[model] for model in order])
    axis.set_xticks(
        range(6),
        [
            PANEL_LABELS[panel] + "\n" + ("Uncheatable" if target == "uncheatable" else "Suite")
            for panel, target in columns
        ],
    )
    for row in range(len(order)):
        for column in range(6):
            value = delta[row, column]
            axis.text(
                column,
                row,
                f"{value:+.4f}" if abs(value) > 1e-10 else "tie",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.65 * limit else "#202932",
            )
    axis.set_title(
        "Archived-bank selection loss relative to frozen MARINER\n"
        "Positive = worse selected BPB; adaptive development evidence",
        pad=14,
    )
    fig.colorbar(heat, ax=axis, label="Selected-loss difference (BPB)", shrink=0.78)
    fig.savefig(OUTPUT / "bank_selection.png", dpi=180)
    fig.savefig(OUTPUT / "bank_selection.pdf")
    plt.close(fig)


def combine_out_of_fold() -> None:
    """Average the same frozen finalists on matched outer-fold predictions."""
    manifest = json.loads((OUTPUT / "bank_combination_manifest.json").read_text())
    members = manifest["equal_convex_members"]
    paths = [
        ROOT / "convex_round2/report_certify/aggregate_predictions.csv",
        ROOT / "convex_fitting_round2/report_full/aggregate_predictions.csv",
        ROOT / "convex_envelopes_round2_v2/report_certify/aggregate_predictions.csv",
    ]
    current = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    keys = ["panel", "target", "repeat", "fold", "row_index"]
    indexed = [current[current.model.eq(member)].set_index(keys).sort_index() for member in members]
    template = indexed[0]
    assert len(template) == 1598 and not template.index.has_duplicates
    for own in indexed[1:]:
        assert own.index.equals(template.index)
        np.testing.assert_allclose(own.observed, template.observed, rtol=1e-13, atol=1e-13)
        np.testing.assert_array_equal(own.basin, template.basin)
    average = template.copy()
    average["model"] = round2.ENSEMBLE_ID
    average["prediction"] = np.mean([own.prediction.to_numpy() for own in indexed], axis=0)
    baseline_path = ROOT.parent / "mariner_harm_screen_20260917/fixed_basis/report_certify/aggregate_predictions.csv"
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[baseline.model.eq(REFERENCE)]
    indexed_baseline = baseline.set_index(keys).sort_index()
    assert indexed_baseline.index.equals(template.index)
    np.testing.assert_allclose(indexed_baseline.observed, template.observed, rtol=1e-13, atol=1e-13)
    frame = pd.concat([current, average.reset_index(), baseline], ignore_index=True)
    assert not frame.duplicated(["model", *keys]).any()
    frame.to_csv(OUTPUT / "aggregate_predictions.csv", index=False)
    prior.bench.grouped_metrics(frame, ["model", "panel", "target", "repeat"]).to_csv(
        OUTPUT / "aggregate_metrics.csv", index=False
    )
    prior.bench.grouped_metrics(frame, ["model", "panel", "target", "repeat", "fold"]).to_csv(
        OUTPUT / "aggregate_fold_metrics.csv", index=False
    )
    prior.json_write(
        OUTPUT / "oof_combination_audit.json",
        {
            "members": members,
            "rows_per_model": len(template),
            "exact_shared_keys": True,
            "source_hashes": {str(path): prior.sha256(path) for path in [*paths, baseline_path]},
            "ensemble_weights": [1 / len(members)] * len(members),
            "interpretation": "Pooled OOF is ranking/calibration; selection uses each of the 30 within-fold menus",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", nargs="+", required=True)
    parser.add_argument("--include-oof", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    banks, doses, members = load_and_combine(args.tracks)
    metrics = metrics_and_sensitivity(banks)
    analysis.dose_metrics(doses, OUTPUT)
    plot_bank_gaps(metrics)
    if args.include_oof:
        combine_out_of_fold()
    print(json.dumps({"members": members, "models": banks.model.nunique(), "bank_rows": len(banks)}))


if __name__ == "__main__":
    main()
