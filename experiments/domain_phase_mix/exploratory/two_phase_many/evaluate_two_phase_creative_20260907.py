# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0", "scikit-learn==1.7.2",
#   "cvxpy==1.7.5", "fsspec==2026.1.0", "gcsfs==2026.1.0", "plotly==6.5.1",
#   "tabulate==0.9.0", "threadpoolctl==3.6.0",
# ]
# ///
"""Compare the registered creative phase formulations on identical saved folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import fit_two_phase_link_transfer_20260907 as previous
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRIOR = HERE / "reference_outputs/two_phase_link_transfer_20260907"
TRANSFER = HERE / "reference_outputs/two_phase_hpr_transfer_20260907"
OUTPUT = HERE / "reference_outputs/two_phase_creative_sweep_20260907"
FAMILIES = ("joint_wspu", "path_states", "phase_geometry", "joint_followup", "semantic_followup")
BOOTSTRAP_DRAWS = 3000
BOOTSTRAP_SEED = 20260907


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def baseline_predictions(panel: dict[str, np.ndarray]) -> pd.DataFrame:
    """Import previous controls without changing their row or fold identities."""
    frames = [pd.read_csv(TRANSFER / "aggregate_replacement/predictions.csv")]
    for model in ("separate_heads", "effective_exposure_dsp"):
        for objective in previous.OBJECTIVES:
            for context in previous.CONTEXTS:
                name = "full" if context == "final" else context
                cell = PRIOR / "controls" / model / objective / name
                complete = json.loads((cell / "complete.json").read_text())
                for filename, digest in complete["sha256"].items():
                    assert file_hash(cell / filename) == digest, str(cell / filename)
                values = np.load(cell / "prediction.npz", allow_pickle=False)["prediction"]
                scored = score_mask(panel, context)
                frames.append(
                    pd.DataFrame(
                        {
                            "objective": objective,
                            "model": model,
                            "context": context,
                            "row": np.arange(len(values)),
                            "run": panel["runs"],
                            "group": panel["groups"],
                            "fold": panel["outer_fold"],
                            "tied": panel["physical_tied"],
                            "scored": scored,
                            "measured": panel[f"{objective}_aggregate"],
                            "predicted": values,
                        }
                    )
                )
    return pd.concat(frames, ignore_index=True)


def score_mask(panel: dict[str, np.ndarray], context: str) -> np.ndarray:
    eligible = ~panel["calibration_mask"]
    return eligible if context == "final" else eligible & (panel["outer_fold"] == int(context[-1]))


def validate_predictions(frame: pd.DataFrame, panel: dict[str, np.ndarray]) -> None:
    """Reject incomplete cells and metadata drift before any model comparison."""
    assert np.isfinite(frame[["measured", "predicted"]].to_numpy()).all()
    assert not frame.duplicated(["objective", "model", "context", "row"]).any()
    for model in frame.model.unique():
        cells = frame[frame.model == model]
        assert len(cells.groupby(["objective", "context"])) == 8, model
    for (objective, model, context), cell in frame.groupby(["objective", "model", "context"]):
        cell = cell.sort_values("row")
        assert np.array_equal(cell.row, np.arange(len(panel["runs"]))), (model, context)
        for column, key in (("run", "runs"), ("group", "groups"), ("fold", "outer_fold"), ("tied", "physical_tied")):
            assert np.array_equal(cell[column].to_numpy(), panel[key]), (model, context, column)
        assert np.array_equal(cell.scored.to_numpy(), score_mask(panel, context)), (model, context, "scored")
        error = np.max(np.abs(cell.measured.to_numpy() - panel[f"{objective}_aggregate"]))
        assert error < 1e-10, (objective, model, context, "objective definition", error)


def endpoint_metrics(truth: np.ndarray, prediction: np.ndarray, tied: np.ndarray) -> dict:
    result = previous.metrics(truth, prediction, tied)
    lower = truth <= np.quantile(truth, 0.2)
    result["observed_lower_quintile_rmse"] = float(np.sqrt(np.mean((prediction[lower] - truth[lower]) ** 2)))
    result["observed_lower_quintile_n"] = int(lower.sum())
    result["calibration_slope_observed_on_predicted"] = (
        float(
            np.sum((prediction - prediction.mean()) * (truth - truth.mean()))
            / np.sum((prediction - prediction.mean()) ** 2)
        )
        if np.std(prediction) > 1e-12
        else None
    )
    result["negative_predictions"] = int((prediction < 0).sum())
    return result


def binary_regret(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.where(prediction < 0, np.maximum(truth, 0), np.maximum(-truth, 0))


def collect_metrics(
    frame: pd.DataFrame, panel: dict[str, np.ndarray], output: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoint_records, pair_records, pair_frames, oof_frames = [], [], [], []
    a, t = panel["pair_asymmetric_rows"], panel["pair_tied_rows"]
    for (objective, model), group in frame.groupby(["objective", "model"]):
        oof = group[(group.context != "final") & group.scored].sort_values("row").copy()
        assert len(oof) == 518 and not oof.row.duplicated().any()
        oof["context"] = "oof"
        oof_frames.append(oof)
        contexts = [(context, cell[cell.scored].sort_values("row")) for context, cell in group.groupby("context")]
        contexts.append(("oof", oof))
        for context, cell in contexts:
            for population, subset in (("all", cell), ("tied", cell[cell.tied]), ("asymmetric", cell[~cell.tied])):
                if subset.empty:
                    continue
                endpoint_records.append(
                    {
                        "objective": objective,
                        "model": model,
                        "context": context,
                        "population": population,
                        **endpoint_metrics(
                            subset.measured.to_numpy(), subset.predicted.to_numpy(), subset.tied.to_numpy()
                        ),
                    }
                )
        for context in (*previous.CONTEXTS, "oof"):
            cell = oof if context == "oof" else group[group.context == context]
            indexed = cell.set_index("row")
            eligible = (
                np.ones(len(a), bool) if context in {"final", "oof"} else panel["outer_fold"][a] == int(context[-1])
            )
            aa, tt = a[eligible], t[eligible]
            if len(aa) == 0:
                continue
            truth = panel[f"{objective}_aggregate"][aa] - panel[f"{objective}_aggregate"][tt]
            pred = indexed.loc[aa, "predicted"].to_numpy() - indexed.loc[tt, "predicted"].to_numpy()
            pred[np.abs(pred) < 1e-12] = 0
            regret = binary_regret(truth, pred)
            pair_records.append(
                {
                    "objective": objective,
                    "model": model,
                    "context": context,
                    "n": len(truth),
                    "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
                    "phase_gain_bias": float(np.mean(truth - pred)),
                    "spearman": previous.safe_rank(truth, pred),
                    "sign_accuracy": float(np.mean(np.sign(pred) == np.sign(truth))),
                    "mean_binary_decision_regret": float(regret.mean()),
                    "asymmetric_pick_fraction": float(np.mean(pred < 0)),
                }
            )
            pair_frames.append(
                pd.DataFrame(
                    {
                        "objective": objective,
                        "model": model,
                        "context": context,
                        "asymmetric_row": aa,
                        "tied_row": tt,
                        "group": panel["groups"][aa],
                        "fold": panel["outer_fold"][aa],
                        "measured_delta": truth,
                        "predicted_delta": pred,
                        "decision_regret": regret,
                    }
                )
            )
    endpoint = pd.DataFrame(endpoint_records)
    pairs = pd.concat(pair_frames, ignore_index=True)
    endpoint.to_csv(output / "endpoint_metrics.csv", index=False)
    pd.DataFrame(pair_records).to_csv(output / "pair_metrics.csv", index=False)
    pairs.to_csv(output / "pair_predictions.csv", index=False)
    pd.concat(oof_frames, ignore_index=True).to_csv(output / "oof_predictions.csv", index=False)
    return endpoint, pairs


def bootstrap_pairs(pairs: pd.DataFrame, output: Path) -> None:
    """Report descriptive paired uncertainty and a simultaneous fixed-prediction band."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, 238, size=(BOOTSTRAP_DRAWS, 238))
    records, simultaneous = [], []
    for objective in previous.OBJECTIVES:
        group = pairs[(pairs.objective == objective) & (pairs.context == "oof")]
        table = group.pivot(index="asymmetric_row", columns="model", values="predicted_delta").sort_index()
        truth = (
            group.drop_duplicates("asymmetric_row")
            .set_index("asymmetric_row")
            .loc[table.index, "measured_delta"]
            .to_numpy()
        )
        for model in table.columns:
            if not model.startswith("CRE2-"):
                continue
            prediction = table[model].to_numpy()
            comparators = ["hpr", "separate_heads"]
            if model == "CRE2-011":
                comparators.append("CRE2-012")
            if model in {"CRE2-015", "CRE2-016"}:
                comparators.append("CRE2-011")
            for comparator in comparators:
                reference = table[comparator].to_numpy()
                error, reference_error = (prediction - truth) ** 2, (reference - truth) ** 2
                regret, reference_regret = binary_regret(truth, prediction), binary_regret(truth, reference)
                for metric, point, draws in (
                    (
                        "paired_rmse",
                        np.sqrt(error.mean()) - np.sqrt(reference_error.mean()),
                        np.sqrt(error[samples].mean(axis=1)) - np.sqrt(reference_error[samples].mean(axis=1)),
                    ),
                    (
                        "paired_decision_regret",
                        regret.mean() - reference_regret.mean(),
                        (regret - reference_regret)[samples].mean(axis=1),
                    ),
                ):
                    low, high = np.quantile(draws, [0.025, 0.975])
                    records.append(
                        {
                            "objective": objective,
                            "model": model,
                            "comparator": comparator,
                            "metric": metric,
                            "difference": float(point),
                            "ci_low": float(low),
                            "ci_high": float(high),
                            "groups": 238,
                            "draws": BOOTSTRAP_DRAWS,
                        }
                    )
                    if comparator == "hpr" and metric == "paired_decision_regret":
                        simultaneous.append((len(records) - 1, float(point), draws))
    if simultaneous:
        centered = np.stack([draws - point for _, point, draws in simultaneous])
        scale = centered.std(axis=1, ddof=1)
        z = np.divide(np.abs(centered), scale[:, None], out=np.zeros_like(centered), where=scale[:, None] > 1e-15)
        critical = float(np.quantile(z.max(axis=0), 0.95))
        for (index, point, _), std in zip(simultaneous, scale, strict=True):
            records[index]["simultaneous_ci_low"] = point - critical * std
            records[index]["simultaneous_ci_high"] = point + critical * std
        (output / "bootstrap_scope.json").write_text(
            json.dumps(
                {
                    "draws": BOOTSTRAP_DRAWS,
                    "seed": BOOTSTRAP_SEED,
                    "simultaneous_comparisons": len(simultaneous),
                    "simultaneous_critical": critical,
                    "scope": (
                        "Shared resampling of fixed OOF counterpart groups. Simultaneous band covers new-model versus "
                        "HPR decision-regret comparisons within this saved screen only; not refitting, new sources, "
                        "or prospective confirmation."
                    ),
                },
                indent=2,
            )
            + "\n"
        )
    pd.DataFrame(records).to_csv(output / "paired_bootstrap.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--families", nargs="*", choices=FAMILIES, default=list(FAMILIES))
    args = parser.parse_args()
    _, panel, _ = previous.inputs(str(PRIOR))
    frames = [baseline_predictions(panel)]
    sources = [
        Path(__file__),
        PRIOR / "inputs/panel.npz",
        PRIOR / "inputs/splits.npz",
        TRANSFER / "aggregate_replacement/predictions.csv",
    ]
    for family in args.families:
        path = args.output / family / "predictions.csv"
        frames.append(pd.read_csv(path))
        sources.append(path)
    frame = pd.concat(frames, ignore_index=True)
    validate_predictions(frame, panel)
    destination = args.output / "comparison"
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "all_predictions.csv", index=False)
    endpoint, pairs = collect_metrics(frame, panel, destination)
    bootstrap_pairs(pairs, destination)
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "source_sha256": {str(p): file_hash(p) for p in sources},
                "models": sorted(frame.model.unique().tolist()),
                "validated_rows_per_cell": 520,
                "validated_cells_per_model": 8,
                "scored_oof_rows_per_model_objective": 518,
                "phase_pairs_per_model_objective": 238,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        endpoint[(endpoint.context == "oof") & (endpoint.population == "all")][
            ["objective", "model", "rmse", "regret1", "regret5", "optimism_at_pick"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
