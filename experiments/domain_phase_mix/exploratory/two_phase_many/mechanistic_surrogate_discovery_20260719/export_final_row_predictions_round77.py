# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas>=2.2",
#   "tabulate>=0.9",
# ]
# ///

"""Export every decision-critical 3e18 baseline prediction and residual."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_ROOT = SCRIPT_DIR.parent
OUTPUT_ROOT = TWO_PHASE_ROOT / "reference_outputs/mechanistic_surrogate_discovery_20260719"
FINAL_DIR = OUTPUT_ROOT / "final_synthesis"
ROUND_DIR = OUTPUT_ROOT / "round77_final_row_predictions"
DASHBOARD = TWO_PHASE_ROOT / "mixture_fit_debugger/src/generated/dashboard_data.json"
FROZEN_ADVERSARIAL_PREDICTIONS = OUTPUT_ROOT / "frozen_gate/adversarial_target_matched_predictions.csv"
BASELINE_MODELS = (
    "canonical",
    "effective_exposure",
    "effective_exposure_geometry",
    "separate_heads",
    "grp",
    "compact_retained_state",
    "bucket_family_grp",
    "hierarchical_phase_bucket_replay",
    "bucket_family_power_separate_heads",
)
EXPOSED_ONLY_MODELS = ("early_family_asymmetric", "inverse_deficit_log_link")


def main() -> None:
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    dashboard = json.loads(DASHBOARD.read_text())["swarms"]["delphi_3e18"]
    provenance = pd.read_csv(FINAL_DIR / "heldout_provenance_index.csv").set_index("wandb_run_name")
    adversarial = pd.read_csv(FINAL_DIR / "adversarial_provenance.csv").set_index("wandb_run_name")

    selected_indices = [
        index
        for index, row in enumerate(dashboard["rows"])
        if row["split"] == "heldout" and not bool(row["isSharedAlias"])
    ]
    selected_rows = [dashboard["rows"][index] for index in selected_indices]
    selected_names = {row["name"] for row in selected_rows}
    if len(selected_rows) != 710 or selected_names != set(provenance.index):
        raise ValueError("The row-level dashboard no longer matches the frozen 710-run provenance index")

    records: list[dict[str, object]] = []
    for target in ("uncheatable", "table9"):
        target_models = dashboard["predictions"][target]["two_phase"]
        for model in BASELINE_MODELS:
            predictions = target_models[model]["prediction"]
            if len(predictions) != len(dashboard["rows"]):
                raise ValueError(f"Prediction length changed for {target}/{model}")
            for index, row in zip(selected_indices, selected_rows, strict=True):
                source = provenance.loc[row["name"]]
                adversarial_row = adversarial.loc[row["name"]] if row["name"] in adversarial.index else None
                observed = float(row["observed"][target])
                predicted = float(predictions[index])
                candidate_target = "" if adversarial_row is None else str(adversarial_row["candidate_target"])
                records.append(
                    {
                        "heldout_id": source["heldout_id"],
                        "wandb_run_name": row["name"],
                        "mixture_sha256": source["mixture_sha256"],
                        "archive_split": source["archive_split"],
                        "training_series": source["training_series"],
                        "policy_class": source["policy_class"],
                        "candidate_target": candidate_target,
                        "target_relation": (
                            "non_adversarial"
                            if not candidate_target
                            else "target_matched"
                            if candidate_target == target
                            else "cross_target"
                        ),
                        "selection_stratum": "" if adversarial_row is None else adversarial_row["selection_stratum"],
                        "proposal_origin": "" if adversarial_row is None else adversarial_row["origin"],
                        "proposal_models": "" if adversarial_row is None else adversarial_row["proposal_models"],
                        "target": target,
                        "model": model,
                        "observed": observed,
                        "predicted": predicted,
                        "residual_predicted_minus_observed": predicted - observed,
                        "optimism_observed_minus_predicted": observed - predicted,
                    }
                )

    predictions = pd.DataFrame(records).sort_values(["target", "model", "heldout_id"])
    expected_rows = 710 * 2 * len(BASELINE_MODELS)
    if len(predictions) != expected_rows:
        raise ValueError(f"Expected {expected_rows} row-level predictions, found {len(predictions)}")
    key = ["heldout_id", "target", "model"]
    if predictions.duplicated(key).any():
        raise ValueError("Row-level prediction keys are not unique")
    predictions.to_csv(ROUND_DIR / "all_3e18_row_predictions.csv", index=False)

    adversarial_predictions = predictions.loc[predictions["target_relation"].ne("non_adversarial")].copy()
    exposed_only = pd.read_csv(FROZEN_ADVERSARIAL_PREDICTIONS)
    exposed_only = exposed_only.loc[exposed_only["model"].isin(EXPOSED_ONLY_MODELS)].copy()
    expected_exposed_only = 2 * 2 * 60
    if len(exposed_only) != expected_exposed_only:
        raise ValueError(f"Expected {expected_exposed_only} exposed-only predictions, found {len(exposed_only)}")
    exposed_only = exposed_only.merge(
        adversarial.reset_index(),
        on="candidate_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_provenance"),
    )
    if exposed_only["heldout_id"].isna().any():
        raise ValueError("Exposed-only predictions do not match the frozen adversarial provenance")
    if not exposed_only["target"].eq(exposed_only["candidate_target"]).all():
        raise ValueError("Exposed-only predictions must be target matched")
    exposed_records = pd.DataFrame(
        {
            "heldout_id": exposed_only["heldout_id"],
            "wandb_run_name": exposed_only["wandb_run_name"],
            "mixture_sha256": exposed_only["mixture_sha256"],
            "archive_split": "adversarial_stress_panel",
            "training_series": exposed_only["origin"],
            "policy_class": exposed_only["policy_class_provenance"],
            "candidate_target": exposed_only["candidate_target"],
            "target_relation": "target_matched",
            "selection_stratum": exposed_only["selection_stratum_provenance"],
            "proposal_origin": exposed_only["origin"],
            "proposal_models": exposed_only["proposal_models_provenance"],
            "target": exposed_only["target"],
            "model": exposed_only["model"],
            "observed": exposed_only["observed"],
            "predicted": exposed_only["predicted"],
        }
    )
    exposed_records["residual_predicted_minus_observed"] = exposed_records["predicted"] - exposed_records["observed"]
    exposed_records["optimism_observed_minus_predicted"] = exposed_records["observed"] - exposed_records["predicted"]
    adversarial_predictions = pd.concat([adversarial_predictions, exposed_records], ignore_index=True).sort_values(
        ["target", "model", "heldout_id"]
    )
    if len(adversarial_predictions) != 2_400:
        raise ValueError(f"Expected 2,400 adversarial predictions, found {len(adversarial_predictions)}")
    if adversarial_predictions.duplicated(key).any():
        raise ValueError("Adversarial prediction keys are not unique")
    adversarial_predictions.to_csv(ROUND_DIR / "adversarial_row_predictions.csv", index=False)

    summary_source = pd.concat(
        [predictions, exposed_records],
        ignore_index=True,
    )
    summary = (
        summary_source.groupby(["target", "model", "target_relation"], as_index=False)
        .agg(
            rows=("heldout_id", "size"),
            rmse=("residual_predicted_minus_observed", lambda values: float((values.pow(2).mean()) ** 0.5)),
            bias=("residual_predicted_minus_observed", "mean"),
            optimism_gt_0p05=("optimism_observed_minus_predicted", lambda values: int((values > 0.05).sum())),
            worst_optimism=("optimism_observed_minus_predicted", "max"),
        )
        .sort_values(["target", "model", "target_relation"])
    )
    summary.to_csv(ROUND_DIR / "row_prediction_summary.csv", index=False)
    report = "\n".join(
        [
            "# Round 77: terminal row-level prediction export",
            "",
            f"Exported {len(predictions):,} uniquely keyed baseline predictions: 710 coordinate-disjoint heldout "
            f"runs, two targets, and {len(BASELINE_MODELS)} Pareto-baseline models.",
            "",
            f"The adversarial export contains {len(adversarial_predictions):,} predictions. All 11 exposed-panel "
            "models have target-matched residuals; the nine archive-wide models also have cross-target residuals.",
            "",
            "The export carries residual and optimism signs explicitly and joins immutable mixture hashes, archive "
            "splits, policy classes, candidate targets, selection strata, proposer series, and proposal origins. It "
            "introduces no model or hyperparameter choice and reads no sealed confirmation outcome.",
            "",
            summary.to_markdown(index=False, floatfmt=".5f"),
        ]
    )
    (ROUND_DIR / "report.md").write_text(report + "\n")
    print(f"exported {len(predictions)} row-level predictions")


if __name__ == "__main__":
    main()
