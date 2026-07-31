# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "wandb"]
# ///
"""Collect and summarize the exact-280 corrective HPR tied-control panel."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
PANEL_DIR = SCRIPT_DIR / "reference_outputs/corrective_hpr_280_decomposed_panel_20260727"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/corrective_hpr_280_tied_controls_results_20260728"

TRAIN_PROJECT = "marin-community/marin"
EVAL_PROJECT = "marin-community/marin-eval"
TRAIN_TAG = "delphi-corrective-hpr-280-decomposed"
UNCHEATABLE_METRIC = "eval/uncheatable_eval/bpb"
TABLE9_METRIC = "olmo_base_easy/table9_51_component_macro_bpb"
RUN_SUFFIX = "_3e18"

CANDIDATE_IDS = (
    "hprc280_unch_aklb0p25_eps0",
    "hprc280_unch_aklb0p5_eps0",
    "hprc280_unch_aklb0p75_eps0",
    "hprc280_t9_aklb0p25_eps0",
    "hprc280_t9_aklb0p5_eps0",
    "hprc280_t9_aklb0p75_eps0",
)

RUN_NOISE_SD = {"uncheatable": 0.00091299968961728, "table9": 0.003771768091801164}

# Frozen before this panel was analyzed. The archive minima include selection-biased
# exploratory draws, so established model-derived and tied references are also shown.
FRONTIERS = {
    "uncheatable": {
        "archive_minimum": (
            0.9824552536010742,
            "dphase_unch05_eff_e0p005_3e18-2cef98",
            "two-phase surrogate-derived",
        ),
        "tied_minimum": (
            0.9847821593284608,
            "agphase_014_agphase_a0_center_center_tv000_seed14-8d4c70",
            "selected tied repeat",
        ),
    },
    "table9": {
        "archive_minimum": (
            1.053310609293393,
            "rphase_216_random_phase_a1_d21_r50-09005b",
            "selection-biased random-population draw",
        ),
        "tied_minimum": (
            1.05532798746225,
            "fiber_102_fiber_1_center_s2-9af710",
            "selected tied repeat",
        ),
        "surrogate_two_phase": (
            1.056690469761157,
            "dphase_t9b075_can_e0p005_3e18-6c15f1",
            "two-phase surrogate-derived",
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=PANEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wandb-timeout", type=int, default=180)
    return parser.parse_args()


def unique_run(runs: list[wandb.apis.public.Run], prefix: str) -> wandb.apis.public.Run:
    matches = [run for run in runs if run.name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one W&B run matching {prefix!r}, found {len(matches)}")
    return matches[0]


def collect_results(manifest: pd.DataFrame, timeout: int) -> pd.DataFrame:
    api = wandb.Api(timeout=timeout)
    training = list(api.runs(TRAIN_PROJECT, filters={"tags": {"$in": [TRAIN_TAG]}}, per_page=100))
    evaluations = list(api.runs(EVAL_PROJECT, filters={"display_name": {"$regex": "t9_hprc280"}}, per_page=100))

    rows: list[dict[str, object]] = []
    for candidate_id in CANDIDATE_IDS:
        source = manifest.loc[manifest["candidate_id"].eq(candidate_id)]
        if len(source) != 1:
            raise ValueError(f"Expected one manifest row for {candidate_id}, found {len(source)}")
        record = source.iloc[0]
        train = unique_run(training, f"{candidate_id}{RUN_SUFFIX}-")
        evaluation = unique_run(evaluations, f"t9_{candidate_id}{RUN_SUFFIX}")
        if train.state != "finished" or evaluation.state != "finished":
            raise ValueError(f"Incomplete candidate {candidate_id}: train={train.state}, eval={evaluation.state}")

        uncheatable = float(train.summary.get(UNCHEATABLE_METRIC, math.nan))
        table9 = float(evaluation.summary.get(TABLE9_METRIC, math.nan))
        if not math.isfinite(uncheatable) or not math.isfinite(table9):
            raise ValueError(f"Missing metric for {candidate_id}: uncheatable={uncheatable}, table9={table9}")

        target = str(record["target"])
        observed_target = uncheatable if target == "uncheatable" else table9
        predicted_target = float(record["one_phase_tied_prediction"])
        rows.append(
            {
                "candidate_id": candidate_id,
                "target": target,
                "policy_class": str(record["policy_class"]),
                "aggregate_kl_budget": float(record["aggregate_kl_budget"]),
                "phase_information_budget": float(record["phase_information_budget"]),
                "predicted_target_bpb": predicted_target,
                "observed_target_bpb": observed_target,
                "observed_minus_predicted_bpb": observed_target - predicted_target,
                "uncheatable_bpb": uncheatable,
                "table9_macro_bpb": table9,
                "max_bucket_weight": float(record["max_bucket_weight"]),
                "max_simulated_epoch": float(record["max_simulated_epoch"]),
                "min_3e18_heldout_policy_tv": float(record["min_3e18_heldout_policy_tv"]),
                "coordinate_hash": str(record["coordinate_hash"]),
                "training_wandb_name": train.name,
                "training_wandb_url": train.url,
                "eval_wandb_name": evaluation.name,
                "eval_wandb_url": evaluation.url,
            }
        )
    return pd.DataFrame(rows).sort_values(["target", "aggregate_kl_budget"]).reset_index(drop=True)


def path_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, group in results.groupby("target"):
        ordered = group.sort_values("aggregate_kl_budget")
        selected = ordered.loc[ordered["predicted_target_bpb"].idxmin()]
        best = ordered.loc[ordered["observed_target_bpb"].idxmin()]
        spearman = ordered["predicted_target_bpb"].rank().corr(ordered["observed_target_bpb"].rank())
        rows.append(
            {
                "target": target,
                "candidate_count": len(ordered),
                "predicted_observed_spearman": spearman,
                "predicted_selected_candidate": selected["candidate_id"],
                "predicted_selected_aggregate_kl": selected["aggregate_kl_budget"],
                "predicted_selected_observed_bpb": selected["observed_target_bpb"],
                "observed_best_candidate": best["candidate_id"],
                "observed_best_aggregate_kl": best["aggregate_kl_budget"],
                "observed_best_bpb": best["observed_target_bpb"],
                "selection_regret_bpb": selected["observed_target_bpb"] - best["observed_target_bpb"],
                "predicted_path_range_bpb": (
                    ordered["predicted_target_bpb"].max() - ordered["predicted_target_bpb"].min()
                ),
                "observed_path_range_bpb": ordered["observed_target_bpb"].max() - ordered["observed_target_bpb"].min(),
            }
        )
    return pd.DataFrame(rows).sort_values("target").reset_index(drop=True)


def frontier_comparison(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, references in FRONTIERS.items():
        best = results.loc[results["target"].eq(target)].sort_values("observed_target_bpb").iloc[0]
        for reference_kind, (reference_bpb, reference_name, reference_note) in references.items():
            gap = float(best["observed_target_bpb"]) - reference_bpb
            rows.append(
                {
                    "target": target,
                    "panel_best_candidate": best["candidate_id"],
                    "panel_best_bpb": best["observed_target_bpb"],
                    "reference_kind": reference_kind,
                    "reference_name": reference_name,
                    "reference_bpb": reference_bpb,
                    "panel_minus_reference_bpb": gap,
                    "gap_in_run_sd": gap / RUN_NOISE_SD[target],
                    "reference_note": reference_note,
                    "new_frontier": gap < 0.0,
                }
            )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    headers = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for record in frame[columns].to_dict(orient="records"):
        values = []
        for value in record.values():
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([headers, divider, *rows])


def write_report(
    output_dir: Path,
    results: pd.DataFrame,
    paths: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    uncheatable = paths.loc[paths["target"].eq("uncheatable")].iloc[0]
    table9 = paths.loc[paths["target"].eq("table9")].iloc[0]
    report = f"""# Corrective exact-280 HPR tied controls at 3e18

## Coverage

The Iris parent succeeded with zero logical failures and four recovered preemptions. All six
training runs finished at the Delphi 3e18 endpoint, and all six native Table-9 evaluations finished.
This is the narrowed corrective panel: three independently fitted tied policies per objective at
aggregate KL budgets 0.25, 0.50, and 0.75. Every policy has zero phase-information budget, so this
panel tests the fitted one-phase HPR restriction, not HPR phase ordering.

## Verdict

**No new 3e18 frontier was established on either objective.**

- Uncheatable: the best panel value is {uncheatable["observed_best_bpb"]:.6f} at aggregate KL
  {uncheatable["observed_best_aggregate_kl"]:.2f}. The surrogate selected KL
  {uncheatable["predicted_selected_aggregate_kl"]:.2f}, which observed
  {uncheatable["predicted_selected_observed_bpb"]:.6f}; selection regret is
  {uncheatable["selection_regret_bpb"]:.6f} BPB. The three-point path is reversed
  (Spearman {uncheatable["predicted_observed_spearman"]:+.1f}).
- Table-9: the best panel value is {table9["observed_best_bpb"]:.6f} at aggregate KL
  {table9["observed_best_aggregate_kl"]:.2f}. The path ordering is correct
  (Spearman {table9["predicted_observed_spearman"]:+.1f}), but the observed response spans only
  {table9["observed_path_range_bpb"]:.6f} BPB versus {table9["predicted_path_range_bpb"]:.6f}
  predicted, and absolute predictions remain strongly optimistic.

The result sharpens the deployment failure. For Uncheatable, moving farther from proportional makes
the one-phase HPR prediction monotonically better while observed BPB becomes worse. For Table-9,
aggregate direction is useful, but predicted gains are substantially compressed at deployment.
This panel therefore does not rehabilitate raw HPR optimization from the exact 280-row 300M fit.

## Observed results

{markdown_table(results, [
    "candidate_id",
    "target",
    "aggregate_kl_budget",
    "predicted_target_bpb",
    "observed_target_bpb",
    "observed_minus_predicted_bpb",
    "uncheatable_bpb",
    "table9_macro_bpb",
])}

## Frontier comparisons

Negative gaps would establish a strict numerical frontier. The random-population Table-9 archive
minimum and the tied minima are selected draws, so noise-scaled gaps are reported rather than treated
as confirmed population improvements.

{markdown_table(comparisons, [
    "target",
    "panel_best_bpb",
    "reference_kind",
    "reference_bpb",
    "panel_minus_reference_bpb",
    "gap_in_run_sd",
    "new_frontier",
])}

## Files

- `observed_results.csv`: six completed candidates with predictions, both observed targets, geometry,
  and W&B provenance.
- `path_summary.csv`: objective-specific ordering, selected candidate, and selection regret.
- `frontier_comparison.csv`: frozen comparisons against the pre-panel 3e18 archive.
"""
    (output_dir / "report.md").write_text(report)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.panel_dir / "candidate_manifest.csv")
    results = collect_results(manifest, args.wandb_timeout)
    paths = path_summary(results)
    comparisons = frontier_comparison(results)
    results.to_csv(args.output_dir / "observed_results.csv", index=False)
    paths.to_csv(args.output_dir / "path_summary.csv", index=False)
    comparisons.to_csv(args.output_dir / "frontier_comparison.csv", index=False)
    write_report(args.output_dir, results, paths, comparisons)
    print(results.to_string(index=False))
    print()
    print(paths.to_string(index=False))
    print()
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
