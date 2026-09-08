# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "wandb"]
# ///
"""Score the 3e18 composite-proposal validation panel against its preregistered prediction.

The composite surrogate was asked to rank fixed-aggregate two-phase policies at the 3e18
Uncheatable frontier and picked ``technical_specialization`` in the ``plus`` orientation at phase
total variation 0.24, predicting it would beat the tied control by 0.0078 BPB. That prediction, the
panel and its hash were written before any run existed, so this is confirmatory rather than fitted.

Six rows in two matched seed blocks: the proposal, its antithetic partner at the same total
variation, and a tied control, with the aggregate identical across all six to machine precision.
Three quantities matter and they answer different questions.

The **orientation call** asks only whether plus beats minus, which is what the model actually claimed
and the one comparison the antithetic design isolates cleanly. The **odd effect**
``(L+ - L-)/2`` measures how much phase ordering moves the target along this direction. The
**asymmetry cost** ``(L+ + L-)/2 - L0`` measures what any departure from tied training costs
regardless of direction. Two-phase beats tied only when the odd effect exceeds that cost, so
reporting the gain against the control without the cost term would overstate the case.

Both seed blocks are paired against their own same-seed control, since the seed is the dominant
nuisance and comparing across blocks would spend the design's main advantage.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_PANEL_DIR = REFERENCE_OUTPUTS / "delphi_3e18_composite_proposal_validation_20260726"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_3e18_composite_proposal_validation_results_20260727"

TRAIN_PROJECT = "marin-community/marin"
EVAL_PROJECT = "marin-community/marin-eval"
TRAIN_TAG = "delphi-3e18-composite-proposal-validation"
EVAL_GROUP = "olmo_base_eval_table9_delphi_3e18_composite_proposal_validation_20260726"
UNCHEATABLE_METRIC = "eval/uncheatable_eval/bpb"
TABLE9_METRIC = "olmo_base_easy/table9_51_component_macro_bpb"
EXPECTED_ROWS = 6
# Run-to-run standard deviation at 3e18, used to read every effect as a multiple of noise.
RUN_SIGMA = {"uncheatable": 0.000913, "table9": 0.003772}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wandb-timeout", type=int, default=180)
    return parser.parse_args()


def collect_results(panel: pd.DataFrame, timeout: int) -> pd.DataFrame:
    """Join the panel to its training and native Table-9 evaluation runs by candidate id."""
    api = wandb.Api(timeout=timeout)
    training = list(api.runs(TRAIN_PROJECT, filters={"tags": {"$in": [TRAIN_TAG]}}, per_page=EXPECTED_ROWS + 50))
    evaluations = list(api.runs(EVAL_PROJECT, filters={"group": EVAL_GROUP}, per_page=EXPECTED_ROWS + 50))
    logger.info("found %d training runs and %d evaluation runs", len(training), len(evaluations))

    rows = []
    for index, entry in panel.reset_index(drop=True).iterrows():
        # The launcher abbreviates run names to cmpval_<row>_<sign>_s<block>, with a content hash
        # appended for training runs, so the join key is rebuilt from the panel's own row order
        # rather than from the full candidate id.
        short = f"cmpval_{index:02d}_{entry['sign']}_s{int(entry['seed_block'])}"
        train = next((run for run in training if run.name.startswith(short)), None)
        # An evaluation can appear more than once when an attempt crashed and was retried; the
        # finished attempt is the one carrying the metric.
        finished = [run for run in evaluations if run.name == f"t9_{short}" and run.state == "finished"]
        native = finished[0] if finished else None
        rows.append(
            {
                "candidate_id": str(entry["candidate_id"]),
                "wandb_name": short,
                "sign": entry["sign"],
                "seed_block": int(entry["seed_block"]),
                "predicted_gain_vs_tied_bpb": entry.get("model_predicted_gain_vs_tied_bpb"),
                "uncheatable_bpb": float(train.summary.get(UNCHEATABLE_METRIC, np.nan)) if train else np.nan,
                "table9_macro_bpb": float(native.summary.get(TABLE9_METRIC, np.nan)) if native else np.nan,
                "training_wandb_url": train.url if train else None,
                "eval_wandb_url": native.url if native else None,
            }
        )
    return pd.DataFrame(rows)


def decompose(block: pd.DataFrame, column: str) -> dict[str, float]:
    """Odd effect, asymmetry cost and orientation call within one seed block.

    ``plus`` places the named technical group later. The odd effect is signed so that a negative
    value means the plus orientation lowers the target, which is what the model predicted.
    """
    plus = float(block.loc[block["sign"] == "plus", column].iloc[0])
    minus = float(block.loc[block["sign"] == "minus", column].iloc[0])
    tied = float(block.loc[block["sign"] == "center", column].iloc[0])
    odd = 0.5 * (plus - minus)
    cost = 0.5 * (plus + minus) - tied
    return {
        "plus": plus,
        "minus": minus,
        "tied": tied,
        "odd_effect": odd,
        "asymmetry_cost": cost,
        "best_orientation_gain_vs_tied": min(plus, minus) - tied,
        "plus_gain_vs_tied": plus - tied,
        "orientation_call_correct": bool(plus < minus),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel_files = sorted(args.panel_dir.glob("validation_panel-*.csv"))
    assert len(panel_files) == 1, f"expected exactly one panel csv, found {panel_files}"
    panel = pd.read_csv(panel_files[0])
    proposal = json.loads((args.panel_dir / "proposal.json").read_text())
    assert len(panel) == EXPECTED_ROWS, f"panel has {len(panel)} rows, expected {EXPECTED_ROWS}"

    results = collect_results(panel, args.wandb_timeout)
    results.to_csv(args.output_dir / "observed_results.csv", index=False)
    missing = results[results["uncheatable_bpb"].isna() | results["table9_macro_bpb"].isna()]
    if len(missing):
        logger.warning("missing metrics for %d rows:\n%s", len(missing), missing[["candidate_id"]].to_string())

    print(f"\nPREREGISTERED: {proposal['winning_sign']} wins at TV {proposal['phase_tv']}, ")
    print(f"predicted gain vs tied {proposal['predicted_gain_vs_tied_bpb']:+.5f} BPB, ")
    print(f"predicted tied BPB {proposal['predicted_tied_bpb']:.5f}\n")
    print(results[["candidate_id", "sign", "seed_block", "uncheatable_bpb", "table9_macro_bpb"]].to_string(index=False))

    records = []
    for target, column in (("uncheatable", "uncheatable_bpb"), ("table9", "table9_macro_bpb")):
        sigma = RUN_SIGMA[target]
        print("\n" + "=" * 96)
        print(f"{target.upper()}   (run sigma {sigma:.6f})")
        print("=" * 96)
        for seed_block, block in results.groupby("seed_block"):
            if block[column].isna().any() or set(block["sign"]) != {"plus", "minus", "center"}:
                logger.warning("seed block %s incomplete for %s", seed_block, target)
                continue
            summary = decompose(block, column)
            records.append({"target": target, "seed_block": int(seed_block), **summary})
            print(
                f"  block {seed_block}: plus {summary['plus']:.6f}  minus {summary['minus']:.6f}  "
                f"tied {summary['tied']:.6f}"
            )
            print(
                f"    odd effect {summary['odd_effect']:+.6f} ({summary['odd_effect'] / sigma:+.2f}s)   "
                f"asymmetry cost {summary['asymmetry_cost']:+.6f} ({summary['asymmetry_cost'] / sigma:+.2f}s)   "
                f"orientation call {'CORRECT' if summary['orientation_call_correct'] else 'WRONG'}"
            )
            print(
                f"    best orientation vs tied {summary['best_orientation_gain_vs_tied']:+.6f} "
                f"({summary['best_orientation_gain_vs_tied'] / sigma:+.2f}s)   "
                f"two-phase beats tied: {'YES' if summary['best_orientation_gain_vs_tied'] < 0 else 'NO'}"
            )

    table = pd.DataFrame(records)
    table.to_csv(args.output_dir / "pair_decomposition.csv", index=False)

    print("\n" + "=" * 96)
    print("VERDICT AGAINST THE PREREGISTERED PREDICTION")
    print("=" * 96)
    for target, group in table.groupby("target"):
        sigma = RUN_SIGMA[target]
        correct = int(group["orientation_call_correct"].sum())
        mean_odd = float(group["odd_effect"].mean())
        mean_cost = float(group["asymmetry_cost"].mean())
        mean_gain = float(group["plus_gain_vs_tied"].mean())
        beats = int((group["best_orientation_gain_vs_tied"] < 0).sum())
        print(f"\n  {target}")
        print(f"    orientation call correct in {correct}/{len(group)} seed blocks")
        print(f"    mean odd effect     {mean_odd:+.6f} ({mean_odd / sigma:+.2f}s)")
        print(f"    mean asymmetry cost {mean_cost:+.6f} ({mean_cost / sigma:+.2f}s)")
        print(f"    |odd| > cost (two-phase can win): {'YES' if abs(mean_odd) > mean_cost else 'NO'}")
        print(f"    best orientation beats tied in {beats}/{len(group)} blocks")
        if target == "uncheatable":
            predicted = float(proposal["predicted_gain_vs_tied_bpb"])
            print(f"    predicted plus gain {predicted:+.6f}, observed {mean_gain:+.6f} ")
            print(f"    prediction error {mean_gain - predicted:+.6f} ({(mean_gain - predicted) / sigma:+.2f}s)")

    provenance = {
        "panel": str(panel_files[0].name),
        "source_panel_sha256": proposal["source_panel_sha256"],
        "train_tag": TRAIN_TAG,
        "eval_group": EVAL_GROUP,
        "rows": len(results),
        "rows_with_missing_metrics": len(missing),
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
