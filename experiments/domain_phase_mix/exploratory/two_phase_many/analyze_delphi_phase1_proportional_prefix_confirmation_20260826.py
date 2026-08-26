# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2"]
# ///
"""Score the frozen crossed confirmation for proportional-prefix branches."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_RESULTS_DIR = REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_confirmation_results_20260826"
DEFAULT_DESIGN_DIR = REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_confirmation_20260826"
DEFAULT_CONTRACT = REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_wave2_contract_20260826" / "contract.json"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_confirmation_analysis_20260826"
TARGET = "bpb"
FRONTIER_BPB = 0.9798883332146539
CANDIDATE_PATTERN = re.compile(r"confirm_candidate(?P<candidate>\d+)_seed\d+_data\d+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR / "branch_results.csv")
    parser.add_argument("--coverage", type=Path, default=DEFAULT_RESULTS_DIR / "coverage.json")
    parser.add_argument("--design-summary", type=Path, default=DEFAULT_DESIGN_DIR / "continuation_summary.csv")
    parser.add_argument("--design-manifest", type=Path, default=DEFAULT_DESIGN_DIR / "manifest.json")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bytes_exact(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to replace a different frozen artifact: {path}")
        return
    path.write_bytes(payload)


def scoring_contract(path: Path) -> dict[str, object]:
    contract = cast(dict[str, object], json.loads(path.read_text()))
    confirmation_contract = cast(dict[str, object], contract.get("confirmation", {}))
    scoring = cast(dict[str, object], confirmation_contract.get("scoring", {}))
    expected = {
        "bootstrap_draws": 50_000,
        "bootstrap_seed": 20_260_826,
        "primary_candidate_index": 0,
        "candidate_count": 3,
        "crossed_prefix_data_blocks": 9,
    }
    observed = {
        "bootstrap_draws": scoring.get("bootstrap_draws"),
        "bootstrap_seed": scoring.get("bootstrap_seed"),
        "primary_candidate_index": confirmation_contract.get("primary_candidate_index"),
        "candidate_count": confirmation_contract.get("candidate_count"),
        "crossed_prefix_data_blocks": confirmation_contract.get("crossed_prefix_data_blocks"),
    }
    if observed != expected:
        raise ValueError(f"Confirmation scoring contract changed: {observed}")
    return {**expected, "frontier_bpb": FRONTIER_BPB}


def validate_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    results = pd.read_csv(args.results)
    summary = pd.read_csv(args.design_summary)
    coverage = cast(dict[str, object], json.loads(args.coverage.read_text()))
    manifest = cast(dict[str, object], json.loads(args.design_manifest.read_text()))
    if coverage.get("status") != "complete" or coverage.get("missing_rows") != 0:
        raise ValueError("Confirmation results are incomplete")
    if coverage.get("expected_rows") != 36 or coverage.get("visible_result_rows") != 36:
        raise ValueError("Confirmation result count changed")
    if coverage.get("sealed_referee_rows") != 0 or coverage.get("referee_outcomes_opened") is not False:
        raise ValueError("Confirmation unexpectedly contains sealed referees")
    artifacts = cast(dict[str, object], manifest.get("artifacts", {}))
    if artifacts.get("continuation_summary.csv") != file_sha256(args.design_summary):
        raise ValueError("Confirmation manifest references a different design summary")
    inputs = cast(dict[str, object], manifest.get("inputs", {}))
    if inputs.get("wave2_contract_sha256") != file_sha256(args.contract):
        raise ValueError("Confirmation manifest references a different scoring contract")
    if len(summary) != 36 or len(results) != 36:
        raise ValueError("Confirmation panel must contain exactly 36 rows")
    if summary.fit_budget.astype(bool).any() or results.fit_budget.astype(bool).any():
        raise ValueError("Confirmation rows leaked into the fit budget")
    if set(summary.continuation_id) != set(results.continuation_id):
        raise ValueError("Confirmation results do not match the frozen design")
    if results.run_id.nunique() != 36 or results.run_name.nunique() != 36:
        raise ValueError("Confirmation runtime identities repeat")
    if TARGET not in results or not np.isfinite(results[TARGET]).all():
        raise ValueError("Confirmation Uncheatable BPB is incomplete")
    return results, summary, coverage


def paired_differences(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["prefix_repeat_seed", "data_seed"]
    tied = results[results.role.eq("paired_tied_confirmation")]
    if len(tied) != 9 or tied.groupby(keys).size().ne(1).any():
        raise ValueError("Expected one tied row in each crossed prefix-data block")
    tied_values = tied.set_index(keys)[TARGET]
    rows = []
    candidates = results[results.role.eq("predicted_branch_confirmation")]
    for row in candidates.itertuples(index=False):
        match = CANDIDATE_PATTERN.fullmatch(str(row.continuation_id))
        if match is None:
            raise ValueError(f"Malformed confirmation candidate identity: {row.continuation_id}")
        key = (int(row.prefix_repeat_seed), int(row.data_seed))
        if key not in tied_values.index:
            raise ValueError(f"Candidate lacks a paired tied control: {key}")
        tied_bpb = float(tied_values.loc[key])
        rows.append(
            {
                "candidate_index": int(match.group("candidate")),
                "prefix_repeat_seed": key[0],
                "data_seed": key[1],
                "candidate_bpb": float(row.bpb),
                "tied_bpb": tied_bpb,
                "candidate_minus_tied_bpb": float(row.bpb) - tied_bpb,
            }
        )
    paired = pd.DataFrame(rows).sort_values(["candidate_index", *keys]).reset_index(drop=True)
    if len(paired) != 27 or paired.groupby("candidate_index").size().to_dict() != {0: 9, 1: 9, 2: 9}:
        raise ValueError("Expected three complete 3x3 candidate panels")
    return paired


def crossed_bootstrap(
    paired: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    prefix_seeds = tuple(sorted(paired.prefix_repeat_seed.unique()))
    data_seeds = tuple(sorted(paired.data_seed.unique()))
    if len(prefix_seeds) != 3 or len(data_seeds) != 3:
        raise ValueError("Crossed bootstrap requires three prefix and three data seeds")
    rng = np.random.default_rng(seed)
    prefix_draws = rng.integers(0, len(prefix_seeds), size=(draws, len(prefix_seeds)))
    data_draws = rng.integers(0, len(data_seeds), size=(draws, len(data_seeds)))
    rows = []
    for candidate_index, group in paired.groupby("candidate_index", sort=True):
        grid = (
            group.pivot(index="prefix_repeat_seed", columns="data_seed", values="candidate_minus_tied_bpb")
            .reindex(index=prefix_seeds, columns=data_seeds)
            .to_numpy(dtype=float)
        )
        if not np.isfinite(grid).all():
            raise ValueError(f"Candidate {candidate_index} has an incomplete crossed grid")
        samples = grid[prefix_draws[:, :, None], data_draws[:, None, :]].mean(axis=(1, 2))
        mean_effect = float(grid.mean())
        low, high = np.quantile(samples, [0.025, 0.975])
        mean_candidate_bpb = float(group.candidate_bpb.mean())
        rows.append(
            {
                "candidate_index": int(candidate_index),
                "mean_candidate_bpb": mean_candidate_bpb,
                "mean_tied_bpb": float(group.tied_bpb.mean()),
                "mean_candidate_minus_tied_bpb": mean_effect,
                "crossed_bootstrap_ci95_low_bpb": float(low),
                "crossed_bootstrap_ci95_high_bpb": float(high),
                "mean_candidate_minus_historical_frontier_bpb": mean_candidate_bpb - FRONTIER_BPB,
                "paired_wins_out_of_9": int((grid < 0.0).sum()),
                "primary_promotion_gate_passed": bool(candidate_index == 0 and mean_effect < 0.0 and high < 0.0),
                "claim_status": "primary" if candidate_index == 0 else "descriptive_secondary",
            }
        )
    return pd.DataFrame(rows)


def report_markdown(summary: pd.DataFrame) -> str:
    primary = summary[summary.candidate_index.eq(0)].iloc[0]
    verdict = "passes" if bool(primary.primary_promotion_gate_passed) else "does not pass"
    lines = [
        "# Proportional-prefix branch confirmation",
        "",
        (
            f"The preregistered primary candidate {verdict} the paired tied-continuation gate. "
            f"Its mean candidate-minus-tied effect is {primary.mean_candidate_minus_tied_bpb:+.6f} BPB "
            f"with crossed-bootstrap 95% interval "
            f"[{primary.crossed_bootstrap_ci95_low_bpb:+.6f}, {primary.crossed_bootstrap_ci95_high_bpb:+.6f}]."
        ),
        "",
        (
            "Candidates 1 and 2 are descriptive robustness checks. The comparison with the historical frontier "
            "is also descriptive because it is not paired to these prefix-data blocks. Prefixes were trained on "
            "v5p and continuations on v6e, so a canonical frontier claim requires same-hardware confirmation."
        ),
        "",
        "| candidate | mean BPB | candidate - tied | crossed 95% CI | paired wins | status |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.candidate_index} | {row.mean_candidate_bpb:.6f} | "
            f"{row.mean_candidate_minus_tied_bpb:+.6f} | "
            f"[{row.crossed_bootstrap_ci95_low_bpb:+.6f}, {row.crossed_bootstrap_ci95_high_bpb:+.6f}] | "
            f"{row.paired_wins_out_of_9}/9 | {row.claim_status} |"
        )
    return "\n".join(lines) + "\n"


def analyze(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    contract = scoring_contract(args.contract)
    results, _, coverage = validate_inputs(args)
    paired = paired_differences(results)
    summary = crossed_bootstrap(
        paired,
        draws=int(contract["bootstrap_draws"]),
        seed=int(contract["bootstrap_seed"]),
    )
    primary = summary[summary.candidate_index.eq(int(contract["primary_candidate_index"]))].iloc[0]
    report: dict[str, object] = {
        "contract_version": "delphi_phase1_proportional_prefix_confirmation_analysis_20260826_v1",
        "status": "complete",
        "target": "Uncheatable BPB",
        "estimand": "candidate minus paired tied continuation; negative is better",
        "primary_candidate_index": int(contract["primary_candidate_index"]),
        "primary_promotion_gate_passed": bool(primary.primary_promotion_gate_passed),
        "canonical_frontier_claim_allowed": False,
        "canonical_frontier_blocker": "Prefix training used v5p while continuation training used v6e.",
        "historical_frontier_bpb": FRONTIER_BPB,
        "bootstrap": {
            "draws": int(contract["bootstrap_draws"]),
            "seed": int(contract["bootstrap_seed"]),
            "unit": "independently resampled prefix-seed and data-seed clusters",
        },
        "inputs": {
            "results_sha256": file_sha256(args.results),
            "coverage_sha256": file_sha256(args.coverage),
            "design_summary_sha256": file_sha256(args.design_summary),
            "design_manifest_sha256": file_sha256(args.design_manifest),
            "contract_sha256": file_sha256(args.contract),
            "runtime_manifest_sha256": coverage["manifest_sha256"],
        },
        "candidates": summary.to_dict(orient="records"),
    }
    return paired, summary, report


def main() -> None:
    args = parse_args()
    paired, summary, report = analyze(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = args.output_dir / "paired_differences.csv"
    summary_path = args.output_dir / "candidate_summary.csv"
    report_path = args.output_dir / "report.json"
    markdown_path = args.output_dir / "report.md"
    write_bytes_exact(paired_path, paired.to_csv(index=False).encode())
    write_bytes_exact(summary_path, summary.to_csv(index=False).encode())
    write_bytes_exact(report_path, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    write_bytes_exact(markdown_path, report_markdown(summary).encode())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
