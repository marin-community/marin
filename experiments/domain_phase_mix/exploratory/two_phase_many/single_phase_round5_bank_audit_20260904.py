# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Bank residual audit and aggregation checks behind the round-5 report.

Correlates the frozen successor's Table-9 residuals on the development bank (observed minus predicted, positive =
optimistic) with mixture shares, and checks that the harness macro equals the W&B macro and that the sweep's stored
components equal the W&B evaluations. Diagnostic only: nothing is fitted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round3_heldout_selection_20260903 as selection,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round5_olmix_gap_20260904 as gap,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round5_remedies_20260904 as remedies,
)

REGRESSORS = (
    "arc_easy",
    "arc_challenge",
    "mmlu_stem",
    "mmlu_humanities",
    "mmlu_social_sciences",
    "mmlu_other",
    "socialiqa",
    "drop",
    "jeopardy",
    "naturalqs",
    "sciq",
    "lambada",
    "medmcqa",
    "basic_skills_common_knowledge",
    "basic_skills_logical_reasoning",
)
SHARE_COLUMNS = ("share_stack", "share_cc_low", "share_cc_high", "share_synth_qa", "share_olmocr", "share_curated")
MACRO_KEY = "macro"


def aggregation_audit(evals: Path, components: tuple[str, ...]) -> pd.DataFrame:
    table = pd.read_csv(evals, index_col=0)
    values = gap.evaluation_table(evals, components)
    sweep = pd.read_csv(gap.SWEEP_DIR / "measured_table9_components.csv")
    rows = []
    for run in table.index:
        row = {"run": run, "macro_gap": float(abs(values.loc[run].mean() - float(table.loc[run, MACRO_KEY])))}
        if run.endswith("_orig"):
            cap = int(run.split("_cap")[1].split("_")[0])
            stored = sweep[sweep["candidate_id"].eq(f"wspu_table9_cap{cap:02d}")].set_index("component")["bpb"]
            observed = pd.Series(values.loc[run].to_numpy(), index=[gap.short_name(c) for c in components])
            row["sweep_component_gap"] = float((stored - observed.reindex(stored.index)).abs().max())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument("--evals", type=Path, required=True, help="CSV of W&B Table-9 component BPBs per run")
    parser.add_argument("--shard-dir", type=Path, default=harness.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=gap.DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    harness.HELDOUT_DIR = args.registry_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = harness.load_panel(gap.PANEL)
    components = tuple(panel.group("table9").components)
    short = [gap.short_name(component) for component in components]
    bank = selection.load_bank(panel, "table9")
    _frame, features = harness.heldout_features(panel, "table9")
    predicted = remedies.shard_matrix(args.shard_dir, panel, bank.coordinate_id)
    observed = remedies.bank_components(panel, bank)
    residual = pd.DataFrame(observed - predicted, columns=short)
    frame = remedies.descriptors(features.weights, panel, bank.distance)
    frame["macro_resid"] = residual.mean(axis=1).to_numpy()
    frame["regressor_resid"] = residual[list(REGRESSORS)].mean(axis=1).to_numpy()
    code = [name for name in short if gap.family(f"x/{name}/bpb") == "code"]
    frame["code_resid"] = residual[code].mean(axis=1).to_numpy()
    frame["stack_edu_epochs"] = features.exposures[:, list(panel.buckets).index("dolma3_stack_edu")]
    frame["coordinate_id"] = bank.coordinate_id
    frame.to_csv(args.output_dir / "bank_residual_audit.csv", index=False)

    rows = []
    x = np.column_stack([np.ones(len(frame)), frame[list(SHARE_COLUMNS[:5])].to_numpy(float)])
    for target in ("macro_resid", "regressor_resid", "code_resid"):
        row = {"residual": target, "mean": float(frame[target].mean())}
        row.update({f"corr_{column}": float(frame[target].corr(frame[column])) for column in SHARE_COLUMNS})
        beta = np.linalg.lstsq(x, frame[target].to_numpy(float), rcond=None)[0]
        row.update(
            {f"ols_{name}": float(value) for name, value in zip(("const", *SHARE_COLUMNS[:5]), beta, strict=True)}
        )
        high_stack = frame["stack_edu_epochs"] >= 5.0
        row["mean_at_stack_edu_ge5"] = float(frame.loc[high_stack, target].mean())
        rows.append(row)
    correlations = pd.DataFrame(rows)
    correlations.to_csv(args.output_dir / "bank_residual_correlations.csv", index=False)
    audit = aggregation_audit(args.evals, components)
    audit.to_csv(args.output_dir / "aggregation_audit.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"bank {len(bank.measured)} coordinates (matched-seed coordinates included; diagnostic only)")
    print(correlations.round(4).to_string(index=False))
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
