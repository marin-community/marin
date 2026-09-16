# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Paired comparison of the trained comparator proposals with MARINER's seed-matched runs at 3e18.

Reads the collector's ``measured_results.csv`` for the comparator launch
(`collect_delphi_3e18_validation_results_20260906.py --launch comparator_proposals`) and MARINER's runs of the same
policies at the same data and trainer seeds (frozen-procedure validation for trainer seed 0, fairness repeats for
seeds 1 and 2). Every comparator run is paired with MARINER's run at its trainer seed; candidates with three seeds
report the mean and SE of the paired differences, single-seed candidates report the one difference, which the
paper labels descriptive. A proposal that coincides with MARINER's mixture (the Uncheatable power-one twin) was
not trained again and repeats MARINER's runs. Optimism is the measured loss minus the proposer's own prediction.

Writes ``paired_results.csv`` and ``paired_rows.tex`` (rows for the appendix table) next to the proposals.

usage: uv run --offline --no-sync python summarize_delphi_comparator_proposals_20260909.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# One OpenMP thread: see materialize_delphi_comparator_proposals_20260909.py.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_THREAD_LIMIT"] = "1"

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_delphi_comparator_proposals_20260909 as proposals,
)

REFERENCE = SCRIPT_DIR / "reference_outputs"
PROPOSAL_DIR = proposals.OUTPUT
# MARINER's measured runs of the reference policies: trainer seed 0 from the frozen-procedure validation, seeds 1
# and 2 from the fairness repeats (same data seeds 666200 and 662009 as the comparator runs).
MARINER_RESULTS = (
    REFERENCE / "delphi_frozen_procedure_validation_3e18_20260908/measured_results.csv",
    REFERENCE / "delphi_fairness_repeats_3e18_20260908/measured_results.csv",
)
LADDER = REFERENCE / "complexity_ladder_20260909/complexity_ladder.csv"
# The matched Olmix policies (capped at four epochs, KL-penalized) trained at the same seeds, for the same table.
OLMIX_DIR = REFERENCE / "delphi_matched_olmix_3e18_20260908"
OLMIX_POLICY = {"uncheatable": "olmixq_u_kl0p05_cap04", "table9": "olmixq_t9_kl0p005_cap04"}
OLMIX_MODEL_ID = "olmix_loglinear_taskwise"
OLMIX_LABEL = "Olmix"
MARINER_MODEL_ID = "weibull_softplus_unscaled@kappa_floor_link_flat15_nocap"
NONPARAMETRIC_FAMILY = "nonparametric"
OBJECTIVE_COLUMN = {"uncheatable": "uncheatable_bpb", "table9": "table9_macro_bpb"}
TARGET_LABELS = {"uncheatable": "Uncheatable", "table9": "OlmoBaseEval Easy"}
COMPARATOR_LABELS = {
    "quad": "Quadratic in log-epochs, floor link",
    "spline": "Natural cubic spline in log-epochs, floor link",
    "lgbm": "LightGBM (RegMix)",
    "krr": "Hellinger kernel ridge",
    "mk1": "MARINER, exponential benefit",
    "cvx": "MARINER, fully convex objective (convex harm)",
    "add": "MARINER, additive response",
}
# The second batch (2026-09-14): proposals and measured runs live in separate packages.
CONVEX_ADDITIVE_PROPOSALS = REFERENCE / "delphi_convex_additive_proposals_3e18_20260914"
CONVEX_ADDITIVE_MEASURED = REFERENCE / "delphi_convex_additive_3e18_20260914" / "measured_results.csv"
# Nominal parameters per task for models absent from the ladder: the convex harm keeps MARINER's 2M+5, the additive
# response drops the floor multiplier (2M+4), as in Table 1.
EXTRA_PARAMETERS = {f"{MARINER_MODEL_ID}_raw_epoch_hinge": 83, "weibull_softplus_unscaled": 82}
DEFAULT_TRAINER_SEED = 0


@dataclass(frozen=True)
class PairedRow:
    candidate_id: str
    target: str
    comparator: str
    parameters: int | None  # nominal per task at M=39; None for nonparametric models
    seeds: int
    own_prediction: float
    mariner_prediction: float
    measured_mean: float
    measured_sd: float
    difference_mean: float
    difference_se: float
    optimism: float


def measured_runs(path: Path, target_column: str) -> pd.DataFrame:
    """Measured runs of one launch with the objective in column ``measured`` and integer trainer seeds."""
    table = pd.read_csv(path)
    table = table[table.status == "measured"].copy()
    if "trainer_seed" not in table.columns:
        table["trainer_seed"] = DEFAULT_TRAINER_SEED
    table["trainer_seed"] = table["trainer_seed"].fillna(DEFAULT_TRAINER_SEED).astype(int)
    table["measured"] = [float(row[OBJECTIVE_COLUMN[row[target_column]]]) for _, row in table.iterrows()]
    return table.dropna(subset=["measured"])[["candidate_id", "target", "trainer_seed", "measured"]]


def nominal_parameters() -> dict[str, int | None]:
    """Nominal response parameters per task by model id from the complexity ladder; None for nonparametric models."""
    ladder = pd.read_csv(LADDER)
    table = {row.model: None if row.family == NONPARAMETRIC_FAMILY else int(row.params) for row in ladder.itertuples()}
    return {**EXTRA_PARAMETERS, **table}


def mariner_runs() -> dict[str, pd.Series]:
    """MARINER's measured objective per target, indexed by trainer seed."""
    frames = [measured_runs(path, "target") for path in MARINER_RESULTS]
    table = pd.concat(frames, ignore_index=True)
    out = {}
    for target, policy in proposals.MARINER_POLICY.items():
        rows = table[(table.candidate_id == policy) & (table.target == target)]
        series = rows.set_index("trainer_seed")["measured"].sort_index()
        if series.index.duplicated().any():
            raise ValueError(f"duplicate MARINER seeds for {policy}: {series.index.tolist()}")
        out[target] = series
    return out


def paired_row(
    candidate_id: str,
    target: str,
    comparator: str,
    parameters: int | None,
    series: pd.Series,
    reference: pd.Series,
    own_prediction: float,
    mariner_prediction: float,
) -> PairedRow:
    """Summary of one proposal's runs (indexed by trainer seed) against MARINER's runs at the same seeds."""
    differences = series - reference.loc[series.index]
    several = len(series) > 1
    return PairedRow(
        candidate_id=candidate_id,
        target=target,
        comparator=comparator,
        parameters=parameters,
        seeds=len(series),
        own_prediction=own_prediction,
        mariner_prediction=mariner_prediction,
        measured_mean=float(series.mean()),
        measured_sd=float(series.std(ddof=1)) if several else math.nan,
        difference_mean=float(differences.mean()),
        difference_se=float(differences.std(ddof=1) / math.sqrt(len(series))) if several else math.nan,
        optimism=float(series.mean() - own_prediction),
    )


def paired_rows(
    comparators: pd.DataFrame, mariner: dict[str, pd.Series], summary: dict, parameters: dict[str, int | None]
) -> list[PairedRow]:
    rows = []
    for candidate_id, group in comparators.groupby("candidate_id", sort=False):
        record = summary[candidate_id]
        series = group.set_index("trainer_seed")["measured"].sort_index()
        rows.append(
            paired_row(
                candidate_id,
                record["target"],
                COMPARATOR_LABELS[candidate_id.split("_")[2]],
                parameters[record["model_id"]],
                series,
                mariner[record["target"]],
                float(record["self_prediction_runtime"]),
                float(record["mariner_prediction_runtime"]),
            )
        )
    measured_ids = set(comparators.candidate_id)
    for candidate_id, record in summary.items():
        if candidate_id in measured_ids or record["tv_to_mariner"] > 0:
            continue
        # The proposal is MARINER's own mixture, so MARINER's runs measure it.
        series = mariner[record["target"]]
        rows.append(
            paired_row(
                candidate_id,
                record["target"],
                COMPARATOR_LABELS[candidate_id.split("_")[2]],
                parameters[record["model_id"]],
                series,
                series,
                float(record["self_prediction_runtime"]),
                float(record["mariner_prediction_runtime"]),
            )
        )
    return rows


def mariner_rows(mariner: dict[str, pd.Series], summary: dict, parameters: dict[str, int | None]) -> list[PairedRow]:
    rows = []
    for target, series in mariner.items():
        prefix = "cmp_u_" if target == "uncheatable" else "cmp_t9_"
        record = summary[next(c for c in summary if c.startswith(prefix))]
        prediction = float(record["mariner_prediction_at_mariner"])
        rows.append(
            paired_row(
                proposals.MARINER_POLICY[target],
                target,
                "MARINER",
                parameters[MARINER_MODEL_ID],
                series,
                series,
                prediction,
                prediction,
            )
        )
    return rows


def mariner_predictions_of_olmix() -> dict[str, float]:
    """MARINER's reference fit evaluated at each matched Olmix policy's runtime mixture."""
    ms = proposals.ms
    swarm = ms.read_swarm(ms.DATA / "swarm_weights.csv", ms.DATA / "swarm_outcomes.csv", ms.DATA / "buckets.csv")
    solutions = pd.read_csv(OLMIX_DIR / "solutions.csv")
    out = {}
    for target, policy in OLMIX_POLICY.items():
        weights = solutions[solutions.candidate_id == policy].set_index("bucket").loc[list(swarm.buckets), "runtime"]
        fit = proposals.reference_fit(swarm, target, None)
        out[target] = float(fit.predict(weights.to_numpy(float)[None, :])[0])
    return out


def olmix_rows(mariner: dict[str, pd.Series], parameters: dict[str, int | None]) -> list[PairedRow]:
    table = measured_runs(OLMIX_DIR / "measured_results.csv", "target")
    summary = json.loads((OLMIX_DIR / "summary.json").read_text())
    predictions = mariner_predictions_of_olmix()
    rows = []
    for target, policy in OLMIX_POLICY.items():
        series = table[(table.candidate_id == policy) & (table.target == target)]
        series = series.set_index("trainer_seed")["measured"].sort_index()
        rows.append(
            paired_row(
                policy,
                target,
                OLMIX_LABEL,
                parameters[OLMIX_MODEL_ID],
                series,
                mariner[target],
                float(summary[policy]["predicted_runtime"]),
                predictions[target],
            )
        )
    return rows


def signed(value: float, digits: int) -> str:
    return f"{value:+.{digits}f}"


def latex_row(row: PairedRow, best: bool) -> str:
    """One table row; ``best`` sets the lowest measured loss of the objective in bold."""
    measured = f"{row.measured_mean:.4f}" if row.seeds == 1 else f"${row.measured_mean:.4f} \\pm {row.measured_sd:.4f}$"
    if best:
        measured = f"{{\\boldmath{measured}}}" if measured.startswith("$") else f"\\textbf{{{measured}}}"
    if row.comparator == "MARINER":
        difference = "--"
    elif row.difference_mean == 0.0 and row.difference_se == 0.0:
        difference = "$0$"  # the proposal is MARINER's mixture
    elif row.seeds == 1:
        difference = f"${signed(row.difference_mean, 3)}$"
    else:
        difference = f"${signed(row.difference_mean, 3)} \\pm {row.difference_se:.3f}$"
    parameters = "--" if row.parameters is None else str(row.parameters)
    return (
        f"{row.comparator} & {parameters} & {row.seeds} & {row.own_prediction:.3f} & {row.mariner_prediction:.3f} & "
        f"{measured} & {difference} & ${signed(row.optimism, 3)}$ \\\\"
    )


def main() -> None:
    summary = {
        **json.loads((PROPOSAL_DIR / "summary.json").read_text()),
        **json.loads((CONVEX_ADDITIVE_PROPOSALS / "summary.json").read_text()),
    }
    comparators = pd.concat(
        [
            measured_runs(PROPOSAL_DIR / "measured_results.csv", "target"),
            measured_runs(CONVEX_ADDITIVE_MEASURED, "target"),
        ],
        ignore_index=True,
    )
    mariner = mariner_runs()
    parameters = nominal_parameters()
    rows = (
        mariner_rows(mariner, summary, parameters)
        + olmix_rows(mariner, parameters)
        + paired_rows(comparators, mariner, summary, parameters)
    )
    order = {target: index for index, target in enumerate(TARGET_LABELS)}
    rank = {"MARINER": 0, OLMIX_LABEL: 1}
    rows.sort(key=lambda row: (order[row.target], rank.get(row.comparator, 2), row.candidate_id))
    best = {target: min(row.measured_mean for row in rows if row.target == target) for target in TARGET_LABELS}
    table = pd.DataFrame([row.__dict__ for row in rows])
    table.to_csv(PROPOSAL_DIR / "paired_results.csv", index=False)
    lines = []
    for target, label in TARGET_LABELS.items():
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{label}}}}} \\\\")
        lines.extend(latex_row(row, row.measured_mean == best[target]) for row in rows if row.target == target)
    (PROPOSAL_DIR / "paired_rows.tex").write_text("\n".join(lines) + "\n")
    pending = [c for c in summary if c not in {row.candidate_id for row in rows}]
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("pending:", ", ".join(pending) or "none")


if __name__ == "__main__":
    main()
