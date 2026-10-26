# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""What does MARINER propose when OlmoBaseEval Easy is restricted to its question-answering components?

Refits MARINER's reference procedure on the frozen 3e18 swarm for two equal-weight objectives, the 15 question-answering
components (ten lm-eval-overlap tasks and five answer-selection tasks) and those plus the four MMLU categories, and
optimizes each without a cap, as `analyze_table9_snr_subset_optima_20260928.py` does for SNR thresholds. Reports each
optimum's distance to the full optimum and to Olmix's matched OlmoBaseEval Easy mixture, predicted losses on every
objective at every policy, the repeat SD of each objective over the ten 3e18 proportional repeats, the largest weight
moves, and the per-component predicted changes.

Writes `policies.csv`, `component_predictions.csv` and `summary.json` to
`reference_outputs/table9_qa_subset_optimum_20260929/`. Nothing is launched.

usage: uv run --offline --no-sync python analyze_table9_qa_subset_optimum_20260929.py
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_THREAD_LIMIT"] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[3]))

import materialize_delphi_comparator_proposals_20260909 as mat  # noqa: E402

ms = mat.ms
REFERENCE = SCRIPT_DIR / "reference_outputs"
OUTPUT = REFERENCE / "table9_qa_subset_optimum_20260929"
NOISE_MATRIX = REFERENCE / "delphi_3e18_proportional_noise_floor_20260703" / "noise_component_matrix.csv"
OLMIX_SOLUTIONS = REFERENCE / "delphi_matched_olmix_3e18_20260908" / "solutions.csv"
OLMIX_POLICY = "olmixq_t9_kl0p005_cap04"
TARGET = "table9"
QA_LM_EVAL = (
    "arc_challenge",
    "arc_easy",
    "csqa",
    "hellaswag",
    "lambada",
    "medmcqa",
    "piqa",
    "sciq",
    "socialiqa",
    "winogrande",
)
QA_ANSWER_SELECTION = ("coqa", "drop", "jeopardy", "naturalqs", "squad")
MMLU = ("mmlu_humanities", "mmlu_other", "mmlu_social_sciences", "mmlu_stem")
SUBSETS = {"table9_qa": QA_LM_EVAL + QA_ANSWER_SELECTION, "table9_qa_mmlu": QA_LM_EVAL + QA_ANSWER_SELECTION + MMLU}
TOP_MOVES = 8


def tv(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum() / 2)


def optimum(fit: ms.ObjectiveFit, swarm: ms.Swarm) -> np.ndarray:
    weights, _ = ms.continuous_optimum(fit, swarm.weights)
    counts, _ = ms.runtime_policy(fit, weights)
    return counts / ms.MIXTURE_BLOCK_SIZE


def short_name(component: str) -> str:
    return component.split("/")[2] if "/" in component else component


def component_keys(full: ms.Objective, names: tuple[str, ...]) -> tuple[str, ...]:
    by_name = {short_name(c): c for c in full.components}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f"components not in the objective: {missing}")
    return tuple(by_name[n] for n in names)


def repeat_sd(names: tuple[str, ...]) -> float:
    """Repeat SD of the equal-weight mean over the ten 3e18 proportional repeats."""
    matrix = pd.read_csv(NOISE_MATRIX, index_col=0)
    matrix.columns = [c.split("/")[2] for c in matrix.columns]
    return float(matrix[list(names)].mean(axis=1).std(ddof=1))


def olmix_policy(buckets: tuple[str, ...]) -> np.ndarray:
    table = pd.read_csv(OLMIX_SOLUTIONS)
    rows = table[table.candidate_id == OLMIX_POLICY].set_index("bucket")
    return rows.loc[list(buckets), "runtime"].to_numpy(float)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = mat.bench.load_panel(mat.PANEL)
    swarm, _ = mat.reference_swarm_in_panel_order(panel)
    full = ms.read_objectives(ms.DATA / "objectives.csv")[TARGET]
    anchors = ms.read_anchors(ms.DATA / "anchors.csv", TARGET)
    folds = ms.final_inner_folds(swarm, pd.read_csv(ms.DATA / "splits.csv"))
    snr = {c: float(swarm.outcomes[c].std(ddof=1) / anchors[c].repeat_sd) for c in full.components}
    natural = 1.0 / swarm.inventory
    natural = natural / natural.sum()

    started = time.monotonic()
    full_fit = ms.fit_objective(swarm, full, anchors, folds)
    full_policy = optimum(full_fit, swarm)
    parity = tv(full_policy, mat.reference_policy(mat.MARINER_POLICY[TARGET], swarm.buckets))
    print(
        f"full objective: optimum reproduces the paper's policy to TV {parity:.4f} ({time.monotonic() - started:.0f}s)"
    )
    if parity > mat.PARITY_TV:
        raise ValueError(f"reference MARINER fit does not reproduce {mat.MARINER_POLICY[TARGET]}: TV {parity}")
    olmix = olmix_policy(swarm.buckets)
    uncheatable = mat.reference_policy(mat.MARINER_POLICY["uncheatable"], swarm.buckets)

    fits = {"full": full_fit}
    policies = {"full": full_policy}
    members = {"full": tuple(short_name(c) for c in full.components)}
    for name, subset in SUBSETS.items():
        kept = component_keys(full, subset)
        objective = ms.Objective(name, kept, np.full(len(kept), 1.0 / len(kept)))
        started = time.monotonic()
        fits[name] = ms.fit_objective(swarm, objective, anchors, folds)
        policies[name] = optimum(fits[name], swarm)
        members[name] = subset
        print(f"{name}: {len(kept)} components ({time.monotonic() - started:.0f}s)", flush=True)
    policies["olmix_matched"] = olmix

    swarm_tv = lambda policy: float(np.abs(swarm.weights - policy[None, :]).sum(axis=1).min() / 2)  # noqa: E731
    sds = {name: repeat_sd(names) for name, names in members.items()}
    summary: dict[str, object] = {
        "reference_policy_parity_tv": parity,
        "objective_repeat_sd": sds,
        "subset_mean_snr": {
            name: float(np.mean([snr[k] for k in component_keys(full, names)])) for name, names in members.items()
        },
        "policies": {},
    }
    for name, policy in policies.items():
        record = {
            "tv_to_full_optimum": tv(policy, full_policy),
            "tv_to_olmix_matched": tv(policy, olmix),
            "tv_to_uncheatable_optimum": tv(policy, uncheatable),
            "tv_to_proportional": tv(policy, natural),
            "tv_to_nearest_swarm_run": swarm_tv(policy),
            "max_epochs": float((policy * swarm.inventory).max()),
            "active_buckets": int((policy > 0).sum()),
        }
        for objective, fit in fits.items():
            record[f"predicted_{objective}"] = float(fit.predict(policy[None])[0])
            record[f"predicted_{objective}_minus_full_optimum"] = float(
                fit.predict(policy[None])[0] - fit.predict(full_policy[None])[0]
            )
        if name in fits and name != "full":
            record["own_objective_gain_over_full_optimum"] = -record[f"predicted_{name}_minus_full_optimum"]
            record["own_objective_gain_in_repeat_sd"] = record["own_objective_gain_over_full_optimum"] / sds[name]
            record["own_objective_gain_over_olmix"] = float(
                fits[name].predict(olmix[None])[0] - fits[name].predict(policy[None])[0]
            )
        summary["policies"][name] = record
    for name in SUBSETS:
        moves = pd.Series(policies[name] - full_policy, index=list(swarm.buckets))
        summary[f"{name}_largest_moves_vs_full"] = {
            bucket: {"full": float(full_policy[i]), name: float(policies[name][i]), "delta": float(moves[bucket])}
            for bucket, i in (
                (b, list(swarm.buckets).index(b)) for b in moves.abs().sort_values(ascending=False).index[:TOP_MOVES]
            )
        }

    pd.DataFrame(policies, index=list(swarm.buckets)).rename_axis("bucket").to_csv(OUTPUT / "policies.csv")
    rows = []
    for name, policy in policies.items():
        tasks = full_fit.predict_tasks(policy[None])[0]
        for component, value in zip(full.components, tasks, strict=True):
            group = (
                "qa"
                if short_name(component) in SUBSETS["table9_qa"]
                else "mmlu"
                if short_name(component) in MMLU
                else "other"
            )
            rows.append(
                {
                    "policy": name,
                    "component": short_name(component),
                    "group": group,
                    "snr": snr[component],
                    "predicted_bpb": float(value),
                }
            )
    table = (
        pd.DataFrame(rows)
        .pivot(index=["component", "group", "snr"], columns="policy", values="predicted_bpb")
        .reset_index()
    )
    for name in policies:
        if name != "full":
            table[f"{name}_minus_full"] = table[name] - table["full"]
    table.sort_values(["group", "snr"]).to_csv(OUTPUT / "component_predictions.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "policies"}, indent=1))
    print(pd.DataFrame(summary["policies"]).T.to_string(float_format=lambda v: f"{v:.4f}"))
    print(table[table.group.ne("other")].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
