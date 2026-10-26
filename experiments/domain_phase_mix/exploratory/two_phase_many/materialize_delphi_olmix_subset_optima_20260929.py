# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Olmix's optima for the two subset objectives that MARINER's subset optima were trained on (four-panel figure).

Reuses the matched Olmix policies' machinery (`materialize_delphi_matched_olmix_20260908.py`: the per-task log-linear
laws fitted on the frozen Qwen3 3e18 swarm, the exact KL-regularized proposer with per-bucket caps at four epochs,
and the 1/2048 runtime grid) with the objective restricted to a subset: the broad-language Uncheatable trio (AO3,
BBC News, Wikipedia, with the weights of `delphi_three_component_optimum_3e18_20260908/objectives.csv`) at KL 0.05,
and the 15 question-answering components of OlmoBaseEval Easy with equal weights at KL 0.005, the KL weights of the
matched full optima. Reports each solution's distance to Olmix's full optimum and to MARINER's subset optimum and
the predicted objectives.

Writes `candidate_weights.csv` (+ sha256), `solutions.csv` and `summary.json` to
`reference_outputs/delphi_olmix_subset_optima_3e18_20260929/`. Nothing is launched.

usage: uv run --offline --with cvxpy python materialize_delphi_olmix_subset_optima_20260929.py
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[3]))

import materialize_delphi_matched_olmix_20260908 as matched  # noqa: E402

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base  # noqa: E402
from experiments.domain_phase_mix import olmix_loglinear_fit as olmix  # noqa: E402
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import TOP_LEVEL_DOMAIN_TOKEN_COUNTS  # noqa: E402

REFERENCE = SCRIPT_DIR / "reference_outputs"
OUTPUT = REFERENCE / "delphi_olmix_subset_optima_3e18_20260929"
MATCHED = REFERENCE / "delphi_matched_olmix_3e18_20260908"
TRIO_OBJECTIVE = REFERENCE / "delphi_three_component_optimum_3e18_20260908" / "objectives.csv"
MARINER_SUBSET = {
    "uncheatable": (
        REFERENCE / "delphi_three_component_optimum_3e18_20260908" / "candidate_weights.csv",
        "lwspu_w3_snc_cap06",
    ),
    "table9": (REFERENCE / "delphi_table9_qa_subset_3e18_20260929" / "candidate_weights.csv", "qa15_t9_cap04"),
}
OLMIX_FULL = {"uncheatable": "olmixq_u_kl0p05_cap04", "table9": "olmixq_t9_kl0p005_cap04"}
QA = (
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
    "coqa",
    "drop",
    "jeopardy",
    "naturalqs",
    "squad",
)
CASES = (("olmixq_w3_kl0p05_cap04", "uncheatable", 0.05), ("olmixq_qa15_kl0p005_cap04", "table9", 0.005))
SURROGATE = "olmix_loglinear_d001_qwen3e18"


def laws(target: str) -> tuple[list[str], list[olmix.OlmixLoglinearFit]]:
    records = json.loads((MATCHED / f"laws_{target}.json").read_text())
    return [r["component"] for r in records], [
        olmix.OlmixLoglinearFit(r["log_c"], tuple(r["coefficients"]), r["huber_loss"]) for r in records
    ]


def subset_weights(target: str, components: list[str]) -> np.ndarray:
    if target == "uncheatable":
        table = pd.read_csv(TRIO_OBJECTIVE).set_index("component")["weight"]
        weights = np.asarray([float(table.get(c, 0.0)) for c in components])
    else:
        names = [c.split("/")[2] if "/" in c else c for c in components]
        weights = np.asarray([1.0 / len(QA) if n in QA else 0.0 for n in names])
    if abs(weights.sum() - 1.0) > 1e-9 or int((weights > 0).sum()) not in (3, 15):
        raise ValueError(f"{target}: subset weights sum to {weights.sum()} over {(weights > 0).sum()} components")
    return weights


def policy_from_table(path: Path, candidate_id: str, buckets: list[str]) -> np.ndarray:
    table = pd.read_csv(path)
    rows = table[table.candidate_id.eq(candidate_id)].set_index("domain")["weight"]
    return rows.loc[buckets].to_numpy(float)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = matched.benchmark.read_npz(matched.FROZEN_PANEL)
    buckets = [str(b) for b in panel["buckets"]]
    launcher_order = [buckets.index(domain) for domain in base.DOMAIN_NAMES]
    tokens = np.asarray([TOP_LEVEL_DOMAIN_TOKEN_COUNTS[b] for b in buckets], float)
    natural = tokens / tokens.sum()
    caps = np.minimum(1.0, tokens * matched.EPOCH_CAP / matched.TARGET_BUDGET)
    solutions = pd.read_csv(MATCHED / "solutions.csv")
    rows, summary, frames = [], {}, []
    for candidate_id, target, kl_reg in CASES:
        components, fits = laws(target)
        if list(map(str, panel[f"{target}_components"])) != components:
            raise ValueError(f"{target}: law components differ from the panel's")
        full_weights = np.asarray(panel[f"{target}_aggregation_weights"], float)
        weights = subset_weights(target, components)
        continuous, status = matched.solve_exact(fits, weights, natural, caps, kl_reg)
        counts = matched.quantize_under_cap(continuous, tokens)
        runtime = counts / matched.MIXTURE_BLOCK_SIZE
        epochs = matched.TARGET_BUDGET * runtime / tokens
        olmix_full = (
            solutions[solutions.candidate_id.eq(OLMIX_FULL[target])]
            .set_index("bucket")
            .loc[buckets, "runtime"]
            .to_numpy(float)
        )
        mariner_subset = policy_from_table(*MARINER_SUBSET[target], buckets)
        summary[candidate_id] = {
            "target": target,
            "kl_reg": kl_reg,
            "solver": status,
            "subset_components": int((weights > 0).sum()),
            "predicted_subset_runtime": matched.predict_objective(fits, weights, runtime),
            "predicted_subset_at_olmix_full_optimum": matched.predict_objective(fits, weights, olmix_full),
            "predicted_subset_at_mariner_subset_optimum": matched.predict_objective(fits, weights, mariner_subset),
            "predicted_full_runtime": matched.predict_objective(fits, full_weights, runtime),
            "predicted_full_at_olmix_full_optimum": matched.predict_objective(fits, full_weights, olmix_full),
            "tv_to_olmix_full_optimum": float(np.abs(runtime - olmix_full).sum() / 2),
            "tv_to_mariner_subset_optimum": float(np.abs(runtime - mariner_subset).sum() / 2),
            "tv_runtime_to_continuous": float(np.abs(runtime - continuous).sum() / 2),
            "max_epochs_runtime": float(epochs.max()),
            "active_buckets_runtime": int((counts > 0).sum()),
            "buckets_at_cap_runtime": int((epochs > matched.EPOCH_CAP - 0.05).sum()),
            "largest_weights": {buckets[i]: float(runtime[i]) for i in np.argsort(-runtime)[:6]},
        }
        frames.append(
            pd.DataFrame(
                {
                    "candidate_id": candidate_id,
                    "bucket": buckets,
                    "continuous": continuous,
                    "runtime": runtime,
                    "olmix_full": olmix_full,
                    "mariner_subset": mariner_subset,
                }
            )
        )
        for index in launcher_order:
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "target": target,
                    "target_label": matched.TARGET_LABELS[target],
                    "epoch_cap": matched.EPOCH_CAP,
                    "surrogate": SURROGATE,
                    "domain": buckets[index],
                    "runtime_count": int(counts[index]),
                    "weight": float(runtime[index]),
                    "materialized_epochs": float(epochs[index]),
                }
            )
        print(candidate_id, json.dumps(summary[candidate_id], indent=1), flush=True)
    with (OUTPUT / "candidate_weights.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256((OUTPUT / "candidate_weights.csv").read_bytes()).hexdigest()
    (OUTPUT / "candidate_weights.sha256").write_text(f"{digest}  candidate_weights.csv\n")
    pd.concat(frames).to_csv(OUTPUT / "solutions.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("sha256", digest)


if __name__ == "__main__":
    main()
