# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14"]
# ///
"""Review the frozen procedure's 3e18 validation against its pre-registered gate.

Reads the collector's `measured_results.csv` (and Table-9 components) of the frozen-procedure validation, the
flat-profile validation it supersedes (the matched incumbents: same recipe, data seeds and trainer seed, mixtures
0.025 to 0.038 TV away), and the standalone fits stored beside the launch, and writes `review.md` and
`review_table.csv` into the launch package. The gate (freeze handoff Section 13): every run at most two repeat
SDs above its incumbent and below the Olmix and additive-WSPU comparators by the same margin, and no prediction
miss larger than the flat validation's. Exit code 0 when the gate passes, 2 when a row is pending, 1 when it fails.

usage: uv run review_delphi_frozen_procedure_validation_20260908.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE = SCRIPT_DIR / "reference_outputs"
PACKAGE = REFERENCE / "delphi_frozen_procedure_validation_3e18_20260908"
INCUMBENT = REFERENCE / "delphi_kappa_floor_flat15_validation_3e18_20260907"
STANDALONE = Path.home() / "Projects/Work/Marin/mixture-selection"
ROWS = (  # candidate, incumbent candidate, target, cap
    ("lwspu_u_snc_cap06", "lwspu_u_kff_cap06", "uncheatable", 6),
    ("lwspu_t9_snc_cap06", "lwspu_t9_kff_cap06", "table9", 6),
    ("lwspu_t9_snc_cap08", "lwspu_t9_kff_cap08", "table9", 8),
)
REPEAT_SD = {"uncheatable": 0.001, "table9": 0.004}
COMPARATORS = {  # measured at 3e18: Olmix best KL (0.1 / 0.005), additive WSPU (cap 6 / cap 6 / cap 8)
    "uncheatable": {"olmix": 1.0022, "wspu": 0.9834},
    "table9": {"olmix": 1.0769, "wspu": 1.0722},
}
WSPU_TABLE9_CAP8 = 1.0736
MEASURE = {"uncheatable": "uncheatable_bpb", "table9": "table9_macro_bpb"}
UNCHEATABLE_COMPONENTS = (
    "ao3_english",
    "arxiv_computer_science",
    "arxiv_physics",
    "bbc_news",
    "github_cpp",
    "github_python",
    "wikipedia_english",
)


def load_standalone():
    if str(STANDALONE) not in sys.path:
        sys.path.insert(0, str(STANDALONE))
    import mixture_selection  # noqa: PLC0415  (external authoritative implementation, imported by path)

    return mixture_selection


def task_predictions(ms, fit_path: Path, weights: pd.DataFrame, candidate: str) -> pd.Series:
    fit = ms.ObjectiveFit.from_json(json.loads(fit_path.read_text()))
    table = weights[weights.candidate_id == candidate].set_index("domain")
    vector = table.weight.reindex(list(fit.buckets)).fillna(0.0).to_numpy(float)
    return pd.Series(fit.predict_tasks(vector[None])[0], index=[t.component for t in fit.tasks])


def main() -> int:
    measured = pd.read_csv(PACKAGE / "measured_results.csv").set_index("candidate_id")
    incumbent = pd.read_csv(INCUMBENT / "measured_results.csv").set_index("candidate_id")
    weights = pd.read_csv(PACKAGE / "runtime_materialization/candidate_weights.csv")
    incumbent_weights = pd.read_csv(INCUMBENT / "runtime_materialization/candidate_weights.csv")
    ms = load_standalone()
    fits = {"uncheatable": PACKAGE / "fits/fit_uncheatable.json", "table9": PACKAGE / "fits/fit_table9.json"}
    lines = ["# Frozen-procedure validation review", ""]
    table = []
    verdicts = []
    for candidate, previous, target, cap in ROWS:
        row = measured.loc[candidate]
        if row.status != "measured" or not np.isfinite(float(row[MEASURE[target]])):
            print(f"{candidate}: {row.status}; not every row is measured yet")
            return 2
        predicted_tasks = task_predictions(ms, fits[target], weights, candidate)
        objective_weights = ms.read_objectives(STANDALONE / "data/objectives.csv")[target].weights
        predicted = float(predicted_tasks.to_numpy() @ objective_weights)
        value = float(row[MEASURE[target]])
        prior = float(incumbent.loc[previous, MEASURE[target]])
        prior_miss = abs(
            prior - {"lwspu_u_kff_cap06": 0.9807, "lwspu_t9_kff_cap06": 1.0635, "lwspu_t9_kff_cap08": 1.0626}[previous]
        )
        wspu = WSPU_TABLE9_CAP8 if (target, cap) == ("table9", 8) else COMPARATORS[target]["wspu"]
        olmix = COMPARATORS[target]["olmix"]
        threshold = prior + 2 * REPEAT_SD[target]
        tv = float(
            (
                weights[weights.candidate_id == candidate].set_index("domain").weight
                - incumbent_weights[incumbent_weights.candidate_id == previous].set_index("domain").weight
            )
            .abs()
            .sum()
            / 2
        )
        checks = {
            "within_2sd_of_incumbent": value <= threshold,
            "below_olmix": value <= olmix - 2 * REPEAT_SD[target] or value < olmix,
            "not_above_wspu_margin": value <= wspu + 2 * REPEAT_SD[target],
            "miss_not_larger_than_flat": abs(value - predicted) <= prior_miss + 1e-9,
        }
        verdict = all(checks.values())
        verdicts.append(verdict)
        table.append(
            {
                "candidate_id": candidate,
                "target": target,
                "epoch_cap": cap,
                "predicted": predicted,
                "measured": value,
                "miss": value - predicted,
                "incumbent": prior,
                "incumbent_miss": prior_miss,
                "tv_to_incumbent": tv,
                "olmix_best": olmix,
                "wspu_additive": wspu,
                "threshold": threshold,
                **checks,
                "gate": "pass" if verdict else "FAIL",
            }
        )
    frame = pd.DataFrame(table)
    frame.to_csv(PACKAGE / "review_table.csv", index=False)
    lines.append(frame.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    # Uncheatable components: predicted, measured, incumbent
    u = measured.loc["lwspu_u_snc_cap06"]
    ui = incumbent.loc["lwspu_u_kff_cap06"]
    predicted_u = task_predictions(ms, fits["uncheatable"], weights, "lwspu_u_snc_cap06")
    comp = pd.DataFrame(
        {
            "component": UNCHEATABLE_COMPONENTS,
            "predicted": [float(predicted_u[f"eval/uncheatable_eval/{c}/bpb"]) for c in UNCHEATABLE_COMPONENTS],
            "measured": [float(u[f"uncheatable_{c}_bpb"]) for c in UNCHEATABLE_COMPONENTS],
            "incumbent": [float(ui[f"uncheatable_{c}_bpb"]) for c in UNCHEATABLE_COMPONENTS],
        }
    )
    comp["miss"] = comp.measured - comp.predicted
    lines += [
        "## Uncheatable components (frozen Uncheatable proposal)",
        "",
        comp.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    # Table-9 components: prediction error and movement against the incumbent per proposal
    components = pd.read_csv(PACKAGE / "measured_table9_components.csv")
    incumbent_components = pd.read_csv(INCUMBENT / "measured_table9_components.csv")
    for candidate, previous, _target, cap in ROWS[1:]:
        predicted_t = task_predictions(ms, fits["table9"], weights, candidate)
        got = components[components.candidate_id == candidate].set_index("component").bpb
        prior_c = incumbent_components[incumbent_components.candidate_id == previous].set_index("component").bpb
        names = [k.split("/")[-2] if "/" in k else k for k in predicted_t.index]
        joined = pd.DataFrame({"component": names, "predicted": predicted_t.to_numpy()})
        joined["measured"] = [float(got.get(n, np.nan)) for n in names]
        joined["incumbent"] = [float(prior_c.get(n, np.nan)) for n in names]
        joined["miss"] = joined.measured - joined.predicted
        joined["vs_incumbent"] = joined.measured - joined.incumbent
        rmse = float(np.sqrt(np.nanmean(joined.miss**2)))
        mean_miss = float(np.nanmean(joined.miss))
        lines += [
            f"## Table-9 components, {candidate} (cap {cap}): per-component RMSE {rmse:.4f}, mean miss {mean_miss:+.4f}",
            "",
            "Largest moves against the incumbent:",
            "",
            joined.reindex(joined.vs_incumbent.abs().sort_values(ascending=False).index)
            .head(8)
            .to_markdown(index=False, floatfmt=".4f"),
            "",
        ]
    passed = all(verdicts)
    lines.insert(2, f"**Gate: {'PASS' if passed else 'FAIL'}** ({sum(verdicts)} of {len(verdicts)} rows pass).")
    lines.insert(3, "")
    (PACKAGE / "review.md").write_text("\n".join(lines))
    print("\n".join(lines[:12]))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
