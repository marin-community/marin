# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Every model's predicted loss at every trained proposal, beside the proposals' measured losses (Qwen3 360M/1.6B, 3e18).

The four proposals are MARINER's, Olmix's (matched policy), the natural cubic spline's and tuned RegMix's, each
trained with three trainer seeds. Predictions come from the full-swarm fits that
``predict_delphi_path_midpoints_20260912`` froze before the Figure 10 midpoints were trained (its cached component
fits; nothing is refitted), at the runtime weights of ``plot_comparator_proposal_diagnostics_20260909``. The script
checks that those weights are the trained runs' weights and that the predictions reproduce the archived path
endpoints. Measured Uncheatable losses are the fixed byte-share weighting of each run's seven components.

Writes ``cross_predictions.csv`` and ``measured_runs.csv`` under
``reference_outputs/proposal_cross_predictions_20261024``.

usage: OMP_NUM_THREADS=1 uv run --offline --no-sync --with lightgbm==4.7.0 python -m \\
    experiments.domain_phase_mix.exploratory.two_phase_many.collect_proposal_cross_predictions_20261024
"""

import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    predict_delphi_path_midpoints_20260912 as midpoints,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.uncheatable_objective import (
    fixed_uncheatable_bpb,
    uncheatable_weights,
)

REFERENCE = Path(__file__).resolve().parent / "reference_outputs"
OUT = REFERENCE / "proposal_cross_predictions_20261024"
MODELS = ("mariner", "olmix", "spline", "lgbm")  # predictor and proposal keys of the path figure
TARGETS = ("uncheatable", "table9")
# The fingerprint directory of the frozen fits (prediction_freeze_<target>.json names it for both objectives).
FREEZE = (
    midpoints.DEFAULT_OUTPUT / "prediction_freeze" / "78b225d045b6d42ed175dd91d5b06cebecca7e7b04ad07102597fd447809337c"
)
# Result folder and candidate-weight table of each launch, as collect_delphi_3e18_validation_results_20260906
# registers them (that module imports its siblings as top-level modules, so it is not imported here).
LAUNCHES = {
    "frozen_procedure": (
        "delphi_frozen_procedure_validation_3e18_20260908",
        "delphi_corrected_screen_20260908/materialized_flat15_nocap/runtime_materialization/candidate_weights.csv",
    ),
    "fairness_repeats": (
        "delphi_fairness_repeats_3e18_20260908",
        "delphi_fairness_repeats_3e18_20260908/candidate_weights.csv",
    ),
    "matched_olmix": ("delphi_matched_olmix_3e18_20260908", "delphi_matched_olmix_3e18_20260908/candidate_weights.csv"),
    "comparator_proposals": (
        "delphi_comparator_proposals_3e18_20260909",
        "delphi_comparator_proposals_3e18_20260909/candidate_weights.csv",
    ),
    "table5_repeats": (
        "delphi_table5_repeats_3e18_20260928",
        "delphi_table5_repeats_3e18_20260928/candidate_weights.csv",
    ),
}
# The trained runs of each proposal: (target, proposal, launch, candidate id, trainer seeds).
RUNS = (
    ("uncheatable", "mariner", "frozen_procedure", "lwspu_u_snc_cap06", (0,)),
    ("uncheatable", "mariner", "fairness_repeats", "lwspu_u_snc_cap06", (1, 2)),
    ("uncheatable", "olmix", "matched_olmix", "olmixq_u_kl0p05_cap04", (0, 1, 2)),
    ("uncheatable", "spline", "comparator_proposals", "cmp_u_spline_cap08", (0,)),
    ("uncheatable", "spline", "table5_repeats", "cmp_u_spline_cap08", (1, 2)),
    ("uncheatable", "lgbm", "comparator_proposals", "cmp_u_lgbm_cap08", (0,)),
    ("uncheatable", "lgbm", "table5_repeats", "cmp_u_lgbm_cap08", (1, 2)),
    ("table9", "mariner", "frozen_procedure", "lwspu_t9_snc_cap08", (0,)),
    ("table9", "mariner", "fairness_repeats", "lwspu_t9_snc_cap08", (1, 2)),
    ("table9", "olmix", "matched_olmix", "olmixq_t9_kl0p005_cap04", (0, 1, 2)),
    ("table9", "spline", "comparator_proposals", "cmp_t9_spline_cap12", (0,)),
    ("table9", "spline", "table5_repeats", "cmp_t9_spline_cap12", (1, 2)),
    ("table9", "lgbm", "comparator_proposals", "cmp_t9_lgbm_cap12", (0,)),
    ("table9", "lgbm", "table5_repeats", "cmp_t9_lgbm_cap12", (1, 2)),
)


def run_loss(result: pd.Series, target: str) -> float:
    if target == "table9":
        return float(result["table9_macro_bpb"])
    metrics = {
        key: result["uncheatable_" + key.removeprefix("eval/uncheatable_eval/").removesuffix("/bpb") + "_bpb"]
        for key in uncheatable_weights()
    }
    return fixed_uncheatable_bpb(metrics)


def measured_runs(buckets: tuple[str, ...], weights: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    """Each trained run's loss, after checking that it trained the weights the predictors see."""
    rows = []
    for target, proposal, name, candidate_id, seeds in RUNS:
        folder, weight_table = LAUNCHES[name]
        results = pd.read_csv(REFERENCE / folder / "measured_results.csv")
        results = results[results["status"].eq("measured") & results["candidate_id"].eq(candidate_id)]
        tables = pd.read_csv(REFERENCE / weight_table)
        trained = tables[tables["candidate_id"].eq(candidate_id)].drop_duplicates("domain").set_index("domain")
        if not np.allclose(trained.loc[list(buckets), "weight"].to_numpy(float), weights[target][proposal], atol=1e-12):
            raise ValueError(f"{name}/{candidate_id}: trained weights differ from the predicted proposal")
        for seed in seeds:
            match = results[results["trainer_seed"].eq(seed)] if "trainer_seed" in results else results
            if len(match) != 1:
                raise ValueError(f"{name}/{candidate_id}: {len(match)} measured runs for trainer seed {seed}")
            rows.append(
                {
                    "target": target,
                    "proposal": proposal,
                    "launch": name,
                    "candidate_id": candidate_id,
                    "trainer_seed": seed,
                    "loss": run_loss(match.iloc[0], target),
                }
            )
    return pd.DataFrame(rows)


def cross_predictions(panel, weights: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    """Every model's prediction at every proposal, checked against the archived path endpoints."""
    swarm, _ = midpoints.paths.proposals.reference_swarm_in_panel_order(panel)
    archived = pd.read_csv(midpoints.DEFAULT_FIGURE_CSV)
    rows = []
    for target in TARGETS:
        predictors = midpoints.fit_predictors(panel, swarm, target, FREEZE / target)
        for predictor in MODELS:
            for proposal in MODELS:
                prediction = float(predictors[predictor](weights[target][proposal][None])[0])
                rows.append({"target": target, "predictor": predictor, "proposal": proposal, "prediction": prediction})
    table = pd.DataFrame(rows).set_index(["target", "predictor", "proposal"]).prediction
    # Path position 0 is MARINER's proposal and position 1 the baseline's.
    for target in TARGETS:
        for baseline in MODELS[1:]:
            for predictor in ("mariner", baseline):
                path = archived[archived.target.eq(target) & archived.comparator.eq(baseline)]
                path = path[path.predictor.eq(predictor)].set_index("position").prediction
                for position, proposal in ((0.0, "mariner"), (1.0, baseline)):
                    if abs(path.loc[position] - table[target, predictor, proposal]) > 1e-9:
                        raise ValueError(f"{target}/{predictor} at {proposal}: archived endpoint not reproduced")
    return table.reset_index()


def main() -> None:
    if importlib.metadata.version("lightgbm") != "4.7.0":
        raise ValueError("Use the audited LightGBM 4.7.0 environment")
    OUT.mkdir(parents=True, exist_ok=True)
    panel = midpoints.paths.bench.load_panel(midpoints.paths.proposals.PANEL)
    weights = midpoints.paths.proposal_weights(tuple(panel.buckets))
    measured = measured_runs(tuple(panel.buckets), weights)
    predictions = cross_predictions(panel, weights)
    measured.to_csv(OUT / "measured_runs.csv", index=False)
    predictions.to_csv(OUT / "cross_predictions.csv", index=False)
    means = measured.groupby(["target", "proposal"]).loss.mean().rename("measured")
    wide = predictions.pivot_table(index=["target", "proposal"], columns="predictor", values="prediction")
    print(wide.join(means)[[*MODELS, "measured"]].round(4).to_string())


if __name__ == "__main__":
    main()
