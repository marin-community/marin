# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derived tables: MARINER's fitted parameters and predictions, reliability, learning curves, proposal predictions.

The orchestrator writes every table as Parquet. Runs are referenced by observation id, keyed as the run tables key the
same runs; tasks by the paper's task ids (`naming.task_public`); objectives as `obe_easy` (OlmoBaseEval Easy) and
`uncheatable`, and the subsets some proposals optimized as `obe_easy_qa15`, `obe_easy_snr5`, `uncheatable_worsened`, ...

- `fits_task_parameters`: MARINER's reference fit to the Qwen3 360M/1.6B swarm, one row per objective and task: benefit
  `rate` and `power`, harm `threshold`, `ridge`, the floor multiplier `kappa` (the paper's gamma), the `floor` and the
  inner cross-validated RMSE (mixture-selection's `reference_task_fits.csv`). The archived swarms' fits exist in no
  source; the release's mixture-selection regeneration fits and appends them.
- `fits_swarm_predictions`: the reference fit's per-task predictions at the swarm's 280 fit mixtures. The fit trained on
  these runs, so these are in-sample fitted values, not out-of-fold predictions: mixture-selection's self-test compares
  them with `fit.predict_tasks(swarm.weights)` of the full fit. The bundle names each run by its hash-less training run
  name, as `swarm_qwen3` keys its fit rows.
- `fits_heldout_predictions`: the reference fit's per-task predictions at the held-out mixtures of mixture-selection's
  bank (247 OlmoBaseEval Easy and 408 Uncheatable mixtures of the Qwen3 held-out registry), which are the MARINER
  predictions of the held-out analysis figures. The bank keys each prediction by registry coordinate (one exposure);
  every held-out run at that coordinate carries it, keyed by `heldout.observation_keys`.
- `fits_reliability`: per-task signal-to-noise ratios and the out-of-fold fit quality of MARINER, Olmix and tuned RegMix
  over ten fold assignments of the 279 scored swarm mixtures (`table9_reliability_mariner_20260908`, the source of the
  appendix's SNR and fit tables). `source_table` names the block: OlmoBaseEval Easy components, its collapsed tasks, or
  the Uncheatable corpora; aggregate rows and collapsed tasks have no `task_id`.
- `fits_learning_curves`: the per-draw metrics of every surrogate at every swarm size from three learning-curve studies
  (`study`), with `target` as the objective and per-task strata as task ids. A selection from the held-out bank names
  the selected mixture by `selected_exposure_id`, the `heldout` table's exposure id of that registry coordinate.
- `fits_proposal_predictions`: the predictions that the validation launches' folders record for trained proposals, at
  the trained (runtime-rounded) mixtures: the proposing model's own (`own_prediction`) and, where a folder compares
  models, every other model's. A prediction belongs to its proposal's mixture, so every `validation_3e18` run of that
  mixture carries it, under that table's observation id, launch and variant. `predictor` names models as the learning
  curves do (`mariner`, `olmix`, `quadratic`, `spline`, `regmix` for the tuned LightGBM trees, `regmix_official` for the
  released recipe and `regmix_official_components` for its per-component diagnostic, `krr`, `mariner_cvx`,
  `mariner_add`, `mariner_pbt`, `mariner_mk1`); MARINER's development variants keep their registry names. The
  epoch-cap and KL sweep recorded Olmix's predictions at continuous optima only, and no folder records predictions for
  the native Olmix proposals, the QA-subset proposals (relative to MARINER only), the best swarm run or UniMax-8; those
  runs have no rows.
- `fits_olmix_task_laws`: the matched Olmix baseline's per-task log-linear laws on the Qwen3 swarm, one row per
  objective and task: BPB(w) = exp(log_c) + exp(sum over buckets b of coef__b * w_b), fitted to the 280 fit runs with
  a Huber loss (delta 0.01; `huber_loss` is the fit's loss) (`delphi_matched_olmix_3e18_20260908/laws_*.json`). The
  matched Olmix proposals minimized their aggregate under the `tasks` objective weights, and the held-out analysis
  evaluates single laws at held-out mixtures for Olmix's per-task predictions. The JSON lists coefficients in the
  folder's bucket order, which its `solutions.csv` records.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.release import naming, paths, tables
from experiments.domain_phase_mix.release.configs import heldout, validation
from experiments.domain_phase_mix.release.context import ExportContext

SWARM = "qwen3_360m_1p6b"
PANEL = {swarm: panel for panel, swarm in heldout.SWARM.items()}[SWARM]  # the held-out registry's name of the swarm
MIXSEL_OBJECTIVES = ("table9", "uncheatable")  # mixture-selection's names, which name its files
PUBLIC_OBJECTIVES = {
    "obe_easy",
    "uncheatable",
    "uncheatable_worsened",
    "obe_easy_qa15",
    "obe_easy_qa19",
    "obe_easy_snr2",
    "obe_easy_snr5",
    "obe_easy_snr8",
}
DATA = paths.MIXSEL / "data"
TASK_FITS = DATA / "reference_task_fits.csv"
SWARM_RUNS = DATA / "swarm_weights.csv"
SWARM_PREDICTIONS = "reference_swarm_predictions"
BANK_PREDICTIONS = "reference_bank_predictions"
BANK = "heldout_bank"
RELIABILITY = paths.RO / "table9_reliability_mariner_20260908"
# Source table -> spelling of the metric its `component` column names (the collapsed-task table has no components).
RELIABILITY_TABLES = {
    "snr_fit_components_delphi": "table9/{}/bpb",
    "snr_fit_tasks_delphi": None,
    "snr_fit_uncheatable_delphi": "eval/uncheatable_eval/{}/bpb",
}
LEARNING_CURVES = {
    "mariner_regmix_oof": paths.RO / "learning_curve_mariner_delphi_3e18_20260908/regmix_oof_20260924/metrics_long.csv",
    "pbt": paths.RO / "learning_curve_pbt_delphi_3e18_20260929/metrics_long.csv",
    "extra": paths.RO / "learning_curve_extra_delphi_3e18_20261006/metrics_long.csv",
}
COMPONENT_MEAN = "component_mean"  # the per-task evaluation's stratum that averages the tasks

# Proposal predictions. Runtime materializations record the proposing surrogate's prediction at each trained policy.
MATERIALIZATIONS = [
    paths.RO / f"{folder}/runtime_materialization/candidate_mapping.csv"
    for folder in (
        "delphi_link_validation_3e18_20260906",
        "delphi_link_hub_validation_3e18_20260906",
        "delphi_coupling_validation_3e18_20260906",
        "delphi_kappa_floor_validation_3e18_20260907",
        "delphi_kappa_floor_validation_kl05_3e18_20260907",
        "delphi_kappa_floor_flat15_validation_3e18_20260907",
        "delphi_corrected_screen_20260908/materialized_flat15_nocap",  # the frozen procedure's candidates
    )
]
MARINER_SURROGATE = "weibull_softplus_unscaled@kappa_floor_link_flat15_nocap"  # the frozen procedure
COUPLED_POLICY = r"(?P<model>wspu_coupled)_(?:uncheatable|table9)_cap\d+_(?P<kappa>kappa[0-9p]+)_kl0"
# Folders comparing proposals: cross_predictions_<objective>.csv holds every model's (column) prediction of every
# model's proposal (row); the candidate ids name each proposal's model.
COMPARISONS = [
    paths.RO / "delphi_comparator_proposals_3e18_20260909",
    paths.RO / "delphi_convex_additive_proposals_3e18_20260914",
    paths.RO / "delphi_per_bucket_threshold_proposals_3e18_20260924",
]
COMPARISON_FILES = ("candidate_weights.csv", "cross_predictions_uncheatable.csv", "cross_predictions_table9.csv")
COMPARISON_CANDIDATE = r"cmp_(?P<objective>u|t9)_(?P<code>[a-z0-9]+)_cap\d+"
MODEL_CODES = {
    "mariner": "mariner",
    "quad": "quadratic",
    "spline": "spline",
    "lgbm": "regmix",  # LightGBM tuned as RegMix
    "krr": "krr",  # Hellinger kernel ridge
    "olmix": "olmix",
    "mk1": "mariner_mk1",  # MARINER with Weibull power 1 (an exponential benefit)
    "cvx": "mariner_cvx",  # fully convex objective
    "add": "mariner_add",  # additive response
    "pbt": "mariner_pbt",  # per-bucket thresholds
}
# The frozen MARINER proposals, which the comparisons call `mariner` (their own predictions agree with the frozen
# materialization's, which checks the pairing).
FROZEN_MARINER = {"uncheatable": "lwspu_u_snc_cap06", "table9": "lwspu_t9_snc_cap08"}
KL_ABLATION = paths.RO / "delphi_kl_ablation_3e18_20260908"
KL_FILES = ("policy_summary.csv", "kl_ablation_summary.csv")
MATCHED_OLMIX_DIR = paths.RO / "delphi_matched_olmix_3e18_20260908"
MATCHED_OLMIX = MATCHED_OLMIX_DIR / "summary.json"
OLMIX_LAWS = [MATCHED_OLMIX_DIR / f"laws_{name}.json" for name in MIXSEL_OBJECTIVES]
OLMIX_SOLUTIONS = MATCHED_OLMIX_DIR / "solutions.csv"  # bucket order of the laws' coefficients
OLMIX_LAW_FIELDS = {"component", "log_c", "coefficients", "huber_loss"}
OLMIX_SUBSET = paths.RO / "delphi_olmix_subset_optima_3e18_20260929/summary.json"
REGMIX_ARGMIN = paths.RO / "regmix_tuned_argmin_3e18_20260930/summary.json"
SNR_SUBSET = paths.RO / "delphi_table9_snr_subset_3e18_20260928/subset_validation.csv"
MIDPOINTS = paths.RO / "delphi_path_midpoints_3e18_20260912/prediction_freeze/midpoint_predictions.csv"
REGMIX_REFERENCE = paths.RO / "regmix_official_rerun_20260913"
REGMIX_REFERENCE_FILES = ("candidate_weights.csv", "cross_predictions.csv")
# The rerun's predictors: the released recipe's objective head and its per-component diagnostic, the tuned trees
# (`prior_only` keeps them and changes only the sampler) and MARINER.
REGMIX_REFERENCE_PREDICTORS = {
    "official_objective": "regmix_official",
    "official_components": "regmix_official_components",
    "prior_only": "regmix",
    "mariner": "mariner",
}
THREE_COMPONENT = paths.RO / "delphi_three_component_optimum_3e18_20260908"
THREE_COMPONENT_FILES = (
    "mixtures_for_prediction.csv",
    "candidate_weights.csv",
    "predictions_uncheatable_worsened.csv",
    "predictions_uncheatable.csv",
)
FACTORIAL = paths.RO / "factorial_vs_mariner_20260928/predicted_vs_measured.csv"
AGREEMENT = 1e-12  # two folders' predictions of one proposal by one model


def required_sources() -> list[Path]:
    """The files the export reads itself; it also runs the `heldout` and `validation` exporters on their sources."""
    data = [
        DATA / f"{stem}_{name}.csv" for stem in (SWARM_PREDICTIONS, BANK_PREDICTIONS, BANK) for name in MIXSEL_OBJECTIVES
    ]
    proposals = [
        *MATERIALIZATIONS,
        *(folder / name for folder in COMPARISONS for name in COMPARISON_FILES),
        *(KL_ABLATION / name for name in KL_FILES),
        MATCHED_OLMIX,
        *OLMIX_LAWS,
        OLMIX_SOLUTIONS,
        OLMIX_SUBSET,
        REGMIX_ARGMIN,
        SNR_SUBSET,
        MIDPOINTS,
        *(REGMIX_REFERENCE / name for name in REGMIX_REFERENCE_FILES),
        *(THREE_COMPONENT / name for name in THREE_COMPONENT_FILES),
        FACTORIAL,
    ]
    return sorted(
        {
            TASK_FITS,
            SWARM_RUNS,
            *data,
            heldout.EXPANDED / "heldout_runs.csv",
            *(RELIABILITY / f"{table}.csv" for table in RELIABILITY_TABLES),
            *LEARNING_CURVES.values(),
            *proposals,
        }
    )


def _csv(path: Path, **kwargs) -> pd.DataFrame:
    return tables.read_csv(paths.require(path), **kwargs)


def _json(path: Path) -> dict:
    return json.loads(paths.require(path).read_text())


def _public_objective(name: str) -> str:
    """The public name of an objective from a source's spelling, in which `table9` is OlmoBaseEval Easy."""
    public = re.sub(r"^table9", "obe_easy", name)
    if public not in PUBLIC_OBJECTIVES:
        raise ValueError(f"unknown objective {name!r}")
    return public


def _task_parameters() -> pd.DataFrame:
    t = _csv(TASK_FITS)
    t["task_id"] = t.component.map(naming.task_public)
    t["objective"] = t.objective.map(_public_objective)
    t["swarm"] = SWARM
    return t[["swarm", "objective", "task_id", "rate", "power", "threshold", "ridge", "kappa", "floor", "inner_cv_rmse"]]


def _long_tasks(wide: pd.DataFrame, key: str) -> pd.DataFrame:
    """Per-task predictions with one column per internal metric, as (key, task_id, predicted_bpb) rows.

    Every column but the key must name one of the paper's tasks; `naming.task_public` raises on any other.
    """
    tasks = {column: naming.task_public(column) for column in wide.columns.drop(key)}
    return wide.rename(columns=tasks).melt(id_vars=[key], var_name="task_id", value_name="predicted_bpb")


def _swarm_predictions(ctx: ExportContext) -> pd.DataFrame:
    runs = _csv(SWARM_RUNS, usecols=["run"]).run
    frames = []
    for name in MIXSEL_OBJECTIVES:
        wide = _csv(DATA / f"{SWARM_PREDICTIONS}_{name}.csv")
        assert wide.run.is_unique and set(wide.run) == set(runs), f"{name}: predictions cover other runs than the swarm"
        long = _long_tasks(wide, key="run")
        long.insert(0, "objective", _public_objective(name))
        long.insert(1, "observation_id", long.pop("run").map(ctx.registry.id_for))
        frames.append(long)
    return pd.concat(frames, ignore_index=True)


def _heldout_runs(ctx: ExportContext, heldout_table: pd.DataFrame) -> pd.DataFrame:
    """Coordinate, observation id and exposure id of each Qwen3 held-out run; a registry coordinate is one exposure."""
    runs = _csv(
        heldout.EXPANDED / "heldout_runs.csv", usecols=["panel", "row_id", "training_wandb_run_id", "coordinate_id"]
    )
    runs = runs[runs.panel == PANEL]
    frame = pd.DataFrame(
        {
            "coordinate_id": runs.coordinate_id,
            "observation_id": heldout.observation_keys(runs).map(ctx.registry.id_for),
        }
    )
    frame = frame.merge(
        heldout_table[["observation_id", "exposure_id"]], on="observation_id", how="left", validate="one_to_one"
    )
    assert frame.exposure_id.notna().all(), "every Qwen3 held-out run is in the heldout table"
    assert (frame.groupby("coordinate_id").exposure_id.nunique() == 1).all(), "a coordinate spans several exposures"
    return frame


def _heldout_predictions(runs: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for name in MIXSEL_OBJECTIVES:
        run_count = _csv(DATA / f"{BANK}_{name}.csv", usecols=["coordinate_id", "run_count"]).set_index("coordinate_id")
        wide = _csv(DATA / f"{BANK_PREDICTIONS}_{name}.csv")
        assert wide.coordinate_id.is_unique and set(wide.coordinate_id) == set(run_count.index)
        per_run = runs[["coordinate_id", "observation_id"]].merge(
            wide, on="coordinate_id", how="inner", validate="many_to_one"
        )
        counts = per_run.coordinate_id.value_counts().reindex(run_count.index)
        assert (counts == run_count.run_count).all(), f"{name}: the bank counts other runs per mixture than the registry"
        long = _long_tasks(per_run.drop(columns="coordinate_id"), key="observation_id")
        long.insert(0, "objective", _public_objective(name))
        frames.append(long)
    return pd.concat(frames, ignore_index=True)


def _reliability() -> pd.DataFrame:
    frames = []
    for table, spelling in RELIABILITY_TABLES.items():
        f = _csv(RELIABILITY / f"{table}.csv")
        if spelling is not None:  # aggregate rows leave `component` empty
            task_id = f.pop("component").map(lambda c, s=spelling: naming.task_public(s.format(c)), na_action="ignore")
            f.insert(0, "task_id", task_id)
        f.insert(0, "source_table", table)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def _learning_curves(exposures: pd.Series) -> pd.DataFrame:
    """`exposures`: the exposure id of each held-out registry coordinate."""
    frames = []
    for study, path in LEARNING_CURVES.items():
        f = _csv(path)
        f.insert(0, "study", study)
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df["target"] = df.target.map(_public_objective)
    per_task = (df.evaluation == "oof_component") & (df.stratum != COMPONENT_MEAN)
    df.loc[per_task, "stratum"] = df.stratum[per_task].map(naming.task_public)
    unknown = set(df.selected_coordinate_id.dropna()) - set(exposures.index)
    if unknown:
        raise ValueError(f"{len(unknown)} selected mixtures are not held-out registry coordinates")
    df["selected_coordinate_id"] = df.selected_coordinate_id.map(exposures)
    df = df.rename(columns={"selected_coordinate_id": "selected_exposure_id"})
    return df.astype({"n_basin": "Int64", "n_failed": "Int64", "basin_hit": "boolean"})


def _row(variant: str, objective: str, predictor: str, own: bool, predicted: float) -> dict:
    return {
        "variant": variant,
        "objective": objective,
        "predictor": predictor,
        "own_prediction": own,
        "predicted_bpb": predicted,
    }


def _olmix_task_laws(ctx: ExportContext) -> pd.DataFrame:
    """One row per objective and task; the producer wrote solutions.csv's bucket column from the bucket list that orders
    the laws' coefficients."""
    orders = (
        _csv(OLMIX_SOLUTIONS, usecols=["candidate_id", "bucket"]).groupby("candidate_id", sort=False).bucket.agg(tuple)
    )
    if orders.nunique() != 1:
        raise ValueError(f"{OLMIX_SOLUTIONS}: candidates list buckets in different orders")
    buckets = [naming.bucket_public(b) for b in orders.iloc[0]]
    if sorted(buckets) != sorted(ctx.buckets):
        raise ValueError(f"{OLMIX_SOLUTIONS}: buckets differ from the partition's")
    rows = []
    for path, name in zip(OLMIX_LAWS, MIXSEL_OBJECTIVES, strict=True):
        for law in _json(path):
            if set(law) != OLMIX_LAW_FIELDS or len(law["coefficients"]) != len(buckets):
                raise ValueError(f"{path}: unexpected law for {law.get('component')}")
            coefficients = dict(zip((f"coef__{b}" for b in buckets), law["coefficients"], strict=True))
            rows.append(
                {
                    "swarm": SWARM,
                    "objective": _public_objective(name),
                    "task_id": naming.task_public(law["component"]),
                    "log_c": law["log_c"],
                    "huber_loss": law["huber_loss"],
                    **coefficients,
                }
            )
    laws = pd.DataFrame(rows)
    assert not laws.duplicated(["objective", "task_id"]).any(), "two laws for one task"
    return laws[["swarm", "objective", "task_id", "log_c", "huber_loss", *(f"coef__{b}" for b in ctx.buckets)]]


def _rows(variant, objective, predictor, own, predicted) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant": variant,
            "objective": objective,
            "predictor": predictor,
            "own_prediction": own,
            "predicted_bpb": predicted,
        }
    )


def _materialization(path: Path) -> pd.DataFrame:
    """The proposing surrogate's prediction at each trained policy; the coupled-WSPU launch names its model (coupling
    kappa) in each policy id rather than in a surrogate column."""
    m = _csv(path)
    if "surrogate" in m:
        predictor = m.surrogate.replace({MARINER_SURROGATE: "mariner"})
    else:
        parts = m.policy_id.str.extract(f"^{COUPLED_POLICY}$")
        if parts.isna().any().any():
            raise ValueError(f"{path}: policy ids without a coupled model")
        predictor = parts.model.str.cat(parts.kappa, sep="_")
    return _rows(m.candidate_id, m.target, predictor, True, m.runtime_prediction)


def _comparison(folder: Path) -> pd.DataFrame:
    candidates = _csv(folder / "candidate_weights.csv", usecols=["candidate_id"]).candidate_id.drop_duplicates()
    parts = candidates.str.extract(f"^{COMPARISON_CANDIDATE}$")
    if parts.isna().any().any() or parts.duplicated().any():
        raise ValueError(f"{folder.name}: candidate ids do not name one model per objective")
    frames = []
    for objective, prefix in (("uncheatable", "u"), ("table9", "t9")):
        mine = parts.objective == prefix
        variants = dict(zip(parts.code[mine], candidates[mine], strict=True)) | {"mariner": FROZEN_MARINER[objective]}
        matrix = _csv(folder / f"cross_predictions_{objective}.csv").set_index("proposal")
        long = matrix.rename_axis(columns="model").stack().rename("predicted_bpb").reset_index()
        unknown = (set(long.proposal) - set(variants)) | (set(long.model) - set(MODEL_CODES))
        if unknown:
            raise ValueError(f"{folder.name}: unknown models {sorted(unknown)}")
        own = long.proposal == long.model
        frames.append(
            _rows(long.proposal.map(variants), objective, long.model.map(MODEL_CODES), own, long.predicted_bpb)
        )
    return pd.concat(frames, ignore_index=True)


def _kl_ablation() -> pd.DataFrame:
    """MARINER's prediction, without the penalty, of each KL-penalized proposal: the summary records it by objective and
    KL weight, the policy summary names each weight's policy; both record the penalized objective, which pairs them."""
    policies, summary = (_csv(KL_ABLATION / name) for name in KL_FILES)
    joined = policies.merge(
        summary, left_on=["target", "kl_coefficient"], right_on=["target", "kl"], how="left", validate="one_to_one"
    )
    if not np.allclose(joined.penalized, joined.penalized_objective, rtol=0, atol=AGREEMENT):
        raise ValueError("the KL ablation's summary and policies record different penalized objectives")
    return _rows(joined.candidate_id, joined.target, "mariner", True, joined.predicted_raw)


def _matched_olmix() -> pd.DataFrame:
    summary = _json(MATCHED_OLMIX)
    return pd.DataFrame(
        [_row(variant, s["target"], "olmix", True, s["predicted_runtime"]) for variant, s in summary.items()]
    )


def _olmix_subset(objectives: pd.Series) -> pd.DataFrame:
    """Olmix's predictions of the subset objective each proposal optimized (`objectives`, by variant) and of the full
    objective."""
    rows = []
    for variant, s in _json(OLMIX_SUBSET).items():
        rows.append(_row(variant, objectives[variant], "olmix", True, s["predicted_subset_runtime"]))
        rows.append(_row(variant, s["target"], "olmix", True, s["predicted_full_runtime"]))
    return pd.DataFrame(rows)


def _regmix_argmin() -> pd.DataFrame:
    """The tuned trees' and MARINER's predictions at the MILP solutions that name a candidate."""
    rows = []
    for objective, entry in _json(REGMIX_ARGMIN).items():
        if entry["target"] != objective:
            raise ValueError(f"{REGMIX_ARGMIN}: {objective} entry targets {entry['target']}")
        for p in entry["predictions"].values():
            if "candidate_id" in p:
                rows.append(_row(p["candidate_id"], objective, "regmix", True, p["trees_runtime"]))
                rows.append(_row(p["candidate_id"], objective, "mariner", False, p["mariner_runtime"]))
    return pd.DataFrame(rows)


def _snr_subset(objectives: pd.Series) -> pd.DataFrame:
    """MARINER's predictions of the full objective and of each SNR subset at the subset proposals and at the frozen
    proposal (`full`); the file names a proposal by the subset it optimized (`objectives`, by variant)."""
    variants = {objective: variant for variant, objective in objectives.items()} | {"obe_easy": FROZEN_MARINER["table9"]}
    assert len(variants) == len(objectives) + 1, "two subset proposals optimized one objective"
    table = _csv(SNR_SUBSET)
    policy = table.policy.replace({"full": "table9"}).map(lambda p: variants[_public_objective(p)])
    return _rows(policy, table.subset.replace({"full": "table9"}), "mariner", True, table.predicted)


def _midpoints() -> pd.DataFrame:
    """MARINER's and the blended baseline's predictions at each trained 50% blend; a blend has no proposing model."""
    m = _csv(MIDPOINTS)
    m = m[m.coordinate == "runtime"]
    unknown = set(m.predictor) - set(MODEL_CODES)
    if unknown:
        raise ValueError(f"{MIDPOINTS}: unknown models {sorted(unknown)}")
    return _rows(m.candidate_id, m.target, m.predictor.map(MODEL_CODES), False, m.prediction_bpb)


def _regmix_reference() -> pd.DataFrame:
    """The rerun's predictions at its trained endpoints and midpoints (policy `official_objective_<role>`) and at the
    frozen MARINER proposals (`mariner`); only an endpoint has a proposing model, the released recipe."""
    weights, predictions = (_csv(REGMIX_REFERENCE / name) for name in REGMIX_REFERENCE_FILES)
    trained = weights[["candidate_id", "target", "surrogate", "role"]].drop_duplicates()
    if not (trained.surrogate == "regmix_official_objective").all():
        raise ValueError(f"{REGMIX_REFERENCE}: trained candidates of another surrogate")
    policies = pd.concat(
        [
            pd.DataFrame(
                {
                    "variant": trained.candidate_id,
                    "target": trained.target,
                    "policy": "official_objective_" + trained.role,
                }
            ),
            pd.DataFrame(
                {"variant": list(FROZEN_MARINER.values()), "target": list(FROZEN_MARINER), "policy": "mariner"}
            ),
        ]
    )
    proposer = {"official_objective_endpoint": "official_objective", "mariner": "mariner"}
    joined = policies.merge(predictions, on=["target", "policy"], how="left", validate="one_to_many")
    unknown = set(joined.predictor) - set(REGMIX_REFERENCE_PREDICTORS)
    if joined.prediction.isna().any() or unknown:
        raise ValueError(f"{REGMIX_REFERENCE}: missing or unknown predictions ({sorted(map(str, unknown))})")
    own = joined.policy.map(proposer) == joined.predictor
    return _rows(
        joined.variant, joined.target, joined.predictor.map(REGMIX_REFERENCE_PREDICTORS), own, joined.prediction
    )


def _three_component() -> pd.DataFrame:
    """MARINER's predictions of the worsened-trio and full Uncheatable objectives at the trained three-component optimum
    and at the frozen Uncheatable proposal (`full_uncheatable_optimum`)."""
    mixtures, weights, *predictions = (_csv(THREE_COMPONENT / name) for name in THREE_COMPONENT_FILES)
    (variant,) = weights.candidate_id.unique()
    trained = weights.set_index("domain").weight
    predicted_at = mixtures.set_index("mixture").T["three_component_optimum"].reindex(trained.index)
    if not (predicted_at == trained).all():
        raise ValueError(f"{THREE_COMPONENT}: the predicted mixture is not the trained one")
    variants = {"three_component_optimum": variant, "full_uncheatable_optimum": FROZEN_MARINER["uncheatable"]}
    frames = []
    for p, objective in zip(predictions, ("uncheatable_worsened", "uncheatable"), strict=True):
        p = p[p.mixture.isin(variants)]
        frames.append(_rows(p.mixture.map(variants), objective, "mariner", True, p[f"predicted_{objective}"]))
    return pd.concat(frames, ignore_index=True)


def _factorial() -> pd.DataFrame:
    """MARINER's predictions at the factorial's centre and corners; the design has no proposing model."""
    p = _csv(FACTORIAL)
    return _rows(p.candidate_id, p.target, "mariner", False, p.predicted_bpb)


def _proposal_objectives(validation_table: pd.DataFrame, launch: str) -> pd.Series:
    """The public name of the objective each of a launch's proposals optimized, by variant."""
    rows = validation_table[validation_table.launch == launch].drop_duplicates("variant").set_index("variant")
    return rows.proposal_objective.map(_public_objective)


def _proposal_predictions(validation_table: pd.DataFrame) -> pd.DataFrame:
    predictions = pd.concat(
        [
            *(_materialization(path) for path in MATERIALIZATIONS),
            *(_comparison(folder) for folder in COMPARISONS),
            _kl_ablation(),
            _matched_olmix(),
            _olmix_subset(_proposal_objectives(validation_table, "olmix_subset")),
            _regmix_argmin(),
            _snr_subset(_proposal_objectives(validation_table, "table9_snr_subset")),
            _midpoints(),
            _regmix_reference(),
            _three_component(),
            _factorial(),
        ],
        ignore_index=True,
    )
    predictions["objective"] = predictions.objective.map(_public_objective)
    # Several folders predict some proposals with the same model (the frozen MARINER proposals above all); they agree.
    key = ["variant", "objective", "predictor"]
    spread = predictions.groupby(key).agg(
        low=("predicted_bpb", "min"), high=("predicted_bpb", "max"), own=("own_prediction", "nunique")
    )
    conflicts = spread[(spread.high - spread.low > AGREEMENT) | (spread.own > 1)]
    if len(conflicts):
        raise ValueError(f"folders disagree on {len(conflicts)} predictions: {conflicts.index.tolist()[:3]}")
    predictions = predictions.drop_duplicates(key)

    # A prediction belongs to its proposal's mixture: attach it to every validation run of that mixture. Proposals that
    # no launch trained (alternatives in the same files) drop out.
    assert (validation_table.groupby("variant").mixture_id.nunique() == 1).all(), "a variant names several mixtures"
    mixtures = validation_table.drop_duplicates("variant").set_index("variant").mixture_id
    predictions["mixture_id"] = predictions.pop("variant").map(mixtures)
    runs = validation_table[["observation_id", "launch", "variant", "mixture_id"]]
    out = runs.merge(predictions.dropna(subset=["mixture_id"]), on="mixture_id", how="inner", validate="many_to_many")
    assert not out.duplicated(["observation_id", "objective", "predictor"]).any(), "two proposals share a run's mixture"
    return out.drop(columns="mixture_id").reset_index(drop=True)


def export(ctx: ExportContext) -> dict[str, pd.DataFrame]:
    heldout_table = heldout.export(ctx)["heldout"]
    runs = _heldout_runs(ctx, heldout_table)
    exposures = runs.drop_duplicates("coordinate_id").set_index("coordinate_id").exposure_id
    return {
        "fits_task_parameters": _task_parameters(),
        "fits_swarm_predictions": _swarm_predictions(ctx),
        "fits_heldout_predictions": _heldout_predictions(runs),
        "fits_reliability": _reliability(),
        "fits_learning_curves": _learning_curves(exposures),
        "fits_proposal_predictions": _proposal_predictions(validation.export(ctx)["validation_3e18"]),
        "fits_olmix_task_laws": _olmix_task_laws(ctx),
    }
