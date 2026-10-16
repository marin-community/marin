# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The cumulative held-out registry: frozen (September 2) and expanded (September 13) cohorts of all three swarms.

Observation ids are W&B run ids (`observation_ids`). The 116 validation-launch runs take the run directory of their
evaluation-metrics URI, as `validation_3e18` does. 679 frozen runs take the registry's W&B training run id, which for
the 187 `delphi_archive::` rows is the row id without its prefix. The registry records no W&B run id for the 134 Llama
200M runs; theirs come from W&B, frozen in LLAMA_200M_RUNS. The Llama swarm tables use the same ids, so the 39 Llama
160M deletions, rows of both tables, have one id each; the Qwen3 swarm's fit rows are in no held-out cohort.

Objectives take their public names (`naming.objective_public`): the registry's one baseline, the UniMax-8 run of the
June ladder at 3e18, has objective `none` and, as in `validation_3e18`, its construction's epoch cap of 8.

Seeds come from the launch records that `validation_3e18` reads, wherever they hold the run: each validation-launch
run's from the directory that holds it (`run_names.seeds_from_uri`), and the UniMax-8 run's from the UniMax-8 repeats'
record. The Llama 200M runs' come from their W&B configs (LLAMA_200M_RUNS), and so does the data seed of each run in
WANDB_PULL. They must agree with the registry's seeds, which it records for 42 of the 116 validation-launch runs. Every
other run keeps the registry's seeds, which are null for the other Llama 160M runs and for some frozen Qwen3 runs.

Aggregates follow `tables.add_aggregates`: recomputed from a run's components where it has them all, else the
registry's recorded aggregate where that used the fixed rule. Every run whose recorded Uncheatable aggregate uses other
weights (the Llama panels' legacy weights, or the BPB-schema-2 pooled value of the 14 path-midpoint and RegMix-reference
runs) has its Uncheatable components: the registry's, or for one archived Llama 160M run, which the registry scores by
its pooled aggregate only, those of its W&B run summary (WANDB_PULL).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.release import keys, naming, paths, run_names, tables, wandb_pulls
from experiments.domain_phase_mix.release.context import ExportContext

EXPANDED = paths.RO / "single_phase_heldout_benchmark_expanded_20260913"
FROZEN = paths.RO / "single_phase_heldout_benchmark_20260902"
FROZEN_FILES = ("heldout_runs.csv", "heldout_components.csv")
EXPANDED_FILES = ("heldout_runs.csv", "heldout_coordinate_components.csv")
# The UniMax-8 construction, which validation_3e18 and the ladder also read: a baseline training it has its epoch cap.
# No other baseline's cap is known.
UNIMAX8_WEIGHTS = paths.REPO / "experiments/domain_phase_mix/release/unimax8_weights.csv"
UNIMAX8_EPOCH_CAP = 8.0
# The UniMax-8 repeats' launch record, which validation_3e18 also reads: it names the seed-0 run with its seeds.
UNIMAX8_REPEATS = paths.RO / "delphi_unimax8_repeats_3e18_20260922/measured_results.csv"
LAUNCH_FILES = ("measured_results.csv", "measured_table9_components.csv")  # in each validation launch's folder
SWARM = {"delphi_3e18_39bucket": "qwen3_360m_1p6b", "60m_39bucket": "llama_160m_1p2b", "300m_39bucket": "llama_200m_6b"}
META = {
    "proposal_model": "proposal_method",
    "proposal_target": "proposal_objective",
    "epoch_cap": "epoch_cap",
    "data_seed": "data_seed",
    "trainer_seed": "trainer_seed",
    "subset_seed": "subset_seed",
    "phase_tv": "phase_tv",
}
SEEDS = ["data_seed", "trainer_seed"]
# The panel whose recorded Uncheatable aggregates use the paper's fixed byte shares, which the frozen registry's builder
# checked to 3e-6; the Llama panels' use the archived swarms' legacy weights. Every panel records the OlmoBaseEval Easy
# mean.
FIXED_RULE_PANEL = "delphi_3e18_39bucket"
# These launches ran after the evaluator moved to BPB schema 2 (about 12 September 2026). Their recorded Uncheatable
# aggregate is its pooled one, 0.008 to 0.011 BPB above the fixed weighting of the same runs' components
# (two_phase_many/uncheatable_objective.py); the expanded registry copied it unchecked.
POOLED_SOURCES = ("validation::path_midpoints", "validation::regmix_reference")
# Uncheatable components and data seeds that the registry lacks, frozen from W&B run summaries and configs
# (`wandb_pulls`). The pooled aggregate ties each pull to the run's registry record, which holds the same W&B value at
# full precision.
WANDB_PULL = paths.REPO / "experiments/domain_phase_mix/release/configs/heldout_wandb_components_20261015.csv"
# The W&B run ids and the seeds of the W&B configs of the 134 Llama 200M runs, which the registry records without them
# (`wandb_pulls.read_runs`). W&B shortened 20 of the ids by dropping leading characters of the run name.
LLAMA_200M_RUNS = paths.REPO / "experiments/domain_phase_mix/release/configs/heldout_llama200m_wandb_runs_20261015.csv"
LLAMA_200M_ROW = "300m::"  # the registry's prefix before a Llama 200M run's name
# Result folders of the validation launches in the expanded cohort, from the launch registry of
# collect_delphi_3e18_validation_results_20260906.py, which the expanded registry's builder read.
VALIDATION_LAUNCHES = {
    "link": "delphi_link_validation_3e18_20260906",
    "hub": "delphi_link_hub_validation_3e18_20260906",
    "coupling": "delphi_coupling_validation_3e18_20260906",
    "kappa_floor": "delphi_kappa_floor_validation_3e18_20260907",
    "kappa_floor_kl05": "delphi_kappa_floor_validation_kl05_3e18_20260907",
    "kappa_floor_flat": "delphi_kappa_floor_flat15_validation_3e18_20260907",
    "frozen_procedure": "delphi_frozen_procedure_validation_3e18_20260908",
    "kl_ablation": "delphi_kl_ablation_3e18_20260908",
    "fairness_repeats": "delphi_fairness_repeats_3e18_20260908",
    "olmix_kl005_repeats": "delphi_olmix_kl005_repeats_3e18_20260908",
    "matched_olmix": "delphi_matched_olmix_3e18_20260908",
    "comparator_proposals": "delphi_comparator_proposals_3e18_20260909",
    "path_midpoints": "delphi_path_midpoints_3e18_20260912",
    "regmix_reference": "delphi_regmix_reference_3e18_20260913",
    "three_component_optimum": "delphi_three_component_optimum_3e18_20260908",
    "factorial": "delphi_frontier_factorial_design_20260906",
}


def required_sources() -> list[Path]:
    launches = [paths.RO / folder / name for folder in VALIDATION_LAUNCHES.values() for name in LAUNCH_FILES]
    frozen = [FROZEN / name for name in FROZEN_FILES]
    expanded = [EXPANDED / name for name in EXPANDED_FILES]
    return [*frozen, *expanded, *launches, UNIMAX8_WEIGHTS, UNIMAX8_REPEATS, WANDB_PULL, LLAMA_200M_RUNS]


def _unimax8_mixture() -> str:
    weights = tables.read_csv(paths.require(UNIMAX8_WEIGHTS))
    return keys.mixture_id(dict(zip(weights.domain.map(naming.bucket_public), weights.weight, strict=True)))


def _frozen_components() -> pd.DataFrame:
    """Run-level components of the frozen cohort, which its builder checked against each run's recorded aggregates."""
    long = tables.read_csv(paths.require(FROZEN / "heldout_components.csv"))
    return pd.DataFrame({"row_id": long.row_id, "task": long.component.map(naming.task_public), "bpb": long.bpb})


def _validation_row_id(launch: str, run: str) -> str:
    """The expanded registry's key of a validation run."""
    return f"delphi_validation::{launch}::{run}"


def _launch_results(launch: str, folder: str) -> pd.DataFrame:
    """A validation launch's measured runs, indexed by the expanded registry's key of each run.

    A launch that repeats candidates over trainer seeds names each run `<candidate>@t<seed>`, otherwise `<candidate>`.
    """
    results = tables.read_csv(paths.require(paths.RO / folder / "measured_results.csv"))
    results = results[results.status == "measured"]
    run = list(results.candidate_id)
    if "trainer_seed" in results:
        run = [f"{c}@t{int(s)}" for c, s in zip(run, results.trainer_seed, strict=True)]
    return results.set_axis([_validation_row_id(launch, r) for r in run])


def _validation_components(ctx: ExportContext) -> pd.DataFrame:
    """Run-level components of the validation launches (long), keyed by the expanded registry's row_id.

    Uncheatable components are `uncheatable_<corpus>_bpb` result columns. The OlmoBaseEval Easy file is long; it names
    each run as the results do and each component by its public task id.
    """
    uncheatable_columns = {f"uncheatable_{t.removeprefix('uncheatable/')}_bpb": t for t in ctx.tasks[:7]}
    frames = []
    for launch, folder in VALIDATION_LAUNCHES.items():
        uncheatable = _launch_results(launch, folder)[list(uncheatable_columns)].rename(columns=uncheatable_columns)
        obe = tables.read_csv(paths.require(paths.RO / folder / "measured_table9_components.csv"))
        assert set(obe.component) == set(ctx.tasks[7:]), f"{launch}: unexpected OlmoBaseEval Easy components"
        rows = pd.MultiIndex.from_arrays([[_validation_row_id(launch, r) for r in obe.candidate_id], obe.component])
        both = pd.concat([uncheatable.stack(), obe.bpb.set_axis(rows)])
        frames.append(both.rename_axis(["row_id", "task"]).reset_index(name="bpb"))
    return pd.concat(frames, ignore_index=True)


def _validation_uris() -> pd.Series:
    """Evaluation-metrics URI of each validation run, indexed by the registry's row_id."""
    uris = [_launch_results(launch, folder).eval_metrics_uri for launch, folder in VALIDATION_LAUNCHES.items()]
    return pd.concat(uris)


def _validation_run_names() -> pd.Series:
    """Training run name of each validation run, from its evaluation-metrics URI, indexed by the registry's row_id."""
    return _validation_uris().map(run_names.training_run_name_from_uri)


def _llama_200m_runs() -> pd.DataFrame:
    """LLAMA_200M_RUNS by registry row id, after checking that each W&B run id names its registry run.

    The id is the run's name and six-hex-digit hash. The registry row id is LLAMA_200M_ROW and the same name, of which
    the id keeps the end where W&B shortened it. Raises ValueError for an id that ends in another name.
    """
    runs = wandb_pulls.read_runs(LLAMA_200M_RUNS)
    names = runs.index.str.removeprefix(LLAMA_200M_ROW)
    kept = runs.training_wandb_run_id.str.extract(r"^(?P<name>.+)-[0-9a-f]{6}$")["name"]
    named = [isinstance(k, str) and name.endswith(k) for name, k in zip(names, kept, strict=True)]
    if not runs.index.str.startswith(LLAMA_200M_ROW).all() or not all(named):
        raise ValueError(f"{LLAMA_200M_RUNS.name}: W&B run ids that do not end in their registry run's name")
    return runs


def _launch_seeds() -> pd.DataFrame:
    """Data and trainer seed of each run that a launch record or a W&B pull holds, indexed by W&B run id.

    A validation run's seeds come from the directory that holds it, as in `validation_3e18`; the UniMax-8 repeats'
    record gives its three runs' seeds, LLAMA_200M_RUNS the seeds in the Llama 200M runs' W&B configs, and WANDB_PULL
    the data seed in each pulled run's W&B config.
    """
    uris = _validation_uris()
    names = uris.map(run_names.training_run_name_from_uri)
    validation_seeds = pd.DataFrame([run_names.seeds_from_uri(uri) for uri in uris], index=names, columns=SEEDS)
    repeats = tables.read_csv(paths.require(UNIMAX8_REPEATS)).set_index("run")[SEEDS]
    llama = _llama_200m_runs().set_index("training_wandb_run_id")[SEEDS]
    pulled = wandb_pulls.read(WANDB_PULL).data_seed.to_frame().assign(trainer_seed=np.nan)
    seeds = pd.concat([validation_seeds, repeats, llama, pulled])
    assert seeds.index.is_unique, "a training run appears in two launch records"
    return seeds


def _with_pulled_components(out: pd.DataFrame, runs: pd.DataFrame, run_ids: pd.Series) -> pd.DataFrame:
    """`out` with the Uncheatable components of the runs in WANDB_PULL, which the registry lacks.

    Raises unless each pulled run is a held-out run without registry components for those tasks whose recorded
    Uncheatable aggregate is the pulled pooled value.
    """
    pulls = wandb_pulls.read(WANDB_PULL)
    pulled = run_ids[run_ids.isin(pulls.index)]
    if sorted(pulled) != sorted(pulls.index):
        raise ValueError(f"pulled runs that are not held-out runs: {sorted(set(pulls.index) - set(pulled))}")
    pull = pulls.loc[pulled].set_axis(pulled.index)
    columns = [c for c in pulls.columns if c.startswith("bpb__")]
    if out.loc[pulled.index, columns].notna().any().any():
        raise ValueError(f"pulled runs that already have registry components: {pulled.tolist()}")
    off = pull.pooled != runs.uncheatable_bpb[pulled.index]
    if off.any():
        raise ValueError(f"pulled pooled aggregates that differ from the registry's: {pulled[off].tolist()}")
    out = out.copy()
    out.loc[pulled.index, columns] = pull[columns]
    return out


def _seeds(runs: pd.DataFrame, run_ids: pd.Series) -> pd.DataFrame:
    """Each run's seeds: the registry's, completed from the launch record or W&B pull that holds the run.

    `run_ids` is `observation_ids(runs)`. Raises AssertionError where the registry and a launch record or W&B pull give
    different seeds.
    """
    recorded = runs[SEEDS]
    launched = _launch_seeds().reindex(run_ids).set_axis(runs.index)
    agree = (recorded == launched) | recorded.isna() | launched.isna()
    assert agree.all().all(), "the registry's seeds differ from the launch records'"
    return recorded.fillna(launched)


def _components(ctx: ExportContext, runs: pd.DataFrame) -> pd.DataFrame:
    """Per-run components, wide, after checking that they average to the expanded registry's coordinate components.

    The expanded registry publishes components per coordinate only; those means differ from the run values wherever
    seeds repeat a mixture, so the run values come from the frozen registry and from the validation launches.
    """
    long = pd.concat([_frozen_components(), _validation_components(ctx)], ignore_index=True)
    coordinate = long.row_id.map(runs.set_index("row_id").coordinate_id).rename("coordinate_id")
    means = long.groupby([coordinate, long.task]).bpb.mean()
    published = tables.read_csv(paths.require(EXPANDED / "heldout_coordinate_components.csv"))
    published = published.set_index([published.coordinate_id, published.component.map(naming.task_public)]).bpb_mean
    assert set(means.index) == set(published.index), "run-level components cover other coordinates than the registry"
    assert np.allclose(means, published.reindex(means.index), rtol=0, atol=1e-12), "components differ from the registry"
    wide = long.pivot(index="row_id", columns="task", values="bpb")
    wide.columns = [f"bpb__{t}" for t in wide.columns]
    return wide


def observation_ids(runs: pd.DataFrame) -> pd.Series:
    """Each held-out run's observation id, its W&B run id, aligned with `runs` (rows of heldout_runs.csv).

    The module docstring gives the sources; `fits` keys its held-out predictions with the same rule. Raises ValueError
    for a run without a W&B run id, or one that LLAMA_200M_RUNS gives an id the registry already records.
    """
    frozen_ids = set(tables.read_csv(paths.require(FROZEN / "heldout_runs.csv")).row_id)
    names = _validation_run_names()
    assert set(runs.row_id) - frozen_ids <= set(names.index), "every expanded run needs its training run name"
    recorded = runs.training_wandb_run_id
    assert recorded.dropna().str.fullmatch(run_names.RUN_DIRECTORY.pattern).all(), "W&B run ids must be run names"
    pulled = _llama_200m_runs().training_wandb_run_id
    if runs.row_id[recorded.notna()].isin(pulled.index).any():
        raise ValueError(f"{LLAMA_200M_RUNS.name} gives W&B run ids to runs whose registry records one")
    ids = runs.row_id.map(names).fillna(recorded).fillna(runs.row_id.map(pulled))
    if ids.isna().any():
        raise ValueError(f"held-out runs without a W&B run id: {runs.row_id[ids.isna()].tolist()[:5]}")
    return ids


def export(ctx: ExportContext) -> dict[str, pd.DataFrame]:
    runs = tables.read_csv(paths.require(EXPANDED / "heldout_runs.csv"))
    assert runs.eligible.all() and runs.exclusion_reason.isna().all()
    assert set(runs.panel) == set(SWARM), f"panels {sorted(set(runs.panel))}"
    frozen_ids = set(tables.read_csv(paths.require(FROZEN / "heldout_runs.csv")).row_id)
    run_ids = observation_ids(runs)
    runs[SEEDS] = _seeds(runs, run_ids)

    out = pd.DataFrame({"observation_id": run_ids})
    out["swarm"] = runs.panel.map(SWARM)
    out["cohort"] = ["frozen" if r in frozen_ids else "expanded" for r in runs.row_id]
    for src, dst in META.items():
        out[dst] = runs[src]
    # The 160M panel and some archived Qwen3 runs record no proposal model; their registry source names the experiment.
    out["proposal_method"] = out.proposal_method.fillna(runs.source)
    # Public objective names (`table9` is `obe_easy`); the Llama panels record no objective.
    out["proposal_objective"] = out.proposal_objective.map(naming.objective_public, na_action="ignore")
    wcols = [c for c in runs.columns if c.startswith("weight::")]
    pcols = [c for c in runs.columns if c.startswith("pool_fraction::")]
    for c in wcols:
        out[f"w__{naming.bucket_public(c.removeprefix('weight::'))}"] = runs[c]
    for c in pcols:
        out[f"pf__{naming.bucket_public(c.removeprefix('pool_fraction::'))}"] = runs[c]
    bucket_order = ctx.buckets
    out["mixture_id"] = [
        keys.mixture_id(dict(zip(bucket_order, row, strict=True)))
        for row in out[[f"w__{b}" for b in bucket_order]].to_numpy()
    ]
    out["exposure_id"] = [
        keys.exposure_id(
            dict(zip(bucket_order, w, strict=True)), dict(zip(bucket_order, p, strict=True)), paths.PARTITION_REVISION
        )
        for w, p in zip(
            out[[f"w__{b}" for b in bucket_order]].to_numpy(),
            out[[f"pf__{b}" for b in bucket_order]].to_numpy(),
            strict=True,
        )
    ]
    unimax8 = (out.proposal_objective == "none") & (out.mixture_id == _unimax8_mixture())
    out.loc[unimax8, "epoch_cap"] = UNIMAX8_EPOCH_CAP
    out = out.join(_components(ctx, runs).reindex(runs.row_id).reset_index(drop=True))
    out = _with_pulled_components(out, runs, run_ids)
    out["components_available"] = out[[f"bpb__{t}" for t in ctx.tasks]].notna().all(axis=1)
    recorded = pd.DataFrame({"uncheatable": runs.uncheatable_bpb, "obe_easy": runs.table9_macro_bpb})
    fixed_rule = pd.DataFrame(
        {"uncheatable": (runs.panel == FIXED_RULE_PANEL) & ~runs.source.isin(POOLED_SOURCES), "obe_easy": True}
    )
    # Raises for a run that would need a recorded aggregate under other weights; null where neither exists.
    out = tables.add_aggregates(out, ctx.weights, recorded, fixed_rule)
    return {"heldout": tables.order_columns(out, ctx.buckets, ctx.tasks)}
