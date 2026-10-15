# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The cumulative held-out registry: frozen (September 2) and expanded (September 13) cohorts of all three swarms.

Observation ids key on each run's training run name with its six-hex-digit hash (as W&B records it) wherever one is
derivable. The swarm tables key on hash-less training names; no held-out run is in a swarm table. The 116
validation-launch runs take the name from their evaluation-metrics URI, as `validation_3e18` does. 679 frozen runs take
the registry's W&B training run id; for the 187 `delphi_archive::` rows that is the row_id without its prefix. The 134
Llama 200M runs record no run name and key on the registry's `row_id`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.release import keys, naming, paths, run_names, tables
from experiments.domain_phase_mix.release.context import ExportContext

EXPANDED = paths.RO / "single_phase_heldout_benchmark_expanded_20260913"
FROZEN = paths.RO / "single_phase_heldout_benchmark_20260902"
FROZEN_FILES = ("heldout_runs.csv", "heldout_components.csv")
EXPANDED_FILES = ("heldout_runs.csv", "heldout_coordinate_components.csv")
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
    "uncheatable_bpb": "bpb__uncheatable_source",
    "table9_macro_bpb": "bpb__obe_easy_source",
}
# Weights behind each panel's recorded Uncheatable aggregate; the frozen registry's builder checked them to 3e-6.
AGGREGATE_VERSION = {"delphi_3e18_39bucket": "fixed_20260927", "60m_39bucket": "legacy", "300m_39bucket": "legacy"}
# These launches ran after the evaluator moved to BPB schema 2 (about 12 September 2026). Their recorded Uncheatable
# aggregate is its pooled one, 0.008 to 0.011 BPB above the fixed weighting of the same runs' components
# (two_phase_many/uncheatable_objective.py); the expanded registry copied it unchecked.
POOLED_SOURCES = ("validation::path_midpoints", "validation::regmix_reference")
POOLED_VERSION = "pooled_schema2"
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
    return [*(FROZEN / name for name in FROZEN_FILES), *(EXPANDED / name for name in EXPANDED_FILES), *launches]


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


def _validation_run_names() -> pd.Series:
    """Training run name of each validation run, from its evaluation-metrics URI, indexed by the registry's row_id."""
    uris = [_launch_results(launch, folder).eval_metrics_uri for launch, folder in VALIDATION_LAUNCHES.items()]
    return pd.concat(uris).map(run_names.training_run_name_from_uri)


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


def observation_keys(runs: pd.DataFrame) -> pd.Series:
    """The string each held-out run's observation id hashes, aligned with `runs` (rows of heldout_runs.csv).

    The training run name wherever one is known (the module docstring gives the sources), else the registry's row_id.
    `fits` keys its held-out predictions with the same rule.
    """
    frozen_ids = set(tables.read_csv(paths.require(FROZEN / "heldout_runs.csv")).row_id)
    names = _validation_run_names()
    assert set(runs.row_id) - frozen_ids <= set(names.index), "every expanded run needs its training run name"
    recorded = runs.training_wandb_run_id
    assert recorded.dropna().str.fullmatch(run_names.RUN_DIRECTORY.pattern).all(), "W&B run ids must be run names"
    return runs.row_id.map(names).fillna(recorded).fillna(runs.row_id)


def export(ctx: ExportContext) -> dict[str, pd.DataFrame]:
    runs = tables.read_csv(paths.require(EXPANDED / "heldout_runs.csv"))
    assert runs.eligible.all() and runs.exclusion_reason.isna().all()
    assert set(runs.panel) == set(SWARM), f"panels {sorted(set(runs.panel))}"
    frozen_ids = set(tables.read_csv(paths.require(FROZEN / "heldout_runs.csv")).row_id)

    out = pd.DataFrame({"observation_id": observation_keys(runs).map(ctx.registry.id_for)})
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
    out["aggregate_version"] = runs.panel.map(AGGREGATE_VERSION).mask(runs.source.isin(POOLED_SOURCES), POOLED_VERSION)
    out = out.join(_components(ctx, runs).reindex(runs.row_id).reset_index(drop=True))
    out = tables.add_aggregates(out, ctx.weights)  # NaN where components are missing; the manifest records coverage
    return {"heldout": tables.order_columns(out, ctx.buckets, ctx.tasks)}
