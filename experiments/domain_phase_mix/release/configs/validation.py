# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Every trained proposal at 3e18, joined launch by launch through validation_manifest.csv.

Each manifest row names one launch's measured results, its per-task components and its candidate weight tables, as
paths relative to the repository root; `;` separates several weight tables. The 24 launches of
collect_delphi_3e18_validation_results_20260906.py name each run by candidate id and keep the OlmoBaseEval Easy
components in their own folder; a grouped launch repeats candidates over (data seed, trainer seed) groups. The UniMax-8
repeats train one policy at three trainer seeds and leave it unnamed in their results; unimax8_weights.csv holds the
UniMax-8 construction of launch_delphi_baseline_mixtures.py, and only the seed-0 run's components exist locally, in
the frozen held-out registry. `components_key` formats each run's key in its components file from `{candidate}`,
`{trainer_seed}` and `{run_name}`; `component_rows` counts the launch's runs with all 58 per-task values.

Observation ids key on the training run name, as in `heldout`: the hashed run directory in each run's
evaluation-metrics URI, or the `run` column where a launch records run names directly. Seeds come from the directory
that holds each run, checked against any seeds the results record. `method` and `proposal_objective` are one value,
or `pattern:value` entries (glob patterns over candidate ids) for a launch whose candidates differ; an empty
`proposal_objective` keeps the results' `target`, and the manifest names the objective of each subset proposal.
Objectives take their public names (`naming.objective_public`): `obe_easy` for the results' `table9`, and `none` for
the UniMax-8 baseline, which optimizes no objective. `epoch_cap` gives the cap of a launch whose results record none
(UniMax-8's 8); an empty cell keeps the results' `epoch_cap`.
"""

import fnmatch
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.domain_phase_mix.release import keys, naming, paths, run_names, tables
from experiments.domain_phase_mix.release.context import ExportContext

TABLE = "validation_3e18"
MANIFEST = paths.REPO / "experiments/domain_phase_mix/release/validation_manifest.csv"
AGGREGATE_VERSION = "fixed_20260927"
# These launches recorded the BPB-schema-2 pooled Uncheatable aggregate, 0.008 to 0.011 BPB above the fixed weighting of
# the same runs' components; `heldout` labels the same runs the same way.
POOLED_LAUNCHES = ("path_midpoints", "regmix_reference")
POOLED_VERSION = "pooled_schema2"
SEEDS = ["data_seed", "trainer_seed"]
UNCHEATABLE_TASKS = [t for t in naming.TASKS if t.startswith("uncheatable/")]
RESULT_META = {
    "candidate_id": "variant",
    "target": "proposal_objective",
    "epoch_cap": "epoch_cap",
    "uncheatable_bpb": "bpb__uncheatable_source",
    "table9_macro_bpb": "bpb__obe_easy_source",
    "predicted_uncheatable_bpb": "predicted_bpb_uncheatable",
    "predicted_table9_macro_bpb": "predicted_bpb_obe_easy",
}
RESULT_META.update(
    {
        column: f"bpb__{naming.task_public(column)}"
        for column in (f"uncheatable_{t.removeprefix('uncheatable/')}_bpb" for t in UNCHEATABLE_TASKS)
    }
)


def _csv(cell: str) -> pd.DataFrame:
    """The tables named by one manifest cell, stacked."""
    return pd.concat([tables.read_csv(paths.require(paths.REPO / path)) for path in cell.split(";")], ignore_index=True)


def _weights(candidates: pd.DataFrame, ctx: ExportContext) -> pd.DataFrame:
    """Public weights per candidate from a long weight table (`candidate_id, domain, weight` plus launch columns)."""
    wide = candidates.pivot(index="candidate_id", columns="domain", values="weight")  # raises on a repeated pair
    wide.columns = [f"w__{naming.bucket_public(c)}" for c in wide.columns]
    return wide[[f"w__{b}" for b in ctx.buckets]]


def _components(path: Path) -> pd.DataFrame:
    """Per-run BPB by public task id from a long components file, indexed by each run's key in that file.

    A launch's measured_table9_components.csv keys runs by `candidate_id` and names the 51 OlmoBaseEval Easy tasks by
    public id; the frozen held-out registry's heldout_components.csv keys runs by `row_id` and names all 58 tasks by
    internal metric.
    """
    long = tables.read_csv(path)
    public = {c: c if c in naming.TASKS else naming.task_public(c) for c in long.component.unique()}
    long["task"] = long.component.map(public)
    wide = long.pivot(index="row_id" if "row_id" in long else "candidate_id", columns="task", values="bpb")
    return wide.add_prefix("bpb__")  # pivot raises on a repeated (run, task) pair


def _training_run_names(results: pd.DataFrame) -> list[str]:
    """Hashed training run names: from each evaluation-metrics URI, or the `run` column where a launch keeps them."""
    if "eval_metrics_uri" in results:
        return [run_names.training_run_name_from_uri(uri) for uri in results.eval_metrics_uri]
    names = list(results.run)
    assert all(run_names.RUN_DIRECTORY.fullmatch(name) for name in names), f"run names without a hash: {names}"
    return names


def _seeds(results: pd.DataFrame, launch: str) -> pd.DataFrame:
    """Each run's data and trainer seed from the directory that holds it, or as recorded where no URI is kept."""
    if "eval_metrics_uri" not in results:
        return results[SEEDS].astype(int)
    seeds = pd.DataFrame([run_names.seeds_from_uri(uri) for uri in results.eval_metrics_uri], columns=SEEDS)
    if set(SEEDS) <= set(results) and not (results[SEEDS].to_numpy() == seeds.to_numpy()).all():
        raise ValueError(f"launch {launch}: recorded seeds differ from the runs' seed directories")
    return seeds


def _per_candidate(spec: str, candidates: pd.Series) -> list[str]:
    """A manifest value per run: the cell's value, or that of the one `pattern:value` entry its candidate matches."""
    if ":" not in spec:
        return [spec] * len(candidates)
    rules = [entry.split(":") for entry in spec.split(";")]
    values = []
    for candidate in candidates:
        matched = [value for pattern, value in rules if fnmatch.fnmatchcase(candidate, pattern)]
        if len(matched) != 1:
            raise ValueError(f"candidate {candidate} matches {len(matched)} patterns of {spec!r}")
        values.append(matched[0])
    return values


def _launch(ctx: ExportContext, launch: Any) -> pd.DataFrame:
    """One launch's measured runs with metadata, weights and per-task BPB; raises ValueError naming the launch."""
    results = tables.read_csv(paths.require(paths.REPO / launch.results_csv))
    if "status" in results:  # a results file without status lists measured runs only
        results = results[results.status == "measured"].reset_index(drop=True)
    weights = _weights(_csv(launch.candidates_csv), ctx)
    if launch.candidate_key not in results:
        # A launch that trains one policy (the UniMax-8 repeats) leaves it unnamed in its results.
        if len(weights) != 1:
            raise ValueError(f"launch {launch.launch}: results name no candidate, the weights hold {len(weights)}")
        results[launch.candidate_key] = weights.index[0]
    candidates = results[launch.candidate_key]
    missing = sorted(set(candidates) - set(weights.index))
    if missing:
        raise ValueError(f"launch {launch.launch}: {len(missing)} measured candidates have no weights: {missing[:5]}")

    names = _training_run_names(results)
    seeds = _seeds(results, launch.launch)
    df = results[[c for c in RESULT_META if c in results]].rename(columns=RESULT_META)
    df.insert(0, "observation_id", [ctx.registry.id_for(name) for name in names])
    df.insert(1, "launch", launch.launch)
    df.insert(2, "method", _per_candidate(launch.method, candidates))
    if not pd.isna(launch.proposal_objective):  # a subset proposal's objective, or UniMax-8's none
        df["proposal_objective"] = _per_candidate(launch.proposal_objective, candidates)
    if "proposal_objective" not in df or df.proposal_objective.isna().any():
        raise ValueError(f"launch {launch.launch}: runs without a proposal objective")
    df["proposal_objective"] = df.proposal_objective.map(naming.objective_public)
    if not pd.isna(launch.epoch_cap):
        df["epoch_cap"] = float(launch.epoch_cap)
    df[SEEDS] = seeds
    run_weights = candidates.to_frame("candidate").merge(
        weights, left_on="candidate", right_index=True, how="left", validate="many_to_one"
    )
    assert run_weights.notna().all().all(), f"launch {launch.launch}: a candidate lacks a bucket weight"
    df = pd.concat([df, run_weights.drop(columns="candidate").reset_index(drop=True)], axis=1)

    run_keys = [
        launch.components_key.format(candidate=candidate, trainer_seed=trainer_seed, run_name=name)
        for candidate, trainer_seed, name in zip(candidates, seeds.trainer_seed, names, strict=True)
    ]
    components = _components(paths.require(paths.REPO / launch.components_csv))
    assert not set(components.columns) & set(df.columns), f"launch {launch.launch}: a task appears in two sources"
    joined = pd.DataFrame({"run": run_keys}).merge(
        components, left_on="run", right_index=True, how="left", validate="one_to_one"
    )
    df = pd.concat([df, joined.drop(columns="run").reset_index(drop=True)], axis=1)
    values = df.reindex(columns=[f"bpb__{t}" for t in ctx.tasks]).notna()
    complete = values.all(axis=1)
    if (values.any(axis=1) & ~complete).any() or complete.sum() != launch.component_rows:
        raise ValueError(
            f"launch {launch.launch}: expected {launch.component_rows} runs with all per-task values, found "
            f"{complete.sum()} and {(values.any(axis=1) & ~complete).sum()} with a partial set"
        )
    if len(df) != launch.expected_rows:
        raise ValueError(f"launch {launch.launch}: expected {launch.expected_rows} rows, exported {len(df)}")
    return df


def export_launches(ctx: ExportContext, manifest: pd.DataFrame) -> pd.DataFrame:
    """The validation table of the launches in `manifest`, which has the columns of validation_manifest.csv."""
    out = pd.concat([_launch(ctx, launch) for launch in manifest.itertuples(index=False)], ignore_index=True)
    assert out.observation_id.is_unique, "a training run appears in more than one launch"
    wcols = [f"w__{b}" for b in ctx.buckets]
    for b in ctx.buckets:
        out[f"pf__{b}"] = 1.0  # every validation run trains on full pools
    pf = dict.fromkeys(ctx.buckets, 1.0)
    out["mixture_id"] = [keys.mixture_id(dict(zip(ctx.buckets, row, strict=True))) for row in out[wcols].to_numpy()]
    out["exposure_id"] = [
        keys.exposure_id(dict(zip(ctx.buckets, row, strict=True)), pf, paths.PARTITION_REVISION)
        for row in out[wcols].to_numpy()
    ]
    out["aggregate_version"] = AGGREGATE_VERSION
    out.loc[out.launch.isin(POOLED_LAUNCHES), "aggregate_version"] = POOLED_VERSION
    out = tables.add_aggregates(out, ctx.weights)  # NaN for runs without per-task values
    return tables.order_columns(out, ctx.buckets, ctx.tasks)


def export(ctx: ExportContext) -> dict[str, pd.DataFrame]:
    return {TABLE: export_launches(ctx, tables.read_csv(paths.require(MANIFEST)))}
