# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Every trained proposal at 3e18, joined launch by launch through validation_manifest.csv.

Each manifest row names one launch's measured results, its OlmoBaseEval Easy components and its candidate weight
tables, as paths relative to the repository root; `;` separates several weight tables, and an empty components path
marks a launch that kept aggregates only. The 24 launches of collect_delphi_3e18_validation_results_20260906.py name
each run by candidate id. A grouped launch repeats candidates over (data seed, trainer seed) groups, and its components
file keys each run `<candidate>@t<trainer seed>`. The UniMax-8 repeats train one policy at three trainer seeds and leave
it unnamed in their results; unimax8_weights.csv holds the UniMax-8 construction of launch_delphi_baseline_mixtures.py.

Observation ids key on the training run name, as in `heldout`: the hashed run directory in each run's
evaluation-metrics URI, or the `run` column where a launch records run names directly. The manifest's `method` is one
method, or `pattern:method` entries (glob patterns over candidate ids) for a launch that mixes proposing methods.
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
UNCHEATABLE_TASKS = [t for t in naming.TASKS if t.startswith("uncheatable/")]
OBE_TASKS = [t for t in naming.TASKS if t not in UNCHEATABLE_TASKS]
RESULT_META = {
    "candidate_id": "variant",
    "target": "proposal_objective",
    "epoch_cap": "epoch_cap",
    "data_seed": "data_seed",
    "trainer_seed": "trainer_seed",
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
    return pd.concat([pd.read_csv(paths.require(paths.REPO / path)) for path in cell.split(";")], ignore_index=True)


def _weights(candidates: pd.DataFrame, ctx: ExportContext) -> pd.DataFrame:
    """Public weights per candidate from a long weight table (`candidate_id, domain, weight` plus launch columns)."""
    wide = candidates.pivot(index="candidate_id", columns="domain", values="weight")  # raises on a repeated pair
    wide.columns = [f"w__{naming.bucket_public(c)}" for c in wide.columns]
    return wide[[f"w__{b}" for b in ctx.buckets]]


def _components(path: Path) -> pd.DataFrame:
    """Per-run OlmoBaseEval Easy BPB from a long components file named by public task id, indexed by run key."""
    long = pd.read_csv(path)
    assert set(long.component) == set(OBE_TASKS), f"{path}: unexpected OlmoBaseEval Easy components"
    wide = long.pivot(index="candidate_id", columns="component", values="bpb")  # raises on a repeated pair
    return wide[OBE_TASKS].add_prefix("bpb__")


def _training_run_names(results: pd.DataFrame) -> list[str]:
    """Hashed training run names: from each evaluation-metrics URI, or the `run` column where a launch keeps them."""
    if "eval_metrics_uri" in results:
        return [run_names.training_run_name_from_uri(uri) for uri in results.eval_metrics_uri]
    names = list(results.run)
    assert all(run_names.RUN_DIRECTORY.fullmatch(name) for name in names), f"run names without a hash: {names}"
    return names


def _methods(spec: str, candidates: pd.Series) -> list[str]:
    """Each run's method: the manifest's method, or that of the one `pattern:method` entry its candidate matches."""
    if ":" not in spec:
        return [spec] * len(candidates)
    rules = [entry.split(":") for entry in spec.split(";")]
    methods = []
    for candidate in candidates:
        matched = [method for pattern, method in rules if fnmatch.fnmatchcase(candidate, pattern)]
        if len(matched) != 1:
            raise ValueError(f"candidate {candidate} matches {len(matched)} method patterns of {spec!r}")
        methods.append(matched[0])
    return methods


def _launch(ctx: ExportContext, launch: Any) -> pd.DataFrame:
    """One launch's measured runs with metadata, weights and per-task BPB; raises ValueError naming the launch."""
    results = pd.read_csv(paths.require(paths.REPO / launch.results_csv))
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

    df = results[[c for c in RESULT_META if c in results]].rename(columns=RESULT_META)
    df.insert(0, "observation_id", [ctx.registry.id_for(name) for name in _training_run_names(results)])
    df.insert(1, "launch", launch.launch)
    df.insert(2, "method", _methods(launch.method, candidates))
    run_weights = candidates.to_frame("candidate").merge(
        weights, left_on="candidate", right_index=True, how="left", validate="many_to_one"
    )
    assert run_weights.notna().all().all(), f"launch {launch.launch}: a candidate lacks a bucket weight"
    df = pd.concat([df, run_weights.drop(columns="candidate").reset_index(drop=True)], axis=1)
    if not pd.isna(launch.components_csv):
        runs = list(candidates)
        if launch.grouped:
            runs = [f"{c}@t{int(seed)}" for c, seed in zip(candidates, results.trainer_seed, strict=True)]
        components = pd.DataFrame({"run": runs}).merge(
            _components(paths.require(paths.REPO / launch.components_csv)),
            left_on="run",
            right_index=True,
            how="left",
            validate="one_to_one",
        )
        unmatched = components.run[components.drop(columns="run").isna().any(axis=1)].tolist()
        if unmatched:
            raise ValueError(f"launch {launch.launch}: {len(unmatched)} runs have no components: {unmatched[:5]}")
        df = pd.concat([df, components.drop(columns="run").reset_index(drop=True)], axis=1)
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
    out = tables.add_aggregates(out, ctx.weights)  # NaN for a launch that kept aggregates only
    return tables.order_columns(out, ctx.buckets, ctx.tasks)


def export(ctx: ExportContext) -> dict[str, pd.DataFrame]:
    return {TABLE: export_launches(ctx, pd.read_csv(paths.require(MANIFEST)))}
