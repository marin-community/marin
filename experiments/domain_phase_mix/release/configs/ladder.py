# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The scaling ladder above 3e18 FLOPs and the per-task accuracies of its four 1e21 mixtures.

`ladder` has one row per training run of the six ladder mixtures at 2e19, 3e20 and 1e21 FLOPs: MARINER and Olmix,
each proposed for Uncheatable and for OlmoBaseEval Easy, and the proportional and UniMax-8 baselines. These are the
rungs above the swarm's scale in the paper's scaling figure (fig:r1) and table (tab:scaling-main); the figure's
`plotted_points.csv` names its sources. The 3e18 rungs are validation runs, which `validation_3e18` holds.

MARINER and Olmix runs come from the audited results of their scaling launches, with each run's 51 OlmoBaseEval Easy
components from the Table 9 result saved beside them. The baselines come from the June ladder's W&B export, which
records their seven Uncheatable components but no OlmoBaseEval Easy component. Their 1e21 components are those of the
accuracy table, whose mean reproduces each run's recorded aggregate; their 2e19 and 3e20 components exist only on W&B,
so those four rows have Uncheatable components only (`components_available` false).

Observation ids key on the hashed training run name: the audited `run_name` (the run directory in its evaluation
metrics URI) and, for the baselines, the W&B run id, which is the same name. Data seeds are as the sources record them,
checked against a `_seed<n>` in the run name. A `_t<n>` suffix names a nonzero trainer seed; no ladder run has one,
and the scaling launches record trainer seed 0. Model sizes and token counts are the launch records' exact counts.

`ladder_accuracy` has the BPB and accuracy (percent; MT-MBPP as executable pass@1) on the 51 OlmoBaseEval Easy tasks
of the four 1e21 runs that the paper scores, keyed by observation id, with the accuracy example counts that the paper's
table tab:a-accuracy-components prints.
"""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.release import keys, naming, paths, run_names, tables
from experiments.domain_phase_mix.release.configs import partition_tasks
from experiments.domain_phase_mix.release.context import ExportContext

LADDER = "ladder"
ACCURACY_TABLE = "ladder_accuracy"
FROZEN = paths.RO / "frozen_scaling_update_20260913"
# Each holds audited_measured_results.csv and every run's Table 9 result, `<run name>_table9.json`.
AUDITED = (FROZEN / "mariner", FROZEN / "matched_olmix")
PROGRESS = paths.RO / "delphi_scaling_progress_20260625/delphi_scaling_completed_wandb.csv"
CANDIDATE_WEIGHTS = (
    paths.RO / "delphi_frozen_procedure_validation_3e18_20260908/runtime_materialization/candidate_weights.csv",
    paths.RO / "delphi_matched_olmix_3e18_20260908/candidate_weights.csv",
    paths.REPO / "experiments/domain_phase_mix/release/unimax8_weights.csv",
)
ACCURACY = paths.RO / "accuracy_component_table_20260922"  # components.csv and the appendix table's rows.tex
# The audits recorded the fixed weighting of each run's components; the June ladder's legacy-schema aggregates agree
# with it to 8e-8.
AGGREGATE_VERSION = "fixed_20260927"
SWARM_FLOPS = 3e18
ACCURACY_FLOPS = 1e21
SEEDS = ["data_seed", "trainer_seed"]


def _scale(total_parameters: int, non_embedding_parameters: int, train_tokens: int) -> dict[str, float]:
    return {
        "total_parameters": total_parameters,
        "non_embedding_parameters": non_embedding_parameters,
        "train_tokens": train_tokens,
        "tpp": train_tokens / non_embedding_parameters,
    }


# Exact counts from the launch records (the 3e18 launches' run_specs.json, each scaling launch's run_manifest.json),
# which Appendix table tab:a-scaling-configurations prints rounded: 358M, 128M, 1.58B and TPP 12.3 at 3e18. Tokens
# are the trained steps times batches of 128, 128, 256 and 512 sequences of 4,096 tokens.
SCALES = {
    3e18: _scale(358_304_128, 128_469_376, 1_576_534_016),
    2e19: _scale(669_157_120, 340_821_760, 5_191_499_776),
    3e20: _scale(1_934_710_784, 1_409_374_208, 24_675_090_432),
    1e21: _scale(3_383_104_000, 2_726_433_280, 46_256_881_664),
}
# Each ladder mixture's name in its sources (the candidate id of the weight tables and of validation_3e18.variant; the
# June ladder's mixture) -> (method, proposal_objective).
POLICY = {
    "lwspu_t9_snc_cap08": ("mariner", "obe_easy"),
    "lwspu_u_snc_cap06": ("mariner", "uncheatable"),
    "olmixq_t9_kl0p005_cap04": ("olmix", "obe_easy"),
    "olmixq_u_kl0p05_cap04": ("olmix", "uncheatable"),
    "proportional": ("proportional", "none"),
    "unimax8": ("unimax8", "none"),
}
PROPORTIONAL = "proportional"
BASELINES = ("proportional", "unimax8")
OBJECTIVE = {"table9": "obe_easy", "uncheatable": "uncheatable"}  # the audits' `target`, in the public vocabulary
# The mixtures whose 1e21 runs the accuracy table scores, in its column order; it names its columns by method.
ACCURACY_POLICIES = ("proportional", "unimax8", "olmixq_t9_kl0p005_cap04", "lwspu_t9_snc_cap08")
UNCHEATABLE_TASKS = [t for t in naming.TASKS if t.startswith("uncheatable/")]
OBE_TASKS = [t for t in naming.TASKS if t not in UNCHEATABLE_TASKS]
# A run name is `<name>[_seed<data seed>][_t<trainer seed>]-<six hex digits>`.
RUN_NAME = re.compile(r".+?(?:_seed(?P<data_seed>\d+))?(?:_t(?P<trainer_seed>\d+))?-[0-9a-f]{6}")
# A task row of appendix table tab:a-accuracy-components: `\quad <paper name> & <examples> & <BPB of the mixtures> ...`.
EXAMPLE_ROW = re.compile(r"\\quad (?P<name>.+?) & (?P<n>\d[\d,]*) & ")


def _policy_weights(ctx: ExportContext) -> pd.DataFrame:
    """Public bucket weights (`w__<bucket>`) of each ladder mixture, indexed by its POLICY name."""
    long = pd.concat([pd.read_csv(paths.require(path)) for path in CANDIDATE_WEIGHTS], ignore_index=True)
    long = long[long.candidate_id.isin(POLICY)]
    wide = long.pivot(index="candidate_id", columns="domain", values="weight")  # raises on a repeated pair
    wide.columns = [f"w__{naming.bucket_public(c)}" for c in wide.columns]
    # The proportional mixture weights each bucket by its share of the pooled tokens.
    partition = partition_tasks.export(ctx)["partition"]
    shares = partition.pool_tokens / partition.pool_tokens.sum()
    wide.loc[PROPORTIONAL] = pd.Series(shares.to_numpy(), index=[f"w__{b}" for b in partition.bucket_id])
    wide = wide.reindex(columns=[f"w__{b}" for b in ctx.buckets])
    assert sorted(wide.index) == sorted(POLICY) and wide.notna().all().all(), "a ladder mixture lacks bucket weights"
    return wide


def _table9_components(folder: Path, run: Any) -> dict[str, float]:
    """A run's 51 OlmoBaseEval Easy BPB from the Table 9 result saved beside its audited row."""
    record = json.loads(paths.require(folder / f"{run.run_name}_table9.json").read_text())
    evaluated = record["provenance"]["source_run_name"], record["provenance"]["mixture"]
    if evaluated != (run.run_name.rsplit("-", 1)[0], run.policy):
        raise ValueError(f"{run.run_name}: its Table 9 result evaluated {evaluated}")
    components = record["table9_components"]
    if sorted(components) != sorted(OBE_TASKS):
        raise ValueError(f"{run.run_name}: its Table 9 result does not hold the 51 OlmoBaseEval Easy tasks")
    return {f"bpb__{t}": float(components[t]) for t in OBE_TASKS}


def _audited(folder: Path) -> pd.DataFrame:
    """One scaling launch's measured MARINER or Olmix runs with their 58 per-task BPB."""
    results = pd.read_csv(paths.require(folder / "audited_measured_results.csv"))
    results = results[results.status == "measured"].reset_index(drop=True)
    if [run_names.training_run_name_from_uri(uri) for uri in results.eval_metrics_uri] != list(results.run_name):
        raise ValueError(f"{folder.name}: run names differ from the evaluation-metrics URIs")
    if not (results.target.map(OBJECTIVE) == [POLICY[p][1] for p in results.policy]).all():
        raise ValueError(f"{folder.name}: a mixture's proposal objective differs from its runs' target")
    runs = pd.DataFrame(
        {
            "run_name": results.run_name,
            "policy": results.policy,
            "compute_flops": results.target_flops.astype(float),
            "data_seed": results.data_seed.astype(int),
            "bpb__uncheatable_source": results.uncheatable_bpb,
            "bpb__obe_easy_source": results.table9_macro_bpb,
        }
    )
    uncheatable = results[[f"uncheatable_{t.removeprefix('uncheatable/')}_bpb" for t in UNCHEATABLE_TASKS]]
    obe = pd.DataFrame([_table9_components(folder, run) for run in results.itertuples(index=False)])
    return pd.concat([runs, uncheatable.rename(columns=lambda c: f"bpb__{naming.task_public(c)}"), obe], axis=1)


def _baselines(accuracy: pd.DataFrame) -> pd.DataFrame:
    """The June ladder's proportional and UniMax-8 runs with their Uncheatable components, and at 1e21 the
    OlmoBaseEval Easy components of the accuracy table (`accuracy`, components.csv)."""
    wandb = pd.read_csv(paths.require(PROGRESS))
    wandb = wandb[wandb.mixture.isin(BASELINES)].reset_index(drop=True)
    assert wandb.is_completed.all() and set(wandb.mixture) == set(BASELINES)
    # A training run's W&B id, the last component of its W&B URL, is its hashed run name.
    if not (wandb.wandb_name == wandb.wandb_url.str.rsplit("/", n=1).str[-1]).all():
        raise ValueError("a baseline's W&B name differs from its run id")
    runs = pd.DataFrame(
        {
            "run_name": wandb.wandb_name,
            "policy": wandb.mixture,
            "compute_flops": wandb.flops.astype(float),
            "data_seed": wandb.data_seed.astype(int),
            "bpb__uncheatable_source": wandb.eval_uncheatable_eval_bpb,
            "bpb__obe_easy_source": wandb.olmo_base_easy_table9_51_component_macro_bpb,
        }
    )
    uncheatable = wandb[[f"eval/uncheatable_eval/{t.removeprefix('uncheatable/')}/bpb" for t in UNCHEATABLE_TASKS]]
    obe = pd.DataFrame(np.nan, index=runs.index, columns=[f"bpb__{t}" for t in OBE_TASKS])
    by_task = accuracy.set_index("component").reindex(OBE_TASKS)
    for policy in BASELINES:
        (row,) = runs.index[(runs.policy == policy) & (runs.compute_flops == ACCURACY_FLOPS)]
        obe.loc[row] = by_task[f"bpb_{POLICY[policy][0]}"].to_numpy()
    return pd.concat([runs, uncheatable.rename(columns=lambda c: f"bpb__{naming.task_public(c)}"), obe], axis=1)


def _seeds(run_name: str, data_seed: int) -> tuple[int, int]:
    """The recorded data seed, checked against a `_seed<n>` in the run name, and the trainer seed of a `_t<n>`
    suffix (0 without one)."""
    match = RUN_NAME.fullmatch(run_name)
    if match is None:
        raise ValueError(f"unexpected run name {run_name!r}")
    if match["data_seed"] is not None and int(match["data_seed"]) != data_seed:
        raise ValueError(f"{run_name}: recorded data seed {data_seed} differs from its name")
    return data_seed, int(match["trainer_seed"] or 0)


def _ladder(ctx: ExportContext, runs: pd.DataFrame) -> pd.DataFrame:
    """The ladder table of `runs` (source rows with `run_name`, `policy`, `compute_flops`, `data_seed` and BPB)."""
    ids = pd.DataFrame(
        {
            "observation_id": runs.run_name.map(ctx.registry.id_for),
            "method": [POLICY[p][0] for p in runs.policy],
            "proposal_objective": [POLICY[p][1] for p in runs.policy],
            "compute_flops": runs.compute_flops,
        }
    )
    sizes = pd.DataFrame([SCALES[flops] for flops in runs.compute_flops])
    seeds = [_seeds(name, seed) for name, seed in zip(runs.run_name, runs.data_seed, strict=True)]
    table = pd.concat([ids, sizes, pd.DataFrame(seeds, columns=SEEDS)], axis=1)
    expected = sorted((*POLICY[p], flops) for p in POLICY for flops in SCALES if flops > SWARM_FLOPS)
    found = sorted(zip(table.method, table.proposal_objective, table.compute_flops, strict=True))
    if found != expected:
        raise ValueError(f"expected one run per ladder mixture and rung above 3e18, found {found}")
    assert table.observation_id.is_unique

    weights = runs[["policy"]].merge(
        _policy_weights(ctx), left_on="policy", right_index=True, how="left", validate="many_to_one"
    )
    wcols = [f"w__{b}" for b in ctx.buckets]
    pf = pd.DataFrame(1.0, index=runs.index, columns=[f"pf__{b}" for b in ctx.buckets])  # full pools at every rung
    table = pd.concat([table, weights[wcols], pf, runs[[c for c in runs if c.startswith("bpb__")]]], axis=1)
    pool_fractions = dict.fromkeys(ctx.buckets, 1.0)
    mixtures = [dict(zip(ctx.buckets, row, strict=True)) for row in table[wcols].to_numpy()]
    table["mixture_id"] = [keys.mixture_id(w) for w in mixtures]
    table["exposure_id"] = [keys.exposure_id(w, pool_fractions, paths.PARTITION_REVISION) for w in mixtures]
    table["aggregate_version"] = AGGREGATE_VERSION
    table["components_available"] = table[[f"bpb__{t}" for t in ctx.tasks]].notna().all(axis=1)
    table = tables.add_aggregates(table, ctx.weights)  # NaN OlmoBaseEval Easy where components are missing
    return tables.order_columns(table, ctx.buckets, ctx.tasks)


def _example_counts(rows_tex: Path) -> dict[str, int]:
    """Accuracy examples per task from the rows of appendix table tab:a-accuracy-components, which name each task by
    its paper name (task_metadata.csv)."""
    names = naming.task_table()
    task_of = {name: task for task, name in zip(names.task_id, names.paper_name, strict=True) if task in OBE_TASKS}
    assert len(task_of) == len(OBE_TASKS), "every OlmoBaseEval Easy task needs its own paper name"
    counts: dict[str, int] = {}
    for line in paths.require(rows_tex).read_text().splitlines():
        match = EXAMPLE_ROW.match(line)
        if match is None:
            continue
        task = task_of[match["name"]]
        if task in counts:
            raise ValueError(f"{rows_tex.name}: {task} appears twice")
        counts[task] = int(match["n"].replace(",", ""))
    if sorted(counts) != sorted(OBE_TASKS):
        raise ValueError(f"{rows_tex.name} gives example counts for {len(counts)} of the 51 tasks")
    return counts


def _accuracy(accuracy: pd.DataFrame, ladder_table: pd.DataFrame) -> pd.DataFrame:
    """Per-task BPB and accuracy of each scored 1e21 run, from components.csv (`accuracy`)."""
    counts = _example_counts(ACCURACY / "rows.tex")
    task_ids = [naming.task_public(f"table9/{component}/bpb") for component in accuracy.component]
    parts = []
    for policy in ACCURACY_POLICIES:
        method, objective = POLICY[policy]
        (observation_id,) = ladder_table.observation_id[
            (ladder_table.method == method)
            & (ladder_table.proposal_objective == objective)
            & (ladder_table.compute_flops == ACCURACY_FLOPS)
        ]
        part = pd.DataFrame(
            {
                "method": method,
                "task_id": task_ids,
                "group": accuracy.group,
                "bpb": accuracy[f"bpb_{method}"],
                "accuracy": accuracy[f"accuracy_{method}"],
                "n_examples": [counts[t] for t in task_ids],
            }
        )
        part.insert(0, "observation_id", observation_id)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def export(ctx: ExportContext) -> dict[str, pd.DataFrame]:
    accuracy = pd.read_csv(paths.require(ACCURACY / "components.csv"))
    if sorted(accuracy.component) != sorted(OBE_TASKS):
        raise ValueError("components.csv must hold each of the 51 OlmoBaseEval Easy tasks once")
    runs = pd.concat([*(_audited(folder) for folder in AUDITED), _baselines(accuracy)], ignore_index=True)
    runs = runs[runs.compute_flops > SWARM_FLOPS].reset_index(drop=True)
    # The mean of a run's components is its recorded aggregate, which ties each component set to its run.
    obe = runs[[f"bpb__{t}" for t in OBE_TASKS]]
    complete = obe.notna().all(axis=1)
    if not np.allclose(obe[complete].mean(axis=1), runs.bpb__obe_easy_source[complete], rtol=0, atol=1e-12):
        raise ValueError("a run's OlmoBaseEval Easy components do not reproduce its recorded aggregate")
    ladder_table = _ladder(ctx, runs)
    return {LADDER: ladder_table, ACCURACY_TABLE: _accuracy(accuracy, ladder_table)}
