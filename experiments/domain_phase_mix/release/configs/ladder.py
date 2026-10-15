# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The scaling ladder and the per-task accuracies of its four 1e21 mixtures.

`ladder` has one row per training run of the six ladder mixtures at 3e18, 2e19, 3e20 and 1e21 FLOPs: MARINER and
Olmix, each proposed for Uncheatable and for OlmoBaseEval Easy, and the proportional and UniMax-8 baselines. These are
the points of the paper's scaling figure (fig:r1) and table (tab:scaling-main); the figure's `plotted_points.csv`
names their sources. Above 3e18 each mixture has one run. At 3e18 the figure averages three trainer seeds of each
mixture, which are validation runs (`validation_3e18`), and eleven proportional runs: the swarm's ten proportional
repeats (`swarm_qwen3_360m_1p6b`) and the June ladder's run. `export` holds the June run, which no other table holds;
`with_3e18_rows` adds copies of the other 3e18 runs' rows, which keep their observation ids.

MARINER and Olmix runs come from the audited results of their scaling launches, with each run's 51 OlmoBaseEval Easy
components from the Table 9 result saved beside them. The baselines come from the June ladder's W&B export, which
records their seven Uncheatable components but no OlmoBaseEval Easy component. Their 1e21 components are those of the
accuracy table, and the 3e18 proportional run's are those the UniMax cap sweep collected again beside its own runs;
each set's mean reproduces the run's recorded aggregate. The 2e19 and 3e20 components exist only on W&B, so those four
rows have Uncheatable components only (`components_available` false) and keep their recorded OlmoBaseEval Easy mean:
every ladder source scored both aggregates with the fixed rule, and `tables.add_aggregates` recomputes them wherever a
run has the components.

Observation ids key on the hashed training run name: the audited `run_name` (the run directory in its evaluation
metrics URI) and, for the baselines, the W&B run id, which is the same name. Data seeds are as the sources record them,
checked against a `_seed<n>` in the run name. A `_t<n>` suffix names a nonzero trainer seed; no ladder run has one,
and the scaling launches record trainer seed 0. Model sizes and token counts are the launch records' exact counts.

`ladder_accuracy` has the BPB and accuracy (percent; MT-MBPP as executable pass@1) on the 51 OlmoBaseEval Easy tasks
of the four 1e21 runs that the paper scores, keyed by observation id, with the accuracy example counts that the paper's
table tab:a-accuracy-components prints.

`ladder_compute_equivalence` holds the compute multiples of table tab:a-compute-equivalence, as
`compute_equivalence_20260915.py` computed them from the scaling figure's points: for each objective and baseline, the
law `law_e + law_a * (C / 1e18) ** -law_alpha` fitted to the baseline's mean loss at the four rungs (`law_rmse`, its
residual over them), and at each MARINER rung (`compute_flops`, MARINER's mean loss `mariner_bpb`) the compute at which
the law reaches that loss, its multiple of MARINER's compute, the saving (one minus the inverse multiple, in percent),
the loss improvement at matched compute (percent) and whether the match lies beyond the baseline's largest rung.

`ladder_unimax_cap_sweep` holds the UniMax cap sweep's runs at caps 1, 4 and 12 (table tab:a-unimax-cap), one run each
at the 3e18 rung's model size and token count: each run's cap, seeds, 58 components and both aggregates. The sweep's
Table 9 results give the 51 OlmoBaseEval Easy components, whose mean reproduces each run's recorded mean. The runs' W&B
summaries give the seven Uncheatable components (CAP_SWEEP_PULL): each W&B run id is a run name of the launch record
with its hash, and the data seed of its W&B config must be the launch record's. The sweep's cap-8 and proportional runs
are ladder rows. The table has no observation ids.
"""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.release import keys, naming, paths, run_names, tables, wandb_pulls
from experiments.domain_phase_mix.release.configs import partition_tasks
from experiments.domain_phase_mix.release.context import ExportContext

LADDER = "ladder"
ACCURACY_TABLE = "ladder_accuracy"
EQUIVALENCE_TABLE = "ladder_compute_equivalence"
# compute_equivalence.csv (one row per objective, baseline and MARINER rung) and fits.json (each baseline's law).
COMPUTE_EQUIVALENCE = paths.RO / "compute_equivalence_20260915"
EQUIVALENCE_BASELINES = {"Proportional": "proportional", "UniMax-8": "unimax8", "Olmix": "olmix"}
FROZEN = paths.RO / "frozen_scaling_update_20260913"
# Each holds audited_measured_results.csv and every run's Table 9 result, `<run name>_table9.json`.
AUDITED = (FROZEN / "mariner", FROZEN / "matched_olmix")
AUDIT_FILE = "audited_measured_results.csv"
TABLE9_RESULT = "{run}_table9.json"
PROGRESS = paths.RO / "delphi_scaling_progress_20260625/delphi_scaling_completed_wandb.csv"
CANDIDATE_WEIGHTS = (
    paths.RO / "delphi_frozen_procedure_validation_3e18_20260908/runtime_materialization/candidate_weights.csv",
    paths.RO / "delphi_matched_olmix_3e18_20260908/candidate_weights.csv",
    paths.REPO / "experiments/domain_phase_mix/release/unimax8_weights.csv",
)
ACCURACY = paths.RO / "accuracy_component_table_20260922"  # components.csv and the appendix table's rows.tex
# The UniMax cap sweep's Table 9 results, which hold the 51 components of the June proportional run at 3e18.
CAP_SWEEP_TABLE9 = paths.AG / "unimax_cap_sweep_20260922/table9_results.json"
CAP_SWEEP_PROPORTIONAL = "proportional_3e18"  # the sweep's key of that run
CAP_SWEEP_TABLE = "ladder_unimax_cap_sweep"
# The sweep's launch record: each run's mixture, seeds and size (run_specs.json) and each mixture's cap (summary.json).
CAP_SWEEP_LAUNCH = paths.RO / "delphi_baseline_mixtures_issue6607_20260623"
# The seven Uncheatable components of the sweep's runs, which their Table 9 results lack, and the data seeds of their W&B
# configs, frozen from W&B (`wandb_pulls`). Each pull's pooled aggregate uses other weights than the fixed rule, so the
# table computes bpb__uncheatable from the components.
CAP_SWEEP_PULL = (
    paths.REPO / "experiments/domain_phase_mix/release/configs/unimax_cap_sweep_wandb_components_20261015.csv"
)
# Every ladder source scored both aggregates with the fixed rule: the audits recorded the fixed weighting of each run's
# components, and the June ladder's legacy-schema aggregates agree with it to 8e-8.
FIXED_RULE = {"uncheatable": True, "obe_easy": True}
SWARM_FLOPS = 3e18
ACCURACY_FLOPS = 1e21
SEEDS = ["data_seed", "trainer_seed"]


def required_sources() -> list[Path]:
    """The files the export reads; each audited folder's per-run Table 9 results are the ones present."""
    per_run = [path for folder in AUDITED for path in sorted(folder.glob(TABLE9_RESULT.format(run="*")))]
    audits = [folder / AUDIT_FILE for folder in AUDITED]
    return [
        *CANDIDATE_WEIGHTS,
        *audits,
        *per_run,
        PROGRESS,
        ACCURACY / "components.csv",
        ACCURACY / "rows.tex",
        CAP_SWEEP_TABLE9,
        COMPUTE_EQUIVALENCE / "compute_equivalence.csv",
        COMPUTE_EQUIVALENCE / "fits.json",
        CAP_SWEEP_LAUNCH / "run_specs.json",
        CAP_SWEEP_LAUNCH / "summary.json",
        CAP_SWEEP_PULL,
    ]


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
# The mixtures whose 1e21 runs the accuracy table scores, in its column order; it names its columns by method.
ACCURACY_POLICIES = ("proportional", "unimax8", "olmixq_t9_kl0p005_cap04", "lwspu_t9_snc_cap08")
# The validation runs the scaling figure plots at 3e18 (plotted_points.csv `source_rows`), by launch: trainer seeds 0
# to 2 of MARINER's frozen proposals and of the matched Olmix proposals, and the UniMax-8 repeats. The Olmix proposals'
# other validation runs (the epoch-cap and KL sweep's) are not points of the figure.
VALIDATION_3E18 = {
    "frozen_procedure": ("lwspu_t9_snc_cap08", "lwspu_u_snc_cap06"),
    "fairness_repeats": ("lwspu_t9_snc_cap08", "lwspu_u_snc_cap06"),
    "matched_olmix": ("olmixq_t9_kl0p005_cap04", "olmixq_u_kl0p05_cap04"),
    "unimax8_repeats": ("unimax8",),
}
SWARM_REPEATS = "repeat"  # the swarm table's design kind of its proportional repeats
UNCHEATABLE_TASKS = [t for t in naming.TASKS if t.startswith("uncheatable/")]
OBE_TASKS = [t for t in naming.TASKS if t not in UNCHEATABLE_TASKS]
# A run name is `<name>[_seed<data seed>][_t<trainer seed>]-<six hex digits>`.
RUN_NAME = re.compile(r".+?(?:_seed(?P<data_seed>\d+))?(?:_t(?P<trainer_seed>\d+))?-[0-9a-f]{6}")
# A task row of appendix table tab:a-accuracy-components: `\quad <paper name> & <examples> & <BPB of the mixtures> ...`.
EXAMPLE_ROW = re.compile(r"\\quad (?P<name>.+?) & (?P<n>\d[\d,]*) & ")


def _policy_weights(ctx: ExportContext) -> pd.DataFrame:
    """Public bucket weights (`w__<bucket>`) of each ladder mixture, indexed by its POLICY name."""
    long = pd.concat([tables.read_csv(paths.require(path)) for path in CANDIDATE_WEIGHTS], ignore_index=True)
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
    record = json.loads(paths.require(folder / TABLE9_RESULT.format(run=run.run_name)).read_text())
    evaluated = record["provenance"]["source_run_name"], record["provenance"]["mixture"]
    if evaluated != (run.run_name.rsplit("-", 1)[0], run.policy):
        raise ValueError(f"{run.run_name}: its Table 9 result evaluated {evaluated}")
    components = record["table9_components"]
    if sorted(components) != sorted(OBE_TASKS):
        raise ValueError(f"{run.run_name}: its Table 9 result does not hold the 51 OlmoBaseEval Easy tasks")
    return {f"bpb__{t}": float(components[t]) for t in OBE_TASKS}


def _audited(folder: Path) -> pd.DataFrame:
    """One scaling launch's measured MARINER or Olmix runs with their 58 per-task BPB."""
    results = tables.read_csv(paths.require(folder / AUDIT_FILE))
    results = results[results.status == "measured"].reset_index(drop=True)
    if [run_names.training_run_name_from_uri(uri) for uri in results.eval_metrics_uri] != list(results.run_name):
        raise ValueError(f"{folder.name}: run names differ from the evaluation-metrics URIs")
    if not (results.target.map(naming.objective_public) == [POLICY[p][1] for p in results.policy]).all():
        raise ValueError(f"{folder.name}: a mixture's proposal objective differs from its runs' target")
    runs = pd.DataFrame(
        {
            "run_name": results.run_name,
            "policy": results.policy,
            "compute_flops": results.target_flops.astype(float),
            "data_seed": results.data_seed.astype(int),
            "recorded_uncheatable": results.uncheatable_bpb,
            "recorded_obe_easy": results.table9_macro_bpb,
        }
    )
    uncheatable = results[[f"uncheatable_{t.removeprefix('uncheatable/')}_bpb" for t in UNCHEATABLE_TASKS]]
    obe = pd.DataFrame([_table9_components(folder, run) for run in results.itertuples(index=False)])
    return pd.concat([runs, uncheatable.rename(columns=lambda c: f"bpb__{naming.task_public(c)}"), obe], axis=1)


def _cap_sweep_components(run_name: str) -> np.ndarray:
    """The 51 OlmoBaseEval Easy BPB of the June proportional run at 3e18 from the UniMax cap sweep's results, after
    checking that they scored that run's checkpoint."""
    record = json.loads(paths.require(CAP_SWEEP_TABLE9).read_text())[CAP_SWEEP_PROPORTIONAL]
    if run_name not in record["checkpoint"].split("/"):
        raise ValueError(f"{CAP_SWEEP_TABLE9.name}: {CAP_SWEEP_PROPORTIONAL} scored another run than {run_name}")
    components = record["components"]
    if sorted(components) != sorted(OBE_TASKS):
        raise ValueError(f"{CAP_SWEEP_TABLE9.name}: {CAP_SWEEP_PROPORTIONAL} lacks the 51 OlmoBaseEval Easy tasks")
    return np.array([components[t] for t in OBE_TASKS], dtype=float)


def _baselines(accuracy: pd.DataFrame) -> pd.DataFrame:
    """The June ladder's proportional and UniMax-8 runs with their Uncheatable components, and the OlmoBaseEval Easy
    components of the accuracy table (`accuracy`, components.csv) at 1e21 and of the UniMax cap sweep for the 3e18
    proportional run."""
    wandb = tables.read_csv(paths.require(PROGRESS))
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
            "recorded_uncheatable": wandb.eval_uncheatable_eval_bpb,
            "recorded_obe_easy": wandb.olmo_base_easy_table9_51_component_macro_bpb,
        }
    )
    uncheatable = wandb[[f"eval/uncheatable_eval/{t.removeprefix('uncheatable/')}/bpb" for t in UNCHEATABLE_TASKS]]
    obe = pd.DataFrame(np.nan, index=runs.index, columns=[f"bpb__{t}" for t in OBE_TASKS])
    by_task = accuracy.set_index("component").reindex(OBE_TASKS)
    for policy in BASELINES:
        (row,) = runs.index[(runs.policy == policy) & (runs.compute_flops == ACCURACY_FLOPS)]
        obe.loc[row] = by_task[f"bpb_{POLICY[policy][0]}"].to_numpy()
    (row,) = runs.index[(runs.policy == PROPORTIONAL) & (runs.compute_flops == SWARM_FLOPS)]
    obe.loc[row] = _cap_sweep_components(runs.run_name[row])
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
    """The ladder table of `runs` (source rows with `run_name`, `policy`, `compute_flops`, `data_seed`, per-task BPB and
    recorded aggregates)."""
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
    above = [(*POLICY[p], flops) for p in POLICY for flops in SCALES if flops > SWARM_FLOPS]
    expected = sorted([*above, (*POLICY[PROPORTIONAL], SWARM_FLOPS)])
    found = sorted(zip(table.method, table.proposal_objective, table.compute_flops, strict=True))
    if found != expected:
        raise ValueError(f"expected one run per ladder mixture and rung above 3e18 and the June 3e18 run, found {found}")
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
    table["components_available"] = table[[f"bpb__{t}" for t in ctx.tasks]].notna().all(axis=1)
    recorded = pd.DataFrame({objective: runs[f"recorded_{objective}"] for objective in ctx.weights})
    fixed_rule = pd.DataFrame(FIXED_RULE, index=runs.index)
    table = tables.add_aggregates(table, ctx.weights, recorded, fixed_rule)
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


def _compute_equivalence() -> pd.DataFrame:
    """Each baseline's fitted law and, at each MARINER rung, the compute at which it reaches MARINER's loss."""
    fits = json.loads(paths.require(COMPUTE_EQUIVALENCE / "fits.json").read_text())
    if any(fit["form"] != "power" for fit in fits):
        raise ValueError("every compute-equivalence fit must be the power law")
    laws = pd.DataFrame(
        {
            "objective": [naming.objective_public(fit["target"]) for fit in fits],
            "baseline": [EQUIVALENCE_BASELINES[fit["baseline"]] for fit in fits],
            "law_e": [fit["parameters"]["E"] for fit in fits],
            "law_a": [fit["parameters"]["A"] for fit in fits],
            "law_alpha": [fit["parameters"]["alpha"] for fit in fits],
            "law_rmse": [fit["rmse"] for fit in fits],
        }
    )
    matches = tables.read_csv(paths.require(COMPUTE_EQUIVALENCE / "compute_equivalence.csv"))
    matches["objective"] = matches.target.map(naming.objective_public)
    matches["baseline"] = matches.baseline.map(EQUIVALENCE_BASELINES)
    table = matches.merge(laws, on=["objective", "baseline"], how="left", validate="many_to_one")
    if table[["baseline", "law_e"]].isna().any().any():
        raise ValueError("a compute-equivalence row names a baseline without a fitted law")
    table = table.rename(columns={"mariner_compute": "compute_flops", "mariner_loss": "mariner_bpb"})
    columns = ["objective", "baseline", *laws.columns[2:], "compute_flops", "mariner_bpb", "baseline_compute_to_match"]
    return table[[*columns, "multiplier", "compute_saved_pct", "loss_improvement_pct", "extrapolated"]]


def _cap_sweep_uncheatable(specs: list[dict[str, Any]]) -> pd.DataFrame:
    """The pulled Uncheatable components (`bpb__<task>`) of the runs of `specs` (run_specs.json), in their order.

    Raises unless each run's name with a six-hex-digit hash names exactly one pulled run, every pulled run is one of
    them, and each pulled data seed is the run's.
    """
    pulls = wandb_pulls.read(CAP_SWEEP_PULL)
    names = {i: i.rsplit("-", 1)[0] for i in pulls.index if run_names.RUN_DIRECTORY.fullmatch(i)}
    matched = []
    for spec in specs:
        ids = [i for i, name in names.items() if name == spec["run_name"]]
        if len(ids) != 1:
            raise ValueError(f"{spec['run_name']}: expected one pulled run of that name, found {ids}")
        matched += ids
    if sorted(matched) != sorted(pulls.index):
        raise ValueError(f"the pulled runs {sorted(pulls.index)} are not the sweep's runs {matched}, one for one")
    pull = pulls.loc[matched]
    seeds = [spec["data_seed"] for spec in specs]
    if pull.data_seed.tolist() != seeds:
        raise ValueError(f"pulled data seeds {pull.data_seed.tolist()} differ from the launch record's {seeds}")
    return pull[[f"bpb__{t}" for t in UNCHEATABLE_TASKS]].reset_index(drop=True)


def _unimax_cap_sweep(ctx: ExportContext) -> pd.DataFrame:
    """The cap sweep's runs that no other table holds, with their 58 components and both aggregates."""
    results = json.loads(paths.require(CAP_SWEEP_TABLE9).read_text())
    specs = json.loads(paths.require(CAP_SWEEP_LAUNCH / "run_specs.json").read_text())
    caps = json.loads(paths.require(CAP_SWEEP_LAUNCH / "summary.json").read_text())["unimax_epoch_caps"]
    scale = SCALES[SWARM_FLOPS]
    rows = []
    for spec in specs:
        size = (spec["total_trainable_params"], spec["non_embedding_params"], spec["realized_train_tokens"])
        if size != (scale["total_parameters"], scale["non_embedding_parameters"], scale["train_tokens"]):
            raise ValueError(f"{spec['run_name']} is not at the 3e18 rung's model size and token count")
        components = results[spec["run_name"]]["components"]
        if sorted(components) != sorted(OBE_TASKS):
            raise ValueError(f"{spec['run_name']} needs each of the 51 OlmoBaseEval Easy components once")
        seeds = {"data_seed": spec["data_seed"], "trainer_seed": spec["trainer_seed"]}
        rows.append({"epoch_cap": caps[spec["mixture"]], **seeds, **{f"bpb__{t}": components[t] for t in OBE_TASKS}})
    table = pd.concat([pd.DataFrame(rows), _cap_sweep_uncheatable(specs)], axis=1)
    table["components_available"] = table[[f"bpb__{t}" for t in ctx.tasks]].notna().all(axis=1)
    table = tables.add_aggregates(table, ctx.weights)
    recorded = [results[spec["run_name"]]["macro"] for spec in specs]
    if not np.allclose(table.bpb__obe_easy, recorded, rtol=0, atol=1e-12):
        raise ValueError("a cap-sweep run's components do not reproduce its recorded OlmoBaseEval Easy mean")
    return tables.order_columns(table, ctx.buckets, ctx.tasks)


def export(ctx: ExportContext) -> dict[str, pd.DataFrame]:
    accuracy = tables.read_csv(paths.require(ACCURACY / "components.csv"))
    if sorted(accuracy.component) != sorted(OBE_TASKS):
        raise ValueError("components.csv must hold each of the 51 OlmoBaseEval Easy tasks once")
    runs = pd.concat([*(_audited(folder) for folder in AUDITED), _baselines(accuracy)], ignore_index=True)
    # Of the 3e18 runs, the June proportional run is in no other table; `with_3e18_rows` copies the others.
    june = (runs.policy == PROPORTIONAL) & (runs.compute_flops == SWARM_FLOPS)
    runs = runs[(runs.compute_flops > SWARM_FLOPS) | june].reset_index(drop=True)
    # The mean of a run's components is its recorded aggregate, which ties each component set to its run.
    obe = runs[[f"bpb__{t}" for t in OBE_TASKS]]
    complete = obe.notna().all(axis=1)
    if not np.allclose(obe[complete].mean(axis=1), runs.recorded_obe_easy[complete], rtol=0, atol=1e-12):
        raise ValueError("a run's OlmoBaseEval Easy components do not reproduce its recorded aggregate")
    ladder_table = _ladder(ctx, runs)
    return {
        LADDER: ladder_table,
        ACCURACY_TABLE: _accuracy(accuracy, ladder_table),
        EQUIVALENCE_TABLE: _compute_equivalence(),
        CAP_SWEEP_TABLE: _unimax_cap_sweep(ctx),
    }


def with_3e18_rows(
    ladder_table: pd.DataFrame, validation_table: pd.DataFrame, swarm_table: pd.DataFrame
) -> pd.DataFrame:
    """The ladder with the rest of its 3e18 rung: copies of the `validation_3e18` rows of VALIDATION_3E18 and of the
    proportional repeats of `swarm_qwen3_360m_1p6b`.

    A copy keeps its run's observation id, weights, seeds, BPB and `components_available`, and takes the rung's model
    size and token count (SCALES) and its mixture's method and objective (POLICY).
    """
    chosen = [
        validation_table.launch.eq(launch) & validation_table.variant.isin(variants)
        for launch, variants in VALIDATION_3E18.items()
    ]
    validated = validation_table[np.logical_or.reduce(chosen)].assign(policy=lambda t: t.variant)
    seeds = validated.groupby("policy").trainer_seed.agg(sorted)
    if sorted(seeds.index) != sorted(set(POLICY) - {PROPORTIONAL}) or any(s != [0, 1, 2] for s in seeds):
        raise ValueError(f"expected trainer seeds 0 to 2 of each validated ladder mixture, found {seeds.to_dict()}")
    repeats = swarm_table[swarm_table.design_kind == SWARM_REPEATS].assign(policy=PROPORTIONAL)
    if len(repeats) != 10:
        raise ValueError(f"expected the swarm's ten proportional repeats, found {len(repeats)}")

    rows = pd.concat([validated, repeats], ignore_index=True)
    rows["method"] = [POLICY[p][0] for p in rows.policy]
    rows["proposal_objective"] = [POLICY[p][1] for p in rows.policy]
    rows["compute_flops"] = SWARM_FLOPS
    for column, value in SCALES[SWARM_FLOPS].items():
        rows[column] = value
    ladder = pd.concat([ladder_table, rows[ladder_table.columns]], ignore_index=True)
    if not ladder.observation_id.is_unique:
        raise ValueError("a 3e18 run is already a ladder row")
    return ladder.sort_values("compute_flops", kind="stable").reset_index(drop=True)
