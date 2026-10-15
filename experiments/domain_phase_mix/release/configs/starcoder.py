# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The two-bucket Nemotron-CC/StarCoder experiments: the tied-curve atlas, the fixed-TPP diagonal, the curricula, the
mixture-switch study and the per-step training loss around the switch.

The atlas (`starcoder_all_tied_curves_canonical_dsp_20260902`) fits 45 single-phase curves of programming-languages
BPB against the StarCoder weight, in four families: the fixed-model token ladder, the matched model-size and
token-budget ladder, the training horizon by StarCoder subset ("replay burden"), and the coupled learning-rate onset.

- `starcoder_observations`: one row per training run (954): model size (`total_parameters`,
  `non_embedding_parameters`), tokens (`train_tokens`, D), StarCoder subset (`starcoder_subset_tokens`), seeds,
  StarCoder weight, `starcoder_epochs` and measured BPB.
- `starcoder_curve_membership`: one row per point of a curve (1,058): the run, its `starcoder_epochs` on the curve's
  coordinate, and the three fits' predictions. 52 full-pool runs of the replay family are points of several curves:
  the dense-support design reuses a full-pool run for a smaller subset at a weight whose StarCoder draw never exhausts
  that subset (its `deterministic_aliases`), since the run then sees the same data. Such a run's observation carries
  the full pool's design and epochs; each of its points carries the epochs of its curve's subset, which the fits used.
- `starcoder_curves`: one row per curve (45): reference, family, label, epochs at full StarCoder share, the fit metrics,
  the curve's N, D and StarCoder subset, `fitter` and `grid_points`.
- `starcoder_fit_parameters` (990 rows) and `starcoder_dense_grid` (45,045 rows: 1,001 weights per curve): the three
  fits' parameters and predictions.

N and D come from the curve inventory the atlas fitted (its SHA-256 is in the atlas's `protocol.json`). The StarCoder
subset is the realized subset of the dense-support and coupled-onset designs (whole batches of the shuffled pool; for
the full pool, the pool's 216,567,300,822 tokens), and for the simulated-epoching families (the fixed-model and
matched ladders) the pool's share D / 5,729,908,864,777, the target budget, of which the loader keeps whole sequences.
Every atlas run used the reference seed 20260711 as trainer and data seed (the inventory's `seed_semantics`).
`starcoder_epochs` is the atlas's materialized-epoch coordinate, on which the Weibull-softplus fit places StarCoder: the
weight times the curve's `starcoder_epochs_at_p1`, which is D over the pool for full-pool curves and otherwise the
target's repetition (26.46 epochs at full share) times the subset's multiplier. The realized whole-batch subsets of the
replay family give up to 0.26% more epochs than that coordinate.

Observation ids key on the training run name. The atlas records a run as `run:<name>` or `wandb:<name>` and its
endpoint observation as `<that>@endpoint`; the export keys on `<name>`. These runs appear in no other config.

- `starcoder_diagonal`: the fixed-total-TPP-5 diagonal, 60 runs (4 matched-compute cells x 15 weights) with each
  cell's measured optimum.
- `starcoder_curricula`: the confirmed two-phase gains by training horizon and StarCoder subset (28 cells).
- `starcoder_curricula_matched_nd`: the best tied and two-phase schedules and their fitted gap in each of the 10
  matched model-size and token-budget cells; `starcoder_curricula_matched_nd_confirmation`: the eight fresh-seed pairs
  that confirm the selected two-phase schedule of the largest cell (mean gain 0.0061 BPB).
- `starcoder_transition`: the switch study's tables stacked, named by `table`: `ramp_endpoints` (offline final PL and
  C4 BPB of the 30 ramp and abrupt runs), `ramp_gains_over_october_abrupt`, `switch_window_offline_evals` (offline
  PL and C4 BPB of switch-window checkpoints), `train_loss_jump`, `ramp_eval_history` (the ramp runs' training-time
  PL and C4 BPB every 1,000 steps, which at the final step equal the offline values within 6e-7), and the changes in
  the switched-minus-tied gap around each switch: `eval_double_differences` from the training-time evaluations and
  `switch_window_offline_double_differences` from the offline ones. A column a table lacks is null; the integer
  columns are nullable integers.
- `starcoder_ramp_designs`: the launched schedules of the 30 ramp and abrupt runs, one row per stage (a 16-step
  mixture block of a ramp, or the constant stretch before or after it) with its first step and StarCoder weight, and
  the run's August abrupt source run and tied comparator. The loader fills each block's 2,048 sequences in integers
  (Levanter's `MixtureDataset`, which gives the remainder to the largest component), so a block's StarCoder count
  differs from weight x 2,048 by up to 4.5 sequences.
- `starcoder_trajectories`: the training loss at every logged step of the 50 runs whose seed means the switch-trace
  figure plots, re-pulled from W&B by `release/pull_switch_traces.py`, with the step of each run's switch; `run_label`
  is the series' label in `train_loss_jump`.

The transition tables and the trajectories key on training run names too (the gradient-conflict trajectory id, without
its W&B prefix), so a run has one id across them: the 4x confirmation winners are both the `conf_m400_untied`
trajectories and the offline evaluations' `august_abrupt` runs.
"""

import gzip
import hashlib
import json
import re

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.release import paths, tables
from experiments.domain_phase_mix.release.context import ExportContext

ATLAS = paths.RO / "starcoder_all_tied_curves_canonical_dsp_20260902"
INVENTORY = paths.RO / "starcoder_single_phase_curve_inventory_20260902/curve_inventory.csv"
DENSE_DESIGN = paths.REPO / "experiments/domain_phase_mix/starcoder_wsd80_dense_support_surface_design_20260808.json"
COUPLED_DESIGN = (
    paths.REPO / "experiments/domain_phase_mix/starcoder_wsd80_coupled_onset_dense_surface_design_20260830.json.gz"
)
DIAGONAL = paths.RO / "starcoder_wsd80_fixed_total_tpp5_diagonal_20260905"
CURRICULA = (
    paths.RO
    / "starcoder_wsd80_dense_horizon_replay_confirmation_scaling_20260811"
    / "confirmed_selected_policy_gain_by_horizon_and_replay.csv"
)
TRANSITION = paths.RO / "starcoder_wsd80_transition_transient_20261006"
TRANSITION_TABLES = (
    "ramp_endpoints",
    "ramp_gains_over_october_abrupt",
    "switch_window_offline_evals",
    "train_loss_jump",
    "ramp_eval_history",
    "eval_double_differences",
    "switch_window_offline_double_differences",
)
MATCHED_ND = paths.RO / "starcoder_wsd80_matched_nd_stage1_20260731"
MATCHED_OPTIMA = MATCHED_ND / "optimum_scaling_20260802/discovered_optimum_scaling.csv"
MATCHED_CONFIRMATION = MATCHED_ND / "confirmation_results_20260801"
RAMP_DESIGNS = tuple(
    paths.REPO / f"experiments/domain_phase_mix/starcoder_wsd80_transition_ramp_{name}.json"
    for name in ("design_20261006", "quick_design_20261007", "abrupt_design_20261007")
)
RAMP_DESIGN_FIELDS = ["support_id", "cell_id", "total_steps", "boundary_step", "steps_per_block"]
RAMP_RUN_FIELDS = [
    "arm",
    "ramp_width",
    "data_seed",
    "realized_starcoder_sequences",
    "target_starcoder_sequences",
    "pre_boundary_starcoder_sequences",
    "max_weight_nudge",
]
# The training-time evaluations kept from ramp_eval_history.csv, named as in ramp_endpoints.csv.
EVAL_HISTORY_METRICS = {
    "eval/paloma/dolma_100_programing_languages-llama3/bpb": "pl_bpb",
    "eval/paloma/c4_en-llama3/bpb": "c4_bpb",
}
RUN_NAME_COLUMNS = ("run_name", "trajectory_id")
INTERNAL_COLUMNS = (
    "training_run_id",
    "wandb_run_id",
    "wandb_id",
    "wandb_state",
    "wandb_url",
    "checkpoint_root",
    "metric_uri",
    "metric_source",
    "stream_identity_sha256",
)
FITTER = (
    "three independent full-data fits per curve: canonical DSP (full_fit_*; per-bucket benefit and harm amplitudes, "
    "thresholds and log rates, and an intercept; StarCoder at 26.46 and Nemotron-CC at 1 epoch per unit weight on "
    "every curve), a Weibull-benefit, softplus-harm response with a shared three-parameter shape and an unscaled "
    "nonnegative per-bucket head (weibull_softplus_unscaled_*; StarCoder at starcoder_epochs_at_p1 per unit weight, "
    "Nemotron-CC at D over its 5,729,908,864,777-token pool on full-pool curves and at 1 elsewhere), and Olmix's "
    "log-linear law c + exp(beta_N (1 - p) + beta_S p) under a summed Huber loss (olmix_*); dense grid of 1,001 "
    "StarCoder weights from 0 to 1"
)
GRID_POINTS = 1001
ATLAS_RUN_SOURCES = ("run", "wandb")
DENSE_FAMILY, COUPLED_FAMILY = "dense_horizon_replay", "coupled_lr_onset"
SEED = re.compile(r"seed (\d+)")
DESIGN = ["total_parameters", "non_embedding_parameters", "train_tokens", "starcoder_subset_tokens"]
SEEDS = ["data_seed", "trainer_seed"]
FIT_PREDICTIONS = [
    "full_fit_prediction_bpb",
    "weibull_softplus_unscaled_full_fit_prediction_bpb",
    "olmix_full_fit_prediction_bpb",
]


def _curve_designs(curves: pd.DataFrame, dense: dict) -> pd.DataFrame:
    """N, D, StarCoder subset and seed of each atlas curve, indexed by `curve_id`."""
    protocol = json.loads(paths.require(ATLAS / "protocol.json").read_text())
    if hashlib.sha256(paths.require(INVENTORY).read_bytes()).hexdigest() != protocol["input_hashes"][INVENTORY.name]:
        raise ValueError(f"{INVENTORY.name} is not the inventory the atlas fitted")
    inventory = tables.read_csv(INVENTORY).set_index("curve_id").loc[curves.curve_id]
    if not (inventory.family == curves.family.to_numpy()).all():
        raise ValueError("the inventory and the atlas disagree on a curve's family")
    axis = protocol["materialized_epoch_axis"]
    pool, target = axis["full_starcoder_source_tokens"], axis["simulated_epoch_target_budget"]
    cells = pd.DataFrame(dense["cells"]).set_index("cell_id")
    supports = pd.DataFrame(dense["supports"]).set_index(["cell_id", "support_id"])
    coupled = json.loads(gzip.decompress(paths.require(COUPLED_DESIGN).read_bytes()))

    subsets = []
    for curve_id, curve in inventory.iterrows():
        if curve.family == DENSE_FAMILY:
            _, cell_id, support_id, _ = curve_id.split("__")
            cell, support = cells.loc[cell_id], supports.loc[(cell_id, support_id)]
        elif curve.family == COUPLED_FAMILY:
            cell, support = coupled["cell"], coupled["support"]
            support_id = support["support_id"]
        else:  # simulated epoching: the loader keeps the share D / target of the pool
            subsets.append(round(pool * curve.planned_materialized_tokens / target))
            continue
        if (cell["materialized_tokens"], cell["total_parameters"], support_id) != (
            curve.planned_materialized_tokens,
            curve.total_parameters,
            curve.support_id,
        ):
            raise ValueError(f"{curve_id}: the design's cell or subset differs from the inventory's")
        subsets.append(int(support["starcoder_realized_support_tokens"]))

    seeds = [{int(s) for s in SEED.findall(text)} for text in inventory.seed_semantics]
    if any(len(s) != 1 for s in seeds):
        raise ValueError("a curve's seed semantics name no seed or several")
    design = pd.DataFrame(
        {
            "total_parameters": inventory.total_parameters,
            "non_embedding_parameters": inventory.non_embedding_parameters,
            "train_tokens": inventory.planned_materialized_tokens,
            "starcoder_subset_tokens": subsets,
            "data_seed": [s.pop() for s in seeds],
        },
        index=inventory.index,
    )
    design["trainer_seed"] = design.data_seed  # the inventory's "trainer/data seed"
    simulated = ~inventory.family.isin([DENSE_FAMILY, COUPLED_FAMILY]).to_numpy()
    epochs = design.train_tokens[simulated] / design.starcoder_subset_tokens[simulated]
    at_full_share = curves.set_index("curve_id").starcoder_epochs_at_p1.reindex(epochs.index)  # rounded to 1e-6
    if not np.allclose(epochs, at_full_share, rtol=0, atol=1e-6):
        raise ValueError("a simulated-epoching subset disagrees with the atlas's epochs at full share")
    return design


def _atlas_runs(
    ctx: ExportContext, curves: pd.DataFrame, designs: pd.DataFrame, aliases: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The atlas's observations, one per training run, and its (curve, run) memberships.

    `designs` is `_curve_designs`'s table and `aliases` the dense-support design's `deterministic_aliases`.
    """
    pred = tables.read_csv(paths.require(ATLAS / "predictions.csv"))
    source, run = pred.training_run_id.str.split(":", n=1, expand=True).T.to_numpy()
    if (
        not pd.Series(source).isin(ATLAS_RUN_SOURCES).all()
        or not (pred.training_run_id.map("{}@endpoint".format) == pred.observation_id).all()
    ):
        raise ValueError("an atlas run id is not `<run|wandb>:<name>` with observation `<id>@endpoint`")
    pred["run"] = run
    pred["observation_id"] = pred.run.map(ctx.registry.id_for)
    at_full_share = curves.set_index("curve_id").starcoder_epochs_at_p1
    pred["starcoder_epochs"] = pred.starcoder_weight * pred.curve_id.map(at_full_share)
    points = pred.merge(designs, left_on="curve_id", right_index=True, how="left", validate="many_to_one")
    assert points[DESIGN].notna().all().all(), "an atlas curve lacks a design"

    per_run = points.groupby("observation_id")
    if per_run[["run", "starcoder_weight", "observed_bpb", *DESIGN[:3], *SEEDS]].nunique().gt(1).any().any():
        raise ValueError("a run's curves disagree on its name, weight, BPB, model size, tokens or seeds")
    # A run in several curves is a full-pool run that the dense-support design reuses for smaller subsets; it keeps
    # the largest subset among its curves, and every other membership must be one of the design's aliases of it.
    own = points.sort_values("starcoder_subset_tokens", kind="stable").drop_duplicates("observation_id", keep="last")
    alias_curves = [
        f"dense_replay__{c}__{s}__endpoint" for c, s in zip(aliases.cell_id, aliases.support_id, strict=True)
    ]
    alias_points = set(zip(aliases.alias_of_run_name, alias_curves, strict=True))
    stand_ins = points.drop(index=own.index)
    if not all(point in alias_points for point in zip(stand_ins.run, stand_ins.curve_id, strict=True)):
        raise ValueError("a run belongs to a curve the dense-support design does not reuse it for")

    columns = ["observation_id", *DESIGN, *SEEDS, "starcoder_weight", "starcoder_epochs", "observed_bpb"]
    observations = own.sort_index()[columns].reset_index(drop=True)
    membership = pred[["curve_id", "observation_id", "starcoder_epochs", *FIT_PREDICTIONS]]
    return observations, membership


def _diagonal(ctx: ExportContext) -> pd.DataFrame:
    """The TPP-5 diagonal's runs with each cell's measured optimum; the cell constants the two files share must agree."""
    runs = tables.read_csv(paths.require(DIAGONAL / "observations.csv"))
    cells = tables.read_csv(paths.require(DIAGONAL / "curve_summary.csv"))
    table = runs.merge(cells, on="cell_id", how="left", validate="many_to_one", suffixes=("", "_cell"))
    assert table.observed_optimum_bpb.notna().all(), "a diagonal run's cell has no summary"
    shared = [c for c in cells if c != "cell_id" and c in runs]
    if not all((table[c] == table[f"{c}_cell"]).all() for c in shared):
        raise ValueError("a diagonal run and its cell summary disagree on a cell constant")
    table = table.drop(columns=[f"{c}_cell" for c in shared])
    table.insert(0, "observation_id", table.pop("run_name").map(ctx.registry.id_for))
    return table


def _transition(ctx: ExportContext) -> pd.DataFrame:
    """The switch study's tables stacked under a `table` column, with run names replaced by observation ids."""
    parts = []
    for name in TRANSITION_TABLES:
        part = tables.read_csv(paths.require(TRANSITION / f"{name}.csv"))
        if name == "ramp_eval_history":
            part = part[["run_name", "step", *EVAL_HISTORY_METRICS]].rename(columns=EVAL_HISTORY_METRICS)
        for column in [c for c in RUN_NAME_COLUMNS if c in part]:
            part.insert(0, "observation_id", part.pop(column).map(ctx.registry.id_for))
        part.insert(0, "table", name)
        parts.append(part)
    stacked = pd.concat(parts, ignore_index=True)
    integer = [c for c in stacked if all(pd.api.types.is_integer_dtype(part[c]) for part in parts if c in part)]
    return stacked.astype(dict.fromkeys(integer, "Int64"))


def _matched_confirmation(ctx: ExportContext) -> pd.DataFrame:
    """The matched-N,D confirmation pairs, each run keyed by its training run name."""
    pairs = tables.read_csv(paths.require(MATCHED_CONFIRMATION / "confirmation_pairs.csv"))
    runs = tables.read_csv(paths.require(MATCHED_CONFIRMATION / "confirmation_observations.csv"))
    # The last component of a W&B run URL is the run's id, which for these runs is the training run name.
    if not (runs.wandb_url.str.rsplit("/", n=1).str[-1] == runs.run_name).all():
        raise ValueError("a confirmation run's W&B id differs from its run name")
    for role in ("candidate", "comparator"):
        names = pairs.pop(f"{role}_wandb_url").str.rsplit("/", n=1).str[-1]
        if not names.isin(runs.run_name).all():
            raise ValueError(f"a confirmation {role} is not among the confirmation runs")
        pairs.insert(pairs.columns.get_loc(f"{role}_bpb"), f"{role}_observation_id", names.map(ctx.registry.id_for))
    return pairs


def _ramp_designs(ctx: ExportContext) -> pd.DataFrame:
    """The launched ramp and abrupt schedules, one row per stage of each run."""
    rows = []
    for path in RAMP_DESIGNS:
        design = json.loads(paths.require(path).read_text())
        for run in design["runs"]:
            ids = {
                f"{role}observation_id": ctx.registry.id_for(run[f"{role}run_name"]) for role in ("", "source_", "tied_")
            }
            for stage, (start_step, weight) in enumerate(run["stages"]):
                rows.append(
                    {
                        "design": design["design_version"],
                        **ids,
                        **{field: run[field] for field in RAMP_RUN_FIELDS},
                        **{field: design[field] for field in RAMP_DESIGN_FIELDS},
                        "stage": stage,
                        "start_step": start_step,
                        "starcoder_weight": weight,
                    }
                )
    table = pd.DataFrame(rows).astype({"ramp_width": "Int64"})
    starts = table.groupby("observation_id").start_step
    if table.groupby("observation_id").design.nunique().gt(1).any() or not starts.diff().dropna().gt(0).all():
        raise ValueError("a ramp run appears twice or its stages do not start at increasing steps")
    if not (starts.min().eq(0).all() and table.start_step.mod(table.steps_per_block).eq(0).all()):
        raise ValueError("a ramp schedule does not start at step 0 or a stage starts inside a mixture block")
    return table


def _trajectories(ctx: ExportContext) -> pd.DataFrame:
    trajectories = pd.read_parquet(paths.require(TRANSITION / "trajectories.parquet"))
    ids = {name: ctx.registry.id_for(name) for name in trajectories.run_name.unique()}
    trajectories.insert(0, "observation_id", trajectories.pop("run_name").map(ids))
    return trajectories


def export(ctx: ExportContext) -> dict[str, pd.DataFrame]:
    curves = tables.read_csv(paths.require(ATLAS / "curve_reference.csv")).merge(
        tables.read_csv(paths.require(ATLAS / "metrics.csv")),
        on=["curve_id", "curve_number", "curve_ref", "family"],
        how="left",
        validate="one_to_one",
    )
    assert curves.rows.notna().all(), "an atlas curve has no fit metrics"
    dense = json.loads(paths.require(DENSE_DESIGN).read_text())
    designs = _curve_designs(curves, dense)
    obs, membership = _atlas_runs(ctx, curves, designs, pd.DataFrame(dense["deterministic_aliases"]))
    curves = curves.merge(designs[DESIGN], left_on="curve_id", right_index=True, how="left", validate="one_to_one")
    curves["fitter"], curves["grid_points"] = FITTER, GRID_POINTS
    params = tables.read_csv(paths.require(ATLAS / "full_fit_parameters.csv"))
    grid = tables.read_csv(paths.require(ATLAS / "dense_curves.csv"))
    if not grid.groupby("curve_id").size().eq(GRID_POINTS).all() or set(grid.curve_id) != set(curves.curve_id):
        raise ValueError(f"the dense grid must hold {GRID_POINTS} weights for each atlas curve")
    out = {
        "starcoder_observations": obs,
        "starcoder_curve_membership": membership,
        "starcoder_curves": curves,
        "starcoder_fit_parameters": params,
        "starcoder_dense_grid": grid,
        "starcoder_diagonal": _diagonal(ctx),
        "starcoder_curricula": tables.read_csv(paths.require(CURRICULA)),
        "starcoder_curricula_matched_nd": tables.read_csv(paths.require(MATCHED_OPTIMA)),
        "starcoder_curricula_matched_nd_confirmation": _matched_confirmation(ctx),
        "starcoder_transition": _transition(ctx),
        "starcoder_ramp_designs": _ramp_designs(ctx),
        "starcoder_trajectories": _trajectories(ctx),
    }
    for name, t in out.items():
        drop = [c for c in t.columns if c in INTERNAL_COLUMNS or c.endswith(("_url", "_uri"))]
        out[name] = t.drop(columns=drop)
    return out
