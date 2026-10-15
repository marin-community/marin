# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The main swarm: 280 fit rows from the mixture-selection bundle plus the ten proportional repeats."""

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.release import keys, naming, paths, tables
from experiments.domain_phase_mix.release.context import ExportContext

TABLE = "swarm_qwen3_360m_1p6b"
SWARM_WEIGHTS = paths.MIXSEL / "data/swarm_weights.csv"
SWARM_OUTCOMES = paths.MIXSEL / "data/swarm_outcomes.csv"
PACKET = paths.RO / "two_phase_surrogate_collaborator_packet_20260721/data/canonical/delphi_3e18_one_phase_fit.csv"
INVENTORY = paths.AG / "sft_midtraining_probe/checkpoint_inventory.csv"
NOISE = paths.RO / "delphi_3e18_proportional_noise_floor_20260703"
UNCH_REPEATS = paths.RO / "table9_reliability_20260905/proportional_uncheatable_components.csv"
REPEAT_SEEDS = (
    paths.RO / "two_phase_surrogate_collaborator_packet_20260721/data/raw/delphi_3e18_heldouts/heldout_registry.csv"
)
UNCH_POOLED = "eval/uncheatable_eval/bpb"
REPEAT_RUN = r"^proportional_noise_3e18_(?P<letter>[a-j])_3e18-[0-9a-f]{6}$"
META = ["observation_id", "run_label", "design_kind", "fit_inclusion", "data_seed", "trainer_seed"]
# The packet's candidate_kind separates the 238 newly trained designs from the 42 phase-constant designs whose
# checkpoints are the two-phase swarm's runs; the design name separates the 3 baselines from the 39 deletions.
NEW_TRAINING = "scheduled_new_training"
TWO_PHASE_ALIAS = "reused_exact_phase_tied_alias"


def required_sources() -> list[Path]:
    return [
        SWARM_WEIGHTS,
        SWARM_OUTCOMES,
        PACKET,
        INVENTORY,
        NOISE / "noise_component_matrix.csv",
        NOISE / "noise_panel.csv",
        UNCH_REPEATS,
        REPEAT_SEEDS,
    ]


def _public_weights(frame: pd.DataFrame, internal_cols: list[str], prefix: str = "w__") -> pd.DataFrame:
    return frame[internal_cols].rename(columns={c: f"{prefix}{naming.bucket_public(c)}" for c in internal_cols})


def _public_metrics(frame: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    return frame[metric_cols].rename(columns={c: f"bpb__{naming.task_public(c)}" for c in metric_cols})


def _design_kind(design: str, candidate_kind: str) -> str:
    if candidate_kind == NEW_TRAINING and design.startswith("run_"):
        return "sampled"
    if candidate_kind == TWO_PHASE_ALIAS and design.startswith("baseline_"):
        return "baseline"
    if candidate_kind == TWO_PHASE_ALIAS and design.startswith("pctrl_del_"):
        return "deletion"
    raise ValueError(f"no design kind for {design!r} with candidate_kind {candidate_kind!r}")


def _finish(df: pd.DataFrame, ctx: ExportContext) -> pd.DataFrame:
    wcols = [f"w__{b}" for b in ctx.buckets]
    for b in ctx.buckets:
        df[f"pf__{b}"] = 1.0  # the swarm trains on full pools; only the held-out a-priori runs vary pool fractions
    df["mixture_id"] = [keys.mixture_id(dict(zip(ctx.buckets, row, strict=True))) for row in df[wcols].to_numpy()]
    pf = dict.fromkeys(ctx.buckets, 1.0)
    df["exposure_id"] = [
        keys.exposure_id(dict(zip(ctx.buckets, row, strict=True)), pf, paths.PARTITION_REVISION)
        for row in df[wcols].to_numpy()
    ]
    df["components_available"] = df[[f"bpb__{t}" for t in ctx.tasks]].notna().all(axis=1)
    df = tables.add_aggregates(df, ctx.weights)  # every run has all 58 components; `export` asserts it
    return tables.order_columns(df, ctx.buckets, ctx.tasks)


def export(ctx: ExportContext) -> dict[str, pd.DataFrame]:
    weights = tables.read_csv(paths.require(SWARM_WEIGHTS))
    outcomes = tables.read_csv(paths.require(SWARM_OUTCOMES)).set_index("run")
    packet = tables.read_csv(paths.require(PACKET))[["row_id", "candidate_kind"]]
    inventory = tables.read_csv(paths.require(INVENTORY))
    inventory = inventory[inventory.panel == "p2_one_phase_swarm"][
        ["run_name", "anchor_or_design_id", "data_seed", "trainer_seed"]
    ]

    internal_buckets = [c for c in weights.columns if c not in ("run", "calibration")]
    fit = pd.concat(
        [
            weights[["run"]],
            _public_weights(weights, internal_buckets),
            _public_metrics(outcomes.loc[weights.run].reset_index(drop=True), list(outcomes.columns)),
        ],
        axis=1,
    )
    # The bundle's `run`, the packet's `row_id` and the inventory's `run_name` are the training run name
    # (`singleavg_fit_002_run_00002`); the inventory's `anchor_or_design_id` is the design name (`run_00002`).
    fit = fit.merge(packet, left_on="run", right_on="row_id", how="left", validate="one_to_one")
    assert fit.candidate_kind.notna().all(), "every fit row needs a candidate_kind from the July packet"
    fit = fit.merge(inventory, left_on="run", right_on="run_name", how="left", validate="one_to_one")
    assert fit.run_name.notna().all(), "every fit row needs its design name and seeds from the SFT inventory"
    fit["design_kind"] = [_design_kind(d, k) for d, k in zip(fit.anchor_or_design_id, fit.candidate_kind, strict=True)]
    fit["observation_id"] = fit.run_name.map(ctx.registry.id_for)
    fit["run_label"] = fit.anchor_or_design_id  # design names (run_00002, baseline_proportional, pctrl_del_<bucket>)
    fit["fit_inclusion"] = True

    columns = [*META, *(f"w__{b}" for b in ctx.buckets), *(f"bpb__{t}" for t in ctx.tasks)]  # exported allowlist
    fit = fit[columns]
    repeats = _repeats(ctx, proportional=fit[fit.run_label == "baseline_proportional"])
    table = pd.concat([fit, repeats[columns]], ignore_index=True)
    out = _finish(table, ctx)
    assert out.notna().all().all(), f"missing values in {out.columns[out.isna().any()].tolist()}"
    return {TABLE: out}


def _repeats(ctx: ExportContext, proportional: pd.DataFrame) -> pd.DataFrame:
    """The ten proportional repeats, paired across sources on the run letter.

    The OlmoBaseEval Easy matrix is indexed by letter; the Uncheatable file and the July held-out registry (seeds)
    name the runs `proportional_noise_3e18_<letter>_3e18-<hash>`. The Uncheatable file also holds
    `proportional_3e18-ebc4aa`, the June scaling-ladder run at 3e18 (data seed 660700); it is neither a repeat nor
    the swarm's `baseline_proportional`, so it is left out.
    """
    assert len(proportional) == 1, f"expected one baseline_proportional fit row, found {len(proportional)}"
    t9 = tables.read_csv(paths.require(NOISE / "noise_component_matrix.csv"), index_col=0)  # 10 rows, index = letter
    panel = tables.read_csv(paths.require(NOISE / "noise_panel.csv"), index_col="letter")
    unch = tables.read_csv(paths.require(UNCH_REPEATS))
    unch["letter"] = unch.run.str.extract(REPEAT_RUN)["letter"]
    unch = unch.dropna(subset=["letter"]).set_index("letter")
    assert len(t9) == 10 and sorted(unch.index) == sorted(t9.index), f"repeat letters {sorted(unch.index)}"
    # noise_panel.csv records each letter's pooled Uncheatable BPB; equality with the named runs confirms the pairing.
    assert np.allclose(unch[UNCH_POOLED], panel.uncheatable_bpb.reindex(unch.index), rtol=0, atol=1e-12)
    recorded = tables.read_csv(
        paths.require(REPEAT_SEEDS),
        usecols=["wandb_run_name", "data_seed", "trainer_seed", "fit_panel_run_name", "fit_panel_max_abs_distance"],
    ).set_index("wandb_run_name")
    recorded = recorded.loc[unch.run]
    assert (recorded.fit_panel_run_name == "baseline_proportional").all(), "the repeats train the proportional mixture"
    assert (recorded.fit_panel_max_abs_distance < 1e-9).all(), "the repeats train the proportional mixture"
    unch_metrics = unch.drop(columns=["run", UNCH_POOLED])
    rows = []
    for letter in map(str, t9.index):
        run_name = str(unch.run[letter])
        row = {
            "observation_id": ctx.registry.id_for(run_name),
            "run_label": f"proportional_repeat_{letter}",
            "design_kind": "repeat",
            "fit_inclusion": False,
            "data_seed": recorded.data_seed[run_name],
            "trainer_seed": recorded.trainer_seed[run_name],
        }
        row.update({f"w__{b}": proportional[f"w__{b}"].iloc[0] for b in ctx.buckets})
        row.update({f"bpb__{naming.task_public(c)}": t9.at[letter, c] for c in t9.columns})
        row.update({f"bpb__{naming.task_public(c)}": unch_metrics.at[letter, c] for c in unch_metrics.columns})
        rows.append(row)
    return pd.DataFrame(rows)
