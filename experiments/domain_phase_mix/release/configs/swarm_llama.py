# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The two archived single-phase swarms: Llama 160M/1.2B (242 fit runs) and Llama 200M/6B (280 fit runs).

Observation ids key on each run's training run name with its six-hex-digit hash (its checkpoint directory, which W&B
records as the run id), as the held-out table does: the two swarms reuse hash-less names such as `singleavg_run_00125`
for different runs. `run_label` is the design name (`run_00125`, `baseline_proportional`, `pctrl_del_<bucket>`).

Every row holds its own run's evaluation. The 200M swarm trained 240 runs and reuses 40 single-mixture runs of the
two-phase swarm (the 39 deletions and the stratified baseline), whose native evaluations the parity panel carries. The
panel, and the July packet copied from it, replaced the proportional run's targets by the mean of 11 proportional runs;
the table keeps the run's own values. Uncheatable components are the W&B endpoints that
export_single_phase_uncheatable_components_20260902.py froze after checking each run's aggregate.
"""

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.release import keys, naming, paths, run_names, tables
from experiments.domain_phase_mix.release.context import ExportContext

SMALL = "swarm_llama_160m_1p2b"
LARGE = "swarm_llama_200m_6b"
AUDIT_160 = paths.RO / "60m_39bucket_checkpoint_audit_20260724"
GAP_160 = paths.RO / "60m_table9_gap_completion_20260725/table9_eval_results.csv"
LAUNCH_160 = paths.RO / "single_phase_exposure_average_60m_1p2b/single_phase_exposure_average_manifest.csv"
PACKET = paths.RO / "two_phase_surrogate_collaborator_packet_20260721/data/canonical"
PARITY_200 = paths.RO / "olmo_base_easy_one_phase_parity_panel_300m_20260628"
LAUNCH_200 = (
    paths.RO
    / "single_phase_exposure_average_qsplit240_300m_6b"
    / "single_phase_exposure_average_qsplit240_300m_manifest.csv"
)
DELETIONS_200 = paths.RO / "proportional_controllability_300m_20260520/training_manifest.csv"
UNCHEATABLE = paths.RO / "single_phase_componentwise_canonical_dsp_20260902/input"
UNCH_PREFIX = "eval/uncheatable_eval/"
UNCH_POOLED = "eval/uncheatable_eval/bpb"
PROPORTIONAL_200 = "singleavg_baseline_proportional"
# Training setup from Appendix table tab:a-hyperparameters, which prints these exact counts rounded (157.5M, 59M, 1.2B;
# 201.1M, 102.6M, 6.0B); the counts come from the 160M audit's compute_comparison.csv. Steps are the launch manifests'
# num_train_steps: the tokens over batches of 128 sequences of 2,048 tokens.
SETUP = {
    SMALL: dict(
        model_family="llama",
        total_parameters=157_499_136,
        non_embedding_parameters=58_998_528,
        train_tokens=1_199_833_088,
        train_steps=4_577,
        protocol="MuonH, peak learning rate 0.02, warmup-stable-decay",
    ),
    LARGE: dict(
        model_family="llama",
        total_parameters=201_149_184,
        non_embedding_parameters=102_648_576,
        train_tokens=5_999_951_872,
        train_steps=22_888,
        protocol="MuonH, peak learning rate 0.01, warmup-stable-decay",
    ),
}
DESIGN_KINDS = {"run_": "sampled", "baseline_": "baseline", "pctrl_del_": "deletion"}
# The launch manifests leave trainer_seed unset, which trains with Marin's default seed (experiments/defaults.py).
DEFAULT_TRAINER_SEED = 0


def _join(key: pd.Series, source: pd.DataFrame, on: str, label: str) -> pd.DataFrame:
    """`source`'s row for each key, indexed like `key`; raises unless every key matches exactly one row."""
    joined = key.rename(on).to_frame().merge(source, on=on, how="left", validate="one_to_one", indicator=True)
    unmatched = key[(joined._merge != "both").to_numpy()]
    assert unmatched.empty, f"{label}: no source row for {unmatched.tolist()}"
    return joined.drop(columns="_merge").set_axis(key.index)


def _metrics(frame: pd.DataFrame, columns: list[str], tasks: list[str]) -> pd.DataFrame:
    """`columns` renamed to `bpb__<task>` in the order of `tasks`, which they must cover exactly."""
    out = frame[columns].rename(columns={c: f"bpb__{naming.task_public(c)}" for c in columns})
    expected = [f"bpb__{t}" for t in tasks]
    assert sorted(out.columns) == sorted(expected), f"metric columns {sorted(out.columns)} do not cover {tasks}"
    return out[expected]


def _uncheatable(frame: pd.DataFrame, ctx: ExportContext) -> pd.DataFrame:
    return _metrics(frame, [c for c in frame if c.startswith(UNCH_PREFIX) and c != UNCH_POOLED], ctx.tasks[:7])


def _single_phase_weights(frame: pd.DataFrame, ctx: ExportContext) -> pd.DataFrame:
    """`w__<bucket>` from a panel's `phase_0_<bucket>` columns, after checking that both phases use one mixture."""
    phase0 = [c for c in frame.columns if c.startswith("phase_0_")]
    phase1 = [f"phase_1_{c.removeprefix('phase_0_')}" for c in phase0]
    assert len(phase0) == 39, f"expected 39 phase-0 weight columns, found {len(phase0)}"
    assert np.array_equal(frame[phase0].to_numpy(), frame[phase1].to_numpy()), "a single-phase run has one mixture"
    weights = frame[phase0].rename(columns={c: f"w__{naming.bucket_public(c.removeprefix('phase_0_'))}" for c in phase0})
    return weights[[f"w__{b}" for b in ctx.buckets]]


def _design_kind(design: str) -> str:
    for prefix, kind in DESIGN_KINDS.items():
        if design.startswith(prefix):
            return kind
    raise ValueError(f"no design kind for {design!r}")


def _finish(table: pd.DataFrame, name: str, ctx: ExportContext) -> pd.DataFrame:
    for column, value in SETUP[name].items():
        table[column] = value
    table["train_flops"] = 6.0 * table.total_parameters * table.train_tokens  # the paper's FLOPs per run: 6ND, total N
    for b in ctx.buckets:
        table[f"pf__{b}"] = 1.0  # the swarms train on full pools
    pool_fractions = dict.fromkeys(ctx.buckets, 1.0)
    mixtures = [dict(zip(ctx.buckets, row, strict=True)) for row in table[[f"w__{b}" for b in ctx.buckets]].to_numpy()]
    table["mixture_id"] = [keys.mixture_id(w) for w in mixtures]
    table["exposure_id"] = [keys.exposure_id(w, pool_fractions, paths.PARTITION_REVISION) for w in mixtures]
    table["aggregate_version"] = "legacy"  # both swarms' source Uncheatable aggregates use the legacy weights
    table["components_available"] = table[[f"bpb__{t}" for t in ctx.tasks]].notna().all(axis=1)
    table = tables.add_aggregates(table, ctx.weights)  # NaN wherever a component is missing
    return tables.order_columns(table, ctx.buckets, ctx.tasks)


def export(ctx: ExportContext) -> dict[str, pd.DataFrame]:
    return {SMALL: _export_160(ctx), LARGE: _export_200(ctx)}


def _export_160(ctx: ExportContext) -> pd.DataFrame:
    fit = pd.read_csv(paths.require(AUDIT_160 / "fit_single_phase.csv"))
    assert len(fit) == 242 and (fit.run_name == "singleavg_" + fit.paired_run_name).all()
    hashed = fit.checkpoint_root.map(run_names.training_run_name_from_uri)  # `singleavg_run_00125-9ca4a6`

    gap = _join(fit.run_name, pd.read_csv(paths.require(GAP_160)), "run_name", "160M OlmoBaseEval Easy")
    obe = _metrics(gap, [c for c in gap if c.startswith("table9/")], ctx.tasks[7:])
    assert np.allclose(obe.mean(axis=1), fit.table9_macro_bpb, rtol=0, atol=1e-12), "components differ from the audit"
    unch = pd.read_csv(paths.require(UNCHEATABLE / "60m_uncheatable_components.csv"))
    unch = _join(hashed, unch, "wandb_run_id", "160M Uncheatable")
    assert np.allclose(unch[UNCH_POOLED], fit.uncheatable_bpb, rtol=0, atol=1e-12), "endpoints differ from the audit"
    launch = _join(fit.run_name, pd.read_csv(paths.require(LAUNCH_160)), "run_name", "160M launch manifest")
    assert (launch.num_train_steps == SETUP[SMALL]["train_steps"]).all()

    table = pd.DataFrame(
        {
            "observation_id": hashed.map(ctx.registry.id_for),
            "run_label": fit.paired_run_name,  # design names: run_00125, baseline_<method>
            "design_kind": fit.paired_run_name.map(_design_kind),
            "fit_inclusion": True,
            "data_seed": launch.data_seed.astype("Int64"),
            "trainer_seed": launch.trainer_seed.fillna(DEFAULT_TRAINER_SEED).astype("Int64"),
        }
    )
    table = pd.concat([table, _single_phase_weights(fit, ctx), _uncheatable(unch, ctx), obe], axis=1)
    table["bpb__uncheatable_source"] = fit.uncheatable_bpb
    table["bpb__obe_easy_source"] = fit.table9_macro_bpb
    return _finish(table, SMALL, ctx)


def _export_200(ctx: ExportContext) -> pd.DataFrame:
    panel = pd.read_csv(paths.require(PARITY_200 / "one_phase_augmented_fit_panel.csv"))
    assert len(panel) == 280 and panel.run_name.is_unique
    trained = panel.panel_source == "single_phase_qsplit_signal"
    alias = panel.is_shared_checkpoint_alias  # the stratified baseline, which is also a two-phase swarm row
    assert trained.sum() == 240 and alias.sum() == 1 and not (trained & alias).any()
    # The packet and the component file prefix the deletions' names (`singleavg_pctrl_del_<bucket>`).
    row_id = "singleavg_" + panel.run_name.str.removeprefix("singleavg_")

    unch = pd.read_csv(paths.require(UNCHEATABLE / "300m_uncheatable_components.csv"))
    unch = _join(row_id, unch, "row_id", "200M Uncheatable")
    hashed = unch.wandb_run_id  # `singleavg_run_00002-f84c49`, `pctrl_del_dolma3_arxiv-079b36`
    physical = panel.shared_checkpoint_run_name.fillna(panel.run_name)
    assert (hashed.str.replace(r"-[0-9a-f]{6}$", "", regex=True) == physical).all(), "run ids must name the panel's runs"

    obe_columns = [c for c in panel if c.startswith(("olmo_base_eval/easy_bpb/", "mmlu_"))]
    panel_obe = _metrics(panel, obe_columns, ctx.tasks[7:])
    native = pd.read_csv(paths.require(PARITY_200 / "single_phase_table9_wide.csv"))
    native = _join(panel.run_name[trained], native, "run_name", "200M native OlmoBaseEval Easy")
    native_obe = _metrics(native, obe_columns, ctx.tasks[7:])
    replaced = (panel_obe[trained] - native_obe).abs().max(axis=1) > 1e-12
    assert panel.run_name[trained][replaced].tolist() == [PROPORTIONAL_200], "the panel replaced other runs' values"
    references = pd.read_csv(paths.require(PARITY_200 / "proportional_reference_table9.csv"))
    replaced_macro = panel.table9_macro_bpb[panel.run_name == PROPORTIONAL_200]
    assert len(references) == 11 and np.allclose(replaced_macro, references.table9_macro_bpb.mean(), rtol=0, atol=1e-12)
    # Native evaluations: the wide file's for the 240 trained runs, the panel's for the 40 reused ones.
    obe = pd.concat([native_obe, panel_obe[~trained]]).loc[panel.index]
    obe_source = pd.concat([native.native_table9_macro_bpb, panel.table9_macro_bpb[~trained]]).loc[panel.index]

    packet = _join(row_id, pd.read_csv(paths.require(PACKET / "300m_one_phase_fit.csv")), "row_id", "200M packet")
    own = panel.run_name != PROPORTIONAL_200
    assert np.allclose(packet.uncheatable_bpb[own], unch[UNCH_POOLED][own], rtol=0, atol=1e-12)
    assert np.allclose(packet.table9_macro_bpb[own], obe_source[own], rtol=0, atol=1e-12)

    # The stratified baseline's two-phase launch left no manifest in the sources, so its seeds stay null.
    launches = pd.concat([pd.read_csv(paths.require(LAUNCH_200)), pd.read_csv(paths.require(DELETIONS_200))])
    launch = _join(panel.run_name[~alias], launches, "run_name", "200M launch manifests")
    assert (launch.num_train_steps == SETUP[LARGE]["train_steps"]).all()

    table = pd.DataFrame(
        {
            "observation_id": hashed.map(ctx.registry.id_for),
            "run_label": panel.run_name.str.removeprefix("singleavg_"),
            "design_kind": panel.run_name.str.removeprefix("singleavg_").map(_design_kind),
            "fit_inclusion": True,
            "data_seed": launch.data_seed.reindex(panel.index).astype("Int64"),
            "trainer_seed": launch.trainer_seed.fillna(DEFAULT_TRAINER_SEED).reindex(panel.index).astype("Int64"),
        }
    )
    table = pd.concat([table, _single_phase_weights(panel, ctx), _uncheatable(unch, ctx), obe], axis=1)
    table["bpb__uncheatable_source"] = unch[UNCH_POOLED]
    table["bpb__obe_easy_source"] = obe_source
    return _finish(table, LARGE, ctx)
