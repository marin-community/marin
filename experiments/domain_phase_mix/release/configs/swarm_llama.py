# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The two archived single-phase swarms: Llama 160M/1.2B (281 fit runs) and Llama 200M/6B (280 fit runs), each with
the ten proportional repeats outside the fit set.

Observation ids key on each run's training run name with its six-hex-digit hash (its checkpoint directory, which W&B
records as the run id), as the held-out table does: the two swarms reuse hash-less names such as `singleavg_run_00125`
for different runs. `run_label` is the design name (`run_00125`, `baseline_proportional`, `pctrl_del_<bucket>`,
`proportional_repeat_<trainer seed>`).

Every row holds its own run's evaluation. The 160M fit swarm is the audit's 242 single-phase runs plus the 39
single-bucket deletions, which the audit files with its held-out runs and the held-out table also publishes. The 200M
swarm trained 240 runs and reuses 40 single-mixture runs of the two-phase swarm (the 39 deletions and the stratified
baseline), whose native evaluations the parity panel carries. The panel, and the July packet copied from it, replaced
the 200M proportional run's targets by the mean of 11 proportional runs: the ten repeats here and the two-phase swarm's
proportional run, which no table holds yet. Uncheatable components are W&B endpoints that a builder froze after
checking each run's aggregate.

The sources record each run's Uncheatable aggregate under the archived swarms' legacy corpus weights, so every row's
aggregates are recomputed from its components (`tables.add_aggregates`), which every row has.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from marin.evaluation.olmo_base_eval import components as obe_components

from experiments.domain_phase_mix.release import keys, naming, paths, run_names, tables
from experiments.domain_phase_mix.release.context import ExportContext

SMALL = "swarm_llama_160m_1p2b"
LARGE = "swarm_llama_200m_6b"
AUDIT_160 = paths.RO / "60m_39bucket_checkpoint_audit_20260724"
GAP_160 = paths.RO / "60m_table9_gap_completion_20260725/table9_eval_results.csv"
LAUNCH_160 = paths.RO / "single_phase_exposure_average_60m_1p2b/single_phase_exposure_average_manifest.csv"
# The audit's Table-9 source for the deletions and repeats: 47 leaf tasks and 57 MMLU subjects per run.
NATIVE_160 = paths.RO / "olmo_base_easy_full_results_60m_300m_20260625/olmo_base_easy_full_results_60m_300m_wide.csv"
# Uncheatable components of the 160M held-out runs, which the held-out table publishes for the same runs.
FROZEN_COMPONENTS = paths.RO / "single_phase_heldout_benchmark_20260902/heldout_components.csv"
REPEATS = paths.REPO / "experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/raw_metric_matrix_300m"
PACKET = paths.RO / "two_phase_surrogate_collaborator_packet_20260721/data/canonical"
PARITY_200 = paths.RO / "olmo_base_easy_one_phase_parity_panel_300m_20260628"
LAUNCH_200 = (
    paths.RO
    / "single_phase_exposure_average_qsplit240_300m_6b"
    / "single_phase_exposure_average_qsplit240_300m_manifest.csv"
)
DELETIONS_200 = paths.RO / "proportional_controllability_300m_20260520/training_manifest.csv"
UNCHEATABLE = paths.RO / "single_phase_componentwise_canonical_dsp_20260902/input"
UNCH_COMPONENT = re.compile(r"eval/uncheatable_eval/[^/]+/bpb")  # per corpus, not the pooled `.../eval/bpb`
UNCH_POOLED = "eval/uncheatable_eval/bpb"
OBE_200 = ("olmo_base_eval/easy_bpb/", "mmlu_")  # column prefixes of the 200M sources' 51 components
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
DESIGN_KINDS = {"run_": "sampled", "baseline_": "baseline", "pctrl_del_": "deletion", "proportional_repeat_": "repeat"}
# The launch manifests leave trainer_seed unset, which trains with Marin's default seed (experiments/defaults.py).
DEFAULT_TRAINER_SEED = 0


def required_sources() -> list[Path]:
    return [
        AUDIT_160 / "fit_single_phase.csv",
        AUDIT_160 / "heldout_observations.csv",
        AUDIT_160 / "repeat_observations.csv",
        GAP_160,
        LAUNCH_160,
        NATIVE_160,
        FROZEN_COMPONENTS,
        UNCHEATABLE / "60m_uncheatable_components.csv",
        UNCHEATABLE / "300m_uncheatable_components.csv",
        REPEATS / "noise_baseline_proportional_variable_subset_60m_1p2b.csv",
        REPEATS / "noise_baseline_proportional_variable_subset_300m_6b.csv",
        PARITY_200 / "one_phase_augmented_fit_panel.csv",
        PARITY_200 / "single_phase_table9_wide.csv",
        PARITY_200 / "proportional_reference_table9.csv",
        PACKET / "300m_one_phase_fit.csv",
        LAUNCH_200,
        DELETIONS_200,
    ]


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
    return _metrics(frame, [c for c in frame if UNCH_COMPONENT.fullmatch(c)], ctx.tasks[:7])


def _native_obe(frame: pd.DataFrame, ctx: ExportContext) -> pd.DataFrame:
    """The 51 OlmoBaseEval Easy components from a native export of the 47 leaf tasks and the 57 MMLU subjects.

    Each MMLU category is the size-weighted mean of its subjects' BPB, as the evaluator computes it.
    """
    leaves = {f"bpb__{t}": frame[f"olmo_base_eval/easy_bpb/{t}/bpb"] for t in obe_components.leaf_components()}
    out = pd.DataFrame(leaves)
    for category, weights in obe_components.MMLU_CATEGORY_WEIGHTS.items():
        subjects = [f"olmo_base_eval/easy_bpb/{subject}_rc/bpb" for subject in weights]
        out[f"bpb__{category}"] = frame[subjects].to_numpy() @ np.fromiter(weights.values(), float)
    return out[[f"bpb__{t}" for t in ctx.tasks[7:]]]


def _single_phase_weights(frame: pd.DataFrame, ctx: ExportContext) -> pd.DataFrame:
    """`w__<bucket>` from a panel's `phase_0_<bucket>` columns, after checking that both phases use one mixture."""
    phase0 = [c for c in frame.columns if c.startswith("phase_0_")]
    phase1 = [f"phase_1_{c.removeprefix('phase_0_')}" for c in phase0]
    assert len(phase0) == 39, f"expected 39 phase-0 weight columns, found {len(phase0)}"
    assert np.array_equal(frame[phase0].to_numpy(), frame[phase1].to_numpy()), "a single-phase run has one mixture"
    weights = frame[phase0].rename(columns={c: f"w__{naming.bucket_public(c.removeprefix('phase_0_'))}" for c in phase0})
    return weights[[f"w__{b}" for b in ctx.buckets]]


def _check_deletions(weights: pd.DataFrame, removed: pd.Series, proportional: pd.Series) -> None:
    """Each deletion zeroes one bucket and renormalizes the proportional mixture over the other 38; all 39 occur."""
    assert sorted(f"w__{b}" for b in removed) == sorted(proportional.index), "each bucket must be deleted once"
    for (_, row), bucket in zip(weights.iterrows(), removed, strict=True):
        expected = proportional / (1.0 - proportional[f"w__{bucket}"])
        expected[f"w__{bucket}"] = 0.0
        assert np.allclose(row[expected.index], expected, rtol=0, atol=1e-12), f"the deletion of {bucket}"


def _design_kind(design: str) -> str:
    for prefix, kind in DESIGN_KINDS.items():
        if design.startswith(prefix):
            return kind
    raise ValueError(f"no design kind for {design!r}")


def _rows(
    ctx: ExportContext,
    *,
    run_name: pd.Series,
    label: pd.Series,
    fit: bool,
    seeds: pd.DataFrame,
    weights: pd.DataFrame,
    uncheatable: pd.DataFrame,
    obe: pd.DataFrame,
    recorded_uncheatable: pd.Series,
    recorded_obe_easy: pd.Series,
) -> pd.DataFrame:
    """Rows keyed on hashed training run names (`run_name`), with each run's recorded aggregates, which `_finish`
    consumes; every argument is indexed alike."""
    meta = pd.DataFrame(
        {
            "observation_id": run_name.map(ctx.registry.id_for),
            "run_label": label,
            "design_kind": label.map(_design_kind),
            "fit_inclusion": fit,
            "data_seed": seeds.data_seed.astype("Int64"),
            "trainer_seed": seeds.trainer_seed.astype("Int64"),
        }
    )
    table = pd.concat([meta, weights, uncheatable, obe], axis=1)
    assert len(table) == len(meta), "row parts must share one index"
    table["recorded_uncheatable"] = recorded_uncheatable
    table["recorded_obe_easy"] = recorded_obe_easy
    return table


def _proportional(table: pd.DataFrame, ctx: ExportContext) -> pd.Series:
    """The weights of the fit row that trains the proportional mixture."""
    rows = table.loc[table.run_label == "baseline_proportional", [f"w__{b}" for b in ctx.buckets]]
    assert len(rows) == 1, f"expected one proportional fit row, found {len(rows)}"
    return rows.iloc[0]


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
    table["components_available"] = table[[f"bpb__{t}" for t in ctx.tasks]].notna().all(axis=1)
    recorded = pd.DataFrame({objective: table.pop(f"recorded_{objective}") for objective in ctx.weights})
    # A legacy-weighted Uncheatable value is never used: a row without its components raises.
    fixed_rule = pd.DataFrame({"uncheatable": False, "obe_easy": True}, index=table.index)
    table = tables.add_aggregates(table, ctx.weights, recorded, fixed_rule)
    return tables.order_columns(table, ctx.buckets, ctx.tasks)


def export(ctx: ExportContext) -> dict[str, pd.DataFrame]:
    return {SMALL: _export_160(ctx), LARGE: _export_200(ctx)}


def _export_160(ctx: ExportContext) -> pd.DataFrame:
    fit = _fit_160(ctx)
    proportional = _proportional(fit, ctx)
    native = tables.read_csv(paths.require(NATIVE_160), low_memory=False)
    native = native[native.scale == "60m_1p2b"]
    parts = [fit, _deletions_160(ctx, native, proportional), _repeats_160(ctx, native, proportional)]
    return _finish(pd.concat(parts, ignore_index=True), SMALL, ctx)


def _fit_160(ctx: ExportContext) -> pd.DataFrame:
    fit = tables.read_csv(paths.require(AUDIT_160 / "fit_single_phase.csv"))
    assert len(fit) == 242 and (fit.run_name == "singleavg_" + fit.paired_run_name).all()
    hashed = fit.checkpoint_root.map(run_names.training_run_name_from_uri)  # `singleavg_run_00125-9ca4a6`

    gap = _join(fit.run_name, tables.read_csv(paths.require(GAP_160)), "run_name", "160M OlmoBaseEval Easy")
    obe = _metrics(gap, [c for c in gap if c.startswith("table9/")], ctx.tasks[7:])
    assert np.allclose(obe.mean(axis=1), fit.table9_macro_bpb, rtol=0, atol=1e-12), "components differ from the audit"
    unch = tables.read_csv(paths.require(UNCHEATABLE / "60m_uncheatable_components.csv"))
    unch = _join(hashed, unch, "wandb_run_id", "160M Uncheatable")
    assert np.allclose(unch[UNCH_POOLED], fit.uncheatable_bpb, rtol=0, atol=1e-12), "endpoints differ from the audit"
    launch = _join(fit.run_name, tables.read_csv(paths.require(LAUNCH_160)), "run_name", "160M launch manifest")
    assert (launch.num_train_steps == SETUP[SMALL]["train_steps"]).all()

    return _rows(
        ctx,
        run_name=hashed,
        label=fit.paired_run_name,  # design names: run_00125, baseline_<method>
        fit=True,
        seeds=launch[["data_seed"]].assign(trainer_seed=launch.trainer_seed.fillna(DEFAULT_TRAINER_SEED)),
        weights=_single_phase_weights(fit, ctx),
        uncheatable=_uncheatable(unch, ctx),
        obe=obe,
        recorded_uncheatable=fit.uncheatable_bpb,
        recorded_obe_easy=fit.table9_macro_bpb,
    )


def _deletions_160(ctx: ExportContext, native: pd.DataFrame, proportional: pd.Series) -> pd.DataFrame:
    """The 39 single-bucket deletions. No launch manifest of theirs is among the sources, so their seeds stay null."""
    audit = tables.read_csv(paths.require(AUDIT_160 / "heldout_observations.csv"))
    deletions = audit[audit.run_name.str.startswith("p60_del_")].reset_index(drop=True)
    assert len(deletions) == 39 and (deletions.intervention_type == "domain_deletion").all()
    hashed = deletions.checkpoint_root.map(run_names.training_run_name_from_uri)  # `p60_del_00-1035b3`
    weights = _single_phase_weights(deletions, ctx)
    _check_deletions(weights, deletions.target_domain.map(naming.bucket_public), proportional)

    evaluations = _join(deletions.run_name, native, "run_name", "160M deletion evaluations")
    obe = _native_obe(evaluations, ctx)
    assert np.allclose(obe.mean(axis=1), deletions.table9_macro_bpb, rtol=0, atol=1e-12), "components differ"
    frozen = tables.read_csv(paths.require(FROZEN_COMPONENTS))
    frozen = frozen[frozen.target == "uncheatable"].pivot(index="row_id", columns="component", values="bpb")
    unch = _join("60m::" + deletions.observation_id, frozen.reset_index(), "row_id", "160M deletion Uncheatable")

    return _rows(
        ctx,
        run_name=hashed,
        label="pctrl_del_" + deletions.target_domain.str.replace("/", "_"),  # the 200M swarm's deletion labels
        fit=True,
        seeds=pd.DataFrame({"data_seed": np.nan, "trainer_seed": np.nan}, index=deletions.index),
        weights=weights,
        uncheatable=_uncheatable(unch, ctx),
        obe=obe,
        recorded_uncheatable=deletions.uncheatable_bpb,
        recorded_obe_easy=deletions.table9_macro_bpb,
    )


def _repeats_160(ctx: ExportContext, native: pd.DataFrame, proportional: pd.Series) -> pd.DataFrame:
    """The ten proportional repeats, independently trained; the audit's other repeat rows mirror fit runs or repeat
    other mixtures. Their launch set trainer seeds and no data seed, so data order follows the trainer seed."""
    audit = tables.read_csv(paths.require(AUDIT_160 / "repeat_observations.csv"))
    repeats = audit[(audit.source_kind == "proportional_noise") & ~audit.is_fit_source_alias].reset_index(drop=True)
    assert len(repeats) == 10
    hashed = repeats.checkpoint_root.map(run_names.training_run_name_from_uri)
    weights = _single_phase_weights(repeats, ctx)
    assert np.allclose(weights, proportional[weights.columns], rtol=0, atol=1e-12), "repeats train the proportional mix"

    evaluations = _join(repeats.run_name, native, "run_name", "160M repeat evaluations")
    obe = _native_obe(evaluations, ctx)
    assert np.allclose(obe.mean(axis=1), repeats.table9_macro_bpb, rtol=0, atol=1e-12), "components differ"
    endpoints = tables.read_csv(paths.require(REPEATS / "noise_baseline_proportional_variable_subset_60m_1p2b.csv"))
    endpoints = _join(hashed, endpoints, "wandb_run_id", "160M repeat endpoints")
    assert np.allclose(endpoints[UNCH_POOLED], repeats.uncheatable_bpb, rtol=0, atol=1e-12), "endpoints differ"

    return _rows(
        ctx,
        run_name=hashed,
        label="proportional_repeat_" + endpoints.trainer_seed.astype("Int64").astype(str),
        fit=False,
        seeds=endpoints[["data_seed", "trainer_seed"]],
        weights=weights,
        uncheatable=_uncheatable(endpoints, ctx),
        obe=obe,
        recorded_uncheatable=repeats.uncheatable_bpb,
        recorded_obe_easy=repeats.table9_macro_bpb,
    )


def _export_200(ctx: ExportContext) -> pd.DataFrame:
    fit = _fit_200(ctx)
    table = pd.concat([fit, _repeats_200(ctx, _proportional(fit, ctx))], ignore_index=True)
    return _finish(table, LARGE, ctx)


def _fit_200(ctx: ExportContext) -> pd.DataFrame:
    panel = tables.read_csv(paths.require(PARITY_200 / "one_phase_augmented_fit_panel.csv"))
    assert len(panel) == 280 and panel.run_name.is_unique
    trained = panel.panel_source == "single_phase_qsplit_signal"
    alias = panel.is_shared_checkpoint_alias  # the stratified baseline, which is also a two-phase swarm row
    assert trained.sum() == 240 and alias.sum() == 1 and not (trained & alias).any()
    deletion = panel.panel_source == "domain_deletion"
    # The packet and the component file prefix the deletions' names (`singleavg_pctrl_del_<bucket>`).
    row_id = "singleavg_" + panel.run_name.str.removeprefix("singleavg_")

    unch = tables.read_csv(paths.require(UNCHEATABLE / "300m_uncheatable_components.csv"))
    unch = _join(row_id, unch, "row_id", "200M Uncheatable")
    hashed = unch.wandb_run_id  # `singleavg_run_00002-f84c49`, `pctrl_del_dolma3_arxiv-079b36`
    physical = panel.shared_checkpoint_run_name.fillna(panel.run_name)
    assert (hashed.str.replace(r"-[0-9a-f]{6}$", "", regex=True) == physical).all(), "run ids must name the panel's runs"
    weights = _single_phase_weights(panel, ctx)
    proportional = weights[panel.run_name == PROPORTIONAL_200].iloc[0]  # run names are unique
    _check_deletions(weights[deletion], panel.source_run_name[deletion].map(naming.bucket_public), proportional)

    obe_columns = [c for c in panel if c.startswith(OBE_200)]
    panel_obe = _metrics(panel, obe_columns, ctx.tasks[7:])
    native = tables.read_csv(paths.require(PARITY_200 / "single_phase_table9_wide.csv"))
    native = _join(panel.run_name[trained], native, "run_name", "200M native OlmoBaseEval Easy")
    native_obe = _metrics(native, obe_columns, ctx.tasks[7:])
    replaced = (panel_obe[trained] - native_obe).abs().max(axis=1) > 1e-12
    assert panel.run_name[trained][replaced].tolist() == [PROPORTIONAL_200], "the panel replaced other runs' values"
    references = tables.read_csv(paths.require(PARITY_200 / "proportional_reference_table9.csv"))
    replaced_macro = panel.table9_macro_bpb[panel.run_name == PROPORTIONAL_200]
    assert len(references) == 11 and np.allclose(replaced_macro, references.table9_macro_bpb.mean(), rtol=0, atol=1e-12)
    # Native evaluations: the wide file's for the 240 trained runs, the panel's for the 40 reused ones.
    obe = pd.concat([native_obe, panel_obe[~trained]]).loc[panel.index]
    recorded_obe = pd.concat([native.native_table9_macro_bpb, panel.table9_macro_bpb[~trained]]).loc[panel.index]

    packet = _join(row_id, tables.read_csv(paths.require(PACKET / "300m_one_phase_fit.csv")), "row_id", "200M packet")
    own = panel.run_name != PROPORTIONAL_200
    assert np.allclose(packet.uncheatable_bpb[own], unch[UNCH_POOLED][own], rtol=0, atol=1e-12)
    assert np.allclose(packet.table9_macro_bpb[own], recorded_obe[own], rtol=0, atol=1e-12)

    # The stratified baseline's two-phase launch left no manifest in the sources, so its seeds stay null.
    launches = pd.concat([tables.read_csv(paths.require(LAUNCH_200)), tables.read_csv(paths.require(DELETIONS_200))])
    launch = _join(panel.run_name[~alias], launches, "run_name", "200M launch manifests")
    assert (launch.num_train_steps == SETUP[LARGE]["train_steps"]).all()
    seeds = launch[["data_seed"]].assign(trainer_seed=launch.trainer_seed.fillna(DEFAULT_TRAINER_SEED))

    return _rows(
        ctx,
        run_name=hashed,
        label=panel.run_name.str.removeprefix("singleavg_"),
        fit=True,
        seeds=seeds.reindex(panel.index),
        weights=weights,
        uncheatable=_uncheatable(unch, ctx),
        obe=obe,
        recorded_uncheatable=unch[UNCH_POOLED],
        recorded_obe_easy=recorded_obe,
    )


def _repeats_200(ctx: ExportContext, proportional: pd.Series) -> pd.DataFrame:
    """The ten proportional repeats. The parity panel averaged them with the two-phase swarm's proportional run
    (`panel == "parity"`), a different run from the fit row; the paper counts the ten repeats only."""
    references = tables.read_csv(paths.require(PARITY_200 / "proportional_reference_table9.csv"))
    repeats = references[references.panel == "proportional_noise"].reset_index(drop=True)
    assert len(repeats) == 10
    endpoints = tables.read_csv(paths.require(REPEATS / "noise_baseline_proportional_variable_subset_300m_6b.csv"))
    endpoints = _join(repeats.run_name, endpoints, "run_name", "200M repeat endpoints")
    weights = _single_phase_weights(endpoints, ctx)
    assert np.allclose(weights, proportional[weights.columns], rtol=0, atol=1e-12), "repeats train the proportional mix"

    return _rows(
        ctx,
        run_name=endpoints.wandb_run_id,  # `propvar_300m_6b_trainer_seed_10000-919fa5`
        label="proportional_repeat_" + endpoints.trainer_seed.astype("Int64").astype(str),
        fit=False,
        seeds=endpoints[["data_seed", "trainer_seed"]],
        weights=weights,
        uncheatable=_uncheatable(endpoints, ctx),
        obe=_metrics(repeats, [c for c in repeats if c.startswith(OBE_200)], ctx.tasks[7:]),
        recorded_uncheatable=endpoints[UNCH_POOLED],
        recorded_obe_easy=repeats.table9_macro_bpb,
    )
