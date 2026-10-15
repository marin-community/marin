# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.release import context, naming, paths, run_names, tables
from experiments.domain_phase_mix.release.configs import heldout, swarm_llama

pytestmark = pytest.mark.skipif(
    not (paths.RO / "60m_39bucket_checkpoint_audit_20260724").exists(), reason="archived swarm sources absent"
)
APPENDIX = paths.DRIVE / "sections/appendix.tex"
# Exports the 200M swarm's builder wrote independently of the sources the exporter reads.
SCORES_200 = paths.RO / "one_phase_swarm_scores_export_300m_20260630"
PAPER_COLUMN = {"swarm_llama_160m_1p2b": 2, "swarm_llama_200m_6b": 3}  # columns of Table tab:a-hyperparameters
PAPER_ROWS = {
    "Total parameters": "total_parameters",
    "Non-embedding parameters": "non_embedding_parameters",
    "Training tokens": "train_tokens",
    "FLOPs per run": "train_flops",
}


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    return context.build(private_dir=tmp_path_factory.mktemp("private"))


@pytest.fixture(scope="module")
def out(ctx):
    return swarm_llama.export(ctx)


def _cells(appendix: str, row: str) -> list[str]:
    """The cells of one row of the hyperparameter table, label first."""
    line = next(line for line in appendix.splitlines() if line.startswith(f"{row} &"))
    return [cell.strip() for cell in line.rstrip().removesuffix("\\\\").split("&")]


def _rounds_to(value: float, cell: str) -> bool:
    """Whether `value` prints as a cell such as `157.5M`, `59M`, `6.0B` or `$1.1\\times10^{18}$`."""
    m = re.fullmatch(r"\$?(?P<digits>[\d.]+)(?:(?P<unit>[MB])|\\times10\^\{(?P<exponent>\d+)\}\$)", cell)
    assert m, f"unparsed cell {cell!r}"
    scale = {"M": 1e6, "B": 1e9}[m["unit"]] if m["unit"] else 10.0 ** int(m["exponent"])
    decimals = len(m["digits"].partition(".")[2])
    return abs(value - float(m["digits"]) * scale) <= 0.5 * 10.0**-decimals * scale


def test_160m_rows_and_components(out):
    t = out["swarm_llama_160m_1p2b"]
    assert len(t) == 281 + 10 and t.observation_id.is_unique
    assert t.fit_inclusion.sum() == 281 and t[t.fit_inclusion].mixture_id.is_unique
    assert t.components_available.all()
    assert t.design_kind.value_counts().to_dict() == {"sampled": 238, "deletion": 39, "repeat": 10, "baseline": 4}
    assert (t.fit_inclusion == (t.design_kind != "repeat")).all()
    # Seeds come from the launch manifests; none of the deletions' launch is among the sources.
    launched = t[t.design_kind.isin(["sampled", "baseline"])]
    assert launched.data_seed.notna().all() and (launched.trainer_seed == 0).all()
    assert t.loc[t.design_kind == "deletion", ["data_seed", "trainer_seed"]].isna().all().all()


def test_200m_rows_and_components(out):
    t = out["swarm_llama_200m_6b"]
    assert len(t) == 280 + 10 and t.observation_id.is_unique
    # 240 runs trained for the single-phase swarm, plus the 39 deletions and the stratified baseline, whose native
    # evaluations the parity panel carries.
    assert t.fit_inclusion.sum() == 280 and t.components_available.all()
    assert t.design_kind.value_counts().to_dict() == {"sampled": 238, "deletion": 39, "repeat": 10, "baseline": 3}
    assert (t.fit_inclusion == (t.design_kind != "repeat")).all()
    # The July panel replaced the proportional run's targets by the mean of 11 proportional runs; the table keeps the
    # run's own evaluation.
    native = pd.read_csv(swarm_llama.PARITY_200 / "single_phase_table9_wide.csv").set_index("run_name")
    proportional = t[t.run_label == "baseline_proportional"]
    expected = native.native_table9_macro_bpb["singleavg_baseline_proportional"]
    assert np.allclose(proportional.bpb__obe_easy, expected, rtol=0, atol=1e-12)
    # The stratified baseline is a two-phase swarm run whose launch left no manifest in the sources.
    assert t.run_label[t.trainer_seed.isna()].tolist() == ["baseline_stratified"]


@pytest.mark.skipif(not APPENDIX.exists(), reason="paper folder absent")
def test_setup_columns_match_the_paper(out):
    appendix = APPENDIX.read_text()
    for name, t in out.items():
        assert t.model_family.iloc[0] == "llama"
        assert (
            t[["total_parameters", "non_embedding_parameters", "train_tokens", "train_flops", "train_steps"]]
            .nunique()
            .eq(1)
            .all()
        )
        assert f"{t.train_tokens.iloc[0] / 1e9:.1f}B" in appendix or f"{t.train_tokens.iloc[0] / 1e9:.2f}B" in appendix
        column = PAPER_COLUMN[name]
        for row, field in PAPER_ROWS.items():
            cell = _cells(appendix, row)[column]
            assert _rounds_to(t[field].iloc[0], cell), f"{name}: {field} = {t[field].iloc[0]}, the paper prints {cell}"
        batch = int(_cells(appendix, "Batch (sequences)")[column])
        context_length = int(_cells(appendix, "Context length")[column].replace(",", ""))
        assert t.train_steps.iloc[0] * batch * context_length == t.train_tokens.iloc[0]
        assert int(_cells(appendix, "Fit swarm (runs)")[column]) == t.fit_inclusion.sum()
        assert int(_cells(appendix, "Proportional repeats")[column]) == (t.design_kind == "repeat").sum()


def test_weights_sum_to_one(out):
    for t in out.values():
        assert np.allclose(t[[f"w__{b}" for b in naming.BUCKETS]].sum(axis=1), 1.0, rtol=0, atol=1e-9)


def _recorded_aggregates(ctx) -> pd.DataFrame:
    """Each run's aggregates as its sources record them, indexed by observation id: the 160M audit's runs, and the 200M
    builder's exports of its fit panel and repeats. The fit panel's proportional row records the mean of 11 runs, so it
    is left out."""
    audit = pd.concat(
        [
            tables.read_csv(swarm_llama.AUDIT_160 / name)
            for name in ("fit_single_phase.csv", "heldout_observations.csv", "repeat_observations.csv")
        ]
    )
    small = pd.DataFrame(
        {"uncheatable": audit.uncheatable_bpb.to_numpy(), "obe_easy": audit.table9_macro_bpb.to_numpy()},
        index=audit.checkpoint_root.map(run_names.training_run_name_from_uri).to_numpy(),
    ).drop_duplicates()  # the audit lists some runs in two files
    panel = tables.read_csv(SCORES_200 / "one_phase_augmented_fit_panel_uncheatable_table9_scores_300m.csv")
    panel = panel[~panel.target_uses_proportional_reference_mean]
    references = tables.read_csv(SCORES_200 / "proportional_reference_uncheatable_table9_scores_300m.csv")
    references = references[references.panel == "proportional_noise"]
    large = pd.DataFrame(
        {
            "uncheatable": [*panel.eval_uncheatable_eval_bpb, *references.eval_uncheatable_eval_bpb],
            "obe_easy": [*panel.table9_macro_bpb, *references.table9_macro_bpb],
        },
        index=[*panel.training_wandb_id, *references.wandb_run_id],
    )
    recorded = pd.concat([small, large])
    assert recorded.index.is_unique
    return recorded.set_axis(recorded.index.map(ctx.registry.id_for))


def test_aggregates_recompute_the_recorded_values(ctx, out):
    # The sources record each run's Uncheatable aggregate under the legacy weights, which differ from the fixed rule by
    # at most 2.5e-4 per corpus, so the components reproduce it within 1e-4; a component row paired with another run
    # would not. No row copies its legacy record. OlmoBaseEval Easy is a plain mean in every source.
    recorded = _recorded_aggregates(ctx)
    for name, t in out.items():
        run = recorded.reindex(t.observation_id).set_axis(t.index)
        compared = run.uncheatable.notna()
        left_out = (t.run_label == "baseline_proportional") & (name == "swarm_llama_200m_6b")
        assert compared.equals(~left_out), name
        gap = (t.bpb__uncheatable - run.uncheatable)[compared].abs()
        assert gap.max() < 1e-4 and (gap > 0).all(), name
        assert np.allclose(t.bpb__obe_easy[compared], run.obe_easy[compared], rtol=0, atol=1e-6)


def test_observation_ids_key_on_hashed_training_run_names(ctx, out):
    # Both swarms name runs `singleavg_run_00125` without the hash (241 such names appear in both), so only the training
    # run name with its hash, the checkpoint directory that W&B records as the run id, identifies one run.
    small, large = out["swarm_llama_160m_1p2b"], out["swarm_llama_200m_6b"]
    assert not set(small.observation_id) & set(large.observation_id)
    audit = pd.read_csv(swarm_llama.AUDIT_160 / "fit_single_phase.csv").set_index("paired_run_name")
    hashed = audit.checkpoint_root.str.rsplit("/", n=1).str[-1]  # `singleavg_run_00125-9ca4a6`
    fit = small[small.run_label.isin(hashed.index)]
    assert len(fit) == 242 and (fit.observation_id == fit.run_label.map(hashed).map(ctx.registry.id_for)).all()
    # The held-out table keys the deletions on the run ids its registry records, so each deletion has one id.
    registry = pd.read_csv(heldout.EXPANDED / "heldout_runs.csv")
    deletions = registry[registry.row_id.str.startswith("60m::pctrl:p60_del_")].training_wandb_run_id
    assert set(small.observation_id[small.design_kind == "deletion"]) == set(deletions.map(ctx.registry.id_for))
    # Exports the 200M swarm's builder wrote independently of the sources the exporter reads.
    scores = paths.RO / "one_phase_swarm_scores_export_300m_20260630"
    panel = pd.read_csv(scores / "one_phase_augmented_fit_panel_uncheatable_table9_scores_300m.csv")
    hashed = panel.set_index(panel.run_name.str.removeprefix("singleavg_")).training_wandb_id
    fit = large[large.fit_inclusion]
    assert (fit.observation_id == fit.run_label.map(hashed).map(ctx.registry.id_for)).all()
    references = pd.read_csv(scores / "proportional_reference_uncheatable_table9_scores_300m.csv")
    repeats = references[references.panel == "proportional_noise"].wandb_run_id
    assert set(large.observation_id[large.design_kind == "repeat"]) == set(repeats.map(ctx.registry.id_for))


def test_proportional_repeats(out):
    # Ten runs of the proportional mixture per swarm, outside the fit set as in the Qwen3 table. Their launches set
    # trainer seeds 10000-10009 and no data seed, so data order follows the trainer seed.
    for t in out.values():
        repeats = t[t.design_kind == "repeat"]
        assert len(repeats) == 10 and not repeats.fit_inclusion.any()
        assert set(repeats.mixture_id) == set(t.mixture_id[t.run_label == "baseline_proportional"])
        assert sorted(repeats.trainer_seed) == list(range(10_000, 10_010)) and repeats.data_seed.isna().all()


def test_deletions_remove_one_bucket_from_the_proportional_mixture(out):
    labels = []
    for t in out.values():
        weights = t.set_index("run_label")[[f"w__{b}" for b in naming.BUCKETS]]
        proportional = weights.loc["baseline_proportional"]
        deletions = weights[weights.index.str.startswith("pctrl_del_")]
        assert len(deletions) == 39 and deletions.eq(0).sum(axis=1).eq(1).all() and deletions.eq(0).any().all()
        for _, row in deletions.iterrows():
            removed = row.index[row == 0][0]
            expected = (proportional / (1 - proportional[removed])).mask(proportional.index == removed, 0.0)
            assert np.allclose(row, expected, rtol=0, atol=1e-12)
        labels.append(set(deletions.index))
    assert labels[0] == labels[1]  # both swarms label a deletion by its bucket


def test_no_internal_strings(out):
    for t in out.values():
        text = t.to_csv(index=False)
        for needle in ["gs://", "wandb", "pinlin_calvin_xu", "/Users/", "singleavg"]:
            assert needle not in text
        assert not re.search(r"\w-[0-9a-f]{6}(?![0-9a-z])", text)  # run names end in a six-hex-digit hash
