# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.release import context, naming, paths
from experiments.domain_phase_mix.release.configs import swarm_llama

pytestmark = pytest.mark.skipif(
    not (paths.RO / "60m_39bucket_checkpoint_audit_20260724").exists(), reason="archived swarm sources absent"
)
APPENDIX = paths.DRIVE / "sections/appendix.tex"
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
    assert len(t) == 242 and t.observation_id.is_unique and t.mixture_id.is_unique
    assert t.fit_inclusion.all()
    assert t.components_available.sum() >= 200
    with_components = t[t.components_available]
    assert np.allclose(with_components.bpb__obe_easy, with_components.bpb__obe_easy_source, rtol=0, atol=1e-6)
    assert t.design_kind.value_counts().to_dict() == {"sampled": 238, "baseline": 4}
    assert t.data_seed.notna().all() and (t.trainer_seed == 0).all()  # seeds from the launch manifest


def test_200m_rows_and_components(out):
    t = out["swarm_llama_200m_6b"]
    assert len(t) == 280 and t.observation_id.is_unique
    # 240 runs trained for the single-phase swarm, plus the 39 deletions and the stratified baseline, whose native
    # evaluations the parity panel carries.
    assert t.components_available.sum() == 280
    with_components = t[t.components_available]
    assert np.allclose(with_components.bpb__obe_easy, with_components.bpb__obe_easy_source, rtol=0, atol=1e-6)
    assert t.design_kind.value_counts().to_dict() == {"sampled": 238, "deletion": 39, "baseline": 3}
    # The July panel replaced the proportional run's targets by the mean of 11 proportional runs; the table keeps the
    # run's own evaluation.
    native = pd.read_csv(swarm_llama.PARITY_200 / "single_phase_table9_wide.csv").set_index("run_name")
    proportional = t[t.run_label == "baseline_proportional"]
    expected = native.native_table9_macro_bpb["singleavg_baseline_proportional"]
    assert np.allclose(proportional.bpb__obe_easy, expected, rtol=0, atol=1e-12)
    # The stratified baseline is a two-phase swarm run whose launch left no manifest in the sources.
    assert t.run_label[t.data_seed.isna()].tolist() == ["baseline_stratified"]


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


def test_weights_sum_to_one_and_legacy_aggregates(out):
    for t in out.values():
        assert np.allclose(t[[f"w__{b}" for b in naming.BUCKETS]].sum(axis=1), 1.0, rtol=0, atol=1e-9)
        assert (t.aggregate_version == "legacy").all()
        assert t.bpb__uncheatable_source.notna().all()
        # The legacy Uncheatable weights differ from the fixed rule by at most 2.5e-4 per corpus, so the components
        # reproduce each run's source aggregate within 1e-4; a component row paired with another run would not.
        assert (t.bpb__uncheatable_fixed - t.bpb__uncheatable_source).abs().max() < 1e-4


def test_observation_ids_key_on_hashed_training_run_names(ctx, out):
    # Both swarms name runs `singleavg_run_00125` without the hash (241 such names appear in both), so only the training
    # run name with its hash, the checkpoint directory that W&B records as the run id, identifies one run.
    small, large = out["swarm_llama_160m_1p2b"], out["swarm_llama_200m_6b"]
    assert not set(small.observation_id) & set(large.observation_id)
    audit = pd.read_csv(swarm_llama.AUDIT_160 / "fit_single_phase.csv").set_index("paired_run_name")
    hashed = audit.checkpoint_root.str.rsplit("/", n=1).str[-1]  # `singleavg_run_00125-9ca4a6`
    assert (small.observation_id == small.run_label.map(hashed).map(ctx.registry.id_for)).all()
    # An export the 200M swarm's builder wrote independently of the sources the exporter reads.
    scores = pd.read_csv(
        paths.RO
        / "one_phase_swarm_scores_export_300m_20260630/one_phase_augmented_fit_panel_uncheatable_table9_scores_300m.csv"
    )
    hashed = scores.set_index(scores.run_name.str.removeprefix("singleavg_")).training_wandb_id
    assert (large.observation_id == large.run_label.map(hashed).map(ctx.registry.id_for)).all()


def test_no_internal_strings(out):
    for t in out.values():
        text = t.to_csv(index=False)
        for needle in ["gs://", "wandb", "pinlin_calvin_xu", "/Users/", "singleavg"]:
            assert needle not in text
        assert not re.search(r"\w-[0-9a-f]{6}(?![0-9a-z])", text)  # run names end in a six-hex-digit hash
