# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.release import context, naming, paths
from experiments.domain_phase_mix.release.configs import swarm_qwen3

pytestmark = pytest.mark.skipif(not paths.MIXSEL.exists(), reason="mixture-selection checkout absent")
BUNDLE_RUN_PREFIX = r"^singleavg_fit_\d{3}_"  # the bundle's `run` is `singleavg_fit_<NNN>_<design name>`


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    ctx = context.build(private_dir=tmp_path_factory.mktemp("private"))
    return swarm_qwen3.export(ctx)["swarm_qwen3_360m_1p6b"]


def test_counts_and_fit_inclusion(table):
    assert len(table) == 290
    assert table.fit_inclusion.sum() == 280
    assert table.design_kind.value_counts().to_dict() == {"sampled": 238, "deletion": 39, "baseline": 3, "repeat": 10}
    assert table.observation_id.is_unique


def test_fit_rows_equal_mixture_selection_bundle(table):
    w = pd.read_csv(paths.MIXSEL / "data/swarm_weights.csv").set_index("run")
    w.index = w.index.str.replace(BUNDLE_RUN_PREFIX, "", regex=True)
    fit = table[table.fit_inclusion].set_index("run_label")
    for internal in w.columns.drop("calibration"):
        assert np.allclose(fit[f"w__{naming.bucket_public(internal)}"].reindex(w.index), w[internal], atol=1e-12)
    y = pd.read_csv(paths.MIXSEL / "data/swarm_outcomes.csv").set_index("run")
    y.index = y.index.str.replace(BUNDLE_RUN_PREFIX, "", regex=True)
    for metric in y.columns:
        assert np.allclose(fit[f"bpb__{naming.task_public(metric)}"].reindex(y.index), y[metric], atol=1e-12)


def test_weights_pool_fractions_and_keys(table):
    w = table[[c for c in table if c.startswith("w__")]]
    assert w.shape[1] == 39 and np.allclose(w.sum(axis=1), 1.0, atol=1e-9)
    pf = table[[c for c in table if c.startswith("pf__")]]
    assert pf.shape[1] == 39 and (pf == 1.0).all().all()
    repeats = table[table.design_kind == "repeat"]
    assert repeats.mixture_id.nunique() == 1  # all ten are the proportional mixture
    proportional = table[(table.design_kind == "baseline") & (table.run_label == "baseline_proportional")]
    assert repeats.mixture_id.iloc[0] == proportional.mixture_id.iloc[0]
    assert table[table.design_kind != "repeat"].mixture_id.is_unique


def test_repeats_pair_obe_and_uncheatable_rows_by_run_letter(table):
    # The sources index OBE Easy rows by letter and Uncheatable rows by run name, in different orders; the noise panel
    # records both pooled values per letter. Fixed byte shares stay within 1e-7 of the pooled Uncheatable BPB, and the
    # letters' pooled values lie at least 1.2e-5 apart, so a row paired with another letter's run fails.
    panel = pd.read_csv(swarm_qwen3.NOISE / "noise_panel.csv", index_col="letter")
    repeats = table[table.design_kind == "repeat"]
    letters = repeats.run_label.str.removeprefix("proportional_repeat_")
    obe = panel.table9_macro_bpb.reindex(letters).to_numpy()
    assert np.allclose(repeats.bpb__obe_easy.to_numpy(), obe, rtol=0, atol=1e-12)
    uncheatable = panel.uncheatable_bpb.reindex(letters).to_numpy()
    assert np.allclose(repeats.bpb__uncheatable_fixed.to_numpy(), uncheatable, rtol=0, atol=1e-6)


def test_aggregates_recomputed_with_fixed_weights(table):
    weights = naming.task_weights()["uncheatable_fixed"]
    manual = sum(table[f"bpb__{t}"] * w for t, w in weights.items())
    assert np.allclose(table.bpb__uncheatable_fixed, manual, atol=1e-12)
    assert np.allclose(table.bpb__obe_easy, table[[f"bpb__{t}" for t in naming.TASKS[7:]]].mean(axis=1), atol=1e-12)
    assert (table.aggregate_version == "fixed_20260927").all()


def test_no_internal_strings(table):
    text = table.to_csv(index=False)
    for needle in ["gs://", "wandb", "pinlin_calvin_xu", "/Users/"]:
        assert needle not in text
    # Fieldbook ids are `exp_`/`run_` plus a 26-character ULID; public exposure ids are `exp_` plus 16 hex digits.
    assert not re.search(r"(?:exp|run)_01[0-9a-hjkmnp-tv-z]{24}", text)
    assert not table.run_label.str.contains(r"-[0-9a-f]{6}$").any()
