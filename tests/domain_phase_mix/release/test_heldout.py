# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.release import context, naming, paths, tables
from experiments.domain_phase_mix.release.configs import heldout, validation
from experiments.domain_phase_mix.release.run_names import training_run_name_from_uri

SRC = paths.RO / "single_phase_heldout_benchmark_expanded_20260913"
pytestmark = pytest.mark.skipif(not SRC.exists(), reason="held-out registry absent")


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    return context.build(private_dir=tmp_path_factory.mktemp("private"))


@pytest.fixture(scope="module")
def table(ctx):
    return heldout.export(ctx)["heldout"]


def test_cohorts_and_swarms(table):
    assert len(table) == 929 and table.observation_id.is_unique
    assert table.swarm.value_counts().to_dict() == {"qwen3_360m_1p6b": 621, "llama_160m_1p2b": 174, "llama_200m_6b": 134}
    qwen = table[table.swarm == "qwen3_360m_1p6b"]
    assert qwen.cohort.value_counts().to_dict() == {"frozen": 505, "expanded": 116}
    assert (table[table.swarm != "qwen3_360m_1p6b"].cohort == "frozen").all()


def test_expanded_rows_key_on_training_run_names(ctx, table):
    # validation_3e18 keys these runs on the hashed training run names in their eval_metrics_uri. The table keeps the
    # registry's row order.
    runs = pd.read_csv(SRC / "heldout_runs.csv")
    expanded = runs[table.cohort == "expanded"]
    names = {}
    for source, group in expanded.groupby("source"):
        folder = heldout.VALIDATION_LAUNCHES[source.removeprefix("validation::")]
        results = pd.read_csv(paths.RO / folder / "measured_results.csv")
        results = results[results.status == "measured"].rename(columns={"candidate_id": "source_row_id"})
        on = ["source_row_id", "trainer_seed"] if "trainer_seed" in results else ["source_row_id"]
        uris = group[on].merge(results[[*on, "eval_metrics_uri"]], on=on, how="left", validate="one_to_one")
        names.update(zip(group.index, map(training_run_name_from_uri, uris.eval_metrics_uri), strict=True))
    expected = pd.Series({i: ctx.registry.id_for(name) for i, name in names.items()})
    assert len(expected) == 116 and (table.observation_id[expected.index] == expected).all()
    assert not table.observation_id[expanded.index].isin([ctx.registry.id_for(r) for r in expanded.row_id]).any()


def test_frozen_rows_key_on_recorded_run_names(ctx, table):
    # The registry records the W&B training run id (the run name with its hash) for every frozen run except the 200M
    # ones; `delphi_archive::` row ids wrap that name.
    runs = pd.read_csv(SRC / "heldout_runs.csv")
    archive = runs.row_id.str.startswith("delphi_archive::")
    stripped = runs.row_id[archive].str.removeprefix("delphi_archive::").map(ctx.registry.id_for)
    assert archive.sum() == 187 and (table.observation_id[archive] == stripped).all()
    recorded = runs.training_wandb_run_id.notna()
    assert recorded.sum() == 679 and (table.cohort[recorded] == "frozen").all()
    assert (table.observation_id[recorded] == runs.training_wandb_run_id[recorded].map(ctx.registry.id_for)).all()
    unnamed = (table.cohort == "frozen") & ~recorded
    assert (table.swarm[unnamed] == "llama_200m_6b").all() and unnamed.sum() == 134
    assert (table.observation_id[unnamed] == runs.row_id[unnamed].map(ctx.registry.id_for)).all()


def test_runs_shared_with_validation_have_its_seeds(ctx, table):
    # The registry records seeds for 42 of the 116 expanded runs. Like validation_3e18, heldout reads each validation
    # run's seeds from the directory that holds it, and the UniMax-8 seed-0 run's (a frozen row) from its launch record.
    expanded = table[table.cohort == "expanded"]
    assert len(expanded) == 116 and expanded[["data_seed", "trainer_seed"]].notna().all().all()
    validation_table = validation.export(ctx)["validation_3e18"]
    shared = table.merge(validation_table, on="observation_id", suffixes=("", "_validation"), validate="one_to_one")
    assert len(shared) == 117
    for seed in ["data_seed", "trainer_seed"]:
        assert (shared[seed] == shared[f"{seed}_validation"]).all(), seed


def test_exposure_ids_split_what_mixture_ids_merge(table):
    frozen = table[(table.swarm == "qwen3_360m_1p6b") & (table.cohort == "frozen")]
    assert frozen.mixture_id.nunique() == 410
    assert frozen.exposure_id.nunique() == 425


def test_labels_are_kept(table):
    assert table.proposal_method.notna().all()
    # Objectives take their public names (the registry's `table9`, `table9_macro_bpb`, `uncheatable_and_table9` and
    # `baseline` among them); the Llama panels record none.
    qwen = table.swarm == "qwen3_360m_1p6b"
    assert table.proposal_objective[~qwen].isna().all()
    assert set(table.proposal_objective[qwen]) == {
        "obe_easy",
        "uncheatable",
        "uncheatable_worsened",
        "uncheatable_and_obe_easy",
        "none",
        "other",
    }
    # The registry's one baseline is the UniMax-8 construction's 3e18 ladder run, labeled as validation_3e18 labels
    # it: no objective and the construction's epoch cap of 8.
    baseline = table[table.proposal_objective == "none"]
    assert len(baseline) == 1 and baseline.epoch_cap.tolist() == [8.0]


def test_aggregates_recompute_the_recorded_values(table):
    # The registry records each run's aggregates, row for row with the table. The Qwen3 panel recorded Uncheatable
    # under the fixed rule, except the path-midpoint and RegMix-reference launches, which recorded the BPB-schema-2
    # pooled aggregate, about 0.01 BPB above the fixed weighting of the same components; the Llama panels recorded it
    # under the legacy weights, within 1e-4 of the fixed rule. OlmoBaseEval Easy is a plain mean in every panel, so it
    # checks each swarm's component join.
    runs = tables.read_csv(SRC / "heldout_runs.csv")
    complete = {
        objective: table[[f"bpb__{t}" for t in weights]].notna().all(axis=1)
        for objective, weights in naming.task_weights().items()
    }
    pooled = runs.source.isin(heldout.POOLED_SOURCES)
    assert pooled.sum() == 14 and set(table.proposal_method[pooled]) == {"path_midpoints", "regmix_reference"}
    assert complete["uncheatable"][pooled].all()
    assert (runs.uncheatable_bpb - table.bpb__uncheatable)[pooled].between(0.005, 0.013).all()
    qwen = table.swarm == "qwen3_360m_1p6b"
    fixed = qwen & ~pooled & runs.uncheatable_bpb.notna()
    assert np.allclose(table.bpb__uncheatable[fixed], runs.uncheatable_bpb[fixed], rtol=0, atol=2e-6)
    legacy = ~qwen & complete["uncheatable"] & runs.uncheatable_bpb.notna()
    assert (table.bpb__uncheatable - runs.uncheatable_bpb)[legacy].abs().max() < 1e-4
    obe = runs.table9_macro_bpb.notna()
    assert np.allclose(table.bpb__obe_easy[obe], runs.table9_macro_bpb[obe], rtol=0, atol=1e-12)
    # Without all of an objective's components, a run takes its fixed-rule record or has no value; one Llama 160M run
    # records Uncheatable under the legacy weights only and has none.
    assert table.bpb__uncheatable.notna().equals(complete["uncheatable"] | fixed)
    assert table.bpb__obe_easy.notna().equals(complete["obe_easy"] | obe)
    assert (~complete["uncheatable"] & ~fixed & runs.uncheatable_bpb.notna()).sum() == 1


def test_weights_and_pool_fractions(table):
    w = table[[f"w__{b}" for b in naming.BUCKETS]]
    assert np.allclose(w.sum(axis=1), 1.0, rtol=0, atol=1e-9)
    pf = table[[f"pf__{b}" for b in naming.BUCKETS]]
    assert ((pf > 0) & (pf <= 1)).all().all()
    assert (pf < 1).any().any()  # the a-priori runs vary pool fractions


def test_no_internal_strings(table):
    text = table.to_csv(index=False)
    for needle in ["gs://", "wandb", "pinlin_calvin_xu", "/Users/", "http"]:
        assert needle not in text
    # Fieldbook ids are `exp_`/`run_` plus a 26-character ULID; public exposure ids are `exp_` plus 16 hex digits.
    assert not re.search(r"(?:exp|run)_01[0-9a-hjkmnp-tv-z]{24}", text)
    assert not re.search(r"[0-9a-f]{64}", text)  # registry coordinate ids
    assert not re.search(r"\w-[0-9a-f]{6}(?![0-9a-z])", text)  # run names end in a six-hex-digit hash
