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
# The launch records of the 134 Llama 200M held-out runs, which name each run and its data seed. The perturbation and
# gradient-step launches trained each run at two scales.
LLAMA_200M_LAUNCHES = [
    paths.RO / f"{folder}/training_manifest.csv"
    for folder in (
        "proportional_controllability_300m_20260520",
        "proportional_perturbation_scale_transfer_20260507",
        "proportional_gradient_step_validation_20260510",
    )
]
LLAMA_200M_SCALE = "300m_6b"


@pytest.fixture(scope="module")
def ctx():
    return context.build()


@pytest.fixture(scope="module")
def table(ctx):
    return heldout.export(ctx)["heldout"]


def test_cohorts_and_swarms(table):
    assert len(table) == 929 and table.observation_id.is_unique
    assert table.swarm.value_counts().to_dict() == {"qwen3_360m_1p6b": 621, "llama_160m_1p2b": 174, "llama_200m_6b": 134}
    qwen = table[table.swarm == "qwen3_360m_1p6b"]
    assert qwen.cohort.value_counts().to_dict() == {"frozen": 505, "expanded": 116}
    assert (table[table.swarm != "qwen3_360m_1p6b"].cohort == "frozen").all()


def test_expanded_rows_key_on_training_run_names(table):
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
    expected = pd.Series(names)
    assert len(expected) == 116 and (table.observation_id[expected.index] == expected).all()


def test_frozen_rows_key_on_recorded_run_names(table):
    # The registry records the W&B training run id (the run name with its hash) for every frozen run except the 200M
    # ones; `delphi_archive::` row ids wrap that name.
    runs = pd.read_csv(SRC / "heldout_runs.csv")
    archive = runs.row_id.str.startswith("delphi_archive::")
    stripped = runs.row_id[archive].str.removeprefix("delphi_archive::")
    assert archive.sum() == 187 and (table.observation_id[archive] == stripped).all()
    recorded = runs.training_wandb_run_id.notna()
    assert recorded.sum() == 679 and (table.cohort[recorded] == "frozen").all()
    assert (table.observation_id[recorded] == runs.training_wandb_run_id[recorded]).all()


def test_llama_200m_runs_take_their_w_and_b_ids_and_seeds(table):
    # The registry records neither the W&B run id nor the seeds of its 134 Llama 200M runs; both come from W&B. A W&B
    # run id is the run's name, which W&B shortened from the front for 20 runs, and a six-hex-digit hash. Each run's
    # launch record gives its data seed; the launches left the trainer seed at Marin's default, 0.
    runs = pd.read_csv(SRC / "heldout_runs.csv")
    llama = runs.panel == "300m_39bucket"
    assert llama.sum() == 134 and runs.training_wandb_run_id[llama].isna().all()
    rows = table[llama].set_axis(runs.row_id[llama].str.removeprefix("300m::"))
    kept = rows.observation_id.str.extract(r"^(?P<name>.+)-[0-9a-f]{6}$")["name"]
    assert kept.notna().all() and all(name.endswith(k) for name, k in zip(rows.index, kept, strict=True))
    assert (kept != rows.index).sum() == 20 and rows.observation_id.is_unique
    launches = pd.concat([pd.read_csv(path) for path in LLAMA_200M_LAUNCHES])
    launches = launches[launches.scale == LLAMA_200M_SCALE].set_index("run_name")
    assert (rows.data_seed == launches.data_seed.reindex(rows.index)).all()
    assert launches.trainer_seed.reindex(rows.index).isna().all() and (rows.trainer_seed == 0).all()


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
    # under the legacy weights, within 1e-4 of the fixed rule for the registry's components, and no row copies a legacy
    # record. The Llama 160M run whose components come from W&B pooled its record 3.6e-4 away from the fixed rule; its
    # own test checks its pull. OlmoBaseEval Easy is a plain mean in every panel, so it checks each swarm's component
    # join.
    runs = tables.read_csv(SRC / "heldout_runs.csv")
    pulls = pd.concat([tables.read_csv(path, comment="#") for path in heldout.WANDB_PULLS])
    pulled = table.observation_id.isin(pulls.training_wandb_run_id)
    complete = {
        objective: table[[f"bpb__{t}" for t in weights]].notna().all(axis=1)
        for objective, weights in naming.task_weights().items()
    }
    pooled = runs.source.isin(heldout.POOLED_SOURCES)
    assert pooled.sum() == 14 and set(table.proposal_method[pooled]) == {"path_midpoints", "regmix_reference"}
    assert complete["uncheatable"][pooled].all()
    assert (runs.uncheatable_bpb - table.bpb__uncheatable)[pooled].between(0.008, 0.0115).all()
    qwen = table.swarm == "qwen3_360m_1p6b"
    fixed = qwen & ~pooled & runs.uncheatable_bpb.notna()
    assert np.allclose(
        table.bpb__uncheatable[fixed], runs.uncheatable_bpb[fixed], rtol=0, atol=tables.FIXED_RULE_TOLERANCE
    )
    legacy = ~qwen & ~pulled & complete["uncheatable"] & runs.uncheatable_bpb.notna()
    gap = (table.bpb__uncheatable - runs.uncheatable_bpb)[legacy].abs()
    assert gap.max() < 1e-4 and (gap > 0).all()
    obe = runs.table9_macro_bpb.notna()
    assert np.allclose(table.bpb__obe_easy[obe], runs.table9_macro_bpb[obe], rtol=0, atol=1e-12)
    # Without all of an objective's components, a run takes its fixed-rule record or has no value. No run is left with
    # only a record under other weights.
    assert table.bpb__uncheatable.notna().equals(complete["uncheatable"] | fixed)
    assert table.bpb__obe_easy.notna().equals(complete["obe_easy"] | obe)
    assert not (~complete["uncheatable"] & ~fixed & runs.uncheatable_bpb.notna()).any()


def test_wandb_pull_completes_the_run_the_registry_scores_by_its_pooled_aggregate(table):
    # One archived Llama 160M run has only its pooled Uncheatable aggregate in the registry. Its seven components and
    # data seed come from its W&B summary and config, frozen in the repository; it logged no OlmoBaseEval Easy metrics.
    pull = tables.read_csv(heldout.WANDB_PULLS[0], comment="#")
    (name,) = pull.training_wandb_run_id.unique()
    values = pull.set_index("metric").value
    run = table.set_index("observation_id").loc[name]
    weights = naming.task_weights()["uncheatable"]
    components = {t: values[f"eval/uncheatable_eval/{t.removeprefix('uncheatable/')}/bpb"] for t in weights}
    assert all(run[f"bpb__{t}"] == value for t, value in components.items())
    fixed = sum(value * weights[t] for t, value in components.items())
    assert abs(run.bpb__uncheatable - fixed) <= tables.FIXED_RULE_TOLERANCE
    assert run[[f"bpb__{t}" for t in naming.TASKS[7:]] + ["bpb__obe_easy"]].isna().all()
    assert run.data_seed == 245 and pd.isna(run.trainer_seed) and not run.components_available


def test_llama_200m_runs_without_registry_uncheatable_values_aggregate_their_pulled_components(table):
    # The registry records no Uncheatable component or aggregate for 17 Llama 200M runs. Their seven components come
    # from their W&B summaries, and bpb__uncheatable is the fixed byte-share sum of those components, never the pooled
    # aggregate that W&B also logged, which weights the corpora otherwise (1.5e-5 to 7.2e-5 BPB apart for these runs).
    runs = tables.read_csv(SRC / "heldout_runs.csv")
    without = (runs.panel == "300m_39bucket") & runs.uncheatable_bpb.isna()
    pull = tables.read_csv(heldout.WANDB_PULLS[1], comment="#").pivot(
        index="training_wandb_run_id", columns="metric", values="value"
    )
    assert without.sum() == 17 and set(table.observation_id[without]) == set(pull.index)
    rows = table[without].set_index("observation_id").loc[pull.index]
    weights = naming.task_weights()["uncheatable"]
    components = pull[[f"eval/uncheatable_eval/{t.removeprefix('uncheatable/')}/bpb" for t in weights]]
    assert (rows[[f"bpb__{t}" for t in weights]].to_numpy() == components.to_numpy()).all()
    fixed = components.to_numpy() @ np.array(list(weights.values()))
    assert np.allclose(rows.bpb__uncheatable, fixed, rtol=0, atol=1e-12)
    assert (rows.bpb__uncheatable - pull["eval/uncheatable_eval/bpb"]).abs().min() > 1e-6
    assert rows.components_available.all() and (rows.data_seed == pull.data_seed).all()


def test_weights_and_pool_fractions(table):
    w = table[[f"w__{b}" for b in naming.BUCKETS]]
    assert np.allclose(w.sum(axis=1), 1.0, rtol=0, atol=1e-9)
    pf = table[[f"pf__{b}" for b in naming.BUCKETS]]
    assert ((pf > 0) & (pf <= 1)).all().all()
    assert (pf < 1).any().any()  # the a-priori runs vary pool fractions


def test_no_internal_strings(table):
    text = table.to_csv(index=False)
    for needle in ["gs://", "pinlin_calvin_xu", "/Users/", "http"]:
        assert needle not in text
    # Fieldbook ids are `exp_`/`run_` plus a 26-character ULID; public exposure ids are `exp_` plus 16 hex digits.
    assert not re.search(r"(?:exp|run)_01[0-9a-hjkmnp-tv-z]{24}", text)
    assert not re.search(r"[0-9a-f]{64}", text)  # registry coordinate ids
