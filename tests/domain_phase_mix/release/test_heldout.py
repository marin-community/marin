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
# The four Qwen3 runs whose W&B logging crashed before training ended, and their launch's record of their results.
FINAL_EVALUATION_RUNS = {
    "onephase_dsp_table9_macro_cap04-981094",
    "onephase_dsp_uncheatable_cap06-3b6be6",
    "onephase_dsp_uncheatable_cap08-3f2eac",
    "onephase_dsp_uncheatable_cap10-94f0c7",
}
DSP_EPOCH_CAP_RESULTS = paths.RO / "delphi_one_phase_dsp_epoch_cap_sweep_20260828/measured_results.csv"


@pytest.fixture(scope="module")
def ctx():
    return context.build()


@pytest.fixture(scope="module")
def validation_table(ctx):
    return validation.export(ctx)["validation_3e18"]


@pytest.fixture(scope="module")
def table(ctx):
    return heldout.export(ctx)["heldout"]


@pytest.fixture(scope="module")
def registry_rows(table):
    # The registry's runs keep its row order; the scale-transfer cohort follows them.
    rows = table[table.cohort != heldout.SCALE_TRANSFER_COHORT]
    assert rows.index.equals(pd.RangeIndex(929))
    return rows


@pytest.fixture(scope="module")
def scale_transfer(table):
    return table[table.cohort == heldout.SCALE_TRANSFER_COHORT].set_index("observation_id")


def test_cohorts_and_swarms(table):
    assert len(table) == 937 and table.observation_id.is_unique
    assert table.swarm.value_counts().to_dict() == {"qwen3_360m_1p6b": 621, "llama_160m_1p2b": 178, "llama_200m_6b": 138}
    qwen = table[table.swarm == "qwen3_360m_1p6b"]
    assert qwen.cohort.value_counts().to_dict() == {"frozen": 505, "expanded": 116}
    # The paper counts the frozen held-out runs of each swarm (tab:a-hyperparameters); the scale-transfer runs add none.
    llama = table[table.swarm != "qwen3_360m_1p6b"].groupby(["swarm", "cohort"]).size().to_dict()
    assert llama == {
        ("llama_160m_1p2b", "frozen"): 174,
        ("llama_160m_1p2b", "scale_transfer"): 4,
        ("llama_200m_6b", "frozen"): 134,
        ("llama_200m_6b", "scale_transfer"): 4,
    }


def test_expanded_rows_key_on_training_run_names(registry_rows):
    # validation_3e18 keys these runs on the hashed training run names in their eval_metrics_uri. The table keeps the
    # registry's row order.
    table = registry_rows
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


def test_frozen_rows_key_on_recorded_run_names(registry_rows):
    # The registry records the W&B training run id (the run name with its hash) for every frozen run except the 200M
    # ones; `delphi_archive::` row ids wrap that name.
    table = registry_rows
    runs = pd.read_csv(SRC / "heldout_runs.csv")
    archive = runs.row_id.str.startswith("delphi_archive::")
    stripped = runs.row_id[archive].str.removeprefix("delphi_archive::")
    assert archive.sum() == 187 and (table.observation_id[archive] == stripped).all()
    recorded = runs.training_wandb_run_id.notna()
    assert recorded.sum() == 679 and (table.cohort[recorded] == "frozen").all()
    assert (table.observation_id[recorded] == runs.training_wandb_run_id[recorded]).all()


def test_llama_200m_runs_take_their_w_and_b_ids_and_seeds(registry_rows):
    # The registry records neither the W&B run id nor the seeds of its 134 Llama 200M runs; both come from W&B. A W&B
    # run id is the run's name, of which 20 ids keep only the last 54 characters, and a six-hex-digit hash. Each run's
    # launch record gives its data seed; the launches left the trainer seed at Marin's default, 0.
    runs = pd.read_csv(SRC / "heldout_runs.csv")
    llama = runs.panel == "300m_39bucket"
    assert llama.sum() == 134 and runs.training_wandb_run_id[llama].isna().all()
    rows = registry_rows[llama].set_axis(runs.row_id[llama].str.removeprefix("300m::"))
    kept = rows.observation_id.str.extract(r"^(?P<name>.+)-[0-9a-f]{6}$")["name"]
    assert kept.notna().all() and all(name.endswith(k) for name, k in zip(rows.index, kept, strict=True))
    assert set(kept[kept != rows.index].str.len()) == {54} and (kept != rows.index).sum() == 20
    assert rows.observation_id.is_unique
    launches = pd.concat([pd.read_csv(path) for path in LLAMA_200M_LAUNCHES])
    launches = launches[launches.scale == LLAMA_200M_SCALE].set_index("run_name")
    assert (rows.data_seed == launches.data_seed.reindex(rows.index)).all()
    assert launches.trainer_seed.reindex(rows.index).isna().all() and (rows.trainer_seed == 0).all()


def test_runs_shared_with_validation_have_its_seeds(table, validation_table):
    # The registry records seeds for 42 of the 116 expanded runs. Like validation_3e18, heldout reads each validation
    # run's seeds from the directory that holds it, and the UniMax-8 seed-0 run's (a frozen row) from its launch record.
    expanded = table[table.cohort == "expanded"]
    assert len(expanded) == 116 and expanded[["data_seed", "trainer_seed"]].notna().all().all()
    shared = table.merge(validation_table, on="observation_id", suffixes=("", "_validation"), validate="one_to_one")
    assert len(shared) == 117
    for seed in ["data_seed", "trainer_seed"]:
        assert (shared[seed] == shared[f"{seed}_validation"]).all(), seed


def test_exposure_ids_split_what_mixture_ids_merge(table):
    frozen = table[(table.swarm == "qwen3_360m_1p6b") & (table.cohort == "frozen")]
    assert frozen.mixture_id.nunique() == 410
    assert frozen.exposure_id.nunique() == 425


def test_labels_are_kept(table, registry_rows):
    assert table.proposal_method.notna().all()
    # Objectives take their public names (the registry's `table9`, `table9_macro_bpb`, `uncheatable_and_table9` and
    # `baseline` among them); the registry's Llama panels record none.
    qwen = table.swarm == "qwen3_360m_1p6b"
    assert registry_rows.proposal_objective[registry_rows.swarm != "qwen3_360m_1p6b"].isna().all()
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


def test_aggregates_recompute_the_recorded_values(registry_rows):
    # The registry records each run's aggregates, row for row with the table. The Qwen3 panel records Uncheatable under
    # the fixed rule; the Llama panels recorded it under the legacy weights, within 1e-4 of the fixed rule for the
    # registry's components, and no row copies a legacy record. The Llama 160M run whose components come from W&B
    # pooled its record 3.6e-4 away from the fixed rule; its own test checks its pull. OlmoBaseEval Easy is a plain mean
    # in every panel, so it checks each swarm's component join.
    table = registry_rows
    runs = tables.read_csv(SRC / "heldout_runs.csv")
    pulls = pd.concat([tables.read_csv(path, comment="#") for path in heldout.WANDB_PULLS])
    pulled = table.observation_id.isin(pulls.training_wandb_run_id)
    complete = {
        objective: table[[f"bpb__{t}" for t in weights]].notna().all(axis=1)
        for objective, weights in naming.task_weights().items()
    }
    qwen = table.swarm == "qwen3_360m_1p6b"
    fixed = qwen & runs.uncheatable_bpb.notna()
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


def test_llama_200m_runs_without_registry_uncheatable_values_aggregate_their_pulled_components(registry_rows):
    # The registry records no Uncheatable component or aggregate for 17 Llama 200M runs. Their seven components come
    # from their W&B summaries, and bpb__uncheatable is the fixed byte-share sum of those components, never the pooled
    # aggregate that W&B also logged, which weights the corpora otherwise (1.5e-5 to 7.2e-5 BPB apart for these runs).
    table = registry_rows
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


def test_runs_whose_w_and_b_logging_crashed_take_their_saved_final_evaluations(registry_rows):
    # The registry scores four Qwen3 runs by their pooled Uncheatable aggregate only, and their W&B summaries hold the
    # evaluation at step 2000 or 1000. Their components come from the final evaluation, at step 3006, that each run
    # saved; their launch recorded the same file, step and pooled aggregate, and two of the components. On these legacy
    # payloads the registry's pooled record lies within 4.9e-8 of the fixed byte-share sum.
    runs = tables.read_csv(SRC / "heldout_runs.csv")
    pull = tables.read_csv(heldout.FINAL_EVALUATIONS, comment="#").set_index("training_wandb_run_id")
    assert set(pull.index) == FINAL_EVALUATION_RUNS and (pull.step == 3006).all()
    launch = tables.read_csv(DSP_EPOCH_CAP_RESULTS)
    launch = launch.set_index(launch.eval_metrics_uri.map(training_run_name_from_uri)).loc[pull.index]
    assert (launch.eval_metrics_uri == pull.eval_metrics_uri).all() and (launch.final_step == pull.step).all()
    assert (launch.uncheatable_bpb == pull["eval/uncheatable_eval/bpb"]).all()
    for corpus in ("github_cpp", "github_python"):
        assert (launch[f"{corpus}_bpb"] == pull[f"eval/uncheatable_eval/{corpus}/bpb"]).all()
    evaluated = registry_rows.observation_id.isin(pull.index)
    rows = registry_rows[evaluated].set_index("observation_id").loc[pull.index]
    weights = naming.task_weights()["uncheatable"]
    columns = [f"bpb__{t}" for t in weights]
    components = pull[[f"eval/uncheatable_eval/{t.removeprefix('uncheatable/')}/bpb" for t in weights]]
    assert (rows[columns].to_numpy() == components.to_numpy()).all() and rows.components_available.all()
    recorded = runs.uncheatable_bpb[evaluated].set_axis(registry_rows.observation_id[evaluated]).loc[pull.index]
    assert (rows.bpb__uncheatable - recorded).abs().max() < 5e-8
    # No source holds the Uncheatable components of the one other run that the registry scores by its pooled aggregate.
    missing = registry_rows[registry_rows[columns].isna().any(axis=1)]
    assert missing.observation_id.tolist() == ["hprv_002_hpr300m_t9_1p_aklc0p05-9af63a"]


def test_scale_transfer_runs_carry_figure_14s_proposals(scale_transfer, validation_table):
    # The launch trained the four Uncheatable proposals of validation_3e18 at both Llama settings. Each run carries its
    # proposal's validation_3e18 variant, mixture and epoch cap, and the final evaluation that the launch's collector
    # scored for Figure 14: the fixed byte-share sum of its components equals deployed_optima.csv to within one ulp,
    # because the collector added the terms with Python's compensated sum.
    deployed = tables.read_csv(heldout.SCALE_TRANSFER / "deployed_optima.csv").set_index("run_name")
    columns = {"llama_160m_1p2b": "bpb_60m", "llama_200m_6b": "bpb_300m"}
    names = scale_transfer.index.str.extract(r"^(?P<name>.+)-[0-9a-f]{6}$", expand=False)
    expected = [deployed.loc[name, columns[swarm]] for name, swarm in zip(names, scale_transfer.swarm, strict=True)]
    assert np.allclose(scale_transfer.bpb__uncheatable, expected, rtol=0, atol=2.3e-16)
    assert scale_transfer.swarm.value_counts().to_dict() == {"llama_160m_1p2b": 4, "llama_200m_6b": 4}
    assert (scale_transfer.proposal_objective == "uncheatable").all()
    proposals = validation_table.drop_duplicates("variant").set_index("variant").loc[scale_transfer.proposal_method]
    assert set(proposals.method) == {"mariner", "olmix", "regmix"}
    for column in ("mixture_id", "exposure_id", "epoch_cap"):
        assert (proposals[column].to_numpy() == scale_transfer[column].to_numpy()).all(), column
    uncheatable = [f"bpb__{t}" for t in naming.TASKS[:7]]
    assert scale_transfer[uncheatable].notna().all().all()
    assert scale_transfer[[f"bpb__{t}" for t in naming.TASKS[7:]] + ["bpb__obe_easy"]].isna().all().all()


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
