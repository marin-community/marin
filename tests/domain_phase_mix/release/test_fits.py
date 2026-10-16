# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.release import context, naming, paths, tables
from experiments.domain_phase_mix.release.configs import fits, heldout, swarm_qwen3, validation

pytestmark = pytest.mark.skipif(not all(path.exists() for path in fits.required_sources()), reason="fits sources absent")


@pytest.fixture(scope="module")
def ctx():
    return context.build()


@pytest.fixture(scope="module")
def out(ctx):
    return fits.export(ctx)


@pytest.fixture(scope="module")
def heldout_table(ctx):
    return heldout.export(ctx)["heldout"]


@pytest.fixture(scope="module")
def validation_table(ctx):
    return validation.export(ctx)["validation_3e18"]


def _aggregate(predictions: pd.DataFrame, objective: str, weights: dict[str, float]) -> pd.Series:
    wide = predictions[predictions.objective == objective].pivot(
        index="observation_id", columns="task_id", values="predicted_bpb"
    )
    assert set(wide.columns) == set(weights)
    return (wide[list(weights)] * pd.Series(weights)).sum(axis=1)


def test_task_parameters(out):
    t = out["fits_task_parameters"]
    assert len(t) == 3 * 58
    for swarm in ("qwen3_360m_1p6b", "llama_160m_1p2b", "llama_200m_6b"):
        assert sorted(t[t.swarm == swarm].task_id) == sorted(naming.TASKS), swarm
    assert t.kappa.between(1, 6).all() and (t.ridge >= 0).all()


def test_learning_curves_round_trip_parquet(out, tmp_path):
    t = out["fits_learning_curves"]
    assert len(t) == 165_360
    assert set(t.study) == {"mariner_regmix_oof", "pbt", "extra"}
    path = tmp_path / "lc.parquet"
    t.to_parquet(path, index=False)
    back = pd.read_parquet(path)
    pd.testing.assert_frame_equal(back, t, check_exact=True)
    assert str(back.k.dtype) in ("int64", "Int64") and str(back.n_failed.dtype) in ("int64", "Int64", "float64")


def test_learning_curves_name_tasks_and_selections_publicly(out, heldout_table):
    t = out["fits_learning_curves"]
    assert set(t.target) == {"obe_easy", "uncheatable"}
    assert str(t.n_basin.dtype) == "Int64" and str(t.basin_hit.dtype) == "boolean"
    per_task = t[(t.evaluation == "oof_component") & (t.stratum != "component_mean")]
    assert set(per_task.stratum) == set(naming.TASKS)
    # A held-out selection names the selected mixture by its exposure id; the measured BPB the study recorded for it
    # is the mean of the objective's recorded aggregate over that exposure's held-out runs. The registry records the
    # aggregates row for row with the held-out table.
    selected = t[t.selected_exposure_id.notna()]
    assert (selected.evaluation == "heldout").all() and len(selected) == 50_400
    runs = tables.read_csv(heldout.EXPANDED / "heldout_runs.csv")
    recorded = pd.DataFrame({"obe_easy": runs.table9_macro_bpb, "uncheatable": runs.uncheatable_bpb})
    recorded["exposure_id"] = heldout_table.exposure_id
    means = recorded.groupby("exposure_id").mean()
    for objective, rows in selected.groupby("target"):
        expected = rows.selected_exposure_id.map(means[objective])
        assert expected.notna().all()
        assert ((rows.selected_measured_bpb - expected).abs() <= 1e-9).all()


def test_swarm_predictions_name_tasks_publicly(out):
    p = out["fits_swarm_predictions"]
    assert set(p.task_id) <= set(naming.TASKS)
    assert p.groupby(["objective", "observation_id"]).size().max() <= 58


def test_swarm_predictions_key_on_the_fit_rows_of_the_swarm_table(ctx, out):
    p = out["fits_swarm_predictions"]
    swarm = swarm_qwen3.export(ctx)["swarm_qwen3_360m_1p6b"]
    fit = swarm[swarm.fit_inclusion].set_index("observation_id")
    assert set(p.observation_id) == set(fit.index)
    # The reference fit's values at its own training mixtures follow the measured BPB (rank correlation 0.94 on
    # OlmoBaseEval Easy, 0.98 on Uncheatable); a run keyed to another run's predictions would break that.
    for objective, weights in ctx.weights.items():
        predicted = _aggregate(p, objective, weights)
        measured = fit[f"bpb__{objective}"].reindex(predicted.index)
        assert predicted.corr(measured, method="spearman") > 0.9


def test_heldout_predictions_key_through_the_heldout_registry(ctx, out, heldout_table):
    p = out["fits_heldout_predictions"]
    table = heldout_table.set_index("observation_id")
    assert p.observation_id.isin(table.index).all()
    # The bank predicts 247 OlmoBaseEval Easy and 408 Uncheatable held-out mixtures, trained in 310 and 471 runs; every
    # run of a mixture carries that mixture's predictions.
    exposure = p.observation_id.map(table.exposure_id)
    assert p.groupby("objective").observation_id.nunique().to_dict() == {"obe_easy": 310, "uncheatable": 471}
    assert exposure.groupby(p.objective).nunique().to_dict() == {"obe_easy": 247, "uncheatable": 408}
    assert p.groupby([p.objective, exposure, p.task_id]).predicted_bpb.nunique().max() == 1
    for objective, weights in ctx.weights.items():
        predicted = _aggregate(p, objective, weights)
        measured = table[f"bpb__{objective}"].reindex(predicted.index)
        assert predicted.corr(measured, method="spearman") > 0.9


def test_reliability_reproduces_the_papers_snr_table(out):
    r = out["fits_reliability"]
    components = r[r.source_table == "snr_fit_components_delphi"]
    assert set(components.task_id.dropna()) == set(naming.TASKS[7:])
    # Rows of the appendix's SNR table (tab:a-snr): swarm mean and std, proportional mean and std, SNR, and MARINER's
    # and Olmix's out-of-fold Spearman correlations, as printed. The OlmoBaseEval Easy mean has no task id.
    printed = {
        None: ("1.223", "0.047", "1.198", "0.0038", "12.6", "0.89", "0.80"),
        "minerva_math_algebra": ("1.068", "0.058", "1.157", "0.0080", "7.2", "0.91", "0.69"),
        "hellaswag": ("1.042", "0.026", "0.986", "0.0016", "16.3", "0.89", "0.88"),
        "winogrande": ("1.459", "0.023", "1.421", "0.0088", "2.6", "0.69", "0.67"),
        "sciq": ("1.351", "0.150", "1.192", "0.0093", "16.1", "0.74", "0.63"),
    }
    for task, values in printed.items():
        rows = components[components.task_id.isna() if task is None else components.task_id == task]
        assert len(rows) == 1
        row = rows.iloc[0]
        formatted = (
            f"{row.panel_mean:.3f}",
            f"{row.panel_sd:.3f}",
            f"{row.proportional_mean:.3f}",
            f"{row.repeat_sd:.4f}",
            f"{row.snr:.1f}",
            f"{row.mariner_spearman:.2f}",
            f"{row.olmix_spearman:.2f}",
        )
        assert formatted == values, task
    uncheatable = r[r.source_table == "snr_fit_uncheatable_delphi"]
    assert set(uncheatable.task_id.dropna()) == set(naming.TASKS[:7])


def _predicted(p: pd.DataFrame, variant: str, objective: str, predictor: str) -> set[str]:
    rows = p[(p.variant == variant) & (p.objective == objective) & (p.predictor == predictor)]
    return {f"{value:.4f}" for value in rows.predicted_bpb}


def test_proposal_predictions_key_on_validation_runs(out, validation_table):
    p = out["fits_proposal_predictions"]
    runs = validation_table.set_index("observation_id")
    assert p.observation_id.isin(runs.index).all()
    assert (p.launch == p.observation_id.map(runs.launch)).all()
    assert (p.variant == p.observation_id.map(runs.variant)).all()
    assert not p.duplicated(["observation_id", "objective", "predictor"]).any()
    # The frozen procedure's review table records MARINER's prediction of each of its proposals' objectives.
    review = pd.read_csv(
        paths.RO / "delphi_frozen_procedure_validation_3e18_20260908/review_table.csv", float_precision="round_trip"
    )
    review["objective"] = review.target.map({"table9": "obe_easy", "uncheatable": "uncheatable"})
    mariner = p[(p.predictor == "mariner") & p.own_prediction].merge(
        review[["candidate_id", "objective", "predicted"]],
        left_on=["variant", "objective"],
        right_on=["candidate_id", "objective"],
        validate="many_to_one",
    )
    assert mariner.observation_id.nunique() == 9  # trainer seeds 0 to 2 of the three proposals
    assert ((mariner.predicted_bpb - mariner.predicted).abs() <= 1e-12).all()


def test_runs_without_predictions_are_the_documented_gaps(out, validation_table):
    p = out["fits_proposal_predictions"]
    missing = validation_table[~validation_table.observation_id.isin(p.observation_id)]
    # The epoch-cap and KL sweep recorded Olmix's predictions at its continuous optima only; its five settings that the
    # matched Olmix launch also trained carry that launch's predictions.
    sweep = missing.launch == "olmix_cap_kl_sweep"
    assert sweep.sum() == 34
    # No folder records predictions for the native Olmix proposals, the best swarm run, the QA-subset proposals (whose
    # validation records predictions relative to MARINER only) or UniMax-8.
    assert set(missing.variant[~sweep]) == {
        "olmix_u_kl0p1_cap04",
        "olmix_t9_kl0p005_cap05",
        "olmix_u_kl0p05_cap04",
        "swarm_run00125_cap08",
        "qa15_t9_cap04",
        "qa19_t9_cap05",
        "unimax8",
    }


def test_proposal_predictions_reproduce_the_kl_and_regmix_tables(out):
    p = out["fits_proposal_predictions"]
    # tab:a-kl-ablation: MARINER's predicted BPB, without the penalty, of its KL-penalized proposals.
    printed = {
        "0p005": ("0.9817", "1.0633"),
        "0p025": ("0.9862", "1.0670"),
        "0p1": ("0.9946", "1.0806"),
        "0p5": ("1.0103", "1.1163"),
    }
    for kl, (uncheatable, obe_easy) in printed.items():
        assert _predicted(p, f"lwspu_u_kl{kl}_cap06", "uncheatable", "mariner") == {uncheatable}
        assert _predicted(p, f"lwspu_t9_kl{kl}_cap08", "obe_easy", "mariner") == {obe_easy}
    # tab:a-regmix-argmin: the tuned RegMix trees' and MARINER's predictions of MARINER's proposal, RegMix's sampled
    # proposal and its MILP proposal.
    printed = {
        ("lwspu_u_snc_cap06", "uncheatable"): ("1.0084", "0.9810"),
        ("cmp_u_lgbm_cap08", "uncheatable"): ("1.0138", "0.9982"),
        ("rgx_u_argmin_center_cap08", "uncheatable"): ("0.9831", "0.9933"),
        ("lwspu_t9_snc_cap08", "obe_easy"): ("1.1018", "1.0631"),
        ("cmp_t9_lgbm_cap12", "obe_easy"): ("1.1220", "1.0861"),
        ("rgx_t9_argmin_center_cap12", "obe_easy"): ("1.0854", "1.0860"),
    }
    for (variant, objective), (trees, mariner) in printed.items():
        assert _predicted(p, variant, objective, "regmix") == {trees}
        assert _predicted(p, variant, objective, "mariner") == {mariner}


def test_olmix_task_laws_reproduce_the_matched_olmix_predictions(ctx, out, validation_table):
    laws = out["fits_olmix_task_laws"]
    assert len(laws) == 58 and set(laws.task_id) == set(naming.TASKS)
    assert laws.groupby("objective").size().to_dict() == {"obe_easy": 51, "uncheatable": 7}
    coefficients = [f"coef__{b}" for b in naming.BUCKETS]
    assert [c for c in laws.columns if c.startswith("coef__")] == coefficients
    assert laws[coefficients].notna().all().all()
    # Each law is BPB(w) = exp(log_c) + exp(sum_b coef__b * w_b). The matched Olmix folder records the objective each
    # trained proposal was predicted to reach; the laws at that run's published weights must give it.
    recorded = pd.read_json(paths.RO / "delphi_matched_olmix_3e18_20260908/summary.json", orient="index")
    p = out["fits_proposal_predictions"]
    for variant, row in recorded.iterrows():
        run = validation_table[(validation_table.launch == "matched_olmix") & (validation_table.variant == variant)]
        weights = run[[f"w__{b}" for b in naming.BUCKETS]].iloc[0].to_numpy()
        objective = {"table9": "obe_easy", "uncheatable": "uncheatable"}[row.target]
        task_laws = laws[laws.objective == objective].set_index("task_id")
        per_task = np.exp(task_laws.log_c) + np.exp(task_laws[coefficients].to_numpy() @ weights)
        predicted = (per_task * pd.Series(ctx.weights[objective])).sum()
        assert abs(predicted - row.predicted_runtime) <= 1e-9, variant
        own = p[(p.observation_id.isin(run.observation_id)) & (p.predictor == "olmix") & (p.objective == objective)]
        assert ((own.predicted_bpb - predicted).abs() <= 1e-9).all() and len(own) == len(run)


def test_factorial_predictions_reproduce_the_predicted_effects(out):
    # tab:a-factorial-predictions: MARINER's predicted centre loss and main effects (mean prediction at a factor's
    # upper setting minus that at its lower setting over the 16 corners; corners are named by their A-E settings).
    printed = {
        "obe_easy": ("1.0782", ["-0.0057", "-0.0004", "+0.0027", "+0.0060", "+0.0005"]),
        "uncheatable": ("0.9941", ["+0.0010", "+0.0004", "-0.0013", "+0.0034", "-0.0025"]),
    }
    p = out["fits_proposal_predictions"]
    factorial = p[(p.launch == "factorial") & (p.predictor == "mariner")]
    for objective, (centre, effects) in printed.items():
        rows = factorial[factorial.objective == objective]
        centres = rows[rows.variant.str.startswith("centre_")]
        assert len(centres) == 2 and f"{centres.predicted_bpb.mean():.4f}" == centre
        corners = rows[rows.variant.str.startswith("fac_")].set_index("variant").predicted_bpb
        assert len(corners) == 16
        settings = corners.index.str.extract(r"^fac_([pm]{5})_cap16$", expand=False)
        for factor, effect in enumerate(effects):
            upper = settings.str[factor] == "p"
            assert f"{corners[upper].mean() - corners[~upper].mean():+.4f}" == effect


def test_surrogate_benchmark_holds_every_model_on_every_swarm(out):
    bench = out["fits_surrogate_benchmark"]
    assert len(bench) == 28 * 6 and not bench.duplicated(["model", "swarm", "objective"]).any()
    assert set(bench.swarm) == set(heldout.SWARM.values()) and set(bench.objective) == {"obe_easy", "uncheatable"}
    # Epochs rescale weights invertibly, so the two linear models fit identically (the paper's two linear rows).
    linear = bench[bench.model.str.startswith("linear_")].pivot(index=["swarm", "objective"], columns="model")
    weight, exposure = (linear.xs(model, axis=1, level=1) for model in ("linear_weight", "linear_exposure"))
    assert np.allclose(weight, exposure, rtol=0, atol=1e-12) and (weight.spearman == exposure.spearman).all()


def test_no_internal_strings(out):
    # The registry's row ids and coordinate ids are internal; observation ids are the runs' W&B run ids.
    runs = pd.read_csv(heldout.EXPANDED / "heldout_runs.csv", usecols=["row_id"])
    patterns = [
        re.compile(r"gs://|http|/Users/|pinlin_calvin_xu"),
        re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),  # email addresses
        re.compile(r"[0-9a-f]{64}"),  # registry coordinate ids and policy hashes
        re.compile(r"(?:exp|run|job)_01[a-z0-9]{24}"),  # Fieldbook ids
    ]
    for name, t in out.items():
        assert name.startswith("fits_")
        strings = set(t.columns)
        for column in t.columns:
            if not pd.api.types.is_numeric_dtype(t[column]) and not pd.api.types.is_bool_dtype(t[column]):
                strings |= set(t[column].dropna().astype(str))
        assert not strings & set(runs.row_id), (name, sorted(strings & set(runs.row_id))[:3])
        text = "\n".join(sorted(strings))
        for pattern in patterns:
            assert not pattern.search(text), (name, pattern.pattern, pattern.search(text))


def test_prebuilt_tables_give_the_same_export(ctx, out, heldout_table, validation_table):
    # The orchestrator passes the held-out and validation tables it built instead of exporting them again.
    prebuilt = fits.export(ctx, heldout_table=heldout_table, validation_table=validation_table)
    assert prebuilt.keys() == out.keys()
    for name, table in out.items():
        pd.testing.assert_frame_equal(prebuilt[name], table, check_exact=True)


def test_required_sources_are_the_files_the_export_reads(ctx, monkeypatch):
    # The skip guard above relies on this list. The export also runs the held-out and validation exporters, whose own
    # sources are theirs to list.
    read = set()
    require = paths.require
    monkeypatch.setattr(paths, "require", lambda path: read.add(path) or require(path))
    heldout.export(ctx)
    validation.export(ctx)
    upstream = set(read)
    read.clear()
    fits.export(ctx)
    required = set(fits.required_sources())
    assert required <= read and read - upstream <= required
