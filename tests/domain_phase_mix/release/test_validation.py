# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import re

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.release import context, naming, paths
from experiments.domain_phase_mix.release.configs import heldout, ladder, validation

pytestmark = pytest.mark.skipif(
    not (paths.RO / "delphi_frozen_procedure_validation_3e18_20260908").exists(), reason="validation sources absent"
)


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    return context.build(private_dir=tmp_path_factory.mktemp("private"))


@pytest.fixture(scope="module")
def table(ctx):
    return validation.export(ctx)["validation_3e18"]


def test_row_count_matches_manifest(table):
    manifest = pd.read_csv(validation.MANIFEST)
    assert len(table) == manifest.expected_rows.sum() == 218
    assert table.observation_id.is_unique
    assert set(table.launch) == set(manifest.launch)


def test_headline_numbers(table):
    mariner = table[
        (table.launch.isin(["frozen_procedure", "fairness_repeats"])) & (table.variant == "lwspu_t9_snc_cap08")
    ]
    assert len(mariner) == 3 and (mariner.method == "mariner").all()
    assert mariner.bpb__obe_easy.mean() == pytest.approx(1.067779, abs=5e-6)
    assert mariner.bpb__obe_easy.std(ddof=1) == pytest.approx(0.005174, abs=5e-6)
    unch = table[(table.launch == "frozen_procedure") & (table.variant == "lwspu_u_snc_cap06")]
    assert unch.bpb__uncheatable_fixed.iloc[0] == pytest.approx(0.981424, abs=5e-6)


def test_every_row_has_weights_and_components(table):
    w = table[[f"w__{b}" for b in naming.BUCKETS]]
    assert w.notna().all().all() and np.allclose(w.sum(axis=1), 1.0, rtol=0, atol=1e-9)
    components = table[[f"bpb__{t}" for t in naming.TASKS]]
    assert components.notna().all(axis=1).mean() > 0.95
    # Only UniMax-8 trainer seeds 1 and 2 lack per-task values (their components exist only on GCS); they lack all 58.
    missing = components.isna().any(axis=1)
    unimax = sorted(zip(table.variant[missing], table.trainer_seed[missing], strict=True))
    assert unimax == [("unimax8", 1), ("unimax8", 2)]
    assert components[missing].isna().all().all()
    assert table[["bpb__uncheatable_source", "bpb__obe_easy_source"]].notna().all().all()


def test_seeds_come_from_run_directories(table):
    # MARINER's three trainer seeds share one data seed; the frozen procedure's run sits in a group without `_t`.
    mariner = table[(table.variant == "lwspu_t9_snc_cap08") & table.trainer_seed.isin([0, 1, 2])]
    assert sorted(mariner.trainer_seed) == [0, 1, 2] and mariner.data_seed.nunique() == 1
    assert set(mariner.launch) == {"frozen_procedure", "fairness_repeats"}
    assert table[["data_seed", "trainer_seed"]].notna().all().all()


def test_subset_proposals_name_their_objective(table):
    subset = {
        "lwspu_w3_snc_cap06": "uncheatable_worsened",
        "olmixq_w3_kl0p05_cap04": "uncheatable_worsened",
        "qa15_t9_cap04": "obe_easy_qa15",
        "olmixq_qa15_kl0p005_cap04": "obe_easy_qa15",
        "qa19_t9_cap05": "obe_easy_qa19",
        "snr5_t9_cap09": "obe_easy_snr5",
        "snr8_t9_cap07": "obe_easy_snr8",
    }
    for variant, objective in subset.items():
        rows = table[table.variant == variant]
        assert len(rows) == 3 and (rows.proposal_objective == objective).all(), variant
    # Every other proposal optimized a full objective, so these filters select full-objective runs only; the UniMax-8
    # baseline optimizes none.
    others = table[~table.variant.isin(subset)]
    assert set(others.proposal_objective[others.variant != "unimax8"]) == {"uncheatable", "obe_easy"}
    olmix_uncheatable = table[(table.method == "olmix") & (table.proposal_objective == "uncheatable")]
    assert not olmix_uncheatable.variant.isin(subset).any()


def test_unimax8_rows_carry_the_ladder_labels(table):
    # The ladder's 3e18 rung copies these runs; both tables name the baseline and its cap alike.
    unimax = table[table.variant == "unimax8"]
    method, objective = ladder.POLICY["unimax8"]
    assert len(unimax) == 3 and (unimax.method == method).all() and (unimax.proposal_objective == objective).all()
    assert (unimax.epoch_cap == 8).all()


def test_validation_join_reports_unmatched_candidates(tmp_path):
    manifest = pd.read_csv(validation.MANIFEST).iloc[[0]].copy()
    candidates = pd.read_csv(paths.REPO / manifest.candidates_csv.iloc[0])
    candidates["candidate_id"] = [f"{c}_typo" for c in candidates.candidate_id]
    broken = tmp_path / "candidates.csv"
    candidates.to_csv(broken, index=False)
    manifest["candidates_csv"] = str(broken)
    with pytest.raises(ValueError, match=manifest.launch.iloc[0]):
        validation.export_launches(context.build(private_dir=tmp_path / "p"), manifest)


def test_runs_shared_with_heldout_keep_their_ids_and_values(ctx, table):
    # Both tables key a run on its hashed training run name, so the 116 validation runs of the expanded held-out
    # cohort and the ladder's UniMax-8 seed-0 run (a frozen archive row) get one id and one set of values. The seeds
    # differ: `heldout` leaves them empty for 75 of these runs (a Task 4 follow-up).
    held = heldout.export(ctx)["heldout"]
    expanded = held[(held.swarm == "qwen3_360m_1p6b") & (held.cohort == "expanded")]
    assert len(expanded) == 116 and expanded.observation_id.isin(table.observation_id).all()
    shared = held.merge(table, on="observation_id", suffixes=("_heldout", "_validation"), validate="one_to_one")
    assert len(shared) == 117 and set(shared.cohort) == {"expanded", "frozen"}
    for column in ["mixture_id", "exposure_id", "aggregate_version"]:
        assert (shared[f"{column}_heldout"] == shared[f"{column}_validation"]).all()
    aggregates = ["bpb__uncheatable_source", "bpb__obe_easy_source", "bpb__uncheatable_fixed", "bpb__obe_easy"]
    for column in [f"bpb__{t}" for t in naming.TASKS] + aggregates:
        values = shared[[f"{column}_heldout", f"{column}_validation"]].dropna()
        assert np.allclose(values.iloc[:, 0], values.iloc[:, 1], rtol=0, atol=1e-12)


def test_aggregates_are_versioned(table):
    # path_midpoints and regmix_reference recorded the BPB-schema-2 pooled Uncheatable aggregate, about 0.01 BPB above
    # the fixed weighting of the same components; every other launch recorded the fixed weighting.
    pooled = table[table.aggregate_version == "pooled_schema2"]
    assert len(pooled) == 14 and set(pooled.launch) == {"path_midpoints", "regmix_reference"}
    assert (pooled.bpb__uncheatable_source - pooled.bpb__uncheatable_fixed).between(0.005, 0.013).all()
    fixed = table[(table.aggregate_version == "fixed_20260927") & table.bpb__uncheatable_fixed.notna()]
    assert len(fixed) == 202  # every fixed-version run except UniMax-8 trainer seeds 1 and 2
    assert np.allclose(fixed.bpb__uncheatable_fixed, fixed.bpb__uncheatable_source, rtol=0, atol=2e-6)
    # OlmoBaseEval Easy is a plain mean, so it checks that each run, seed by seed, got its own components.
    obe = table[table.bpb__obe_easy.notna()]
    assert np.allclose(obe.bpb__obe_easy, obe.bpb__obe_easy_source, rtol=0, atol=1e-12)


def test_unimax8_weights_are_the_trained_mixture():
    # The UniMax-8 weights come from launch_delphi_baseline_mixtures.py's construction; the frozen held-out registry
    # records the ladder's seed-0 run and the launch record holds both phases of seeds 1 and 2.
    manifest = pd.read_csv(validation.MANIFEST).set_index("launch")
    weights = pd.read_csv(paths.REPO / manifest.candidates_csv["unimax8_repeats"]).set_index("domain").weight
    assert len(weights) == 39
    frozen = pd.read_csv(paths.RO / "single_phase_heldout_benchmark_20260902/heldout_runs.csv")
    seed0 = frozen.loc[frozen.row_id == "delphi_archive::unimax8_3e18-cb3b49", [f"weight::{d}" for d in weights.index]]
    assert len(seed0) == 1 and np.allclose(seed0.to_numpy(dtype=float)[0], weights.to_numpy(), rtol=0, atol=1e-15)
    specs = json.loads((paths.AG / "unimax8_repeats_20260922/dry_run/run_specs.json").read_text())
    assert sorted(spec["trainer_seed"] for spec in specs) == [1, 2]
    for spec in specs:
        for phase in spec["phase_weights"].values():
            assert np.allclose([phase[d] for d in weights.index], weights.to_numpy(), rtol=0, atol=1e-15)


def test_no_internal_strings(table):
    text = table.to_csv(index=False)
    for needle in ["gs://", "http", "wandb", "pinlin_calvin_xu", "/Users/"]:
        assert needle not in text
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)  # email addresses
    # Fieldbook ids are `exp_`/`run_`/`job_` plus a 26-character ULID; public exposure ids are `exp_` plus 16 hex digits.
    assert not re.search(r"(?:exp|run|job)_01[a-z0-9]{24}", text)
    assert not re.search(r"[0-9a-f]{64}", text)
    assert not re.search(r"\w-[0-9a-f]{6}(?![0-9a-z])", text)  # run names end in a six-hex-digit hash
