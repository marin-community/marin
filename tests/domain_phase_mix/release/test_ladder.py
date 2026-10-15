# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import re

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.release import context, naming, paths, tables
from experiments.domain_phase_mix.release.configs import ladder, swarm_qwen3, validation

FIG = paths.RO / "frozen_scaling_update_20260913/figure/plotted_points.csv"
pytestmark = pytest.mark.skipif(not FIG.exists(), reason="ladder sources absent")
# Each scaling launch's record of what its runs trained, and the run specs of the frozen procedure's 3e18 runs.
LAUNCH_RECORDS = [
    paths.RO / "delphi_frozen_procedure_scaling_v6e_20260908/launch_dry_run/run_manifest.json",
    paths.RO / "delphi_matched_olmix_scaling_v6e_20260910/launch_dry_run/run_manifest.json",
]
RUN_SPECS_3E18 = paths.RO / "delphi_frozen_procedure_validation_3e18_20260908/launch_dry_run/table9/run_specs.json"
# The figure's proportional point at 3e18 averages the June run with the swarm's ten repeats; this file holds the
# Uncheatable components of all eleven.
PROPORTIONAL_UNCHEATABLE = paths.RO / "table9_reliability_20260905/proportional_uncheatable_components.csv"


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    return context.build(private_dir=tmp_path_factory.mktemp("private"))


@pytest.fixture(scope="module")
def out(ctx):
    return ladder.export(ctx)


@pytest.fixture(scope="module")
def validation_table(ctx):
    return validation.export(ctx)["validation_3e18"]


@pytest.fixture(scope="module")
def swarm_table(ctx):
    return swarm_qwen3.export(ctx)["swarm_qwen3_360m_1p6b"]


@pytest.fixture(scope="module")
def plotted(out, validation_table, swarm_table):
    # The scaling figure plots the ladder's rows and, at 3e18, validation and swarm runs, which with_3e18_rows copies.
    return ladder.with_3e18_rows(out["ladder"], validation_table, swarm_table)


@pytest.mark.parametrize("flops", [3e18, 2e19, 3e20, 1e21])
def test_every_plotted_point_is_reproduced(plotted, flops):
    points = tables.read_csv(FIG)
    points = points[points.flops == flops]
    assert len(points) == 8  # four mixtures on each of the two objectives
    t = plotted
    label = {"MARINER": "mariner", "Olmix": "olmix", "Proportional": "proportional", "UniMax-8": "unimax8"}
    for p in points.itertuples(index=False):
        rows = t[(t.method == label[p.label]) & (t.compute_flops == p.flops)]
        if p.target == "table9":
            rows = rows[rows.proposal_objective.isin(["obe_easy", "none"])]
            values = rows.bpb__obe_easy_source
        else:
            rows = rows[rows.proposal_objective.isin(["uncheatable", "none"])]
            values = rows.bpb__uncheatable_source
        assert len(rows) == p.n, (p.label, p.flops, p.target, len(rows))
        assert np.isclose(values.mean(), p.mean, rtol=0, atol=1e-12), (p.label, p.flops, p.target)


def test_scale_constants_and_weights(out):
    t = out["ladder"]
    assert sorted(t.compute_flops.unique()) == [3e18, 2e19, 3e20, 1e21]
    # At 3e18 the export holds only the June proportional run, which no other table holds; with_3e18_rows adds the rest.
    assert t.method[t.compute_flops == 3e18].tolist() == ["proportional"]
    assert t.groupby("compute_flops").total_parameters.nunique().eq(1).all()
    w = t[[f"w__{b}" for b in naming.BUCKETS]]
    assert np.allclose(w.sum(axis=1), 1.0, rtol=0, atol=1e-9)
    assert t.groupby(["method", "proposal_objective"]).mixture_id.nunique().eq(1).all()


def test_3e18_rows_copy_their_runs(plotted, validation_table, swarm_table):
    rung = plotted[plotted.compute_flops == 3e18].set_index("observation_id")
    # Trainer seeds 0 to 2 of each validated mixture, and the June proportional run with the swarm's ten repeats.
    assert rung.groupby(["method", "proposal_objective"]).size().to_dict() == {
        ("mariner", "obe_easy"): 3,
        ("mariner", "uncheatable"): 3,
        ("olmix", "obe_easy"): 3,
        ("olmix", "uncheatable"): 3,
        ("proportional", "none"): 11,
        ("unimax8", "none"): 3,
    }
    scale = pd.Series(ladder.SCALES[3e18])
    assert (rung[scale.index] == scale).all().all()
    copied = 0
    for source in (validation_table, swarm_table):
        source = source.set_index("observation_id")
        ids = rung.index.intersection(source.index)
        shared = [c for c in rung.columns if c in source.columns and c not in ("method", "proposal_objective")]
        pd.testing.assert_frame_equal(rung.loc[ids, shared], source.loc[ids, shared], check_dtype=False)
        copied += len(ids)
    assert copied == len(rung) - 1  # all but the June run
    # The swarm table records no OlmoBaseEval Easy source aggregate; its copies take their components' mean.
    repeats = rung.loc[rung.index.intersection(swarm_table.observation_id)]
    components = repeats[[f"bpb__{task}" for task in naming.TASKS[7:]]]
    assert len(repeats) == 10 and np.allclose(repeats.bpb__obe_easy_source, components.mean(axis=1), rtol=0, atol=0)
    # UniMax-8 trainer seeds 1 and 2 carry aggregates only.
    assert rung.components_available.sum() == len(rung) - 2


def test_june_proportional_run_has_the_figures_components(ctx, out):
    t = out["ladder"]
    june = t[t.compute_flops == 3e18].iloc[0]
    assert june.components_available
    # The figure's input lists the June run among the eleven proportional runs; its seven components are the run's.
    source = tables.read_csv(PROPORTIONAL_UNCHEATABLE)
    source = source.set_index(source.run.map(ctx.registry.id_for)).loc[june.observation_id]
    for task in naming.TASKS[:7]:
        assert june[f"bpb__{task}"] == source[f"eval/uncheatable_eval/{task.removeprefix('uncheatable/')}/bpb"]


def test_accuracy_table(out):
    a = out["ladder_accuracy"]
    assert len(a) == 4 * 51 and set(a.method) == {"mariner", "olmix", "proportional", "unimax8"}
    assert set(a.task_id) == set(naming.TASKS[7:])
    assert a.accuracy.between(0, 100).all() and a.n_examples.gt(0).all()


def test_runs_match_their_launch_records(out):
    # The scaling launches recorded each run's seeds, model size, token count and the weights of both phases.
    t = out["ladder"]
    records = [record for path in LAUNCH_RECORDS for record in json.loads(path.read_text())["runs"]]
    assert len(records) == 12
    for record in records:
        run = record["run"]
        method, objective = ladder.POLICY[record["policy"]["candidate_id"]]
        rows = t[(t.method == method) & (t.proposal_objective == objective) & (t.compute_flops == run["target_flops"])]
        assert len(rows) == 1, run["run_name"]
        row = rows.iloc[0]
        assert (row.data_seed, row.trainer_seed) == (run["data_seed"], run["trainer_seed"])
        size = (row.total_parameters, row.non_embedding_parameters, row.train_tokens)
        assert size == (run["total_trainable_params"], run["non_embedding_params"], run["realized_train_tokens"])
        weights = row[[f"w__{b}" for b in naming.BUCKETS]].to_numpy(dtype=float)
        for phase in run["phase_weights"].values():
            trained = [phase[d] for d in naming.bucket_table().internal_name]
            assert np.allclose(weights, trained, rtol=0, atol=1e-15), run["run_name"]
    # The 3e18 constants, which Task 12 gives the validation runs at that rung, are what those runs trained.
    spec = json.loads(RUN_SPECS_3E18.read_text())[0]
    scale = ladder.SCALES[3e18]
    size = (scale["total_parameters"], scale["non_embedding_parameters"], scale["train_tokens"])
    assert size == (spec["total_trainable_params"], spec["non_embedding_params"], spec["realized_train_tokens"])


def test_components_cover_every_run_but_the_mid_ladder_baselines(out):
    t = out["ladder"]
    obe = t[[f"bpb__{task}" for task in naming.TASKS[7:]]]
    assert t[[f"bpb__{task}" for task in naming.TASKS[:7]]].notna().all().all()
    # The June baselines' OlmoBaseEval Easy components at 2e19 and 3e20 are on W&B only.
    missing = t[~t.components_available]
    assert sorted(zip(missing.method, missing.compute_flops, strict=True)) == [
        ("proportional", 2e19),
        ("proportional", 3e20),
        ("unimax8", 2e19),
        ("unimax8", 3e20),
    ]
    assert obe[~t.components_available].isna().all().all() and obe[t.components_available].notna().all().all()
    # Each run's components reproduce its recorded aggregates: Uncheatable within the 1e-7 by which legacy payloads
    # differ from the fixed weighting, OlmoBaseEval Easy (a plain mean) to rounding.
    assert np.allclose(t.bpb__uncheatable_fixed, t.bpb__uncheatable_source, rtol=0, atol=1e-7)
    complete = t[t.components_available]
    assert np.allclose(complete.bpb__obe_easy, complete.bpb__obe_easy_source, rtol=0, atol=1e-12)


def test_accuracy_rows_are_the_1e21_ladder_runs(out):
    t = out["ladder"].set_index("observation_id")
    a = out["ladder_accuracy"]
    assert a.observation_id.nunique() == 4 and a.groupby("observation_id").method.nunique().eq(1).all()
    for observation_id, rows in a.groupby("observation_id"):
        run = t.loc[observation_id]
        assert run.compute_flops == 1e21 and run.method == rows.method.iloc[0]
        assert run.proposal_objective in ("obe_easy", "none")
        values = run[[f"bpb__{task}" for task in rows.task_id]].to_numpy(dtype=float)
        assert np.allclose(rows.bpb, values, rtol=0, atol=1e-12)


def test_no_internal_strings(out):
    text = out["ladder"].to_csv(index=False) + out["ladder_accuracy"].to_csv(index=False)
    for needle in ["gs://", "http", "wandb", "pinlin_calvin_xu", "/Users/"]:
        assert needle not in text
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)  # email addresses
    # Fieldbook ids are `exp_`/`run_`/`job_` plus a 26-character ULID; public exposure ids are `exp_` plus 16 hex digits.
    assert not re.search(r"(?:exp|run|job)_01[a-z0-9]{24}", text)
    assert not re.search(r"[0-9a-f]{64}", text)
    assert not re.search(r"\w-[0-9a-f]{6}(?![0-9a-z])", text)  # run names end in a six-hex-digit hash
