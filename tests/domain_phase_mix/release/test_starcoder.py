# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import re

import numpy as np
import pandas as pd
import pytest
from levanter.utils.flop_utils import lm_flops_per_token
from scipy import stats

from experiments.domain_phase_mix.release import paths, pull_switch_traces
from experiments.domain_phase_mix.release.configs import domain_sweeps, starcoder

ATLAS = paths.RO / "starcoder_all_tied_curves_canonical_dsp_20260902"
pytestmark = pytest.mark.skipif(not ATLAS.exists(), reason="StarCoder atlas absent")
# Per-run launch results of three atlas families, which the export does not read.
RUN_RECORDS = {
    "dense_horizon_replay": (
        paths.RO / "starcoder_wsd80_dense_support_calibration_results_20260813/coverage_with_calibration_weights.csv"
    ),
    "coupled_lr_onset": paths.RO / "starcoder_wsd80_coupled_onset_dense_surface_results_20260901/observations.csv",
    "matched_nd": (
        paths.RO
        / "starcoder_wsd80_matched_nd_stage1_20260731/stage3_dense_surface_results_20260802"
        / "combined_discovery_observations.csv"
    ),
}
# The W&B run id of a family's run is its run name after a prefix: the coupled learning-rate-onset launch put `c2v4c_`
# before every run name (its deployment in launch_starcoder_wsd80_coupled_onset_dense_surfaces.py).
WANDB_PREFIX = {"dense_horizon_replay": "", "coupled_lr_onset": "c2v4c_", "matched_nd": ""}
TRANSITION = paths.RO / "starcoder_wsd80_transition_transient_20261006"
# The heteroskedasticity analysis's per-mixture summary of the repeat-noise panel, which the paper's table reports.
NOISE_SUMMARY = paths.RO / "starcoder_heteroskedasticity_mechanisms_20260617/anchor_metric_noise_features.csv"
# The paper builder's record of Figure 1's surrogate panel, with the share at which it places each design-grid weight.
FIGURE1_RECORD = paths.DRIVE / "figure1_real_data_drafts/v43/sources/response_subset.json"


@pytest.fixture(scope="module")
def out():
    return starcoder.export()


def test_observations_and_membership(out):
    obs, mem = out["starcoder_observations"], out["starcoder_curve_membership"]
    assert len(obs) == 954 and obs.observation_id.is_unique
    assert len(mem) == 1058 and set(mem.observation_id) <= set(obs.observation_id)
    assert mem.groupby("curve_id").size().sum() == 1058
    assert obs[["starcoder_weight", "observed_bpb", "starcoder_epochs"]].notna().all().all()


def test_fits_and_grid(out):
    assert len(out["starcoder_curves"]) == 45
    assert len(out["starcoder_fit_parameters"]) == 990
    grid = out["starcoder_dense_grid"]
    assert len(grid) == 45045 and grid.starcoder_weight.between(0, 1).all()
    assert set(out["starcoder_curves"].columns) >= {"curve_id", "family", "curve_label", "fitter", "grid_points"}


def test_trajectories_cover_the_switch_figure(out):
    traj = out["starcoder_trajectories"]
    assert {"run_label", "seed", "step", "train_loss"} <= set(traj.columns)
    assert traj.groupby(["run_label", "seed"]).step.max().ge(22608).all()
    assert traj.groupby(["run_label", "seed"]).size().ge(1000).all()


def test_repeat_noise_panel_reproduces_the_analysis(out):
    panel = out["starcoder_repeat_noise"]
    assert len(panel) == 50 and panel.observation_id.is_unique and panel.data_seed.is_unique
    mixtures = panel.groupby("mixture_label")
    assert len(mixtures) == 10 and mixtures.size().eq(5).all()
    assert mixtures[["phase_0_starcoder", "phase_1_starcoder"]].nunique().eq(1).all().all()
    summary = pd.read_csv(NOISE_SUMMARY)
    summary = summary[summary.metric == "eval/paloma/dolma_100_programing_languages/bpb"].set_index("anchor_id")
    stats = mixtures.pl_bpb.agg(["mean", "std"])
    for column in ("mean", "std"):
        assert np.allclose(stats[column], summary.loc[stats.index, column], rtol=0, atol=1e-12)


def test_observations_match_their_runs_launch_records(out):
    # Model size, tokens, StarCoder subset and seeds against each run's own launch record, for the 856 runs of the
    # three families that keep one; each run is keyed by its W&B run id.
    obs = out["starcoder_observations"].set_index("observation_id")
    family = out["starcoder_curve_membership"].merge(out["starcoder_curves"][["curve_id", "family"]], on="curve_id")
    family = family.drop_duplicates("observation_id").set_index("observation_id").family
    checked = 0
    for name, source in RUN_RECORDS.items():
        runs = pd.read_csv(source)
        if "is_alias" in runs:
            runs = runs[~runs.is_alias]  # an alias row names the full-pool run it reuses
        runs.index = WANDB_PREFIX[name] + runs.run_name
        ids = family.index[family == name]
        mine, theirs = obs.loc[ids], runs.loc[ids]
        assert (mine.total_parameters == theirs.total_parameters).all()
        assert (mine.non_embedding_parameters == theirs.non_embedding_parameters).all()
        assert (mine.train_tokens == theirs.materialized_tokens).all()
        assert (mine.data_seed == theirs.data_seed).all()
        assert np.allclose(mine.starcoder_weight, theirs.phase_1_starcoder, rtol=0, atol=1e-12)
        if "trainer_seed" in theirs:
            assert (mine.trainer_seed == theirs.trainer_seed).all()
        if "starcoder_realized_support_tokens" in theirs:
            assert (mine.starcoder_subset_tokens == theirs.starcoder_realized_support_tokens).all()
        checked += len(ids)
    assert checked == 856


def test_runs_shared_by_curves_trained_on_the_full_pool(out):
    # A full-pool run that never exhausts a smaller subset stands in for it; the run keeps the full pool's design.
    obs = out["starcoder_observations"].set_index("observation_id")
    curves = out["starcoder_curves"].set_index("curve_id")
    mem = out["starcoder_curve_membership"]
    shared = mem[mem.duplicated("observation_id", keep=False)]
    assert shared.observation_id.nunique() == 52 and len(shared) == 156
    subsets = shared.curve_id.map(curves.starcoder_subset_tokens).groupby(shared.observation_id).max()
    assert (obs.starcoder_subset_tokens[subsets.index] == subsets).all()
    full_pool = curves.starcoder_subset_tokens.max()
    assert (subsets == full_pool).all()
    # A full-pool run's epochs are its weight times D over the pool (from the atlas's epochs at full share, which
    # curve_reference.csv rounds to 1e-6); it uses under one epoch of each subset it stands in for, on the atlas's
    # coordinate and on the realized subset.
    rows = obs.loc[subsets.index]
    assert np.allclose(rows.starcoder_epochs, rows.starcoder_weight * rows.train_tokens / full_pool, rtol=0, atol=5e-7)
    assert shared.starcoder_epochs.lt(1).all()
    exposure = shared.observation_id.map(obs.starcoder_weight * obs.train_tokens)
    assert (exposure / shared.curve_id.map(curves.starcoder_subset_tokens)).lt(1).all()


def test_trajectories_hold_every_logged_step(out):
    # train/loss is logged at every step from 2 to the run's last (28,259; 28,159 for the H5 runs); none is thinned.
    steps = out["starcoder_trajectories"].groupby(["observation_id", "run_label", "seed"]).step
    assert steps.ngroups == 50 and steps.min().eq(2).all() and steps.max().isin([28159, 28259]).all()
    assert (steps.nunique() == steps.size()).all() and (steps.max() - steps.min() + 1 == steps.size()).all()


def _jump(series: pd.Series, switch: int) -> tuple[float, float, int]:
    """train_loss_jump.csv's per-run statistics: the mean loss over the first 8 and 64 steps from the switch minus
    the mean over the 512 steps before it, and the first step at which the 8-step running mean falls below that."""
    pre = series.loc[switch - 512 : switch - 1].mean()
    running = series.loc[switch : switch + 2047].rolling(8).mean()
    below = running[running < pre]
    duration = int(below.index[0] - switch + 1) if len(below) else 9999
    return series.loc[switch : switch + 7].mean() - pre, series.loc[switch : switch + 63].mean() - pre, duration


def test_trajectories_reproduce_the_training_loss_jumps(out):
    # The re-pulled series give the jumps and durations that the analysis computed from the original pull.
    traj = out["starcoder_trajectories"]
    transition = out["starcoder_transition"]
    jumps = transition[transition.table == "train_loss_jump"].set_index("arm")
    assert set(traj.run_label) <= set(jumps.index)
    for label, runs in traj.groupby("run_label"):
        stats = [
            _jump(run.set_index("step").train_loss.astype(float), run.switch_step.iloc[0])
            for _, run in runs.groupby("seed")
        ]
        jump8, jump64, duration = (np.array(values) for values in zip(*stats, strict=True))
        expected = jumps.loc[label]
        assert len(stats) == expected.n
        assert jump8.mean() == pytest.approx(expected.jump8, rel=0, abs=1e-12)
        assert jump64.mean() == pytest.approx(expected.jump64, rel=0, abs=1e-12)
        assert (np.median(duration), duration.max()) == (expected.dur_median, expected.dur_max)


def test_required_sources_are_the_files_the_export_reads(monkeypatch):
    # The skip guard above and the manifest's provenance rely on this list.
    read = set()
    require = paths.require
    monkeypatch.setattr(paths, "require", lambda path: read.add(path) or require(path))
    starcoder.export()
    assert read == set(starcoder.required_sources())


def test_trajectories_key_on_the_w_and_b_runs_they_were_pulled_from(out):
    # pull_switch_traces.py read each series from its W&B run: a confirmation run under its run name, a
    # gradient-conflict trajectory as gcfv9_<trajectory id>.
    pulled = pull_switch_traces.series()
    pulled = dict(zip(zip(pulled.run_label, pulled.seed.astype(int), strict=True), pulled.wandb_id, strict=True))
    traj = out["starcoder_trajectories"].drop_duplicates(["run_label", "seed"])
    keyed = dict(zip(zip(traj.run_label, traj.seed.astype(int), strict=True), traj.observation_id, strict=True))
    assert len(keyed) == 50 and keyed == pulled


def test_transition_runs_share_ids_with_the_trajectories(out):
    transition = out["starcoder_transition"]
    sources = {name: len(pd.read_csv(TRANSITION / f"{name}.csv")) for name in starcoder.TRANSITION_TABLES}
    assert transition.table.value_counts().to_dict() == sources
    evals = transition[transition.table == "switch_window_offline_evals"]
    traj = out["starcoder_trajectories"].drop_duplicates("observation_id")
    for label, arm in {"conf_m400_untied": "august_abrupt", "h5_boundary_beta_0p60": "boundary_beta_0p60"}.items():
        assert set(traj.observation_id[traj.run_label == label]) == set(evals.observation_id[evals.arm == arm])
    p3 = evals[(evals.arm == "p3") & (evals.support_id == "m100a")]
    assert set(traj.observation_id[traj.run_label == "p3_m100a"]) == set(p3.observation_id)
    # Each ramp run's final checkpoint is evaluated once offline and has a training-time evaluation history.
    ramps = transition[transition.table == "ramp_endpoints"]
    history = transition[transition.table == "ramp_eval_history"]
    assert set(ramps.observation_id) == set(history.observation_id) <= set(evals.observation_id)


def test_diagonal_cells_and_curricula(out):
    diagonal = out["starcoder_diagonal"]
    assert len(diagonal) == 60 and diagonal.observation_id.is_unique
    assert diagonal.groupby("cell_id").size().eq(15).all() and diagonal.cell_id.nunique() == 4
    # Each cell's summary is its best measured run.
    best = diagonal.loc[diagonal.groupby("cell_id").starcoder_bpb.idxmin()]
    assert (best.observed_optimum_bpb == best.starcoder_bpb).all()
    assert (best.observed_optimum_weight == best.starcoder_weight).all()
    curricula = out["starcoder_curricula"]
    assert len(curricula) == 28 and not curricula.duplicated(["cell_id", "support_id"]).any()


def test_epoch_matching_reproduces_figure2(out):
    # Figure 2's selections and printed numbers from the table alone. Each proxy curve averages its trainer seeds; the
    # unmatched runs at weight 0 are also every matched curve's point at 0.
    sweep = out["starcoder_epoch_matching"].assign(percent=lambda t: (100 * t.starcoder_weight).round(6))
    assert len(sweep) == 103 and sweep.observation_id.is_unique
    zero = sweep[(sweep.arm != "target") & (sweep.percent == 0)]
    assert len(zero) == 2 and (zero.arm == "unmatched").all() and zero.subset_seed.isna().all()
    unmatched = sweep[sweep.arm == "unmatched"].groupby("percent").programming_languages_bpb.mean()
    assert unmatched.idxmin() == 100
    tracks = []
    for _, runs in sweep[sweep.arm == "matched"].groupby("subset_seed"):
        curve = pd.concat([runs, zero])
        assert curve.groupby("percent").programming_languages_bpb.mean().idxmin() == 50
        tracks += [
            seed.set_index("percent").programming_languages_bpb.idxmin() for _, seed in curve.groupby("trainer_seed")
        ]
    # The six track minima of the follow-up's plan, whose mean, 8/15, the follow-up target run trains.
    assert tracks == [50, 50, 65, 50, 50, 55] and np.mean(tracks) == pytest.approx(100 * 8 / 15, abs=1e-9)
    target = sweep[sweep.arm == "target"].set_index("percent").programming_languages_bpb
    grid = target[target.index.isin(unmatched.index)]
    low, matched, naive = grid.min(), target[round(100 * 8 / 15, 6)], grid[100]
    assert grid.idxmin() == 70
    assert [f"{v:.3f}" for v in (matched, naive, low)] == ["0.774", "0.801", "0.766"]
    assert f"{100 * (1 - (matched - low) / (naive - low)):.1f}" == "74.7"  # "74.7% less excess loss"
    # The epochs printed beside the selections and above p = 0.6, 0.8 and 1 on the top axes, and the panels' FLOPs.
    epochs = sweep.groupby(["arm", "percent"]).starcoder_epochs
    assert epochs.nunique().eq(1).all()
    epochs = epochs.first()
    assert (f"{epochs['target', round(100 * 8 / 15, 6)]:.1f}", f"{epochs['unmatched', 100]:.2f}") == ("8.5", "0.87")
    rows = {"matched": (1, "9.5 12.7 15.8"), "unmatched": (2, "0.52 0.70 0.87"), "target": (1, "9.5 12.7 15.8")}
    for arm, (digits, printed) in rows.items():
        assert " ".join(f"{epochs[arm, p]:.{digits}f}" for p in (60, 80, 100)) == printed
    flops = sweep.groupby(sweep.arm == "target").compute_flops  # proxies, then the target
    assert flops.nunique().eq(1).all() and [f"{f:.2e}" for f in flops.first()] == ["2.49e+16", "6.66e+18"]


def test_epoch_matching_web_pool_gives_figure1s_web_epochs(out):
    # Figure 1's surrogate panel fits MARINER with Nemotron-CC at its finite-pool epochs, (1 - share) x tokens / pool;
    # its builder records 0.04558802099128514 epochs per unit weight for the proxies. The design sized the pool for the
    # target, so no run trains on it for a whole epoch.
    sweep = out["starcoder_epoch_matching"]
    assert sweep.web_pool_tokens.dtype == np.int64
    proxies = sweep[sweep.arm != "target"]
    assert np.allclose(proxies.train_tokens / proxies.web_pool_tokens, 0.04558802099128514, rtol=0, atol=1e-15)
    assert ((1 - sweep.starcoder_weight) * sweep.train_tokens).lt(sweep.web_pool_tokens).all()


@pytest.mark.skipif(not FIGURE1_RECORD.exists(), reason="paper folder absent")
def test_epoch_matching_grid_gives_figure1s_realized_shares(out):
    # Figure 1's surrogate panel maps every weight of the design's grid to the share that the proxies' sampler realized.
    # At the 12 trained weights, the grid's counts are the ones behind the runs' materialized epochs.
    grid = out["starcoder_epoch_matching_grid"]
    mapping = json.loads(FIGURE1_RECORD.read_text())["track_metadata"]["fraction_mapping"]
    matched = grid[grid.arm == "matched"]
    assert matched.starcoder_weight.tolist() == mapping["requested"]
    assert np.allclose(matched.realized_starcoder_fraction, mapping["actual"], rtol=0, atol=1e-15)
    trained = out["starcoder_epoch_matching"].merge(grid, on=["arm", "starcoder_weight"], validate="many_to_one")
    assert len(trained) == 102  # every run but the follow-up at 8/15, which is off the grid
    realized = trained.starcoder_epochs * trained.starcoder_subset_tokens / trained.train_tokens
    assert np.allclose(realized, trained.realized_starcoder_fraction, rtol=0, atol=1e-15)


@pytest.mark.skipif(
    not all(path.exists() for path in domain_sweeps.required_sources()), reason="domain sweep sources absent"
)
def test_epoch_matching_web_only_runs_are_finemath_weight0_rows(out):
    # One run, one id: FineMath's weight-0 rows are two of the sweep's runs, evaluated again on the same set later.
    sweeps = domain_sweeps.export()["domain_sweeps"]
    shared = sweeps[sweeps.observation_id.isin(out["starcoder_epoch_matching"].observation_id)]
    finemath = shared[shared.evaluation == "paloma_bpb"].set_index("observation_id")
    assert set(shared.domain) == {"finemath"} and len(finemath) == 2 and (finemath.domain_weight == 0).all()
    runs = out["starcoder_epoch_matching"].set_index("observation_id").loc[finemath.index]
    assert (runs.trainer_seed == finemath.trainer_seed).all() and (runs.train_tokens == finemath.train_tokens).all()
    assert (runs.total_parameters == finemath.model_total_parameters).all()
    assert np.allclose(runs.programming_languages_bpb, finemath.bpb, rtol=0, atol=1.3e-6)


def test_curve_compute_prints_as_figure3_and_follows_levanter(out):
    # Figure 3's insets print each row's FLOPs as f"{flops / 1e18:.2g}e18": panels A (fixed D) and B (fixed N) read
    # atlas curves, panel C the diagonal's cells.
    curves = out["starcoder_curves"].set_index("curve_ref")
    assert curves.compute_flops.notna().all()
    panels = (
        curves.compute_flops[["C05", "C06", "C07", "C08"]],
        curves.compute_flops[["C05", "C10", "C13", "C14"]],
        out["starcoder_diagonal"].drop_duplicates("cell_id").sort_values("rung").compute_flops,
    )
    for panel in panels:
        assert [f"{flops / 1e18:.2g}" for flops in panel] == ["0.88", "1.7", "3.4", "6.5"]
    # The fixed-model ladder's FLOPs, logged to W&B, are Levanter's count for the Llama model its design describes
    # (10 layers of width 768, MLP width 1,536, 8 heads) with the Llama 3 vocabulary at sequence length 2,048.
    per_token = lm_flops_per_token(
        hidden_dim=768,
        intermediate_dim=1536,
        num_layers=10,
        num_kv_heads=8,
        num_heads=8,
        seq_len=2048,
        vocab_size=128256,
        glu=True,
    )
    ladder = curves[curves.family == "fixed_model_token_ladder"]
    assert len(ladder) == 4 and np.allclose(
        ladder.compute_flops, 3 * per_token * ladder.train_tokens, rtol=1e-12, atol=0
    )
    # Curves that train the same model on the same tokens cost the same.
    assert (
        curves.groupby(["total_parameters", "non_embedding_parameters", "train_tokens"])
        .compute_flops.nunique()
        .eq(1)
        .all()
    )


def test_figure25_reads_both_double_difference_tables(out):
    # Figure 25 (a_switch_smoothing) reads both files, which keep every row under their own table name.
    transition = out["starcoder_transition"]
    tables = {}
    for name in ("eval_double_differences", "switch_window_offline_double_differences"):
        source = pd.read_csv(TRANSITION / f"{name}.csv")
        tables[name] = transition[transition.table == name][list(source.columns)].reset_index(drop=True)
        pd.testing.assert_frame_equal(tables[name], source, check_dtype=False)
    # Panel A: the 0.02 -> 0.82 switch at the decay onset, offline points (no note) over the training-time ones.
    columns = ["arm", "metric", "lag", "double_diff", "double_diff_ci"]
    offline = tables["switch_window_offline_double_differences"]
    offline = offline[offline.note.isna()].rename(columns={"ci": "double_diff_ci"})
    both = pd.concat([tables["eval_double_differences"][columns], offline[columns]])
    both = both.drop_duplicates(["arm", "metric", "lag"], keep="last")
    switch = both[(both.arm == "p3_vs_p4_m100a") & both.lag.between(-256, 1500)]
    for metric in ("PL", "C4"):
        assert sorted(switch.lag[switch.metric == metric]) == [-256, 0, 64, 256, 392, 1392]


def test_figure24_matched_nd_tables(out):
    optima = out["starcoder_curricula_matched_nd"]
    assert len(optima) == len(pd.read_csv(starcoder.MATCHED_OPTIMA)) == 10
    # Panel C draws four cells along each path: more tokens, larger models, and both.
    for track in ("increase_d", "increase_n", "increase_nd"):
        assert optima.track_memberships.str.contains(f"'{track}'", regex=False).sum() == 4
    pairs = out["starcoder_curricula_matched_nd_confirmation"]
    assert len(pairs) == len(pd.read_csv(starcoder.MATCHED_CONFIRMATION / "confirmation_pairs.csv")) == 8
    assert pd.concat([pairs.candidate_observation_id, pairs.comparator_observation_id]).is_unique
    # Panel C's star, which the builder prints: the mean gain over the eight pairs and its 95% paired-t interval.
    gain = pairs.gain_tied_minus_untied_bpb
    half = stats.t.ppf(0.975, len(gain) - 1) * gain.std(ddof=1) / np.sqrt(len(gain))
    star = [gain.mean(), gain.mean() - half, gain.mean() + half]
    assert np.allclose(star, [0.0061012, 0.0052691, 0.0069332], rtol=0, atol=5e-8)


def test_figure26_ramp_designs(out):
    designs = out["starcoder_ramp_designs"]
    sources = [json.loads(path.read_text()) for path in starcoder.RAMP_DESIGNS]
    assert len(designs) == sum(len(run["stages"]) for design in sources for run in design["runs"]) == 2740
    # One design for each of the 30 ramp and abrupt runs, whose sources and comparators are the August runs.
    runs = designs.drop_duplicates("observation_id")
    transition = out["starcoder_transition"]
    endpoints = transition[transition.table == "ramp_endpoints"]
    assert len(runs) == 30 and set(runs.observation_id) == set(endpoints.observation_id)
    evals = transition[transition.table == "switch_window_offline_evals"]
    assert set(runs.source_observation_id) == set(evals.observation_id[evals.arm == "august_abrupt"])
    assert set(runs.tied_observation_id) == set(evals.observation_id[evals.arm == "august_tied"])
    # The figure draws the six schedules of seed 20260821.
    shown = runs[runs.data_seed == 20260821]
    assert sorted(shown.arm) == ["abrupt", "ramp0128", "ramp0256", "ramp0512", "ramp2048", "ramp5632"]


def test_ramp_designs_carry_the_loaders_starcoder_counts(out):
    # Figure 27 plots the StarCoder share of each 2,048-sequence block as the loader allocated it. The ramps keep each
    # bucket's total: every run draws as many StarCoder sequences as the abrupt run of its data seed, and its full
    # blocks hold 204,387 of them. A centred ramp moves some of them to just before the decay onset.
    designs = out["starcoder_ramp_designs"]
    runs = designs.groupby("observation_id")
    first, totals = runs.first(), runs.starcoder_stage_sequences.sum()
    assert (totals == first.realized_starcoder_sequences).all() and (totals == first.target_starcoder_sequences).all()
    abrupt = first[first.arm == "abrupt"].set_index("data_seed").realized_starcoder_sequences
    assert len(abrupt) == 5 and (totals == first.data_seed.map(abrupt)).all()
    # A stage's full blocks end at the next stage's first step, or at training's last full block.
    last_block = designs.total_steps - designs.total_steps % designs.steps_per_block
    stop = runs.start_step.shift(-1).fillna(last_block)
    blocks = (stop - designs.start_step) // designs.steps_per_block
    assert set((blocks * designs.starcoder_block_sequences).groupby(designs.observation_id).sum()) == {204_387}
    before = (
        designs[designs.start_step < designs.boundary_step].groupby("observation_id").starcoder_stage_sequences.sum()
    )
    moved = before.groupby(first.arm).agg(set).to_dict()
    assert moved == {
        "abrupt": {0},
        "ramp0128": {581},
        "ramp0256": {1163},
        "ramp0512": {2327},
        "ramp2048": {9312},
        "ramp5632": {25622},
    }
    # The loader's counts are not the design weights times the block: they differ by up to 4.5 sequences.
    gap = (designs.starcoder_block_sequences - designs.starcoder_weight * 2048).abs()
    assert 1 < gap.max() < 4.5


def test_no_internal_strings_anywhere(out):
    # Every string a table holds: its column names and the values of its non-numeric columns.
    patterns = [
        re.compile(r"gs://|http|/Users/|pinlin_calvin_xu"),
        re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),  # email addresses
        re.compile(r"[0-9a-f]{64}"),
        re.compile(r"(?:exp|run|job)_01[a-z0-9]{24}"),  # Fieldbook ids
    ]
    for name, t in out.items():
        strings = set(t.columns)
        for column in t.columns:
            if not pd.api.types.is_numeric_dtype(t[column]):
                strings |= set(t[column].dropna().astype(str))
        text = "\n".join(sorted(strings))
        for pattern in patterns:
            assert not pattern.search(text), (name, pattern.pattern)
