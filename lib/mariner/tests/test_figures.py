# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mariner.cli import INPUT_ERROR_STATUS, PREDICTION_TOLERANCE, main
from mariner.data import (
    FIGURES_INDEX,
    MAIN_SWARM,
    OBJECTIVES,
    objective_from_tasks,
    read_table,
    release_anchors,
    release_runs,
    release_swarm,
    release_tasks,
    released_olmix_laws,
    trained_counts,
)
from mariner.figures import FIGURES, write_figures
from mariner.figures.analysis import (
    COMMON_CRAWL,
    EPOCH_CAPS,
    FINEMATH,
    FLAN,
    PROXY,
    TARGET,
    epoch_text,
    matched_proxy,
    sweep_olmix_law,
)
from mariner.figures.epochs import Track
from mariner.figures.layout import OBJECTIVE_TITLES, Method
from mariner.figures.pipeline import ALL_TASKS, scaling_values
from mariner.figures.schedules import ComputePath
from mariner.figures.starcoder import EPOCH_MATCHING_TABLE, epoch_matching
from mariner.figures.surrogate import TRAINER_SEEDS
from mariner.figures.swarm import holm
from mariner.fit import fit_objective
from mariner.optimize import MIXTURE_BLOCK_SIZE

# figures_tables.csv marks configuration constants and the two-phase data the release omits among a figure's configs.
INDEX_MARKERS = {"constant", "deferred"}
TERMS = "fig:mariner-terms"  # the one figure that reads no table: mariner.model's terms at the paper's values
# Figure 3's inset tables print each sweep's training FLOPs and fitted optimum in materialized StarCoder epochs, by
# path and scale; the paths match compute at each scale (its caption: "within 3% across columns").
PRINTED_FLOPS = ["8.8e+17", "1.7e+18", "3.4e+18", "6.5e+18"]
PRINTED_OPTIMUM_EPOCHS = {
    Track.FIXED_TOKENS: [8.8, 8.3, 8.0, 7.7],
    Track.FIXED_PARAMETERS: [8.8, 11.1, 14.0, 17.5],
    Track.FIXED_TPP: [9.2, 9.7, 10.9, 12.5],
}
COMPUTE_MATCH = 0.03
# And each path's optimum-share law, p* = coefficient (q / q0)^exponent with q0 its smallest scale to three digits.
PRINTED_SHARE_LAWS = {
    Track.FIXED_TOKENS: (0.332, -0.078),
    Track.FIXED_PARAMETERS: (0.335, 0.338),
    Track.FIXED_TPP: (0.340, 0.326),
}
# Figure 5's harm legend: each threshold's activation, e^tau - 1 epochs, to two significant digits.
PRINTED_ACTIVATIONS = {1.0: "1.7", 2.0: "6.4", 3.0: "19"}
SOFTPLUS_AT_ZERO = math.log(2.0)
# The plotted curve, linearly interpolated between its 1,500 log-spaced epochs, at an activation (measured 1.7e-6).
HARM_CURVE_TOLERANCE = 1e-5
# BPB by which the 1.00B-token sweep falls from its smallest share (sections/surrogate_properties.tex:22).
LOWEST_COMPUTE_DIP = 0.22
# The dataset recomputes the ladder's Uncheatable aggregate from its components, up to 7.7e-8 BPB from the means the
# paper fitted, which moves the fitted laws by up to 3.1e-7 (measured); the module's fit of the paper's own means gives
# the released laws exactly. The OlmoBaseEval Easy laws match the released ones within 4.4e-9 (measured).
LAW_TOLERANCE = 1e-6
# Figures 1 and 9's compute multiples of MARINER over Olmix and over proportional mixing at 3e18, 2e19, 3e20 and 1e21
# FLOPs: Uncheatable (Figure 1, Figure 9's left panel) and OlmoBaseEval Easy (Figure 9's right panel).
PRINTED_MULTIPLES = {"olmix": [1.3, 1.4, 1.5, 1.6], "proportional": [2.0, 2.4, 2.8, 3.2]}
PRINTED_OBE_EASY_MULTIPLES = {"olmix": [1.2, 1.3, 1.4, 1.3], "proportional": [2.8, 3.1, 4.8, 5.9]}
# sections/abstract.tex:1: "MARINER's mixtures require 69--83\% less final-training compute than proportional mixing
# to reach the same loss, and 26--38\% less than Olmix" at 1e21 FLOPs: the savings 1 - 1 / multiple (%) over the two
# objectives.
PRINTED_SAVINGS = {"proportional": [69, 83], "olmix": [26, 38]}
LARGEST_SCALE = 1e21
# tab:scaling-main's accuracy column (sections/results.tex:22-25), mean accuracy (%) over the 51 components at 1e21.
PRINTED_TABLE_ACCURACY = {"proportional": "30.1", "unimax8": "32.7", "olmix": "35.1", "mariner": "36.4"}
# tab:scaling-main's losses (sections/results.tex:22-25) of the mixtures Figure 10 draws on their own objective: the
# 3e18 mean and standard deviation, then 2e19, 3e20 and 1e21 FLOPs.
PRINTED_ON_OBJECTIVE = {
    ("uncheatable", "proportional"): ["1.038", "0.001", "0.902", "0.770", "0.721"],
    ("uncheatable", "olmix"): ["1.002", "0.001", "0.873", "0.745", "0.700"],
    ("uncheatable", "mariner"): ["0.982", "0.001", "0.854", "0.729", "0.686"],
    ("obe_easy", "proportional"): ["1.198", "0.004", "0.972", "0.754", "0.673"],
    ("obe_easy", "olmix"): ["1.092", "0.004", "0.892", "0.676", "0.607"],
    ("obe_easy", "mariner"): ["1.068", "0.005", "0.868", "0.656", "0.589"],
}
# Figure 11's All bars against the weighted sums of their rows' bars: they differ by floating rounding alone (measured
# 2.4e-16), since each run's aggregate and its proportional mean are those sums of their components.
AGGREGATE_TOLERANCE = 1e-12
# The rows of Figure 11 that sections/results.tex:53 names: "MARINER's full-objective mixtures do not improve over
# proportional mixing on AO3, BBC News, or Wikipedia on Uncheatable, or question answering on OlmoBaseEval Easy.
# MARINER's mixtures optimized for these components lower their loss below proportional mixing. Olmix's mixtures
# optimized for the same components lower it less. Both, however, raise the full-objective loss above proportional
# mixing."
TRADEOFF_ROWS = {
    "uncheatable": ("uncheatable/ao3_english", "uncheatable/bbc_news", "uncheatable/wikipedia_english"),
    "obe_easy": ("QA",),
}
# Figure 13's caption (sections/analysis.tex:29), "Proportional mixing gives every bucket 0.9 epochs", and the largest
# number it prints in each column, by objective and mixture.
PRINTED_PROPORTIONAL_EPOCHS = "0.9"
PRINTED_EPOCH_MAXIMA = {
    ("uncheatable", "MARINER"): "5.3",
    ("uncheatable", "Olmix, cap 4, KL 0.05"): "3.2",
    ("uncheatable", "Olmix, no cap, no KL"): "16",
    ("obe_easy", "MARINER"): "7.5",
    ("obe_easy", "Olmix, cap 4, KL 0.005"): "4.0",
    ("obe_easy", "Olmix, no cap, no KL"): "17",
}
# The tasks Figure 15 labels, by the names the redraw gives them; the paper prints them as "Basic Skills: pattern
# recognition", "CSQA" and "SciQ" (sections/analysis.tex:61: "labels identify tasks with negative Olmix $R^2$").
LABELED_TASKS = {"Basic Skills: Simple Pattern Recognition", "CSQA", "SciQ"}
# Figure 14's correlations over the 280 designs (its caption, sections/analysis.tex:41: "Top: Uncheatable BPB for 280
# swarm mixtures"), as its columns print them: Pearson r of their BPB and Spearman rho of their ranks, by (source,
# target) setting.
TRANSFER_DESIGNS = 280
PRINTED_TRANSFER_CORRELATIONS = {
    ("llama_160m_1p2b", "qwen3_360m_1p6b"): ("0.94", "0.93"),
    ("llama_160m_1p2b", "llama_200m_6b"): ("0.83", "0.80"),
    ("qwen3_360m_1p6b", "llama_200m_6b"): ("0.65", "0.59"),
}
# sections/analysis.tex:48: "the best swarm mixture at Qwen3 360M/1.6B still ranks second of 280 at Llama 200M/6B".
PRINTED_BEST_TRANSFER_RANK = 2
# Figure 16's Spearman correlations over each column's 30 best held-out mixtures, MARINER's and then Olmix's, as the
# paper's figure prints them since its rebuild from the released fits on 2026-11-07
# (revision_notes/20261107_fig16_released_fits/CHANGES.md in the paper's folder).
FRONTIER_MIXTURES = 30  # its caption (sections/analysis.tex:75): "the 30 lowest-loss held-out mixtures"
PRINTED_FRONTIER_SPEARMAN = {
    "Uncheatable": ("0.84", "0.71"),
    "OlmoBaseEval Easy": ("0.34", "0.18"),
    "Arithmetic": ("0.70", "0.59"),
    "arXiv physics": ("0.73", "0.59"),
    "SciQ": ("-0.33", "0.40"),
}
# Its titles' signal-to-noise ratios. The paper prints arXiv physics's as 14, dividing by the standard deviation of
# eleven proportional runs, the June run among them; fits_reliability's ten repeats give 16, which the redraw prints.
PRINTED_FRONTIER_SNR = {"Uncheatable": 28, "OlmoBaseEval Easy": 13, "Arithmetic": 6, "SciQ": 16}
# Figure 17 draws every held-out mixture of the two banks that has the component: 169 of the 170 Uncheatable ones (one
# has no Uncheatable components) and the 157 OlmoBaseEval Easy ones, as the paper's figure does since its rebuild
# (revision_notes/20261107_fig16_released_fits/CHANGES.md).
EXTRAPOLATION_POINTS = {"arXiv physics": 169, "SciQ": 157}
# sections/analysis.tex:8: "the matched proxies place both minima where the target runs do, and FineMath's optimal
# epoch count is about seven times FLAN's".
PRINTED_EPOCH_RATIO = 7
# The package's fits of Figure 12 against the dataset's domain_sweeps_olmix_fits, in BPB over the mixture fraction
# (measured 6.8e-8).
DOMAIN_FIT_TOLERANCE = 1e-6
# Mean OlmoBaseEval Easy accuracy at 1e21 FLOPs: the abstract's 30.1% and 36.4%, and Figure 1's MARINER bar labels.
PRINTED_ACCURACY = {("All", "proportional"): "30.1", ("All", "mariner"): "36.4"}
PRINTED_MARINER_GROUPS = {"Basic": "78.5", "Math": "9.9", "Code": "15.9", "MMLU": "36.8", "QA": "57.8"}
# The shares that Figures 1 and 2 mark with diamonds and a star, positions rather than printed text: the matched
# proxies' mean selection, 8/15, which the target's follow-up run trains in starcoder_epoch_matching; the unmatched
# proxy's pure StarCoder (sections/simulated_epoching.tex:17: "The unmatched proxy selects pure StarCoder"); and the
# target's optimum.
MARKED_SHARES = {"matched": 8 / 15, "unmatched": 1.0}
MARKED_OPTIMUM = 0.7
SHARE_TOLERANCE = 1e-12
PRINTED_EXCESS_AVOIDED = "75"  # Figure 1: "Excess loss avoided: 75%"
# Figure 2's labels: each selection's materialized epochs and target loss, the target's optimum loss, the share of
# excess loss avoided, the epochs its top axes print at p = 0.6, 0.8 and 1.0, and its titles' FLOPs.
PRINTED_SELECTIONS = {"matched": ("8.5", "0.774"), "unmatched": ("0.87", "0.801")}
PRINTED_OPTIMUM_BPB = "0.766"
PRINTED_AVOIDED = "74.7"
PRINTED_EPOCH_ROWS = {
    "matched": ["9.5", "12.7", "15.8"],
    "unmatched": ["0.52", "0.70", "0.87"],
    "target": ["9.5", "12.7", "15.8"],
}
PRINTED_COMPUTE = ("2.49e+16", "6.66e+18")
PROPOSAL_PREDICTION_GAP = 0.005  # BPB: "Its proposed mixtures measure within 0.005 BPB of its predictions" (abstract)
# Figure 1's surrogate panel shows MARINER's predicted loss falling and then rising (sections/surrogate.tex:40: "so a
# task's predicted loss can fall and then rise as a bucket is upweighted"), with its interior optimum at p = 0.578 of the
# 400-point grid (predicted_minimum_fraction 0.5781954887 in figure1_real_data_drafts/v40/
# figure1_real_data_verification.json); Olmix's monotone fit picking pure StarCoder (sections/abstract.tex:1: "common
# monotone surrogates cannot represent a task loss that first falls and then rises"); and boosted trees that "reach the
# 90% mean but stay flat past it, below the measured 100% mean" (figure1_real_data_drafts/v40/build_figure.py:14-15).
MARKED_MARINER_OPTIMUM = "0.578"
# Figures 18 and 19 (their captions, sections/appendix.tex:370 and :380): "Bands are 95% t-intervals over ten
# subset/fold draws", and Figure 19's "Hollow diamonds mark swarm sizes at which released RegMix's fits are constant":
# its fits are constant at the five smallest swarms, of 21 to 61 mixtures, on both objectives.
LEARNING_CURVE_DRAWS = 10
CONSTANT_FIT_MIXTURES = [21, 31, 41, 51, 61]
# Figure 18's caption (sections/appendix.tex:370): "MARINER's ablations select mixtures with similar regret, although
# they rank mixtures with different accuracy." On the full swarm every ablation's regret interval overlaps MARINER's,
# and these two ablations' Spearman intervals lie below MARINER's.
WORSE_RANKING_ABLATIONS = (Method.MARINER_CONVEX, Method.MARINER_ADDITIVE)
# sections/appendix.tex:263: the natural cubic spline "needs more runs than MARINER to reach the same out-of-fold rank
# correlation". The smallest swarm, in mixtures, at which each model's pooled Spearman reaches a level, 0.9 on
# Uncheatable and 0.8 on OlmoBaseEval Easy; and the same line's "The natural cubic spline ranks the full Qwen3
# 360M/1.6B swarm slightly better than MARINER".
SPEARMAN_LEVELS = {"uncheatable": 0.9, "obe_easy": 0.8}
FIRST_MIXTURES_AT_LEVEL = {Method.MARINER: 101, Method.SPLINE: 161}
# sections/appendix.tex:388: "At every baseline's endpoint except Olmix's, and at all ten midpoints, MARINER's
# prediction is closer to the measured loss than the baseline's own."
BLEND_MIDPOINTS = 10
# Olmix's released curves are its released per-task laws at each blend of the two proposals (measured 2.2e-16 BPB).
OLMIX_CURVE_TOLERANCE = 1e-12
# sections/appendix_analysis.tex:103: "Only two of the 39 removals change the OlmoBaseEval Easy mean detectably,
# although that mean varies across the swarm by 12.6 times its repeat standard deviation". The paper's Figure 23
# annotates the two, Stack-Edu and Stack-Edu FIM, 11 and 8, and 24 cells in all, 20 of them red.
REMOVALS = 39
PRINTED_MEAN_REMOVALS = {"Stack-Edu": "11", "Stack-Edu FIM": "8"}
PRINTED_MEAN_SNR = "12.6"
ANNOTATED_CELLS = 24
RAISING_CELLS = 20
# sections/appendix_analysis.tex:118: Holm's method's "first step requires $|T_{t,i}|>4.4$ with one run per removal";
# and :120: "only two of these 26 removals move any task detectably".
PRINTED_HOLM_THRESHOLD = "4.4"
DETECTABLE_TOPIC_REMOVALS = {"Art & design (H)", "Health (H)"}
# sections/appendix_analysis.tex:217 and Figure 24's caption (:222): the Uncheatable proposal exceeds the swarm's
# sampled range for "Common Crawl literature (H) and olmOCR PDFs HQ", and the OlmoBaseEval Easy proposal "stays within
# every bucket's sampled range"; the faint dots are "the 280 mixtures of the Qwen3 360M/1.6B swarm".
OUTSIDE_SUPPORT = {"uncheatable": {"Literature (H)", "olmOCR PDFs HQ"}, "obe_easy": set()}
SWARM_FIT_RUNS = 280
# Figure 22's legends print each token budget's minimizing StarCoder epochs, by the StarCoder subset's repetition
# relative to the target run's, from the smallest budget (1.00B tokens) to the largest (7.41B).
PRINTED_REPLAY_EPOCHS = {
    0.125: ["3.0", "3.3", "3.3", "3.3"],
    0.25: ["6.0", "5.3", "6.6", "6.6"],
    0.5: ["8.6", "10.6", "10.6", "11.9"],
    1.0: ["9.3", "10.6", "13.2", "18.5"],
    2.0: ["9.0", "13.2", "15.9", "21.2"],
    4.0: ["10.6", "14.9", "15.9", "21.2"],
}
# Its caption (sections/appendix.tex:414): "the shift is not monotone at every degree of downsampling". The minimizing
# epochs fall from one budget to the next only at 1/4x, from the smallest budget to the second: (repetition, step).
REPLAY_EPOCH_FALLS = [(0.25, 0)]
# Figure 25's caption (sections/appendix_analysis.tex:307): Holm correction "over the 28 blocks" of token budget and
# StarCoder pool, each measured on "five fresh paired seeds", and one schedule pair measured "over eight fresh paired
# seeds".
CURRICULA_BLOCKS = 28
BLOCK_PAIRS = 5
CONFIRMATION_PAIRS = 8
# The confirmed cell's mean gain and 95% paired-t interval over its eight pairs (1e-3 BPB), which Figure 25C's star
# plots: a normal quantile or a population standard deviation would give a narrower interval.
CONFIRMATION_INTERVAL = ("6.101", "5.269", "6.933")
# sections/appendix_analysis.tex:302: "A cell went to a fresh-seed test only if its first 23 schedules gave a
# single-seed gain of at least 0.005 BPB."
FRESH_SEED_GAIN = 0.005
# Figure 26's caption (sections/appendix_analysis.tex:358): panel A's switch "from 0.02 to 0.82 StarCoder when the
# learning-rate decay begins", its gaps "relative to the same gap 64 steps before the switch" over "eight seeds", and
# panel B's "five paired seeds".
SWITCH_SHARES = (0.02, 0.82)
SWITCH_LR_STATE = "decay onset"  # starcoder_transition's name for the start of the learning-rate decay
# The same caption: the switch is "at the target run's repetition", the StarCoder subset that starcoder_transition names
# m100a, against single-phase runs of the schedule's average share, 0.8 x 0.02 + 0.2 x 0.82 = 0.18 ("vs tied 0.18").
SWITCH_POOL = "m100a"
FIRST_PHASE = 0.8
SWITCH_BASELINE_LEAD = 64
SWITCH_PAIRS = 8
RAMP_PAIRS = 5
# sections/appendix_analysis.tex:362: "Ramps of up to 512 steps changed the mean final programming-languages BPB by at
# most 0.0005, and the 2,048- and 5,632-step ramps raised it in every seed."
NARROW_RAMP_WIDTH = 512
NARROW_RAMP_CHANGE = 0.0005
WIDE_RAMPS = 2
# Figure 27's caption (sections/appendix_analysis.tex:368): "each mixture block of 2,048 sequences (16 training steps)",
# and "All ramps are centred on the start of the decay, 80% of the way through training."
BLOCK_SEQUENCES = 2048
BLOCK_STEPS = 16
DECAY_START = 0.8
# sections/appendix_analysis.tex:362: the selected schedule "raises StarCoder from 0 to 0.28 when the decay begins".
PRINTED_DECAY_SHARE = "0.28"


def epoch_label(value: float) -> str:
    return f"{value:.2f}" if value < 1 else f"{value:.1f}"


def test_write_figures_writes_the_named_figure_reproducibly(tmp_path):
    out = tmp_path / "out"
    paths = write_figures(tmp_path, out, [TERMS])
    assert paths == [out / f"{FIGURES[TERMS].stem}.pdf"]
    assert [path.name for path in out.iterdir()] == [paths[0].name]
    written = paths[0].read_bytes()
    assert written.startswith(b"%PDF")
    assert b"CreationDate" not in written
    # The PDFs carry no creation date, so the committed figures can be regenerated byte for byte.
    write_figures(tmp_path, out, [TERMS])
    assert paths[0].read_bytes() == written


def test_write_figures_names_an_unknown_label(tmp_path):
    # fig:b5 comes first and would fail to read the empty snapshot if anything were drawn before the check.
    with pytest.raises(ValueError, match="fig:nope"):
        write_figures(tmp_path, tmp_path, ["fig:b5", "fig:nope"])
    assert list(tmp_path.iterdir()) == []


def test_cli_figures_exits_nonzero_on_unknown_label(tmp_path, capsys):
    assert main(["figures", "--data", str(tmp_path), "--out", str(tmp_path), "--only", "fig:nope"]) != 0
    assert "fig:nope" in capsys.readouterr().err


def test_mariner_terms_mark_each_harm_activation_at_the_papers_epochs(tmp_path):
    """Figure 5's caption: "dots mark activations at E = e^tau - 1 epochs" on each harm curve."""
    values = FIGURES[TERMS].draw(tmp_path).values
    harm = values["harm"]
    for activation in values["activations"].itertuples():
        assert f"{activation.epochs:.2g}" == PRINTED_ACTIVATIONS[activation.tau]
        # The squared softplus activates where its argument log(1 + E) - tau crosses zero.
        assert activation.harm == pytest.approx(SOFTPLUS_AT_ZERO**2, rel=1e-12)
        curve = harm[harm["tau"] == activation.tau]
        on_curve = np.interp(np.log(activation.epochs), np.log(curve["epochs"]), curve["harm"])
        assert on_curve == pytest.approx(activation.harm, abs=HARM_CURVE_TOLERANCE)


def small_epoch_matching() -> pd.DataFrame:
    """A two-share sweep: two trainer seeds of each proxy and one matched subset, whose share-0 runs are the unmatched
    proxy's, and the target at both shares."""
    rows = []
    for seed in (1, 2):
        rows.append({"observation_id": f"u0_{seed}", "arm": "unmatched", "trainer_seed": seed, "starcoder_weight": 0.0})
        rows.append({"observation_id": f"u1_{seed}", "arm": "unmatched", "trainer_seed": seed, "starcoder_weight": 1.0})
        rows.append(
            {
                "observation_id": f"m1_{seed}",
                "arm": "matched",
                "subset_seed": 7,
                "trainer_seed": seed,
                "starcoder_weight": 1.0,
            }
        )
    rows += [
        {"observation_id": f"t{share:g}", "arm": "target", "trainer_seed": 1, "starcoder_weight": share}
        for share in (0.0, 1.0)
    ]
    runs = pd.DataFrame(rows)
    return runs.assign(
        starcoder_epochs=runs["starcoder_weight"],
        programming_languages_bpb=1.0 + runs.index / 100,
        train_tokens=100,
        starcoder_subset_tokens=100,
        compute_flops=1e16,
    )


def write_starcoder_table(root: Path, name: str, frame: pd.DataFrame) -> None:
    """A snapshot at ``root`` of the one StarCoder table ``name``, as CSV."""
    table = Path("data/starcoder") / f"{name}.csv"
    (root / table).parent.mkdir(parents=True)
    frame.to_csv(root / table, index=False)
    entry = {"table": name, "path": table.as_posix(), "format": "csv"}
    (root / "MANIFEST.json").write_text(json.dumps({"tables": [entry]}))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda runs: pd.concat([runs, runs.iloc[[2]].assign(observation_id="m1_1_again")]), "repeats runs"),
        (lambda runs: runs[runs["observation_id"] != "m1_2"], "lacks a run"),
    ],
    ids=["repeated", "missing"],
)
def test_epoch_matching_refuses_repeated_or_missing_runs(tmp_path, edit, message):
    write_starcoder_table(tmp_path, EPOCH_MATCHING_TABLE, edit(small_epoch_matching()))
    with pytest.raises(ValueError, match=message):
        epoch_matching(tmp_path)


def test_cli_figures_names_the_column_a_snapshot_lacks(tmp_path, capsys):
    """A snapshot whose starcoder_ramp_designs predates the data loader's per-block StarCoder counts cannot draw
    Figure 27: the command stops with the input-error status and names the missing column."""
    designs = pd.DataFrame(
        {
            "arm": ["abrupt", "abrupt"],
            "ramp_width": [math.nan, math.nan],
            "data_seed": [1, 1],
            "stage": [0, 1],
            "start_step": [0, 80],
            "total_steps": [100, 100],
            "boundary_step": [80, 80],
            "steps_per_block": [16, 16],
            "starcoder_weight": [0.0, 0.3],
        }
    )
    write_starcoder_table(tmp_path, "starcoder_ramp_designs", designs)
    argv = ["figures", "--data", str(tmp_path), "--out", str(tmp_path / "out"), "--only", "fig:a-ramp-schedules"]
    assert main(argv) == INPUT_ERROR_STATUS
    assert "starcoder_block_sequences" in capsys.readouterr().err


def test_holm_scales_each_p_value_by_the_hypotheses_left():
    """Holm's step-down correction, which Figure 23 applies: the p-values in increasing order times 4, 3, 2 and 1,
    then made non-decreasing in that order. Bonferroni's correction would give 0.12, 0.04, 0.16 and 0.02."""
    p_values = np.array([0.03, 0.01, 0.04, 0.005])
    np.testing.assert_allclose(holm(p_values), [0.06, 0.03, 0.06, 0.02], rtol=0, atol=1e-15)


def check_pipeline(values: dict[str, pd.DataFrame]) -> None:
    """Figure 1: its repetition panel's "Excess loss avoided: 75%" and the shares it marks, its surrogate panel's
    optima and trees, its compute multiples, and the abstract's accuracies (sections/abstract.tex:1: "from
    proportional mixing's 30.1\\% to 36.4\\%")."""
    assert f"{100 * values['summary']['excess_loss_avoided'].iloc[0]:.0f}" == PRINTED_EXCESS_AVOIDED
    selections = values["selections"].set_index("arm")["share"]
    for arm, share in MARKED_SHARES.items():
        assert selections[arm] == pytest.approx(share, abs=SHARE_TOLERANCE)
    assert values["target_optimum"]["share"].iloc[0] == MARKED_OPTIMUM
    optima = values["surrogate_optima"].set_index("model")["share"]
    grid = values["surrogate_curves"]["share"]
    assert grid.min() < optima["mariner"] < grid.max()  # an interior optimum
    assert f"{optima['mariner']:.3f}" == MARKED_MARINER_OPTIMUM
    assert optima["olmix"] == MARKED_SHARES["unmatched"]
    plateaus = values["surrogate_trees"]
    means = values["surrogate_means"].set_index("share")["bpb"]
    last = plateaus.iloc[-1]  # the trees' last plateau runs to pure StarCoder
    assert last["share_left"] < max(share for share in means.index if share < 1.0)  # flat from below the 90% mean on
    assert last["bpb"] < means[1.0]
    multiples = values["multiples"].sort_values("compute_flops")
    for baseline, printed in PRINTED_MULTIPLES.items():
        assert multiples.loc[multiples["baseline"] == baseline, "multiple"].round(1).tolist() == printed
    accuracy = values["accuracy"].set_index(["group", "method"])["accuracy"]
    for key, printed in PRINTED_ACCURACY.items():
        assert f"{accuracy.loc[key]:.1f}" == printed
    for group, printed in PRINTED_MARINER_GROUPS.items():
        assert f"{accuracy.loc[(group, Method.MARINER.value)]:.1f}" == printed


def check_epoch_matching(values: dict[str, pd.DataFrame]) -> None:
    """Figure 2: the shares it marks, each selection's printed epochs and target loss ("0.774", "0.801"), the target's
    optimum ("0.766"), "74.7% less excess loss", the epochs its top axes print, and its titles' FLOPs."""
    for selection in values["selections"].itertuples():
        assert selection.share == pytest.approx(MARKED_SHARES[selection.arm], abs=SHARE_TOLERANCE)
        assert (epoch_label(selection.epochs), f"{selection.target_bpb:.3f}") == PRINTED_SELECTIONS[selection.arm]
    optimum = values["target_optimum"].iloc[0]
    assert optimum["share"] == MARKED_OPTIMUM
    assert f"{optimum['bpb']:.3f}" == PRINTED_OPTIMUM_BPB
    summary = values["summary"].iloc[0]
    assert f"{100 * summary['excess_loss_avoided']:.1f}" == PRINTED_AVOIDED
    assert (f"{summary['proxy_flops']:.3g}", f"{summary['target_flops']:.3g}") == PRINTED_COMPUTE
    for arm, rows in values["epochs"].groupby("arm"):
        assert [epoch_label(epochs) for epochs in rows.sort_values("share")["epochs"]] == PRINTED_EPOCH_ROWS[arm]


def check_matched_compute_scaling(values: dict[str, pd.DataFrame]) -> None:
    """Figure 3's caption (sections/simulated_epoching.tex:37): "Increasing training tokens shifts the optimum toward
    more epochs; increasing model size shifts it toward fewer. Scaling both at fixed TPP produces a milder shift toward
    more epochs. Compute is matched within 3% across columns.\" """
    minima = values["minima"]
    epochs = {track: rows["epochs"].to_numpy() for track, rows in minima.groupby("track")}
    assert (np.diff(epochs[Track.FIXED_PARAMETERS]) > 0).all()  # more tokens at one model size
    assert (np.diff(epochs[Track.FIXED_TOKENS]) < 0).all()  # a larger model at one token budget
    assert (np.diff(epochs[Track.FIXED_TPP]) > 0).all()
    tpp_shift = epochs[Track.FIXED_TPP][-1] - epochs[Track.FIXED_TPP][0]
    tokens_shift = epochs[Track.FIXED_PARAMETERS][-1] - epochs[Track.FIXED_PARAMETERS][0]
    assert tpp_shift < tokens_shift
    # The paths' k-th sweeps train at matched compute: the columns of the paper's panels.
    flops = np.stack([rows["compute_flops"].to_numpy(float) for _, rows in minima.groupby("track")])
    assert np.isfinite(flops).all()
    assert (flops.max(axis=0) / flops.min(axis=0) - 1 <= COMPUTE_MATCH).all()


def check_olmix_capacity_failure(values: dict[str, pd.DataFrame]) -> None:
    """Figure 4: in every sweep, Olmix's pick is an endpoint and MARINER's fitted minimum is interior. And
    sections/surrogate_properties.tex:22: "the lowest-compute curve dips by more than 0.22 BPB".
    """
    observations = values["observations"]
    picks = values["olmix_picks"].set_index("curve")
    minima = values["mariner_minima"].set_index("curve")
    for curve, rows in observations.groupby("curve"):
        low, high = rows["share"].min(), rows["share"].max()
        assert picks.loc[curve, "share"] in (low, high)
        assert low < minima.loc[curve, "share"] < high
    lowest = observations[observations["train_tokens"] == observations["train_tokens"].min()].sort_values("share")
    assert lowest["bpb"].iloc[0] - lowest["bpb"].min() > LOWEST_COMPUTE_DIP


def check_learning_curves(values: dict[str, pd.DataFrame]) -> None:
    """sections/surrogate.tex:106: "MARINER's selections have lower regret than those of Olmix and both RegMix variants
    at essentially every swarm size (Figure~\\ref{fig:r6})."

    Of the 24 swarm sizes and objectives, MARINER's mean ratio is the lowest at all but one: OlmoBaseEval Easy at 240
    runs, where tuned RegMix's is 0.002 lower.
    """
    ratios = values["regret_ratios"].pivot_table(index=["objective", "swarm_size"], columns="model", values="ratio")
    assert ratios.notna().all().all()
    others = ratios[[Method.OLMIX.value, Method.REGMIX.value, Method.REGMIX_RELEASED.value]].min(axis=1)
    assert len(ratios) == 24
    assert (ratios[Method.MARINER.value] >= others).sum() <= 1


def check_proposal_predictions(values: dict[str, pd.DataFrame]) -> None:
    """sections/abstract.tex:1: "Its proposed mixtures measure within 0.005 BPB of its predictions." The redraw raises
    unless the snapshot holds all four models' predictions of every proposal."""
    measured = values["measured_summary"].set_index(["objective", "proposal"])["mean_bpb"]
    predicted = values["predictions"].set_index(["objective", "proposal", "predictor"])["predicted_bpb"]
    mariner = Method.MARINER.value
    for objective in OBJECTIVES:
        gap = measured.loc[(objective, mariner)] - predicted.loc[(objective, mariner, mariner)]
        assert abs(gap) <= PROPOSAL_PREDICTION_GAP


def check_accuracy_vs_bpb(values: dict[str, pd.DataFrame]) -> None:
    """Figure 8: the per-task accuracies average to tab:scaling-main's accuracy column (sections/results.tex:22-25),
    30.1, 32.7, 35.1 and 36.4%, and "mean accuracy on OlmoBaseEval Easy follows the BPB ordering"
    (sections/results.tex:6)."""
    accuracy = values["accuracy"]
    all_tasks = accuracy[accuracy["group"] == ALL_TASKS].set_index("method")
    assert {method: f"{value:.1f}" for method, value in all_tasks["accuracy"].items()} == PRINTED_TABLE_ACCURACY
    by_bpb = all_tasks.sort_values("bpb")
    assert by_bpb["accuracy"].is_monotonic_decreasing


def check_scaling_curves(values: dict[str, pd.DataFrame]) -> None:
    """Figure 9's compute multiples on both objectives, and the savings at 1e21 FLOPs behind the abstract's "69--83\\%
    less final-training compute than proportional mixing ... and 26--38\\% less than Olmix" (sections/abstract.tex:1).
    """
    multiples = values["multiples"].sort_values("compute_flops")
    printed = {"uncheatable": PRINTED_MULTIPLES, "obe_easy": PRINTED_OBE_EASY_MULTIPLES}
    for objective, baselines in printed.items():
        for baseline, row in baselines.items():
            rows = multiples[(multiples["objective"] == objective) & (multiples["baseline"] == baseline)]
            assert rows["multiple"].round(1).tolist() == row
    largest = multiples[multiples["compute_flops"] == LARGEST_SCALE]
    for baseline, saved in PRINTED_SAVINGS.items():
        multiple = largest.loc[largest["baseline"] == baseline, "multiple"]
        assert sorted(round(100 * (1 - 1 / value)) for value in multiple) == saved


def check_on_off_objective(values: dict[str, pd.DataFrame]) -> None:
    """Figure 10: the on-objective values equal tab:scaling-main's (sections/results.tex:22-25), and
    sections/results.tex:44: "MARINER's still outperform Olmix's at every compute scale ... Off-objective mixtures all
    have lower loss than proportional mixing but higher loss than their on-objective counterparts." """
    tracks = values["tracks"].sort_values("compute_flops")
    on_objective = tracks[tracks["on_objective"]]
    for (objective, method), printed in PRINTED_ON_OBJECTIVE.items():
        rows = on_objective[(on_objective["objective"] == objective) & (on_objective["method"] == method)]
        first = rows.iloc[0]
        drawn = [first["mean_bpb"], first["sd_bpb"], *rows["mean_bpb"].iloc[1:]]
        assert [f"{value:.3f}" for value in drawn] == printed
    loss = tracks.pivot_table(
        index=["objective", "compute_flops"], columns=["method", "on_objective"], values="mean_bpb"
    )
    proportional = loss[(Method.PROPORTIONAL.value, True)]
    for method in (Method.OLMIX.value, Method.MARINER.value):
        off, on = loss[(method, False)], loss[(method, True)]
        assert ((on < off) & (off < proportional)).all()
    assert (loss[(Method.MARINER.value, False)] < loss[(Method.OLMIX.value, False)]).all()


def check_component_tradeoffs(values: dict[str, pd.DataFrame]) -> None:
    """Figure 11: on OlmoBaseEval Easy, each All bar is the change of the mean over the 51 tasks, the task-weighted
    mean of the group bars (its caption: "All is the aggregate"); and the rows that sections/results.tex:53 names
    (``TRADEOFF_ROWS``). ``test_component_tradeoffs_all_bars_are_the_aggregate_changes`` checks Uncheatable's All."""
    bars = values["bars"]
    obe_easy = bars[(bars["objective"] == "obe_easy") & (bars["item"] != ALL_TASKS)]
    for (panel, method), rows in obe_easy.groupby(["panel", "method"]):
        everything = bars[(bars["panel"] == panel) & (bars["method"] == method) & (bars["item"] == ALL_TASKS)]
        weighted = np.average(rows["delta_bpb"], weights=rows["tasks"])
        assert everything["delta_bpb"].iloc[0] == pytest.approx(weighted, rel=0, abs=AGGREGATE_TOLERANCE)
    change = bars.set_index(["objective", "proposal_objective", "method", "item"])["delta_bpb"]
    mariner, olmix = Method.MARINER.value, Method.OLMIX.value
    for objective, rows in TRADEOFF_ROWS.items():
        subset = bars.loc[
            (bars["objective"] == objective) & (bars["proposal_objective"] != objective), "proposal_objective"
        ]
        (proposal,) = subset.unique()
        for row in rows:
            assert change[(objective, objective, mariner, row)] >= 0  # the full objective does not improve it
            assert change[(objective, proposal, mariner, row)] < change[(objective, proposal, olmix, row)] < 0
        for method in (mariner, olmix):
            assert change[(objective, proposal, method, ALL_TASKS)] > 0


def check_domain_repetition(values: dict[str, pd.DataFrame]) -> None:
    """Figure 12's caption (sections/analysis.tex:13): "Olmix selects zero FLAN under every cap and saturates each
    FineMath cap, missing both interior minima." And sections/analysis.tex:8: "the matched proxies place both minima
    where the target runs do, and FineMath's optimal epoch count is about seven times FLAN's"."""
    selections = values["selections"].set_index(["domain", "cap"])
    for cap in EPOCH_CAPS:
        assert selections.loc[(FLAN.domain, cap), "share"] == 0.0
        assert selections.loc[(FINEMATH.domain, cap), "epochs"] == pytest.approx(cap, rel=1e-12)
    minima = {}
    for scale in (PROXY, TARGET):
        curves = values[scale]
        minima[scale] = curves.loc[curves.groupby("domain")["bpb"].idxmin()].set_index("domain")
    assert (minima[PROXY]["share"] == minima[TARGET]["share"]).all()
    for domain, row in minima[TARGET].iterrows():
        assert not np.isclose(selections.xs(domain, level="domain")["share"], row["share"]).any()
    target_epochs = minima[TARGET]["epochs"]
    assert round(target_epochs[FINEMATH.domain] / target_epochs[FLAN.domain]) == PRINTED_EPOCH_RATIO


def check_mixture_epochs(values: dict[str, pd.DataFrame]) -> None:
    """Figure 13: "Proportional mixing gives every bucket 0.9 epochs" (its caption, sections/analysis.tex:29), and the
    largest number each column prints equals the paper's."""
    assert {epoch_text(epochs) for epochs in values["proportional"]["epochs"]} == {PRINTED_PROPORTIONAL_EPOCHS}
    largest = values["epochs"].groupby(["objective", "column"])["epochs"].max()
    assert {key: epoch_text(epochs) for key, epochs in largest.items()} == PRINTED_EPOCH_MAXIMA


def check_scale_transfer(values: dict[str, pd.DataFrame]) -> None:
    """Figure 14's printed correlations; its caption (sections/analysis.tex:41): "MARINER's proposal beats every swarm
    mixture at all three settings", which sections/analysis.tex:45 extends to "every baseline"; and
    sections/analysis.tex:48: "the best swarm mixture at Qwen3 360M/1.6B still ranks second of 280 at Llama 200M/6B"."""
    pairs = values["pairs"].set_index(["source", "target"])
    printed = {pair: (f"{row['pearson']:.2f}", f"{row['spearman']:.2f}") for pair, row in pairs.iterrows()}
    assert printed == PRINTED_TRANSFER_CORRELATIONS
    designs = values["designs"].pivot(index="run_label", columns="setting", values="bpb")
    ranks = values["designs"].pivot(index="run_label", columns="setting", values="rank")
    proposals = values["proposals"].pivot(index="method", columns="setting", values="bpb")
    assert len(designs) == TRANSFER_DESIGNS
    others = proposals.drop(index=Method.MARINER.value)
    for setting, design_bpb in designs.items():
        mariner = proposals.loc[Method.MARINER.value, setting]
        assert mariner < design_bpb.min()
        assert mariner < others[setting].min()
    best = designs["qwen3_360m_1p6b"].idxmin()
    assert ranks.loc[best, "llama_200m_6b"] == PRINTED_BEST_TRANSFER_RANK


def check_r2_vs_snr(values: dict[str, pd.DataFrame]) -> None:
    """Figure 15's caption (sections/analysis.tex:61): "labels identify tasks with negative Olmix $R^2$". The redraw
    labels the paper's three tasks."""
    points = values["points"]
    assert set(points.loc[points["labeled"], "name"]) == LABELED_TASKS


def check_frontier(values: dict[str, pd.DataFrame]) -> None:
    """Figure 16's printed Spearman correlations and signal-to-noise ratios; its caption's 30 mixtures per column; and
    sections/analysis.tex:90: "MARINER's predictions of the best held-out mixtures are instead mostly pessimistic", which
    holds in every column. ``test_frontier_mixtures_beat_every_swarm_run`` checks the rest of its caption."""
    correlation = values["spearman"].pivot(index="column", columns="model", values="spearman")
    printed = {
        column: (f"{row[Method.MARINER.value]:.2f}", f"{row[Method.OLMIX.value]:.2f}")
        for column, row in correlation.iterrows()
    }
    assert printed == PRINTED_FRONTIER_SPEARMAN
    snr = values["columns"].set_index("column")["snr"]
    assert {column: round(snr[column]) for column in PRINTED_FRONTIER_SNR} == PRINTED_FRONTIER_SNR
    points = values["points"]
    assert (points.groupby(["column", "model"]).size() == FRONTIER_MIXTURES).all()
    mariner = points[points["model"] == Method.MARINER.value]
    pessimistic = (mariner["predicted"] > mariner["measured"]).groupby(mariner["column"]).sum()
    assert (pessimistic > FRONTIER_MIXTURES / 2).all()


def check_extrapolation_residuals(values: dict[str, pd.DataFrame]) -> None:
    """Figure 17's caption (sections/analysis.tex:84): "MARINER overestimates SciQ loss more for held-out mixtures
    farther from the swarm, whereas arXiv physics shows no comparable overestimate". SciQ's residual, measured minus
    predicted, falls with the distance by least squares, and its median over the farther half of the mixtures lies
    below its median over the nearer half; arXiv physics's slope is not negative, and none of its residuals reaches
    SciQ's median over the farther half."""
    residuals = values["residuals"]
    assert residuals.groupby("component").size().to_dict() == EXTRAPOLATION_POINTS
    slope, near, far = {}, {}, {}
    for component, rows in residuals.groupby("component"):
        slope[component] = np.polyfit(rows["distance"], rows["residual"], 1)[0]
        farther = rows["distance"] > rows["distance"].median()
        near[component] = rows.loc[~farther, "residual"].median()
        far[component] = rows.loc[farther, "residual"].median()
    assert slope["SciQ"] < 0 <= slope["arXiv physics"]
    assert far["SciQ"] < near["SciQ"]
    assert residuals.loc[residuals["component"] == "arXiv physics", "residual"].min() > far["SciQ"]


def check_downsampling_by_budget(values: dict[str, pd.DataFrame]) -> None:
    """Figure 21's caption (sections/appendix.tex:405): "At every token budget $D$, stronger downsampling shifts the
    minimum toward a smaller StarCoder fraction", with the full StarCoder pool added. At each of the four budgets the
    grid minimum's share never rises from the full pool through the six subsets in order of repetition, and the most
    repeated subset's lies below the full pool's. The shift is weak: the minimum stays put in 8 of the 24 steps between
    neighbouring pools, so sections/appendix.tex:400's "each smaller StarCoder subset puts the minimum at a smaller
    StarCoder fraction" holds only in that weak sense."""
    minima = values["minima"]
    assert minima.groupby("train_tokens").size().tolist() == [7, 7, 7, 7]
    for _, rows in minima.groupby("train_tokens"):
        shares = rows.sort_values("support_order")["share"].to_numpy()
        assert (np.diff(shares) <= 0).all()
        assert shares[-1] < shares[0]


def check_downsampling_by_repetition(values: dict[str, pd.DataFrame]) -> None:
    """Figure 22: its legends' minimizing StarCoder epochs (``PRINTED_REPLAY_EPOCHS``), and its caption
    (sections/appendix.tex:414): "Larger token budgets generally favor more repetition, although the shift is not
    monotone at every degree of downsampling" (``REPLAY_EPOCH_FALLS``)."""
    printed, falls = {}, []
    for multiplier, rows in values["minima"].groupby("epoch_multiplier"):
        epochs = rows.sort_values("train_tokens")["epochs"].to_numpy()
        printed[multiplier] = [f"{value:.1f}" for value in epochs]
        falls += [(multiplier, int(step)) for step in np.flatnonzero(np.diff(epochs) < 0)]
    assert printed == PRINTED_REPLAY_EPOCHS
    assert falls == REPLAY_EPOCH_FALLS


def check_curricula_sweeps(values: dict[str, pd.DataFrame]) -> None:
    """Figure 25 and sections/appendix_analysis.tex:298-302:

    - the selected two-phase schedule "beat the selected single-phase mixture only at the longest budget and only with
      StarCoder repeated at least as often as in the target run": the blocks that stay positive after Holm correction
      are exactly the longest budget's at 1x repetition or more;
    - "At 7.4B tokens the gain increases across all six downsampled subsets, and none of the four full-corpus blocks
      has a positive mean gain";
    - "the fitted gain grew with tokens at a fixed model size but not with model size at a fixed number of tokens": it
      rises along the tokens path, and every larger model's gain stays below the smallest model's;
    - "A cell went to a fresh-seed test only if its first 23 schedules gave a single-seed gain of at least 0.005 BPB.
      Only the cell with the most tokens qualified, and its selected two-phase schedule beat its selected single-phase
      mixture in all eight fresh paired seeds."
    """
    blocks = values["blocks"]
    assert len(blocks) == CURRICULA_BLOCKS
    assert (blocks["pairs"] == BLOCK_PAIRS).all()
    longest = blocks["train_tokens"] == blocks["train_tokens"].max()
    repeated = blocks["epoch_multiplier"] >= 1.0  # False for the full pool, whose multiplier is NaN
    assert blocks["holm_positive"].equals(longest & repeated)
    subsets = blocks[longest & blocks["epoch_multiplier"].notna()].sort_values("epoch_multiplier")
    assert len(subsets) == 6
    assert (np.diff(subsets["gain_bpb"]) > 0).all()
    full_pool = blocks[blocks["epoch_multiplier"].isna()]
    assert len(full_pool) == 4
    assert (full_pool["gain_bpb"] <= 0).all()
    paths = values["fitted_gains"].sort_values("compute_flops")
    tokens = paths.loc[paths["path"] == ComputePath.TOKENS, "gain_bpb"].to_numpy()
    parameters = paths.loc[paths["path"] == ComputePath.PARAMETERS, "gain_bpb"].to_numpy()
    assert (np.diff(tokens) > 0).all()
    assert (parameters[1:] < parameters[0]).all()
    cells = values["cells"]
    assert cells["confirmation_eligible"].equals(cells["stage1_gain_bpb"] >= FRESH_SEED_GAIN)
    (qualified,) = cells.loc[cells["confirmation_eligible"], "cell_id"]
    assert cells.set_index("cell_id")["train_tokens"].idxmax() == qualified
    confirmation = values["confirmation"].iloc[0]
    assert confirmation["cell_id"] == qualified
    assert confirmation["pairs"] == confirmation["seeds"] == confirmation["positive_pairs"] == CONFIRMATION_PAIRS
    interval = (confirmation[column] * 1e3 for column in ("gain_bpb", "low_bpb", "high_bpb"))
    assert tuple(f"{value:.3f}" for value in interval) == CONFIRMATION_INTERVAL


def check_switch_smoothing(values: dict[str, pd.DataFrame]) -> None:
    """Figure 26: its caption's switch, baseline step and seeds (``SWITCH_SHARES`` to ``RAMP_PAIRS``);
    sections/appendix_analysis.tex:353: the PL and C4 gaps "move monotonically to their new levels", each step up to
    the 95% interval of the point it reaches; and sections/appendix_analysis.tex:362: "Ramps of up to 512 steps changed
    the mean final programming-languages BPB by at most 0.0005, and the 2,048- and 5,632-step ramps raised it in every
    seed"."""
    switch = values["switch"].iloc[0]
    assert (switch["p0"], switch["p1"]) == SWITCH_SHARES
    assert SWITCH_POOL in switch["label"].split()
    assert switch["label"].endswith(f"vs tied {FIRST_PHASE * switch['p0'] + (1 - FIRST_PHASE) * switch['p1']:.2f}")
    assert switch["lr_state"] == SWITCH_LR_STATE
    assert switch["switch_step"] - switch["baseline_step"] == SWITCH_BASELINE_LEAD
    assert switch["pairs"] == SWITCH_PAIRS
    for _, rows in values["gaps"].groupby("metric"):
        rows = rows.sort_values("lag")
        change = rows["gap_change_bpb"].to_numpy()
        steps = np.sign(change[-1] - change[0]) * np.diff(change)
        assert (steps >= -rows["half_width_bpb"].to_numpy()[1:]).all()
    ramps = values["ramp_gains"]
    assert (ramps["pairs"] == RAMP_PAIRS).all()
    narrow = ramps["width"] <= NARROW_RAMP_WIDTH
    assert (ramps.loc[narrow, "gain_bpb"].abs() <= NARROW_RAMP_CHANGE).all()
    assert (~narrow).sum() == WIDE_RAMPS
    assert (ramps.loc[~narrow, "worse_in"] == ramps.loc[~narrow, "pairs"]).all()


def check_ramp_schedules(values: dict[str, pd.DataFrame]) -> None:
    """Figure 27's caption (sections/appendix_analysis.tex:368): "Smoothing keeps StarCoder's total exposure and moves
    some of it from just after the start of the learning-rate decay to just before it"; each line gives "the StarCoder
    share of each mixture block of 2,048 sequences (16 training steps), as the data loader allocated it", a whole number
    of sequences per block, which the design's weights are not; "All ramps are centred on the start of the decay, 80%
    of the way through training." Every plotted schedule puts the same number of StarCoder sequences in its full
    blocks and rises from 0 to the 0.28 of sections/appendix_analysis.tex:362. At every data seed each ramp's StarCoder
    total equals the abrupt switch's, and a wider ramp draws more of it before the decay. Each plotted ramp runs from
    half its width before the decay to half its width after it."""
    training = values["training"].iloc[0]
    decay = training["decay_step"]
    assert decay / training["total_steps"] == DECAY_START
    assert training["steps_per_block"] == BLOCK_STEPS
    schedules = values["schedules"]
    sequences = schedules["share"].to_numpy() * BLOCK_SEQUENCES
    assert np.array_equal(sequences, np.round(sequences))
    # The plotted form of the totals claim: every schedule puts as many StarCoder sequences in its full blocks.
    full_blocks = set()
    for _, rows in schedules.groupby("arm"):
        rows = rows.sort_values("stage")
        shares = rows["share"].to_numpy()
        assert shares[0] == 0.0
        assert f"{shares[-1]:.2f}" == PRINTED_DECAY_SHARE
        blocks = (rows["end_step"] - rows["start_step"]).to_numpy() // training["steps_per_block"]
        full_blocks.add(float(np.sum(blocks * shares * BLOCK_SEQUENCES)))
    assert len(full_blocks) == 1
    for _, rows in values["totals"].groupby("data_seed"):
        abrupt = rows["ramp_width"].isna()
        assert abrupt.sum() == 1
        assert (rows["starcoder_sequences"] == rows.loc[abrupt, "starcoder_sequences"].iloc[0]).all()
        before = rows.sort_values("ramp_width", na_position="first")["before_decay"].to_numpy()
        assert before[0] == 0
        assert (np.diff(before) > 0).all()
    for _, rows in values["schedules"].groupby("arm"):
        starts = rows.sort_values("stage")["start_step"].to_numpy()
        half = np.nan_to_num(rows["ramp_width"].iloc[0]) / 2
        assert (starts[1], starts[-1]) == (decay - half, decay + half)


def learning_curve_cells(values: dict[str, pd.DataFrame], objective: str, model: Method) -> pd.DataFrame:
    curves = values["learning_curves"]
    rows = curves[(curves["objective"] == objective) & (curves["model"] == model.value)]
    return rows.sort_values("mixtures").set_index("mixtures")


def check_learning_ablations(values: dict[str, pd.DataFrame]) -> None:
    """Figure 18's caption (sections/appendix.tex:370): ten draws per band, and "MARINER's ablations select mixtures
    with similar regret, although they rank mixtures with different accuracy" (``WORSE_RANKING_ABLATIONS``)."""
    curves = values["learning_curves"]
    assert (curves["draws"] == LEARNING_CURVE_DRAWS).all()
    assert not curves["constant_fit"].any()
    full = curves["mixtures"].max()
    for objective in OBJECTIVES:
        mariner = learning_curve_cells(values, objective, Method.MARINER).loc[full]
        for model in (Method.MARINER_CONVEX, Method.MARINER_ADDITIVE, Method.MARINER_BUCKET_THRESHOLDS):
            ablation = learning_curve_cells(values, objective, model).loc[full]
            assert ablation["low"] <= mariner["high"] and mariner["low"] <= ablation["high"]
        for model in WORSE_RANKING_ABLATIONS:
            ablation = learning_curve_cells(values, objective, model).loc[full]
            assert ablation["spearman_high"] < mariner["spearman_low"]


def check_learning_baselines(values: dict[str, pd.DataFrame]) -> None:
    """Figure 19: ten draws per band; released RegMix's constant fits (``CONSTANT_FIT_MIXTURES``), whose correlation
    is undefined; and sections/appendix.tex:263 on the natural cubic spline (``FIRST_MIXTURES_AT_LEVEL``)."""
    curves = values["learning_curves"]
    assert (curves["draws"] == LEARNING_CURVE_DRAWS).all()
    constant = curves[curves["constant_fit"]]
    assert set(constant["model"]) == {Method.REGMIX_RELEASED.value}
    for objective in OBJECTIVES:
        sizes = constant.loc[constant["objective"] == objective, "mixtures"].sort_values().tolist()
        assert sizes == CONSTANT_FIT_MIXTURES
    assert constant["spearman"].isna().all()
    assert curves.loc[~curves["constant_fit"], "spearman"].notna().all()
    full = curves["mixtures"].max()
    for objective, level in SPEARMAN_LEVELS.items():
        for model, mixtures in FIRST_MIXTURES_AT_LEVEL.items():
            spearman = learning_curve_cells(values, objective, model)["spearman"]
            assert spearman[spearman >= level].index.min() == mixtures
        spline = learning_curve_cells(values, objective, Method.SPLINE).loc[full, "spearman"]
        assert spline > learning_curve_cells(values, objective, Method.MARINER).loc[full, "spearman"]


def check_comparator_paths(values: dict[str, pd.DataFrame]) -> None:
    """sections/appendix.tex:388: at the endpoint of every baseline but Olmix, by the mean of its three trainer seeds,
    and at all ten midpoints, MARINER's prediction lies closer to the measured loss than the baseline's own."""
    curves = values["curves"].set_index(["objective", "path", "predictor", "position"])["predicted_bpb"]
    endpoints = values["endpoints"]
    measured_end = endpoints[endpoints["position"] == 1.0].groupby(["objective", "path"])["bpb"].agg(["mean", "size"])
    assert (measured_end["size"] == len(TRAINER_SEEDS)).all()
    midpoints = values["midpoints"].set_index(["objective", "path"])["bpb"]
    assert len(midpoints) == BLEND_MIDPOINTS
    for (objective, path), measured in measured_end["mean"].items():
        own, mariner = curves[(objective, path, path)], curves[(objective, path, Method.MARINER.value)]
        closer_at_end = abs(mariner[1.0] - measured) < abs(own[1.0] - measured)
        assert closer_at_end == (path != Method.OLMIX.value)
        midpoint = midpoints[(objective, path)]
        assert abs(mariner[0.5] - midpoint) < abs(own[0.5] - midpoint)


def check_deletion_matrix(values: dict[str, pd.DataFrame]) -> None:
    """Figure 23 and sections/appendix_analysis.tex:103-120: the OlmoBaseEval Easy mean's two detectable removals and
    their printed T, its swarm-to-repeat ratio of 12.6, Holm's first-step threshold of 4.4, the two Common Crawl topic
    removals that move any task, and the paper figure's 24 annotated cells, 20 of them red."""
    cells = values["cells"]
    assert (cells.groupby("task").size() == REMOVALS).all()
    significant = cells[cells["significant"]]
    mean = significant[significant["task"] == values["summary"]["mean_task"].iloc[0]]
    assert dict(zip(mean["bucket"], (f"{t:.0f}" for t in mean["t"]), strict=True)) == PRINTED_MEAN_REMOVALS
    summary = values["summary"].iloc[0]
    assert f"{summary['mean_snr']:.1f}" == PRINTED_MEAN_SNR
    assert f"{summary['holm_first_step']:.1f}" == PRINTED_HOLM_THRESHOLD
    topics = significant[significant["source"] == COMMON_CRAWL]
    assert set(topics["bucket"]) == DETECTABLE_TOPIC_REMOVALS
    assert (len(significant), int((significant["t"] > 0).sum())) == (ANNOTATED_CELLS, RAISING_CELLS)


def check_swarm_support(values: dict[str, pd.DataFrame]) -> None:
    """sections/appendix_analysis.tex:217 and Figure 24's caption (``OUTSIDE_SUPPORT``): the buckets in which each
    MARINER proposal lies outside the swarm's sampled positive epochs, among the 280 swarm mixtures; the buckets run in
    order of pool size."""
    assert values["swarm"]["run"].nunique() == SWARM_FIT_RUNS
    ranges = values["ranges"]
    assert ranges["pool_tokens"].is_monotonic_decreasing
    proposals = values["proposals"]
    outside = {objective: set(rows.loc[rows["outside"], "bucket"]) for objective, rows in proposals.groupby("objective")}
    assert outside == OUTSIDE_SUPPORT


PAPER_CHECKS: dict[str, Callable[[dict[str, pd.DataFrame]], None]] = {
    "fig:pipeline": check_pipeline,
    "fig:epoch-matching": check_epoch_matching,
    "fig:b5": check_matched_compute_scaling,
    "fig:b6": check_olmix_capacity_failure,
    "fig:r6": check_learning_curves,
    "fig:proposal-predictions": check_proposal_predictions,
    "fig:accuracy": check_accuracy_vs_bpb,
    "fig:r1": check_scaling_curves,
    "fig:off-objective": check_on_off_objective,
    "fig:component-tradeoffs": check_component_tradeoffs,
    "fig:domain-repetition": check_domain_repetition,
    "fig:mixture-epochs": check_mixture_epochs,
    "fig:r3": check_scale_transfer,
    "fig:r7": check_r2_vs_snr,
    "fig:analysis-floor": check_frontier,
    "fig:analysis-extrapolation": check_extrapolation_residuals,
    "fig:a-learning-ablations": check_learning_ablations,
    "fig:a-learning-baselines": check_learning_baselines,
    "fig:a-comparator-paths": check_comparator_paths,
    "fig:b5a": check_downsampling_by_budget,
    "fig:a-downsampling": check_downsampling_by_repetition,
    "fig:a-deletion": check_deletion_matrix,
    "fig:a-support": check_swarm_support,
    "fig:a-curricula-sweeps": check_curricula_sweeps,
    "fig:a-switch-smoothing": check_switch_smoothing,
    "fig:a-ramp-schedules": check_ramp_schedules,
}


@pytest.mark.data_integration
@pytest.mark.timeout(600)
@pytest.mark.parametrize("label", [label for label, spec in FIGURES.items() if spec.tables])
def test_redraw_reproduces_the_paper(snapshot, label):
    PAPER_CHECKS[label](FIGURES[label].draw(snapshot).values)


@pytest.mark.data_integration
@pytest.mark.timeout(60)
def test_registry_matches_the_dataset_index(snapshot):
    index = pd.read_csv(snapshot / FIGURES_INDEX).set_index("label")
    for label, spec in FIGURES.items():
        configs = [config for config in index.loc[label, "configs"].split(";") if config not in INDEX_MARKERS]
        assert (spec.stem, sorted(spec.tables)) == (index.loc[label, "file"], sorted(configs))


@pytest.mark.data_integration
@pytest.mark.timeout(300)
def test_starcoder_fits_reproduce_the_papers_printed_optima_and_compute(snapshot):
    values = FIGURES["fig:b5"].draw(snapshot).values
    for track, rows in values["minima"].groupby("track"):
        assert [round(epochs, 1) for epochs in rows["epochs"]] == PRINTED_OPTIMUM_EPOCHS[Track(track)]
        assert [f"{flops:.2g}" for flops in rows["compute_flops"]] == PRINTED_FLOPS
    for law in values["path_laws"].itertuples():
        rounded = float(f"{law.reference:.3g}")
        coefficient = law.share_coefficient * (rounded / law.reference) ** law.share_exponent
        assert (round(coefficient, 3), round(law.share_exponent, 3)) == PRINTED_SHARE_LAWS[Track(law.track)]


@pytest.mark.data_integration
@pytest.mark.timeout(120)
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_scaling_laws_match_the_released_compute_equivalence(snapshot, objective):
    """Figures 1 and 9 fit the baselines' ladder laws that ladder_compute_equivalence releases."""
    laws = scaling_values(snapshot, objective)["laws"].set_index("method")
    equivalence = read_table(snapshot, "ladder_compute_equivalence")
    released = equivalence[equivalence["objective"] == objective].drop_duplicates("baseline").set_index("baseline")
    for baseline in released.index:
        fitted = laws.loc[baseline, ["e", "a", "alpha"]].to_numpy(float)
        expected = released.loc[baseline, ["law_e", "law_a", "law_alpha"]].to_numpy(float)
        np.testing.assert_allclose(fitted, expected, rtol=0, atol=LAW_TOLERANCE)


@pytest.mark.data_integration
@pytest.mark.timeout(120)
def test_component_tradeoffs_all_bars_are_the_aggregate_changes(snapshot):
    """Figure 11's caption: "All is the aggregate". Each Uncheatable All bar, the change of the runs' recorded
    aggregate, equals its component bars weighted by the tasks table's byte shares, as the dataset records the
    aggregate: the shares as printed, which sum to 1 + 2.1e-8."""
    bars = FIGURES["fig:component-tradeoffs"].draw(snapshot).values["bars"]
    objective = objective_from_tasks(release_tasks(snapshot), "uncheatable")
    weights = pd.Series(objective.weights, index=list(objective.tasks))
    uncheatable = bars[bars["objective"] == "uncheatable"]
    for _, rows in uncheatable.groupby(["panel", "method"]):
        change = rows.set_index("item")["delta_bpb"]
        weighted = float(change.loc[weights.index] @ weights)
        assert change.loc[ALL_TASKS] == pytest.approx(weighted, rel=0, abs=AGGREGATE_TOLERANCE)


@pytest.mark.data_integration
@pytest.mark.timeout(120)
def test_domain_repetition_fits_reproduce_the_released_olmix_fits(snapshot):
    """Figure 12's Olmix fits to the matched proxies are the dataset's domain_sweeps_olmix_fits."""
    sweeps = read_table(snapshot, "domain_sweeps")
    released = read_table(snapshot, "domain_sweeps_olmix_fits").set_index("domain")
    shares = np.linspace(0.0, 1.0, 101)
    weights = np.column_stack([shares, 1.0 - shares])
    for sweep in (FLAN, FINEMATH):
        proxy = matched_proxy(sweeps, sweep)
        law = sweep_olmix_law(proxy)
        fit = released.loc[sweep.domain]
        expected = np.exp(fit["log_c"]) + np.exp(
            weights @ fit[["coefficient_domain", "coefficient_web"]].to_numpy(float)
        )
        np.testing.assert_allclose(law.predict(weights), expected, rtol=0, atol=DOMAIN_FIT_TOLERANCE)


@pytest.mark.data_integration
@pytest.mark.timeout(120)
def test_frontier_mixtures_beat_every_swarm_run(snapshot):
    """Figure 16's caption (sections/analysis.tex:75): its 30 lowest-loss held-out mixtures "beat every swarm run on
    both objectives". The 30 of each objective's column measure below every run of the main swarm on that objective.
    (Across objectives the claim would fail: 26 of the Uncheatable column's 30 beat the swarm on OlmoBaseEval Easy.)"""
    points = FIGURES["fig:analysis-floor"].draw(snapshot).values["points"]
    swarm = release_runs(snapshot, MAIN_SWARM)
    for objective in OBJECTIVES:
        column = points[(points["column"] == OBJECTIVE_TITLES[objective]) & (points["model"] == Method.MARINER.value)]
        assert len(column) == FRONTIER_MIXTURES
        assert column["measured"].max() < swarm[f"bpb__{objective}"].min()


@pytest.mark.data_integration
@pytest.mark.timeout(600)
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_comparator_paths_are_the_package_fits_along_each_blend(snapshot, objective):
    """Figure 20 draws the released path predictions. MARINER's equal the package's refit of the main swarm, and
    Olmix's its released per-task laws, at every blend w(s) = (1 - s) w_MARINER + s w_baseline of the trained
    proposals."""
    values = FIGURES["fig:a-comparator-paths"].draw(snapshot).values
    tasks = objective_from_tasks(release_tasks(snapshot), objective)
    swarm = release_swarm(snapshot, MAIN_SWARM, objective)
    fit = fit_objective(swarm, tasks, release_anchors(snapshot, MAIN_SWARM, objective))
    laws = released_olmix_laws(snapshot, objective)
    curves = values["curves"]
    for path in values["paths"][values["paths"]["objective"] == objective].itertuples():
        start, end = (
            trained_counts(snapshot, mixture) / MIXTURE_BLOCK_SIZE
            for mixture in (path.start_mixture_id, path.end_mixture_id)
        )
        rows = curves[(curves["objective"] == objective) & (curves["path"] == path.path)]
        positions = np.sort(rows["position"].unique())
        blends = (1.0 - positions)[:, None] * start[None, :] + positions[:, None] * end[None, :]
        mariner = rows[rows["predictor"] == Method.MARINER.value].sort_values("position")["predicted_bpb"]
        np.testing.assert_allclose(fit.predict(blends), mariner, rtol=0, atol=PREDICTION_TOLERANCE)
        if path.path == Method.OLMIX.value:
            olmix = rows[rows["predictor"] == Method.OLMIX.value].sort_values("position")["predicted_bpb"]
            per_task = np.column_stack([law.predict(blends) for law in laws])
            np.testing.assert_allclose(per_task @ tasks.weights, olmix, rtol=0, atol=OLMIX_CURVE_TOLERANCE)
