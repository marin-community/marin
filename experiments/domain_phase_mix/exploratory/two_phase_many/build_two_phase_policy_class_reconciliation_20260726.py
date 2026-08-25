# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Build the phase-order policy-class reconciliation report and evidence panel."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import statistics
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/two_phase_policy_class_reconciliation_20260726"
REPLAY_SOURCE_TAR = Path.home() / "Zotero/storage/PU8ULP89/arXiv-2603.04964v1.tar.gz"
REPLAY_SOURCE_SHA256 = "e0f13d9f42eb5d1ca8839fa6f0c01b1c5a25e0fc05c031df52fe6bd8031d5df7"
REPLAY_SOURCE_ASSERTIONS = {
    "sections/appendix.tex": (
        r"We actually use a target data fraction of $\frac{1}{1024}$",
        r"our replay fractions were $0.25, 0.5, 0.75, 0.875$",
        r"the Stage 2 target weight is automatically fixed as $w_2 = 1 - \rho$",
        "up to 32 repetitions of the target data",
    ),
    "sections/two_stage.tex": (
        r"subject to the data constraint",
        r"We sweep over replay fraction $\rho$ and target stage 2 allocation $\alpha$",
        (
            r"We show the full results of sweeping over replay fraction and Stage 2 allocation in Figure "
            r"\ref{fig:main-sweep}"
        ),
    ),
    "figures/main-sweep.tex": (
        r"\textbf{Full data schedule sweep.}",
        "parameterized by their replay fraction and fraction of target data allocated to Stage 2",
        r"\label{fig:main-sweep}",
    ),
}


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    setting: str
    objective: str
    comparison: str
    gain_bpb: float
    ci_low: float | None
    ci_high: float | None
    sample_size: str
    design: str
    inference: str
    status: str
    source: str


@dataclass(frozen=True)
class Literature:
    paper: str
    regime: str
    reported_mechanism: str
    implication_for_marin: str
    why_not_a_contradiction: str
    url: str
    exposure_control: str
    estimand_verdict: str


EVIDENCE = [
    Evidence(
        "starcoder_wsd80_paired",
        "StarCoder, 80/20 WSD",
        "Code BPB",
        "Selected off-diagonal schedule versus best sampled tied schedule",
        0.005794,
        0.001259,
        0.010329,
        "4 fresh paired seeds",
        "Selected global policies; phase TV 0.372; aggregate exposure differs",
        (
            "The surface-selection seed is excluded from inference. All four fresh pairs favor the phased schedule; "
            "the interval is recomputed from those four pairs, assumes normal differences, and has exact two-sided "
            "sign-flip p=0.125. Dropping the selection seed raises the point estimate from 0.005339 to 0.005794 BPB "
            "while weakening the sign test from p=0.0625 to p=0.125. Phase 1 exactly coincides with WSD decay, so this "
            "estimates a selected annealing-data schedule, not chronology independent of learning-rate position or "
            "continuous policy-class optima."
        ),
        "supported",
        "starcoder_wsd80_surface_refined_20260714/report.md",
    ),
    Evidence(
        "starcoder_fixed_aggregate_low",
        "StarCoder, 80/20 WSD",
        "Code BPB",
        "Late-only versus tied at aggregate StarCoder weight 0.140704",
        0.022978,
        None,
        None,
        "1 shared seed",
        "Exact fixed-aggregate triplet; phase TV 0.704; phase 1 is WSD decay",
        "Large path dependence at one dose; learning-rate position and phase order are collinear.",
        "exploratory",
        "starcoder_wsd80_surface_refined_20260714/report.md",
    ),
    Evidence(
        "starcoder_fixed_aggregate_high",
        "StarCoder, 80/20 WSD",
        "Code BPB",
        "Late-only versus tied at aggregate StarCoder weight 0.170050",
        -0.015906,
        None,
        None,
        "1 shared seed",
        "Exact fixed-aggregate triplet; phase TV 0.850; phase 1 is WSD decay",
        "The phase effect reverses when both total exposure and contrast dose change.",
        "exploratory",
        "starcoder_wsd80_surface_refined_20260714/report.md",
    ),
    Evidence(
        "300m_paired_uncheatable",
        "300M swarm",
        "Uncheatable BPB",
        "Mean tied-minus-two-phase gain over 240 exposure-matched pairs",
        -0.000854,
        -0.003113,
        0.001405,
        "240 pairs",
        "Matched qsplit population; mean phase TV 0.507",
        (
            "The sampled high-asymmetry population has a near-zero mean and median effect, with exactly 120/240 pairs "
            "favoring each class. The normal-approximation interval spans zero. Original and tied retrains used "
            "non-identical code paths, so this is provenance-limited and not a local frontier test."
        ),
        "unresolved",
        "one_vs_two_phase_swarm_debug_20260630/one_vs_two_phase_swarm_summary.csv",
    ),
    Evidence(
        "300m_paired_table9",
        "300M swarm",
        "Table-9 macro BPB",
        "Mean tied-minus-two-phase gain over 240 exposure-matched pairs",
        -0.002285,
        -0.006834,
        0.002264,
        "240 pairs",
        "Matched qsplit population; mean phase TV 0.507",
        (
            "The mean is tail-driven: the recomputed median gain is +0.00113 and 123/239 non-tied pairs favor "
            "two-phase. The "
            "normal-approximation interval spans residual gains and harms. Phase-tied baselines differ by 0.003-0.004 "
            "BPB across the original and retrain code paths, so the mean is provenance-limited."
        ),
        "unresolved",
        "one_vs_two_phase_swarm_debug_20260630/paired_table9_canonical_correction_audit.json",
    ),
    Evidence(
        "delphi_within_family_uncheatable",
        "Delphi, TPP 4.4",
        "Uncheatable BPB",
        "Exposure-matched two-phase versus tied siblings at model-proposed aggregates",
        0.005695,
        None,
        None,
        "8 pairs across 3 surrogate families",
        "Aggregate shift <=0.0009 TV; phase contrast 0.11-0.46 TV; 5/8 pairs share a data seed",
        (
            "All eight observations, representing six distinct policy contrasts, and all three family means favor "
            "two-phase, but the pooled mean is heterogeneous: family means are 0.00932, 0.00535, and 0.00079 BPB. The "
            "last is of the same order as the 0.000924 same-seed paired-noise SD. The three-family bootstrap endpoints "
            "are merely the extrema of those family means, not an inferential 95% CI."
        ),
        "suggestive",
        "within_family_policy_class_pairs_20260726/matched_pairs.csv",
    ),
    Evidence(
        "delphi_within_family_table9",
        "Delphi, TPP 4.4",
        "Table-9 macro BPB",
        "Exposure-matched two-phase versus tied siblings at model-proposed aggregates",
        0.003422,
        None,
        None,
        "6 pairs across 2 surrogate families",
        "Aggregate-matched siblings; 3/6 pairs share a data seed",
        (
            "The two family means are -0.00241 and +0.00926 BPB; four of six pairs favor two-phase. With two families, "
            "the source bootstrap endpoints equal the two family means and are not an inferential 95% CI."
        ),
        "unresolved",
        "within_family_policy_class_pairs_20260726/matched_pairs.csv",
    ),
    Evidence(
        "delphi_loweps_uncheatable",
        "Delphi, TPP 4.4",
        "Uncheatable BPB",
        "Best low-phase-information candidate versus exact tied aggregate",
        0.002665,
        None,
        None,
        "1 selected seed",
        "Frozen epsilon path",
        "Suggestive local gain, but not a repeat-confirmed policy-class gap.",
        "suggestive",
        "decoupled_phase_information_validation_results_20260712/report.md",
    ),
    Evidence(
        "delphi_loweps_table9",
        "Delphi, TPP 4.4",
        "Table-9 macro BPB",
        "Best frontier-comparable low-phase-information candidate versus exact tied aggregate",
        0.000840,
        None,
        None,
        "1 selected seed",
        "Frozen epsilon path; same-seed control was a documented 0.007638-BPB low draw",
        (
            "The same-seed gain is below the measured independent-run difference scale. Relative to the later "
            "fresh-control mean, the gain is 0.008478 BPB; neither reference is an unbiased causal effect from one seed."
        ),
        "unresolved",
        "decoupled_phase_information_validation_results_20260712/report.md",
    ),
    Evidence(
        "delphi_loweps_table9_suboptimal_anchor",
        "Delphi, TPP 4.4",
        "Table-9 macro BPB",
        "Best small phase split at the weaker t9s05 aggregate versus its exact tied control",
        0.009430,
        None,
        None,
        "1 selected seed; best of 3 surrogate paths",
        "Exact fixed aggregate; same seed; weaker Table-9 aggregate than t9b075",
        (
            "Effective-exposure, canonical, and separate-heads paths gain 0.009430, 0.005711, and 0.005376 BPB against "
            "their same-seed control. Re-referencing the stronger t9b075 anchor to its fresh-control mean gives "
            "0.002242, 0.008478, and 0.006582 BPB for the same three families, so the cross-anchor ordering is not "
            "stable. No fresh t9s05 controls exist, so this correction is one-sided. The result is one-seed, "
            "control-sensitive, and selected over epsilon."
        ),
        "unresolved",
        "decoupled_phase_information_validation_results_20260712/report.md",
    ),
    Evidence(
        "delphi_random_uncheatable_r25",
        "Delphi, TPP 4.4",
        "Uncheatable BPB",
        "Mean isotropic phase tilt at 25% feasible radius versus tied anchor",
        -0.000035,
        None,
        None,
        "48 directions",
        "Fixed aggregate; median phase TV 0.0072; isotropic tangent directions",
        "The local random-direction distribution is near zero; 38% of treatments improve.",
        "null",
        "pi_meeting_weekly_progress_20260721/source_data/03_random_phase_population.csv",
    ),
    Evidence(
        "delphi_random_uncheatable_r50",
        "Delphi, TPP 4.4",
        "Uncheatable BPB",
        "Mean isotropic phase tilt at 50% feasible radius versus tied anchor",
        -0.000012,
        None,
        None,
        "48 directions",
        "Fixed aggregate; median phase TV 0.0143; isotropic tangent directions",
        "The local random-direction distribution is centered near zero; 52% of treatments improve.",
        "null",
        "pi_meeting_weekly_progress_20260721/source_data/03_random_phase_population.csv",
    ),
    Evidence(
        "delphi_random_uncheatable_r75",
        "Delphi, TPP 4.4",
        "Uncheatable BPB",
        "Mean isotropic phase tilt at 75% feasible radius versus tied anchor",
        -0.000027,
        None,
        None,
        "48 directions",
        "Fixed aggregate; median phase TV 0.0215; isotropic tangent directions",
        "The local random-direction distribution is centered near zero; 52% of treatments improve.",
        "null",
        "pi_meeting_weekly_progress_20260721/source_data/03_random_phase_population.csv",
    ),
    Evidence(
        "delphi_random_table9_r25",
        "Delphi, TPP 4.4",
        "Table-9 macro BPB",
        "Mean isotropic phase tilt at 25% feasible radius versus tied anchor",
        0.000085,
        None,
        None,
        "48 directions",
        "Fixed aggregate; median phase TV 0.0109; isotropic tangent directions",
        "A near-zero mean target-matched effect; 50% of treatments improve.",
        "null",
        "pi_meeting_weekly_progress_20260721/source_data/03_random_phase_population.csv",
    ),
    Evidence(
        "delphi_random_table9_r50",
        "Delphi, TPP 4.4",
        "Table-9 macro BPB",
        "Mean isotropic phase tilt at 50% feasible radius versus tied anchor",
        0.001131,
        None,
        None,
        "48 directions",
        "Fixed aggregate; median phase TV 0.0218; isotropic tangent directions",
        "A small mean dip did not persist coherently along adjacent radii; 60% of treatments improve.",
        "unresolved",
        "pi_meeting_weekly_progress_20260721/source_data/03_random_phase_population.csv",
    ),
    Evidence(
        "delphi_random_table9_r75",
        "Delphi, TPP 4.4",
        "Table-9 macro BPB",
        "Mean isotropic phase tilt at 75% feasible radius versus tied anchor",
        0.000148,
        None,
        None,
        "48 directions",
        "Fixed aggregate; median phase TV 0.0327; isotropic tangent directions",
        "The mean returns near zero and only 48% of treatments improve.",
        "null",
        "pi_meeting_weekly_progress_20260721/source_data/03_random_phase_population.csv",
    ),
    Evidence(
        "delphi_hybrid_effexp_uncheatable",
        "Delphi, TPP 4.4",
        "Uncheatable BPB",
        "Mean effective-exposure phase ordering versus aggregate-matched tied controls",
        0.002412,
        None,
        None,
        "20 correlated policies, 4 shared controls",
        "Pairwise aggregate-KL-matched controls across four aggregate anchors; model-specific phase ordering",
        (
            "The clearest local model-directed ordering signal, but not a global frontier estimate: policies are "
            "correlated, controls are shared, and the selected model policy missed the ex-post low draw."
        ),
        "suggestive",
        "pi_meeting_weekly_progress_20260721/report.md",
    ),
    Evidence(
        "delphi_aggressive_uncheatable",
        "Delphi, TPP 4.4",
        "Uncheatable BPB",
        "Even-channel gain: tied control minus antithetic pair mean at phase TV 0.5",
        -0.007207,
        -0.008469,
        -0.005945,
        "16 antithetic directions",
        "Fixed aggregate; pair mean cancels the odd order channel",
        "Large generic asymmetry has a reliably harmful even cost; this is not the signed phase-order effect.",
        "disfavored",
        "delphi_3e18_aggressive_phase_asymmetry_results_20260723/report.md",
    ),
    Evidence(
        "delphi_aggressive_table9",
        "Delphi, TPP 4.4",
        "Table-9 macro BPB",
        "Even-channel gain: tied control minus antithetic pair mean at phase TV 0.5",
        -0.012204,
        -0.014663,
        -0.009746,
        "16 antithetic directions",
        "Fixed aggregate; pair mean cancels the odd order channel",
        "Large generic asymmetry has a reliably harmful even cost; this is not the signed phase-order effect.",
        "disfavored",
        "delphi_3e18_aggressive_phase_asymmetry_results_20260723/report.md",
    ),
    Evidence(
        "delphi_aggressive_selected_table9",
        "Delphi, TPP 4.4",
        "Table-9 macro BPB",
        "Synthetic-all late recipe at phase TV 0.10 versus same-seed tied frontier control",
        0.010186,
        None,
        None,
        "1 selected recipe; 129 treatments overall, including 40 at TV 0.10",
        "Fixed aggregate; same seed; frontier anchor; handcrafted late-quality direction",
        (
            "The strongest same-seed target-matched frontier result in the archive. It also gains 0.008071 BPB versus "
            "the fresh-control mean, but its 0.010186-BPB magnitude lies inside a normal-null winner's-curse scale: "
            "the expected best is 0.01057 over 129 treatments or 0.00881 over the 40 TV-0.10 treatments. The same "
            "recipe gains 0.010651 BPB versus its same-seed control and 0.006734 BPB versus its fresh-control mean at "
            "an off-Table-9-frontier aggregate under the same data seed. This makes the direction worth an "
            "independent-seed repeat but is not a second frontier confirmation; it reverses at larger radii at this "
            "anchor."
        ),
        "suggestive",
        "delphi_3e18_aggressive_phase_asymmetry_results_20260723/report.md",
    ),
    Evidence(
        "delphi_aggressive_cross_anchor_table9",
        "Delphi, TPP 4.4",
        "Table-9 macro BPB",
        "Synthetic-all late recipe at phase TV 0.10 versus same-seed control at the Uncheatable anchor",
        0.010651,
        None,
        None,
        "1 recipe at the Uncheatable frontier anchor; data seed 7224001 (seed block 1), shared with the Table-9 result",
        (
            "Fixed aggregate within anchor; cross-target metric; its tied Table-9 control is 0.035241 BPB worse than "
            "the same-seed Table-9-anchor control; the two anchors' fresh-control means differ by 0.033439 BPB"
        ),
        (
            "The same-seed gain is 0.010651 BPB and the gain versus the 16-run fresh-control mean is 0.006734 BPB. "
            "The recipe has the same sign at both aggregates and remains beneficial at larger radii at the "
            "Uncheatable anchor. Because both treatments share data seed 7224001 (seed block 1), this shows robustness "
            "across a frontier and an off-frontier aggregate, but the off-frontier gain may include aggregate repair "
            "and is not independent-seed confirmation."
        ),
        "suggestive",
        "delphi_3e18_aggressive_phase_asymmetry_results_20260723/observed_results_with_control_deltas.csv",
    ),
    Evidence(
        "delphi_tpp4p4_table9",
        "Delphi, TPP 4.4",
        "Table-9 macro BPB",
        "Best Table-9 value along the Uncheatable-optimized epsilon path versus its tied aggregate",
        0.002116,
        None,
        None,
        "1 seed; selected best of epsilon grid",
        (
            "Uncheatable-optimized aggregate and phase path; the tied aggregate is 0.040124 BPB worse than the selected "
            "Table-9 tied frontier and 0.032486 worse than its fresh-control mean"
        ),
        (
            "This is consistent with cross-target improvement at a Table-9-suboptimal aggregate, not a Table-9-frontier "
            "phase-order result. It anchors the TPP series but confounds horizon with aggregate-dependent behavior."
        ),
        "exploratory",
        "delphi_fixed_n_tpp_phase_sweep_results_20260713/observed_results.csv",
    ),
    Evidence(
        "delphi_tpp10_table9",
        "Delphi, TPP 10",
        "Table-9 macro BPB",
        "Best Table-9 phase schedule versus exact tied aggregate",
        0.010103,
        None,
        None,
        "1 seed; selected best of epsilon grid",
        (
            "Uncheatable-optimized aggregate and phase path; fixed model size and longer token horizon; TPP 4.4 used "
            "a different data seed"
        ),
        (
            "The best Table-9 value along this target-mismatched path improves at the longer horizon, but this is not "
            "a Table-9-frontier comparison and the cross-TPP trend is unpaired."
        ),
        "suggestive",
        "delphi_fixed_n_tpp_phase_sweep_results_20260713/observed_results.csv",
    ),
    Evidence(
        "delphi_tpp20_table9",
        "Delphi, TPP 20",
        "Table-9 macro BPB",
        "Best Table-9 phase schedule versus exact tied aggregate",
        0.013458,
        None,
        None,
        "1 seed; selected best of epsilon grid",
        (
            "Uncheatable-optimized aggregate and phase path; fixed model size and longer token horizon; TPP 4.4 used "
            "a different data seed"
        ),
        (
            "The largest Table-9-selected value along the target-mismatched path is suggestive, not a replicated "
            "growth curve or a Table-9-frontier result."
        ),
        "suggestive",
        "delphi_fixed_n_tpp_phase_sweep_results_20260713/observed_results.csv",
    ),
    Evidence(
        "delphi_tpp10_uncheatable",
        "Delphi, TPP 10",
        "Uncheatable BPB",
        "Best phase schedule versus exact tied aggregate",
        0.001818,
        None,
        None,
        "1 seed; selected best of epsilon grid",
        (
            "Fixed model size; seed 714000; max simulated epochs held near 12.92 across TPP per the run manifest; "
            "TPP 4.4 used another seed"
        ),
        "The selected gain is smaller than at TPP 4.4, so the Uncheatable best-of-grid path is non-monotone.",
        "suggestive",
        "delphi_fixed_n_tpp_phase_sweep_results_20260713/report.md",
    ),
    Evidence(
        "delphi_tpp20_uncheatable",
        "Delphi, TPP 20",
        "Uncheatable BPB",
        "Best phase schedule versus exact tied aggregate",
        0.003383,
        None,
        None,
        "1 seed; selected best of epsilon grid",
        (
            "Fixed model size and longer token horizon; max simulated epochs held near 12.92 per the run manifest; "
            "TPP 4.4 used a different data seed"
        ),
        "The selected Uncheatable effect remains modest; cross-TPP comparisons are not paired.",
        "suggestive",
        "delphi_fixed_n_tpp_phase_sweep_results_20260713/report.md",
    ),
    Evidence(
        "delphi_dolmino_late_75_uncheatable",
        "Delphi, TPP 4.4",
        "Uncheatable BPB",
        "Late-phase Dolmino share 0.75 versus same-seed tied control",
        -0.002504,
        -0.004646,
        -0.000363,
        "3 paired repeats",
        "Fixed aggregate; phase TV 0.411; conventional Dolmino-late schedule",
        "All three repeats are worse; harm is already visible before the 90-100% late-Dolmino extremes.",
        "disfavored",
        "delphi_3e18_aggressive_phase_asymmetry_results_20260723/report.md",
    ),
    Evidence(
        "delphi_dolmino_late_90_uncheatable",
        "Delphi, TPP 4.4",
        "Uncheatable BPB",
        "Late-phase Dolmino share 0.90 versus same-seed tied control",
        -0.005956,
        -0.008668,
        -0.003245,
        "3 paired repeats",
        "Fixed aggregate; phase TV 0.599; conventional Dolmino-late schedule",
        "All three repeats are worse: a strong conventional late-quality schedule is harmful at this anchor.",
        "disfavored",
        "delphi_3e18_aggressive_phase_asymmetry_results_20260723/report.md",
    ),
    Evidence(
        "delphi_dolmino_late_100_uncheatable",
        "Delphi, TPP 4.4",
        "Uncheatable BPB",
        "Late-phase Dolmino share 1.00 versus same-seed tied control",
        -0.009102,
        -0.011067,
        -0.007137,
        "3 paired repeats",
        "Fixed aggregate; phase TV 0.724; conventional Dolmino-late schedule",
        "All three repeats are worse, strengthening the evidence for a large-asymmetry cost.",
        "disfavored",
        "delphi_3e18_aggressive_phase_asymmetry_results_20260723/report.md",
    ),
    Evidence(
        "delphi_dolmino_late_75_table9",
        "Delphi, TPP 4.4",
        "Table-9 macro BPB",
        "Late-phase Dolmino share 0.75 versus same-seed tied control",
        0.000315,
        -0.015469,
        0.016099,
        "3 paired repeats",
        "Fixed aggregate; phase TV 0.329; conventional Dolmino-late schedule",
        "The mean is nearly tied and only one of three repeats improves; the descriptive interval is wide.",
        "unresolved",
        "delphi_3e18_aggressive_phase_asymmetry_results_20260723/report.md",
    ),
    Evidence(
        "delphi_dolmino_late_90_table9",
        "Delphi, TPP 4.4",
        "Table-9 macro BPB",
        "Late-phase Dolmino share 0.90 versus same-seed tied control",
        -0.007265,
        -0.008780,
        -0.005750,
        "3 paired repeats",
        "Fixed aggregate; phase TV 0.517; conventional Dolmino-late schedule",
        "All three repeats are worse despite the quality-late prior.",
        "disfavored",
        "delphi_3e18_aggressive_phase_asymmetry_results_20260723/report.md",
    ),
    Evidence(
        "delphi_dolmino_late_100_table9",
        "Delphi, TPP 4.4",
        "Table-9 macro BPB",
        "Late-phase Dolmino share 1.00 versus same-seed tied control",
        -0.007460,
        -0.015025,
        0.000105,
        "3 paired repeats",
        "Fixed aggregate; phase TV 0.642; conventional Dolmino-late schedule",
        "All three point estimates are worse, but the descriptive three-repeat interval reaches zero.",
        "disfavored",
        "delphi_3e18_aggressive_phase_asymmetry_results_20260723/report.md",
    ),
    Evidence(
        "60m_frontier_mean_uncheatable",
        "60M",
        "Uncheatable BPB",
        "Mean fixed-aggregate treatment at the Uncheatable frontier anchor",
        -0.000011,
        None,
        None,
        "66 treatments",
        "Designed antithetic and directional panel; anchor selected on Uncheatable",
        "The population mean is tied even though individual directions are learnable.",
        "null",
        "60m_fixed_aggregate_phase_order_results_20260726/report.md",
    ),
    Evidence(
        "60m_frontier_best_uncheatable",
        "60M",
        "Uncheatable BPB",
        "Best selected fixed-aggregate treatment at the Uncheatable frontier anchor",
        0.005721,
        None,
        None,
        "Best of 66",
        "Designed antithetic and directional panel; anchor selected on Uncheatable",
        (
            "A descriptive best-of-66 extreme consistent with selection from the observed dispersion: an iid-normal "
            "Blom approximation gives 0.00671 BPB. The panel is designed and correlated, so this is not a lower bound."
        ),
        "exploratory",
        "60m_fixed_aggregate_phase_order_results_20260726/report.md",
    ),
    Evidence(
        "60m_frontier_mean_table9",
        "60M",
        "Table-9 macro BPB",
        "Mean fixed-aggregate treatment at the Uncheatable frontier anchor",
        -0.001635,
        None,
        None,
        "66 treatments",
        "Designed antithetic and directional panel; anchor selected on Uncheatable, not Table-9",
        "The population mean is close to tied relative to its 0.00929-BPB treatment dispersion.",
        "null",
        "60m_fixed_aggregate_phase_order_results_20260726/report.md",
    ),
    Evidence(
        "60m_frontier_best_table9",
        "60M",
        "Table-9 macro BPB",
        "Best selected fixed-aggregate treatment at the Uncheatable frontier anchor",
        0.024435,
        None,
        None,
        "Best of 66",
        "Designed antithetic and directional panel; anchor selected on Uncheatable, not Table-9",
        (
            "A descriptive best-of-66 extreme consistent with selection from the observed dispersion; a Blom "
            "normal-order-statistic approximation using the panel mean and SD gives 0.0202 BPB. It needs untouched "
            "directional confirmation."
        ),
        "exploratory",
        "60m_fixed_aggregate_phase_order_results_20260726/report.md",
    ),
]

WITHIN_FAMILY_DIAGNOSTICS = [
    {
        "target": "Uncheatable",
        "family": "geomfront",
        "pairs": 3,
        "distinct_policy_contrasts": 3,
        "same_seed_pairs": 0,
        "mean_gain_bpb": 0.009315,
        "min_gain_bpb": 0.007475,
        "max_gain_bpb": 0.011023,
        "paired_noise_sd_bpb": 0.00134,
    },
    {
        "target": "Uncheatable",
        "family": "sepfront",
        "pairs": 3,
        "distinct_policy_contrasts": 1,
        "same_seed_pairs": 3,
        "mean_gain_bpb": 0.005345,
        "min_gain_bpb": 0.004820,
        "max_gain_bpb": 0.006135,
        "paired_noise_sd_bpb": 0.000924,
    },
    {
        "target": "Uncheatable",
        "family": "retstate",
        "pairs": 2,
        "distinct_policy_contrasts": 2,
        "same_seed_pairs": 2,
        "mean_gain_bpb": 0.000790,
        "min_gain_bpb": 0.000344,
        "max_gain_bpb": 0.001236,
        "paired_noise_sd_bpb": 0.000924,
    },
    {
        "target": "Table-9",
        "family": "geomfront",
        "pairs": 3,
        "distinct_policy_contrasts": 3,
        "same_seed_pairs": 0,
        "mean_gain_bpb": -0.002414,
        "min_gain_bpb": -0.013141,
        "max_gain_bpb": 0.017087,
        "paired_noise_sd_bpb": 0.00560,
    },
    {
        "target": "Table-9",
        "family": "sepfront",
        "pairs": 3,
        "distinct_policy_contrasts": 1,
        "same_seed_pairs": 3,
        "mean_gain_bpb": 0.009258,
        "min_gain_bpb": 0.005383,
        "max_gain_bpb": 0.012571,
        "paired_noise_sd_bpb": 0.004083,
    },
]

ODD_EVEN_DIAGNOSTICS = [
    {
        "target": "Uncheatable",
        "phase_tv": 0.10,
        "odd_rms_bpb": 0.000422,
        "odd_latent_rms_snr": 0.000,
        "odd_snr_at_noise_ci_upper": 0.000,
        "even_mean_cost_bpb": 0.000298,
        "stable_sign_directions": "12/16 across all radii",
        "stable_sign_null_p": 0.000038,
        "consistently_better_directions": "0/16",
    },
    {
        "target": "Uncheatable",
        "phase_tv": 0.25,
        "odd_rms_bpb": 0.000936,
        "odd_latent_rms_snr": 1.762,
        "odd_snr_at_noise_ci_upper": 0.000,
        "even_mean_cost_bpb": 0.001719,
        "stable_sign_directions": "12/16 across all radii",
        "stable_sign_null_p": 0.000038,
        "consistently_better_directions": "0/16",
    },
    {
        "target": "Uncheatable",
        "phase_tv": 0.50,
        "odd_rms_bpb": 0.003312,
        "odd_latent_rms_snr": 7.100,
        "odd_snr_at_noise_ci_upper": 3.099,
        "even_mean_cost_bpb": 0.007207,
        "stable_sign_directions": "12/16 across all radii",
        "stable_sign_null_p": 0.000038,
        "consistently_better_directions": "0/16",
    },
    {
        "target": "Table-9",
        "phase_tv": 0.10,
        "odd_rms_bpb": 0.002357,
        "odd_latent_rms_snr": 0.576,
        "odd_snr_at_noise_ci_upper": 0.000,
        "even_mean_cost_bpb": -0.000356,
        "stable_sign_directions": "6/16 across all radii",
        "stable_sign_null_p": 0.189655,
        "consistently_better_directions": "0/16",
    },
    {
        "target": "Table-9",
        "phase_tv": 0.25,
        "odd_rms_bpb": 0.003627,
        "odd_latent_rms_snr": 1.468,
        "odd_snr_at_noise_ci_upper": 0.000,
        "even_mean_cost_bpb": 0.001549,
        "stable_sign_directions": "6/16 across all radii",
        "stable_sign_null_p": 0.189655,
        "consistently_better_directions": "0/16",
    },
    {
        "target": "Table-9",
        "phase_tv": 0.50,
        "odd_rms_bpb": 0.004724,
        "odd_latent_rms_snr": 2.086,
        "odd_snr_at_noise_ci_upper": 0.322,
        "even_mean_cost_bpb": 0.012204,
        "stable_sign_directions": "6/16 across all radii",
        "stable_sign_null_p": 0.189655,
        "consistently_better_directions": "0/16",
    },
]

DIRECTIONAL_DIAGNOSTICS = [
    {
        "target": "Uncheatable",
        "rows": 240,
        "phase_tv": "mean 0.507",
        "oof_r2": 0.583851,
        "oof_spearman": 0.789615,
        "oof_sign_accuracy": 0.833333,
        "top24_selected_realized_gain_bpb": 0.020778,
        "support_warning": "Shared candidate pool: 32/46 exceed at least one signed-feature support bound",
    },
    {
        "target": "Table-9",
        "rows": 240,
        "phase_tv": "mean 0.507",
        "oof_r2": 0.601408,
        "oof_spearman": 0.776880,
        "oof_sign_accuracy": 0.832636,
        "top24_selected_realized_gain_bpb": 0.049646,
        "support_warning": "Same 46-candidate pool as Uncheatable; the 32/46 count is not target-specific",
    },
]

SOURCE_ASSERTIONS = {
    "starcoder_wsd80_surface_refined_20260714/wsd80_paired_contrasts.csv": (
        "20260712,global,constant,-0.007825553417206033",
        "20260715,global,constant,-0.006541013717651367",
    ),
    "one_vs_two_phase_swarm_debug_20260630/procedural_audit_checkpoint.md": (
        "original qsplit checkpoints were trained earlier under the original launcher",
        "roughly `0.003-0.004` Table-9 BPB",
    ),
    "one_vs_two_phase_swarm_debug_20260630/local_phase_signal_diagnostics.json": (
        '"oof_r2": 0.5838505726804721',
        '"oof_r2": 0.6014075016882449',
        '"top24_pred_two_phase_actual_mean_delta": 0.04964628984075837',
    ),
    "within_family_policy_class_pairs_20260726/report.md": (
        "mean delta -0.005695 BPB, cluster-bootstrap 95% CI",
        "[-0.009315, -0.000790]",
        "correlation\n-0.910",
        "slope of -0.361",
        "retstate's anchor was only 0.006",
        "Only 20 of 45 pairs share a data seed",
    ),
    "within_family_policy_class_pairs_20260726/matched_pairs.csv": (
        "sepfront_unch_2p_s0_3e18",
        "sepfront_unch_2p_s1_3e18",
        "sepfront_unch_2p_s2_3e18",
        "retstate_unch_grouped_2p_3e18",
        "retstate_unch_nogroup_2p_3e18",
    ),
    "delphi_3e18_fixed_aggregate_phase_snr_20260724/report.md": (
        "| Table-9 macro     |               0.003772",
        "| uncheatable | odd_order        |          0.500000",
        "| table9      | odd_order        |          0.500000",
        "| table9_frontier      | table9      |                       0.004083",
        "| uncheatable_frontier | uncheatable |                       0.000924",
    ),
    "delphi_3e18_aggressive_phase_asymmetry_results_20260723/report.md": (
        "| table9_frontier      | table9      |              1.057530 |                 1.065168",
        "0.003121",
        "0.007638",
        "| uncheatable_frontier | uncheatable |          0.500000",
        "| table9_frontier      | table9      |          0.500000",
        "| uncheatable_frontier | uncheatable |             0.900000",
        "agphase_a1_recipe_synthetic_all_plus_tv010",
        "-0.010186",
        "best_sign_better_all_three_tv",
    ),
    "delphi_3e18_aggressive_phase_asymmetry_results_20260723/observed_results_with_control_deltas.csv": (
        "agphase_a0_recipe_synthetic_all_plus_tv010",
        "-0.010651165352405068",
        "agphase_a1_recipe_synthetic_all_plus_tv010",
        "-0.010186020589482814",
    ),
    "delphi_3e18_aggressive_phase_asymmetry_results_20260723/fresh_control_summary.csv": (
        "table9_frontier,table9,True,1.0575300915544252,1.065167814825838",
        "uncheatable_frontier,table9,False,1.097653785269499,1.0986067877202284",
    ),
    "decoupled_phase_information_validation_results_20260712/path_summary.csv": (
        "dphase_t9b075_can_e0p005",
        "0.0008396217932682415",
        "dphase_t9b075_sep_e0p05",
        "-0.001056283626033938",
        "dphase_t9b075_eff_e0p05",
        "-0.005395932681948823",
        "dphase_t9s05_eff_e0p01",
        "0.009430305339706013",
        "dphase_t9s05_can_e0p005",
        "0.005710779900099494",
        "dphase_t9s05_sep_e0p005",
        "0.0053755059983955356",
    ),
    "delphi_fixed_n_tpp_phase_sweep_results_20260713/report.md": (
        "| Uncheatable | 10.000 | 0.050",
        "0.002665 BPB at TPP 4.4, 0.001818 at TPP 10, and 0.003383 at TPP 20",
    ),
    "delphi_fixed_n_tpp_phase_sweep_20260712/run_manifest.json": ('"expected_max_simulated_epoch": 12.918367',),
}

PRIMARY_SOURCE_PROVENANCE = [
    {
        "paper": "Replaying pre-training data improves fine-tuning",
        "source": "arXiv:2603.04964v1 source tar",
        "source_sha256": "e0f13d9f42eb5d1ca8839fa6f0c01b1c5a25e0fc05c031df52fe6bd8031d5df7",
        "location": "sections/appendix.tex, Appendix D.2 (Magic number justifications)",
        "claim": "Base target fraction 1/1024 and replay grid {0.25, 0.5, 0.75, 0.875}.",
    },
    {
        "paper": "Replaying pre-training data improves fine-tuning",
        "source": "arXiv:2603.04964v1 source tar",
        "source_sha256": "e0f13d9f42eb5d1ca8839fa6f0c01b1c5a25e0fc05c031df52fe6bd8031d5df7",
        "location": (
            "sections/appendix.tex, Appendix E.1.1 (Mid-training experiments / Fine-tuning baseline / Repetitions)"
        ),
        "claim": "The selected midtraining setup uses 32 repetitions.",
    },
    {
        "paper": "Replaying pre-training data improves fine-tuning",
        "source": "arXiv:2603.04964v1 source tar",
        "source_sha256": "e0f13d9f42eb5d1ca8839fa6f0c01b1c5a25e0fc05c031df52fe6bd8031d5df7",
        "location": "sections/appendix.tex, Appendix A (Data schedule equivalences)",
        "claim": "Stage-2 target weight is w2=1-rho and total target-step fraction is gamma.",
    },
    {
        "paper": "Replaying pre-training data improves fine-tuning",
        "source": "arXiv:2603.04964v1 source tar",
        "source_sha256": "e0f13d9f42eb5d1ca8839fa6f0c01b1c5a25e0fc05c031df52fe6bd8031d5df7",
        "location": "sections/two_stage.tex, Data schedule space and Data schedule experiments",
        "claim": (
            "The Figure 7 experiment varies replay fraction rho and Stage-2 target allocation alpha subject to a "
            "fixed target-data constraint."
        ),
    },
    {
        "paper": "Replaying pre-training data improves fine-tuning",
        "source": "arXiv:2603.04964v1 source tar",
        "source_sha256": "e0f13d9f42eb5d1ca8839fa6f0c01b1c5a25e0fc05c031df52fe6bd8031d5df7",
        "location": "figures/main-sweep.tex, caption and label fig:main-sweep",
        "claim": (
            "The Full data schedule sweep is parameterized by replay fraction and the fraction of target data "
            "allocated to Stage 2."
        ),
    },
]


LITERATURE = [
    Literature(
        "Replaying pre-training data improves fine-tuning",
        "150M models; 4M target tokens; 4B total tokens; C4 plus one target domain",
        (
            "Replay and early target exposure both reduce the abrupt Stage-1-to-Stage-2 shift. Replay becomes less "
            "important, and can hurt, when target data is already present in Stage 1."
        ),
        (
            "A tied mixture already replays every bucket throughout training. A Marin-side extrapolation is that the "
            "incremental order effect should shrink once the aggregate policy removes the stage-distribution "
            "discontinuity; this is not a claim directly tested by the paper."
        ),
        (
            "In arXiv v1, Appendix D.2 gives a 1/1024 base target fraction and replay grid "
            "{0.25, 0.5, 0.75, 0.875}; Appendix E.1.1 gives 32 repetitions for the selected midtraining setup; "
            "Appendix A defines w2=1-rho. Marin's algebra then gives gamma=1/32 and the tied line at rho=31/32, "
            "outside that grid. The derivation is specific to that setup."
        ),
        "https://arxiv.org/abs/2603.04964",
        exposure_control=(
            "Figure 7 fixes total target exposure, but the exact tied policy lies outside the sampled rho grid."
        ),
        estimand_verdict="Closest prior design, but it does not compare its best schedule with the tied policy.",
    ),
    Literature(
        "The Finetuner's Fallacy",
        "1B models; scarce specialized domain datasets; pretraining followed by finetuning",
        (
            "Early specialized exposure reduces later overfitting and forgetting. The preferred timing and mixture "
            "fraction vary with corpus size and training horizon."
        ),
        (
            "It rejects a universal 'rare data only at the end' rule and supports exposure-dependent repetition harm, "
            "which is consistent with the sign reversals and horizon dependence in Marin."
        ),
        (
            "The endpoint is post-finetuning domain loss, not a broad pretraining macro BPB with every bucket already "
            "present in both phases."
        ),
        "https://arxiv.org/abs/2603.16177",
        exposure_control="Specialized pretraining adds target exposure before the common finetuning stage.",
        estimand_verdict="Supports early exposure and repetition mechanisms, not fixed-aggregate phase superiority.",
    ),
    Literature(
        "Midtraining Bridges Pretraining and Posttraining Distributions",
        "Pretraining, midtraining, then supervised finetuning; domain-specific posttraining targets",
        (
            "Midtraining helps most when it bridges a large pretraining-to-posttraining distribution gap. Timing can "
            "matter more than mixture weight, and earlier specialized exposure can be better."
        ),
        (
            "Phase value should scale with a future distribution shift. Marin evaluates the pretraining endpoint "
            "itself, so it omits the posttraining bridge that supplies much of this paper's benefit."
        ),
        (
            "The paper compares after supervised finetuning and optimizes proximity to that future target. It does not "
            "test whether a tied aggregate is optimal for broad endpoint BPB."
        ),
        "https://arxiv.org/abs/2510.14865",
        exposure_control="Start time and specialized mixture weight jointly change total specialized-token exposure.",
        estimand_verdict="Strong bridge-to-future-target result, but not an aggregate-matched phase-order comparison.",
    ),
    Literature(
        "Curriculum Learning for LLM Pretraining: An Analysis of Learning Dynamics",
        "Pythia models from 14M to 1B; 300B-token linguistic curricula",
        (
            "Curricula mainly change exposure within shared latent phases and reduce gradient noise or output "
            "saturation in smaller models; gains shrink at larger scale."
        ),
        (
            "This supports a finite-horizon optimization-stability mechanism rather than an invariant new endpoint "
            "optimum. It also predicts scale and schedule dependence."
        ),
        "Its curricula order examples by linguistic scores, not aggregate-preserving domain-mixture contrasts.",
        "https://arxiv.org/abs/2601.21698",
        exposure_control="Several experiments fix the sample multiset and total compute while changing order.",
        estimand_verdict=(
            "Causal order effects exist, but are sparse, capability-specific, and often shrink at 410M-1B scale."
        ),
    ),
    Literature(
        "On The Power of Curriculum Learning in Training Deep Networks",
        "Image classifiers and an idealized curriculum analysis",
        (
            "Curriculum can accelerate optimization and improve finite-compute generalization, while the ideal "
            "curriculum need not change the underlying global minimum."
        ),
        (
            "The classical theory separates trajectory and finite-compute generalization benefits from changing the "
            "underlying optimization problem's global minimum. A small finite-endpoint gap is therefore not contrary "
            "to curriculum theory."
        ),
        "Different model family and task, but the claim distinction is directly relevant.",
        "https://proceedings.mlr.press/v97/hacohen19a.html",
        exposure_control="Theoretical and image-classification curricula alter presentation order and pacing.",
        estimand_verdict=(
            "Separates finite-time optimization value from changing the optimization function's global minimum."
        ),
    ),
    Literature(
        "Skill-It",
        "Ordered prerequisite skills; online mixture updates; continual pretraining and finetuning",
        (
            "Large ordering gains arise when the data groups have a directed prerequisite graph and the sampler "
            "targets a specific downstream skill."
        ),
        (
            "A useful phase direction can be low-rank and structured rather than aligned with generic 'quality'. "
            "Random 38-dimensional phase contrasts should rarely discover it."
        ),
        "It assumes or learns skill dependencies and uses online adaptation, unlike a fixed two-phase broad objective.",
        "https://arxiv.org/abs/2307.14430",
        exposure_control="Total tokens are fixed, but cumulative exposure allocated to each skill changes.",
        estimand_verdict="Supports narrow structured order effects; does not isolate fixed aggregate exposure.",
    ),
    Literature(
        "DoReMi",
        "Static pretraining-domain mixture optimization",
        "Aggregate domain proportions substantially affect cross-domain performance and compute efficiency.",
        (
            "This matches Marin's dominant empirical fact: aggregate-mixture gains are much larger and higher-SNR than "
            "fixed-aggregate phase-order gains."
        ),
        "It optimizes one static mixture and makes no claim that order is unimportant.",
        "https://arxiv.org/abs/2305.10429",
        exposure_control="The policy is static; aggregate domain weights are the optimized treatment.",
        estimand_verdict="Shows that most accessible gain can reside in the tied aggregate itself.",
    ),
    Literature(
        "RegMix-D",
        "Dynamic mixtures inferred from proxy loss trajectories",
        (
            "Dynamic schedules can outperform static mixtures when stage-specific preferences are identified from "
            "training trajectories."
        ),
        (
            "It highlights information missing from endpoint-only swarms: trajectory supervision can identify "
            "stage-specific preferences more efficiently than fitting only terminal losses."
        ),
        (
            "Its result is not a fixed-aggregate two-phase comparison and uses trajectory data unavailable to Marin's "
            "endpoint surrogate."
        ),
        "https://arxiv.org/abs/2606.18663",
        exposure_control="Stage preferences and cumulative domain totals can both change.",
        estimand_verdict=(
            "Shows gains over selected static baselines while aggregate totals may move; it does not test a globally "
            "optimized tied class under fixed aggregate exposure."
        ),
    ),
    Literature(
        "Does your data spark joy?",
        "7B MPT; 1T tokens; the final 5-30% is replaced by domain-upsampled data",
        (
            "Late domain upsampling can improve broad downstream performance, but the paper explicitly notes that a "
            "better full-run mixture might match the scheduled result."
        ),
        (
            "This is compatible with Marin's observation that optimizing aggregate exposure can absorb most apparent "
            "phase benefit."
        ),
        "Replacing the final segment changes total exposure to both the source and target distributions.",
        "https://arxiv.org/abs/2406.03476",
        exposure_control="The final-stage replacement changes aggregate source exposure.",
        estimand_verdict="Recipe evidence for annealing, with an explicit unresolved static-mixture alternative.",
    ),
    Literature(
        "2 OLMo 2 Furious",
        "1B-32B models; 4-7T pretraining tokens followed by 50-300B Dolmino annealing",
        (
            "A high-quality annealing stage improves downstream results when coupled to terminal learning-rate decay "
            "and, in some comparisons, checkpoint averaging."
        ),
        (
            "It motivates structured late data, but also shows why phase semantics, extra compute, and LR position "
            "cannot be inferred from a data-mixture comparison alone."
        ),
        "The annealing recipes jointly change data, extra tokens, LR trajectory, and sometimes checkpoint averaging.",
        "https://arxiv.org/abs/2501.00656",
        exposure_control="Aggregate data exposure and compute differ between recipes.",
        estimand_verdict="Strong production recipe, not an isolated fixed-compute phase-order causal effect.",
    ),
    Literature(
        "Reuse, Don't Retrain",
        "15B model; general continued pretraining followed by a QA-heavy blend",
        (
            "A staged QA-heavy continuation can outperform placing the QA blend throughout, despite using fewer QA "
            "tokens."
        ),
        (
            "The result supports stage-specific utility, but it also demonstrates that repetition amount and LR "
            "position are part of the treatment."
        ),
        "Staged and throughout policies receive substantially different QA-token totals.",
        "https://arxiv.org/abs/2407.07263",
        exposure_control="Neither target exposure nor LR position is held fixed.",
        estimand_verdict="Does not identify a pure order effect.",
    ),
    Literature(
        "Don't Stop Pretraining",
        "RoBERTa domain-adaptive and task-adaptive continued pretraining",
        (
            "Continued pretraining on domain or task data improves downstream classification, while mismatched "
            "cross-domain adaptation can hurt."
        ),
        (
            "It establishes conditional adaptation value and negative transfer, both consistent with a signed, "
            "aggregate-conditional phase gradient."
        ),
        "Domain-adaptive training adds data and compute after the original pretraining run.",
        "https://aclanthology.org/2020.acl-main.740/",
        exposure_control="The comparison adds target exposure and extra optimization steps.",
        estimand_verdict="Not an order-only or fixed-compute result.",
    ),
    Literature(
        "When Do Curricula Work?",
        "ResNet-50 on CIFAR and Food101 with paced difficulty curricula",
        (
            "Under standard regimes, ordered curricula show no significant advantage and random curricula often "
            "match them; benefits concentrate under severe time limits or label noise."
        ),
        (
            "A broad-loss null is therefore an established empirical possibility rather than an extraordinary "
            "violation of curriculum research."
        ),
        "Training steps are fixed, but pacing gives examples unequal presentation counts.",
        "https://arxiv.org/abs/2012.03107",
        exposure_control="Per-example exposure is not matched across paced curricula.",
        estimand_verdict="Direct precedent for regime-dependent null curriculum effects.",
    ),
    Literature(
        "Beyond Random Sampling",
        "0.5B Llama-style models; limited-data, unlimited-data, and continual-training scenarios",
        (
            "In the exact-multiset limited-data scenario, curriculum effects are clearest in early and mid-training "
            "and terminal losses mostly converge. Curriculum warmup before random sampling yields persistent gains up "
            "to 3.5% in broader continual or unlimited-pool designs."
        ),
        (
            "The paper supports both transient and persistent order effects depending on design. Only the "
            "fixed-multiset scenario is a close analogue to Marin's fixed-aggregate order estimand."
        ),
        (
            "Only Scenario 1 holds the exact token multiset fixed. The headline sustained warmup comparison changes "
            "the sampling design and is not an exact fixed-exposure test."
        ),
        "https://arxiv.org/abs/2506.11300",
        exposure_control=(
            "Exact fixed exposure in Scenario 1; unlimited-pool and continual scenarios do not preserve the same "
            "sample multiset."
        ),
        estimand_verdict=(
            "Supports trajectory effects under fixed exposure and persistent gains in other designs; the latter do "
            "not establish a fixed-aggregate policy-class gap."
        ),
    ),
    Literature(
        "Adaptive Data Optimization",
        "124M and 1.3B language models with online Pile-domain mixture adaptation",
        (
            "Adaptive mixture updates can help, but a strong static natural mixture remains close, especially at "
            "1.3B scale."
        ),
        (
            "This bounds the practical dynamic-mixture headroom in a related broad pretraining setting and supports "
            "scale-dependent diminishing returns."
        ),
        "The adaptive policy changes cumulative domain totals.",
        "https://arxiv.org/abs/2410.11820",
        exposure_control="No fixed aggregate across policies.",
        estimand_verdict="Compatible with limited scheduling headroom after choosing a strong static mixture.",
    ),
    Literature(
        "Curriculum Learning for Language Modeling",
        "ELMo on WikiText-2 and WikiText-103 over ten epochs",
        (
            "Difficulty curricula generally fail to improve perplexity; the stochastic baseline is strongest on "
            "WikiText-103."
        ),
        "This is direct negative evidence for a universal broad-language-model curriculum advantage.",
        "Sampling replacement and padding compute differ across policies.",
        "https://arxiv.org/abs/2108.02170",
        exposure_control="Per-example exposure and effective compute are not perfectly matched.",
        estimand_verdict="Explicit language-modeling null result.",
    ),
    Literature(
        "Scaling Data-Constrained Language Models",
        "Models up to 9B parameters and 900B tokens under repeated-data constraints",
        (
            "A few repeated epochs can be nearly harmless before repetition value decays, after which additional "
            "exposure creates diminishing returns and overfitting."
        ),
        (
            "This supplies a mechanism for the observed token-horizon dependence: phase order matters more once some "
            "buckets approach repetition or saturation limits."
        ),
        "The paper varies repetition at fixed-compute comparison points rather than phase order.",
        "https://arxiv.org/abs/2305.16264",
        exposure_control="Useful for repetition mechanisms, not an ordering estimand.",
        estimand_verdict="Explains a potential horizon threshold without proving a phase advantage.",
    ),
    Literature(
        "Curriculum Learning",
        "Foundational easy-to-hard continuation experiments and theory",
        (
            "A sequence of easier distributions can guide optimization toward better basins or accelerate learning "
            "when the original objective is difficult."
        ),
        (
            "Continuation is a finite-path mechanism; it does not imply that every curriculum family has a distinct "
            "or better terminal optimum under matched exposure."
        ),
        "The curriculum changes support, pacing, and presentation counts early in training.",
        "https://icml.cc/2009/papers/119.pdf",
        exposure_control="Not a fixed-exposure comparison.",
        estimand_verdict="Foundational path-dependence motivation, not evidence for a universal strict gap.",
    ),
]


CLAIMS = [
    {
        "claim": "The two-phase policy class is at least as strong as the tied class.",
        "status": "established",
        "basis": "Set containment: every tied policy is a valid two-phase policy.",
        "boundary": "Containment does not imply a strict or practically meaningful gap.",
    },
    {
        "claim": "Phase order can change endpoint loss.",
        "status": "established",
        "basis": (
            "StarCoder fixed-dose sign reversals and fresh repeats, 300M directional OOF prediction, 60M designed "
            "directions, and robust Uncheatable Delphi antithetic odd effects."
        ),
        "boundary": (
            "The sign and magnitude depend on aggregate exposure, contrast dose, objective, optimizer/LR schedule, and "
            "training horizon."
        ),
    },
    {
        "claim": "Generic or aggressive asymmetry is beneficial.",
        "status": "rejected",
        "basis": (
            "Low-radius random means are small, while TV 0.5 and extreme Dolmino-late schedules have a reliably harmful "
            "even asymmetry cost at Delphi 3e18."
        ),
        "boundary": (
            "This is specific to the tested Delphi frontier anchors. The high-TV 300M qsplit design has learnable "
            "directional effects, and a narrow structured direction can still help."
        ),
    },
    {
        "claim": "The 80/20 WSD StarCoder study proves the continuous two-phase optimum is globally better.",
        "status": "not-established",
        "basis": (
            "Excluding the surface-selection seed, the selected off-diagonal point beats the selected tied point by "
            "0.00579 BPB across four fresh paired seeds."
        ),
        "boundary": (
            "Only one tied candidate was repeated, exact two-sided sign-flip p=0.125, aggregate exposure differs, and "
            "the data phase is collinear with WSD learning-rate position."
        ),
    },
    {
        "claim": "Two-phase has a repeatable 0.01 BPB advantage at Delphi TPP 4.4.",
        "status": "not-supported",
        "basis": (
            "A selected Table-9 synthetic-all recipe improves its same-seed frontier control by 0.010186 BPB, and "
            "some suboptimal-anchor sibling pairs improve by 0.005-0.011. Neither result is repeat-confirmed; the "
            "Table-9 recipe is one selected direction and reverses at larger radii. Its magnitude is inside the "
            "normal-null expected-best scale for the 129-treatment panel."
        ),
        "boundary": (
            "The same recipe has a 0.010651-BPB same-seed gain and a 0.006734-BPB gain versus the fresh-control mean "
            "at the off-Table-9-frontier Uncheatable anchor under the same data seed, showing aggregate robustness but "
            "not a second frontier confirmation. It remains a concrete direction for independent-seed confirmation, "
            "not a repeatable policy-class gap. The longer-horizon Table-9 epsilon path reaches 0.010-0.013 gains on "
            "an Uncheatable-optimized, Table-9-suboptimal aggregate."
        ),
    },
    {
        "claim": "Single-phase is globally optimal in the 39-bucket problem.",
        "status": "unresolved",
        "basis": "No experiment covers the full 38-dimensional contrast polytope or proves a zero phase gradient.",
        "boundary": (
            "The data justify a near-tied empirical conclusion at the current anchors and compute, not a universal "
            "theorem."
        ),
    },
]

HYPOTHESES = [
    {
        "hypothesis_id": "H0",
        "name": "Odd order exists but does not outrun even cost",
        "mechanism": (
            "At the sampled frontier anchors, signed chronology effects grow with contrast radius, but the even cost "
            "of withholding or overconcentrating buckets grows faster along the tested directions."
        ),
        "falsifiable_signature": (
            "Odd antithetic effects are identifiable, yet no orientation beats tied consistently across radii; a "
            "low-rank direction optimized from one radius must fail to transfer to untouched radii or seeds."
        ),
        "current_evidence": (
            "At TV 0.5, point-estimate odd latent RMS SNR is 7.10 on Uncheatable and 2.09 on Table-9, but at the upper "
            "95% noise-SD bound these fall to 3.10 and 0.32. Cross-radius sign stability is 12/16 on Uncheatable "
            "(null p=0.000038) and 6/16 on Table-9 (p=0.190). Even costs are 0.00721 and 0.01220 BPB."
        ),
        "assessment": (
            "Supported on Uncheatable for the tested Delphi directions; Table-9 order identification is unresolved "
            "after propagating noise uncertainty. Neither result excludes a narrow untested direction."
        ),
    },
    {
        "hypothesis_id": "H1",
        "name": "Narrow useful direction, poor high-dimensional search",
        "mechanism": (
            "The order gradient is low-rank and structured, while random directions in the 38-dimensional tangent "
            "space have weak alignment and still pay an even asymmetry cost."
        ),
        "falsifiable_signature": (
            "The same signed low-rank direction wins across radii, seeds, nearby aggregates, and at least one new "
            "objective or scale."
        ),
        "current_evidence": (
            "StarCoder and 60M named directions support structure; whole-swarm versus local-fiber SNR shows why the "
            "fixed-aggregate problem is sample-inefficient, while Delphi random directions do not identify a stable "
            "frontier winner."
        ),
        "assessment": "Plausible, but current 3e18 evidence has not identified the transferable direction.",
    },
    {
        "hypothesis_id": "H2",
        "name": "Effective-aggregate repair",
        "mechanism": (
            "Phase contrast changes retained or influence-weighted exposure, acting like a correction to an imperfect "
            "aggregate rather than an irreducible chronology benefit."
        ),
        "falsifiable_signature": (
            "Two-phase gains scale with tied-anchor suboptimality and disappear at the independently optimized tied "
            "frontier."
        ),
        "current_evidence": (
            "Six distinct exposure-matched within-family contrasts improve across eight observations at suboptimal "
            "model-proposed aggregates, while the frontier-anchored random and aggressive panels do not establish "
            "comparable headroom. Family means decline from 0.00932 to 0.00535 to 0.00079 BPB as the tied anchors "
            "approach the frontier. The one-seed decoupled Table-9 cross-anchor contrast is control-sensitive: after "
            "re-referencing t9b075 to its fresh-control mean, only effective exposure retains the weak-anchor "
            "advantage, while canonical and separate heads reverse it. At 60M, best-of-66 gains are larger at the "
            "worse proportional anchor than at the Uncheatable frontier anchor on both targets, though this is a "
            "selected-extreme comparison."
        ),
        "assessment": (
            "Plausible and directionally supported: the reported correlation (-0.910) and slope (-0.361) are invariant "
            "to the constant error in the reference frontier. Re-referencing gives a descriptive frontier intercept of "
            "about 0.00077 BPB against the selected tied point or 0.00128 against the fresh-control mean, both at the "
            "noise scale. The slope is weakly identified across only three families, and the existing cross-anchor "
            "panel does not provide stable evidence after control-draw correction, so a replicated, preregistered "
            "cross-anchor experiment remains necessary."
        ),
    },
    {
        "hypothesis_id": "H3",
        "name": "Cumulative optimization-time weighting",
        "mechanism": (
            "Changing the token horizon changes cumulative optimization, learning-rate influence, and the time over "
            "which optimizer state and retained features evolve, even when the materialized maximum simulated epochs "
            "are held fixed."
        ),
        "falsifiable_signature": (
            "A preregistered fixed phase direction changes reproducibly with TPP under paired seeds while aggregate "
            "mixture, simulated epoching, optimizer, and architecture remain fixed."
        ),
        "current_evidence": (
            "Along an Uncheatable-optimized aggregate and phase path, best-of-epsilon Table-9 gains are "
            "0.0021/0.0101/0.0135 at TPP 4.4/10/20, while Uncheatable is 0.0027/0.0018/0.0034. At TPP 4.4 that tied "
            "aggregate is 0.032-0.040 BPB worse than the Table-9 frontier controls. The winning epsilon changes, TPP "
            "4.4 uses another seed, and max simulated epochs remain near 12.92 by design."
        ),
        "assessment": (
            "Mechanism not identified because horizon is entangled with aggregate-dependent behavior for Table-9. The "
            "result motivates a fixed-policy, target-matched paired horizon test, not a repetition or saturation claim."
        ),
    },
    {
        "hypothesis_id": "H4",
        "name": "Many-bucket and macro-objective dilution",
        "mechanism": (
            "Bucket- and benchmark-specific order gradients disagree, so macro averaging cancels odd gains while "
            "positive even asymmetry costs remain."
        ),
        "falsifiable_signature": (
            "Component-level order gradients are individually stable and stronger than the macro gradient, with "
            "predictable cancellation in the weighted aggregate."
        ),
        "current_evidence": (
            "Cross-objective order signs are not universal and selected Table-9 and Uncheatable directions differ, "
            "but the existing component audit does not show enough sign cancellation to establish this mechanism."
        ),
        "assessment": "Under-tested and presently unsupported; retain only as a falsifiable alternative.",
    },
    {
        "hypothesis_id": "H5",
        "name": "Winner's curse and selected noise dips",
        "mechanism": (
            "Optimization over many low-SNR candidates selects negative noise and model extrapolation error rather "
            "than a stable phase-order effect."
        ),
        "falsifiable_signature": (
            "Apparent winners regress under fresh paired seeds, fail at adjacent radii, or are not selected by a model "
            "fit without their outcomes."
        ),
        "current_evidence": (
            "The isolated 0.01 Table-9 dip reverses at larger radii; epsilon-path predicted gains grow while observed "
            "performance flattens or worsens."
        ),
        "assessment": "Clearly contributes; cannot alone explain the repeated StarCoder or 60M signed effects.",
    },
    {
        "hypothesis_id": "H6",
        "name": "Support extrapolation hides a learnable local field",
        "mechanism": (
            "A low-dimensional signed phase field is learnable inside the broad qsplit design, but surrogate "
            "optimization proposes contrast patterns outside that field's empirical support."
        ),
        "falsifiable_signature": (
            "A frozen directional model selects held-out in-support schedules successfully, while performance degrades "
            "as signed-feature distance from support increases."
        ),
        "current_evidence": (
            "The 300M nine-feature grouped model attains OOF R2 0.584/0.601 and sign accuracy 0.833 on both objectives, "
            "but 32/46 optimized candidates cross at least one feature support bound."
        ),
        "assessment": (
            "Strong explanation for the gap between learnable qsplit deltas and failed global optimization; it does not "
            "show that the same directional field transfers to the Delphi frontier."
        ),
    },
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table to {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compute_derived_statistics() -> dict[str, Any]:
    contrast_path = OUTPUT_DIR.parent / "starcoder_wsd80_surface_refined_20260714/wsd80_paired_contrasts.csv"
    with contrast_path.open() as handle:
        rows = list(csv.DictReader(handle))
    deltas = [
        float(row["delta_bpb_a_minus_b"])
        for row in rows
        if row["schedule_a"] == "global" and row["schedule_b"] == "constant" and int(row["data_seed"]) != 20260711
    ]
    if len(deltas) != 4:
        raise ValueError(f"Expected four fresh StarCoder pairs, found {len(deltas)}")
    mean_delta = statistics.fmean(deltas)
    standard_error = statistics.stdev(deltas) / math.sqrt(len(deltas))
    t_critical_df3 = 3.182446305284263
    delta_low = mean_delta - t_critical_df3 * standard_error
    delta_high = mean_delta + t_critical_df3 * standard_error
    nonzero_deltas = [delta for delta in deltas if delta != 0]
    positive_count = sum(delta > 0 for delta in nonzero_deltas)
    dominant_sign_count = max(positive_count, len(nonzero_deltas) - positive_count)
    exact_sign_one_sided_p = sum(
        math.comb(len(nonzero_deltas), count) for count in range(dominant_sign_count, len(nonzero_deltas) + 1)
    ) / 2 ** len(nonzero_deltas)
    result = {
        "starcoder_fresh_pairs": {
            "excluded_selection_seed": 20260711,
            "deltas_bpb_global_minus_constant": deltas,
            "mean_gain_bpb": -mean_delta,
            "paired_t_ci95_gain_low_bpb": -delta_high,
            "paired_t_ci95_gain_high_bpb": -delta_low,
            "exact_sign_flip_one_sided_p": exact_sign_one_sided_p,
            "exact_sign_flip_two_sided_p": min(1.0, 2 * exact_sign_one_sided_p),
        }
    }
    evidence_row = next(row for row in EVIDENCE if row.evidence_id == "starcoder_wsd80_paired")
    expected = (
        evidence_row.gain_bpb,
        evidence_row.ci_low,
        evidence_row.ci_high,
    )
    observed = (
        result["starcoder_fresh_pairs"]["mean_gain_bpb"],
        result["starcoder_fresh_pairs"]["paired_t_ci95_gain_low_bpb"],
        result["starcoder_fresh_pairs"]["paired_t_ci95_gain_high_bpb"],
    )
    if any(abs(float(actual) - float(reference)) > 5e-7 for actual, reference in zip(observed, expected, strict=True)):
        raise ValueError(f"StarCoder fresh-pair evidence drifted: observed={observed}, expected={expected}")

    qsplit_sources = {
        "uncheatable": (
            "one_vs_two_phase_swarm_debug_20260630/paired_uncheatable_one_vs_two_phase_qsplit240.csv",
            "300m_paired_uncheatable",
        ),
        "table9": (
            "one_vs_two_phase_swarm_debug_20260630/paired_table9_one_vs_two_phase_qsplit240_canonical.csv",
            "300m_paired_table9",
        ),
    }
    qsplit_statistics: dict[str, Any] = {}
    for target, (relative_path, evidence_id) in qsplit_sources.items():
        with (OUTPUT_DIR.parent / relative_path).open() as handle:
            paired_rows = list(csv.DictReader(handle))
        paired_deltas = [float(row["delta_single_minus_two"]) for row in paired_rows]
        paired_mean = statistics.fmean(paired_deltas)
        paired_sd = statistics.stdev(paired_deltas)
        paired_se = paired_sd / math.sqrt(len(paired_deltas))
        qsplit_statistics[target] = {
            "n": len(paired_deltas),
            "mean_gain_bpb": paired_mean,
            "median_gain_bpb": statistics.median(paired_deltas),
            "sd_bpb": paired_sd,
            "normal_approx_ci95_low_bpb": paired_mean - 1.96 * paired_se,
            "normal_approx_ci95_high_bpb": paired_mean + 1.96 * paired_se,
            "single_phase_better_count": sum(delta < 0 for delta in paired_deltas),
            "two_phase_better_count": sum(delta > 0 for delta in paired_deltas),
            "tie_count": sum(delta == 0 for delta in paired_deltas),
        }
        qsplit_evidence = next(row for row in EVIDENCE if row.evidence_id == evidence_id)
        qsplit_observed = (
            paired_mean,
            qsplit_statistics[target]["normal_approx_ci95_low_bpb"],
            qsplit_statistics[target]["normal_approx_ci95_high_bpb"],
        )
        qsplit_expected = (qsplit_evidence.gain_bpb, qsplit_evidence.ci_low, qsplit_evidence.ci_high)
        if any(
            abs(float(actual) - float(reference)) > 5e-7
            for actual, reference in zip(qsplit_observed, qsplit_expected, strict=True)
        ):
            raise ValueError(
                f"300M qsplit {target} evidence drifted: observed={qsplit_observed}, expected={qsplit_expected}"
            )
    result["qsplit_300m"] = qsplit_statistics

    table9_median = float(qsplit_statistics["table9"]["median_gain_bpb"])
    if abs(table9_median - 0.0011251989516664054) > 5e-7:
        raise ValueError(f"300M Table-9 median drifted: {table9_median}")

    snr_path = OUTPUT_DIR.parent / "delphi_3e18_fixed_aggregate_phase_snr_20260724/macro_phase_snr_by_design.csv"
    with snr_path.open() as handle:
        snr_rows = list(csv.DictReader(handle))
    odd_source_rows = {
        (row["target"], float(row["asymmetry_level"])): row
        for row in snr_rows
        if row["panel"] == "aggressive_phase_asymmetry"
        and row["design_family"] == "balanced_partition"
        and row["effect_channel"] == "odd_order"
        and row["target_matched"] == "True"
    }
    direction_path = (
        OUTPUT_DIR.parent / "delphi_3e18_aggressive_phase_asymmetry_results_20260723/balanced_direction_consistency.csv"
    )
    with direction_path.open() as handle:
        direction_rows = list(csv.DictReader(handle))
    target_direction_rows = {row["target"]: row for row in direction_rows if row["is_primary_target"] == "True"}
    odd_statistics: dict[str, Any] = {}
    for diagnostic in ODD_EVEN_DIAGNOSTICS:
        target = str(diagnostic["target"]).lower().replace("-", "")
        source_target = "table9" if target == "table9" else target
        phase_tv = float(diagnostic["phase_tv"])
        source_row = odd_source_rows[(source_target, phase_tv)]
        direction_row = target_direction_rows[source_target]
        stable_count = int(direction_row["same_better_sign_all_three_tv"])
        direction_count = int(direction_row["direction_count"])
        stable_sign_p = sum(
            math.comb(direction_count, count) * 0.25**count * 0.75 ** (direction_count - count)
            for count in range(stable_count, direction_count + 1)
        )
        observed = (
            float(source_row["rms_effect_bpb"]),
            float(source_row["latent_rms_snr"]),
            float(source_row["latent_rms_snr_noise_ci95_low"]),
            stable_sign_p,
        )
        expected = (
            float(diagnostic["odd_rms_bpb"]),
            float(diagnostic["odd_latent_rms_snr"]),
            float(diagnostic["odd_snr_at_noise_ci_upper"]),
            float(diagnostic["stable_sign_null_p"]),
        )
        if any(abs(actual - reference) > 5e-4 for actual, reference in zip(observed, expected, strict=True)):
            raise ValueError(
                f"Odd-channel diagnostic drifted for {source_target} at TV {phase_tv}: "
                f"observed={observed}, expected={expected}"
            )
        odd_statistics[f"{source_target}_tv_{phase_tv:.2f}"] = {
            "odd_rms_bpb": observed[0],
            "odd_latent_rms_snr_point_noise": observed[1],
            "odd_latent_rms_snr_upper_noise_sd": observed[2],
            "stable_sign_count": stable_count,
            "direction_count": direction_count,
            "stable_sign_null_p_binomial_p_0p25": stable_sign_p,
        }
    result["aggressive_odd_channel"] = odd_statistics

    aggressive_path = (
        OUTPUT_DIR.parent
        / "delphi_3e18_aggressive_phase_asymmetry_results_20260723/observed_results_with_control_deltas.csv"
    )
    with aggressive_path.open() as handle:
        aggressive_rows = list(csv.DictReader(handle))
    table9_anchor_treatments = [
        row
        for row in aggressive_rows
        if row["anchor_id"] == "table9_frontier" and row["contrast_family"] != "center_control"
    ]
    table9_anchor_tv010 = [row for row in table9_anchor_treatments if abs(float(row["target_phase_tv"]) - 0.1) < 1e-8]
    if len(table9_anchor_treatments) != 129 or len(table9_anchor_tv010) != 40:
        raise ValueError(
            "Aggressive Table-9 treatment counts drifted: "
            f"all={len(table9_anchor_treatments)}, tv010={len(table9_anchor_tv010)}"
        )
    same_seed_noise_path = OUTPUT_DIR.parent / "delphi_3e18_fixed_aggregate_phase_snr_20260724/same_seed_delta_noise.csv"
    with same_seed_noise_path.open() as handle:
        same_seed_noise_rows = list(csv.DictReader(handle))
    table9_noise_row = next(
        row for row in same_seed_noise_rows if row["anchor_id"] == "table9_frontier" and row["target"] == "table9"
    )
    table9_same_seed_noise_sd = float(table9_noise_row["same_seed_delta_noise_sd_bpb"])
    selection_scales: dict[str, Any] = {}
    for label, rows_for_scale in (
        ("all_table9_anchor_treatments", table9_anchor_treatments),
        ("table9_anchor_tv010_treatments", table9_anchor_tv010),
    ):
        treatment_count = len(rows_for_scale)
        blom_quantile = statistics.NormalDist().inv_cdf((treatment_count - 0.375) / (treatment_count + 0.25))
        selection_scales[label] = {
            "treatment_count": treatment_count,
            "same_seed_delta_noise_sd_bpb": table9_same_seed_noise_sd,
            "blom_normal_max_quantile": blom_quantile,
            "null_expected_best_gain_bpb": table9_same_seed_noise_sd * blom_quantile,
            "approximation_caveat": "Assumes iid normal null draws; treatments share controls and designed directions.",
        }
    selected_rows = {
        row["anchor_id"]: row
        for row in aggressive_rows
        if row["direction_id"] == "recipe_synthetic_all"
        and row["sign"] == "plus"
        and abs(float(row["target_phase_tv"]) - 0.1) < 1e-8
    }
    selected_gains = {anchor_id: -float(row["table9_delta_vs_control"]) for anchor_id, row in selected_rows.items()}
    expected_selected_gains = {
        "uncheatable_frontier": 0.010651165352405068,
        "table9_frontier": 0.010186020589482814,
    }
    if any(
        abs(selected_gains[anchor_id] - expected_gain) > 5e-10
        for anchor_id, expected_gain in expected_selected_gains.items()
    ):
        raise ValueError(f"Synthetic-all selected gains drifted: {selected_gains}")
    tv010_null_expected_best = selection_scales["table9_anchor_tv010_treatments"]["null_expected_best_gain_bpb"]
    selected_excess_over_tv010_null_sd = (
        selected_gains["table9_frontier"] - tv010_null_expected_best
    ) / table9_same_seed_noise_sd
    if abs(selected_excess_over_tv010_null_sd - 0.33824399053757553) > 5e-10:
        raise ValueError(f"Selected-gain null-scale diagnostic drifted: {selected_excess_over_tv010_null_sd}")
    uncheatable_anchor_control = float(selected_rows["uncheatable_frontier"]["table9_same_seed_control_bpb"])
    table9_anchor_control = float(selected_rows["table9_frontier"]["table9_same_seed_control_bpb"])
    fresh_control_path = (
        OUTPUT_DIR.parent / "delphi_3e18_aggressive_phase_asymmetry_results_20260723/fresh_control_summary.csv"
    )
    with fresh_control_path.open() as handle:
        fresh_control_rows = list(csv.DictReader(handle))
    table9_fresh_control_by_anchor = {
        row["anchor_id"]: float(row["fresh_control_mean_bpb"]) for row in fresh_control_rows if row["target"] == "table9"
    }
    fresh_control_gains = {
        anchor_id: table9_fresh_control_by_anchor[anchor_id] - float(row["table9_macro_bpb"])
        for anchor_id, row in selected_rows.items()
    }
    expected_uncheatable_anchor_control = 1.1025239751684304
    expected_table9_anchor_control = 1.0672830083284397
    expected_fresh_control_gains = {
        "uncheatable_frontier": 0.006733977904203048,
        "table9_frontier": 0.008070827086881227,
    }
    if abs(uncheatable_anchor_control - expected_uncheatable_anchor_control) > 5e-10:
        raise ValueError(f"Uncheatable-anchor Table-9 control drifted: {uncheatable_anchor_control}")
    if abs(table9_anchor_control - expected_table9_anchor_control) > 5e-10:
        raise ValueError(f"Table-9-anchor Table-9 control drifted: {table9_anchor_control}")
    if any(
        abs(fresh_control_gains[anchor_id] - expected_gain) > 5e-10
        for anchor_id, expected_gain in expected_fresh_control_gains.items()
    ):
        raise ValueError(f"Synthetic-all fresh-control gains drifted: {fresh_control_gains}")
    result["delphi_aggressive_selection"] = {
        "selection_scales": selection_scales,
        "synthetic_all_tv010_gain_by_anchor_bpb": selected_gains,
        "synthetic_all_tv010_gain_vs_fresh_control_mean_by_anchor_bpb": fresh_control_gains,
        "table9_selected_gain_excess_over_tv010_null_expected_best_sd": selected_excess_over_tv010_null_sd,
        "table9_control_bpb_by_anchor": {
            "uncheatable_frontier": uncheatable_anchor_control,
            "table9_frontier": table9_anchor_control,
        },
        "table9_fresh_control_mean_bpb_by_anchor": table9_fresh_control_by_anchor,
        "uncheatable_anchor_control_gap_vs_selected_table9_frontier_bpb": (
            uncheatable_anchor_control - table9_anchor_control
        ),
        "uncheatable_anchor_fresh_control_gap_vs_table9_frontier_fresh_control_mean_bpb": (
            table9_fresh_control_by_anchor["uncheatable_frontier"] - table9_fresh_control_by_anchor["table9_frontier"]
        ),
        "shared_data_seed": selected_rows["table9_frontier"]["data_seed"],
        "shared_seed_block": selected_rows["table9_frontier"]["seed_block"],
    }

    decoupled_path = OUTPUT_DIR.parent / "decoupled_phase_information_validation_results_20260712/path_summary.csv"
    with decoupled_path.open() as handle:
        decoupled_rows = list(csv.DictReader(handle))
    decoupled_by_anchor_family = {
        (row["anchor_tag"], row["family"]): row for row in decoupled_rows if row["objective"] == "table9"
    }
    fresh_control_mean = 1.065168
    t9b075_fresh_gains = {
        family: fresh_control_mean - float(decoupled_by_anchor_family[("t9b075", family)]["best_observed_target_bpb"])
        for family in ("effective_exposure", "canonical", "separate_heads")
    }
    expected_fresh_gains = {
        "effective_exposure": 0.002241975763625836,
        "canonical": 0.008477530238843,
        "separate_heads": 0.006581624819540804,
    }
    if any(
        abs(t9b075_fresh_gains[family] - expected_gain) > 5e-7 for family, expected_gain in expected_fresh_gains.items()
    ):
        raise ValueError(f"Decoupled fresh-control gains drifted: {t9b075_fresh_gains}")
    canonical_t9b075 = decoupled_by_anchor_family[("t9b075", "canonical")]
    selected_control = float(canonical_t9b075["best_observed_target_bpb"]) + float(
        canonical_t9b075["best_observed_gain_vs_tied"]
    )
    control_offset = fresh_control_mean - selected_control
    paired_delta_sd = table9_same_seed_noise_sd
    whole_swarm_run_sd = 0.003772
    fresh_control_run_sd = 0.003121
    result["decoupled_cross_anchor_control_sensitivity"] = {
        "selected_t9b075_control_bpb": selected_control,
        "fresh_t9b075_control_mean_bpb": fresh_control_mean,
        "control_offset_bpb": control_offset,
        "t9b075_gain_vs_fresh_control_mean_bpb": t9b075_fresh_gains,
        "implied_same_seed_correlation_using_whole_swarm_run_sd": 1 - paired_delta_sd**2 / (2 * whole_swarm_run_sd**2),
        "implied_same_seed_correlation_using_fresh_control_run_sd": (
            1 - paired_delta_sd**2 / (2 * fresh_control_run_sd**2)
        ),
        "correlation_identity": "Var(delta)=2*sigma^2*(1-rho)",
    }

    treatment_path = OUTPUT_DIR.parent / "60m_fixed_aggregate_phase_order_results_20260726/treatment_summary.csv"
    with treatment_path.open() as handle:
        treatment_rows = list(csv.DictReader(handle))
    selection_statistics: dict[str, Any] = {}
    expected_approximations = {"uncheatable": 0.00671, "table9": 0.02019}
    for target in ("uncheatable", "table9"):
        frontier_row = next(
            row for row in treatment_rows if row["target"] == target and row["anchor_id"] == "uncheatable_frontier"
        )
        treatment_count = int(frontier_row["observed_treatments"])
        gain_mean = -float(frontier_row["mean_delta"])
        gain_sd = float(frontier_row["sd_delta"])
        blom_quantile = statistics.NormalDist().inv_cdf((treatment_count - 0.375) / (treatment_count + 0.25))
        expected_best_gain = gain_mean + gain_sd * blom_quantile
        observed_best_gain = -float(frontier_row["best_delta"])
        mean_evidence = next(row for row in EVIDENCE if row.evidence_id == f"60m_frontier_mean_{target}")
        best_evidence = next(row for row in EVIDENCE if row.evidence_id == f"60m_frontier_best_{target}")
        if abs(gain_mean - mean_evidence.gain_bpb) > 5e-7:
            raise ValueError(f"60M {target} frontier mean drifted: {gain_mean}")
        if abs(observed_best_gain - best_evidence.gain_bpb) > 5e-7:
            raise ValueError(f"60M {target} frontier best drifted: {observed_best_gain}")
        if abs(expected_best_gain - expected_approximations[target]) > 5e-4:
            raise ValueError(f"60M {target} expected best approximation drifted: {expected_best_gain}")
        selection_statistics[target] = {
            "anchor_id": "uncheatable_frontier",
            "treatment_count": treatment_count,
            "gain_mean_bpb": gain_mean,
            "gain_sd_bpb": gain_sd,
            "blom_normal_max_quantile": blom_quantile,
            "blom_expected_best_gain_bpb": expected_best_gain,
            "observed_best_gain_bpb": observed_best_gain,
            "approximation_caveat": "Assumes iid normal draws; the designed antithetic panel is correlated.",
        }
    result["60m_frontier_selection"] = selection_statistics
    return result


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join([head, rule, body])


def render_report() -> str:
    evidence_rows = []
    for row in EVIDENCE:
        interval = (
            f"[{row.ci_low:.6f}, {row.ci_high:.6f}]"
            if row.ci_low is not None and row.ci_high is not None
            else "not estimated"
        )
        evidence_rows.append(
            [
                row.setting,
                row.objective,
                f"{row.gain_bpb:+.6f}",
                interval,
                f"{row.sample_size}; {row.design}",
                row.inference,
                f"[source](../{row.source})",
            ]
        )

    literature_rows = [
        [
            f"[{row.paper}]({row.url})",
            row.regime,
            row.exposure_control,
            row.estimand_verdict,
            row.reported_mechanism,
            row.implication_for_marin,
            row.why_not_a_contradiction,
        ]
        for row in LITERATURE
    ]

    claim_rows = [[row["claim"], row["status"], row["basis"], row["boundary"]] for row in CLAIMS]
    hypothesis_rows = [
        [
            f"{row['hypothesis_id']}: {row['name']}",
            row["mechanism"],
            row["falsifiable_signature"],
            row["current_evidence"],
            row["assessment"],
        ]
        for row in HYPOTHESES
    ]
    claim_table = markdown_table(["Claim", "Status", "Evidence", "Boundary"], claim_rows)
    hypothesis_table = markdown_table(
        ["Hypothesis", "Mechanism", "Falsifiable signature", "Current evidence", "Assessment"],
        hypothesis_rows,
    )
    evidence_table = markdown_table(
        [
            "Setting",
            "Objective",
            "Effect in gain convention",
            "Reported interval",
            "Sample and design",
            "Interpretation",
            "Source",
        ],
        evidence_rows,
    )
    within_family_table = markdown_table(
        [
            "Target",
            "Family",
            "Observations",
            "Distinct policy contrasts",
            "Same-seed pairs",
            "Mean gain BPB",
            "Pair range BPB",
            "Paired-noise SD BPB",
        ],
        [
            [
                str(row["target"]),
                str(row["family"]),
                str(row["pairs"]),
                str(row["distinct_policy_contrasts"]),
                str(row["same_seed_pairs"]),
                f"{row['mean_gain_bpb']:+.6f}",
                f"[{row['min_gain_bpb']:+.6f}, {row['max_gain_bpb']:+.6f}]",
                f"{row['paired_noise_sd_bpb']:.5f}",
            ]
            for row in WITHIN_FAMILY_DIAGNOSTICS
        ],
    )
    odd_even_table = markdown_table(
        [
            "Target",
            "Phase TV",
            "Odd RMS BPB",
            "Odd latent RMS SNR (point noise)",
            "Odd SNR at upper noise-SD bound",
            "Even mean cost BPB",
            "Stable better sign",
            "Null p for sign stability",
            "One sign beats tied at all radii",
        ],
        [
            [
                row["target"],
                f"{row['phase_tv']:.2f}",
                f"{row['odd_rms_bpb']:.6f}",
                f"{row['odd_latent_rms_snr']:.3f}",
                f"{row['odd_snr_at_noise_ci_upper']:.3f}",
                f"{row['even_mean_cost_bpb']:+.6f}",
                row["stable_sign_directions"],
                f"{row['stable_sign_null_p']:.6f}",
                row["consistently_better_directions"],
            ]
            for row in ODD_EVEN_DIAGNOSTICS
        ],
    )
    directional_table = markdown_table(
        [
            "Target",
            "Rows",
            "Phase TV",
            "OOF R2",
            "OOF Spearman",
            "OOF sign accuracy",
            "Top-24 realized gain",
            "Support warning",
        ],
        [
            [
                row["target"],
                str(row["rows"]),
                row["phase_tv"],
                f"{row['oof_r2']:.3f}",
                f"{row['oof_spearman']:.3f}",
                f"{row['oof_sign_accuracy']:.3f}",
                f"{row['top24_selected_realized_gain_bpb']:.6f}",
                row["support_warning"],
            ]
            for row in DIRECTIONAL_DIAGNOSTICS
        ],
    )
    literature_table = markdown_table(
        [
            "Paper",
            "Regime",
            "Exposure control",
            "Verdict for strict estimand",
            "Reported mechanism",
            "Marin implication",
            "Scope relative to Marin estimand",
        ],
        literature_rows,
    )

    return rf"""# Does two-phase mixing materially dominate tied mixtures?

## Executive conclusion

The evidence does **not** support the broad statement that phase order has no
effect. It supports a narrower and more useful statement:

> Phase order is a low-signal, aggregate-conditional, direction-specific
> residual. At the current Delphi 3e18 horizon (TPP 4.4), near the best known
> tied aggregates, no experiment has established a repeatable 0.01 BPB
> target-matched advantage. This is not a proof that the global two-phase
> optimum is tied.

Five observations prevent a universal null conclusion:

1. Excluding the surface-selection seed, StarCoder 80/20 WSD gives a paired
   0.00579 BPB gain for selected schedules across four fresh seeds (paired-\(t\)
   interval [0.00126, 0.01033]; exact two-sided sign-flip \(p=0.125\)).
2. At Delphi TPP 4.4, six exposure-matched within-family policy contrasts
   represented by eight observations at suboptimal model-proposed aggregates
   all favor two-phase, with a descriptive pooled
   mean of 0.00570 BPB. This is heterogeneous: family means are 0.00932,
   0.00535, and 0.00079 BPB, and the nearest-frontier family is on the same
   scale as same-seed paired noise.
3. At TV 0.5, Delphi antithetic odd effects are robustly identifiable on
   Uncheatable. Table-9 is unresolved after propagating uncertainty in its
   six-degree-of-freedom noise estimate. At TV 0.5, the even asymmetry cost
   dominates on both targets, and no orientation beats tied at all three radii.
4. A selected same-seed synthetic-all recipe at the Table-9 frontier improves
   by 0.010186 BPB at TV 0.10, but a normal-null best-of-129 scale is
   approximately 0.01057 BPB. At the Uncheatable frontier anchor, the same
   recipe gains 0.010651 BPB against its same-seed control and 0.006734 BPB
   against the 16-run fresh-control mean. That anchor is 0.035241 BPB worse
   than the same-seed Table-9-anchor control and 0.033439 BPB worse in a
   fresh-mean-to-fresh-mean comparison. This is robustness across a frontier
   and an off-frontier aggregate, not a second frontier confirmation; the
   off-frontier gain may include aggregate repair. Both treatments use data
   seed 7224001, and the recipe reverses at larger radii at the Table-9 anchor.
   Separately,
   the same Delphi model at longer token horizons reaches selected one-seed
   Table-9 gains of 0.01010 at TPP 10 and 0.01346 at TPP 20 along an
   Uncheatable-optimized aggregate path. The horizon comparison is unpaired and
   maximizes over an epsilon grid.
5. The 60M fixed-aggregate panel identifies cross-target directional structure
   even though its frontier treatment-population mean is approximately tied.

The defensible paper claim is therefore about **regime and identifiability**,
not equality of the policy classes.

At Delphi TPP 4.4, the most defensible magnitude statement is:

- no frontier-comparable phase-order gain has been repeat-confirmed;
- the strongest selected frontier-comparable Uncheatable gain is 0.002665 BPB
  on one seed, while suboptimal-anchor sibling pairs show larger gains;
- the strongest selected same-seed Table-9 result is 0.010186 BPB at TV 0.10,
  but this magnitude is inside the expected null selection tail
  (0.01057 BPB over all 129 treatments; 0.00881 over the 40 TV-0.10
  treatments). The same recipe gains 0.010651 BPB against its same-seed control
  and 0.006734 BPB against the fresh-control mean at the
  off-Table-9-frontier Uncheatable anchor under the same data seed. This makes it an
  independent-seed repeat candidate, but the second gain may include aggregate
  repair and is not a second frontier confirmation;
- a repeatable \(0.01\) BPB frontier advantage has not been established, but no
  valid confidence bound currently excludes one in an unsearched direction;
- Table-9 is less settled because its repeat noise is larger. Its TPP-10/20
  gains are materially larger, but they follow an Uncheatable-optimized,
  Table-9-suboptimal aggregate path rather than a Table-9 frontier;
- failure to find a gain is not a proof that \(J^\star_{{\mathrm{{2p}}}}
  =J^\star_{{\mathrm{{1p}}}}\).

## Claim ladder

{claim_table}

## The estimand prior work usually does not test

Let the fixed phase-0 fraction be \(\alpha\), and let
\(p^{{(0)}},p^{{(1)}}\in\Delta^{{K-1}}\). Define

$$
a=\alpha p^{{(0)}}+(1-\alpha)p^{{(1)}},\qquad
p^{{(0)}}=a-(1-\alpha)d,\qquad p^{{(1)}}=a+\alpha d.
$$

The contrast obeys

$$
\mathbf 1^\top d=0,\qquad
-\frac{{a_i}}{{\alpha}}\le d_i\le\frac{{a_i}}{{1-\alpha}}.
$$

The endpoint objective is an expectation over data order, initialization, and
training/evaluation randomness,

$$
J(a,d)=\mathbb E_\xi\!\left[\mathcal L(\theta_T(a,d;\xi))\right].
$$

The strict policy-class question is

$$
J^\star_{{\mathrm{{2p}}}}=\min_{{a,d}}J(a,d)
\quad\text{{versus}}\quad
J^\star_{{\mathrm{{1p}}}}=\min_a J(a,0).
$$

Set containment proves only \(J^\star_{{\mathrm{{2p}}}}\le
J^\star_{{\mathrm{{1p}}}}\). A strict or practically meaningful gap must be
measured. Here \(\alpha\) is fixed rather than optimized. The clean causal
phase-order estimand holds \(a\), total compute,
sample-generation rules, optimizer treatment, and learning-rate trajectory
fixed while varying \(d\). Most finetuning, continued-pretraining, annealing,
and adaptive-mixture papers change at least one of those quantities.

A fixed-anchor experiment estimates only the local gap
\(\min_d J(a^\star_{{\mathrm{{1p}}}},d)-J(a^\star_{{\mathrm{{1p}}}},0)\).
It does not identify the global class gap because the best two-phase aggregate
\(a^\star_{{\mathrm{{2p}}}}\) may differ. The selected StarCoder schedule
comparison changes both \(a\) and \(d\); the fixed-dose StarCoder triplets and
Marin phase-fiber panels isolate \(d\).

## A local theory that reconciles the results

Around a tied policy \(d=0\), write

$$
J(a,d)=J_0(a)+s(a)^\top d+\frac12d^\top R(a)d+O(\lVert d\rVert^3).
$$

- \(s(a)^\top d\) is the **odd signed order effect**. It requires path
  dependence: retention, forgetting, gradient non-commutativity, recency,
  optimizer memory, or time-varying learning-rate influence.
- \(d^\top R(a)d/2\) is the **even asymmetry cost or benefit**. Positive
  curvature prices underexposure, repeated-data overload, and moving useful
  domains away from part of training.
- If \(R(a)\succ0\), the unconstrained local improvement is approximately
  \(\frac12s(a)^\top R(a)^{{-1}}s(a)\). A weak order gradient creates a
  second-order optimum gap even when chronology is real.

Antithetic \(+d/-d\) experiments identify the two channels:

$$
O(d)=\frac{{J(a,d)-J(a,-d)}}2\approx s(a)^\top d,
\qquad
C(d)=\frac{{J(a,d)+J(a,-d)}}2-J(a,0)\approx\frac12d^\top R(a)d.
$$

This explains why sufficiently large symmetric phase perturbations can be
mostly worse than tied even when signed order effects exist. For a symmetric design
\(d\sim(0,\Sigma)\),

$$
\mathbb E[J(a,d)-J(a,0)]
\approx \frac12\operatorname{{tr}}(R(a)\Sigma).
$$

The odd term cancels in expectation; positive even curvature does not.
Symmetric sampling therefore need not produce an outcome distribution centered
at the tied control. The low-radius Delphi random panel itself is close to
centered; the shift is clear in the aggressive-TV panel.

### Why order can matter mechanistically

For two domain-specific SGD maps \(T_i,T_j\), a second-order expansion gives

$$
T_jT_i-T_iT_j
=\eta^2\left(H_jg_i-H_ig_j\right)+O(\eta^3).
$$

Order can matter when the gradient flows do not commute, features change, or
optimizer moments retain history. The commutator is an SGD illustration; Marin
runs use Muon/AdamH-family optimizers, so it is not a literal optimizer model.
Moreover, a nonzero local commutator need not survive to the endpoint because
later dynamics can erase it. More generally, a schedule perturbation has
terminal influence

$$
\delta\theta_T
=-\int_0^T\Phi(T,t)\eta(t)\sum_i\delta p_i(t)g_i(\theta_t)\,dt.
$$

Thus raw token exposure is not the whole state: learning rate \(\eta(t)\),
future transport \(\Phi(T,t)\), and optimizer state weight early and late
tokens differently. \(\Phi\) may contract or amplify. An aggregate-preserving
perturbation has no first-order endpoint effect if
\(\Phi(T,t)\eta(t)g_i(\theta_t)\) is stationary in time for every bucket, because
the time integral of each \(\delta p_i(t)\) is zero. This is why 50/50 cosine
StarCoder, 80/20 WSD StarCoder, and 80/20 WSD Delphi are different
interventions even at matched aggregate exposure.

This model predicts the observed geometry:

1. **One contrast dimension is easy.** In two-domain StarCoder, an
   aggregate-preserving phase contrast has one degree of freedom. A sweep spans
   the useful direction directly.
2. **Thirty-eight contrast dimensions are hard.** In the 39-bucket problem, a
   random unit direction has expected projection \(O(1/\sqrt{{38}})\) on a
   narrow useful \(g\), while the even cost grows with contrast radius.
3. **A good tied aggregate can have a small residual gradient.** The tied
   schedule already exposes every bucket throughout training. It removes much
   of the distribution shift that replay and midtraining are designed to fix.
4. **Objectives may cancel.** Bucket-specific order preferences can disagree,
   but the present component evidence does not yet establish cancellation as
   the dominant mechanism.
5. **Horizon may change influence weighting.** The fixed-N experiment keeps
   the candidate's maximum simulated epochs near 12.92 while changing TPP.
   Along the Uncheatable-optimized path, Table-9's post-hoc selected gain rises
   from 0.00212 to 0.01346, but Uncheatable follows 0.00267, 0.00182, 0.00338.
   At TPP 4.4, that aggregate is materially suboptimal for Table-9. Endpoints
   are unpaired across TPP and selected epsilon changes, so neither monotonicity,
   repetition, nor target-matched frontier headroom is identified.

## Competing reconciliations

The hypotheses below are deliberately non-exclusive. H0/H2 are compatible with
the sampled Uncheatable frontier panels, H6 explains why broad qsplit effects
need not yield safe optimized schedules, and the present experiments do not
exclude a narrow untested \(0.01\)-BPB direction.

{hypothesis_table}

## Internal evidence

The sign convention below is positive when the two-phase policy has lower BPB.
For even-channel rows, the effect is tied loss minus the antithetic pair mean;
those rows measure asymmetry cost rather than signed order. Selected minima are
not unbiased estimates of a policy-class gap. Intervals are copied only when
the source reports a compatible interval; missing intervals must not be read as
zero uncertainty.

{evidence_table}

### Odd order versus even asymmetry cost

The aggressive Delphi panel directly implements the local decomposition.
On Uncheatable, the odd channel is robustly identifiable at TV 0.5 and
cross-radius signs are stable for 12/16 directions. On Table-9, the point
estimate looks identifiable but the conclusion disappears at the upper 95%
noise-SD bound, and 6/16 stable signs are compatible with noise. The blocking
fact on both targets is that none of the tested orientations beats tied at all
three radii, while the even cost grows strongly. The stability \(p\)-value is
one cross-radius exact Binomial(16, 1/4) upper-tail statistic repeated on each
TV row for readability; it assumes independent sign noise across radii, which
shared per-direction seed blocks may weaken.

{odd_even_table}

### Fixed-exposure sibling effects are heterogeneous

The eight Uncheatable sibling observations represent six distinct policy
contrasts and all favor two-phase, but they do not support one homogeneous
\(0.00570\)-BPB effect. The gain shrinks as the tied anchor approaches the
frontier: the retstate family mean is on the same scale as the 0.000924-BPB
same-seed paired-noise SD. Only five of eight Uncheatable observations and
three of six Table-9 observations share a data seed. The source's family-cluster bootstrap has
only three and two clusters; its endpoints equal the extrema of the resampled
family means and are not used here as confidence intervals.

{within_family_table}

The one-seed decoupled Table-9 panel provides a control-sensitive cross-anchor
check. At the weaker t9s05 aggregate, effective exposure, canonical, and
separate heads gain 0.009430, 0.005711, and 0.005376 BPB against their shared
same-seed tied control. At t9b075 the same-seed gains are -0.005396, +0.000840,
and -0.001056 BPB, respectively. The selected t9b075 control was a documented
0.007638-BPB low draw. Re-referencing to its fresh-control mean changes those
three values to +0.002242, +0.008478, and +0.006582 BPB. There are no fresh
t9s05 controls, so this correction is necessarily one-sided. Only effective
exposure then retains the hypothesized weak-anchor advantage; canonical and
separate heads reverse it. From
\(\operatorname{{Var}}(\Delta)=2\sigma^2(1-\rho)\), the available run and
paired-difference SDs imply only \(\rho\approx0.14\)-\(0.41\), so same-seed
pairing does not remove most of the control draw. The present cross-anchor
comparison is therefore inconclusive rather than support for aggregate repair.

### The 300M qsplit phase field is learnable inside support

A nine-feature grouped directional model predicts the paired qsplit deltas
well out of fold despite the near-zero population mean. The top-24 realized
gain is a cross-validated selection diagnostic, not a policy-class estimate.
Its failure mode is support extrapolation: most optimized schedules cross at
least one empirical signed-feature bound.

{directional_table}

### Anchor and design uncertainty

The Delphi frontier anchors used in several selected comparisons were themselves
low draws. Fresh controls moved the selected Table-9 anchor from 1.057530 to
\(1.065168\) BPB (run-to-run SD \(0.003121\)) and the selected
Uncheatable anchor from 0.985120 to \(0.986529\) BPB (run-to-run SD
\(0.000963\))
([source](../delphi_3e18_aggressive_phase_asymmetry_results_20260723/report.md)).
Same-seed control comparisons remain useful, but absolute frontier claims must
not treat the original anchor values as noiseless.

Selection over the aggressive panel is also large enough to explain the
headline 0.010186-BPB Table-9 value under a noise-only approximation. The
same-seed paired-difference SD is 0.004083 BPB. A Blom normal-order-statistic
approximation gives an expected best gain of 0.01057 BPB among all 129
Table-9-anchor treatments, or 0.00881 BPB among the 40 TV-0.10 treatments.
These are descriptive winner's-curse scales, not exact null distributions:
directions share controls and are correlated. The 129-treatment universe is
the primary selection comparison; within the 40-treatment TV-0.10 subset, the
selected gain is only about 0.34 paired-difference SD above the expected
normal-null maximum. The reason to repeat `synthetic_all_plus` is instead its
same-sign result at the Uncheatable frontier aggregate: 0.010651 BPB versus its
same-seed control and 0.006734 BPB versus its 16-run fresh-control mean. That
aggregate is 0.035241 BPB worse than the same-seed Table-9-anchor control and
0.033439 worse in a fresh-mean-to-fresh-mean comparison, so the second gain may
include aggregate repair. Both treatments use data seed 7224001 (seed block 1),
making this cross-aggregate but not cross-seed robustness, and not a second
frontier confirmation.

The whole-swarm versus fixed-aggregate historical SNR contrast is 26.1 versus
1.24 on Uncheatable and 11.4 versus 1.13 on Table-9
([source](../delphi_3e18_fixed_aggregate_phase_snr_20260724/report.md)).
This compares two intervention distributions with very different radii. It
shows that the sampled local phase-order problem is harder to learn; it does not
measure global phase-order headroom. At aggressive radii the odd channel becomes
identifiable, but the direction of benefit remains unknown and the even channel
is predominantly harmful.

## Literature scope relative to the Marin estimand

{literature_table}

### The replay paper's documented midtraining grid omits the tied point

This detail is important because the replay experiment is the closest published
analogue to Marin's fixed-aggregate comparison. In arXiv v1, Appendix D.2
(`Magic number justifications`) documents a base target fraction of
\(1/1024\) and replay values
\(\rho\in\{{0.25,0.5,0.75,0.875\}}\); Appendix E.1.1
(`Mid-training experiments / Fine-tuning baseline / Repetitions`) documents
32 repetitions for the selected midtraining setup; Appendix A defines the
schedule variables. Under that parameterization, the Stage-2 target weight is
\(w_2=1-\rho\), so the documented
effective target fraction is \(\gamma=32/1024=1/32\). If both stages are
parameterized as direct mixture weights, a constant schedule requires

$$
w_1=w_2=\gamma=\frac1{{32}}
\quad\Longrightarrow\quad
\rho=1-\frac1{{32}}=\frac{{31}}{{32}}=0.96875.
$$

Figure 7 therefore holds total target exposure fixed across its sampled
schedules in that setup, but stops before the corresponding tied schedule. Its
highlighted result shows that some sampled schedules beat other sampled
schedules at fixed exposure; it does **not** compare the best phased schedule
with the best constant mixture. This derivation is scoped to the documented
32-repetition midtraining experiment, not the paper's separately tuned
fine-tuning or data-efficiency reference runs.

Exact source locations and the arXiv-v1 source-tar SHA256 are recorded in
`primary_source_provenance.csv`.

The literature makes four distinctions that matter:

1. **Target adaptation versus broad endpoint quality.** Finetuning and
   midtraining studies usually optimize one scarce target after a broad
   pretraining distribution, often after downstream finetuning. Marin optimizes
   broad endpoint BPB with all 39 buckets present in the aggregate.
2. **Trajectory value versus a strict policy-class endpoint gap.** Classical
   curriculum theory can improve convergence trajectories without establishing
   that the best scheduled policy has a different endpoint from the best
   static policy. A benefit that persists to one finite training endpoint is
   evidence for that schedule at that horizon, not a universal strict gap
   between globally optimized policy classes.
3. **Abrupt shift versus continuous replay.** The replay study reports that
   generic replay matters much less when target data is already present in
   Stage 1. A tied mixture is the limiting case where every bucket is replayed
   throughout.
4. **Structured order versus arbitrary phase contrast.** Skill-It and dynamic
   mixture methods exploit prerequisite structure or trajectory feedback. They
   do not predict that random high-dimensional phase asymmetry is beneficial.

There is also affirmative null evidence. *When Do Curricula Work?* finds no
significant standard-regime curriculum benefit and often matches ordered
curricula with random pacing. *Curriculum Learning for Language Modeling*
reports that curricula generally do not improve perplexity.
In *Beyond Random Sampling*, the exact-multiset limited-data scenario shows
clearer early- and mid-training effects than terminal differences, while its
warmup and continual designs report persistent gains up to 3.5% without
preserving the exact same multiset. The Pythia curriculum study likewise finds
capability- and scale-dependent effects. A small fixed-aggregate broad endpoint
effect is therefore consistent with, not extraordinary relative to, the
literature; persistent effects in other exposure designs remain real evidence
for those regimes.

## What StarCoder established

StarCoder established genuine path dependence and a controlled gain for three
selected schedules. It did not establish the continuous policy-class suprema.
The best off-diagonal and \(p^{{(0)}}=0\) schedules were indistinguishable, and
only the sampled tied \(p=0.30\) candidate was repeated. Excluding the
surface-selection seed, all four fresh paired differences favor the phased
candidate: mean gain 0.00579 BPB, paired-\(t\) interval
[0.00126, 0.01033], exact two-sided sign-flip \(p=0.125\). The selected
off-diagonal and tied schedules also have different aggregates, so this is a
fixed-candidate global schedule comparison rather than a fixed-\(a\) order
estimand.

Phase 1 is exactly the WSD decay segment, making data timing collinear with
learning-rate position. The one-seed fixed-dose triplets isolate aggregate
exposure but not that LR-position effect. Late-only code helps by 0.02298 BPB at
one dose and hurts by 0.01591 at another; phase TV also changes from 0.704 to
0.850. The result argues for a dose-dependent \(g(a,d)\), not for a universal
late-data coefficient.

## What can be claimed about Delphi 3e18

Supported:

- Across the sampled designs, aggregate variation is much easier to learn than
  fixed-aggregate order variation. This is a sample-efficiency result, not a
  headroom bound.
- Exposure-matched two-phase siblings improve eight of eight Uncheatable
  comparisons at suboptimal model-proposed aggregates, with a descriptive
  pooled mean of 0.005695 BPB. The family means are 0.009315, 0.005345, and
  0.000790 BPB; the nearest-frontier retstate effect is noise-scale. Table-9
  has opposite-signed family means and is unresolved.
- The strongest selected frontier-comparable Uncheatable gain is 0.002665 BPB
  on one seed. It is not a lower confidence bound.
- A selected synthetic-all Table-9 recipe at the frontier gains 0.010186 BPB
  versus its same-seed control and 0.008071 versus the fresh-control mean at
  TV 0.10. That magnitude is inside the normal-null winner's-curse scale for
  the panel. At the off-Table-9-frontier Uncheatable anchor under the same data
  seed, the recipe gains 0.010651 BPB versus its same-seed control and 0.006734
  versus its fresh-control mean. That anchor is 0.035241 BPB behind the
  same-seed Table-9-anchor control and 0.033439 behind it in a
  fresh-mean-to-fresh-mean comparison. This makes the recipe a concrete
  independent-seed follow-up across aggregate regimes; it is not a second
  frontier confirmation, and the off-frontier gain may include aggregate
  repair.
- Generic large asymmetry has a harmful even cost. At TV 0.5, antithetic pair
  means worsen Uncheatable by 0.00721 and Table-9 by 0.01220; the signed odd
  order RMS values are 0.00331 and 0.00472 BPB. Uncheatable order is robustly
  identifiable; Table-9 order is not after noise-SD uncertainty is propagated.
  No tested orientation beats tied at every radius.
- Conventional 75-100% Dolmino-late schedules are worse on Uncheatable in all
  target-matched repeats at the tested frontier anchor. Table-9 is nearly tied
  at 75% and becomes harmful at 90-100%.
- Effective-exposure ordering across four pairwise aggregate-KL-matched anchors
  shifts Uncheatable favorably by 0.00241 on average, with 19/20 correlated
  candidates improving, but does not produce a comparable Table-9 shift or
  establish a new global frontier.
- Current surrogates overstate phase value: predicted improvement increases
  along epsilon paths where observed performance usually flattens or worsens.
- In-scale 3e18 HPR fitting improves its two-phase proposals relative to a
  300M-source fit, but its raw optima remain optimistic and its validated
  policies do not beat established frontiers; its one-phase Table-9 proposal
  becomes worse.
- The 300M qsplit phase field is learnable out of fold, but optimized candidates
  frequently leave its signed-feature support. This supports a
  learnability-versus-extrapolation diagnosis rather than a no-phase-signal
  diagnosis.
- The longer-horizon Table-9 epsilon path is evaluated on an
  Uncheatable-optimized aggregate. At TPP 4.4 its tied Table-9 score is
  0.032-0.040 BPB behind Table-9 frontier controls, so its larger TPP-10/20
  gains mix horizon dependence with aggregate-dependent behavior.

Not supported:

- A repeatable 0.01 BPB gain at TPP 4.4. One selected Table-9 recipe reaches
  that magnitude, but it has no repeat confirmation.
- A universal Dolmino-late or quality-late rule.
- Equality of the global tied and two-phase optima.
- Treating the best value among dozens of one-seed candidates as an unbiased
  estimate of headroom.

## Recommended presentation language

Use:

> Two-phase policies strictly contain tied policies, and controlled experiments
> show that order effects exist. In the 39-bucket Delphi setting at TPP 4.4,
> however, no frontier-comparable phase-order gain has yet been repeat-confirmed.
> Fixed-exposure gains at suboptimal model-proposed aggregates are
> family-heterogeneous: roughly 0.009, 0.005, and 0.001 BPB as the tied anchor
> approaches the frontier. Generic large asymmetry has a harmful even cost near current
> frontier anchors. One selected Table-9 recipe reaches a 0.0102-BPB
> same-seed gain at low TV, a magnitude inside the panel's expected null
> selection tail. The same recipe has the same sign at an off-Table-9-frontier
> Uncheatable anchor under the same data seed, where its gain is 0.0067 BPB
> against the fresh-control mean. That anchor is 0.033-0.035 BPB worse on
> Table-9 and the gain may include aggregate repair. It is a
> preregistered repeat candidate, not a second frontier confirmation.
> Selected longer-horizon and lower-dimensional experiments show larger
> gains in different or target-mismatched regimes, so the result is
> regime-specific rather than a universal tied-optimality claim; current data
> neither establish nor exclude a repeatable narrow 0.01-BPB frontier
> direction.

Avoid:

- “Two-phase mixing is not better than single-phase.”
- “The StarCoder experiment proved global two-phase superiority.”
- “Curriculum literature contradicts our result.”
- “The absence of a discovered 0.01 gain proves no such direction exists.”

## Remaining decisive experiments

1. **Local frontier quadratic identification.** At the best tied aggregate,
   run 16-24 preregistered family/D-optimal directions as \(\pm d\), at two
   radii, with paired seeds and repeated centers. Fit \(s\) and a low-rank
   \(R\), predict untouched directions, and report a confidence bound on
   \(\max_d[-s^\top d-d^\top Rd/2]\). This directly tests whether local
   \(0.01\)-BPB headroom exists.
2. **Replicated cross-anchor mechanism test.** Repeat the existing one-seed
   t9b075-versus-t9s05 contrast and apply the same preregistered directions at
   the frontier and two deliberately degraded aggregates. Gains proportional
   to anchor error support aggregate repair; a stable gain at every
   anchor supports irreducible chronology.
3. **Repeat the horizon result.** Test the TPP-10/20 winner and exact tied
   control across at least five paired seeds before claiming a phase threshold.
4. **Close the StarCoder tied-line caveat.** Refine \(p=0.25\)-\(0.35\) with
   paired seeds rather than comparing only the sampled \(p=0.30\) point.
5. **LR and optimizer factorial.** Test a promising and null contrast under
   WSD versus cosine, multiple phase boundaries, and retained versus reset
   optimizer moments. Match cumulative LR-weighted exposure where possible.
6. **Component decomposition.** Estimate signed order gradients per benchmark
   and after a common SFT stage. Stable component effects that cancel in the
   macro objective would confirm objective dilution.

## Audit boundaries

- StarCoder 50/50 cosine, StarCoder 80/20 WSD, 60M, 300M, and Delphi differ in
  model size, token budget, phase fractions, LR schedule, and sometimes
  evaluation path. Compare signs and mechanisms, not raw BPB magnitudes.
- The old 300M `279` one-phase artifact is stale; the clean paired causal
  comparison intentionally uses 240 qsplit pairs. Those original two-phase and
  later tied retrain rows nevertheless differ in launcher/export provenance;
  phase-tied baselines show 0.003-0.004 Table-9 offsets. Treat qsplit means as
  provenance-limited, while the directional OOF result remains a useful
  within-design diagnostic.
- The random-population and hybrid checked-in materialization reports are not
  outcome sources; final outcome summaries come from the July 21 PI packet and
  Fieldbook provenance captured there.
- The within-family policy-pair report mislabels BPB 0.982455 as tied; the
  low-epsilon report identifies it as two-phase. The actual selected tied
  Uncheatable frontier is 0.985120; the best within-family two-phase proposal
  is 0.987895, still 0.002775 BPB worse. The exposure-matched sibling deltas do
  not use the bad reference and are included here. Adding a constant to anchor
  error leaves its reported correlation (-0.910) and slope (-0.361) unchanged,
  so those directional diagnostics remain valid but weakly identified across
  three families. The source intercept is contaminated; this audit reports
  re-referenced descriptive intercepts instead.
- The fixed-N TPP report's Finding 5 correctly notes that the policies came
  from an Uncheatable-optimized aggregate and phase path. Its summary table
  then selects the best epsilon separately for Table-9. This audit labels those
  values as post-hoc Table-9 selections within a target-mismatched path, not
  independently Table-9-optimized schedules.
- Delphi's 80/20 data boundary coincides with the onset of WSD decay. Every
  fixed-aggregate phase contrast therefore identifies data order conditional
  on this LR position; it does not separate chronology from LR-weighted
  influence. The Dolmino-late findings are scoped to that coupled schedule.
- The fixed-N maximum simulated-epoch claim is checked against
  `delphi_fixed_n_tpp_phase_sweep_20260712/run_manifest.json`, not inferred
  from the outcome table alone.
- Selected extrema over tens of policies are subject to multiplicity and
  winner's curse. They are labeled exploratory unless repeated or evaluated by
  a frozen selection rule.
"""


def render_html() -> str:
    payload = {
        "evidence": [asdict(row) for row in EVIDENCE],
        "literature": [asdict(row) for row in LITERATURE],
        "claims": CLAIMS,
        "hypotheses": HYPOTHESES,
        "odd_even": ODD_EVEN_DIAGNOSTICS,
        "directional": DIRECTIONAL_DIAGNOSTICS,
        "within_family": WITHIN_FAMILY_DIAGNOSTICS,
    }
    payload_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    claim_cards = "".join(
        f"""
        <article class="claim-card" data-status="{html.escape(row['status'])}">
          <span>{html.escape(row['status'])}</span>
          <h3>{html.escape(row['claim'])}</h3>
          <p>{html.escape(row['basis'])}</p>
          <small>{html.escape(row['boundary'])}</small>
        </article>
        """
        for row in CLAIMS
    )
    literature_cards = "".join(
        f"""
        <article class="paper-card">
          <p class="eyebrow">PRIMARY SOURCE</p>
          <h3><a href="{html.escape(row.url)}">{html.escape(row.paper)}</a></h3>
          <p class="regime">{html.escape(row.regime)}</p>
          <p><strong>Exposure control.</strong> {html.escape(row.exposure_control)}</p>
          <p><strong>Strict-estimand verdict.</strong> {html.escape(row.estimand_verdict)}</p>
          <p><strong>Mechanism.</strong> {html.escape(row.reported_mechanism)}</p>
          <p><strong>Marin implication.</strong> {html.escape(row.implication_for_marin)}</p>
          <details><summary>Scope relative to the Marin estimand</summary>
            <p>{html.escape(row.why_not_a_contradiction)}</p>
          </details>
        </article>
        """
        for row in LITERATURE
    )
    hypothesis_cards = "".join(
        f"""
        <article class="hypothesis-card">
          <p class="eyebrow">{html.escape(row['hypothesis_id'])}</p>
          <h3>{html.escape(row['name'])}</h3>
          <p><strong>Mechanism.</strong> {html.escape(row['mechanism'])}</p>
          <p><strong>Falsifier.</strong> {html.escape(row['falsifiable_signature'])}</p>
          <p><strong>Evidence.</strong> {html.escape(row['current_evidence'])}</p>
          <div>{html.escape(row['assessment'])}</div>
        </article>
        """
        for row in HYPOTHESES
    )
    odd_even_rows = "".join(
        f"""
        <tr>
          <td>{html.escape(str(row["target"]))}</td>
          <td>{row["phase_tv"]:.2f}</td>
          <td>{row["odd_rms_bpb"]:.6f}</td>
          <td>{row["odd_latent_rms_snr"]:.3f}</td>
          <td>{row["odd_snr_at_noise_ci_upper"]:.3f}</td>
          <td>{row["even_mean_cost_bpb"]:+.6f}</td>
          <td>{html.escape(str(row["stable_sign_directions"]))}</td>
          <td>{row["stable_sign_null_p"]:.6f}</td>
          <td>{html.escape(str(row["consistently_better_directions"]))}</td>
        </tr>
        """
        for row in ODD_EVEN_DIAGNOSTICS
    )
    within_family_rows = "".join(
        f"""
        <tr>
          <td>{html.escape(str(row["target"]))}</td>
          <td>{html.escape(str(row["family"]))}</td>
          <td>{row["pairs"]}</td>
          <td>{row["distinct_policy_contrasts"]}</td>
          <td>{row["same_seed_pairs"]}</td>
          <td>{row["mean_gain_bpb"]:+.6f}</td>
          <td>[{row["min_gain_bpb"]:+.6f}, {row["max_gain_bpb"]:+.6f}]</td>
          <td>{row["paired_noise_sd_bpb"]:.5f}</td>
        </tr>
        """
        for row in WITHIN_FAMILY_DIAGNOSTICS
    )
    directional_rows = "".join(
        f"""
        <tr>
          <td>{html.escape(str(row["target"]))}</td>
          <td>{row["rows"]}</td>
          <td>{html.escape(str(row["phase_tv"]))}</td>
          <td>{row["oof_r2"]:.3f}</td>
          <td>{row["oof_spearman"]:.3f}</td>
          <td>{row["oof_sign_accuracy"]:.3f}</td>
          <td>{row["top24_selected_realized_gain_bpb"]:.6f}</td>
        </tr>
        """
        for row in DIRECTIONAL_DIAGNOSTICS
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Phase order: policy-class evidence audit</title>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <script>
    window.MathJax = {{ tex: {{ inlineMath: [['\\\\(','\\\\)']], displayMath: [['$$','$$']] }} }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap');
    :root {{
      --ink:#112c3b; --muted:#61717b; --paper:#f7f2e7; --panel:#fffdf7;
      --line:#d8cfbe; --orange:#d95f32; --green:#1f836f; --gold:#d8a524;
      --red:#b83b42; --blue:#397b9f;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; color:var(--ink); font-family:"IBM Plex Sans",sans-serif;
      background:
        linear-gradient(rgba(17,44,59,.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(17,44,59,.035) 1px, transparent 1px),
        var(--paper);
      background-size:32px 32px;
    }}
    h1,h2,h3 {{ font-family:"Source Serif 4",serif; margin:0; }}
    a {{ color:inherit; text-decoration-color:var(--orange); text-underline-offset:3px; }}
    .hero {{
      min-height:72vh; display:grid; place-items:center; padding:72px 28px;
      background:
        radial-gradient(circle at 78% 20%, rgba(216,165,36,.22), transparent 30%),
        linear-gradient(135deg,#102f40 0%,#173e4e 62%,#245b5c 100%);
      color:#fffdf7;
    }}
    .hero-inner {{ width:min(1180px,100%); }}
    .kicker {{ letter-spacing:.16em; text-transform:uppercase; color:#f1bd54; font-weight:600; }}
    h1 {{ font-size:clamp(3rem,8vw,7.5rem); line-height:.9; max-width:1020px; margin:18px 0 28px; }}
    .hero p {{ font-size:clamp(1.1rem,2vw,1.45rem); line-height:1.55; max-width:880px; color:#dbe6e2; }}
    .verdict {{
      display:grid; grid-template-columns:repeat(3,1fr); gap:1px; margin-top:46px;
      background:#6f827f; border:1px solid #6f827f;
    }}
    .verdict div {{ background:rgba(8,35,46,.88); padding:24px; }}
    .verdict strong {{ display:block; font-family:"Source Serif 4"; font-size:2.2rem; color:#fff; }}
    .verdict span {{ color:#c4d6d2; }}
    main {{ width:min(1240px,calc(100% - 36px)); margin:0 auto; padding:56px 0 100px; }}
    section {{ margin:0 0 72px; }}
    .section-head {{
      display:flex; justify-content:space-between; gap:24px; align-items:end;
      border-bottom:2px solid var(--ink); padding-bottom:14px; margin-bottom:28px;
    }}
    .section-head h2 {{ font-size:clamp(2rem,4vw,3.6rem); }}
    .section-head p {{ max-width:540px; color:var(--muted); line-height:1.55; }}
    .claims {{ display:grid; grid-template-columns:repeat(2,1fr); gap:16px; }}
    .claim-card {{
      background:var(--panel); border:1px solid var(--line); padding:24px;
      box-shadow:4px 4px 0 rgba(17,44,59,.07);
    }}
    .claim-card span {{
      display:inline-block; text-transform:uppercase; letter-spacing:.1em;
      font-size:.72rem; font-weight:600; color:var(--orange);
    }}
    .claim-card h3 {{ font-size:1.55rem; margin:12px 0; }}
    .claim-card p {{ line-height:1.5; }}
    .claim-card small {{ color:var(--muted); line-height:1.45; display:block; }}
    .theory {{ display:grid; grid-template-columns:1.05fr .95fr; background:var(--ink); color:#f8f3e7; }}
    .theory > div {{ padding:36px; }}
    .equation {{
      background:#f4ead3; color:var(--ink); display:flex; flex-direction:column;
      justify-content:center; font-size:1.2rem;
    }}
    .theory h2 {{ font-size:2.8rem; }}
    .theory li {{ margin:14px 0; line-height:1.5; color:#dbe6e2; }}
    .controls {{ display:flex; flex-wrap:wrap; gap:8px; margin:0 0 18px; }}
    button {{
      border:1px solid var(--ink); background:transparent; color:var(--ink);
      padding:9px 14px; font:inherit; cursor:pointer;
    }}
    button.active {{ background:var(--ink); color:#fff; }}
    .plot-shell {{ background:var(--panel); border:1px solid var(--line); padding:18px; min-height:570px; }}
    .axis text {{ fill:var(--muted); font:12px "IBM Plex Sans"; }}
    .axis path,.axis line {{ stroke:#9aa5a7; }}
    .zero {{ stroke:var(--ink); stroke-width:1.5; }}
    .row-label {{ font-size:12px; fill:var(--ink); }}
    .interval {{ stroke:var(--ink); stroke-width:2; }}
    .dot {{ stroke:var(--panel); stroke-width:2; cursor:pointer; }}
    .tooltip {{
      position:fixed; pointer-events:none; max-width:380px; background:#102f40;
      color:#fff; padding:15px 17px; border-left:4px solid #f1bd54;
      opacity:0; z-index:5; box-shadow:0 12px 30px rgba(0,0,0,.25);
    }}
    .tooltip strong {{ display:block; font-family:"Source Serif 4"; font-size:1.1rem; margin-bottom:7px; }}
    .tooltip small {{ color:#c8d8d5; }}
    .papers,.hypotheses {{ display:grid; grid-template-columns:repeat(2,1fr); gap:16px; }}
    .paper-card {{ background:var(--panel); padding:24px; border-top:5px solid var(--orange); }}
    .paper-card h3 {{ font-size:1.5rem; margin:8px 0 18px; }}
    .paper-card p {{ line-height:1.5; }}
    .paper-card .regime {{
      color:var(--muted); font-size:.9rem; border-bottom:1px solid var(--line);
      padding-bottom:14px;
    }}
    .hypothesis-card {{ background:var(--panel); border:1px solid var(--line); padding:24px; }}
    .hypothesis-card h3 {{ font-size:1.55rem; margin:8px 0 16px; }}
    .hypothesis-card p {{ line-height:1.5; }}
    .hypothesis-card div {{
      margin-top:18px; padding:13px 15px; background:#e7efe9;
      border-left:4px solid var(--green); line-height:1.45;
    }}
    .replay-audit {{
      display:grid; grid-template-columns:.72fr 1.28fr;
      border:1px solid var(--line); background:var(--panel);
    }}
    .replay-audit > div {{ padding:30px; }}
    .replay-audit .math-block {{ background:#efe3ca; display:flex; flex-direction:column; justify-content:center; }}
    .replay-audit h3 {{ font-size:2rem; margin-bottom:14px; }}
    .replay-audit p {{ line-height:1.6; }}
    .eyebrow {{ color:var(--orange); letter-spacing:.12em; font-size:.72rem; font-weight:600; }}
    summary {{ cursor:pointer; color:var(--muted); font-weight:600; }}
    .recommendation {{
      background:#e7efe9; border-left:8px solid var(--green); padding:32px;
      font-family:"Source Serif 4"; font-size:clamp(1.4rem,2.5vw,2rem);
      line-height:1.35;
    }}
    .boundary {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
    .boundary article {{ padding:24px; background:var(--panel); border:1px solid var(--line); }}
    .boundary h3 {{ color:var(--orange); font-size:1.6rem; margin-bottom:12px; }}
    .boundary li {{ margin:10px 0; line-height:1.45; }}
    .audit-table {{ width:100%; border-collapse:collapse; background:var(--panel); }}
    .audit-table th,.audit-table td {{
      padding:12px 14px; border:1px solid var(--line); text-align:right; font-variant-numeric:tabular-nums;
    }}
    .audit-table th:first-child,.audit-table td:first-child {{ text-align:left; }}
    .audit-table th {{ background:#e9e1d1; color:var(--ink); font-size:.78rem; letter-spacing:.05em; }}
    .table-scroll {{ overflow-x:auto; border:1px solid var(--line); }}
    footer {{ color:var(--muted); border-top:1px solid var(--line); padding-top:24px; }}
    @media (max-width:800px) {{
      .verdict,.claims,.theory,.papers,.hypotheses,.replay-audit,.boundary {{ grid-template-columns:1fr; }}
      .hero {{ min-height:auto; }} h1 {{ font-size:3.6rem; }}
      .plot-shell {{ overflow-x:auto; }}
    }}
  </style>
</head>
<body>
  <header class="hero">
    <div class="hero-inner">
      <p class="kicker">Policy-class evidence audit · 26 July 2026</p>
      <h1>Phase order matters. The gap is conditional.</h1>
      <p>
        Two-phase mixtures contain tied mixtures, but containment does not imply a large endpoint gap.
        Across Marin and the literature, phase order appears as a harder-to-identify, direction-specific
        residual whose value depends on aggregate exposure, token horizon, objective, and future distribution shift.
      </p>
      <div class="verdict">
        <div><strong>0.00579</strong><span>fresh-seed StarCoder gain for selected schedules</span></div>
        <div>
          <strong>6 / 8</strong>
          <span>distinct contrasts / observations favor two-phase; 5 observations share a seed</span>
        </div>
        <div><strong>0 / 16</strong><span>aggressive directions beat tied at all three radii</span></div>
      </div>
    </div>
  </header>
  <main>
    <section>
      <div class="section-head">
        <h2>Claim ladder</h2>
        <p>
          Mathematical nesting, path dependence, strict superiority, and a practically meaningful gap are separate
          claims.
        </p>
      </div>
      <div class="claims">{claim_cards}</div>
    </section>

    <section class="theory">
      <div>
        <p class="kicker">A falsifiable local model</p>
        <h2>Aggregate plus contrast</h2>
        <ul>
          <li>The signed term prices retention, forgetting, gradient non-commutativity, and recency.</li>
          <li>The even term prices underexposure and repeated-data overload.</li>
          <li>In 38 contrast dimensions, random directions rarely align with a narrow useful gradient.</li>
          <li>
            At a strong tied anchor, the residual phase gradient can be small even when path dependence exists
            elsewhere.
          </li>
        </ul>
      </div>
      <div class="equation">
        $$a=\\alpha_0w^{{(0)}}+\\alpha_1w^{{(1)}},\\qquad d=w^{{(1)}}-w^{{(0)}}$$
        $$L(a,d)=L_0(a)+g(a)^\\top d+\\frac12d^\\top H(a)d+O(\\lVert d\\rVert^3)$$
      </div>
    </section>

    <section>
      <div class="section-head">
        <h2>Competing explanations</h2>
        <p>
          The null, poor-search, aggregate-repair, horizon, objective-dilution, and winner's-curse explanations make
          different predictions.
        </p>
      </div>
      <div class="hypotheses">{hypothesis_cards}</div>
    </section>

    <section>
      <div class="section-head">
        <h2>Evidence field</h2>
        <p>
          Positive values favor two-phase. Even-channel rows show tied minus antithetic pair mean, not signed order.
          Selected minima are shown, but their selection boundary is explicit.
        </p>
      </div>
      <div class="controls" id="objective-controls"></div>
      <div class="plot-shell"><svg id="evidence-chart"></svg></div>
    </section>

    <section>
      <div class="section-head">
        <h2>Odd signal, even cost</h2>
        <p>
          At larger TV, the odd phase-order gradient is robust on Uncheatable but unresolved on Table-9 after
          propagating uncertainty in its repeat-noise estimate. On both targets, tested directions pay a larger even
          cost and fail to maintain a beneficial orientation across radii. The displayed null p is one cross-radius
          Binomial(16, 1/4) statistic repeated on each TV row and assumes independent sign noise across radii.
        </p>
      </div>
      <div class="table-scroll">
        <table class="audit-table">
          <thead><tr>
            <th>Target</th><th>Phase TV</th><th>Odd RMS BPB</th><th>Odd SNR, point noise</th>
            <th>Odd SNR, upper noise bound</th><th>Even mean cost</th><th>Stable sign</th>
            <th>Stable-sign null p</th><th>Beats tied at all radii</th>
          </tr></thead>
          <tbody>{odd_even_rows}</tbody>
        </table>
      </div>
    </section>

    <section>
      <div class="section-head">
        <h2>Fixed-exposure siblings are heterogeneous</h2>
        <p>
          Six distinct Uncheatable policy contrasts represented by eight observations all favor two-phase, but their
          magnitude falls sharply as the aggregate approaches the frontier. Pair ranges are descriptive; the source
          has too few family clusters for an inferential family bootstrap.
        </p>
      </div>
      <div class="table-scroll">
        <table class="audit-table">
          <thead><tr>
            <th>Target</th><th>Family</th><th>Observations</th><th>Distinct contrasts</th><th>Same-seed pairs</th>
            <th>Mean gain BPB</th><th>Pair range BPB</th><th>Paired-noise SD</th>
          </tr></thead>
          <tbody>{within_family_rows}</tbody>
        </table>
      </div>
    </section>

    <section>
      <div class="section-head">
        <h2>Learnable inside support</h2>
        <p>
          The high-TV 300M qsplit panel contains a predictable phase field even though its mean effect is near zero.
          The deployment failure appears when optimization leaves that signed-feature support.
        </p>
      </div>
      <div class="table-scroll">
        <table class="audit-table">
          <thead><tr>
            <th>Target</th><th>Rows</th><th>Phase TV</th><th>OOF R2</th>
            <th>OOF Spearman</th><th>Sign accuracy</th><th>Top-24 realized gain</th>
          </tr></thead>
          <tbody>{directional_rows}</tbody>
        </table>
      </div>
    </section>

    <section>
      <div class="section-head">
        <h2>Inference guardrails</h2>
        <p>The most important qualifications for reading the evidence field.</p>
      </div>
      <div class="boundary">
        <article>
          <h3>Fixed exposure can help off frontier</h3>
          <p>
            Six exposure-matched Uncheatable policy contrasts represented by eight observations all favor two-phase,
            but family means range from
            0.00079 to 0.00932 BPB. Correlation and slope against anchor error are invariant to the source's constant
            reference shift; re-referenced frontier intercepts are only 0.00077 to 0.00128 BPB and remain noise-scale.
          </p>
        </article>
        <article>
          <h3>Frontier controls drift</h3>
          <p>
            Fresh controls move the selected Uncheatable anchor by +0.00141 BPB and the selected Table-9 anchor by
            +0.00764 BPB. The latter change reverses two of three apparent weak-versus-strong aggregate comparisons.
            One-seed frontier differences are not confidence bounds.
          </p>
        </article>
        <article>
          <h3>The 0.0102 gain is selection-scale</h3>
          <p>
            With same-seed difference SD 0.004083 BPB, a normal-null expected best is 0.01057 BPB over 129 treatments
            or 0.00881 over the 40 TV-0.10 treatments. Its magnitude alone is not evidence. The same recipe has the
            same sign at an off-Table-9-frontier anchor under the same data seed, where its gain is 0.0067 BPB against
            the fresh-control mean. That anchor is 0.033-0.035 BPB worse on Table-9 and its gain may include aggregate
            repair.
          </p>
        </article>
        <article>
          <h3>SNR is design-dependent</h3>
          <p>
            Whole-swarm variation is easier to learn than local phase-fiber variation. These SNR values describe two
            intervention distributions; they do not bound global phase-order headroom.
          </p>
        </article>
        <article>
          <h3>Horizon trend is unpaired and target-mismatched</h3>
          <p>
            TPP 10/20 values are one-seed maxima over epsilon; TPP 4.4 uses another data seed. Uncheatable is
            non-monotone, and maximum simulated epochs stay fixed. The Table-9 values follow an
            Uncheatable-optimized aggregate path that is suboptimal for Table-9 at TPP 4.4. Treat this as an
            optimization-time and aggregate-dependent-behavior hypothesis.
          </p>
        </article>
      </div>
    </section>

    <section>
      <div class="section-head">
        <h2>Literature map</h2>
        <p>
          The papers establish mechanisms and regimes, not the same estimand as Marin's broad fixed-aggregate endpoint
          comparison.
        </p>
      </div>
      <div class="replay-audit">
        <div>
          <p class="eyebrow">THE CLOSEST PRIOR DESIGN</p>
          <h3>The documented midtraining grid omits the tied schedule</h3>
          <p>
            In arXiv v1, Appendix D.2 documents a <strong>1/1024</strong> base
            target fraction and a replay grid ending at <strong>0.875</strong>;
            Appendix E.1.1 documents <strong>32 repetitions</strong> for the
            selected midtraining setup, and Appendix A defines the schedule.
            Marin's algebra puts the phase-tied point at
            <strong>0.96875</strong>. Figure 7 shows that sampled schedules
            differ; it does not compare the phased optimum with the constant
            optimum. Exact source provenance is included with the artifact.
          </p>
        </div>
        <div class="math-block">
          $$w_2=1-\\rho,\\qquad \\gamma=\\frac{{1}}{{32}}$$
          $$w_1=w_2=\\gamma\\Longrightarrow\\rho=\\frac{{31}}{{32}}=0.96875$$
        </div>
      </div>
      <div style="height:18px"></div>
      <div class="papers">{literature_cards}</div>
    </section>

    <section>
      <div class="section-head">
        <h2>Safe conclusion</h2>
        <p>A paper-ready statement that preserves both the positive and negative evidence.</p>
      </div>
      <div class="recommendation">
        Two-phase policies strictly contain tied policies, and controlled experiments show that order effects exist.
        In the 39-bucket Delphi setting at TPP 4.4, however, no frontier-comparable phase-order gain has yet been
        repeat-confirmed. Fixed-exposure gains occur at suboptimal model-proposed aggregates, while generic large
        asymmetry has a harmful even cost near current frontier anchors. One selected low-TV Table-9 recipe reaches a
        0.0102-BPB same-seed gain, but that magnitude is inside the panel's expected null selection tail. The same
        recipe has the same sign at an off-Table-9-frontier Uncheatable anchor under the same data seed, where its
        gain is 0.0067 BPB against the fresh-control mean. That anchor is 0.033-0.035 BPB worse on Table-9 and the gain
        may include aggregate repair. This makes it a repeat candidate rather than a second frontier confirmation.
        Current data neither establish nor exclude a repeatable narrow 0.01-BPB frontier direction.
      </div>
    </section>

    <section>
      <div class="section-head"><h2>Decision boundary</h2><p>What the evidence rules in and rules out.</p></div>
      <div class="boundary">
        <article>
          <h3>Supported</h3>
          <ul>
            <li>Path dependence is real.</li>
            <li>Sampled aggregate variation is higher-SNR than local phase-fiber variation.</li>
            <li>Large generic asymmetry has a harmful even cost near current frontier anchors.</li>
            <li>Fixed-exposure order can improve suboptimal model-proposed aggregates.</li>
          </ul>
        </article>
        <article>
          <h3>Not established</h3>
          <ul>
            <li>A repeatable 0.01 BPB Delphi TPP-4.4 gain.</li>
            <li>A universal quality-late or Dolmino-late rule.</li>
            <li>Equality of the global tied and two-phase optima.</li>
            <li>Continuous policy-class superiority from the selected StarCoder candidates.</li>
          </ul>
        </article>
      </div>
    </section>
    <footer>
      Generated from checked-in Marin reports and linked primary sources. See <code>report.md</code>,
      <code>evidence_matrix.csv</code>, <code>odd_even_diagnostics.csv</code>,
      <code>directional_diagnostics.csv</code>, <code>within_family_diagnostics.csv</code>,
      <code>hypothesis_matrix.csv</code>, and
      <code>literature_matrix.csv</code> beside this file. The fresh-seed StarCoder interval is recomputed in
      <code>derived_statistics.json</code>.
    </footer>
  </main>
  <div class="tooltip" id="tooltip"></div>
  <script>
    const DATA = {payload_json};
    const colors = {{
      supported:'#1f836f', suggestive:'#397b9f', exploratory:'#d8a524',
      unresolved:'#8a7d68', null:'#6f7e85', disfavored:'#b83b42'
    }};
    const objectives = ['All', ...new Set(DATA.evidence.map(d => d.objective))];
    let selected = 'All';
    const controls = d3.select('#objective-controls');
    controls.selectAll('button').data(objectives).join('button')
      .classed('active', d => d === selected)
      .text(d => d)
      .on('click', (_, d) => {{
        selected = d;
        controls.selectAll('button').classed('active', x => x === selected);
        draw();
      }});
    function draw() {{
      const data = DATA.evidence.filter(d => selected === 'All' || d.objective === selected);
      const shell = document.querySelector('.plot-shell');
      const width = Math.max(920, shell.clientWidth - 36);
      const row = 34, height = Math.max(520, data.length * row + 110);
      const margin = {{top:26,right:55,bottom:54,left:300}};
      const svg = d3.select('#evidence-chart').attr('width',width).attr('height',height);
      svg.selectAll('*').remove();
      const lo = d3.min(data, d => Math.min(d.gain_bpb, d.ci_low ?? d.gain_bpb));
      const hi = d3.max(data, d => Math.max(d.gain_bpb, d.ci_high ?? d.gain_bpb));
      const bound = Math.max(.006, Math.abs(lo), Math.abs(hi)) * 1.12;
      const x = d3.scaleLinear().domain([-bound,bound]).range([margin.left,width-margin.right]);
      const y = d3.scaleBand()
        .domain(data.map(d => d.evidence_id))
        .range([margin.top,height-margin.bottom])
        .padding(.34);
      svg.append('g').attr('class','axis').attr('transform',`translate(0,${{height-margin.bottom}})`)
        .call(d3.axisBottom(x).ticks(7).tickFormat(d3.format('+.3f')));
      svg.append('line').attr('class','zero').attr('x1',x(0)).attr('x2',x(0))
        .attr('y1',margin.top).attr('y2',height-margin.bottom);
      svg.append('text').attr('x',x(0)-8).attr('y',16).attr('text-anchor','end')
        .attr('fill','#b83b42').text('tied better');
      svg.append('text').attr('x',x(0)+8).attr('y',16).attr('fill','#1f836f').text('two-phase better');
      svg.selectAll('.row-label').data(data).join('text').attr('class','row-label')
        .attr('x',margin.left-14).attr('y',d => y(d.evidence_id)+y.bandwidth()/2+4)
        .attr('text-anchor','end').text(d => `${{d.setting}} · ${{d.objective.replace(' macro BPB','')}}`);
      svg.selectAll('.interval').data(data.filter(d => d.ci_low !== null)).join('line').attr('class','interval')
        .attr('x1',d => x(d.ci_low)).attr('x2',d => x(d.ci_high))
        .attr('y1',d => y(d.evidence_id)+y.bandwidth()/2).attr('y2',d => y(d.evidence_id)+y.bandwidth()/2);
      const tooltip = d3.select('#tooltip');
      svg.selectAll('.dot').data(data).join('circle').attr('class','dot')
        .attr('cx',d => x(d.gain_bpb)).attr('cy',d => y(d.evidence_id)+y.bandwidth()/2)
        .attr('r',7).attr('fill',d => colors[d.status] || '#6f7e85')
        .on('mousemove',(event,d) => {{
          const interval = d.ci_low === null ? 'not estimated' : `[${{d.ci_low.toFixed(6)}}, ${{d.ci_high.toFixed(6)}}]`;
          tooltip.style('opacity',1).style('left',`${{event.clientX+18}}px`).style('top',`${{event.clientY+18}}px`)
            .html(
              `<strong>${{d.comparison}}</strong>` +
              `<div>Gain: ${{d.gain_bpb.toFixed(6)}} BPB · reported interval ${{interval}}</div>` +
              `<div>${{d.sample_size}} · ${{d.design}}</div><small>${{d.inference}}</small>`
            );
        }})
        .on('mouseleave',()=>tooltip.style('opacity',0));
    }}
    draw();
    addEventListener('resize', draw);
  </script>
</body>
</html>
"""


def validate_replay_primary_source() -> None:
    if not REPLAY_SOURCE_TAR.is_file():
        raise FileNotFoundError(f"Missing Replay paper source archive: {REPLAY_SOURCE_TAR}")
    with REPLAY_SOURCE_TAR.open("rb") as handle:
        source_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    if source_sha256 != REPLAY_SOURCE_SHA256:
        raise ValueError(
            f"Replay paper source archive hash drifted: expected {REPLAY_SOURCE_SHA256}, observed {source_sha256}"
        )
    with tarfile.open(REPLAY_SOURCE_TAR, "r:gz") as archive:
        for member_name, expected_fragments in REPLAY_SOURCE_ASSERTIONS.items():
            member = archive.extractfile(member_name)
            if member is None:
                raise FileNotFoundError(f"Replay paper source member is missing: {member_name}")
            source_text = member.read().decode("utf-8")
            missing_fragments = [fragment for fragment in expected_fragments if fragment not in source_text]
            if missing_fragments:
                raise ValueError(
                    f"Replay paper primary-source consistency check failed for {member_name}: {missing_fragments}"
                )


def main() -> None:
    validate_replay_primary_source()
    missing_sources = [row.source for row in EVIDENCE if not (OUTPUT_DIR.parent / row.source).is_file()]
    if missing_sources:
        raise FileNotFoundError(f"Missing evidence sources: {missing_sources}")
    evidence_ids = [row.evidence_id for row in EVIDENCE]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("Evidence identifiers must be unique")
    for source, expected_fragments in SOURCE_ASSERTIONS.items():
        source_text = (OUTPUT_DIR.parent / source).read_text(encoding="utf-8")
        missing_fragments = [fragment for fragment in expected_fragments if fragment not in source_text]
        if missing_fragments:
            raise ValueError(f"Source consistency check failed for {source}: {missing_fragments}")
    derived_statistics = compute_derived_statistics()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "evidence_matrix.csv", [asdict(row) for row in EVIDENCE])
    write_csv(OUTPUT_DIR / "odd_even_diagnostics.csv", ODD_EVEN_DIAGNOSTICS)
    write_csv(OUTPUT_DIR / "directional_diagnostics.csv", DIRECTIONAL_DIAGNOSTICS)
    write_csv(OUTPUT_DIR / "within_family_diagnostics.csv", WITHIN_FAMILY_DIAGNOSTICS)
    write_csv(OUTPUT_DIR / "literature_matrix.csv", [asdict(row) for row in LITERATURE])
    write_csv(OUTPUT_DIR / "primary_source_provenance.csv", PRIMARY_SOURCE_PROVENANCE)
    write_csv(OUTPUT_DIR / "claim_ladder.csv", CLAIMS)
    write_csv(OUTPUT_DIR / "hypothesis_matrix.csv", HYPOTHESES)
    (OUTPUT_DIR / "report.md").write_text(render_report(), encoding="utf-8")
    (OUTPUT_DIR / "phase_order_policy_class_reconciliation.html").write_text(render_html(), encoding="utf-8")
    (OUTPUT_DIR / "derived_statistics.json").write_text(
        json.dumps(derived_statistics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "claim_count": len(CLAIMS),
        "directional_diagnostic_count": len(DIRECTIONAL_DIAGNOSTICS),
        "evidence_count": len(EVIDENCE),
        "hypothesis_count": len(HYPOTHESES),
        "literature_count": len(LITERATURE),
        "odd_even_diagnostic_count": len(ODD_EVEN_DIAGNOSTICS),
        "primary_source_provenance_count": len(PRIMARY_SOURCE_PROVENANCE),
        "within_family_diagnostic_count": len(WITHIN_FAMILY_DIAGNOSTICS),
        "output_dir": str(OUTPUT_DIR),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
