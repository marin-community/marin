# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Build the evidence audit for phase-order scaling and surrogate requirements."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parent
REFERENCE_OUTPUTS: Final = ROOT / "reference_outputs"
OUTPUT_DIR: Final = REFERENCE_OUTPUTS / "two_phase_scaling_mechanistic_framework_20260803"
STARCODER_DIR: Final = REFERENCE_OUTPUTS / "starcoder_wsd80_matched_nd_stage1_20260731" / "optimum_scaling_20260802"
STARCODER_CONFIRMATION_DIR: Final = (
    REFERENCE_OUTPUTS / "starcoder_wsd80_matched_nd_stage1_20260731" / "confirmation_results_20260801"
)
STARCODER_TOKEN_LADDER_DIR: Final = REFERENCE_OUTPUTS / "starcoder_wsd80_token_budget_surfaces_20260731"
FIBER_DIR: Final = REFERENCE_OUTPUTS / "starcoder_wsd80_scale_specific_tied_fibers_20260731" / "results_20260731"
DELPHI_DIR: Final = REFERENCE_OUTPUTS / "delphi_fixed_n_tpp_phase_sweep_results_20260713"
STARCODER_TOKEN_LADDER_TOTAL_PARAMETERS: Final = 157_499_136
STARCODER_TOKEN_LADDER_NON_EMBEDDING_PARAMETERS: Final = 58_998_528
ZOTERO_DATA_MIXTURE_ITEM_COUNT: Final = 77


@dataclass(frozen=True)
class EvidencePoint:
    setting: str
    panel: str
    target: str
    cell_id: str
    intervention: str
    total_parameters: int
    materialized_tokens: int
    total_tpp: float
    non_embedding_tpp: float | None
    phase_gain_bpb: float
    optimum_distance: float | None
    gain_estimator: str
    selection_scope: str
    evidence_level: str
    source: str


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    claim: str
    status: str
    evidence: str
    implication: str
    falsification: str


@dataclass(frozen=True)
class LiteratureSource:
    source_id: str
    citation: str
    title: str
    url: str
    theme: str
    directness: str
    evidence_regime: str
    finding: str
    marin_implication: str
    scope: str
    model_requirement: str
    in_zotero_data_mixture: bool


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_starcoder_points() -> list[EvidencePoint]:
    rows = read_csv(STARCODER_DIR / "discovered_optimum_scaling.csv")
    points: list[EvidencePoint] = []
    rows_by_cell: dict[str, dict[str, str]] = {}
    for row in rows:
        rows_by_cell[row["cell_id"]] = row
        if row["cell_id"].startswith("r0_shared"):
            intervention = "shared base"
        elif "_increase_nd_" in row["cell_id"]:
            intervention = "increase N and D"
        elif "_increase_d_" in row["cell_id"]:
            intervention = "fixed N, increase D"
        elif "_increase_n_" in row["cell_id"]:
            intervention = "fixed D, increase N"
        else:
            raise ValueError(f"Unknown StarCoder intervention for {row['cell_id']}")
        points.append(
            EvidencePoint(
                setting="StarCoder WSD80",
                panel="Matched-compute N-D grid",
                target="Programming Languages BPB",
                cell_id=row["cell_id"],
                intervention=intervention,
                total_parameters=int(row["total_parameters"]),
                materialized_tokens=int(row["materialized_tokens"]),
                total_tpp=float(row["total_parameter_tpp"]),
                non_embedding_tpp=float(row["non_embedding_tpp"]),
                phase_gain_bpb=float(row["discovery_gain_tied_minus_untied_bpb"]),
                optimum_distance=float(row["optimum_l2_distance"]),
                gain_estimator="raw best sampled tied minus raw best sampled untied",
                selection_scope="same-seed surface discovery after dense coordinate search",
                evidence_level="single-reference-seed grid discovery",
                source=str(STARCODER_DIR / "discovered_optimum_scaling.csv"),
            )
        )

    for confirmation in read_csv(STARCODER_CONFIRMATION_DIR / "cell_confirmation_summary.csv"):
        if confirmation["confirmed"] != "True":
            continue
        cell_id = confirmation["cell_id"]
        row = rows_by_cell[cell_id]
        points.append(
            EvidencePoint(
                setting="StarCoder WSD80",
                panel="Matched-compute N-D grid",
                target="Programming Languages BPB",
                cell_id=f"{cell_id}_fresh_seed_confirmation",
                intervention="fixed N, increase D",
                total_parameters=int(row["total_parameters"]),
                materialized_tokens=int(row["materialized_tokens"]),
                total_tpp=float(row["total_parameter_tpp"]),
                non_embedding_tpp=float(row["non_embedding_tpp"]),
                phase_gain_bpb=float(confirmation["mean_gain_bpb"]),
                optimum_distance=float(row["optimum_l2_distance"]),
                gain_estimator="paired fresh-seed mean at the preselected tied and untied coordinates",
                selection_scope=(
                    f"{confirmation['pair_count']} fresh matched seed pairs; candidate selected before these outcomes"
                ),
                evidence_level=(
                    f"fresh-seed confirmation; 95% CI "
                    f"[{float(confirmation['ci95_low']):.6f}, {float(confirmation['ci95_high']):.6f}]"
                ),
                source=str(STARCODER_CONFIRMATION_DIR / "cell_confirmation_summary.csv"),
            )
        )
    return points


def load_starcoder_token_ladder_points() -> list[EvidencePoint]:
    summaries = json.loads((STARCODER_TOKEN_LADDER_DIR / "surface_summary.json").read_text(encoding="utf-8"))
    if len(summaries) != 4:
        raise ValueError(f"Expected four fixed-model token-ladder surfaces, got {len(summaries)}")

    points: list[EvidencePoint] = []
    for summary in summaries:
        token_budget = int(summary["token_budget_requested"])
        tied_weight = float(summary["best_tied_weight"])
        phase_0 = float(summary["best_observed_p0"])
        phase_1 = float(summary["best_observed_p1"])
        points.append(
            EvidencePoint(
                setting="StarCoder WSD80",
                panel="Fixed-157.5M token ladder",
                target="Programming Languages BPB",
                cell_id=f"fixed157m_tokens_{summary['token_budget_label'].lower()}",
                intervention="fixed N=157.5M, increase D",
                total_parameters=STARCODER_TOKEN_LADDER_TOTAL_PARAMETERS,
                materialized_tokens=token_budget,
                total_tpp=float(summary["total_parameter_tpp"]),
                non_embedding_tpp=float(summary["non_embedding_parameter_tpp"]),
                phase_gain_bpb=float(summary["best_tied_bpb"]) - float(summary["best_observed_bpb"]),
                optimum_distance=math.hypot(phase_0 - tied_weight, phase_1 - tied_weight),
                gain_estimator="raw best observed tied minus raw best observed untied",
                selection_scope=(
                    f"{int(summary['coordinate_count'])} observed coordinates in the assembled "
                    f"{summary['token_budget_label']} surface; raw minima are selection-biased"
                ),
                evidence_level="fixed-model token-ladder surface discovery; single-reference-seed raw minima",
                source=str(STARCODER_TOKEN_LADDER_DIR / "surface_summary.json"),
            )
        )
    return points


def load_delphi_points() -> list[EvidencePoint]:
    rows = read_csv(DELPHI_DIR / "frontier_summary.csv")
    points: list[EvidencePoint] = []
    for row in rows:
        target = "Table-9 macro BPB" if row["metric"] == "table9_macro_bpb" else "Uncheatable BPB"
        tpp = float(row["tpp"])
        points.append(
            EvidencePoint(
                setting="Delphi fixed N",
                panel="Delphi epsilon-path sweep",
                target=target,
                cell_id=f"delphi_{row['metric']}_tpp_{tpp:g}",
                intervention="fixed N, increase D",
                total_parameters=358_304_128,
                materialized_tokens=round(tpp * 358_304_128),
                total_tpp=tpp,
                non_embedding_tpp=tpp * 358_304_128 / 128_469_376,
                phase_gain_bpb=float(row["phase_gain_vs_tied"]),
                optimum_distance=None,
                gain_estimator="per-target maximum observed gain over the epsilon path",
                selection_scope="target-matched epsilon selection; one seed per coordinate; cross-TPP unpaired",
                evidence_level="single-seed path; cross-TPP unpaired",
                source=str(DELPHI_DIR / "frontier_summary.csv"),
            )
        )
    return points


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2 + 1
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True))
    x_norm = math.sqrt(sum((a - x_mean) ** 2 for a in x))
    y_norm = math.sqrt(sum((b - y_mean) ** 2 for b in y))
    return numerator / (x_norm * y_norm)


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(average_ranks(x), average_ranks(y))


def hypotheses() -> list[Hypothesis]:
    return [
        Hypothesis(
            "H1",
            "A phase/LR-weighted cumulative dose is sufficient for endpoint loss.",
            "useful null; inadequate globally",
            "A token-fixed fiber changes LR-weighted dose under 80/20 WSD, so it can create apparent ordering effects. But any globally sufficient dose reachable by tied policies cannot produce a strict policy-class advantage; the dense WSD80 surfaces and high-TPP confirmation require testing residual structure beyond this null.",
            "Fit and subtract the best tied-reachable dose null before attributing residual gain to forgetting, consolidation, or noncommutativity.",
            "Show that one response curve of an explicitly defined dose predicts held-out fibers and that its best tied policy matches the best two-phase policy.",
        ),
        Hypothesis(
            "H2",
            "No policy on the globally optimal tied mixture's fiber can improve it.",
            "population unresolved; unsafe constraint",
            "At the finite-grid 2B anchor a=0.35, d=+0.20 improves by 0.003860 BPB on four fresh seeds with 95% interval [0.001545, 0.006175]. Its Holm p is 0.0261 within the four primary anchors but 0.1194 over all 12 repeated arms. At the nearby a=0.40 anchor the repeated effect is null.",
            "Do not enforce fiber optimality. Permit gains inside the tied-optimal uncertainty basin and report sensitivity to how that basin is defined.",
            "Locate the population tied optimum more precisely and repeat a dense feasible fiber with independent seeds.",
        ),
        Hypothesis(
            "H3",
            "Total-parameter TPP alone determines the two-phase advantage.",
            "descriptive, not causal",
            "Across ten WSD80 cells raw discovered gain has Spearman rho about 0.794 with total TPP; smoothing raises it to 0.976. Cells below TPP 3 have raw gains near zero, while every cell above TPP 4.7 is positive. N, D, data reuse, optimizer time, and support remain confounded.",
            "TPP is a useful moderator and design coordinate, not a sufficient state variable or output calibrator.",
            "Cross N and D at overlapping TPP while holding phase schedule, data reuse, and optimizer time fixed.",
        ),
        Hypothesis(
            "H4",
            "At fixed model size, a longer token horizon increases exploitable phase leverage.",
            "threshold supported; smooth law unresolved",
            "WSD80 fixed-N raw gains are 0.00332, 0.00375, 0.00300, and 0.00808 BPB; only the highest-TPP cell has fresh-seed confirmation, 0.00610 [0.00527, 0.00693]. Delphi selected one-seed maxima are nonmonotone for Uncheatable and rise for Table-9, but are unpaired across TPP.",
            "Condition temporal dynamics on horizon and realized repetition, but do not impose a monotone gain law from these selected endpoints.",
            "The TPP-40 39-bucket replay tests whether the same fixed-N direction survives a full heterogeneous swarm.",
        ),
        Hypothesis(
            "H5",
            "Late-stage gradient drift and covariance make mixture scheduling increasingly valuable.",
            "active, unmeasured mechanism",
            "Gradient-noise-scale theory predicts a larger useful batch as loss falls, but that ratio does not imply larger absolute diffusion. LR decay suppresses the diffusion term. No Marin endpoint panel directly measures per-mixture gradient means, covariances, or target alignment.",
            "Use target-aligned drift and curvature-weighted diffusion as measurement targets, not as an established endpoint explanation.",
            "Measure per-bucket gradient means, covariance projections, and target-gradient alignment at matched checkpoints before and during decay.",
        ),
        Hypothesis(
            "H6",
            "Finite-pool repetition and overload create the horizon-dependent phase gain.",
            "exact replay rejected; broader dose role unresolved",
            "The 18-run WSD80 full-pool intervention expanded eligible cache support 773-fold and eliminated exact cache-index wrap. The fixed A-vs-B gap persisted at +0.111431 BPB and increased by +0.032346 relative to the repeated-subset base. But A did not beat the best tied control C: C-A was -0.000655 BPB with 95% interval [-0.001439,+0.000129]. Exact replay is therefore not the main cause of the fixed A-vs-B effect, while semantic reuse, changed physical content, and reoptimized policy-class gain remain unresolved.",
            "Keep bucket-specific finite-dose curvature in the aggregate and even-cost channels, but do not use exact replay as the explanation for global two-phase gain. The temporal channel must explain performance beyond aggregate quality and a tied-reachable dose response.",
            "Fit dense tied and untied surfaces while independently varying unique-token support at a fixed source distribution, then compare reoptimized policy-class gains rather than one fixed A/B contrast.",
        ),
        Hypothesis(
            "H7",
            "Residual strict phase advantage requires structure beyond a tied-reachable dose statistic.",
            "structural requirement",
            "Unequal phase/LR weights create a first-order orientation effect even for commuting updates. After accounting for that null, noncommuting update fields, forgetting, repetition, or state-dependent plasticity can generate residual path dependence.",
            "A promoted dynamic model needs an explicit transition and tied restriction, but only after showing improvement over the phase-weighted-dose null.",
            "Ablate the transition to the best phase-weighted cumulative-dose model and test residual held-out fibers and policy-class selection.",
        ),
        Hypothesis(
            "H8",
            "Phase gain is aggregate- and target-conditioned rather than a universal late-data multiplier.",
            "plausible; not yet identified",
            "The 2B tied-optimal basin contains a replicated gain at a=0.35 and a null repeated effect at a=0.40, but a formal interaction test is weak. RPL's large broad-text gains are a target-specific model failure, not by themselves proof of the true conditioning law.",
            "Condition phase control on aggregate state and target-sensitive response, not only contrast magnitude or late weight.",
            "Use matched fibers across several aggregates and targets with frozen direction selection.",
        ),
        Hypothesis(
            "H9",
            "The WSD80 gain has reached an approximately 0.006 BPB asymptote.",
            "not identified",
            "A Hill sensitivity fit to smoothed gains gives 0.005877 BPB, but its AICc lead is only 0.83, jackknife asymptote spans 0.005035-0.007219, and the sole fresh-seed mean is 0.006101. The fitted ceiling is not an empirical ceiling.",
            "Do not encode a fixed gain ceiling. The current data support saturation as a possibility, not a law.",
            "Add higher-TPP fixed-N cells and independently repeat the selected tied and untied optima.",
        ),
        Hypothesis(
            "H10",
            "Bucket utility can switch on sharply with model capacity or mixture frequency.",
            "externally supported; untested in Marin smooth BPB",
            "Capacity-allocation theory and controlled synthetic mixtures exhibit model-size and mixing-ratio thresholds. Marin has scale-dependent mixture and phase effects, but no bucket-level threshold intervention yet distinguishes a sharp transition from a steep smooth response.",
            "Permit capacity-conditioned aggregate utility and test for thresholds before imposing a discontinuity. Do not represent the effect as a post-hoc scale calibration.",
            "At multiple N and D values, densely sweep the frequency of a fixed bucket or family and compare smooth, threshold, and change-point models on held-out ratios.",
        ),
        Hypothesis(
            "H11",
            "Aggregate response contains stable sparse interactions between bucket families.",
            "plausible; observational support only",
            "External observational work reports math-code complementarity, and controlled mixture studies find non-additive domain effects. Marin has not yet shown that a specific interaction has a stable sign across grouped folds, targets, scales, and interventions.",
            "Low-rank or sparse family interactions are admissible only after a controlled stability audit; unrestricted pairwise terms are not.",
            "Fit preregistered family interactions on one swarm and require sign stability plus improvement on a second swarm or target under blocked CV.",
        ),
        Hypothesis(
            "H12",
            "A static scale-aware aggregate mixture law is sufficient for two-phase policy selection.",
            "rejected structurally",
            "CAMEL, BiMix, AutoScale, and related laws can improve aggregate allocation across N and D, but their inputs contain no phase state or ordering transition. They cannot represent two schedules with the same aggregate and different endpoints unless temporal state is added.",
            "Use these laws as candidates for A(bar w; N, D), never as the complete O/C temporal model.",
            "Evaluate the law on aggregate-matched contrast pairs; identical predictions with reproducibly different outcomes falsify sufficiency directly.",
        ),
        Hypothesis(
            "H13",
            "Lower aggregate prediction error implies better mixture decisions at deployment scale.",
            "rejected as a general rule",
            "DataDecide finds that more elaborate scaling-law baselines do not beat a simple small-scale ranking frontier; Aioli finds that several learned mixture methods do not consistently beat stratified sampling because their law parameters are estimated inaccurately; and Marin repeatedly observes within-panel fit and held-out optimum quality diverge.",
            "Model selection must retain regret, calibration, optimism, and raw-optimum audits in addition to RMSE.",
            "Use a sealed cross-scale candidate panel and compare paired deployment regret among models selected without its outcomes.",
        ),
        Hypothesis(
            "H14",
            "A semantically plausible bucket partition necessarily exposes useful phase order.",
            "rejected as a general rule; unresolved for Marin buckets",
            "Skill-It finds large gains when groups form an ordered skill graph, but instruction-type groups on Alpaca behave nearly like a complete graph and improve validation loss by only 0.007 over random sampling. Marin's 39 labels are data-source and quality buckets, not demonstrated prerequisite skills.",
            "Do not treat bucket names or families as sufficient evidence for temporal structure. A phase-order model should learn or test transfer relations and should shrink toward no order effect when the partition is uninformative.",
            "Estimate a preregistered cross-bucket transfer graph from local interventions, compare semantic and randomized partitions, and require held-out order gains beyond aggregate and repetition effects.",
        ),
        Hypothesis(
            "H15",
            "The timing of specialized data is fully summarized by its cumulative token dose.",
            "rejected externally; unresolved in exact Marin aggregate-matched form",
            "Midtraining Bridges varies start time and mixture weight in from-scratch pretraining and reports that increasing the later mixture does not compensate for delayed introduction. This is close to a dose-compensated temporal intervention, although it is not Marin's exact fixed-aggregate fiber.",
            "The temporal state should expose optimizer age or plasticity in addition to cumulative dose. This is independent support for testing aggregate-conditioned phase control rather than only phase-weighted exposure.",
            "At fixed model, tokens, LR schedule, and specialized-token dose, randomize introduction time and measure both in-domain BPB and retained broad-domain BPB.",
        ),
    ]


def literature_sources() -> list[LiteratureSource]:
    return [
        LiteratureSource(
            "mc-candlish-2018",
            "McCandlish et al. (2018)",
            "An Empirical Model of Large-Batch Training",
            "https://arxiv.org/abs/1812.06162",
            "Optimization dynamics",
            "Conceptual",
            "Image and language-model training; gradient statistics across training.",
            "The gradient noise scale predicts the useful batch-size regime and commonly changes over training.",
            "Measure per-bucket gradient means and covariance late in training; changing mixture may matter when shared drift shrinks relative to between-bucket disagreement.",
            "Noise scale is a ratio, not proof that absolute late-stage diffusion grows, and the paper does not intervene on data order or mixture.",
            "Treat gradient drift and covariance as measured latent-state candidates, not as a TPP-to-gain law.",
            False,
        ),
        LiteratureSource(
            "smith-2017",
            "Smith et al. (2017)",
            "Don't Decay the Learning Rate, Increase the Batch Size",
            "https://arxiv.org/abs/1711.00489",
            "Optimization dynamics",
            "Conceptual",
            "Supervised image training under learning-rate and batch-size schedules.",
            "Batch growth and learning-rate decay can produce related reductions in stochastic-update noise.",
            "The WSD decay window changes the effective stochastic process, so phase utility should be conditioned on LR mass and optimizer time.",
            "This is not a data-mixture or curriculum result and does not show that late data should differ from early data.",
            "Expose LR mass and normalized optimizer time explicitly when comparing schedules.",
            False,
        ),
        LiteratureSource(
            "sweeney-2026",
            "Sweeney (2026)",
            "The Geometry of Sequential Learning: Lie-Bracket Prediction of Transfer Order",
            "https://arxiv.org/abs/2606.24993",
            "Temporal interaction",
            "Conceptual",
            "Short-horizon sequential learning and transfer-order geometry.",
            "Lie-bracket terms characterize order sensitivity when update fields do not commute.",
            "A target-projected noncommutative term is a principled candidate residual after schedule-weighted dose is removed.",
            "The derivation is local and does not establish that Lie brackets explain Marin endpoints; unequal LR weights create a first-order effect first.",
            "Only retain a noncommutative transition if it predicts held-out contrast reversals beyond the cumulative-dose null.",
            False,
        ),
        LiteratureSource(
            "hacohen-2019",
            "Hacohen and Weinshall (2019)",
            "On The Power of Curriculum Learning in Training Deep Networks",
            "https://arxiv.org/abs/1904.03626",
            "Temporal interaction",
            "Conceptual",
            "Supervised curriculum learning with idealized optimization analysis.",
            "Curricula can improve finite-compute trajectories without changing the ideal global minimizer.",
            "Separate transient optimization gains from a strict endpoint policy-class advantage in Marin learning-curve audits.",
            "The setting is not LLM mixture pretraining and does not predict which buckets should appear early or late.",
            "A temporal model should predict trajectories as well as endpoints when intermediate checkpoints exist.",
            True,
        ),
        LiteratureSource(
            "finetuners-fallacy-2026",
            "Finetuner's Fallacy (2026)",
            "The Finetuner's Fallacy: When to Pretrain with Your Finetuning Data",
            "https://arxiv.org/abs/2603.16177",
            "Temporal interaction",
            "Partial",
            "Fine-tuning and specialization with repeated exposure and forgetting.",
            "Early specialization can overfit repeated data while later updates can erase previously acquired capability.",
            "Retention and replay harm are plausible phase states, especially for small finite Dolmino pools.",
            "Fine-tuning from a pretrained model is not from-scratch two-phase pretraining, and its losses do not identify Marin's aggregate-matched order effect.",
            "Keep retained state and repetition cost separate and require each to survive ablation.",
            True,
        ),
        LiteratureSource(
            "replay-pretraining-2026",
            "Replaying Pre-training Data (2026)",
            "Replaying pre-training data improves fine-tuning",
            "https://arxiv.org/abs/2603.04964",
            "Temporal interaction",
            "Partial",
            "From-scratch 150M models trained for 4B tokens under one WSD schedule, with generic and target data arranged in two-stage schedules before evaluation of specialization.",
            "Moving target data earlier while replaying generic pretraining data can improve target-data efficiency and preserve broad capability.",
            "Late broad-data utility can be modeled as interference control rather than assumed waste, and the study is structurally closer to Marin's two-stage WSD setting than its fine-tuning title suggests.",
            "Its target-data budgets and two-stage schedules are not exact aggregate-matched 39-bucket fibers, and it does not identify a universal replay ratio or a global mixture optimum.",
            "Allow late broad-data utility and optimizer-age-dependent specialization, then test both target by target.",
            True,
        ),
        LiteratureSource(
            "prescriptive-repetition-2026",
            "Prescriptive Scaling Laws (2026)",
            "Prescriptive Scaling Laws for Data Constrained Training",
            "https://arxiv.org/abs/2605.01640",
            "Finite data and repetition",
            "Direct aggregate",
            "Finite-data language-model training with repeated examples and weight-decay variation.",
            "Repeated data eventually becomes counterproductive, with the response depending on data and optimization conditions.",
            "Materialized epochs and finite bucket pools need their own state; repetition cannot be replaced by raw mixture weight.",
            "The result concerns aggregate repeated exposure, not whether repetition is better early or late; its threshold is not a universal per-bucket cap.",
            "Use bucket-size-aware dose and a bounded repetition response, conditioned on optimizer regime.",
            True,
        ),
        LiteratureSource(
            "camel-2026",
            "Li et al. (2026)",
            "Capacity-Aware Mixture Law Enables Efficient LLM Data Optimization",
            "https://arxiv.org/abs/2603.08022",
            "Scale-aware aggregate",
            "Direct aggregate",
            "Static five-domain MoE mixtures; 590M-A12M to 7B-A150M fits and a 55B-A1.2B extrapolation.",
            "CAMEL models validation loss with nonlinear model-capacity by mixture interactions and derives compute allocation across fitting scales.",
            "Our aggregate term should allow bucket utility to vary with capacity; a scale-independent 300M mixture law need not transfer to Delphi or production.",
            "CAMEL has no phase state, order, retention, LR schedule, or finite-pool epochs. MoE active capacity is not automatically equivalent to total parameters or TPP, and its loss-to-benchmark map is not an admissible post-hoc BPB correction here.",
            "Benchmark a capacity-conditioned aggregate response A(bar w; N, D) before adding temporal dynamics.",
            True,
        ),
        LiteratureSource(
            "mixing-phase-transitions-2025",
            "Gu et al. (2025)",
            "Data Mixing Can Induce Phase Transitions in Knowledge Acquisition",
            "https://arxiv.org/abs/2505.18091",
            "Capacity thresholds",
            "Direct aggregate",
            "Static synthetic biographies mixed with web data across Pythia 14M-6.9B, plus limited real-data checks.",
            "Knowledge acquisition can switch sharply at model-size and mixing-ratio thresholds because bounded capacity reallocates discontinuously.",
            "Smooth aggregate laws may miss bucket activation thresholds, and proxy-scale recipes can fail when a valuable bucket is below its critical frequency.",
            "The evidence is static mixing and mostly synthetic factual acquisition. It does not explain phase ordering, the WSD80 odd channel, or a 39-bucket endpoint; real heterogeneous mixtures may smooth the transition, and not all experiments were multi-seed.",
            "Test smooth versus thresholded capacity-conditioned bucket utility before encoding a discontinuity.",
            True,
        ),
        LiteratureSource(
            "domain-synergy-2025",
            "Domain-Aware Scaling Laws (2025)",
            "Domain-Aware Scaling Laws Uncover Data Synergy",
            "https://openreview.net/forum?id=Z26PGqEdW7",
            "Aggregate interaction",
            "Observational",
            "Open-weight models with heterogeneous and partly reconstructed pretraining mixtures.",
            "Direct and pairwise domain terms recover patterns such as math-code complementarity and improve prediction over domain-agnostic laws.",
            "A small number of stable family interactions could improve the 39-bucket aggregate response and explain why bucket effects are not additive.",
            "Corpus metadata and model recipes are observationally confounded; the paper does not identify causal interactions or phase order.",
            "Admit only preregistered sparse or low-rank interactions that retain signs across controlled Marin panels.",
            True,
        ),
        LiteratureSource(
            "mixture-constraints-2026",
            "Mixture Under Data Constraints (2026)",
            "Scaling Laws for Mixture Pretraining Under Data Constraints",
            "https://arxiv.org/abs/2605.12715",
            "Finite data and repetition",
            "Direct aggregate",
            "More than 2,000 static target-plus-generic runs, 101M-805M models, multiple finite target pools.",
            "Optimal target exposure depends on pool size, compute, and model scale; generic data regularizes reuse and target corpora can sometimes be repeated 15-20 times.",
            "This directly supports bucket-size-aware materialized epochs and a scale-conditioned repetition response in A and the even asymmetry cost.",
            "The two-source static design does not identify phase placement, and 15-20 repeats is not a universal safe range for every Marin bucket or target.",
            "Parameterize shortage and replay harm by realized epochs, pool size, N, and D rather than one global rho.",
            True,
        ),
        LiteratureSource(
            "bimix-2024",
            "BiMix (2024)",
            "BiMix: A Bivariate Data Mixing Law for Language Model Pretraining",
            "https://arxiv.org/abs/2405.14908",
            "Scale-aware aggregate",
            "Direct aggregate",
            "Static mixtures with joint variation in domain proportions and total data volume.",
            "A bivariate law can interpolate and extrapolate aggregate mixture loss accurately in the reported low-dimensional settings.",
            "BiMix is a useful aggregate baseline for whether explicit D by mixture structure beats current Marin aggregate heads.",
            "Its strong reported fit does not establish optimum-region calibration in 39 dimensions, cross-scale policy regret, or temporal-order effects; entropy proxies are corpus heuristics rather than phase mechanisms.",
            "Compare it as A(bar w; D), with deployment regret and support audits rather than RMSE alone.",
            True,
        ),
        LiteratureSource(
            "autoscale-2024",
            "AutoScale (2024)",
            "AutoScale: Scale-Aware Data Mixing for Pre-Training LLMs",
            "https://arxiv.org/abs/2407.20177",
            "Scale-aware aggregate",
            "Direct aggregate",
            "Static GPT-2 and BERT mixtures optimized at small scales and extrapolated over token budgets.",
            "The best aggregate composition can move with training scale; a two-stage law extrapolates the trend from manageable experiments.",
            "Scale must enter the aggregate component, and fixed small-scale rankings should be treated as hypotheses rather than invariants.",
            "The study primarily varies data scale, not phase order, and does not prove every extrapolated recipe is globally optimal under realistic data constraints.",
            "Require cross-scale aggregate validation before interpreting residual phase gains.",
            True,
        ),
        LiteratureSource(
            "optimal-mixtures-2025",
            "Scaling Laws for Optimal Data Mixtures (2025)",
            "Scaling Laws for Optimal Data Mixtures",
            "https://arxiv.org/abs/2507.09404",
            "Scale-aware aggregate",
            "Direct aggregate",
            "Static language, vision, and multimodal mixtures modeled as functions of N, D, and domain weights.",
            "Joint laws allow compute-dependent optimal mixtures, but simpler additive laws can extrapolate better when the richer law overfits.",
            "This mirrors Marin's fit-versus-optimum tension: aggregate expressivity must earn its complexity on held-out policy selection.",
            "Mixture weights are fixed throughout training; dynamic schedules are explicitly outside scope, so the law cannot explain aggregate-matched phase contrasts.",
            "Use nested aggregate laws and reject terms that improve train fit but worsen held-out regret or optimism.",
            True,
        ),
        LiteratureSource(
            "datadecide-2025",
            "DataDecide (2025)",
            "DataDecide: How to Predict Best Pretraining Data with Small Experiments",
            "https://arxiv.org/abs/2504.11393",
            "Decision validation",
            "Direct decision",
            "25 corpora, 14 model scales, three seeds, up to 1B parameters and 100B tokens.",
            "A single 150M ranking predicts many 1B corpus comparisons, while eight scaling-law baselines do not beat the compute-decision frontier of simple single-scale prediction.",
            "Model sophistication and lower fit error do not guarantee better mixture decisions; Marin should score regret and selected-policy outcomes directly.",
            "The decision is among discrete corpora rather than continuous 39-bucket two-phase policies, so the simple-ranking result is not sufficient for our optimizer.",
            "Keep a simple scale-transfer ranking baseline and require new laws to improve sealed deployment regret.",
            True,
        ),
        LiteratureSource(
            "curriculum-dynamics-2026",
            "Curriculum Dynamics (2026)",
            "Curriculum Learning for LLM Pretraining: An Analysis of Learning Dynamics",
            "https://arxiv.org/abs/2601.21698",
            "Temporal interaction",
            "Direct temporal",
            "Pythia 14M-1B trained for 300B tokens under linguistic curricula, random order, and a reverse-order control.",
            "Curricula alter time spent in shared latent phases; smaller models show larger GNS and output-head stability differences, and direction can matter.",
            "This motivates measuring trajectory state and testing N dependence rather than assuming phase benefit grows universally with scale or TPP.",
            "The curricula and ordering granularity differ from Marin's two blockwise mixture phases, and the paper does not locate a fixed-aggregate policy optimum.",
            "Evaluate candidate temporal states against intermediate checkpoints and reverse-order controls.",
            True,
        ),
        LiteratureSource(
            "two-phase-pretraining-2024",
            "Two-Phase Pretraining (2024)",
            "Maximize Your Data's Potential: Enhancing LLM Accuracy with Two-Phase Pretraining",
            "https://arxiv.org/abs/2412.15285",
            "Temporal intervention",
            "Direct temporal",
            "Manually designed two-phase blends from 1T to 15T tokens and up to 25B parameters.",
            "Two-phase ordering can improve average downstream accuracy, with outcomes depending on data quality, epoch count, phase duration, and scale.",
            "It establishes practical precedent that phase policy can matter and that epoching belongs in the design state.",
            "The blends are not aggregate-matched causal contrasts, the target is broad downstream accuracy rather than smooth BPB, and the work does not solve a 39-dimensional optimum.",
            "Use it to motivate temporal state, not to hard-code a quality-late rule or expected BPB gain.",
            True,
        ),
        LiteratureSource(
            "spark-joy-2024",
            "Late Upsampling (2024)",
            "Does your data spark joy? Performance gains from domain upsampling at the end of training",
            "https://arxiv.org/abs/2406.03476",
            "Temporal intervention",
            "Direct temporal",
            "A 7B model trained for 1T tokens with domain upsampling during the final 5-30% of training.",
            "Late upsampling of selected domains improves difficult benchmarks, with 10-20% reported as the best duration in that setup.",
            "Late high-density data is a plausible policy direction and phase duration can be optimized rather than fixed by convention.",
            "The intervention changes aggregate exposure as well as order and targets benchmark accuracy; it cannot isolate an aggregate-held phase benefit or justify universal Dolmino-late behavior.",
            "Separate aggregate dose from placement and validate both against tied controls.",
            True,
        ),
        LiteratureSource(
            "cmr-2024",
            "CMR Scaling Law (2024)",
            "CMR Scaling Law: Predicting Critical Mixture Ratios for Continual Pre-training of Language Models",
            "https://aclanthology.org/2024.emnlp-main.903/",
            "Continual pretraining",
            "Partial",
            "Continual pretraining with domain data and general-data replay under a capability-preservation constraint.",
            "Loss, token horizon, and mixture ratio exhibit a predictable tradeoff; general replay controls catastrophic forgetting.",
            "Mixture utility can depend on horizon and on a retained-capability constraint, supporting a replay/forgetting state.",
            "The model starts from a pretrained checkpoint and optimizes a constrained multi-objective ratio, not a from-scratch 80/20 schedule or one smooth BPB target.",
            "Do not identify the CMR with Marin's unconstrained optimum; use it as a forgetting-control benchmark.",
            True,
        ),
        LiteratureSource(
            "ado-2024",
            "ADO (2024)",
            "Adaptive Data Optimization: Dynamic Sample Selection with Scaling Laws",
            "https://arxiv.org/abs/2410.11820",
            "Online scheduling",
            "Partial",
            "Online mixture updates during pretraining from per-domain learning-potential estimates.",
            "A time-varying mixture can be driven by current learning state without an external proxy model.",
            "Temporal policy should depend on state, not only initial mixture coordinates; online estimates suggest measurable state variables may replace arbitrary phase coefficients.",
            "ADO is an adaptive sequential algorithm, whereas Marin seeks an offline two-phase surrogate under a fixed training-data budget; its feedback loop cannot be treated as held-out evidence for our policy class.",
            "Use ADO as a state-observation reference, not as a substitute for an offline transition law.",
            True,
        ),
        LiteratureSource(
            "regmix-2024",
            "RegMix (2024)",
            "RegMix: Data Mixture as Regression for Language Model Pre-training",
            "https://arxiv.org/abs/2407.01492",
            "Aggregate optimization",
            "Direct aggregate",
            "Hundreds of static proxy-scale mixtures and nonparametric regression, followed by larger-model validation.",
            "Mixture effects can be predicted empirically and include unintuitive domain interactions; proxy models can produce useful aggregate recipes.",
            "RegMix is a strong practical aggregate baseline and a reminder that qualitative bucket priors are insufficient.",
            "The regressor is nonmechanistic, has no phase state, and relies on ranking transfer that may fail under capacity thresholds or Marin's out-of-support optimum selection.",
            "Compare mechanistic models against RegMix-like decision quality, but do not import its learner as the explanatory surrogate.",
            True,
        ),
        LiteratureSource(
            "olmix-2026",
            "Olmix (2026)",
            "Olmix: A Framework for Data Mixing Throughout LM Development",
            "https://arxiv.org/abs/2602.12237",
            "Aggregate optimization",
            "Direct aggregate",
            "Static mixture optimization under evolving domain sets, data constraints, and practical compute budgets.",
            "Careful design choices, repetition constraints, near-optimum validation, and mixture reuse make learned recipes practical during model development.",
            "Olmix defines the apples-to-apples 280-row aggregate baseline Marin must beat and supports keeping deployment constraints separate from fit evidence.",
            "It does not model phase order or aggregate-matched temporal effects; its sample efficiency does not automatically extend to a doubled policy dimension.",
            "Preserve its held-out and constrained-optimization discipline while adding an identified temporal component.",
            True,
        ),
        LiteratureSource(
            "data-mixing-laws-2024",
            "Data Mixing Laws (2024)",
            "Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance",
            "https://arxiv.org/abs/2403.16952",
            "Scale-aware aggregate",
            "Direct aggregate",
            "Static RedPajama mixtures with nested laws over mixture, steps, and model size; a continual-training extension.",
            "Mixture response is predictable and can be nested with scale laws to recommend larger-run compositions.",
            "This is a candidate aggregate baseline and supports explicit N/D conditioning rather than one scale-invariant response.",
            "Nested extrapolations can compound error, and the endpoint law contains no phase transition or path state; the continual extension does not identify our two-phase optimum.",
            "Audit each nesting level separately and retain policy-regret gates at the final scale.",
            True,
        ),
        LiteratureSource(
            "aioli-2025",
            "Chen et al. (2025)",
            "Aioli: A Unified Optimization Framework for Language Model Data Mixing",
            "https://arxiv.org/abs/2411.05735",
            "Decision validation",
            "Direct decision",
            "Controlled comparison of static and dynamic mixture optimizers, plus an online method that re-estimates its mixing law during training.",
            "Existing methods do not consistently beat stratified sampling because their mixing-law parameters are estimated inaccurately; Aioli beats stratified sampling on all six reported datasets.",
            "A plausible aggregate equation is not enough: parameter identification and the selected policy must be validated against a stratified baseline. Online state can repair scale-transfer errors, but that does not validate a two-phase mechanism.",
            "The outcome is average group test perplexity under aggregate reweighting, not aggregate-matched phase order. Aioli's adaptive feedback policy is also a different policy class from Marin's offline two-block schedule.",
            "Retain stratified and simple static baselines, audit parameter recovery, and separate online adaptation from the offline temporal surrogate.",
            True,
        ),
        LiteratureSource(
            "midtraining-bridges-2026",
            "Liu et al. (2025; v2 2026)",
            "Midtraining Bridges Pretraining and Posttraining Distributions",
            "https://arxiv.org/abs/2510.14865",
            "Temporal intervention",
            "Direct temporal",
            "From-scratch pretraining with specialized-data midtraining; timing-by-weight ablations use 70M and 160M models and vary introduction from 12B to 105B tokens and specialized weight from 10% to 80%.",
            "Timing and mixture weight interact: high specialized weight works early but degrades late, and increasing later mixture does not compensate for delayed introduction.",
            "This is direct evidence against cumulative specialized dose as a sufficient temporal state. Marin should test an optimizer-age or plasticity gate and a general-data bridge, especially for code, math, and Dolmino-like pools.",
            "The compensation sequence is near dose-matched rather than an exact fixed-aggregate antithetic fiber; endpoints include specialized and retained C4 behavior after fine-tuning, not Marin's global smooth-BPB policy optimum.",
            "Condition phase utility on aggregate, optimizer age, and target distance; preregister an exact dose-matched timing intervention in Marin.",
            True,
        ),
        LiteratureSource(
            "skillit-2023",
            "Chen et al. (2023)",
            "Skill-it! A Data-Driven Skills Framework for Understanding and Training Language Models",
            "https://arxiv.org/abs/2307.14430",
            "Temporal interaction",
            "Partial",
            "Synthetic and real continual-pretraining and fine-tuning experiments with learned prerequisite graphs and online data scheduling.",
            "Ordered skill groups can yield large gains, but arbitrary semantic groups need not: instruction-type groups on Alpaca produce nearly complete transfer and only a 0.007 average validation-loss improvement over random sampling.",
            "The 39 Dolma/Dolmino buckets should not be assumed to be temporal skills merely because they are interpretable. Weak phase signal may reflect an uninformative partition as well as noise or an inadequate model.",
            "The strongest gains are not from-scratch fixed-aggregate 39-bucket BPB optimization, and the learned schedule is online rather than two-block. The negative grouping result is a boundary condition, not proof that Marin's buckets are uninformative.",
            "Estimate and validate sparse cross-bucket transfer structure; shrink phase effects toward zero when groups do not exhibit stable prerequisites.",
            True,
        ),
        LiteratureSource(
            "scaling-data-constrained-2023",
            "Muennighoff et al. (2023; v5 2025)",
            "Scaling Data-Constrained Language Models",
            "https://arxiv.org/abs/2305.16264",
            "Finite data and repetition",
            "Direct aggregate",
            "Four hundred runs up to 9B parameters and 900B training tokens, varying unique-data constraints, repetition, and compute.",
            "At fixed compute, up to four epochs of repeated data cause negligible loss change relative to unique data, but the value of further repeated tokens eventually decays toward zero.",
            "This is the canonical reason to express finite-pool use in realized epochs and to allow replay value to decay with model and horizon. It anchors Marin's per-bucket repetition response below newer high-repeat studies.",
            "The result aggregates repetitions over training and does not identify early-versus-late placement, bucket-specific safe epoch counts, or phase-order benefit.",
            "Fit a bounded, scale-conditioned repetition response in the aggregate/even channel before attributing residuals to temporal order.",
            True,
        ),
        LiteratureSource(
            "doremi-2023",
            "Xie et al. (2023)",
            "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining",
            "https://arxiv.org/abs/2305.10429",
            "Aggregate optimization",
            "Direct aggregate",
            "A 280M group-DRO proxy chooses static domain weights for an 8B model on The Pile and GLaM data.",
            "Proxy-derived domain weights improve reported perplexity and downstream accuracy and reach the default-mixture baseline in fewer steps.",
            "DoReMi is a canonical proxy-scale aggregate baseline and supports robust multi-domain objectives, but Aioli shows that this family is sensitive to inaccurate law parameters.",
            "The chosen weights are static during the full run; there is no phase contrast, finite-pool state, or proof that the same proxy ranking transfers to Marin's scale and target.",
            "Include group-robust aggregate baselines and test scale transfer directly; do not treat proxy success as a temporal mechanism.",
            True,
        ),
        LiteratureSource(
            "doge-2024",
            "Fan et al. (2024)",
            "DoGE: Domain Reweighting with Generalization Estimation",
            "https://arxiv.org/abs/2310.15393",
            "Aggregate optimization",
            "Direct aggregate",
            "A bilevel proxy estimates static domain weights to improve in-distribution and out-of-domain generalization on SlimPajama.",
            "Gradient-based generalization estimates can recover inter-domain dependencies and select useful aggregate proportions for a larger base model.",
            "Target-conditioned gradient information is a plausible way to reduce the 39-bucket response to a small number of identified interactions, and it supplies a stronger aggregate baseline than independent bucket utilities.",
            "The learned result is still a static aggregate recipe transferred from a proxy; it does not model block order, WSD optimizer age, or aggregate-matched phase effects.",
            "Use target-projected gradient features only if they improve blocked policy selection and retain stable signs across scales.",
            True,
        ),
        LiteratureSource(
            "chinchilla-2022",
            "Hoffmann et al. (2022)",
            "Training Compute-Optimal Large Language Models",
            "https://arxiv.org/abs/2203.15556",
            "Scale foundations",
            "Context",
            "More than 400 dense language-model runs from 70M to 16B parameters and 5B to 500B tokens under compute-budget scaling.",
            "Compute-optimal model and token counts scale approximately equally in the studied regime; the 70B Chinchilla run uses about 20 total tokens per parameter.",
            "Chinchilla supplies a conventional total-TPP reference for Marin's 4.4, 29.8, and higher-TPP settings, but it does not predict phase gain. Departing from its compute-optimal ratio changes optimization and data-reuse regimes simultaneously.",
            "Its law concerns aggregate validation loss and compute allocation, not mixture composition, data order, non-embedding TPP, or WSD decay phases.",
            "Report N and D separately alongside both TPP definitions; never use distance from TPP 20 as a phase-effect equation.",
            True,
        ),
        LiteratureSource(
            "minicpm-wsd-2024",
            "Hu et al. (2024)",
            "MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies",
            "https://arxiv.org/abs/2404.06395",
            "Optimization dynamics",
            "Context",
            "Small-language-model scaling experiments introducing the Warmup-Stable-Decay learning-rate schedule.",
            "WSD separates a long stable segment from a decay segment, supports continued training, and changes the optimizer weighting of data seen in each phase.",
            "Marin's 80/20 phases coincide with distinct LR regimes, so raw token aggregate is not a schedule-weighted dose. LR mass and optimizer age must be explicit when comparing phase policies.",
            "MiniCPM establishes the scheduler and its training utility, not that a particular data mixture should be placed in decay or that WSD creates a strict two-phase data advantage.",
            "Expose LR mass and normalized optimizer time in the transition law and retain a non-WSD control when making causal claims.",
            False,
        ),
        LiteratureSource(
            "olmo3-2025",
            "Team Olmo (2025)",
            "Olmo 3",
            "https://arxiv.org/abs/2512.13961",
            "Production recipe",
            "Direct practice",
            "A 5.5T-5.93T-token pretraining stage followed by 100B-token Dolma 3 Dolmino midtraining for 7B and 32B models, with released data, checkpoints, and ablations.",
            "Olmo 3 reserves small high-value and structured task data for a high-quality midtraining mix targeting code, math, QA, and instruction following; it also retains high-quality broad pretraining data in that mix.",
            "This is the direct production context for Marin's Dolmino buckets and supports testing quality-late plus broad-data bridging rather than a pure specialization phase.",
            "The final recipe is selected through a development pipeline, not an aggregate-matched causal comparison against all tied policies. It demonstrates practical success but does not identify the globally optimal phase schedule.",
            "Represent high-value finite pools, retained broad data, and target tradeoffs explicitly; validate recipe components with controlled tied and phase contrasts.",
            True,
        ),
        LiteratureSource(
            "algorithmic-stability-2015",
            "Alabdulmohsin (2015)",
            "Algorithmic Stability and Uniform Generalization",
            "https://papers.nips.cc/paper_files/paper/2015/hash/6512bd43d9caa6e02c990b0a82652dca-Abstract.html",
            "Statistical foundations",
            "Conceptual",
            "Theory relating stability and uniform generalization for bounded parametric losses.",
            "Stable learning procedures and uniform generalization are closely linked under the paper's assumptions.",
            "The result warns that dependence on individual training examples can harm uniform loss generalization, but it does not establish that bootstrap-stable surrogate parameters or argmins are identified.",
            "The formal learner is permutation-invariant, so the theorem is not evidence that curriculum order helps. Its stability functional and risk guarantee are not bootstrap dispersion or an argmin-location guarantee; using it to justify optimum stability would be a category error.",
            "Continue reporting fold, bootstrap, and optimizer stability as independent empirical diagnostics, without attributing that practice to this theorem.",
            True,
        ),
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    assert rows
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_report(points: list[EvidencePoint], registry: list[Hypothesis], literature: list[LiteratureSource]) -> str:
    starcoder = [
        point
        for point in points
        if point.setting == "StarCoder WSD80" and point.evidence_level == "single-reference-seed grid discovery"
    ]
    confirmation = next(point for point in points if "fresh-seed confirmation" in point.evidence_level)
    raw_rho = spearman([point.total_tpp for point in starcoder], [point.phase_gain_bpb for point in starcoder])
    fixed_n = sorted(
        [point for point in starcoder if point.intervention in {"shared base", "fixed N, increase D"}],
        key=lambda point: point.total_tpp,
    )
    fixed_d = sorted(
        [point for point in starcoder if point.intervention in {"shared base", "fixed D, increase N"}],
        key=lambda point: point.total_parameters,
    )
    monotone_triplets = 0
    triplet_count = 0
    for rung in ("r1_", "r2_", "r3_"):
        triplet = sorted(
            [point for point in starcoder if point.cell_id.startswith(rung)], key=lambda point: point.total_tpp
        )
        assert len(triplet) == 3
        triplet_count += 1
        monotone_triplets += int(all(a.phase_gain_bpb <= b.phase_gain_bpb for a, b in pairwise(triplet)))
    hypothesis_rows = "".join(f"| {item.hypothesis_id} | {item.status} | {item.claim} |\n" for item in registry)
    literature_rows = "".join(
        (
            f"| [{item.title}]({item.url})<br>{item.citation} | {item.theme}; {item.directness} | {item.finding} | "
            f"{item.marin_implication} | {item.scope} |\n"
        )
        for item in literature
    )
    zotero_source_count = sum(item.in_zotero_data_mixture for item in literature)
    template = """# Phase-order scaling: evidence and mechanistic requirements

## Decision

The current evidence supports a **threshold-like association**, not a universal TPP scaling law. Across ten WSD80 cells, raw discovered policy-class gain has total-TPP Spearman rho `__RAW_RHO__`; smoothing the same surfaces raises rho to `0.9758`. Every raw cell below total TPP 3 is within `0.00065` BPB of zero, while every cell above TPP 4.7 has a positive raw gain of at least `0.00218` BPB. Only the highest-TPP fixed-N cell has independent confirmation: `__CONFIRMATION_GAIN__` BPB with 95% interval `__CONFIRMATION_CI__` over eight fresh seed pairs.

The evidence does not identify TPP itself as the cause. TPP co-varies with token horizon, model size, optimizer time, materialized epochs, and support. The appropriate modeling framework is:

$$
J(\\bar w,\\delta;s)=A(\\bar w;s)+O(\\bar w,\\delta;s)+C(\\bar w,\\delta;s),
$$

where the decomposition into `A`, odd `O`, and even `C` is an **exact accounting identity under contrast reversal**, not yet a mechanism. A mechanistic surrogate must specify a state transition for `s`, first fit the best phase/LR-weighted cumulative-dose null, and add path-dependent structure only when that residual is identified.

## Evidence hierarchy

1. **Fresh-seed confirmation:** one high-TPP WSD80 cell, eight paired seeds, mean gain `__CONFIRMATION_GAIN__` BPB.
2. **Raw grid discovery:** ten WSD80 cells on a shared reference seed. Useful for shape discovery; selected minima are optimistic.
3. **Fixed-model token ladder:** four additional WSD80 surfaces at 1B, 2B, 4B, and 8B tokens with a fixed 157.5M-parameter model. Their raw selected minima are useful contextual evidence, but the coordinate counts differ substantially by horizon and they are not pooled into the matched-grid correlation.
4. **Smoothed surfaces:** useful sensitivity analysis, but smoothing materially strengthens monotonicity and cannot be reported as raw evidence.
5. **Delphi epsilon paths:** per-target maxima over candidate epsilon values, one seed per coordinate and unpaired across TPP. They are hypothesis-generating only.

## Empirical scaling facts

### StarCoder WSD80

- **Fixed N, increasing D:** raw gains across TPP `__FIXED_N_TPP__` are `__FIXED_N_GAIN__` BPB. The high endpoint is confirmed, but the raw path is not monotone.
- **Fixed D, increasing N:** N increases `__FIXED_D_N_START__M -> __FIXED_D_N_END__M`, TPP falls `__FIXED_D_TPP_START__ -> __FIXED_D_TPP_END__`, and raw gain changes `__FIXED_D_GAIN_START__ -> __FIXED_D_GAIN_END__` BPB.
- **Matched empirical compute:** only `__MONOTONE_TRIPLETS__/__TRIPLET_COUNT__` raw triplets order gain monotonically with TPP; smoothed gains give `3/3`. The triplets rule out compute alone as a sufficient descriptor, but cannot separate TPP from D/N because those ranks coincide within each rung.
- **Saturation:** a Hill fit to smoothed gains estimates `0.005877` BPB, but its AICc lead is `0.83`, jackknife asymptotes span `0.005035-0.007219`, and the fresh-seed mean is `__CONFIRMATION_GAIN__`. No finite ceiling is identified.

### Delphi 39-bucket fixed N

- Uncheatable selected gain at TPP `4.4/10/20` is `0.002665/0.001818/0.003383` BPB.
- Table-9 selected gain is `0.002116/0.010103/0.013458` BPB.
- Each value is the per-target maximum over an epsilon path on one seed; cross-TPP comparisons are unpaired and smaller than roughly `1.1` imported repeat-standard-error units except the largest Table-9 effects. These paths do not establish a scaling trend.
- Epsilon `0.1` changes from harmful at TPP 4.4 to useful at TPP 10 and 20, whereas epsilon `0.2` remains harmful. Horizon changes the useful asymmetry range, not simply its amplitude.

### Mechanistic reading of the three scale interventions

- **Increase D at fixed N:** this extends optimizer time and absolute phase duration, raises per-bucket materialized epochs, and gives state-dependent interference or repetition more time to accumulate. TPP rises, but is only a label for this bundle.
- **Increase N at fixed D:** the number of tokens and schedule steps stays fixed while capacity rises. The lower TPP cells remain earlier in their data-limited optimization trajectory, where aggregate-mixture deficiency can dominate the smaller temporal residual.
- **Change N and D at matched empirical compute:** approximately holding the panel's compute convention fixed moves along an N-D tradeoff. Different gains rule out compute alone as a sufficient coordinate, but cannot identify TPP because higher TPP is simultaneously larger D and smaller N.

A transferable surrogate should therefore consume N and D, or their dimensionless consequences such as optimizer horizon and bucket epochs, rather than applying a scalar TPP-to-BPB correction.

## Structural derivations

### 1. The odd/even split is diagnostic, not mechanistic

At a fixed token aggregate, contrast reversal defines exactly

$$
O(\\bar w,\\delta)=\\frac{J(\\bar w,\\delta)-J(\\bar w,-\\delta)}{2},\\qquad
C(\\bar w,\\delta)=\\frac{J(\\bar w,\\delta)+J(\\bar w,-\\delta)}{2}-J(\\bar w,0).
$$

This identity says which response is orientation-sensitive and which is symmetric. It does not explain either term. Because phase durations remain 80/20, `delta -> -delta` is contrast reversal, not a literal exchange of the two time intervals.

### 2. Phase/LR-weighted cumulative dose is the first null

In two-domain WSD80, let `a=0.8 p0+0.2 p1` and `d=p1-p0`. A first-order LR-mass approximation gives early and late normalized masses about `8/9` and `1/9`, so

$$
\\widetilde p=\\frac{8p_0+p_1}{9}=a-\\frac{0.8}{9}d.
$$

Thus a token-aggregate-matched fiber does **not** hold LR-weighted dose fixed. A response `J=F(tilde p)` can produce variation along that fiber without any path dependence. But if a proposed endpoint statistic is globally sufficient and every such statistic is reachable by a tied policy, no oracle `F` on it can produce a strict advantage over the whole tied policy class. The dose model is therefore the necessary local null, not the headline model.

### 3. Unequal schedule weighting precedes the Lie bracket

For short updates with unequal step weights,

$$
U_B^{\\eta_2}\\circ U_A^{\\eta_1}-U_A^{\\eta_2}\\circ U_B^{\\eta_1}
=(\\eta_2-\\eta_1)(g_A-g_B)+\\eta_1\\eta_2(H_Bg_A-H_Ag_B)+O(\\eta^3).
$$

The first term survives even when the update fields commute; it is schedule weighting. The bracket is a candidate residual mechanism after that weighting is modeled. Its target projection can encode interference or state-dependent plasticity, but current endpoint data do not identify it.

### 4. Mixture choice changes target-aligned drift and stochastic diffusion

For `theta+ = theta - eta ghat_q`, with `E[ghat_q]=mu_q` and `Cov(ghat_q)=Sigma_q/B`, a second-order target-loss expansion gives

$$
E[\\Delta L_E]\\approx
-\\eta\\langle\\nabla L_E,\\mu_q\\rangle
+\\frac{\\eta^2}{2}\\left(\\mu_q^T H_E\\mu_q
+\\operatorname{tr}(H_E\\Sigma_q/B)\\right).
$$

The first term rewards target-aligned progress; the second is curvature-weighted drift and diffusion. McCandlish et al. report that gradient noise scale, a ratio governing useful batch size, tends to rise as loss falls. That does not imply larger absolute diffusion, and WSD LR decay suppresses diffusion through `eta^2/B`. This equation defines trajectory measurements to collect; it is order-blind unless `mu_q`, `Sigma_q`, or curvature depend on state.

For bucket gradients `g_i(theta)`, define

$$
\\mu_q=\\sum_i q_i g_i,\\qquad
V_q=\\sum_i q_i\\lVert g_i-\\mu_q\\rVert^2,\\qquad
R_q=\\frac{V_q}{\\lVert\\mu_q\\rVert^2+\\epsilon}.
$$

The proposed **gradient-divergence threshold** is that the shared mean gradient shrinks late while between-bucket disagreement decays more slowly, so relative divergence `R_q` rises. Then changing `q` can materially change target alignment even when the remaining average gradient is small. This is consistent with the literature and the observed horizon threshold, but it is not identified. Measure `mu_q`, `V_q`, per-target alignment, and within-bucket covariance at matched checkpoints; a flat `R_q` or no association with held-out phase effects would reject it.

### 5. Repetition provides a distinct horizon-dependent state

At fixed finite pools, increasing D increases realized epochs. The complete 60M 39-bucket dose intervention identifies strong nonlinear finite-dose response: the frozen signed-dose model reduces OOF RMSE to `0.430` and `0.479` of its signed-linear ablation on Uncheatable and Table-9, while x32 is worse than each bucket's best measured dose for `17/39` buckets on both targets. Dose and overload therefore belong in the aggregate and even-cost channels.

The StarCoder full-pool intervention rejects the narrower claim that exact cache-index replay creates the fixed A-vs-B phase gap. With no exact index wrap, `B-A=+0.111431` BPB and is `+0.032346` larger than in the repeated-subset base. However, the best tied control C matches A within uncertainty: `C-A=-0.000655` BPB with 95% interval `[-0.001439,+0.000129]`. The large A-vs-B contrast is therefore mostly an aggregate-quality comparison, not evidence of a global two-phase advantage. The intervention also changes physical content and does not remove semantic duplication, so broader data-reuse effects remain unresolved.

The resulting requirement is narrower than the original repetition hypothesis: track finite-dose curvature, but require a temporal mechanism to predict reoptimized gains after aggregate quality is controlled. Dense surfaces under independently varied unique-token support are needed to separate repetition, gradient-SNR, optimizer time, and state-dependent interference.

### 6. Capacity-aware mixture laws belong in the aggregate response

CAMEL derives a static validation-loss law of the form

$$
A(\\bar w,M)=C+\\sum_i\\frac{K_i}{\\langle t_i,\\bar w\\rangle^{\\alpha_i}M^{\\beta_i}},
$$

where `M` is the paper's model-capacity coordinate and `t_i` is an inferred intrinsic-domain profile over observed datasets. This is a concrete reason not to assume additive, scale-invariant bucket utility. It is not a phase model: two schedules with the same aggregate receive the same `A`, and the paper's MoE capacity coordinate must not be silently replaced by total parameters or TPP.

The data-mixing phase-transition work derives a single-fact threshold frequency that scales as

$$
f_{\\mathrm{thres}}\\sim M^{-(\\alpha+1)}.
$$

This suggests testing whether some bucket or family contributions activate steeply with capacity and frequency. It does not establish a discontinuity in Marin's smooth aggregate BPB, and it supplies no odd phase-order channel. The immediate modeling implication is therefore to compare smooth capacity interactions, soft activation, and change-point alternatives inside `A`; only then ask whether residual `O` and `C` require temporal state.

## Fiber-optimality verdict

The theorem remains valid under a globally sufficient phase-weighted dose and tied reachability: every two-phase endpoint statistic is matched by a tied policy, so no strict policy-class gain is possible. The empirical fiber statement has three different evidential levels:

- Literal exact-population claim: unresolved because the exact tied optimum and its continuous fiber are unobserved.
- Primary four-anchor grid family: at `a=0.35, d=+0.20`, four fresh seeds improve by `0.003860` BPB with interval `[0.001545, 0.006175]` and Holm-adjusted `p=0.0261`; all five total seeds improve.
- All 12 repeated arms: the same comparison has Holm-adjusted `p=0.1194`. At nearby `a=0.40`, the repeated effect is null. The conclusion is anchor- and multiplicity-sensitive.
- Modeling consequence: allow aggregate-conditioned phase gains within the tied-optimal uncertainty basin. Fiber optimality is a null to test, not a constraint to impose.

## Requirements imposed on the next surrogate

1. **Separate aggregate and temporal identification.** Use joint fitting with blocked folds, orthogonalized phase features, or exact aggregate-matched controls. A naive sequential residual fit is biased when aggregate and contrast co-vary.
2. **Dose null before dynamics.** Fit the best phase/LR-weighted cumulative-dose model first; only residual predictive structure can justify a transition law.
3. **Odd benefit and even cost.** Contrast reversal must expose an odd orientation channel and an even asymmetry/repetition channel.
4. **Explicit transition law when identified.** Use `s_(k+1)=Psi(s_k,q_k,eta_k,xi_k)` and `J=G(s_T)` for retention, forgetting, repetition, or plasticity; remove unidentifiable states.
5. **Aggregate and target conditioning as a tested interaction.** Nearby fibers and RPL negative-control failures motivate it, but do not establish its exact form.
6. **Dimensionless scale variables.** TPP, realized epochs, normalized optimizer time, LR mass, and phase fractions may modulate dynamics. They cannot be a post-hoc BPB calibration layer.
7. **Plausible raw optimization.** Require feasible bounded responses, bootstrap stability, and explicit boundary sensitivity. A boundary optimum is not automatically invalid.
8. **Capacity-conditioned aggregate utility.** Test whether `A` requires nonlinear `N x mixture` structure or bucket activation thresholds before adding temporal coefficients to absorb scale mismatch.
9. **Finite-pool dose, not weight alone.** Shortage and repetition must use bucket size, realized materialized epochs, model capacity, and horizon; a universal epoch cap is not supported.
10. **Sparse interactions only after identification.** Family synergy is plausible, but unrestricted pairwise terms are confounded and have previously harmed optimum-region generalization.
11. **Optimizer age or plasticity as a falsifiable state.** Midtraining compensation failures require testing whether the same specialized dose has different value at different introduction times; this state is retained only if exact dose-matched Marin contrasts support it.
12. **Bucket semantics are not prerequisite structure.** Phase-order coefficients should shrink toward zero unless a bucket or family partition exhibits stable cross-bucket transfer or held-out temporal gain.

## What the TPP-40 swarm can and cannot resolve

The launched 280-row panel holds architecture, policy coordinates, data seeds, phase fractions, and target definitions fixed while increasing the token horizon to total TPP 40. It tests whether the same heterogeneous policy design contains more **learnable phase structure** at a longer horizon; it is not a direct fixed-aggregate causal phase-gain experiment.

Before outcomes are used for model selection, freeze the same folds, metric definitions, model suite, and hyperparameters used at TPP 4.4. Define the primary statistic for each target as

$$
G_{\\mathrm{phase}}=\\mathrm{RMSE}_{\\mathrm{blind}}-\\mathrm{RMSE}_{\\mathrm{aware}},
$$

and compare `G_phase(TPP40)-G_phase(TPP4.4)` under the same grouped-CV construction. Secondary diagnostics are OOF Spearman, Regret@1/@3/@5, low-tail optimism, algebraically tied versus full fits, and bootstrap stability of the raw optimum.

It does **not** identify gradient noise, forgetting, repetition, or noncommutativity by itself. A positive difference supports more learnable temporal structure at the longer horizon. A null rejects transfer of the WSD threshold pattern to this 39-bucket panel, not all phase effects. Predicted policy-class gain is an optimization diagnostic until validated.

## Hypothesis registry

| ID | Status | Claim |
|:--|:--|:--|
__HYPOTHESIS_ROWS__

## Literature synthesis

This audit expands the original seven anchors to `__LITERATURE_COUNT__` primary sources. The Zotero **Data Mixture** collection contains `__ZOTERO_COLLECTION_COUNT__` parent items; `__ZOTERO_SOURCE_COUNT__` mechanism-relevant sources are promoted into the ledger after excluding adjacent infrastructure, generic offline-RL, and non-mixture work. The central distinction is structural:

- **CAMEL and data-mixing phase transitions constrain the aggregate term.** CAMEL supplies a concrete nonlinear capacity-by-mixture law. The phase-transition work supplies a mechanism by which bucket utility can activate sharply with capacity or frequency. Neither contains phase state, order, retention, LR position, or fixed-aggregate contrast.
- **Repetition laws constrain dose accounting.** The finite-data papers make raw mixture weight inadequate: materialized epochs, pool size, model scale, and horizon jointly govern shortage and replay harm. They do not identify whether a repeated token should occur early or late.
- **Midtraining Bridges supplies the closest external temporal intervention.** Its timing-by-weight compensation test rejects cumulative specialized dose as sufficient and motivates optimizer-age or plasticity state. It is near dose-matched, not Marin's exact fixed-aggregate antithetic design, so it constrains the model class without identifying our phase optimum.
- **Curriculum effects depend on the partition.** Skill-It shows both large ordered-skill gains and a near-null result for an uninformative semantic partition. The 39 Dolma/Dolmino buckets need demonstrated transfer structure; readable bucket labels are not enough.
- **Decision papers impose a statistical gate.** Aioli, DataDecide, and Marin show that a plausible law or lower fit error need not yield a better selected mixture. Stratified baselines, policy regret, calibration, optimism, and raw-optimum stability remain primary evidence.
- **Production recipes are precedent, not causal proof.** OLMo 3's 100B-token Dolmino midtraining stage directly motivates finite high-value pools plus retained broad data, but its final recipe does not establish the globally optimal tied or two-phase policy.
- **Interaction evidence is provisional.** Domain synergy and RegMix motivate nonadditive aggregate response, but observational confounding and unstable high-dimensional terms rule out an unrestricted interaction layer.

### Source-level implication and scope ledger

| Source | Evidence type | What it establishes | Marin implication | Scope against what Marin has tried or observed |
|:--|:--|:--|:--|:--|
__LITERATURE_ROWS__

## Evidence boundary

- WSD80 raw gains outside the highest-TPP fixed-N endpoint are selected grid discoveries from a shared reference seed; winner's curse remains possible.
- The pooled TPP relation is confounded by N, D, and branch support.
- Delphi cross-TPP comparisons are single-seed and not fully paired.
- The drift-diffusion and noncommutativity mechanisms are motivated and falsifiable, not identified by current endpoints.
- No existing shared surrogate passes the full cross-target, high-TPP 39-bucket, negative-control, and raw-optimum gates.
"""
    replacements = {
        "__RAW_RHO__": f"{raw_rho:.4f}",
        "__CONFIRMATION_GAIN__": f"{confirmation.phase_gain_bpb:.6f}",
        "__CONFIRMATION_CI__": confirmation.evidence_level.removeprefix("fresh-seed confirmation; 95% CI "),
        "__FIXED_N_TPP__": "/".join(f"{point.total_tpp:.2f}" for point in fixed_n),
        "__FIXED_N_GAIN__": "/".join(f"{point.phase_gain_bpb:.5f}" for point in fixed_n),
        "__FIXED_D_N_START__": f"{fixed_d[0].total_parameters / 1e6:.1f}",
        "__FIXED_D_N_END__": f"{fixed_d[-1].total_parameters / 1e6:.1f}",
        "__FIXED_D_TPP_START__": f"{fixed_d[0].total_tpp:.2f}",
        "__FIXED_D_TPP_END__": f"{fixed_d[-1].total_tpp:.2f}",
        "__FIXED_D_GAIN_START__": f"{fixed_d[0].phase_gain_bpb:.6f}",
        "__FIXED_D_GAIN_END__": f"{fixed_d[-1].phase_gain_bpb:.6f}",
        "__MONOTONE_TRIPLETS__": str(monotone_triplets),
        "__TRIPLET_COUNT__": str(triplet_count),
        "__HYPOTHESIS_ROWS__": hypothesis_rows.rstrip(),
        "__LITERATURE_COUNT__": str(len(literature)),
        "__ZOTERO_COLLECTION_COUNT__": str(ZOTERO_DATA_MIXTURE_ITEM_COUNT),
        "__ZOTERO_SOURCE_COUNT__": str(zotero_source_count),
        "__LITERATURE_ROWS__": literature_rows.rstrip(),
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    return template


def render_html(points: list[EvidencePoint], registry: list[Hypothesis], literature: list[LiteratureSource]) -> str:
    starcoder = [
        point
        for point in points
        if point.setting == "StarCoder WSD80" and point.evidence_level == "single-reference-seed grid discovery"
    ]
    raw_rho = spearman([point.total_tpp for point in starcoder], [point.phase_gain_bpb for point in starcoder])
    confirmation = next(point for point in points if "fresh-seed confirmation" in point.evidence_level)
    observed_scaling_copy = (
        "Every point uses Llama-3-tokenized Paloma Programming Languages BPB. Green circles are the ten-cell "
        "matched-compute N-D grid; orange diamonds are the separate fixed-157.5M token ladder at 1B, 2B, 4B, and "
        "8B tokens. Both show raw best-observed tied-minus-untied minima and are selection-biased. The gold star is "
        "the matched-grid fresh-seed confirmation. Compare trends within a source panel: coordinate coverage and model "
        "size differ between the two discovery panels."
    )
    payload = json.dumps([asdict(point) for point in points], separators=(",", ":"))
    hypothesis_payload = json.dumps([asdict(item) for item in registry], separators=(",", ":"))
    literature_payload = json.dumps([asdict(item) for item in literature], separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Phase-order scaling evidence</title>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <script>
    window.MathJax = {{tex: {{inlineMath: [['\\\\(','\\\\)']], displayMath: [['$$','$$']]}}}};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,500;6..72,700&display=swap');
    :root {{
      --ink:#15303a; --muted:#68777a; --paper:#f4efe4; --panel:#fffdf7;
      --line:#d5ccbc; --orange:#d85c32; --green:#1f806c; --gold:#d4a129;
      --red:#b44144; --blue:#397b9f; --deep:#123746;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; color:var(--ink); font-family:"DM Sans",sans-serif;
      background:linear-gradient(rgba(21,48,58,.035) 1px,transparent 1px),
        linear-gradient(90deg,rgba(21,48,58,.035) 1px,transparent 1px),var(--paper);
      background-size:32px 32px;
    }}
    h1,h2,h3 {{ font-family:"Newsreader",serif; margin:0; }}
    a {{ color:inherit; text-decoration-color:var(--orange); text-underline-offset:3px; }}
    .hero {{
      color:#fff; padding:70px 28px 56px; min-height:68vh; display:grid; place-items:center;
      background:radial-gradient(circle at 78% 18%,rgba(213,162,41,.28),transparent 27%),
        radial-gradient(circle at 10% 90%,rgba(31,128,108,.28),transparent 35%),
        linear-gradient(135deg,#0d2d3b,#174b53 70%,#275d55);
    }}
    .hero-inner {{ width:min(1200px,100%); }}
    .kicker,.eyebrow {{ text-transform:uppercase; letter-spacing:.15em; font-size:.75rem; font-weight:600; color:#f2bc50; }}
    h1 {{ font-size:clamp(3.5rem,8vw,7.7rem); line-height:.9; max-width:1100px; margin:16px 0 24px; }}
    .hero-copy {{ max-width:930px; font-size:clamp(1.08rem,1.9vw,1.38rem); line-height:1.6; color:#dbe7e4; }}
    .verdicts {{ display:grid; grid-template-columns:repeat(3,1fr); margin-top:44px; border:1px solid #6d8785; background:#6d8785; gap:1px; }}
    .verdicts article {{ background:rgba(8,36,46,.9); padding:23px; }}
    .verdicts strong {{ display:block; font:700 2.35rem "Newsreader"; color:#fff; }}
    .verdicts span {{ color:#bfd3cf; line-height:1.4; }}
    main {{ width:min(1260px,calc(100% - 36px)); margin:auto; padding:58px 0 100px; }}
    section {{ margin-bottom:74px; }}
    .section-head {{ display:flex; gap:28px; align-items:end; justify-content:space-between; border-bottom:2px solid var(--ink); padding-bottom:14px; margin-bottom:26px; }}
    .section-head h2 {{ font-size:clamp(2.2rem,4.3vw,3.8rem); }}
    .section-head p {{ max-width:590px; color:var(--muted); line-height:1.55; }}
    .plot-shell {{ background:var(--panel); border:1px solid var(--line); padding:16px; min-height:590px; position:relative; }}
    .controls {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; }}
    button {{ border:1px solid var(--ink); background:transparent; padding:9px 13px; color:var(--ink); font:inherit; cursor:pointer; }}
    button.active {{ background:var(--ink); color:white; }}
    .axis text {{ fill:var(--muted); font:12px "DM Sans"; }}
    .axis path,.axis line {{ stroke:#98a5a5; }}
    .grid line {{ stroke:#dcd5c8; stroke-dasharray:3 5; }}
    .grid path {{ display:none; }}
    .tooltip {{ position:fixed; max-width:390px; pointer-events:none; opacity:0; background:#102f3e; color:white; border-left:4px solid #f2bc50; padding:14px 16px; z-index:20; box-shadow:0 12px 30px rgba(0,0,0,.24); }}
    .tooltip strong {{ display:block; font:700 1.08rem "Newsreader"; margin-bottom:7px; }}
    .tooltip small {{ color:#c7d8d5; line-height:1.5; }}
    .interventions,.requirements,.papers {{ display:grid; grid-template-columns:repeat(3,1fr); gap:15px; }}
    .card {{ background:var(--panel); border:1px solid var(--line); padding:24px; box-shadow:4px 4px 0 rgba(21,48,58,.06); }}
    .card h3 {{ font-size:1.55rem; margin:10px 0 12px; }}
    .card p,.card li {{ line-height:1.53; }}
    .card .number {{ font:700 2.4rem "Newsreader"; color:var(--green); }}
    .mechanism {{ display:grid; grid-template-columns:1.04fr .96fr; background:var(--deep); color:#f9f5ea; }}
    .mechanism > div {{ padding:34px; }}
    .mechanism .math {{ background:#eee1c8; color:var(--ink); display:flex; flex-direction:column; justify-content:center; }}
    .mechanism h2 {{ font-size:2.8rem; }}
    .mechanism li {{ margin:13px 0; line-height:1.5; color:#d6e4e1; }}
    .pipeline {{ display:grid; grid-template-columns:repeat(4,1fr); gap:0; margin-top:16px; border:1px solid var(--line); }}
    .pipeline article {{ background:var(--panel); padding:23px; min-height:180px; border-right:1px solid var(--line); position:relative; }}
    .pipeline article:last-child {{ border-right:0; }}
    .pipeline article:not(:last-child)::after {{ content:'>'; position:absolute; right:-9px; top:44%; width:18px; height:28px; background:var(--orange); color:white; text-align:center; line-height:28px; z-index:2; }}
    .pipeline h3 {{ font-size:1.35rem; margin:8px 0; }}
    .pipeline p {{ font-size:.92rem; color:var(--muted); line-height:1.45; }}
    .fiber {{ display:grid; grid-template-columns:.9fr 1.1fr; background:var(--panel); border:1px solid var(--line); }}
    .fiber > div {{ padding:30px; }}
    .fiber .result {{ background:#e4efe9; border-left:7px solid var(--green); }}
    .fiber h3 {{ font-size:2rem; margin-bottom:12px; }}
    .fiber p,.fiber li {{ line-height:1.55; }}
    .hypothesis-controls {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:15px; }}
    .hypotheses {{ display:grid; grid-template-columns:repeat(2,1fr); gap:14px; }}
    .hypothesis {{ background:var(--panel); border:1px solid var(--line); padding:23px; }}
    .hypothesis h3 {{ font-size:1.45rem; margin:8px 0 11px; }}
    .status {{ display:inline-block; padding:5px 8px; font-size:.7rem; font-weight:600; text-transform:uppercase; letter-spacing:.08em; background:#e6dfd2; }}
    .hypothesis p {{ line-height:1.48; }}
    .hypothesis .implication {{ background:#e8f0eb; border-left:4px solid var(--green); padding:11px 13px; }}
    .requirements {{ grid-template-columns:repeat(2,1fr); }}
    .requirement {{ border-top:5px solid var(--orange); }}
    .papers .card {{ border-top:5px solid var(--green); }}
    .papers {{ grid-template-columns:repeat(2,1fr); }}
    .paper-meta {{ color:var(--muted); font-size:.9rem; }}
    .paper-implication {{ background:#e8f0eb; border-left:4px solid var(--green); padding:11px 13px; }}
    .paper-scope {{ background:#f1e9dc; border-left:4px solid var(--gold); padding:11px 13px; }}
    .literature-summary {{ color:var(--muted); margin:2px 0 18px; }}
    .launch {{ background:#efe1c7; border:1px solid #c6b79e; padding:31px; display:grid; grid-template-columns:1fr 1fr; gap:30px; }}
    .launch h3 {{ font-size:2rem; }}
    .launch p,.launch li {{ line-height:1.55; }}
    footer {{ border-top:1px solid var(--line); color:var(--muted); padding-top:25px; line-height:1.5; }}
    @media(max-width:850px) {{
      .verdicts,.interventions,.requirements,.papers,.pipeline,.fiber,.mechanism,.launch {{ grid-template-columns:1fr; }}
      .hypotheses {{ grid-template-columns:1fr; }}
      .pipeline article {{ border-right:0; border-bottom:1px solid var(--line); }}
      .pipeline article::after {{ display:none; }}
      h1 {{ font-size:3.8rem; }}
    }}
  </style>
</head>
<body>
  <header class="hero">
    <div class="hero-inner">
      <p class="kicker">Mechanistic evidence audit · 3 August 2026</p>
      <h1>Phase value appears after a horizon threshold.</h1>
      <p class="hero-copy">Raw StarCoder discoveries separate low-TPP near-zero gains from positive high-TPP gains, and one high-TPP cell confirms on fresh seeds. TPP is a moderator, not an identified mechanism: LR weighting, optimizer time, repetition, target-aligned gradients, and path-dependent state all move with the design.</p>
      <div class="verdicts">
        <article><strong>{raw_rho:.4f}</strong><span>raw discovery Spearman rho with total TPP; smoothing gives 0.9758</span></article>
        <article><strong>{confirmation.phase_gain_bpb:.5f}</strong><span>BPB fresh-seed gain at the sole confirmed high-TPP cell</span></article>
        <article><strong>unresolved</strong><span>population fiber optimality; unsafe as a hard model constraint</span></article>
      </div>
    </div>
  </header>
  <main>
    <section>
      <div class="section-head"><h2>Observed scaling</h2><p>{observed_scaling_copy}</p></div>
      <div class="plot-shell" id="scaling-chart"></div>
    </section>

    <section>
      <div class="section-head"><h2>What interventions say</h2><p>The direction of intervention matters more than a pooled correlation.</p></div>
      <div class="interventions">
        <article class="card"><p class="eyebrow">Fixed N · increase D</p><p class="number">0.00610</p><h3>The high-horizon endpoint confirms.</h3><p>Raw gains are nonmonotone before the highest rung. More D also means more optimizer steps and materialized epochs, so the confirmation supports a horizon effect without identifying its cause.</p></article>
        <article class="card"><p class="eyebrow">Fixed D · increase N</p><p class="number">near zero</p><h3>Low-TPP cells lose detectable leverage.</h3><p>At fixed tokens, the two largest models have raw gains +0.00065 and -0.00009 BPB. This is compatible with undertraining, but does not distinguish N from TPP.</p></article>
        <article class="card"><p class="eyebrow">Matched empirical compute</p><p class="number">2 / 3</p><h3>Compute alone is insufficient.</h3><p>Only two raw triplets order gain monotonically with TPP; smoothing gives three. Within a rung, TPP rank is also D rank and inverse-N rank, so the causal coordinate remains unresolved.</p></article>
      </div>
    </section>

    <section class="mechanism">
      <div>
        <p class="eyebrow">Minimal decomposition</p>
        <h2>Aggregate response plus temporal control</h2>
        <ul>
          <li><strong>A</strong> prices total aggregate exposure and data shortage.</li>
          <li><strong>O</strong> is odd under contrast reversal and records orientation sensitivity.</li>
          <li><strong>C</strong> is even and prices asymmetry, overload, or concentration.</li>
          <li><strong>The split is an identity.</strong> A model must still explain O and C through dose or state.</li>
        </ul>
      </div>
      <div class="math">
        $$J(\\bar w,\\delta;s)=A(\\bar w;s)+O(\\bar w,\\delta;s)+C(\\bar w,\\delta;s)$$
        $$O(\\bar w,-\\delta;s)=-O(\\bar w,\\delta;s),\\quad C(\\bar w,-\\delta;s)=C(\\bar w,\\delta;s)$$
        $$s_{{k+1}}=\\Psi(s_k,q_k,\\eta_k,\\xi_k),\\quad J=G(s_T)$$
      </div>
    </section>

    <section>
      <div class="section-head"><h2>Mechanistic chain</h2><p>A candidate model must expose each link; endpoint TPP alone cannot stand in for the chain.</p></div>
      <div class="pipeline">
        <article><p class="eyebrow">01 · schedule</p><h3>Data distribution q(t)</h3><p>Mixture weights, finite bucket pools, phase boundary, and LR position determine what gradients are sampled.</p></article>
        <article><p class="eyebrow">02 · stochastic update</p><h3>Drift and covariance</h3><p>Each q changes mean gradient μq and covariance Σq. Late usefulness depends on target alignment and curvature-weighted noise.</p></article>
        <article><p class="eyebrow">03 · state transition</p><h3>Weighting, then interaction</h3><p>Unequal LR mass creates a first-order effect. Residual forgetting, consolidation, plasticity, or repetition require a state-dependent transition.</p></article>
        <article><p class="eyebrow">04 · endpoint</p><h3>Target-sensitive BPB</h3><p>The terminal response must preserve code-positive effects without inventing broad-text gains.</p></article>
      </div>
    </section>

    <section>
      <div class="section-head"><h2>Mechanistic ladder</h2><p>Fit the cumulative-dose null before interpreting residual phase structure as dynamics.</p></div>
      <div class="mechanism">
        <div>
          <p class="eyebrow">Null · schedule-weighted dose</p>
          <h2>Token matching does not match LR dose.</h2>
          $$\\widetilde p=(8p_0+p_1)/9=a-(0.8/9)d$$
          <p>Under 80/20 WSD, a token-fixed contrast shifts first-order LR-weighted exposure. This can explain fiber variation without path dependence, but a tied-reachable dose cannot explain a strict policy-class advantage.</p>
        </div>
        <div class="math">
          <p class="eyebrow">Residual · unequal updates and state</p>
          $$\\Delta U=(\\eta_2-\\eta_1)(g_A-g_B)+\\eta_1\\eta_2(H_Bg_A-H_Ag_B)+O(\\eta^3)$$
          <p>The first term is ordinary schedule weighting and survives commuting fields. The bracket is a candidate residual interaction, not an established Marin mechanism.</p>
        </div>
      </div>
    </section>

    <section class="mechanism">
      <div>
        <p class="eyebrow">Active hypothesis · gradient divergence</p>
        <h2>Shared drift may vanish before domain disagreement.</h2>
        <p>If the common improvement direction shrinks late while bucket gradients remain different, changing the mixture can dominate the small remaining average update. This would create a horizon threshold, not a universal smooth TPP law.</p>
      </div>
      <div class="math">
        $$\\mu_q=\\sum_i q_i g_i,\\quad V_q=\\sum_i q_i\\lVert g_i-\\mu_q\\rVert^2$$
        $$R_q=V_q/(\\lVert\\mu_q\\rVert^2+\\epsilon)$$
        <p>Measure Rq, target-gradient alignment, and within-bucket covariance at matched checkpoints. Current endpoint BPB does not identify this mechanism.</p>
      </div>
    </section>

    <section class="mechanism">
      <div>
        <p class="eyebrow">Aggregate law · capacity aware</p>
        <h2>Scale can change which data is worth learning.</h2>
        <p>CAMEL makes mixture utility nonseparable from model capacity. The phase-transition result goes further: bounded capacity can create a steep activation threshold in source frequency. Both mechanisms belong in the aggregate response; neither predicts temporal order.</p>
      </div>
      <div class="math">
        $$A(\\bar w,M)=C+\\sum_i\\frac{{K_i}}{{\\langle t_i,\\bar w\\rangle^{{\\alpha_i}}M^{{\\beta_i}}}}$$
        $$f_{{\\mathrm{{thres}}}}\\sim M^{{-(\\alpha+1)}}$$
        <p>Test smooth capacity interactions, soft activation, and change points. Do not substitute total parameters or TPP for CAMEL's capacity variable without validation, and do not add an odd phase term from static evidence.</p>
      </div>
    </section>

    <section class="mechanism">
      <div>
        <p class="eyebrow">Temporal evidence · plasticity and grouping</p>
        <h2>The same dose can have different value at different times.</h2>
        <p>Midtraining Bridges reports that a larger late specialized mixture does not compensate for delayed introduction. Skill-It supplies the complementary boundary condition: order helps only when data groups expose real prerequisite structure, not merely readable semantic labels.</p>
      </div>
      <div class="math">
        $$s_{{k+1}}=\\Psi(s_k,q_k,\\eta_k,\\mathrm{{age}}_k),\\qquad J=G(s_T)$$
        <p>For Marin, optimizer age or plasticity and cross-bucket transfer are testable state variables. The external interventions constrain the form, but neither identifies our 39-bucket optimum or licenses arbitrary per-bucket phase coefficients.</p>
      </div>
    </section>

    <section>
      <div class="section-head"><h2>Fiber hypothesis</h2><p>The theorem and the empirical claim are different.</p></div>
      <div class="fiber">
        <div>
          <p class="eyebrow">The null theorem survives</p>
          <h3>Global weighted-dose factorization</h3>
          <p>If every two-phase endpoint statistic is reachable by a tied policy, no oracle response function on that statistic can yield a strict two-phase advantage. This remains a useful falsifiable null.</p>
        </div>
        <div class="result">
          <p class="eyebrow">The empirical statement is unresolved</p>
          <h3>A finite-grid counterexample is multiplicity-sensitive.</h3>
          <p>At a=0.35, d=+0.20 improves by <strong>0.003860 BPB</strong> on four fresh seeds. Holm p is <strong>0.0261</strong> within four primary anchors but <strong>0.1194</strong> across all 12 arms; the nearby a=0.40 repeat is null. Do not impose fiber optimality, and do not claim the exact population fiber is refuted.</p>
        </div>
      </div>
    </section>

    <section>
      <div class="section-head"><h2>Hypothesis registry</h2><p>Filter claims by their current evidential status. “Active” mechanisms are not established explanations.</p></div>
      <div class="hypothesis-controls" id="hypothesis-controls"></div>
      <div class="hypotheses" id="hypothesis-grid"></div>
    </section>

    <section>
      <div class="section-head"><h2>Requirements for the surrogate</h2><p>These are stronger than “fit TPP as another feature.”</p></div>
      <div class="requirements">
        <article class="card requirement"><p class="eyebrow">Identification</p><h3>Orthogonalize aggregate and contrast.</h3><p>Use blocked joint fitting or exact aggregate-matched controls. Sequential residual fitting is biased when aggregate and contrast co-vary.</p></article>
        <article class="card requirement"><p class="eyebrow">Null first</p><h3>Fit phase/LR-weighted dose.</h3><p>Only held-out residual structure beyond the best tied-reachable cumulative dose can justify forgetting, consolidation, or noncommutativity.</p></article>
        <article class="card requirement"><p class="eyebrow">Dynamics</p><h3>Add only identified state.</h3><p>Retention, replay harm, optimizer age, and plasticity need explicit transitions and nested ablations. Bucket labels are not prerequisite graphs; phase terms should shrink toward zero without stable transfer evidence.</p></article>
        <article class="card requirement"><p class="eyebrow">Optimization</p><h3>Audit the feasible raw surface.</h3><p>Require bounded response, bootstrap stability, and boundary sensitivity. A real optimum may lie on a boundary, so interiority is not a gate.</p></article>
      </div>
    </section>

    <section>
      <div class="section-head"><h2>Literature map</h2><p>Every source states its Marin implication and its scope. Static aggregate laws are not treated as evidence for temporal order.</p></div>
      <div class="controls" id="literature-theme-controls"></div>
      <div class="controls" id="literature-directness-controls"></div>
      <p class="literature-summary" id="literature-summary"></p>
      <div class="papers" id="literature-grid"></div>
    </section>

    <section class="launch">
      <div><p class="eyebrow">Running test</p><h3>Delphi 39-bucket · total TPP 40</h3><p>The exact immutable 280-policy, 39-bucket panel is replayed on the same architecture and 80/20 schedule. Only token horizon and token-aware optimizer accounting change.</p></div>
      <div><p class="eyebrow">Frozen interpretation gate</p><ul><li>Primary: the grouped-OOF RMSE advantage of phase-aware over phase-blind fits, compared with TPP 4.4 under identical folds and frozen hyperparameters.</li><li>A positive difference supports more learnable temporal structure, not TPP sufficiency.</li><li>The panel lacks fixed-aggregate controls, so it cannot causally identify phase gain or its physical mechanism.</li></ul></div>
    </section>

    <footer>Generated from persisted StarCoder and Delphi result tables. Source paths, selection scope, and evidence level are in <code>evidence_points.csv</code>; hypothesis decisions are in <code>hypothesis_registry.csv</code>; source-level claims and scope are in <code>literature_ledger.csv</code>.</footer>
  </main>
  <div class="tooltip" id="tooltip"></div>
  <script>
    const evidence = {payload};
    const hypotheses = {hypothesis_payload};
    const literature = {literature_payload};
    const tooltip = d3.select('#tooltip');
    const scalingEvidence = evidence.filter(d => d.target === 'Programming Languages BPB');

    function drawScaling() {{
      const host = d3.select('#scaling-chart'); host.selectAll('*').remove();
      const data = scalingEvidence;
      const width = Math.max(760, host.node().clientWidth - 32), height = 630;
      const margin = {{top:132,right:32,bottom:78,left:78}};
      const svg = host.append('svg').attr('viewBox', `0 0 ${{width}} ${{height}}`).attr('role','img').attr('aria-label','Programming Languages BPB phase gain versus total tokens per parameter across two StarCoder WSD80 source panels');
      const x = d3.scaleLog().domain([d3.min(data,d=>d.total_tpp)*.78,d3.max(data,d=>d.total_tpp)*1.24]).range([margin.left,width-margin.right]);
      const yMin = Math.min(-.0005,d3.min(data,d=>d.phase_gain_bpb)*1.2);
      const yMax = Math.max(.0065,d3.max(data,d=>d.phase_gain_bpb)*1.14);
      const y = d3.scaleLinear().domain([yMin,yMax]).nice().range([height-margin.bottom,margin.top]);
      const isConfirmation = d => d.evidence_level.includes('confirmation');
      const isTokenLadder = d => d.panel === 'Fixed-157.5M token ladder';
      const symbolType = d => isConfirmation(d) ? d3.symbolStar : isTokenLadder(d) ? d3.symbolDiamond : d3.symbolCircle;
      const symbolSize = d => isConfirmation(d) ? 210 : isTokenLadder(d) ? 125 : 105;
      const symbolFill = d => isConfirmation(d) ? '#d4a129' : isTokenLadder(d) ? '#d85c32' : '#1f806c';
      svg.append('g').attr('class','grid').attr('transform',`translate(${{margin.left}},0)`).call(d3.axisLeft(y).ticks(7).tickSize(-(width-margin.left-margin.right)).tickFormat(''));
      svg.append('g').attr('class','axis').attr('transform',`translate(0,${{height-margin.bottom}})`).call(d3.axisBottom(x).ticks(7,'~g'));
      svg.append('g').attr('class','axis').attr('transform',`translate(${{margin.left}},0)`).call(d3.axisLeft(y).ticks(7).tickFormat(d3.format('.3f')));
      svg.append('line').attr('x1',margin.left).attr('x2',width-margin.right).attr('y1',y(0)).attr('y2',y(0)).attr('stroke','#15303a').attr('stroke-width',1.4);
      svg.append('text').attr('x',(margin.left+width-margin.right)/2).attr('y',height-22).attr('text-anchor','middle').attr('font-weight',600).text('Total-parameter tokens per parameter (log scale)');
      svg.append('text').attr('transform','rotate(-90)').attr('x',-(margin.top+height-margin.bottom)/2).attr('y',22).attr('text-anchor','middle').attr('font-weight',600).text('Two-phase gain over tied (Programming Languages BPB)');
      svg.append('g').selectAll('path').data(data).join('path')
        .attr('d',d=>d3.symbol().type(symbolType(d)).size(symbolSize(d))())
        .attr('transform',d=>`translate(${{x(d.total_tpp)}},${{y(d.phase_gain_bpb)}})`)
        .attr('fill',symbolFill).attr('stroke','#15303a').attr('stroke-width',d=>isConfirmation(d)?2.8:1.8).attr('opacity',.92)
        .on('mouseenter',(event,d)=>{{
          tooltip.style('opacity',1).html(`<strong>${{d.cell_id}}</strong><small>${{d.panel}}<br>${{d.setting}} · ${{d.target}}<br>N ${{(d.total_parameters/1e6).toFixed(1)}}M · D ${{(d.materialized_tokens/1e9).toFixed(3)}}B · total TPP ${{d.total_tpp.toFixed(2)}}<br>gain ${{d.phase_gain_bpb.toFixed(6)}} BPB<br>${{d.intervention}}<br>${{d.gain_estimator}}<br>${{d.selection_scope}}<br><em>${{d.evidence_level}}</em></small>`);
          moveTip(event);
        }}).on('mousemove',moveTip).on('mouseleave',()=>tooltip.style('opacity',0));
      const legendX = margin.left;
      const legendY = 16;
      const legendWidth = width - margin.left - margin.right;
      const legend = svg.append('g').attr('transform',`translate(${{legendX}},${{legendY}})`);
      legend.append('rect')
        .attr('width',legendWidth).attr('height',96)
        .attr('fill','#f4efe4').attr('stroke','#d5ccbc').attr('stroke-width',1.2);
      legend.append('text')
        .attr('x',14).attr('y',22).attr('font-size',11).attr('font-weight',700)
        .attr('letter-spacing','1.2px').attr('fill','#68777a').text('MARKER PROVENANCE');
      const legendEntries = [
        {{label:'Matched N-D grid · 10 raw minima',size:85,width:1.8,type:d3.symbolCircle,fill:'#1f806c'}},
        {{label:'Fixed-157.5M token ladder · 4 raw minima',size:100,width:1.8,type:d3.symbolDiamond,fill:'#d85c32'}},
        {{label:'Matched-grid fresh-seed confirmation',size:145,width:2.8,type:d3.symbolStar,fill:'#d4a129'}}
      ];
      legendEntries.forEach((entry,i)=>{{
        const rowY = 42 + i*22;
        legend.append('path').attr('d',d3.symbol().type(entry.type).size(entry.size)()).attr('transform',`translate(18,${{rowY-4}})`).attr('fill',entry.fill).attr('stroke','#15303a').attr('stroke-width',entry.width);
        legend.append('text').attr('x',38).attr('y',rowY).attr('font-size',12).text(entry.label);
      }});
    }}
    function moveTip(event) {{ tooltip.style('left',`${{event.clientX+15}}px`).style('top',`${{event.clientY+15}}px`); }}
    drawScaling();

    const statuses = ['All', ...new Set(hypotheses.map(d=>d.status))];
    let activeStatus='All';
    d3.select('#hypothesis-controls').selectAll('button').data(statuses).join('button')
      .attr('class',d=>d===activeStatus?'active':null).text(d=>d)
      .on('click',(_,d)=>{{activeStatus=d;d3.select('#hypothesis-controls').selectAll('button').classed('active',x=>x===d);drawHypotheses();}});
    function drawHypotheses() {{
      const data=hypotheses.filter(d=>activeStatus==='All'||d.status===activeStatus);
      d3.select('#hypothesis-grid').selectAll('article').data(data,d=>d.hypothesis_id).join(
        enter=>enter.append('article').attr('class','hypothesis').html(d=>`<span class="status">${{d.hypothesis_id}} · ${{d.status}}</span><h3>${{d.claim}}</h3><p>${{d.evidence}}</p><p class="implication"><strong>Model implication.</strong> ${{d.implication}}</p><details><summary>Cheapest falsification</summary><p>${{d.falsification}}</p></details>`),
        update=>update,
        exit=>exit.remove()
      );
    }}
    drawHypotheses();

    const literatureThemes = ['All themes', ...new Set(literature.map(d=>d.theme))];
    const literatureDirectness = ['All evidence', ...new Set(literature.map(d=>d.directness))];
    let activeLiteratureTheme = 'All themes';
    let activeLiteratureDirectness = 'All evidence';
    d3.select('#literature-theme-controls').selectAll('button').data(literatureThemes).join('button')
      .attr('class',d=>d===activeLiteratureTheme?'active':null).text(d=>d)
      .on('click',(_,d)=>{{activeLiteratureTheme=d;d3.select('#literature-theme-controls').selectAll('button').classed('active',x=>x===d);drawLiterature();}});
    d3.select('#literature-directness-controls').selectAll('button').data(literatureDirectness).join('button')
      .attr('class',d=>d===activeLiteratureDirectness?'active':null).text(d=>d)
      .on('click',(_,d)=>{{activeLiteratureDirectness=d;d3.select('#literature-directness-controls').selectAll('button').classed('active',x=>x===d);drawLiterature();}});
    function drawLiterature() {{
      const data=literature.filter(d=>(activeLiteratureTheme==='All themes'||d.theme===activeLiteratureTheme)&&(activeLiteratureDirectness==='All evidence'||d.directness===activeLiteratureDirectness));
      const zoteroCount=data.filter(d=>d.in_zotero_data_mixture).length;
      d3.select('#literature-summary').text(`${{data.length}} sources shown · ${{zoteroCount}} promoted from the 77-item Zotero Data Mixture collection · filters do not change the evidence hierarchy`);
      d3.select('#literature-grid').selectAll('article').data(data,d=>d.source_id).join(
        enter=>enter.append('article').attr('class','card').html(d=>`<p class="eyebrow">${{d.theme}} · ${{d.directness}}</p><h3><a href="${{d.url}}">${{d.citation}}</a></h3><p class="paper-meta">${{d.title}}<br>${{d.evidence_regime}}</p><p><strong>Finding.</strong> ${{d.finding}}</p><p class="paper-implication"><strong>Marin implication.</strong> ${{d.marin_implication}}</p><p class="paper-scope"><strong>Scope.</strong> ${{d.scope}}</p><details><summary>Model requirement</summary><p>${{d.model_requirement}}</p></details>`),
        update=>update,
        exit=>exit.remove()
      );
    }}
    drawLiterature();
  </script>
</body>
</html>
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    points = load_starcoder_points() + load_starcoder_token_ladder_points() + load_delphi_points()
    registry = hypotheses()
    literature = literature_sources()
    write_csv(OUTPUT_DIR / "evidence_points.csv", [asdict(point) for point in points])
    write_csv(OUTPUT_DIR / "hypothesis_registry.csv", [asdict(item) for item in registry])
    write_csv(OUTPUT_DIR / "literature_ledger.csv", [asdict(item) for item in literature])
    (OUTPUT_DIR / "report.md").write_text(render_report(points, registry, literature), encoding="utf-8")
    (OUTPUT_DIR / "phase_order_scaling_framework.html").write_text(
        render_html(points, registry, literature), encoding="utf-8"
    )
    (OUTPUT_DIR / "review_resolution.md").write_text(
        """# Independent review resolution

Two independent Claude Opus 5 reviews were run after the first artifact draft:

- Mechanistic review session: `4fefe503-6607-4eb6-ad06-8b783e3cac93`.
- Statistical review session: `b54a9872-cc9d-46de-acc7-4f1ff79fe153`.

## Accepted corrections

- Raw discovery, smoothed-surface, and fresh-seed-confirmation gains are now separate estimands.
- The raw TPP association replaces the smoothed correlation as the headline statistic.
- Matched-compute evidence is descriptive: only two of three raw triplets are monotone, and TPP cannot be separated from D/N within a rung.
- Delphi epsilon-path maxima are labeled selected, one-seed, and unpaired across TPP.
- Fiber evidence is split into the unresolved population claim, the four-primary-anchor test, and all-12-arm multiplicity.
- Token-fixed contrast is not LR-dose-fixed under WSD. Unequal schedule weighting enters before any Lie-bracket interpretation.
- The odd/even split is presented as an exact accounting identity, not a physical mechanism.
- Gradient noise scale is a motivation for trajectory measurements, not an endpoint explanation.
- Sequential aggregate-then-residual fitting is replaced by blocked joint fitting, orthogonalization, or exact matched controls.
- Raw optima require boundary sensitivity and bootstrap stability; interiority is not a universal requirement.

## Reviewer claim corrected during adjudication

The mechanistic reviewer initially claimed that any phase-weighted dose is constant along a token-fixed fiber. That is false unless the phase multipliers coincide with token weights. For WSD80, the first-order LR-weighted coordinate changes as `a-(0.8/9)d`. The valid theorem is narrower: if a dose statistic is globally sufficient and every two-phase dose is reachable by a tied policy, it cannot support a strict policy-class advantage.

## Not promoted to conclusions

- The Hill asymptote is not identified and is not encoded as a gain ceiling.
- Aggregate conditioning is plausible but not established by the neighboring-fiber comparison.
- Lie brackets, gradient diffusion, forgetting, and repetition remain candidate mechanisms rather than identified causes.
- TPP40 is not treated as a fixed-aggregate causal phase experiment.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
