# Handoff: an a priori 280-row Delphi 3e18 swarm for the 39-bucket mixing problem (2026-09-04)

Purpose: replace the current 280-run fit panel with a swarm, fixed before any model is fitted, that gives a
surrogate the evidence it needs to select a better Table-9 / Uncheatable optimum than OLMix fitted on the same
swarm. This document is written for a reviewer who has not followed the project; every claim points at an artifact.
Nothing has been launched. The materialized design is
`experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_apriori_swarm_280_20260904/`
(`swarm_mixtures.csv`, `design_rows.csv`, `manifest.json`), produced by
`design_delphi_apriori_swarm_280_20260904.py` from a fixed seed.

## 1. What is wrong with the current swarm (evidence from real runs)

The current panel is 280 rows: proportional, UniMax and uniform baselines, 39 leave-one-bucket-out deletions from
the proportional control, and 238 random spread mixtures. Fitted on it, the best surrogate we have
(`weibull_softplus_unscaled`, per-bucket Weibull benefit and softplus harm in materialized epochs) proposed the
Table-9 cap-6/7/8 mixtures that were validated at data seed 662009 against the OLMix KL=0.005 cap-4 mixture:
OLMix 1.0769, WSPU 1.0727 / 1.0678 / 1.0795. The surrogate had predicted a 0.074–0.079 gain and got 0.004–0.009.
Round 5 (`single_phase_observatory_round5_olmix_gap_report_20260904.md`) traced that to four properties of the
training data, not of the model class:

1. **No support where the good mixtures are.** The panel's mixtures are all spread: 22–39 effective buckets, never
   below 61% Common Crawl (design_rows.csv, block `reused_*`), synthetic QA never above 2.3 epochs, OLMOCR never
   above 3.5, stack above 6 epochs in three rows. The ten best measured Table-9 mixtures in the 247-coordinate
   development bank sit at 17–23 effective buckets, 23–32% CC-high, synthetic QA 0.10–0.20, stack 0.18–0.30, with
   small clean pools (wikipedia, stem-heavy crawl, synth math) at 7–13 epochs; the OLMix mixture has 10 effective
   buckets and synthetic QA at 4 epochs. The nearest panel row to any of them is TV 0.28 away (median 0.34).
2. **Epochs and shares are confounded in every row.** At a fixed token budget a bucket's epochs are its share times
   budget over unique tokens, so no row in the panel (or in any of the 408 registry coordinates) can attribute an
   effect to repetition rather than to allocation. The fitted harm curves are therefore share effects in disguise:
   the successor's per-bucket optimum epochs correlate −0.64 with log inventory, the measured best mixtures +0.58,
   the conditional dose ladders +0.52, and the successor against the ladders −0.55 (round-5 follow-up, this
   session). The heads want 17–20 epochs of synthetic QA, stack and OLMOCR and 3–5 of wikipedia and stem-heavy
   crawl; the measured frontier does the opposite.
3. **Spread is credited by construction.** An additive concave benefit gives every bucket its own first-epoch gain,
   so eighteen 2% CC buckets outscore one 36% bucket. The bank contradicts it: the residual of the 15 regressing
   Table-9 components correlates +0.50 with the CC-low share, and 55% of the predicted cap-7 gain came from buckets
   under 4% share or beyond the panel's exposure range. A spread-only panel cannot expose this, because it has no
   concentrated rows to be wrong on.
4. **Reselecting rows from the existing a priori pool does not help.** Round 6
   (`single_phase_observatory_round6_training_sets_report_20260904.md`) rebuilt 280-row training sets from the
   panel plus the 237 dose ladders and the baseline row under the same rule (no model-proposed coordinate). No
   design improved any of four models' picks on either target (Table-9 regret 0.0157 → 0.0143–0.0151, interval
   covering zero; frontier rank 10th → 13th–23rd), and the over-budget fit with every eligible row picked the same
   coordinates. The dose ladders sit around the proportional anchor (Table-9 macro 1.19–1.38, TV 0.44 from the
   frontier); they fix calibration there and nothing else. The budget is not the constraint; the placement is.

Six rounds of model work on this panel (registry of 135 entries, calibration rules, reliability weights, family
objectives, exposure clamps, share floors, kernel corrections, pooled laws, hub interactions) moved calibration and
never selection. That is the case for changing the data.

## 1b. The six modeling rounds behind section 1

All under `.agents/handoffs/`, each with its Codex and DeepSeek dispositions; commits on
`calvin/swarm-olmo3-regmix-test`. Read these before judging whether the data, and not the model class, is the
constraint.

| round | report | what it tested on the current panel | outcome |
|---|---|---|---|
| 1 | `single_phase_observatory_benchmark_cc_report_20260902.md`, `single_phase_observatory_ablation_and_modeling_cc_handoff_20260902.md` | the Observatory benchmark: Screen / Certify tiers, 135 registry entries, matched ablations for every term; the successor `weibull_softplus_unscaled` | successor selected; its predicted Table-9 gain over the frontier did not validate (sweep report in `reference_outputs/delphi_one_phase_weibull_softplus_epoch_cap_sweep_20260902/`) |
| 2 | `single_phase_observatory_round2_cc_report_20260902.md` | link functions (bounded log-deficit, link by CV), refinement, shape sharing, priors; prospective test on 12 proposed coordinates; StarCoder gate | no mechanism beats the successor at Screen; bounded link is the best coverage user; StarCoder gate passes only as a decision rule |
| 3 | `single_phase_observatory_round3_cc_report_20260903.md` (commits 8240201184, 0decb97c67) | the 247/408-coordinate development bank: frozen-model selection scoring, dose anatomy, shape scans, coarse rules, ensembles, proposals, leave-one-source-out and dose regimes; registry defects found and repaired | Uncheatable is coverage-limited, Table 9 is not for additive models; bank-selected shapes fail split-half; with the frontier's neighbours in training every model degrades on Table 9 |
| 4 | `single_phase_observatory_round4_cc_report_20260903.md`, `single_phase_related_work_review_20260903.md` (commit 94a6d148fa) | seven literature mechanisms nested on the successor (share penalty, onset covariates, hierarchical harm, hub interactions, unique-token benefit) and a pooled effective-data law; typed cap policies; dose knees | none selects better; calibration moves, selection does not (fourth time); knees follow inventory, the successor's harm term does not |
| 5 | `single_phase_observatory_round5_olmix_gap_report_20260904.md` (commits d216cc944d, 2a9f3cf3dd) | why WSPU barely beat matched-seed OLMix: 51-row predicted-vs-observed tables per cap, exact per-bucket decomposition, dose curves, bank residual audit, offline remedies, candidate set | the heads never predicted the 15 regressions; spread credit and an over-steep stack curve explain 89% of the predicted gain; no remedy improves selection |
| 6 | `single_phase_observatory_round6_training_sets_report_20260904.md` (commits 174921a8c6, e6e95cc359) | alternative 280-row training sets from sampled and intervention runs only, four models, both targets, over-budget reference | no design improves any pick; the budget is not what binds, the placement is |

The deck `.agents/handoffs/slides/single_phase_observatory_benchmark_20260902/slides.md` (PDF alongside) carries
per-round timelines and TLDRs; Fieldbook experiment `exp_01m1ge7ye6hz2epd0mjkbkrvt8` holds the round notes.

## 2. The design and what each block is for

All 280 rows are fixed by the script from bucket metadata (unique-token counts, bucket type read from the name) and
the literature's repetition priors. No row comes from a fitted model or a measured optimum. OLMix is refitted on the
identical 280 rows.

| block | rows | new runs | placement rule | problem it addresses |
|---|---|---|---|---|
| baselines: proportional, UniMax, uniform | 3 | 0 | reused | comparability with every earlier swarm |
| proportional repeats | 3 | 3 | reused mixture, fresh seeds | seed noise floor (Table-9 macro SD 0.0031 today) |
| leave-one-bucket-out from proportional | 39 | 0 | reused | per-bucket deletion effects, as before |
| type-level anchors B (small pools forward), C (code and math forward), D (CC halved, low bin dropped) | 3 | 3 | relative epoch levels per type: B = CC-high 1, CC-low 0.5, code 2, curated 4, math 4, synthetic 4; C = CC 0.5, code 6, math 6, curated 2, synthetic 2; D = CC-high 0.5, CC-low 0, others 4 | bases for the ladders and subsampled rows in concentrated regions |
| **fixed-share subsampled pools** | 40 | 40 | anchors A (proportional), B, D; targets synthetic QA, OLMOCR, both stack buckets, wikipedia, stem-heavy crawl, synth math, the CC-high group; pool fractions 1/2 and 1/4 | problem 2: the only rows in which epochs move and shares do not |
| epoch-ratio Latin hypercube | 77 | 77 | six type levels drawn from a log box (CC 1/4–2, others 1/4–16), per-bucket jitter σ 0.35, shares = level × unique tokens normalized | problem 1: joint elevation of clean pools with reduced CC; problem 3: concentrated rows on which spread credit is testable |
| single-bucket share ladders at B and C | 48 | 48 | 8 buckets × multipliers 0.25, 3, 8 | per-bucket knees in concentrated regions rather than around the proportional row |
| CC-removal corners | 9 | 9 | CC-high in {1, 0.5, 0.25}, CC-low in {0, 0.25}, other types at 2/4/8 | how much web can go |
| random panel rows, farthest-point subset | 18 | 0 | reused | keeps the spread region covered |
| highest-epoch conditional dose ladders | 40 | 0 | reused | CC over-exposure and proportional-anchor knees already paid for |

New runs: 180 of 280. The reserved 45 rows are unchanged from the current panel, so every baseline comparison that
exists today remains possible.

## 3. Evidence that the design addresses the problems

All numbers are from `manifest.json` and `design_rows.csv`; the coverage comparison uses measured coordinates only
as a diagnostic after placement.

- **Support (problem 1).** Median TV from the ten best measured Table-9 mixtures to the nearest design row 0.26
  (min 0.18) against 0.34 (min 0.28) for the current panel; 122 design rows under 35% CC (panel 0), 139 rows below
  15 effective buckets (panel 0), 19 rows with synthetic QA at or above 25% share (panel 0), synthetic QA reaching
  7.2 epochs at full pool (panel 2.3), OLMOCR 14.5 (3.5), stack 21.6 (6.9). A random Dirichlet swarm of 280 rows,
  the RegMix-style alternative, does no better than the panel (median 0.38, none under 35% CC).
- **Identifiability (problem 2).** In the panel the spread of log epochs given share is exactly zero for all 39
  buckets. In the design it is 0.16 for synthetic QA, OLMOCR, both stack buckets and the CC-high group (six rows each
  with a reduced pool, at three anchors) and 0.13 for wikipedia, stem-heavy crawl and synth math (four rows each).
  Those rows are usable by an epoch-aware model and invisible to OLMix, whose surrogate sees identical shares.
  Half- and quarter-pool rows are nested subsets of the full pool under the same seed (section 5), so the contrast is
  clean.
- **Spread credit (problem 3).** The lattice and corners give 139 concentrated rows. If the true response does not
  reward spreading, a per-bucket concave benefit fitted on them will misfit those rows and lose the inner
  cross-validation to a pooled form; on the current panel that contest cannot happen because every row is spread.
- **Reuse (problem 4).** The reserved and reused rows are exactly the a priori rows round 6 tested; the 180 new rows
  are the ones round 6 showed to be missing (nothing near the frontier region, nothing separating epochs from
  shares).

What this evidence does not show: that a surrogate fitted on the new swarm will pick a better mixture. No run in
the new blocks exists, so that cannot be measured offline; section 4 states the test that would measure it.

## 4. Why this can reach the goal, and the pre-registered test

Goal: a surrogate optimum from the new swarm that beats OLMix's optimum from the same swarm. The mechanism is not
a better model; it is information OLMix cannot consume. OLMix's surrogate is a positive log-linear law in shares
with a KL pull toward the natural distribution and a uniform 4-epoch cap; its optimum is a cap-saturated corner
(nine buckets at exactly 4 epochs carry 78% of the KL=0.005 mixture). Its cap is where the epoch information enters,
and on Table 9 that uniform cap excludes every one of the ten best measured mixtures (best feasible under cap 4 is
OLMix's own 1.0769; under cap 8, 1.0664; under cap 16, 1.0579). An epoch-aware surrogate fitted on rows where
epochs and shares are separated can derive per-bucket caps from the subsampled rows instead of pricing them from a
harm curve, and search inside the resulting box with a zero-or-≥2% share rule, so the two failure modes of round 5
are removed at the source rather than corrected afterwards.

Pre-registered comparison, to be fixed before any fit:

1. Fit OLMix (delta 0.01, KL 0.005, uniform cap 4, its published fitter) and our surrogate on the same 280 rows.
2. Our optimizer: per-bucket caps read from the subsampled rows (knee = first pool fraction whose target loss is
   worse than the full pool by more than the repeat SD; support cap otherwise), shares zero or ≥ 2%, min-plus search.
3. Materialize both optima at two data seeds each (seed SD 0.0031 on the Table-9 macro; a 0.01 difference is
   detected at two seeds per arm). Report both against the replicated centre (1.0639, 26 seeds) as the fixed
   reference and against each other.
4. Success: our optimum beats OLMix's at both seeds by more than the paired seed SD. Failure modes are informative:
   if the subsampled rows show no repetition harm at 3e18 for the targeted buckets, the epoch information is not
   worth having at this scale and share-space models are the right tool; if our optimum lands in the lattice's
   corners far from any measured point, the extrapolation gate (predicted gain mostly from beyond-support buckets)
   rejects it before a run.

Risks: the lattice box and the three anchors were chosen after seeing this project's results; they are symmetric
across the non-CC types and use no fitted value, but a sceptic should set the box before looking. Eight targeted
buckets and 40 subsampled rows estimate eight knees; buckets outside the targets keep support caps only. The
design counts rows, not compute; every row is a full 3e18 run.

## 5. How the subsampled-pool rows run

Levanter's mixture config already supports this without code changes. `max_train_batches` (per dataset) with
`max_train_batches_subset_seed` takes the first N sequences of a seeded stable permutation of the dataset
(`_stable_simulated_epoch_subset_key(name, "train", seed)`, the same key the simulated-epoch path uses), and the
mixture restarts an exhausted dataset (`StopStrategy.RESTART_STRATEGY`), which is repetition. Constraint: it cannot
be combined with `experiment_budget`/`target_budget`, so for a subsampled row the launcher sets, for every dataset,

    max_train_batches[d] = floor(pool_fraction[d] × (experiment_budget / target_budget) × sequences[d] / batch_size)

with `pool_fraction` from the `pool_fraction_<domain>` columns (1.0 for every other dataset) and
`max_train_batches_subset_seed` equal to the swarm's simulated-epoch subset seed. Rows with all fractions at 1.0 run
through the normal simulated-epoch path and select the same sequences, so the half- and quarter-pool rows are nested
subsets of the full-pool row. `swarm_mixtures.csv` follows the augmented-swarm launcher's row convention
(`phase_0_<domain>` and `phase_1_<domain>` columns, equal in every row, single phase) with the pool-fraction columns
appended; the launcher needs the pool-fraction branch added and nothing else.

## 6. Files

- `experiments/domain_phase_mix/exploratory/two_phase_many/design_delphi_apriori_swarm_280_20260904.py`: the
  design (seeded, deterministic), the launcher table, the row summary, the identifiability and coverage diagnostics.
- `tests/test_design_delphi_apriori_swarm_280_20260904.py`: budget, simplex rows, reserved rows, subsample semantics.
- `reference_outputs/delphi_apriori_swarm_280_20260904/swarm_mixtures.csv` (280 rows), `design_rows.csv`,
  `manifest.json` (seed, script and mixture hashes, unique tokens, identifiability, coverage).
- Evidence documents: round 5 and round 6 reports under `.agents/handoffs/`, Fieldbook notes
  `note_01m1qg02egfday31819cqsgpjp` (round 6) and the round-5 note on experiment `exp_01m1msmaf8p4wenrgyfxw2cm54`.

## 7. Review brief (for Codex)

Read section 1b's round reports first; the claim that the data and not the model class is the constraint rests on them. Check hardest: (a) that no row's placement depends on a fitted model or a measured coordinate (the coverage
diagnostic must be the only place the registry's outcomes are read); (b) that the subsampled rows' semantics in
section 5 match `lib/levanter/src/levanter/data/text/datasets.py` (nested subsets, restart on exhaustion, the
incompatibility with simulated budgets) and that the batch-count formula is right; (c) that the evidence in
sections 1 and 3 is stated no more strongly than the artifacts support, in particular that the design's coverage of
the measured best region is a post-hoc diagnostic and that the goal test in section 4 has not been run; (d) whether
the 40 subsampled rows and 8 targets are enough to estimate per-bucket knees at seed SD 0.0031; (e) whether 180 new
runs is justified against a smaller pilot (for example the 40 subsampled rows and the 3 anchors first).
