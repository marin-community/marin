# Handoff: a prospectively frozen 280-row Delphi 3e18 swarm for the 39-bucket mixing problem (2026-09-04)

Purpose: replace the current 280-run fit panel with a swarm, frozen before any model is fitted on it, that gives a
surrogate the evidence it needs to select a better Table-9 / Uncheatable optimum than OLMix fitted on the same
swarm. The rows were placed without any model-proposed or measured optimum, but the lattice box and the anchors were
chosen with this project's earlier results in view: a prospectively frozen adaptive design, not a blind one.
Revision 2 incorporates the Codex review (section 8). This document is written for a reviewer who has not followed the project; every claim points at an artifact.
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
the literature's repetition priors; the box and anchors reflect this project's earlier results (section 4, risks).
No row comes from a fitted model or a measured optimum. OLMix is refitted on the identical 280 rows.

| block | rows | new runs | placement rule | problem it addresses |
|---|---|---|---|---|
| baselines: proportional, UniMax, uniform | 3 | 0 | reused (panel run names recorded) | comparability with every earlier swarm |
| proportional repeats, seed blocks 1–3 | 3 | 3 | reused mixture, fresh paired seeds | seed noise floor and the full-support controls of the pilot |
| leave-one-bucket-out from proportional | 39 | 0 | reused | per-bucket deletion effects, as before |
| type-level anchors B, C, D at full support, plus B repeated in seed block 1 | 4 | 4 | B = CC-high 1, CC-low 0.5, code 2, curated 4, math 4, synthetic 4; C = CC 0.5, code 6, math 6, curated 2, synthetic 2; D = CC-high 0.5, CC-low 0, others 4 | bases for the ladders and subsampled rows; full-support controls per seed block |
| **fixed-share subsampled pools** | 40 | 40 | pilot: anchors A and B × targets synthetic QA, OLMOCR, stack_edu, CC-high group × pool fractions 1/2, 1/4 × seed blocks 0 and 1 (32 rows); plus anchor D × stack_fim, wikipedia, stem-heavy crawl, synth math × 1/2, 1/4 (8 rows) | problem 2: the only rows in which epochs move and shares do not |
| epoch-ratio Latin hypercube | 76 | 76 | six type levels from a log box (CC 1/4–2, others 1/4–16), per-bucket jitter σ 0.35, shares = level × unique tokens normalized | problem 1: joint elevation of clean pools with reduced CC; problem 3: concentrated rows on which spread credit is testable |
| single-bucket share ladders at B and C | 48 | 48 | 8 buckets × multipliers 0.25, 3, 8 | per-bucket knees in concentrated regions rather than around the proportional row |
| CC-removal corners | 9 | 9 | (CC-high, CC-low, others) ratios 2:0:4, 1:0:4, 1/4:0:4, 1/8:0:4, 1:1/4:4, 1/4:1/4:4, 1:1/2:4, 1/2:1/4:4, 1/10:1/10:4, all distinct from each other and from anchor D | how much web can go |
| random panel rows, farthest-point subset | 18 | 0 | reused (panel run names recorded) | keeps the spread region covered |
| highest-epoch conditional dose ladders with both targets measured | 40 | 0 | reused (registry coordinate ids and run names recorded) | CC over-exposure and proportional-anchor knees already paid for |

New runs: 180 of 280, of which 37 form the pilot (section 4). Every run condition (shares, pool fractions, seed
block) is unique; repeated mixtures differ only by seed block. The reserved 45 rows are unchanged from the current
panel, so every baseline comparison that exists today remains possible.

## 3. Evidence that the design addresses the problems

All numbers are from `manifest.json` and `design_rows.csv`; the coverage comparison uses measured coordinates only
as a diagnostic after placement.

- **Support (problem 1).** Median TV from the ten best measured Table-9 mixtures to the nearest design row 0.28
  (min 0.20) against 0.34 (min 0.28) for the current panel (`manifest.json`, `coverage_diagnostic`); design rows
  under 35% CC, below 15 effective buckets and with synthetic QA at or above 25% share number in the hundreds,
  tens and tens respectively where the panel has none (`design_rows.csv`); synthetic QA reaches 7.2 epochs at full
  pool (panel 2.3), OLMOCR 14.5 (3.5), stack 21.6 (6.9). A random Dirichlet swarm of 280 rows,
  the RegMix-style alternative, does no better than the panel (median 0.38, none under 35% CC).
- **Support separation (problem 2).** The surrogate's inputs are shares and exposures. In the current panel the joint
  matrix [shares | exposures] has rank 39: exposures are shares times a constant, so no exposure direction exists
  that shares do not imply. In the design it has rank 47: eight independent exposure directions, one per subsampled
  target group (`manifest.json`, `support_separation`). Per bucket, the centred (share, exposure) pair has a
  singular-value ratio of 0 in the panel and above 0 for the eight targets in the design. Those rows are usable by
  an epoch-aware model and invisible to OLMix, whose surrogate sees identical shares. Half- and quarter-pool rows
  are nested subsets of the full pool under the same seed (section 5). This shows the directions exist; whether
  eight knees are estimable at the seed noise is the pilot's question, not a property of the matrix.
- **Spread credit (problem 3).** The lattice and corners give 139 concentrated rows. If the true response does not
  reward spreading, a per-bucket concave benefit fitted on them will misfit those rows and lose the inner
  cross-validation to a pooled form; on the current panel that contest cannot happen because every row is spread.
- **Reuse (problem 4).** The reserved and reused rows are exactly the sampled and intervention rows round 6 tested; the 180 new rows
  are the ones round 6 showed to be missing (nothing near the frontier region, nothing separating epochs from
  shares).

What this evidence does not show: that a surrogate fitted on the new swarm will pick a better mixture. No run in
the new blocks exists, so that cannot be measured offline; section 4 states the test that would measure it.

## 4. Why this can reach the goal, the pilot, and the pre-registered test

Goal: a surrogate optimum from the new swarm that beats OLMix's optimum from the same swarm. The mechanism is not
a better model; it is information OLMix cannot consume. OLMix's surrogate is a positive log-linear law in shares
with a KL pull toward the natural distribution and a uniform 4-epoch cap; its optimum is a cap-saturated corner
(nine buckets at exactly 4 epochs carry 78% of the KL=0.005 mixture). Its cap is where the epoch information enters,
and on Table 9 that uniform cap excludes every one of the ten best measured mixtures (best feasible under cap 4 is
OLMix's own 1.0769; under cap 8, 1.0664; under cap 16, 1.0579). An epoch-aware surrogate fitted on rows where
epochs and shares are separated can derive per-bucket caps from the subsampled rows instead of pricing them from a
harm curve, and search inside the resulting box with a zero-or-≥2% share rule, so the two failure modes of round 5
are removed at the source rather than corrected afterwards.

**Pilot first (37 new runs plus the existing proportional baseline).** Two seed blocks; within a block the data seed
and the pool-subset seed are the same number (`data_seed` = `subset_seed` = 662009 + block), so a block's half- and
quarter-pool rows are nested in its own full-support run and the block's training noise is shared. Block 0: the
proportional baseline (existing) and anchor B at full support, then anchors A and B × synthetic QA, OLMOCR,
stack_edu, CC-high group × pool fractions 1/2 and 1/4 (16 rows). Block 1: the same 16 rows plus fresh full-support
runs of A (proportional repeat 1) and B (anchor B repeat). Proportional repeats 2 and 3 (blocks 2 and 3) are the
noise controls. The pilot's question is whether the per-bucket repetition effect at 3e18 is larger than the seed
noise for these four targets; only if it is does the epoch information pay for the remaining 143 runs. The 8
anchor-D subsampled rows, the lattice, the ladders and the corners are the second wave.

Pre-registered comparison, to be fixed before any fit:

1. Fit OLMix (delta 0.01, KL 0.005, uniform cap 4, its published fitter) twice on the new swarm: on the full-support
   rows only (240) and on all 280 rows. Its optimum is the one from whichever fit has the lower held-out Huber loss
   under the swarm's inner folds, decided before either optimum runs. Fit our surrogate on all 280 rows.
2. Our optimizer: per-bucket caps read from the subsampled rows (knee = first pool fraction whose target loss is worse
   than the full-support run in the same seed block by more than the seed SD; support cap otherwise), shares zero or
   ≥ 2%, min-plus search.
3. Materialize both optima at two data seeds each as a screen. The replicated centre's SD is 0.0041 on the Table-9
   macro (26 seeds), so two seeds per arm resolve a 0.01 difference and not a 0.005 one; a screen that passes is
   confirmed with four seeds per arm. Both arms are reported against the replicated centre (1.0639) as the fixed
   reference and against each other.
4. Success: our optimum beats the better OLMix optimum at confirmation. Failure modes are informative: if the pilot
   shows no repetition effect above noise, the epoch information is not worth having at this scale and share-space
   models are the right tool; if our optimum lands in the lattice's corners far from any measured point, the
   extrapolation gate (predicted gain mostly from beyond-support buckets) rejects it before a run.

Risks: the lattice box and the anchors were chosen after seeing this project's results (symmetric across the
non-CC types, no fitted value, but not blind). Eight targeted buckets keep knees; every other bucket keeps a support
cap only. The design counts rows, not compute; every row is a full 3e18 run, about 3e18 FLOPs each, 5.4e20 for the
180 new rows and 1.1e20 for the pilot.

## 5. What has to change end to end for the subsampled rows

The launcher is not the only change; the pool fraction has to reach the model.

1. Launcher (`launch_delphi_augmented_swarm_3e18.py` pattern): for a row with any `pool_fraction_<domain>` below 1,
   set for every dataset `max_train_batches[d] = floor(pool_fraction[d] × (experiment_budget / target_budget) ×
   sequences[d] / batch_size)` and `max_train_batches_subset_seed = subset_seed`, instead of `experiment_budget` /
   `target_budget` (Levanter forbids combining them, `datasets.py` lines 839–842). `max_train_batches` takes the
   first N sequences of the same seeded stable permutation the simulated-epoch path uses
   (`_stable_simulated_epoch_subset_key(name, "train", seed)`, lines 1276–1290), so the half- and quarter-pool
   rows are nested subsets of the full-support run with the same seed; the mixture restarts an exhausted dataset
   (`StopStrategy.RESTART_STRATEGY`), which is repetition. Log `pool_fractions`, `seed_block`, `data_seed` and
   `subset_seed` in the run config so the collector can read them.
2. Registry collector (`prepare_single_phase_heldout_benchmark_20260902.py`): read the pool fractions from the run
   config into `pool_fraction::<domain>` columns on `heldout_runs.csv` and `heldout_coordinates.csv`, and include
   them in the coordinate identity. Today a coordinate is its weight vector; without this, the half- and
   quarter-pool rows collapse into the full-support coordinate and their outcomes are averaged away.
3. Panel and bank features (`benchmark_single_phase_observatory_20260902.py`, `heldout_features` and the panel
   loader; `single_phase_observatory_models_20260902.Features`): build `exposures` as `weights × inventory /
   pool_fraction` per row. `exposures` is already a per-row matrix, so the successor needs no other change;
   anything that reads the per-bucket `inventory` constant (onset covariates, the unique-token benefit input) must
   read the per-row effective inventory instead for subsampled rows.
4. OLMix's fitter sees shares only; it needs no change, which is the point.

`swarm_mixtures.csv` follows the augmented-swarm launcher's row convention (`phase_0_<domain>` and
`phase_1_<domain>` columns, equal in every row, single phase) with `pool_fraction_<domain>`,
`materialized_epochs_<domain>`, provenance (`source_coordinate_id`, `source_run_names`, `repeat_of`), `seed_block`,
`data_seed`, `subset_seed` and `wave` columns appended.

## 6. Files

- `experiments/domain_phase_mix/exploratory/two_phase_many/design_delphi_apriori_swarm_280_20260904.py`: the
  design (seeded, deterministic), the launcher table, the row summary, the support-separation and coverage diagnostics.
- `tests/test_design_delphi_apriori_swarm_280_20260904.py`: budget, simplex rows, reserved rows, subsample semantics.
- `reference_outputs/delphi_apriori_swarm_280_20260904/swarm_mixtures.csv` (280 rows), `design_rows.csv`,
  `manifest.json` (seeds, script and mixture hashes, unique tokens, support separation, coverage).
- Evidence documents: round 5 and round 6 reports under `.agents/handoffs/`, Fieldbook notes
  `note_01m1qg02egfday31819cqsgpjp` (round 6) and the round-5 note on experiment `exp_01m1msmaf8p4wenrgyfxw2cm54`.

## 7. Review brief (for Codex)

Read section 1b's round reports first; the claim that the data and not the model class is the constraint rests on
them. Check hardest: (a) that no row's placement depends on a fitted model or a measured coordinate (the coverage
diagnostic must be the only place the registry's outcomes are read); (b) that section 5 matches
`lib/levanter/src/levanter/data/text/datasets.py` and that the batch-count formula is right; (c) that the evidence
in sections 1 and 3 is stated no more strongly than the artifacts support; (d) whether the pilot in section 4
isolates repetition from subset quality and training noise; (e) whether the OLMix protocol in section 4 is fair to
OLMix.

## 8. Codex review of revision 1, and what changed

- P1, pool fractions never reach the model: correct; revision 1 claimed only the launcher changes. Section 5 now
  lists the four touch points, including the coordinate identity in the registry, and the launcher table carries
  `materialized_epochs_<domain>` so the intended exposures are explicit.
- P1, only 14 of 40 reused dose rows had Table 9: correct. The generator now selects from the 89 dose coordinates
  with both targets measured, ranked by epochs, and raises if fewer than 40 exist; the registry refresh (all 237 with
  Table 9) will widen the pool.
- P1, provenance of reused rows lost: correct. Each reused row now records the panel run name or the registry
  coordinate id and the W&B run ids behind it; `repeat_of` marks intentional repeats.
- P1, the 43-run pilot did not isolate repetition: accepted in full. The pilot is now two paired seed blocks, four
  targets, anchors A and B, with a full-support run of each anchor in each block and three proportional controls
  (37 new runs); the extra targets moved to anchor D in the second wave.
- P1, OLMix degraded by duplicate-share rows: accepted. OLMix is fitted on the full-support rows and on all rows,
  and the better fit by a frozen held-out rule supplies its optimum.
- P2, three unintended duplicate conditions: correct (two corner groups collapsed onto anchor D and onto each
  other). Corners are now nine distinct ratios, and the generator rejects any repeated (shares, pool fractions,
  seed block) condition.
- P2, the identifiability diagnostic was tautological: correct. Replaced by the rank of [shares | exposures] (39 in
  the panel, 47 in the design) and per-bucket singular-value ratios; the estimability of knees is left to the pilot.
- P2, wording and power: "a priori" is now "prospectively frozen adaptive design"; the power statement uses the
  centre's SD 0.0041, two seeds are a screen and four seeds the confirmation.
- Recommendation not to launch the 180 runs or the earlier pilot: adopted. The order is pipeline changes (section
  5), then the 37-run pilot, then a decision on the second wave.

