# Single-phase Observatory, round 4: mechanisms from the repetition-aware mixing literature (2026-09-03)

Fieldbook `exp_01m1ge7ye6hz2epd0mjkbkrvt8`; continues rounds 1-3 (commits 99bea291d7, 954a881eea, 8240201184,
0decb97c67). Plan: `.agents/projects/single_phase_round4_plan_20260903.md`. Literature review:
`.agents/handoffs/single_phase_related_work_review_20260903.md`. Local compute only. Status: closed 2026-09-04 05:10.

## 0. Answers

1. **Do the literature mechanisms give a better Delphi 3e18 optimum?** No. Seven mechanisms nested on the
   successor and one new model class, all fitted on the 280-run panel: at Screen three are indistinguishable from
   the successor and four are worse; on the bank none improves the successor's pick on either target (paired
   coordinate and fixed-bank bootstraps: differences zero or positive, the one negative value -0.0003 with an
   interval including zero), and with the frontier's neighbours in training every one behaves like the successor.
2. **What did they change?** Calibration. The pooled effective-data law is a 3x better predictor than the successor
   far from the panel on Uncheatable (RMSE 0.009 against 0.029, bias +0.008 against -0.033 once its solver is
   run to convergence) and the hub interactions remove most of the same optimism; neither selects better, and the
   pooled law selects significantly worse (+0.0055 [+0.0030, +0.0074] regret). With the two links of round 2 that
   is the fourth independent demonstration that calibration far from the panel and ordering among near-frontier
   candidates are different properties on this bank.
3. **What did the dose curves say about tolerance?** Per-bucket conditional knees correlate +0.69 / +0.87 with log
   inventory, which is a share effect (small pools stay small at any multiplier); the literature's unique-token
   tolerance predicts the negative slope in our parameterization, and that is the slope the panel's inner CV
   chooses in 34 of 58 fits. Either way the fitted onset entries do not change selection.
4. **Cap policy.** For Table 9 the best feasible measured value keeps improving with the cap (1.083 at 4, 1.072 at
   6, 1.066 at 8, 1.064 at 16), but the cap never binds the successor: its unconstrained optimum tops out at 7.81
   epochs, so every cap from 8 up materializes the same mixture (the reason the sweep stopped at 8), and its pick
   inside any policy stays 0.007-0.010 behind the best feasible. Typed caps that hold Common Crawl to 2-4 epochs
   cost Uncheatable 0.005-0.015. The successor's harm term, not the policy, keeps it out of the region.
5. **Fit cost.** A Screen pass is 395 fits per entry; the slow ones are the Michael panels (236-480 columns), where
   Lawson-Hanson NNLS costs 200-550 s per fit. A warm-started projected-gradient solver on the per-design Gram
   matrix with a coarse-to-fine grid is the fix for a future round.
6. **No successor is named.** The next lever for Table 9 remains validation runs around the replicated frontier
   centre and subsample-support runs that over-expose Common Crawl at panel weights; for Uncheatable, the DSP
   epoch-cap family at caps 12 and 16.


## 1. Mechanisms and code

Seven entries nested on `weibull_softplus_unscaled` (registry `ROUND4_ENTRIES`), one new model class, and a
cap-policy analysis; see the plan for the table. The design builder gained defaulted options only
(`share_penalty`, `harm_onset_covariate`, `benefit_input`, harm kind `softplus_bucket_hierarchical`, interaction
kinds `total_hub` / `cc_hub`), so every existing configuration's description hash is unchanged; 57 sampled
round-1-to-3 shards reproduce bit-for-bit under the edited builder and the helper pin was refreshed on that
evidence. Tests: `tests/test_single_phase_round4_mechanisms_20260903.py` (8).

## 2. Cap policies on the bank (`single_phase_round4_cap_policies_20260903.py`)

Bucket types: 13 CC-high, 13 CC-low (the two science-math-and-technology buckets are Common Crawl), 5 curated, 5
synthetic, 2 code, 1 math. Feasible bank coordinates, the best measured among them, and the successor's pick
inside the policy (canonical registry):

| Policy | U feasible / best / pick | T9 feasible / best / pick |
|---|---|---|
| uniform 4 | 186 / 0.9827 / 0.9841 | 91 / 1.0832 / 1.0840 |
| uniform 6 | 231 / 0.9823 / 0.9834 | 134 / 1.0722 / 1.0722 |
| uniform 8 | 295 / 0.9823 / 0.9834 | 174 / 1.0664 / 1.0736 |
| uniform 16 | 378 / 0.9811 / 0.9834 | 230 / 1.0639 / 1.0736 |
| typed tight (synthetic 8, math 6, code 5, curated 4, CC 3 / 2) | 164 / 0.9957 / 0.9976 | 77 / 1.0762 / 1.0762 |
| typed loose (16 / 12 / 8 / 6 / 4 / 3) | 221 / 0.9827 / 0.9834 | 119 / 1.0711 / 1.0736 |
| typed frontier (16 / 16 / 8 / 16 / 4 / 2) | 259 / 0.9865 / 0.9900 | 144 / 1.0639 / 1.0660 |

For Table 9 the best feasible value keeps improving with the cap (1.064-1.066 need 12-16 epochs on synth-math,
finemath and wikipedia), but the cap never binds the successor: its unconstrained Table-9 optimum has a maximum of
7.81 epochs (Uncheatable 5.92), so caps 8, 10, 12, 16 and none materialize the same mixture (predicted 1.0039),
which is why the sweep stopped at 8. The successor's pick inside any policy is 0.007-0.010 behind the best
feasible at caps 8-16: its own harm term keeps it out of the region, so ranking, not policy, is the limitation. The HPR-280 control (16.03 epochs on wikipedia) is infeasible even at a
uniform cap of 16. Uncheatable is within 0.0016 of the frontier at every uniform cap of 4 or more; typed caps
that hold Common Crawl to 2-4 epochs cost it 0.005-0.015 because the Uncheatable frontier family keeps
science-math-high at 4 epochs.

## 2a. Repetition knees from the dose curves (`single_phase_round4_dose_knees_20260903.py`)

For each bucket the conditional dose curve (multipliers 0.25 to 32 of its proportional share) is summarized by
the optimum of a quadratic in log multiplier (Scaling Domain Data Repetition's estimator) converted to epochs, and
by the first multiplier whose change from the anchor exceeds three repeat SDs. Development evidence (the dose runs
are in the bank), used only to choose among covariates that the panel-only entries then carry as two global
parameters.

| Covariate | Spearman with log optimum epochs, Uncheatable (n = 36) | Table 9 (n = 14) |
|---|---|---|
| centred log inventory | +0.69 (p < 0.001) | +0.87 (p < 0.001) |
| declared quality rank | -0.28 (p = 0.10) | -0.51 (p = 0.06) |
| deletion value | +0.66 (p < 0.001) | +0.41 (p = 0.14) |

Conditional optima sit at 0.4-1.2 epochs for most Common Crawl buckets (below the proportional 0.9), 1.2-1.7 for
synth-thinking, finemath, synth-math, wikipedia and arxiv, and 2.0-2.7 for stack-edu and synth-code. The
unique-token covariate (log inventory) is the one the literature predicts and the one the curves support; the
quality rank points the other way because high-quality CC buckets are the smaller ones. Inventory here is epochs
at full share, budget divided by unique tokens, so a large inventory is a small unique pool. The dose-curve
correlation therefore says that small buckets show later conditional knees in epochs, a share effect (their weight
stays tiny at any multiplier, so the damage of concentration is small), not the literature's unique-token
tolerance, which predicts the opposite sign in this parameterization: a negative slope, earlier onset for small
pools. The panel's inner CV chooses that negative slope: on the 58 Delphi heldout fits -0.75 in 34, 0 in 6, positive
in 18 (`@onset_quality`: -1.0 in 31 of 58). The grid contains both signs, so the entry tests the mechanism either way.

## 3. Screen (38 family-macro units, paired against `weibull_softplus_unscaled`; `screen/pooled_screen_contrasts.csv`)

| Entry | RMSE units better / worse | sign p (Holm) | mean RMSE contrast | Spearman units better / worse |
|---|---|---|---|---|
| @share_penalty | 19 / 19 | 1.0 | +0.0002 | 22 / 16 |
| @onset_inventory | 16 / 21 | 0.51 | +0.0008 | 20 / 16 |
| @onset_quality | 17 / 16 | 1.0 | +0.0003 | 19 / 14 |
| @harm_hierarchical | 13 / 25 | 0.073 | +0.0066 | 26 / 12 (p = 0.034) |
| @interaction_total_hub | 13 / 25 | 0.073 | +0.0070 | 31 / 7 (p = 0.0001) |
| @interaction_cc_hub | 11 / 27 | 0.014 | +0.0069 | 32 / 6 (p < 0.0001) |
| @unique_benefit | 0 / 38 | < 0.0001 | +0.0247 | 35 / 3 (p < 0.0001) |

No Nadeau-Bengio interval excludes zero on any metric, so nothing meets the frozen promotion rule. Three
mechanisms are indistinguishable from the successor in in-panel accuracy (share penalty, both onset covariates:
mean contrasts +0.0002 to +0.0008, sign p >= 0.5), the hierarchical harm and
both hub interactions are worse on most units (extra columns the panel cannot pin down), and the unique-token
benefit is worse on every unit: within the panel the epoch coordinate carries the benefit signal, as round 1's
weight-coordinate ablation already showed. All seven still go to the bank, which scores selection rather than
in-panel accuracy. The promotion table (harness now enumerates `ROUND4_ENTRIES`) promotes none by the frozen rule;
the post-hoc two-sided sign rule flags the CC-hub interaction (11/27) and the unique-token benefit (0/38), both in
the worse direction, which is what that rule is for (a significant difference either way is documented at Certify).


## 4. Bank selection (canonical registry, panel-only fits; `heldout_round4/`)

Archive stratum (Uncheatable 171, Table 9 158 coordinates); regret of the predicted argmin, best-of-5, rank the
measured frontier receives, bias at L1 >= 0.75 from the panel:

| Entry | U regret / best-of-5 / frontier rank / far bias | T9 regret / best-of-5 / frontier rank / far bias |
|---|---|---|
| weibull_softplus_unscaled (reference) | 0.0023 / 0.0012 / 6 / -0.033 | 0.0157 / 0.0132 / 10 / -0.030 |
| @share_penalty | 0.0023 / 0.0012 / 6 / -0.032 | 0.0157 / 0.0132 / 12 / -0.030 |
| @onset_inventory | 0.0023 / 0.0023 / 10 / -0.030 | 0.0157 / 0.0132 / 11 / -0.032 |
| @onset_quality | 0.0030 / 0.0012 / 7 / -0.029 | 0.0157 / 0.0132 / 10 / -0.030 |
| @harm_hierarchical | 0.0023 / 0.0012 / 7 / -0.034 | 0.0157 / 0.0132 / 12 / -0.040 |
| @interaction_total_hub | 0.0023 / 0.0023 / 6 / -0.013 | 0.0157 / 0.0132 / **6** / -0.032 |
| @interaction_cc_hub | 0.0030 / 0.0023 / 18 / -0.005 | 0.0151 / 0.0143 / 25 / +0.005 |
| @unique_benefit | 0.0927 / 0.0103 / 155 / -0.001 | 0.2530 / 0.0157 / 20 / +0.027 |
| @log_deficit_bounded_link (round 2) | 0.0030 / 0.0016 / 8 / -0.005 | 0.0143 / 0.0143 / 35 / +0.010 |

- The share penalty, the onset-by-inventory entry, the hierarchical harm and the total-hub interaction make the
  successor's pick on both targets (0.9834, the cap-6 sweep run; 1.0736, the cap-8 run). The onset-by-quality
  entry moves to the cap-4 run on Uncheatable (0.9841, regret 0.0030) and keeps 1.0736; the CC-hub interaction
  moves on both (0.9841; 1.0730, regret 0.0151 against 0.0157). Paired bootstrap over coordinates and the
  fixed-bank measurement bootstrap give differences of zero or positive against the successor on Uncheatable; on
  Table 9 the CC-hub interaction is -0.0003 [-0.0014, +0.0035] and nothing else moves.
- The hub interactions remove most of the far-panel optimism on Uncheatable (bias -0.013 and -0.005 against
  -0.033) the way the bounded link does, and the total-hub entry ranks the Table-9 frontier 6th (successor 10th,
  top-quartile Spearman 0.60 against 0.56) without changing the pick; the CC-hub entry changes it by one
  neighbour on each target with no interval excluding zero. The share penalty, the onset covariates
  and the hierarchical harm leave regret@1 within 0.0007 of the successor (best-of-5 moves by up to 0.0012 for the
  inventory onset): on this panel the extra columns are fitted to zero or to values that do not change the
  ordering far away.
- The unique-token benefit collapses (regret 0.09 / 0.25): within the panel the epoch coordinate carries the
  benefit signal, and a benefit that saturates at the inventory has nothing to say about the frontier region.


## 4a. Leave-one-source-out on Table 9 (`union_round4_*`, archive stratum, 158 coordinates, 15 sources)

Panel plus dose rows plus every archive source but the held-out one in training, as in round 3:

| Entry | pooled frontier rank, panel-only -> loso | far bias, panel-only -> loso | within-source regret, paired difference loso - panel-only |
|---|---|---|---|
| @interaction_total_hub | 6 -> 53 | -0.027 -> +0.004 | +0.0019 [-0.0026, +0.0068] |
| @interaction_cc_hub | 25 -> 46 | +0.002 -> +0.008 | -0.0002 [-0.0039, +0.0039] |
| @share_penalty | 12 -> 60 | -0.029 -> +0.003 | +0.0034 [-0.0009, +0.0083] |
| @onset_inventory | 11 -> 54 | -0.030 -> +0.003 | +0.0025 [-0.0014, +0.0064] |
| pooled_effective_data | 33 -> 50 | -0.013 -> +0.007 | +0.0006 [-0.0039, +0.0053] |
| successor (round 3) | 10 -> 45 | -0.029 -> +0.003 | +0.0020 [-0.0020, +0.0055] |

Every round-4 mechanism behaves like the successor once the frontier's neighbours enter training: the far bias
disappears, the pooled ranking of the frontier region gets worse (46th-60th), and no within-source paired interval
excludes zero. None of them uses coverage better than the successor; the Table-9 limit found in round 3 stands.

## 5. Pooled effective-data law (`single_phase_round4_pooled_law_20260903.py`, registry `pooled_effective_data`)

y = c + (sum_b tau_b U_b (1 + rho(E_b)))^(-alpha) + sum_b gamma_b w_b, U_b the unique share of the budget, rho a
saturating repetition credit with scale r1, tau and gamma nonnegative by bounded least squares with an analytic
Jacobian, alpha in {0.1, 0.3, 0.6, 1.0}, r1 in {2, 6, 20} and a ridge by inner CV (a joint prior: tau toward 1,
gamma toward 0). The registry passes the revision explicitly so it enters the cache key.

Three revisions were fitted, and the bank numbers moved with the solver, which is itself a finding:

| Revision | solver | U archive RMSE / regret / frontier rank | T9 RMSE / regret / frontier rank |
|---|---|---|---|
| 2 | finite-difference Jacobian, 400 evaluations, first finite iterate accepted (scipy `success` mostly false) | 0.0071 / 0.0030 / 11 | 0.0215 / 0.0157 / 33 |
| 3 | same, only `success` iterates accepted, fallbacks by scaling and method | 0.0104 / 0.0068 / 23 | 0.0233 / 0.0157 / 17 |
| 4 (final) | analytic Jacobian, 2000 evaluations, converged on every unit | 0.0088 / 0.0086 / 28 | 0.0238 / 0.0157 / 26 |

Final (revision 4):

- Screen (38 units, none failed; Michael-panel fits 600-720 s each): RMSE better than canonical DSP on 32 / 6 units
  and than OLMix on 36 / 2 (sign p < 0.0001), regret at 1 better than DSP on 27 / 10 (p 0.008); Spearman worse on
  34 / 4 and 36 / 2. No parent contrast exists for a new class; on the Delphi Screen units its inner-CV RMSE is at
  the successor's level.
- Bank, archive stratum: Uncheatable RMSE 0.0088 against 0.029 for the successor and 0.015 for the bounded link,
  bias +0.008 at L1 >= 0.75 against -0.033 (the only model that is not optimistic there); but regret 0.0086 (rank
  21), frontier 28th, paired difference against the successor +0.0055 [+0.0030, +0.0074], worse in every draw. Table
  9: RMSE 0.0238, regret 0.0157 (the successor's pick), frontier 26th, +0.0005 [0.0000, +0.0080].
- Reading: concave pooling of the effective-data total is the best-calibrated single-phase predictor far from the
  panel that this benchmark has seen on Uncheatable, and the worst selector among the calibrated models; the
  revision-2 numbers that looked like both came from iterates scipy had not converged. Together with the two links
  and the hub interactions this is the fourth mechanism on which calibration far from the panel and ordering among
  near-frontier candidates come apart, and the first where better calibration coincides with clearly worse
  selection.

## 6. Fit cost (asked during the Screen run)

A Screen pass is 79 units x 5 folds = 395 fits per entry: 18 anchor components on the three 39-bucket panels,
16 Michael tasks, 45 StarCoder curves. Stored per-fit times: share penalty 43 s, onset entries 61 s,
hierarchical harm 120 s, hub interactions 226 s (max 550 s) on the tabular units; StarCoder fits are free.
Profile of one Delphi component fit on one core: successor 2.2 s = 2520 NNLS solves at 0.87 ms (scipy
Lawson-Hanson 0.83 s, QR reduction 1.0 s); hub entry 21 s = 5040 solves at 4.1 ms (NNLS 15 s). The slow Screen
fits are the Michael panels (236-240 successor columns, 480 hub columns), where Lawson-Hanson scales badly and the
QR reduction is skipped whenever rows < 2 x columns. Remedies for a future round, in order of expected gain:
warm-started projected-gradient NNLS on the per-design Gram matrix shared across the ridge grid and neighbouring
shapes; a coarse-to-fine shape grid with the round-2 bounded refinement; forcing the reduced solve whenever it is
cheaper. A solver change alters results only at tolerance and needs the same equivalence check the QR path had.

## 7. Reviews

DeepSeek (deepseek-v4-pro, max reasoning; 71 messages, read-only) recomputed all 6,350 description hashes against
the gen9 snapshot (0 mismatches), re-ran the round-4 tests in process, compared the old and new design builder on
62 sampled configurations (0 mismatches), and recomputed every number in sections 2-6 from the tables. Codex
(gpt-5.6-sol, max reasoning) found three result-affecting code defects and six text or safeguard items.

| Finding | Verified | Disposition |
|---|---|---|
| DeepSeek B1/B2: section 4 misclassified which entries keep the successor's pick (onset-by-quality and CC-hub move it; total-hub keeps it) | yes | section 4 rewritten from the tables |
| DeepSeek B3: the onset-by-inventory entry does not carry the expected sign; inner CV picks slope -0.75 in 34 of 58 Delphi fits | yes | section 2a rewritten; this is itself a finding |
| DeepSeek N4: the pooled law reported `converged: True` unconditionally | yes | the solver's success flag and a finite cost are now recorded |
| DeepSeek N5: the pooled module is not in the harness source hashes | yes | documented on the `revision` field; adding the file to `source_hashes()` would change every protocol hash, deferred |
| DeepSeek N6: share-penalty columns are collinear with the intercept at ridge 0 | yes | harmless for prediction; noted in section 4 |
| DeepSeek N7: round-4 entries use the unscaled head, as every successor ablation since round 2 | yes | inherited convention, stated |
| DeepSeek N8: CC-hub silently fell back to the total hub without bucket names | yes | now raises |
| DeepSeek N10: `bucket_type` classified the two CC science-math buckets as math | yes | fixed; cap-policy table regenerated (13 / 13 / 5 / 5 / 2 / 1) |
| DeepSeek N11-N14: Spearman header order, "exact ties", "241 parameters", unverifiable inner-CV numbers | yes | corrected (473-481 parameters; Delphi Screen inner-CV 0.0186 / 0.0245 against 0.0199 / 0.0236) |
| DeepSeek N15: the one-line harness change is in a file another session also edits | yes | committed with this round |
| Codex P1: typed caps misclassified the two CC science-math buckets | yes (same as DeepSeek N10) | fixed, table regenerated |
| Codex P1: the pooled `revision` is a defaulted field, so bumping it left the description hash unchanged and the failed Michael shards counted as cached | yes | the registry now passes `POOLED_REVISION` explicitly (default 0), which enters the description; all pooled shards refit |
| Codex P1: non-converged least-squares iterates accepted and reported as converged | yes | only `success` results accepted; otherwise the lowest-cost finite iterate is kept and `converged` is recorded false |
| Codex P2: the inventory covariate's sign was inverted in the text (inventory = budget / unique tokens) | yes | section 2a rewritten; the literature's expectation is the negative slope inner CV chose |
| Codex P2: the pooled ridge is a joint (tau = 1, gamma = 0) prior, not a tau-only prior | yes | described as such |
| Codex P2: `onset_covariate` and `unique_token_input` were not pinned | yes | pinned |
| Codex P3: bank-pick summary, "exact ties", 0.0007 bound | yes | corrected (best-of-5 moves up to 0.0012) |
| Codex P3: the pooled-law test generated its truth with the production helper | yes | independent oracle written out in the test |

