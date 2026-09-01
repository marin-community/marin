# Adaptive sampling for RL curricula under a fixed token budget

TL;DR: We compared six environment-sampling policies for GRPO post-training on a difficulty-graded math ladder (grades 0–13) at two scales, under fixed step budgets with measured generated-token spend, scoring runs with a grade-weighted validation pass@1 that the samplers never observe. On Qwen3-0.6B, adaptive bin sampling composed with DAPO group filtering scored 0.312 grade-weighted against 0.267 for uniform sampling (+17% relative); DAPO alone contributed +0.027 and adaptive sampling alone +0.016. On Snowball 67B-A2B (our in-house 67B-total, 2B-active MoE, after reasoning SFT), the grade-prior arm finished first at 0.310 and was the only arm still holding the hardest grade at the end of training; its final margin over uniform sampling (+0.021) is within single-seed noise, and the decisive evidence at this scale is robustness. Two failure modes broke the pure feedback-driven sampler: a grader that rejects the model's answer format is indistinguishable from an unlearnable bin and gets starved (Snowball scored 0.000 on GSM8K for an entire run because its `\boxed{}` answers fail the `#### N` grader), and the arm that concentrates sampling hardest was the one destabilized by a late high-gradient update. Grade-derived priors prevented the starvation at small measured cost (−0.011 grade-weighted on Qwen). We recommend grade priors + group-informative weighting + DAPO filtering as the default configuration. A fourth round on Snowball (MuonH at the model's own recipe lr, a grader-contract system prompt, and belief reversion) confirmed the frontier result under a healthier regime: prompting alone removed the format handicap for every arm (val-gsm8k ≥ 0.89 by the first eval), the composite naive-vs-curriculum gap shrank to noise, but the curriculum arm was again the only one to sustain the hardest grade — and with format solved, the binding constraint became the 2048-token generation cap, which 40–48% of rollouts still hit at end of training.

## 1. Question

RL post-training pools tasks whose difficulty for the current policy spans everything from always-solved to never-solved. Rollouts at both extremes produce GRPO groups with identical rewards, zero advantage, and no gradient, so the sampling distribution over tasks controls how much of a fixed generation budget becomes learning signal. We measured how much the choice of sampling policy is worth on a realistic difficulty ladder, holding the model, pool, and trainer fixed.

The candidate policies come from the automatic-curriculum literature: sampling by learning progress (Graves et al., 2017, arXiv:1704.03003), by value-loss priority (Prioritized Level Replay; Jiang et al., 2021, arXiv:2010.03934), and by success-rate learnability p(1−p) (Rutherford et al., 2024, arXiv:2408.15099). DAPO (Yu et al., 2025, arXiv:2503.14476) attacks the same waste within a batch: drop zero-advantage groups and draw more until the batch is full. We evaluate bin-level sampling, DAPO filtering, and their composition, plus a weighting derived from the filter itself: sample bins by the probability 1 − pⁿ − (1−p)ⁿ that a size-n GRPO group at pass rate p survives the filter (we call this group-informative weighting; n = 8 here). Unlike p(1−p), this curve is near-flat across mid difficulties and vanishes only at the extremes.

Two earlier rounds on Qwen3-0.6B shaped the design; their full results are in the [tracking issue](https://github.com/marin-community/marin/issues/8765). Round 1 found that adaptive samplers bought token efficiency rather than a better endpoint. Round 2 found DAPO filtering the largest single fixed-budget win, and found samplers driven by pass rate outperform samplers driven by the fraction of mixed-reward groups. Round 3, reported here, replaced the artificial pools with a graded ladder, added the second model scale, and added the group-informative weighting.

## 2. Setup

Pool. One shared partition of 19 train bins over ~23,400 problems, graded 0–13 from elementary arithmetic to graduate-level mathematics: seeded reasoning-gym chain sums (grades 0–2), ASDiv (1–2), SVAMP (2), GSM8K (3), NuminaMath cn_k12 (4), MATH levels 1–5 (grades 5–8), NuminaMath AMC (9), AIME 1983–2024 (10), NuminaMath olympiad and Omni-MATH (11–12), TheoremQA and HARDMath (13). Grades come from the strongest per-source signal: explicit school grades for ASDiv, MATH levels, contest tier otherwise. Validation is six held-out bins, disjoint from training rows, at grades 1 (reasoning-gym sums), 3 (GSM8K), 5 (MATH-500), 9 (AMC), 11 (Omni-MATH), and 13 (TheoremQA), 75–500 rows each.

End metric. Grade-weighted validation pass@1: per-bin pass@1 averaged with weight 1+grade, so grade-13 progress counts 7× grade-1 progress. We also track a frontier grade, the highest grade with pass@1 ≥ 0.25. Both are computed offline from logged evals and are never visible to any sampler; samplers observe only their own rollout rewards.

Models and training. GRPO (Shao et al., 2024, arXiv:2402.03300) on the SkyRL training stack: lr 2e-6, KL loss, FSDP2, evals every 10 steps with each environment's rule-based grader, one sample per prompt. Qwen3-0.6B (non-thinking): 120 steps × 512 prompts × 8 samples, ≤1024 new tokens, on 64 H100s (16 training GPUs, 48 single-GPU vLLM engines). Snowball 67B-A2B SFT: 60 steps × 128 prompts × 8 samples, ≤2048 new tokens, on 64 H100s (32 training GPUs, 4 node-sized expert-parallel vLLM engines). One seed per arm.

Budget accounting. Arms run a fixed step count; token spend is measured, one generation batch per training step (average response length × responses per batch). Measured spend was 229–247M tokens per Qwen arm and 64–75M per Snowball arm, except grade-adaptive: its 120 steps cost only 144M tokens because it dwells on bins with short answers. Extra batches drawn by the DAPO filter are generated but not metered, so DAPO-arm spend is understated by the unmetered redraw fraction; equal-step comparisons are unaffected, and token-axis comparisons of DAPO against non-DAPO arms are biased in DAPO's favor by that fraction.

Arms. All arms share the pool, model, and trainer; only sampling differs. The adaptive samplers keep a per-bin Beta posterior over pass rate from pre-filter rollout counts decayed 0.95 per step, Thompson-sample a rate per bin, map it through the group-informative weight, then normalize with a floor of 0.05/num_bins. The arm name "learnability" is historical (round 2 used p(1−p)); in round 3 both directional arms use the group-informative weight and differ only in their prior.

| arm | bin sampling | DAPO filter |
|---|---|---|
| naive | row-proportional shuffle | no |
| naive-dapo | row-proportional shuffle | yes |
| learnability-dapo | Thompson on pass rate → group-informative weight, flat prior | yes |
| grade-prior | as learnability, prior mean linear in grade from 0.85 (easiest) to 0.05 (hardest) | no |
| grade-prior-dapo | as grade-prior | yes |
| grade-adaptive | 80% of batch from the current grade, 20% row-proportional over the rest; advance when the current grade's informative fraction (share of groups whose 8 rollouts do not all get one reward) stays below 10% for 10 steps | no |

Snowball ran three arms (naive, learnability-dapo, grade-prior-dapo) at otherwise identical configurations.

## 3. Results

### Qwen3-0.6B: sampler and filter compose

| arm | grade-weighted | frontier grade | measured tokens |
|---|---|---|---|
| learnability-dapo | 0.312 | 9 | 247M |
| grade-prior-dapo | 0.301 | 9 | 245M |
| naive-dapo | 0.294 | 5 | 229M |
| grade-prior | 0.283 | 5 | 240M |
| naive | 0.267 | 5 | 231M |
| grade-adaptive | 0.257 | 5 | 144M |

![Qwen grade-weighted pass@1 vs tokens](https://raw.githubusercontent.com/marin-community/marin/assets/curriculum-rl-paper/qwen-grade-weighted-tokens.png)

Decomposing the 0.045 gap between learnability-dapo and naive: DAPO filtering alone is +0.027 (naive-dapo), adaptive bin sampling alone is +0.016 (grade-prior), and the composition adds +0.018 over naive-dapo. The ordering is stable over the last third of training. In all line charts, faint lines are raw evals and bold lines are EMA-smoothed.

The per-grade breakout shows where the gains live. Grades 1–3 saturate for every arm inside 50M tokens (grade 1 near 1.0, grade 3 plateauing near 0.8), grade 5 climbs slowly for all arms, and grades 11–13 never leave the floor at this scale. The separation is at grade 9 (AMC): the two adaptive DAPO arms alone reach 0.25, grade-prior-dapo at 224M tokens and learnability-dapo at its final eval (247M).

![Qwen pass@1 by grade](https://raw.githubusercontent.com/marin-community/marin/assets/curriculum-rl-paper/qwen-grades-tokens.png)

![Qwen tokens to attain each grade](https://raw.githubusercontent.com/marin-community/marin/assets/curriculum-rl-paper/qwen-attainment.png)

grade-adaptive finished 0.010 below naive on 62% of naive's measured tokens (144M vs 231M), repeating the earlier rounds' pattern: strict grade-advancement is token-efficient because it dwells on short-answer bins, and it never reached the contest grades within the run (AMC 0.08 at its final eval, against naive's 0.18).

### Snowball 67B-A2B: the prior decides the outcome

| arm | grade-weighted | without val-gsm8k | frontier grade | measured tokens |
|---|---|---|---|---|
| grade-prior-dapo | 0.310 | 0.256 | 13 | 71M |
| naive | 0.289 | 0.232 | 5 | 64M |
| learnability-dapo | 0.166 | 0.181 | 5 | 75M |

The final ordering overstates the steady-state gap between the top two arms: naive tracked at or slightly above grade-prior-dapo for most of the run, peaked at 0.315 near 46M tokens, and slid to 0.289, while grade-prior-dapo ended at its maximum and was still rising at cutoff. All three arms touched grade 13 during the run (TheoremQA ≥ 0.25); only grade-prior-dapo still held it at the final eval (0.277). With one seed we read the result as "grade-prior-dapo matches naive and degrades less", supported by the failure analysis below rather than by the 0.021 final gap. The without-gsm8k column removes the validation bin whose grader is mis-specified for this model (next section); it widens grade-prior-dapo's margin over naive to 0.024 and shows learnability-dapo's deficit is not explained by that bin alone.

Round-3 Snowball trajectory charts (all three arms, including the learnability collapse analyzed in §4) are on the [assets branch](https://github.com/marin-community/marin/tree/assets/curriculum-rl-paper); the chart treatment of the surviving naive-vs-curriculum comparison is in §7, under the corrected round-4 recipe.

## 4. Analysis

### Failure mode 1: format-contract starvation

The ladder's graders differ per source: the GSM8K environment requires a final `#### N` line, while most other bins accept an answer line or `\boxed{}`. Snowball's SFT distribution answers in `\boxed{}` style, so its observed GSM8K pass rate is 0 even when the mathematics is right; sampled retained rollouts show correct boxed answers rejected by the grader ([details](https://github.com/marin-community/marin/issues/8765#issuecomment-5471696465)). For the group-informative weight, w(0) = 0: the learnability sampler concluded the bin was unlearnable at step 1 and pinned it near the sampling floor (0.9% of the batch, against ~9% for bins it favored) for the whole run, so the model never saw enough GSM8K rollouts to learn the format, and val-gsm8k stayed at 0.000 at every eval.

![Snowball learnability bin weights, log scale](https://raw.githubusercontent.com/marin-community/marin/assets/curriculum-rl-paper/curriculum-snowball-learnability-dapo-weight.png)

The two other arms escaped. naive's row-proportional exposure kept ~54 GSM8K rollouts per step flowing; the rare rollouts that happened to emit `####` earned positive advantage, and RL taught the format inside 10 steps (val-gsm8k 0.891 at step 10, 0.918 final). grade-prior-dapo kept sampling the bin on its grade-3 prior until the same mechanism fired (0.828 at step 10, 0.898 final). A sampler driven purely by observed reward cannot distinguish "too hard for now" from "grader rejects the format"; both look like a zero pass rate. Priors are the cheap insurance; normalizing every bin to one answer contract is the correct fix.

### Failure mode 2: late-run destabilization under concentrated sampling

snowball-learnability-dapo was stable through the step-50 eval; the step-60 eval showed every validation grade damaged at once, and the training log locates a step-60 update with raw gradient norm 19,332 (typical values near 1), max per-token policy-reference KL 9.3, and entropy 0.085. The terminal checkpoint export captured the damaged weights. This is an association from one seed, and snowball-naive also decayed mildly from its mid-run peak, so some late-run regression at this scale is not sampler-specific — but the blowup landed in the arm that concentrates sampling hardest, and no KL or gradient-norm guard was in place. Concentrated adaptive sampling raises the value of such a guard (skip or clip updates past a threshold).

### The weighting objective matches measured learning velocity

Pooling every bin of every arm at both scales, we took consecutive logged transitions of each bin's decayed pass rate and binned Δ(pass rate)/Δ(step) by pass rate. Mean velocity peaks near pass rate 0.45 at ~0.008/step and vanishes at both extremes. On the low side the empirical curve tracks the flat-topped group-informative weighting: bins at 5–25% pass rate still improve at one- to two-thirds of peak velocity, which p(1−p) underweights. On the high side velocity falls off faster than either reference, so mastered bins are even less worth sampling than the weighting assumes. One confound cuts both ways: velocity is per training step, and bins a sampler favors both update their statistics faster and receive more gradient, so the curve mixes intrinsic learnability with allocation. It validates the weighting's shape, not a causal claim.

![Learning velocity vs pass rate](https://raw.githubusercontent.com/marin-community/marin/assets/curriculum-rl-paper/velocity-vs-pass-rate.png)

## 5. Conclusions

1. Under a fixed budget, adaptive bin sampling and DAPO group filtering are complementary, and their composition was the best configuration at both scales (+17% relative grade-weighted pass@1 over uniform sampling on Qwen3-0.6B).
2. The value concentrates at the frontier. Curricula did not move saturated grades and did not lift grades far beyond the model's reach; they bought attainment of the hardest reachable grade (9 for Qwen, 13 for Snowball) within budgets where uniform sampling did not hold it.
3. Pure feedback-driven samplers are unsafe as defaults on a new model, because a mis-specified grader is indistinguishable from an unlearnable bin. Grade priors cost little when wrong (Qwen: grade-prior-dapo finished 0.011 below learnability-dapo, whose flat prior was harmless there) and rescue the run when feedback is corrupted (Snowball: 0.310 vs 0.166).
4. The group-informative weighting 1 − pⁿ − (1−p)ⁿ fits measured learning velocity better than p(1−p) at low pass rates; both overweight high pass rates.
5. Recommended default: grade priors + group-informative weighting + DAPO filtering, one answer contract across all bins, and a KL/gradient-norm guard for long concentrated runs.

## 6. Limitations

One seed per arm; the Snowball top-two final gap (0.021) is within plausible seed noise and the mid-run ordering favored naive. Scores are in-run rule-graded pass@1, one sample per prompt, on 75–500-row validation bins (the grade-9 bin moves in 1/128 quanta). Arms stop on steps, so token spend varies (an 8% spread among Qwen arms, with grade-adaptive at −38%), and DAPO redraw generation is unmetered, which flatters DAPO arms on the token axis by an unmeasured factor. The grade-weighted metric's 1+grade weights are a choice; equal weighting compresses the gaps but does not change the top arm in either family (see the per-grade panels). Snowball ran 60 steps and its best arm was still rising at cutoff. The format-contract failure is specific to models whose SFT style conflicts with a grader, though that condition is common when reusing public graders on in-house models.

## 7. Round 4: removing the format handicap, then re-measuring

Round 3 left three suspicions: the curriculum's Snowball margin might mostly reflect guaranteed exposure to a format-deceptive bin rather than sampling skill; AdamW at 2e-6 might under-drive a model whose entire pretraining and SFT used MuonH; and a starved bin had no path back into the belief distribution. Round 4 changed all three on Snowball and re-ran the two viable arms for 120 steps × 512 rollouts (vs round 3's 60 × 1024; same rollout total, twice-as-fresh policy):

- **Optimizer**: MuonH at lr 1e-5 with weight decay 0 (the model's own pretrain/SFT recipe), gradient clip 1.0, micro batch 8 — halving the FSDP all-gathers that dominate MoE step time (222s/step vs 534s in round 3).
- **Prompt contracts**: a system prompt stating the grader contract plus strengthened per-bin instructions, adopted after a served-model A/B (temp 1.0, rule-graded, 100 problems per variant) lifted gsm8k from 0.00 to 0.55 before any RL; the failures it fixed were correct math ending in `\boxed{18}` instead of `#### 18`.
- **Reversion**: `data.sampling.reversion_mass=2.0` adds per-step pseudo-evidence pulling each bin's belief toward pass 0.5, so a starved bin re-inflates to a probeable weight instead of locking at the floor.

Final evals (round-3 values in parentheses; "metered" excludes DAPO redraw generation, as before):

| arm | grade-weighted | frontier grade | metered tokens |
|---|---|---|---|
| naive | 0.285 (0.289) | 5 (5) | 89M |
| grade-prior-dapo | **0.292** (0.310) | **13** (13) | 81M |

![Snowball round-4 grade-weighted pass@1 vs tokens](https://raw.githubusercontent.com/marin-community/marin/assets/curriculum-rl-paper/snowball-r4-grade-weighted-tokens.png)

1. **Prompting ate most of the curriculum's round-3 margin.** With the contract in the prompt, both arms scored ≥ 0.89 on val-gsm8k at the first eval (step 10) — compliance that round 3's naive arm spent half its run learning and the learnability arm never achieved. The naive-vs-curriculum composite gap shrank from +0.021 to +0.007. The correct reading of round 3 is that most of the curriculum's margin there was guaranteed exposure to a format-deceptive bin, and a two-paragraph prompt buys the same thing for free.
2. **The frontier edge survives.** grade-prior-dapo was again the only arm to sustain the grade-13 bin (val-theoremqa 0.268 final, sustained crossing at ~75M tokens; naive peaked at 0.232 without a sustained crossing) — the same 13-vs-5 frontier split as round 3, reproduced under a different optimizer, batch shape, and prompt. Frontier attainment, not the composite, is the durable value of the curriculum.

   ![Snowball round-4 pass@1 by grade](https://raw.githubusercontent.com/marin-community/marin/assets/curriculum-rl-paper/snowball-r4-grades-tokens.png)

   ![Snowball round-4 tokens to attain each grade](https://raw.githubusercontent.com/marin-community/marin/assets/curriculum-rl-paper/snowball-r4-attainment.png)

3. **MuonH at 5× the round-3 lr was stable end to end.** Raw gradient norms held at 0.17–0.47 under the 1.0 clip with no late-run destabilization in either arm; the dapo arm finished 120 steps in one attempt. The naive arm hit two CUDA OOMs at micro batch 8 — uniform sampling occasionally packs eight near-cap sequences into one micro batch (no sample packing) — and resumed cleanly from checkpoint both times; its 89M metered tokens include the retraced steps.
4. **Reversion worked mechanically.** All 19 bin weights stayed within a ~3× band for the whole run, where round 3's starved bins pinned to the epsilon floor within ~15 steps; hard bins dip early and drift back as decay plus pseudo-evidence erode the pessimistic estimate.

   ![Round-4 grade-prior-dapo bin weights](https://raw.githubusercontent.com/marin-community/marin/assets/curriculum-rl-paper/curriculum-snowball-r4-grade-prior-dapo-weight.png)

5. **A real regression: peak val-math500 fell from 0.656 to 0.562**, in both arms equally. Candidate causes we cannot separate at one seed each: half the per-step rollout count, lr-driven drift, or interaction between the anti-`\boxed{}` contract and the model's native math register. This is the main open question for round 5.
6. **The binding constraint is now generation budget, not sampling.** Retained-trajectory analysis (`trajectory_stats.py`; failures and truncations are retained exhaustively, successes sampled) shows 40–48% of rollouts still hit the 2048-token cap at end of training, and frontier bins truncate almost totally: AIME 98%, Omni-MATH 99%, TheoremQA 88%. Successful rollouts are short (~500–700 tokens), carry canonical think-token structure 85–100% of the time (rising to ~100% by step 100), and always end with the graded answer line; RL ground `\boxed{}` out of successful responses entirely (0.39 → 0.00 over the run). Grades 11+ (except theoremqa) are budget-starved, not merely hard — the model cannot finish its reasoning inside the cap. Raising the generation budget on high grades, or shaping for brevity, is the highest-leverage round-5 change.

## Appendix: reproduction

- Experiment code (pool build, launch presets, chart script): `experiments/post_training/curriculum_rl/` in [marin PR #8769](https://github.com/marin-community/marin/pull/8769); the [tracking issue](https://github.com/marin-community/marin/issues/8765) has per-round reports and mid-run findings.
- Sampler implementation: MarinSkyRL branch `curriculum-sampling` (`data.sampling.kind`, `data.sampling.weighting=group-informative`, checkpoint-resumable), upstreaming in [MarinSkyRL #470](https://github.com/marin-community/MarinSkyRL/pull/470).
- Runs: W&B project `marin-community/marin-curriculum-rl`; run names carry `2026.08.31` (Qwen), `2026.08.31.1` (Snowball round 3), and `2026.09.01.1` (Snowball round 4). Figures and `summary.json`: repo branch [`assets/curriculum-rl-paper`](https://github.com/marin-community/marin/tree/assets/curriculum-rl-paper), which also holds per-grade-vs-step breakouts and per-arm curriculum weight and pass-rate trajectories.
- MoE-under-FSDP throughput note: at micro batch 1, each Snowball update ran 32 sequential micro-steps, each re-gathering the full 134GB of sharded parameters for ~2B active parameters of compute (~29 min/step). Raising `micro_train_batch_size_per_gpu` to 4 cut the policy update from 1660s to 437s (~8.8 min/step). Check this before concluding an MoE is too slow to RL-train.
