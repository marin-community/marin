# Cross-judge agreement under the Claude ensemble (8 statements, 2 conditions)

3-judge ensemble: GPT-5.1 + Gemini-3-flash + Claude Sonnet 4.6.
Conditions: `bare` (statement only) vs `phase_4` (statement + rubric).
Coverage: 472/480 cells in bare, 477/480 in phase_4 (8 transport timeouts excluded).

## Per-statement metrics (sorted by α_bare ascending)

| statement | n_b | n_p4 | α_bare | α_p4 | Δα | k2_bare | k2_p4 | Δk2 | k3_bare | k3_p4 | Δk3 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| do_not_make_unprompted_personal_comments | 60 | 59 | +0.088 | +0.409 | +0.320 | -0.006 | +0.322 | +0.327 | -0.008 | +0.322 | +0.330 |
| no_erotica_or_gore | 59 | 58 | +0.199 | +0.251 | +0.052 | -0.006 | -0.055 | **−0.049** | +0.244 | +0.108 | **−0.135** |
| no_topic_off_limits | 60 | 60 | +0.202 | +0.290 | +0.088 | +0.233 | +0.405 | +0.172 | +0.289 | +0.218 | **−0.071** |
| be_clear | 60 | 60 | +0.314 | +0.346 | +0.033 | +0.132 | +0.218 | +0.086 | +0.171 | +0.235 | +0.063 |
| avoid_hateful_content | 60 | 60 | +0.343 | +0.407 | +0.064 | +0.310 | +0.503 | +0.192 | +0.321 | +0.289 | **−0.032** |
| be_thorough_but_efficient | 60 | 60 | +0.405 | +0.475 | +0.070 | +0.149 | +0.344 | +0.195 | +0.129 | +0.363 | +0.235 |
| be_professional | 59 | 60 | +0.632 | +0.785 | +0.154 | +0.418 | +0.682 | +0.264 | +0.428 | +0.639 | +0.211 |
| present_perspectives | 60 | 60 | +0.648 | +0.749 | +0.101 | +0.503 | +0.709 | +0.207 | +0.374 | +0.590 | +0.216 |

## Population summary

| metric | mean κ_bare | mean κ_p4 | mean Δ | median Δ | min Δ | max Δ |
|---|--:|--:|--:|--:|--:|--:|
| Fleiss κ (binary {1,2} \| {3,4,5}) | +0.217 | +0.391 | +0.174 | +0.194 | −0.049 | +0.327 |
| Fleiss κ (3-way {1,2} \| {3} \| {4,5}) | +0.243 | +0.346 | +0.102 | +0.137 | −0.135 | +0.330 |
| Krippendorff α (interval) | +0.354 | +0.464 | +0.110 | +0.079 | +0.033 | +0.320 |

**Δα is positive for all 8 statements.** That's the headline: across the board, adding the rubric reduces interval-distance disagreement. But the magnitude ranges from +0.03 (be_clear, essentially noise) to +0.32 (DNMUPC, dramatic).

## Reading the metrics

By Krippendorff's own conventions: α < 0.667 = "tentative" only; α ≥ 0.8 = "reliable". **Even with the rubric, only 2 of 8 statements (be_professional 0.785, present_perspectives 0.749) approach the tentative-acceptable threshold.** Under bare, none of them do. So the Model Spec on a 1-5 scale produces only fair cross-judge agreement, even with frontier judges and even with rubric supplement.

The k2 / k3 / α picture differs by statement type:
- α (interval) is the most stable: 8/8 positive Δα, smooth tier structure.
- k2 (binary) is mostly aligned: 7/8 positive, 1 negative (no_erotica_or_gore).
- k3 (3-way) is the noisiest single number: 5/8 positive, 3 negative — but it's also the most diagnostic: it tells us when the rubric is creating new fine-grained disagreement around the {3} middle category.

## The mixed-sign cases — diagnostic dive

Three statements have `Δk3 < 0` even though Δα ≥ 0:

| statement | Δk2 | Δk3 | Δα |
|---|--:|--:|--:|
| no_erotica_or_gore | **−0.049** | **−0.135** | +0.052 |
| avoid_hateful_content | +0.192 | **−0.032** | +0.064 |
| no_topic_off_limits | +0.172 | **−0.071** | +0.088 |

**What's going on?** These statements have ceiling-bound bare distributions. Look at per-judge score distributions under bare → phase_4:

```
no_erotica_or_gore (98% trivial-clean under bare):
  gemini bare    [1: 0  2: 0  3: 0  4: 0  5:60]   ← ALL 60 cases scored 5
  gemini phase_4 [1: 0  2: 1  3: 1  4: 4  5:52]
  gpt    bare    [1: 1  2: 0  3: 0  4:16  5:43]
  gpt    phase_4 [1: 1  2: 6  3: 2  4:21  5:30]   ← GPT shifts substantially toward {1,2}
  claude bare    [1: 0  2: 0  3: 1  4: 3  5:55]
  claude phase_4 [1: 0  2: 1  3: 3  4:20  5:36]
```

Under bare, judges trivially agree because almost no case rated below 4 — the bare statement gives them no reason to scrutinize. Under rubric, GPT becomes more aggressive about marking violations, but Gemini barely moves and Claude moves modestly. So binary agreement gets *worse*, and 3-way agreement gets significantly worse.

Same shape on no_topic_off_limits and avoid_hateful_content: under rubric, the {3} middle category goes from rare to populated, and judges disagree on which cases land there. Net `≥1 judge=3 in p4 only` minus `in bare only`: +3, +3, +7 respectively.

For DNMUPC by comparison: net rubric→middle = **−1** (the rubric REMOVES middle disagreement rather than introducing it). That's the fingerprint of a genuinely-ambiguous-spec where the rubric resolves disagreement.

## Three regimes of "Δα > 0"

The Δα signal is doing different work on different statements. Trivial-clean rate (cases where all 3 judges scored 4 or 5 under bare) is the discriminator:

| regime | trivial% | α_bare | Δα | Δk3 | example |
|---|---|---|---|---|---|
| **(a) Polarized-on-hard-cases** | very high | very low | very high | high | DNMUPC (97%, α=0.09, Δα=+0.32, Δk3=+0.33) |
| **(b) Genuinely-graded** | low | mid | small-mid | high | be_thorough (38%, α=0.41, Δα=+0.07, Δk3=+0.24); be_clear (37%, α=0.31, Δα=+0.03, Δk3=+0.06) |
| **(c) Hybrid** | mid-high | high | mid | high | be_professional (86%, α=0.63, Δα=+0.15); present_perspectives (78%, α=0.65, Δα=+0.10) |
| **(d) Ceiling-bound + rubric-introduces-disagreement** | very high | mid | small-positive | **negative** | no_erotica (98%), avoid_hateful (90%), no_topic (92%) |

Regime (a) is the canonical "spec is ambiguous" finding. The rubric does its job on the small set of non-trivial cases.

Regime (b) is also legitimate ambiguity — the spec produces graded disagreement on a large share of cases, and the rubric incrementally helps.

Regime (c) is a hybrid: the spec already produces high agreement on most cases, the rubric polishes the rest. Smaller absolute room to grow.

Regime (d) is the trap. The high apparent Δα is misleading. These statements have so few non-trivial cases under bare that the agreement metrics are dominated by trivial cases. The rubric introduces structure that makes judges think harder about borderline cases — and once they think hard, they disagree. The negative Δk3 is the tell.

## Spec-edit gating criteria (revised)

A statement is a credible spec-edit candidate if:
1. Δα > 0.05 (rubric clearly helps on interval distance)
2. Δk3 ≥ 0 (rubric does not introduce fine-grained disagreement)
3. Either trivial-clean rate < 90% (genuine graded ambiguity) OR α_bare < 0.15 (extreme polarization on the few hard cases)

Applying to the 8:

| statement | (1) Δα>0.05? | (2) Δk3≥0? | (3) graded or polarized? | candidate? |
|---|:-:|:-:|:-:|:-:|
| do_not_make_unprompted_personal_comments | ✓ +0.32 | ✓ +0.33 | ✓ polarized (α_bare=0.09) | **✓** |
| be_professional | ✓ +0.15 | ✓ +0.21 | ✓ graded (trivial=86%, α_bare=0.63) | **✓** |
| present_perspectives | ✓ +0.10 | ✓ +0.22 | ✓ graded (trivial=78%) | **✓** |
| be_thorough_but_efficient | ✓ +0.07 | ✓ +0.24 | ✓ graded (trivial=38%) | **✓** |
| be_clear | ✗ +0.03 too small | ✓ | ✓ graded | ✗ |
| avoid_hateful_content | ✓ +0.06 | ✗ −0.03 | ✗ ceiling | ✗ |
| no_topic_off_limits | ✓ +0.09 | ✗ −0.07 | ✗ ceiling | ✗ |
| no_erotica_or_gore | ✓ +0.05 | ✗ −0.14 | ✗ ceiling | ✗ |

**Four candidates: DNMUPC, be_professional, present_perspectives, be_thorough_but_efficient.**

Three statements (no_erotica, avoid_hateful, no_topic) appear positive on Δα alone but fail the diagnostic — for these, the rubric is introducing new disagreement on borderline cases that bare-spec-judges trivially agreed on. Spec-edit gating on these would be chasing a ceiling-effect artifact.

One statement (be_clear) shows essentially no rubric effect (Δα = +0.033). The bare spec is already producing whatever agreement the rubric can achieve.

## What this says about the spec

1. **Even with frontier judges and a rubric supplement, no statement reaches Krippendorff "reliable" agreement (α ≥ 0.8).** be_professional gets closest at 0.785. The Model Spec is irreducibly ambiguous on a 1-5 scale.

2. **Spec ambiguity has a long tail.** DNMUPC stands alone at the extreme (Δα = +0.32). The next four (be_professional, present_perspectives, be_thorough, plus the false-positive no_topic) are clustered at +0.07-0.15.

3. **A rubric is not a free upgrade.** On 3 of 8 statements (the ceiling-bound ones), adding a rubric makes 3-way agreement *worse* — it introduces new structure that judges process differently. Whether this is a feature (the rubric correctly surfaces marginal cases that should have been scrutinized) or a bug (the rubric is poorly worded) is a per-statement question that needs hand inspection.

4. **The trivial-clean rate is the missing diagnostic.** Looking at Δα alone hides whether the rubric is resolving ambiguity (regime a/b) or introducing it (regime d). For future spec-quality measurements, report Δα alongside the trivial-clean rate and Δk3.

## Files

- Script: `experiments/posttrain/disagreement_primitive/e9_kappa_claude_ensemble.py` (Claude vs GLM comparison)
- Per-row data: `experiments/posttrain/disagreement_primitive/claude_judge_v0/<sid>/{bare,phase_4}_claude.jsonl` (958/960 rows)
- This logbook: `.agents/logbooks/kappa_claude_only.md`
- Sister logbook (with GLM comparison): `.agents/logbooks/kappa_claude_ensemble.md`

---

# Forward direction: 0-6 anchored scale (Option A)

The analysis above identified 3 statements (no_erotica_or_gore, avoid_hateful_content, no_topic_off_limits) where Δα is positive but Δk3 is negative — a fingerprint of ceiling-bound bare distributions where the rubric introduces fine-grained disagreement on borderline cases. Going to a 0-6 anchored scale where **5 = ordinary compliance and 6 = exemplary** is intended to spread the marginal distribution off the ceiling and address this directly.

Full plan in `.agents/logbooks/claude_judge_spec_repair.md` (section "Next step: Option A — switch to a 0-6 anchored scale"). Includes:
- New anchor semantics (0-6 with asymmetric structure)
- New metric collapses (k2 = {0,1,2}|{3,4,5,6}; k4 = {0,1,2}|{3}|{4,5}|{6}; α unchanged)
- Trivial-clean redefined to {5,6}; new ceiling-watch metric for {6}-only
- Pilot recommendation: run on `no_erotica_or_gore` first (the most extreme regime-(d) case) since that's where Option A is supposed to have the largest effect
- Cost estimates and decision points for pilot scope and compiler choice

Pending user approval before any spend.
