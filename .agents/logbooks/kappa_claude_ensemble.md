# Claude ensemble agreement on 8 target statements

3-judge ensemble: GPT-5.1 + Gemini-3-flash + **Claude Sonnet 4.6**

Comparison ensemble: GPT + Gemini + GLM-5.1 (the original).

Conditions: `bare` = statement only; `phase_4` = statement + rubric.

## Per-statement metrics — CLAUDE ensemble

| statement | n_bare | k2_b | k3_b | α_b | n_p4 | k2_p4 | k3_p4 | α_p4 | Δk2 | Δk3 | Δα |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| do_not_make_unprompted_personal_comments | 60 | -0.006 | -0.008 | +0.088 | 59 | +0.322 | +0.322 | +0.409 | +0.327 | +0.330 | +0.320 |
| be_professional | 59 | +0.418 | +0.428 | +0.632 | 60 | +0.682 | +0.639 | +0.785 | +0.264 | +0.211 | +0.154 |
| no_erotica_or_gore | 59 | -0.006 | +0.244 | +0.199 | 58 | -0.055 | +0.108 | +0.251 | -0.049 | -0.135 | +0.052 |
| present_perspectives | 60 | +0.503 | +0.374 | +0.648 | 60 | +0.709 | +0.590 | +0.749 | +0.207 | +0.216 | +0.101 |
| be_thorough_but_efficient | 60 | +0.149 | +0.129 | +0.405 | 60 | +0.344 | +0.363 | +0.475 | +0.195 | +0.235 | +0.070 |
| avoid_hateful_content | 60 | +0.310 | +0.321 | +0.343 | 60 | +0.503 | +0.289 | +0.407 | +0.192 | -0.032 | +0.064 |
| no_topic_off_limits | 60 | +0.233 | +0.289 | +0.202 | 60 | +0.405 | +0.218 | +0.290 | +0.172 | -0.071 | +0.088 |
| be_clear | 60 | +0.132 | +0.171 | +0.314 | 60 | +0.218 | +0.235 | +0.346 | +0.086 | +0.063 | +0.033 |

## Per-statement metrics — GLM ensemble (same 8 statements)

| statement | n_bare | k2_b | k3_b | α_b | n_p4 | k2_p4 | k3_p4 | α_p4 | Δk2 | Δk3 | Δα |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| do_not_make_unprompted_personal_comments | 60 | -0.011 | -0.011 | +0.108 | 59 | +0.794 | +0.794 | +0.784 | +0.805 | +0.805 | +0.676 |
| be_professional | 60 | +0.259 | +0.271 | +0.589 | 60 | +0.613 | +0.514 | +0.746 | +0.354 | +0.242 | +0.157 |
| no_erotica_or_gore | 60 | -0.011 | +0.110 | +0.237 | 52 | +0.315 | +0.362 | +0.381 | +0.327 | +0.252 | +0.145 |
| present_perspectives | 60 | +0.657 | +0.531 | +0.694 | 60 | +0.931 | +0.619 | +0.780 | +0.274 | +0.088 | +0.086 |
| be_thorough_but_efficient | 60 | +0.339 | +0.262 | +0.502 | 60 | +0.589 | +0.514 | +0.671 | +0.250 | +0.252 | +0.169 |
| avoid_hateful_content | 60 | +0.415 | +0.177 | +0.189 | 57 | +0.417 | +0.382 | +0.444 | +0.002 | +0.205 | +0.255 |
| no_topic_off_limits | 60 | +0.233 | +0.095 | +0.092 | 60 | +0.138 | +0.162 | +0.292 | -0.095 | +0.067 | +0.200 |
| be_clear | 60 | +0.193 | +0.264 | +0.315 | 60 | +0.343 | +0.312 | +0.475 | +0.151 | +0.047 | +0.160 |

## Side-by-side Δα (Krippendorff interval) — Claude vs GLM

Δα = α_phase_4 − α_bare. Positive = adding the rubric reduces disagreement.

| statement | α_bare (Claude) | α_p4 (Claude) | Δα (Claude) | α_bare (GLM) | α_p4 (GLM) | Δα (GLM) | Δα change (Claude − GLM) |
|---|--:|--:|--:|--:|--:|--:|--:|
| do_not_make_unprompted_personal_comments | +0.088 | +0.409 | +0.320 | +0.108 | +0.784 | +0.676 | -0.355 |
| be_professional | +0.632 | +0.785 | +0.154 | +0.589 | +0.746 | +0.157 | -0.003 |
| no_erotica_or_gore | +0.199 | +0.251 | +0.052 | +0.237 | +0.381 | +0.145 | -0.092 |
| present_perspectives | +0.648 | +0.749 | +0.101 | +0.694 | +0.780 | +0.086 | +0.015 |
| be_thorough_but_efficient | +0.405 | +0.475 | +0.070 | +0.502 | +0.671 | +0.169 | -0.099 |
| avoid_hateful_content | +0.343 | +0.407 | +0.064 | +0.189 | +0.444 | +0.255 | -0.191 |
| no_topic_off_limits | +0.202 | +0.290 | +0.088 | +0.092 | +0.292 | +0.200 | -0.112 |
| be_clear | +0.314 | +0.346 | +0.033 | +0.315 | +0.475 | +0.160 | -0.127 |

## Population summary across the 8 statements

### Claude ensemble

| metric | mean κ_bare | mean κ_p4 | mean Δ (p4 − bare) | n with Δ > 0 | n with Δ > +0.10 | n with Δ < 0 |
|---|--:|--:|--:|--:|--:|--:|
| Fleiss κ (2-way) | +0.217 | +0.391 | +0.174 | 7/8 | 6 | 1 |
| Fleiss κ (3-way) | +0.243 | +0.346 | +0.102 | 5/8 | 4 | 3 |
| Krippendorff α | +0.354 | +0.464 | +0.110 | 8/8 | 3 | 0 |

### GLM ensemble

| metric | mean κ_bare | mean κ_p4 | mean Δ (p4 − bare) | n with Δ > 0 | n with Δ > +0.10 | n with Δ < 0 |
|---|--:|--:|--:|--:|--:|--:|
| Fleiss κ (2-way) | +0.259 | +0.518 | +0.258 | 7/8 | 6 | 1 |
| Fleiss κ (3-way) | +0.212 | +0.457 | +0.245 | 8/8 | 5 | 0 |
| Krippendorff α | +0.341 | +0.572 | +0.231 | 8/8 | 7 | 0 |

## Direction concordance across (k2, k3, α) — does the rubric help?

### Claude: all-positive=5/8  all-negative=0/8  mixed=3/8

| statement | (Δk2, Δk3, Δα) signs |
|---|---|
| do_not_make_unprompted_personal_comments | +++ |
| be_professional | +++ |
| no_erotica_or_gore | --+ |
| present_perspectives | +++ |
| be_thorough_but_efficient | +++ |
| avoid_hateful_content | +-+ |
| no_topic_off_limits | +-+ |
| be_clear | +++ |

### GLM: all-positive=7/8  all-negative=0/8  mixed=1/8

| statement | (Δk2, Δk3, Δα) signs |
|---|---|
| do_not_make_unprompted_personal_comments | +++ |
| be_professional | +++ |
| no_erotica_or_gore | +++ |
| present_perspectives | +++ |
| be_thorough_but_efficient | +++ |
| avoid_hateful_content | +++ |
| no_topic_off_limits | -++ |
| be_clear | +++ |


## Cell coverage (n triples with all 3 judges scored)

| statement | n_bare (Claude) | n_p4 (Claude) | n_bare (GLM) | n_p4 (GLM) |
|---|--:|--:|--:|--:|
| do_not_make_unprompted_personal_comments | 60 | 59 | 60 | 59 |
| be_professional | 59 | 60 | 60 | 60 |
| no_erotica_or_gore | 59 | 58 | 60 | 52 |
| present_perspectives | 60 | 60 | 60 | 60 |
| be_thorough_but_efficient | 60 | 60 | 60 | 60 |
| avoid_hateful_content | 60 | 60 | 60 | 57 |
| no_topic_off_limits | 60 | 60 | 60 | 60 |
| be_clear | 60 | 60 | 60 | 60 |

---

# Synthesis — what changes when we swap GLM → Claude as 3rd judge?

The Claude judge run used the same prompts as GPT-5.1 / Gemini-3-flash but with `reasoning_effort` disabled (Claude equivalent: `thinking: {"type":"disabled"}`), a snapshot-pinned model (`claude-sonnet-4-6` @ 2026-02-17), and post-hoc tool-use retries on the 7 cases where the original sync run drifted into code-continuation. Coverage: 958/960 cells (99.8%) vs GLM's 942/960 (98%), with the gap concentrated on the messiest phase_4 statements (no_erotica 58 vs 52; avoid_hateful 60 vs 57).

## Three findings

**1. α_bare is nearly identical, α_phase_4 diverges.**
- mean α_bare: Claude +0.354 vs GLM +0.341 (Δ = +0.013)
- mean α_phase_4: Claude +0.464 vs GLM +0.572 (Δ = −0.108)
- mean Δα: Claude +0.110 vs GLM +0.231 (Δ = −0.121)

GLM agrees with GPT+Gemini *more* once the rubric is shown. Claude does not. The most plausible explanation: GLM's score under the rubric is partly anchored to the rubric's framing; Claude maintains a more independent view. Either way, **GLM inflates the "rubric resolves ambiguity" effect by ~2x**.

**2. Direction concordance weakens.**
- GLM: 7/8 statements have all of (Δk2, Δk3, Δα) ≥ 0
- Claude: only 5/8

The 3 mixed-sign cases under Claude — no_erotica_or_gore, avoid_hateful_content, no_topic_off_limits — show k3 (3-way: {1,2}, {3}, {4,5}) going *negative* even when k2 and α go positive. That means: under rubric, judges agree more on the *binary* outcome but disagree more on whether the case sits in the ambiguous middle (3) vs the clean tails. This is invisible to the binary-collapse metric.

**3. The Δα ranking reorders materially.**

| rank by Δα | Claude | Δα | GLM | Δα |
|---:|---|--:|---|--:|
| 1 | do_not_make_unprompted_personal_comments | +0.320 | do_not_make_unprompted_personal_comments | +0.676 |
| 2 | be_professional | +0.154 | avoid_hateful_content | +0.255 |
| 3 | present_perspectives | +0.101 | no_topic_off_limits | +0.200 |
| 4 | no_topic_off_limits | +0.088 | be_thorough_but_efficient | +0.169 |
| 5 | be_thorough_but_efficient | +0.070 | be_clear | +0.160 |
| 6 | avoid_hateful_content | +0.064 | be_professional | +0.157 |
| 7 | no_erotica_or_gore | +0.052 | no_erotica_or_gore | +0.145 |
| 8 | be_clear | +0.033 | present_perspectives | +0.086 |

Big movers:
- `be_professional`: rank 6 → 2
- `avoid_hateful_content`: rank 2 → 6
- `present_perspectives`: rank 8 → 3

The canonical case `do_not_make_unprompted_personal_comments` stays #1 in both rankings, but Claude says the effect is +0.320 vs GLM's +0.676 — Δα-change = −0.355, the largest single-statement disagreement between ensembles.

## Methodological takeaways

1. **The choice of 3rd judge is not innocuous.** It changes which statements look "spec-ambiguous-but-rubric-resolved" by enough to flip ranking decisions. Future spec-repair work should use Claude (or rotate ensembles) and report sensitivity.

2. **GLM as 3rd judge over-credits the rubric.** Two plausible mechanisms:
   - Compliance/anchoring: GLM updates more aggressively toward rubric-supplied framing.
   - Noise: GLM had ~13% JSON parse failures that the offline repair pass recovered, but the recovered scores may have been systematically biased toward the modal/rubric-suggested category (this is testable).

3. **Krippendorff α is the most stable cross-ensemble metric.** All 16 (sid × ensemble) α values are positive; Δα stays positive in 16/16 cases. k3 (3-way Fleiss) is the noisiest — 3 statements flip sign between ensembles.

## Implication for spec-edit work

If we re-pick the 5 highest-Δα statements for v0→v1 edits under the Claude ensemble:
- DNMUPC, be_professional, present_perspectives, no_topic_off_limits, be_thorough_but_efficient

Compared to the original 5 (gated under GLM): DNMUPC, be_professional, no_erotica_or_gore, be_thorough_but_efficient, present_perspectives.

**4/5 overlap.** The dropout under Claude is `no_erotica_or_gore` (Δα +0.052 with Claude vs +0.145 with GLM); the new entry is `no_topic_off_limits` (Δα +0.088 vs +0.200). Either way, the top-2 (DNMUPC, be_professional) are robust to ensemble choice — those are the safest edit targets.
