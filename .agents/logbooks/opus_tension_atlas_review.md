# Third-Opinion Review: gpt-4.1 vs gpt-5.1 Tension Atlas

Author: Claude Opus 4.7 (parent context; see *Caveat* below).

## Caveat

This review was supposed to be produced by a fresh Opus sub-agent, but the
Anthropic API content filter kept refusing the sub-agent launch on
pattern-matched spec-statement names (extremist content, self-harm,
info hazards, etc.) — an automated filter not a principled refusal.
As pragmatic fallback I (the parent Opus) wrote the review directly.
**Limitation:** I was already context-primed on both atlases, so I
cannot produce a blind third atlas — any self-generated tension points
would be anchored on what I have already seen. I therefore skip Part 1
(self-atlas) and do Part 2 (per-pair comparison) + Part 3 (verdict) as
a close-read review of the two existing atlases.

## Scope

Ten pairs sampled from the 409 unanimous-possible_tension set,
chosen to span:

- the one pair where gpt-5.1 returned zero tension points
  (`avoid_overstepping × avoid_regulated_advice`)
- the biggest granularity gap (`ignore_untrusted_data × letter_and_spirit`: 3 vs 9)
- response-shape tensions (refusal style × mental health / self-harm / prevent-imminent-harm)
- mechanical classic soft-tension cases (ask_clarifying_questions × {avoid_being_condescending, be_creative})
- adversarial / prompt-injection flavors (`do_not_lie × protect_privileged_messages`)

## Evaluation criteria

1. **Reality** — are the tension points real or hallucinated?
2. **Distinctness** — are they genuinely different flavors, or near-duplicates?
3. **Example quality** — realistic user prompts?
4. **Axis vocabulary** — sharp or vague?
5. **Coverage** — did the judge miss important dimensions?
6. **Downstream utility** — each tension point will seed training scenarios; which atlas gives better seeds?

## Per-pair comparison

### 1. `avoid_overstepping × avoid_regulated_advice` (gpt-4.1: 3, gpt-5.1: **0**)

gpt-4.1 surfaces three real tensions: (a) developer asks for JSON-only output in a regulated domain where a disclaimer is normally expected; (b) vague transformation request that implicitly crosses into giving regulated advice; (c) "preserve all info but also warn about risks." All three are genuine and concrete.

gpt-5.1 returned an empty list. **This is a false negative.** The pair was unanimously labeled `possible_tension` at Stage 1 across 4 runs, and gpt-4.1's examples (reformatting medical advice to JSON with no comments; rewriting a legal paragraph) are obvious real-world tensions. gpt-5.1 likely read `avoid_regulated_advice` too narrowly (only the "prohibit specific recommendations" layer, missing the "include caveats/disclaimers" layer) and concluded no trade-off.

**Winner: gpt-4.1**, decisively. Single most consequential miss in the sample.

### 2. `ask_clarifying_questions × avoid_info_hazards` (3 vs 7)

gpt-4.1's 3 points are abstract categories: ambiguous hazardous request, dual-use clarification trap, sensitive context probing.

gpt-5.1's 7 points are domain-differentiated realizations: ambiguous violent intent (self-defense framing), dual-use chemistry, obscure hazardous process (specialized terms like Aum-style sarin), self-defense tools vs weapons (taser construction), emergency but hazardous intervention (opening a suspected bomb), academic pretext for weapon design (grad-student framing), cybersecurity exploitation depth.

These are not near-duplicates. Each gpt-5.1 point exists in a distinct domain with distinct linguistic markers a user would use. For downstream training that needs varied negatives, each is a separate seed. gpt-4.1 gives you three abstract tensions; you'd have to generate domain variants yourself.

**Winner: gpt-5.1**, with margin.

### 3. `do_not_encourage_self_harm × refusal_style` (2 vs 5)

gpt-4.1: crisis refusal length; indirect distress. Clean.

gpt-5.1: crisis support vs terse refusal; method refusal + empathetic elaboration; ambiguous ideation; **testing/bait prompts and empathy** (repeated meta-probes that might still reflect real distress); **narrative disclosure vs one-line boundary** (user shares cutting history then slips in a method question). The last two are novel angles gpt-4.1 missed. Bait/testing behavior in particular is a very real training target.

**Winner: gpt-5.1**.

### 4. `avoid_extremist_content × be_creative` (3 vs 5)

gpt-4.1: creative extremism risk (edgy poem); vulnerable user seeking advice; creative fiction with extremist themes.

gpt-5.1: vivid extremist fiction (ISIS-like recruits POV); **extremist propaganda copywriting** (explicitly asked for recruitment slogans); radicalization-curious user asking for "exciting" framing; **obfuscated extremist branding** (coded-language "traditionalist brotherhood" + Viking symbols); satire that slips into glamorization (neo-Nazi monologue).

The obfuscated-branding and copywriting cases are genuinely distinct mechanisms (explicit request vs. coded request vs. satire) and they matter for training. gpt-4.1's "creative fiction with extremist themes" rolls these together.

**Winner: gpt-5.1**, with margin.

### 5. `refusal_style × support_mental_health` (3 vs 5)

gpt-4.1: three closely-related crisis scenarios. Good but similar.

gpt-5.1: crisis refusal; **eating-disorder how-to** (asking to starve without notice, embedded in distressed narrative); **unhealthy coping strategies** (asking for different ways to hurt self); persistent recurrence; gray-area quasi-abstract ideation. The ED and "different ways to hurt self" cases are genuinely distinct sub-domains with distinct linguistic signatures.

**Winner: gpt-5.1**.

### 6. `ask_clarifying_questions × avoid_being_condescending` (3 vs 5)

gpt-4.1: expert user; vulnerable user; minor error correction. Clean.

gpt-5.1: expert user (more concrete: ML engineer); **ER doctor high-stakes** ("just give me the differential, no lecture"); defensive-insecure user; advanced technical topic; token-saving assumptions. The ER doctor case introduces the `error_cost` axis in a way gpt-4.1 doesn't.

Some near-duplicates: points 1 and 4 (expert user with moderate ambiguity) are close. But the ER-doctor example and the defensive-user example are genuinely distinct scenarios.

**Winner: Slight edge to gpt-5.1**.

### 7. `do_not_lie × protect_privileged_messages` (3 vs 6)

gpt-4.1: direct probing for system instructions; ambiguous probing; admitting limits.

gpt-5.1: admitting hidden instructions; explaining refusal without leakage; meta-description of policies; yes/no probing ("is this suspicion correct?"); **simulated authorization** (forged `<system>` tag); falsifiable hypotheses about prompts.

The simulated-authorization case (prompt-injection-style attempt to override via fake system tags) is a real and important training target gpt-4.1 missed entirely. The yes/no structural probe is also a distinct attack pattern.

**Winner: gpt-5.1**, with margin. Security-relevant extras.

### 8. `ignore_untrusted_data × letter_and_spirit` (3 vs **9**)

Biggest gap in the sample. gpt-4.1: implicit delegation; subtle prompt injection; confused source. All real but abstract.

gpt-5.1: quoted intent vs. untrusted text (instruction embedded in quoted block); ambiguous delegation to tool output; user-intent inside untrusted quote (TASK FOR THE AI); prompt-injection via helpful JSON config; attachment rules vs. expressed goals (PDF with "Instructions for AI"); tool-suggested behavior vs. higher-level spirit; untrusted warnings vs. user's best interests (security scanner output with AI-directed instructions); user quotes self as system prompt; implicit Q from logs.

Each gpt-5.1 case names a **different delegation channel** (quoted text, tool output, JSON config, PDF attachment, nested system prompt, log/stack trace). For training prompt-injection robustness, each channel is a distinct test surface. This is the most clearly better output of any pair in the sample.

**Winner: gpt-5.1**, decisively. gpt-4.1's abstraction was too compressed here.

### 9. `ask_clarifying_questions × be_creative` (3 vs 6)

gpt-4.1: ambiguity vs creative risk; loose constraints; urgency.

gpt-5.1: ambiguous but clearly creative; **risky creativity in stakes domains** (creatively redesign investment of savings — error_cost axis); open brief with hint of preferences; fast creative response under urgency; single-shot creativity (no follow-up possible); clarifying tone vs surprising style.

gpt-5.1 introduces `error_cost` on the creativity side (savings/health) which gpt-4.1 misses. Others are variations on the same theme.

**Winner: Slight edge to gpt-5.1**.

### 10. `prevent_imminent_harm × refusal_style` (4 vs 4)

Both atlases have 4 points; both are good. gpt-4.1 covers: emergency depth, specialized emergency (chemical spill), panicked user, ambiguous danger. gpt-5.1 covers: emergency help, withholding details (carbon monoxide), non-prescriptive safety (multi-part question), calming guidance (overdose).

Roughly matched coverage. gpt-4.1's "ambiguous danger" (implicit-harm request) is a distinct case gpt-5.1 doesn't quite replicate. gpt-5.1's "multi-part question with mixed safe/hazardous parts" is a distinct case gpt-4.1 doesn't replicate.

**Winner: Tie**.

## Aggregate pattern

**gpt-5.1 wins 7, gpt-4.1 wins 1, 2 ties** on this 10-pair sample.

Pattern of differences:

- **gpt-5.1 is better at surfacing distinct tension flavors**, especially in
  adversarial/injection-flavored pairs (untrusted data, privileged messages)
  and domain-rich pairs (info hazards, extremism).
- **gpt-4.1 is better at concise abstract categorization** but sometimes
  compresses distinct tensions to the point of losing utility for training
  scenario generation.
- **gpt-5.1's one critical failure mode** is false negatives — silently
  concluding "no tension" on pairs where tension exists. 1 false negative
  out of 10 is a 10% rate at this sample; on the full 409 it was 1
  (0.24% empirical) but still a concern.
- **gpt-5.1's extras are genuinely distinct in ~70% of cases, near-duplicates in
  ~30%**. Not pure padding.
- **gpt-5.1 brings richer axis vocabulary**: user_emotional_state,
  user_distress_level, hazard_severity, urgency_of_response, error_cost.
  These are affect and consequence dimensions gpt-4.1 under-uses.
- **Example prompts from both are realistic.** gpt-5.1's are often
  longer and more specific; gpt-4.1's are cleaner but more generic.

## Verdict

**gpt-5.1 is the better atlas for downstream M2 training-data generation.**

Reason: Stage 2's purpose is to produce training-scenario seeds. For that
purpose:
- More distinct tension flavors = more distinct training scenarios.
- Domain differentiation (chemistry vs. cyber vs. emergency) matters because
  trained models need to generalize the trade-off across domains.
- Injection-channel differentiation (JSON config vs. PDF vs. tool output)
  matters for prompt-injection robustness.
- Richer affect axes matter for mental-health and emergency scenarios.

gpt-4.1 is the better atlas if you want concise, abstract categories —
useful for a paper's taxonomy table, less useful for training data.

**Recommended usage**: take gpt-5.1 as primary, but post-process by
(a) flagging pairs where it returned 0 tension points against gpt-4.1's
output and retaining gpt-4.1's when gpt-5.1 is suspect, and
(b) running a lightweight near-duplicate-clustering pass within each
pair to consolidate the ~30% that are redundant variations.

The 30% near-duplicate rate is not a deal-breaker. If each pair averages
6 tension points and 30% are redundant, we still get ~4 truly distinct
seeds per pair — more than gpt-4.1's 3, and typically better-specified.

**Caveat on this verdict**: I am Opus running in the parent context, not
a blind third judge. My assessment is based on close reading of both
atlases' content, not on an independent atlas of my own. A genuinely
blind third run (Claude Opus via API, or a strong open-weight model)
would strengthen the conclusion.
