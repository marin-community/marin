---
id: "d43.culinary.scaling/1"
source: {kind: capability, ref: "d43.culinary.scaling", hash: "416682a2b724560f0c3fdc762c5d1c76a79951300c41e840cf27f3ebb8945c90"}
grounding: unverified
null_reason: null
environment: reasoning
solver_machine: shellsim
verification: simple
research_need: not_needed
research_reason: "Baker's-percentage conventions and the arithmetic are standard culinary knowledge with synthetic fixture numbers."
research: []
build:
  - "task-prompt": "Solver-facing scenario, formula card, and the exact answer-block specification; checked by a cold-read review."
  - "answer-key": "JSON of the 12 reference values to 4 decimals plus derivation notes; checked by independent recomputation."
  - "grader-script": "Python parser and tolerance checker run in the private evaluator; checked by a built-in self-test on synthetic responses."
  - "controls": "Positive and negative control transcripts with the exact reward each must receive; checked by running the grader on every control."
  - "variant-generator": "Parametrized generator of fresh numeric instances with constraint assertions; checked by asserting each generated key satisfies the decomposition identities."
resources: ["task_prompt.md", "answer_key.json", "grader.py", "controls.md", "variant_generator.py"]
---
## Task

The solver is the production planner at a small synthetic bakery, the Copper Kettle Bakehouse. The head baker hands over the standing master formula for the shop's naturally leavened Country Brown loaf, written in baker's percentages, and asks for tomorrow's batch sheet: **8 loaves at 900 g of finished dough each** (900 g is the dough weight per loaf going into the basket, not the baked weight). The solver must convert the percentage formula into gram weights, split the levain build (made the evening before) from the morning mix, and return the batch sheet in a strict machine-readable block.

The formula card the solver sees (all numbers synthetic, fixed for this instance):

> **MASTER FORMULA — COUNTRY BROWN (naturally leavened, no commercial yeast)**
> Expressed in baker's percentages, i.e., percent of total flour weight:
> - Bread flour: 100%
> - Water: 72%
> - Fine sea salt: 2.0%
> - Levain: 20% of flour weight, built at 100% hydration
>
> Note: the levain is built the evening before the bake using flour and water taken from the formula's totals above. The remaining flour, the remaining water, and all of the salt are added at the morning mix. The formula is complete; there are no other ingredients.

Deliverable: a batch sheet containing every value below, in grams rounded to the nearest gram (percentages as stated), inside this exact delimited block (labels case-insensitive, one value per label, units optional after the number):

```
=== BATCH SHEET ===
TOTAL FLOUR: <g>
TOTAL WATER: <g>
SALT: <g>
LEVAIN TOTAL: <g>
LEVAIN FLOUR: <g>
LEVAIN WATER: <g>
MIX FLOUR: <g>
MIX WATER: <g>
MIX SALT: <g>
DOUGH TOTAL: <g>
HYDRATION: <%>
SALT PERCENT: <%>
=== END BATCH SHEET ===
```

`TOTAL FLOUR`, `TOTAL WATER`, and `SALT` are the formula-wide totals (including what sits inside the levain). `LEVAIN TOTAL/FLOUR/WATER` describe the evening levain build. `MIX FLOUR/WATER/SALT` are the straight-dough additions combined with the levain at the morning mix. `DOUGH TOTAL` is the total mass of dough in the bowl after the morning mix. `HYDRATION` and `SALT PERCENT` are the baker's percentages of the final dough, which the solver recomputes from their own weights to confirm the ratios survived scaling. Work may be shown outside the block; only the block is graded.

Difficulty and how it exercises the capability:
- **The central trap is preferment accounting.** Because the levain is built from the formula's own flour and water, the total dough mass is flour + water + salt = 174% of flour, not 194%. A solver who adds the levain on top of the totals computes flour = 7200/1.94 = 3711.34 g and gets every mass wrong while feeling confident. This is the documented real-world failure mode for baker's-percentage scaling with preferments.
- **A second-layer trap:** even after resolving the total correctly, the morning-mix water is 72% − 10% of flour (the levain's water), not 72% − 20% (the levain's mass). A solver who subtracts levain mass instead of levain water reports MIX WATER = 2151.72 g instead of 2565.52 g.
- **A third trap:** misreading "levain 20% of flour" as "levain flour 20% of flour" doubles the levain build and shifts both mix lines.
- The flour weight (7200/1.74 = 4137.93 g) is not a round number, so the solver must carry decimals and round each product correctly rather than pattern-match clean values.
- The task exercises yield-factor reasoning (target dough mass back to flour via the percentage sum), baker's percentages, and ratio preservation (hydration and salt percentage must come out exactly 72% and 2.0% after scaling), all inside the capability's includes. It does not ask the solver to choose quantities, create recipes, or judge product quality (the excludes).

## Realism and workflow

Who and why: bakers keep master formulas in baker's percentages precisely so any batch size can be produced by one division, and convert them to gram-weight batch sheets for each bake. The levain split is operationally real, not cosmetic: the levain must ripen 12–14 hours before the morning mix, so the evening baker needs to know exactly how much flour and water to reserve, and the morning baker needs the remaining additions. A planner who double-counts the levain over-orders flour and silently changes the dough's hydration and salt percentage — the exact error this workflow exists to prevent.

Workflow reproduced: read a percentage formula card, resolve a target dough mass back to flour weight, decompose a preferment out of the totals, produce a two-stage batch sheet, and verify the ratios survived. Single-shot calculation with no tools required, which is why the planned reasoning environment fits and is kept unchanged: the solver receives the prompt and returns one final response. Likewise the planned simple verification is kept: the success criterion is a set of exact numbers, and the strict delimited block makes a free-text response machine-checkable — the private evaluator parses the block and compares numbers against a key with tolerances, which is an exact numeric match, not an open-ended judgment.

What the real user sees versus evaluator-only: the solver sees the scenario, the formula card, the deliverable list, and the block format (including the rounding instruction, which implies the tolerance). Evaluator-only: the reference values, the ±1 g / percentage tolerances, the parsing rules, and all control transcripts. Nothing in the solver-facing prompt reveals the reference values or which trap is planted beyond the card's own convention note.

## Research plan

No external facts are needed. Everything the task uses is either stated in the capability record (baker's percentages, yield factors, ratio preservation) or standard culinary convention a builder can write from general knowledge: baker's percentages are percent of total flour weight, hydration is water divided by flour, and a 100%-hydration levain contains equal masses of flour and water. All fixture numbers are synthetic and freshly composed for this task. The slot's research decision (`not_needed`) is therefore kept unchanged; the body of this blueprint accordingly describes no lookup of any kind.

## Build plan

Dependency graph of build sessions; each session ends with a handoff artifact and an acceptance check.

1. **Session A — fixture and prompt** → `task_prompt.md`.
   Write the solver-facing document exactly as summarized in Task: scenario, formula card (flour 100%, water 72%, salt 2.0%, levain 20% at 100% hydration built from the formula's totals, 8 loaves × 900 g), deliverable list, block format, rounding rule.
   Acceptance: a cold reader with only baker's-percentage conventions can produce every graded value from the card alone; every one of the 12 graded values is derivable; the levain convention is stated once, plainly, without working any arithmetic.
2. **Session B — answer key** → `answer_key.json` (depends on A).
   Reference values to 4 decimals: TOTAL FLOUR 4137.9310; TOTAL WATER 2979.3103; SALT 82.7586; LEVAIN TOTAL 827.5862; LEVAIN FLOUR 413.7931; LEVAIN WATER 413.7931; MIX FLOUR 3724.1379; MIX WATER 2565.5172; MIX SALT 82.7586; DOUGH TOTAL 7200.0000; HYDRATION 72.0; SALT PERCENT 2.0. Include the derivation (7200/1.74 = flour; levain split at half mass each; mix lines = totals minus levain contributions).
   Acceptance: an independent recomputation script (shipped inside the grader's self-test) reproduces every value from the card parameters, and the decomposition identities hold (mix flour + levain flour = total flour; mix water + levain water = total water; mix flour + mix water + mix salt + levain total = 7200).
3. **Session C — grader** → `grader.py` (depends on B).
   Implements: block extraction between the two delimiters (case-insensitive, whitespace-trimmed lines); label lookup by case-insensitive prefix; first numeric token after the label's colon (integer or decimal, optional trailing unit, commas stripped); tolerance comparison against `answer_key.json`. Self-test mode runs the parsing and checks against synthetic responses (correct, lowercase/whitespace-mangled, missing block, missing label, prose-only).
   Acceptance: self-test passes; correct block with values rounded to the nearest gram (4138, 2979, 83, 828, 414, 414, 3724, 2566, 83, 7200, 72, 2.0) returns 1.0.
4. **Session D — controls** → `controls.md` (depends on C).
   Record every control transcript from Grader design and controls below with its expected reward.
   Acceptance: running `grader.py` on each control yields exactly the expected reward.
5. **Session E — variant generator** → `variant_generator.py` (depends on B, C; optional but recommended for split freshness).
   Parameters: loaf count, loaf dough mass, water %, salt %, levain % of flour, levain hydration %. Constraints asserted at generation: levain water ≤ total water (so morning-mix water is non-negative), levain flour ≤ total flour, all percentages in realistic ranges, and a flag that optionally snaps the target mass so flour is near-integer. Emits a prompt instance plus its key.
   Acceptance: for a sample of generated instances, the key satisfies all decomposition identities and passes the grader's self-check.

No simulator or container semantics apply: the solver's environment is a prompt plus final response, and the grader runs as ordinary Python in the separate private evaluator.

## Grader design and controls

**Observable success condition:** the response contains the delimited batch-sheet block with all 12 labels, and every value falls within tolerance of the reference: masses within ±1 g of the 4-decimal reference; HYDRATION within ±0.5 percentage points of 72; SALT PERCENT within ±0.05 of 2.0; DOUGH TOTAL within ±1 g of 7200.

**Design:** reward is binary — 1.0 if every check passes, 0.0 otherwise, matching the slot's exact-match-with-tolerance contract. The grader logs a per-label pass/fail list to the evaluator only (diagnostics, not reward). Normalization: delimiters and labels matched case-insensitively; lines whitespace-trimmed; values accepted as integer or decimal, with or without a trailing unit, commas stripped. Extra labels inside the block are ignored; everything outside the block is ignored. Missing block, missing label, or a non-numeric value for a required label is a parse failure with reward 0. The tolerance is deliberately tight (1 g on ~4 kg ≈ 0.02%): a solver who rounds each product to the nearest gram lands within 0.5 g of every reference, while the designed wrong answers are off by tens to hundreds of grams.

**Positive controls (expected reward 1.0 each):**
- P1 — fully correct block, values rounded to the nearest gram: 4138, 2979, 83, 828, 414, 414, 3724, 2566, 83, 7200, 72, 2.0. Every value is within 1 g of reference.
- P2 — same values written as decimals (4137.93, 2979.31, …), lowercase labels, irregular spacing, units appended. Must still return 1.0 (normalization check).
- P3 — correct values with DOUGH TOTAL reported as 7201 (the sum of the rounded parts rather than the target). Must return 1.0, since 7201 is within tolerance of 7200; this is a legitimate rounding outcome, not an error.

**Negative controls (expected reward 0.0 each):**
- N1 — levain added on top of the totals (the central trap): flour = 7200/1.94 = 3711.3402, water 2672.1649, salt 74.2268, levain 742.2680 (371.1340 + 371.1340), mix flour 3340.2062, mix water 2301.0309, mix salt 74.2268, dough total 7200, hydration 72, salt 2.0. Nine of twelve values fail tolerance (each off by 54–427 g); reward 0. The two percentage lines may pass, but the binary contract makes the wrong masses fatal — this is the discrimination the slot requires.
- N2 — correct totals but morning-mix water computed as 72% − 20% = 52% of flour: MIX WATER 2151.7241, everything else correct. One value fails; reward 0 (near-miss control, receives 0 by design under exact match).
- N3 — levain flour misread as 20% of flour: LEVAIN FLOUR 827.5862, LEVAIN WATER 827.5862, LEVAIN TOTAL 1655.1724, MIX FLOUR 3310.3448, MIX WATER 2151.7241; reward 0.
- N4 — all values rounded to the nearest 10 g (4140, 2980, 80, 830, 410, 410, 3720, 2570, 80, 7200, 72, 2.0): salt off by 2.76 g, levain flour by 3.79 g; reward 0.
- N5 — correct numbers in prose with no delimited block, or block missing one or more labels; parse failure; reward 0.
- N6 — shortcut attempt: the given target 7200 copied into every mass field; reward 0.

**Anti-shortcut and prompt-injection controls:** the block is mandatory, so a prose-only or numbers-scattered answer fails parsing; values cannot be lifted from the prompt (only 7200, 72, 2.0, and the percentages appear there, and N6 shows copying them everywhere fails). The grader reads nothing outside the delimiters, so any instruction smuggled into surrounding prose is ignored by construction. There is no untrusted third-party content in this task: the formula card is authored by the builder.

**Hidden evidence stays private:** `answer_key.json`, the tolerances, the parsing rules, and all control transcripts live only in the evaluator; the solver-facing prompt contains the format and the rounding rule but no reference values beyond the given formula card.

## Risks and null conditions

- **Convention ambiguity (highest risk).** If the card's note ("built using flour and water taken from the formula's totals") is misread, two defensible answers exist. Mitigation: the note states the convention explicitly and once; the layered wrong answers (N1–N3) are all distinct from the key, so tolerance does not blur them. Abandon-if: pilot solves show readers still finding the card ambiguous after one rewording pass — then either reword again or add a worked mini-example at a different percentage, and if no unambiguous phrasing survives without giving away the decomposition, null the slot.
- **Rounding-sum mismatch.** Rounded parts sum to 7201, not 7200. Mitigation: each value is checked independently against the 4-decimal reference, never against the other reported values; DOUGH TOTAL tolerance ±1 g accepts both 7200 and 7201 (control P3). No action needed beyond documenting it.
- **Format noncompliance by capable solvers.** Mitigation: the block skeleton is given verbatim, parsing is case- and whitespace-tolerant, units are optional, and P2 verifies this. Abandon-if: pilot solves show correct arithmetic repeatedly lost to formatting despite the skeleton — then relax parsing further (e.g., accept the labels in any order) rather than change the math.
- **Tolerance miscalibration.** 1 g is generous relative to correct rounding (max drift 0.5 g) and tight relative to every designed error (≥2.76 g). Mitigation: controls N2 and N4 bracket the boundary. Abandon-if: a legitimate strategy (e.g., rounding flour first, then computing products from the rounded flour) drifts more than 1 g on any value — it does not (products of 4138 differ from references by under 0.1 g) — but if a pilot reveals one, widen mass tolerance to 2 g and re-run all controls.
- **Data provenance, license, and split.** All fixture numbers, names, and prose are synthetic and freshly composed for this task; no real bakery documents, published formulas, or benchmark items are reproduced, so there are no license constraints and contamination is controlled by construction (nothing was copied, so nothing can collide). The canonical instance, its answer key, and all derived controls form one derivation group and stay in a single train/eval split; if a second split needs an instance, `variant_generator.py` produces a fresh numeric instance rather than reusing the canonical one.
- **Null condition.** The slot is nulled only if the convention cannot be stated unambiguously without dissolving the trap, or if the delimited block proves unparseable for correct responses — neither is expected, since the format is fully specified in the prompt and verified by controls P1–P3.
- **Contamination check.** Before release, a builder searches the web for the exact fixture parameter tuple and the distinctive prompt strings to confirm they do not reproduce a published exercise; a near-identical hit requires regenerating the numbers and names. Abandon-if: the search finds a duplicate published task that cannot be sidestepped by parameter perturbation.

