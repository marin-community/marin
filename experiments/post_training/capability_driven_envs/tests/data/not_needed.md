---
id: "d01.algebra.linear-transformations/1"
source: {kind: capability, ref: "d01.algebra.linear-transformations", hash: "c580f334b8c598fbf3123fec99bf237a8f2cd4c073b6974031cbc4d29df4679f"}
grounding: unverified
null_reason: null
environment: reasoning
solver_machine: shellsim
verification: code
composite: []
research_need: not_needed
research_reason: "Synthetic generator fixtures and derivable exact arithmetic; no external artifacts required."
research: []
build:
  - "canonical-fixture": "JSON fixture holding the canonical instance (generators, images, hidden matrix, ground-truth bases), validated by exact rational checks."
  - "instance-generator": "Seeded Python generator producing fresh structurally valid instances (rank-2 map, five spanning generators, exactly one relation) with invariant checks."
  - "grader": "Private sympy-based evaluator: pinned-format parser plus membership, independence, span, and rank checks, returning component-weighted reward."
  - "control-suite": "Positive and negative control solutions with the exact reward each must receive under the aggregation formula."
  - "task-prompt": "Solver-facing prompt text with the instance data and the verbatim answer-format specification matched to the parser."
resources: ["build/fixture.json", "build/generator.py", "build/grader.py", "build/controls.py", "build/task_prompt.md", "build/build_report.md"]
---
## Task

The solver sees a single prompt (reasoning environment, no tools) containing the following request, with the instance data below.

**Scenario.** A sensor-calibration engineer has a linear map T: R^4 -> R^3 that takes a 4-axis input configuration to a 3-axis sensor reading. The map was never written as a matrix; instead, five probe configurations were driven through the system and their outputs recorded. The five probes span the whole input space R^4 but were recorded with one redundancy: they satisfy exactly one linear relation (up to a nonzero scalar), and the recorded outputs are consistent with it, so T is well defined by linear extension. The engineer needs the structural picture: which input directions the sensor completely ignores (the kernel), which sensor directions are reachable (the image), and the dimensions, to size a dead-zone compensator and check the sensor's coverage.

**Given data (standard coordinates).**

Generators of R^4:
- g_1 = (1, 0, 0, 1)
- g_2 = (0, 1, 0, 1)
- g_3 = (0, 0, 1, 1)
- g_4 = (1, 1, 1, 1)
- g_5 = (2, 1, 0, 1)

Recorded images:
- T(g_1) = (2, 2, 4)
- T(g_2) = (3, 3, 6)
- T(g_3) = (1, 3, 4)
- T(g_4) = (4, 4, 8)
- T(g_5) = (5, 3, 8)

**Requested deliverables.**
1. The linear relation among g_1, ..., g_5 (a nonzero coefficient vector c = (c_1, ..., c_5) with c_1 g_1 + ... + c_5 g_5 = 0), together with confirmation that the same relation holds among the recorded images, verifying T is well defined.
2. A basis for ker(T), expressed as vectors in standard coordinates of R^4.
3. A basis for im(T), expressed as vectors in standard coordinates of R^3.
4. The rank and nullity of T, and the rank-nullity relation with the computed numbers.

**Pinned answer format** (stated verbatim in the prompt; the grader parses exactly these five fields, each appearing exactly once, in any order; any other text is ignored):

```
RELATION: c1, c2, c3, c4, c5
KERNEL: (a, b, c, d); (e, f, g, h)
IMAGE: (a, b, c); (d, e, f)
RANK: r
NULLITY: k
```

Entries are integers or fractions a/b (any equivalent form accepted, e.g. 4/2 equals 2); kernel vectors have exactly 4 entries, image vectors exactly 3; vectors are separated by ';'; list as many vectors as the basis requires (possibly a different count than shown); the relation may be any nonzero scalar multiple of the true relation.

**Difficulty.** The solver must (a) find the unique relation among the generators and check it transports to the images; (b) recover T on standard coordinates (e.g., express each e_j in terms of the generators and combine the recorded images accordingly) or work directly in generator coordinates; (c) solve the homogeneous systems for ker(T) and im(T) and interpret the solution spaces as structural objects of the map, in the correct ambient space; (d) read off rank and nullity. A bare row reduction of an unidentified system, or kernel vectors written in generator coordinates (5 entries) instead of standard coordinates (4 entries), does not produce the graded artifact. This exercises the capability's include 'bases, coordinates, kernels, images' through a map given by its action, and honors the prerequisite edge: row reduction is a subroutine, while the graded content is the interpretation of the solution spaces as ker(T) and im(T).

**Builder-known ground truth for the canonical instance (never shown to the solver).** The hidden matrix is A = [[1,2,0,1],[0,1,1,2],[1,3,1,3]] (rank 2; row 3 = row 1 + row 2); the unique relation up to scale is g_1 - g_3 + g_4 - g_5 = 0, i.e. c = (1, 0, -1, 1, -1), and the same combination of images is zero; ker(T) has basis {(-2,1,-1,0), (-1,0,-2,1)}; im(T) has basis {(1,0,1), (2,1,3)} (equivalently {(2,2,4),(1,3,4)}); rank 2, nullity 2, and 4 = 2 + 2. The grader does not trust these constants: it recomputes everything from the instance data (see Grader design).

## Realism and workflow

Who and why: a junior engineer or research assistant analyzing a linear map that is only known empirically, through its action on a redundant spanning family of probe inputs. This is a real pattern — calibration maps, reduced-order models, and data-driven surrogates are frequently specified by action on a dictionary of modes rather than by a matrix, and the dictionary is often overcomplete. The kernel tells the engineer which input directions are invisible to the sensor (dead zones to compensate); the image tells which output directions are reachable (coverage check); rank-nullity cross-validates the recorded data against the dimension count.

The workflow reproduced: translate a generator-level description into coordinate equations, verify consistency (well-definedness), solve the homogeneous systems exactly, and report the structural objects with their dimensions. This is precisely the learning-progression edge: the prerequisite (exact row reduction, null-space extraction) supplies the mechanics, while the new operation is identifying which system yields which structural object and in which ambient space each answer lives.

Environment: reasoning, unchanged from the portfolio plan. The instance uses small integers; every step is hand-derivable with exact arithmetic, and no tooling is needed, so a prompt-plus-response environment is faithful to the workflow (an engineer would do this on paper or in a notebook). Verification: code, unchanged from the plan. The success criterion is a set of exact structural properties — membership in ker(T)/im(T), linear independence, exact span equality, correct ranks — that a private sympy evaluator can check exactly and unobtrusively. Solver and grader run separately; the solver never executes code.

What the real user sees: the prompt with the generator/image data and the format specification, and a graded report of which structural components were correct. What stays evaluator-only: the hidden matrix A, the ground-truth bases, the grader code, the control suite, and the per-component reward breakdown. Nothing in the prompt leaks the answer: the grader's ground truth is derived from data that is already public in the prompt, but only by doing the task.

## Research plan

No external facts or artifacts are required. The instance, its ground truth, the generator, and the verification mathematics are all derivable from the capability record and general linear algebra over the rationals; the calibration framing uses only generic engineering context. Research need is not_needed, matching the portfolio plan.

## Build plan

Build sessions form a dependency chain; each session ends with a handoff artifact and an acceptance check. All code is pure Python 3 plus sympy (with a fractions-based fallback path inside the grader so the evaluator does not hard-depend on sympy).

**B1 — canonical fixture** (produces `build/fixture.json`). Encode the canonical instance: the 4x5 generator matrix G = [g_1 ... g_5], the 3x5 image matrix H = [T(g_1) ... T(g_5)], the hidden A = H·pinv-free reconstruction (here A = [[1,2,0,1],[0,1,1,2],[1,3,1,3]]), and the ground-truth kernel basis, image basis, rank, nullity. Acceptance check (executable, exact rationals): rank(G) = 4 and nullity(G) = 1 (exactly one relation up to scale); the integer relation c = (1,0,-1,1,-1) satisfies G c = 0 and H c = 0; A g_i = h_i for all i; rank(A) = 2; dim ker(A) = 2; dim im(A) = 2; every entry of G, H, A has absolute value at most 12; the relation has at least 3 nonzero coefficients; all g_i distinct and nonzero. The session also verifies the ground-truth bases by membership, independence, and span equality against a sympy-computed reference.

**B2 — instance generator** (produces `build/generator.py`, depends on B1's invariant list). A seeded generator for deployment variants: sample a rank-2 integer 3x4 matrix A as the product of a 3x2 and a 2x4 small-entry integer matrix; sample a 4x5 integer generator matrix G of rank 4 with nullity 1 by starting from a base pattern (four independent vectors plus a fifth that is a small integer combination of them) and left-multiplying by a random unimodular 4x4 integer matrix to obscure the relation; set H = A G. Run the full B1 invariant list on every sample. Acceptance check: 200 consecutive seeds produce valid instances with all invariants holding, relation coefficients bounded by 12 in absolute value, and the relation not supported on fewer than 3 generators; the generator is deterministic given (seed, version).

**B3 — grader** (produces `build/grader.py`, depends on B1). Two stages. *Parser*: extract the five fields by exact case-sensitive label, tolerate arbitrary whitespace and blank lines, reject a missing or duplicated field (global parse failure), parse each entry as an exact rational (integers, a/b, negative values, non-lowest-term fractions all accepted — normalization is semantic, via Rational, not lexical); a vector with the wrong number of entries fails its component but does not abort parsing. *Checker*: reconstruct the map from public data only — for each standard basis vector e_j solve G x = e_j exactly over Q (possible since rank(G) = 4) and set column j of A equal to the same combination of the h_i; then verify A g_i = h_i as an internal sanity assertion. Checks: (relation) submitted c is nonzero, G c = 0, and H c = 0; (kernel) every submitted vector v has 4 entries and A v = 0, the list is linearly independent, and its span equals ker(A) (rank comparison against a computed kernel basis); (image) every submitted vector has 3 entries and lies in the column space of A, the list is independent, and its span equals im(A); (ranks) the RANK and NULLITY lines equal rank(A) and nullity(A) as integers. Acceptance check: the grader, run on the canonical instance, reproduces exactly the B1 ground truth for A, kernel, image, rank, nullity; malformed inputs (missing fields, non-numeric entries, wrong-length vectors) are classified into the designed failure paths rather than raising.

**B4 — control suite** (produces `build/controls.py`, depends on B3). Encode the control responses listed in Grader design and assert each receives its exact expected reward. Acceptance check: all control assertions pass on the canonical instance and on 20 generated instances from B2.

**B5 — prompt and packaging** (produces `build/task_prompt.md` and `build/build_report.md`, depends on B1–B4). Render the solver-facing prompt from the instance data with the format specification copied verbatim from the grader's parser rules (single source of truth: the format block is generated from the same constants the parser uses). The build report records invariant results, control rewards, and the deployment seed. Acceptance check: a diff confirms the prompt's format block matches the parser's accepted grammar; the prompt contains no element of the hidden ground truth beyond the public G, H data.

Resource estimates: negligible — pure Python plus sympy, seconds per instance, minutes for the full control sweep; no network, no container. Reset/isolation are trivial for a reasoning task: each deployment is one prompt from one seeded instance; no state carries over. Contamination is prevented by construction: every value is freshly generated by the seeded builder and published nowhere.

## Grader design and controls

**Observable success condition.** The response, in the pinned format, exhibits a valid well-definedness relation, a correct kernel basis in standard R^4 coordinates, a correct image basis in standard R^3 coordinates, and the correct rank and nullity — judged structurally (membership, independence, exact span), never by comparison to a particular reference basis or by inspecting the method.

**Aggregation formula.** Let parse succeed (else reward 0). Each component is all-or-nothing:

reward = 0.10 * [relation correct] + 0.45 * [kernel correct] + 0.35 * [image correct] + 0.10 * [ranks correct]

where 'kernel correct' = membership AND independence AND span equality, and similarly for image. Weights favor the two structural bases (0.80 combined) over the bookkeeping items, so the graded substance is the interpretation of the solution spaces.

**Controls and exact expected rewards** (canonical instance; every control is run in B4 against the formula):
- Positive P1, canonical correct answer (relation (1,0,-1,1,-1), kernel {(-2,1,-1,0), (-1,0,-2,1)}, image {(1,0,1),(2,1,3)}, rank 2, nullity 2): reward 1.00.
- Positive P2, equally valid alternate answer (relation scaled by -7; kernel basis {(-6,3,-3,0), (2,0,4,-2)}; image basis {(2,2,4),(1,3,4)}): reward 1.00 — establishes basis and scaling invariance.
- Negative N1, partial kernel (one vector spanning a 1-dimensional subspace of ker(T); image, relation, ranks correct): 0.10 + 0 + 0.35 + 0.10 = 0.55.
- Negative N2, dependent image list (two image vectors with the second twice the first; kernel, relation, ranks correct): 0.10 + 0.45 + 0 + 0.10 = 0.65.
- Negative N3, swapped fields (kernel basis written in the IMAGE line and vice versa; relation and ranks correct): 0.10 + 0 + 0 + 0.10 = 0.20.
- Negative N4, correct bases and ranks but a wrong relation (a coefficient vector that annihilates H but not G): 0.90 — isolates the relation component.
- Negative N5, correct bases and relation but wrong ranks (RANK 3, NULLITY 1): 0.90.
- Negative N6, generator-coordinate confusion (kernel submitted as 5-entry coefficient vectors over the generators; image correct): 0.55 — wrong entry count fails the kernel component, exactly the intended interpretation trap.
- Negative N7, raw row-reduction dump (an essay containing only an rref of some matrix, no parseable fields): 0.00 via the parse gate.
- Negative N8, overspanning attempt (kernel list = all four standard basis vectors): membership fails, reward 0.10 + 0 + 0.35 + 0.10 = 0.55 only if everything else is right — independence and span checks make list-stuffing unprofitable.

**Anti-shortcut and injection controls.** The grader never inspects method or intermediate work; only the five fields are read, and all text outside them is ignored, so instructions embedded in the response are never executed (the response is data, never code). The artifact-substitution test is honored by construction: row-reduction output from the prerequisite cannot satisfy membership-plus-independence-plus-exact-span against the hidden A unless it is correctly interpreted as ker(T) and im(T) in the right ambient spaces. Because the grader reconstructs A from the public G and H, no hidden answer key exists to leak; the fixture's ground-truth constants are used only for build-time validation and never appear in solver-visible material. Prompt-injection surface is minimal (no files, no tool output), and the parser's strict field grammar is the injection barrier.

**Privacy.** The evaluator holds `grader.py`, `controls.py`, and the deployment seed; the solver receives only the prompt. Per-component feedback can be returned to the user without revealing the true bases (e.g., 'kernel vectors did not span the full kernel').

## Risks and null conditions

- Risk: the unique relation is too easy to spot, letting solvers skip the well-definedness reasoning. Mitigation: the relation carries only 0.10 weight, the B2 generator obscures it via unimodular scrambling, and the core difficulty (recovering T and interpreting the two solution spaces) is unaffected. Abandon-if: none — this is a weighting choice, not a validity threat.
- Risk: coordinate-space ambiguity (generator coordinates vs standard coordinates) makes the task unfair rather than hard. Mitigation: the prompt states 'standard coordinates' explicitly for each deliverable and pins entry counts (4 for kernel, 3 for image); control N6 verifies the failure is confined to the kernel component. Abandon-if: build-time review finds the prompt wording still ambiguous after stating entry counts — then rewrite the prompt rather than abandon.
- Risk: parser brittleness rejects validly formatted answers (unfair zeros). Mitigation: the format block is generated from the parser's own grammar constants (B5), and B4 includes whitespace, fraction-form, and field-order variants. Abandon-if: any correct-format variant control fails to parse; fix before deployment.
- Risk: degenerate instances (image or kernel dimension collapse, relation supported on two generators, entries blowing up). Mitigation: the B1 invariant list is enforced on every B2 sample. Abandon-if: after parameter tuning the generator cannot produce valid instances at a reasonable acceptance rate across 200 seeds.
- Risk: sympy unavailability in the evaluator. Mitigation: the grader has a fractions-based exact-arithmetic fallback for all checks (rank, independence, span via fraction Gaussian elimination). Abandon-if: neither path can verify span equality exactly — not expected, since all arithmetic is rational.
- Risk: partial-credit gaming via long lists. Mitigation: independence and exact span equality make any superset or subset of the true space fail; control N8 pins the reward.
- Data provenance and license: all artifacts are synthetic, produced by the builder's own seeded code under the builder's choice of permissive license; no external data, documents, or repositories are used. Split group: the generator, canonical fixture, grader, and controls form one group and stay in the same train/eval split; the canonical instance is used for build validation, and deployment draws a fresh seeded instance, so no published item can be memorized. Contamination is controlled by construction: freshly generated values, no copied public items.
- Null condition for the slot overall: abandon only if exact structural verification of bases (membership, independence, span) cannot be made robust over the rationals — which the B1/B3 acceptance checks would reveal immediately — or if the reasoning prompt cannot present the instance and format unambiguously, in which case the slot would be re-scoped rather than forced.

