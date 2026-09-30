# Task `d14-hardware-fpnorm-combinational-0001`

**Combinational floating-point normalization stage.**  The exported, validated
deliverable of run `/muchanem/cap-recipe-v1-slot3-current-001`.

This document is a rendering of the exported TaskSpec.  The authoritative
bytes are `validated/d14.hardware.digital.combinational_logic-3-d0a49adeb2d7/`
(6 files), which are byte-identical to the item's Harbor package and validate
clean against `vendor/task_spec/task-spec-v0.9.json`.

---

## Identity

| field | value |
| --- | --- |
| `id` | `d14-hardware-fpnorm-combinational-0001` |
| `schema_version` | `0.9` |
| `difficulty` | 7 |
| `success_policy` | `all_required_steps` |
| `metadata.task_shape` | `answer` |
| steps | 1 (`step-1`) |
| `specification_sha256` | `675aab480306b1bfbf906e4308e1010a527a523094178e3eb23fa0b0d5a402fe` |
| `lowering_version` | `0.9` |
| `harbor_revision` | `93147ea9e07b04ec8d2eb5afd2916386f1aacc69` |

`coverage_tags`: `artifact:formula`, `competency:algorithm_design`,
`shape:constrained_generation`, `subject:computing.digital_logic`

### Provenance

| field | value |
| --- | --- |
| `source.dataset` | `synthetic-capability-proposals` |
| `source.revision` | `483538cb7bdf96ccd12bbadae737a335b1aebd1567e0e0b7eff1fc23ca3a4ec6` (the admitted proposal) |
| `source.row` | `d0a49adeb2d7771287719f0d339258ff99cd5b35b3f2b002275ea96cfce67f94` (proposal hash) |
| `source.importer_revision` | `marin/c7cf159173+15531bf06b` |
| catalog | `new_catalog.json` @ `2026.09.20-cross-domain-v3` |
| capability | `d14.hardware.digital.combinational_logic` — *Synthesize combinational logic* |
| subject | D14 — Computer Hardware, Embedded Systems & Robotics |

The capability's stated outcome: *"Derive and implement combinational logic
from a supplied behavioral specification with explicit width, priority,
signedness, and invalid-input behavior."*  Its declared **excludes** are
state-retaining behavior and software implementations of the same function —
both of which this task enforces mechanically (the DAG/acyclicity check kills
any latching construction, and the deliverable is an equation set, not code).

---

## Environment and interface

| field | value |
| --- | --- |
| environment class | **reasoning** — no shell, no files, no toolchain |
| `requirements.state.image` | `null` |
| `requirements.state.workdir` | `/app` |
| `requirements.state.setup_commands` | none |
| `requirements.capabilities` / `action_interfaces` / `resources` | empty |
| `binding.environment` | `{"kind": "none"}` |
| `binding.tools` | `[]` |
| `task.toml` | `allow_internet = false` |
| rendering | `plain` — `assistant_final`, extractor `plain`, surface `original`, v0.2 |
| `step.context_requirement` | `instruction_and_workspace` |
| `step.answer_requirements` | `{"kind": "json"}` |

The solver receives one prompt and returns one JSON object as its final
message.  Nothing else is available to it and nothing else is graded.

---

## The prompt

What follows is `instruction.md` in full — the complete solver-facing surface,
14 KB across 384 lines.  It embeds two normative documents authored by the
build: `spec_fpnorm.md` (the behavioral contract) and `format_contract.md`
(the deliverable contract).

---

We're adding a gradual-underflow mode to the soft-float multiplier datapath on our FPGA, and I need the post-shift normalization stage as a Boolean equation set I can drop into our structural netlist flow. Stage contract: inputs are an 8-bit unsigned fraction frac[7:0] and a 4-bit biased exponent exp[3:0] (bias 7); outputs are a 7-bit normalized mantissa (implicit leading one removed), the adjusted exponent, a zero flag for an all-zero fraction, and a sticky sub flag for when the result lands subnormal at the exponent floor. The full behavioral spec (with worked examples) and our equation-format contract are attached. Please derive the logic and return fpnorm_equations.json.

Return the complete JSON object as your final answer and nothing else: no prose, no explanation, no code fences — the answer message must consist of the JSON text alone.

# Post-shift normalization stage — normative behavioral specification

**Document:** `spec_fpnorm.md` · **Revision:** 1.0 · **Status:** normative

This document is the complete behavioral contract for the post-shift
normalization stage of the gradual-underflow soft-float multiplier datapath.
It is self-contained: every one of the 4096 possible input combinations is
defined by rules R0–R5 below, with no don't-cares and no undefined behavior.
Where this document and any other description disagree, this document wins.

Your deliverable is a Boolean equation set (`fpnorm_equations.json`) whose 13
output bits reproduce this behavior exactly; the file format and expression
grammar are defined in the companion `format_contract.md`.

---

## R0 — Fields and encodings

| Field | Width | Encoding |
| --- | --- | --- |
| `frac` | 8 bits, `frac[7:0]` | unsigned fixed-point fraction; `f7` is the MSB, `f0` the LSB |
| `exp`  | 4 bits, `exp[3:0]`  | unsigned biased exponent; `e3` is the MSB, `e0` the LSB |

- All comparisons and arithmetic in this specification are **unsigned**.
- The bias is 7. The bias is informational only: it plays no role in any rule
  below and you never need it to compute an output.
- Biased exponent 0 is a **valid, ordinary encoding** — there are no reserved
  or special exponent encodings in this stage. Exponent clamping at the biased
  floor (biased 0) is handled by R4, not by treating `exp = 0` as special.
- The stage consumes the multiplier's post-shift product fraction and
  exponent; the product may already be denormalized, which is why the stage
  both left-shifts and detects underflow below the biased floor.

## R1 — Zero short-circuit

If `frac == 0` (all eight fraction bits are zero):

```
zero = 1
sub  = 0
mant = 0000000   (all seven mantissa bits are 0)
nexp = 0000      (all four exponent bits are 0)
```

**regardless of `exp`.** The zero case overrides everything else in this
specification (see worked example C).

## R2 — Leading-zero count

If `frac != 0`:

```
zero = 0
lz   = number of leading zero bits of the 8-bit fraction frac[7:0]
```

`lz` is the count of consecutive zero bits starting at `f7` before the first
1 bit. `lz` ranges over 0..7. Equivalently, `lz = k` means `f7..f(8-k)` are
all 0 and `f(7-k) = 1`.

## R3 — Normalizing shift

If `frac != 0`:

```
s       = min(lz, exp)        // the shift never goes below the biased floor
shifted = frac << s            // arithmetic on the unsigned 8-bit value
mant    = shifted[6:0]        // the low 7 bits of the shifted fraction
```

The shift amount is capped by the exponent: you shift left by `lz` positions
to normalize, but never by more than `exp` positions. Because `s <= lz`,
`frac << s` never exceeds 8 bits — **no bits are ever lost by the shift**;
you may rely on this invariant.

Note the asymmetry between the two paths:

- **Normalized path** (`lz <= exp`, so `s = lz`): the leading 1 lands exactly
  at bit 7 of `shifted` and is dropped by taking `shifted[6:0]` — this is the
  implicit leading one.
- **Subnormal path** (`lz > exp`, so `s = exp`): the shift stops early, the
  fraction's leading 1 remains somewhere below bit 7, and the retained
  leading zeros are part of the mantissa. The mantissa is *not* normalized.

## R4 — Adjusted exponent and subnormal flag

If `frac != 0`:

```
if lz <= exp:   nexp = exp - lz    ; sub = 0     // normalized (possibly at the floor)
else:           nexp = 0           ; sub = 1     // subnormal at the biased floor
```

The biased exponent never goes below 0: when normalization would push it
below the floor, the exponent saturates at 0 and `sub` is raised (sticky
subnormal-at-floor flag). `lz == exp` is the boundary: the result normalizes
**exactly at** the floor with `nexp = 0` and `sub = 0` — it is *not*
subnormal (see worked examples D and F).

Equivalently, in **all** non-zero cases `nexp = exp - s` with `s = min(lz, exp)`
(when `lz > exp` this yields `nexp = exp - exp = 0`), and `sub = 1` exactly
when `lz > exp`.

## R5 — Totality

All 4096 input combinations (`frac` in 0..255 × `exp` in 0..15) are defined
by R1–R4. There are no don't-cares: your equation set must produce the
specified value for every input vector.

---

## Worked examples

Bit strings are written most-significant-bit first; `0b` prefixes binary.
Each example lists the complete 13-bit output
(`mant6..mant0`, `nexp3..nexp0`, `zero`, `sub`).

### Example A — normalized with headroom

`frac = 0b00010110`, `exp = 5`.

`frac != 0`, `lz = 3` (three leading zeros), `lz <= exp` so `s = min(3,5) = 3`.
`shifted = 0b00010110 << 3 = 0b10110000`.

| output | value |
| --- | --- |
| mant | `0110000` |
| nexp | `0010` (= 5 − 3 = 2) |
| zero | 0 |
| sub  | 0 |

### Example B — subnormal (shift stopped at the floor)

`frac = 0b00010110`, `exp = 2`.

`lz = 3 > exp = 2`, so `s = min(3,2) = 2`; the shift is capped.
`shifted = 0b00010110 << 2 = 0b01011000`. The leading 1 stays below bit 7;
the retained leading zero is part of the mantissa.

| output | value |
| --- | --- |
| mant | `1011000` |
| nexp | `0000` (saturated at the floor) |
| zero | 0 |
| sub  | 1 |

### Example C — zero fraction overrides everything

`frac = 0b00000000`, `exp = 13`.

R1 applies regardless of `exp`:

| output | value |
| --- | --- |
| mant | `0000000` |
| nexp | `0000` |
| zero | 1 |
| sub  | 0 |

### Example D — normalized exactly at the floor

`frac = 0b10000001`, `exp = 0`.

`lz = 0`, `lz <= exp` (0 <= 0), so `s = min(0,0) = 0`.
`shifted = frac`; the leading 1 is already at bit 7 and is dropped.

| output | value |
| --- | --- |
| mant | `0000001` |
| nexp | `0000` |
| zero | 0 |
| sub  | 0 |

Even with `exp = 0`, a fraction already in normal position stays normalized.

### Example E — subnormal by one, at the floor

`frac = 0b01000000`, `exp = 0`.

`lz = 1 > exp = 0`, so `s = 0`, no shift, exponent saturates.

| output | value |
| --- | --- |
| mant | `1000000` |
| nexp | `0000` |
| zero | 0 |
| sub  | 1 |

Compare D and E: the same `exp = 0`, one position of leading zero apart,
opposite `sub`.

### Example F — the `lz == exp` boundary pair

`frac = 0b00000001` (`lz = 7`):

- `exp = 7`: `lz == exp` → normalized **exactly at the floor**. `s = 7`,
  `shifted = 0b10000000`, the leading 1 lands at bit 7 and is dropped.

  | output | value |
  | --- | --- |
  | mant | `0000000` |
  | nexp | `0000` |
  | zero | 0 |
  | sub  | 0 |

- `exp = 6`: `lz = 7 > 6` → subnormal by one. `s = 6`,
  `shifted = 0b01000000`; the leading 1 stays below bit 7.

  | output | value |
  | --- | --- |
  | mant | `1000000` |
  | nexp | `0000` |
  | zero | 0 |
  | sub  | 1 |

These two vectors differ in `mant` and `sub` only — the boundary is exactly
one shift position wide.

---

## Boundary self-check set

Before submitting, self-check your derivation against this published set of
input vectors (the rules above define the expected outputs for each):

1. **Exponent sweep:** every `frac` in 0..255 (all 256 values) with each
   `exp` in {0, 1, 2, 6, 7, 8, 15}.
2. **Shift-boundary band:** every (`frac`, `exp`) pair, `frac != 0`, whose
   leading-zero count satisfies `lz ∈ {exp−1, exp, exp+1}` (only `lz` in
   0..7 exists, so high `exp` contributes nothing).
3. **Corner fractions:** each `frac` in {0, 1, 63, 64, 127, 128, 129, 255}
   with each `exp` in {0, 7, 8, 15}.

This set covers every boundary class in the specification: the zero
short-circuit, the `lz == exp` floor boundary on both sides, the
subnormal-by-one band, the transition from a cap-active to a cap-inactive
exponent (exp = 7 vs 8), and the top exponent corner.


# Deliverable contract — `fpnorm_equations.json`

**Document:** `format_contract.md` · **Revision:** 1.0 · **Status:** normative

This document defines the exact deliverable format for the normalization
stage. The behavioral requirements are in `spec_fpnorm.md`; this document
says nothing about behavior. Any answer that violates this contract is rejected as
structurally invalid, before its equations are evaluated.

---

## 1. Top-level JSON object

The answer is a single, well-formed JSON object with **exactly two**
top-level keys:

```json
{
  "format": "fpnorm-equations-v1",
  "signals": {
    "<signal-name>": "<expression>",
    ...
  }
}
```

- `format` must be the string `"fpnorm-equations-v1"`. No other top-level
  keys are permitted; both keys are required.
- `signals` maps signal names to expression strings (see §3 for the
  expression grammar).
- **Duplicate keys are rejected.** A `signals` object that defines the same
  name twice (JSON objects with repeated member names) is a structural
  failure, even if a lenient parser would accept it.
- The answer must be valid JSON overall: no trailing commas, no comments, no
  unquoted keys, no NaN/Infinity.

## 2. Signal names

Names are **case-sensitive** throughout.

**Required outputs** — the `signals` map must define all 13, exactly named:

```
mant6 mant5 mant4 mant3 mant2 mant1 mant0
nexp3 nexp2 nexp1 nexp0
zero  sub
```

`mant6` is the mantissa MSB, `mant0` the LSB; `nexp3` is the adjusted
exponent MSB, `nexp0` the LSB. Missing any required output is a structural
failure.

**Inputs** — read-only, pre-defined, case-sensitive:

```
f7 f6 f5 f4 f3 f2 f1 f0
e3 e2 e1 e0
```

`f7`/`e3` are the MSBs. A signal definition that **redefines an input
name** (any of the twelve above appearing as a key in `signals`) is a
structural failure. Inputs are the only signals defined outside your file.

**Helper signals** — optional. A helper is any `signals` entry other than
the 13 required outputs. Rules:

- Helper names must match the regular expression `[a-z_][a-z0-9_]*`
  (lowercase only; the required output names above also satisfy it).
- A helper name must not collide with an input name or a required output
  name (redefining a required output twice is already excluded by the
  duplicate-key rule; defining a helper with a required output's name is
  that same collision).
- Helpers may reference inputs and other signals (including other helpers).
- **No cycles.** The definitions must form a directed acyclic graph. A
  signal (helper or output) whose value depends on itself, directly or
  through any chain, is a structural failure — the equation set must be
  purely combinational. This is enforced, so no latched or state-retaining
  construction can pass.
- Every referenced identifier must be defined (as an input, helper, or
  required output). An undefined reference — including any case-flipped
  name such as `F7`, `Mant6`, or `SUB`, which are *not* the same identifiers
  as `f7`, `mant6`, `sub` — is a structural failure.

## 3. Expression grammar

Expressions are Boolean formulas over the input/helper/output identifiers
and the constants `0` and `1`. The grammar (EBNF, case-sensitive) is:

```
or_expr  := xor_expr ('|' xor_expr)*
xor_expr := and_expr ('^' and_expr)*
and_expr := unary ('&' unary)*
unary    := ('~' | '!') unary | primary
primary  := IDENT | '0' | '1' | '(' or_expr ')'
```

- Operator precedence, highest to lowest: `~` `!`, then `&`, then `^`,
  then `|`.
- `~` and `!` are both logical negation and may be stacked (`!!x`, `~~x`).
- `&` `^` `|` are the usual bitwise-Boolean AND, XOR, OR on 0/1 values;
  they are left-associative.
- The constants `0` and `1` are the Boolean values false and true.
- Parentheses group. Whitespace (spaces, tabs) between tokens is ignored.
- **Nothing else is accepted.** No other operators, no word operators
  (`and`, `or`, `not`), no arithmetic (`+`, `*`), no comparison, no
  indexing, no functions. Anything outside the grammar above is a parse
  failure and a structural failure of the submission.

## 4. Size limits

- At most **256 signals** total in `signals` (helpers plus the 13 required
  outputs).
- At most **256 KB** total file size.

Exceeding either limit is a structural failure.

## 5. Compactness threshold T

The **literal-occurrence count** of a submission is the total number of
identifier and constant tokens (`IDENT`, `0`, `1`) appearing anywhere in
the expression strings of its `signals` map — every occurrence counts,
repeated references included; operators and parentheses count for nothing.

```
T = max(1500, 3 * L_ref)
```

where `L_ref` is the literal-occurrence count of the build-time primary
reference solution.

**Frozen value for this task: T = 1500.**
(Measured at build time: `L_ref` = 190, so `3 * L_ref` = 570 and the
formula floor of 1500 binds.) The frozen number 1500 is the binding
threshold; the formula above is its derivation of record.

- A functionally perfect answer with literal-occurrence count `<= T` meets
  the compactness requirement; one with a count `> T` does not meet it.
- Compactness never affects correctness credit: an enumerated ROM-style
  answer that is functionally exact is still functionally exact.

Return only the answer in the requested format.

---

## How it is graded

The verifier is private evaluator material and never reaches the solver.

| field | value |
| --- | --- |
| `verifier.kind` | `code_answer` |
| `verifier.output_path` | `candidate.txt` |
| inner kind / mode | `tasktrove` / `script` |
| entrypoint | `verifier/grade_answer.py candidate.txt` |
| script timeout | 120 s |
| runtime | container `docker.io/library/python@sha256:2f17fc04…06a9`, workspace `empty`, timeout 300 s |
| implementation revision | `b76d03131cd88bd9fc711dba206659027edba3a8` |

Three embedded resources, all role `verifier`:
`verifier/grade_answer.py` (TaskTrove wrapper),
`verifier/fpnorm_verifier.py` (the grader core, 848 lines),
`verifier/golden.py` (the private reference model, 177 lines).

### Stage 1 — structural gate (critical)

A structural failure scores **0** and the submission never reaches item
scoring.  Checks run in a documented precedence order:

`too_large` (>256 KiB) → `malformed_json` / `duplicate_json_key` →
`bad_schema` → `too_many_signals` (>256) → `missing_output` →
`input_collision` → `bad_signal_name` → `parse_error` →
`undefined_reference` → `cycle`.

The parser is a recursive-descent implementation of exactly the EBNF
published in `format_contract.md` — the public grammar and the enforced
grammar are the same object, which is what makes the contract checkable
rather than aspirational.  Acyclicity is enforced, so no latched or
state-retaining construction can pass; that is the capability's stated
exclusion, mechanised.

### Stage 2 — exhaustive equivalence

All 13 output signals are evaluated bit-parallel over **all 4096 input
vectors** and compared against `golden.py`.  Each output is a binary weighted
item:

| items | weight each | subtotal |
| --- | --- | --- |
| `mant6` … `mant0` | 1 | 7 |
| `nexp3` … `nexp0` | 1 | 4 |
| `zero` | 1 | 1 |
| `sub` | **3** | 3 |
| | | **15** |

`q = passed_weight / 15`.  The `sub` flag carries triple weight because it is
the one output that encodes the task's actual difficulty driver — the
`lz == exp` boundary — and a solver that gets everything else right while
missing the boundary should not look nearly correct.

### Stage 3 — compactness

`c ∈ {0, 1}`: the literal-occurrence count (every `IDENT` / `0` / `1` token in
the `signals` expressions) against `T = max(1500, 3 × L_ref)`.  Measured at
build time, `L_ref = 190`, so `3 × L_ref = 570` and the floor binds:
**T = 1500, frozen and published**.

### Aggregation

```
reward = 0                                    if the structural gate fails
reward = round(q² * (0.85 + 0.15 * c), 3)     otherwise
```

Squaring `q` is deliberate: partial correctness on an exhaustive equivalence
check is cheap to reach and should not pay linearly.  Compactness is worth
15% and **never** affects correctness credit — a functionally exact
enumerated ROM still scores 0.85, which the control battery confirms
empirically rather than by assertion.

### Infrastructure is not a score

An unexpected verifier exception returns category `internal_error`.  The
wrapper then writes **no** `reward.json` and exits `EX_SOFTWARE` (70); the
pinned TaskTrove script grader reports no reward and TaskCompendium stores
`infra_error` with a **null** reward.  An infrastructure failure is never
recorded as a semantic zero.

*This behaviour is the one change the quality reviewer required.  See
[`build.md`](build.md) § Repair round 1.*

### Calibration — the control battery, as executed

Every row below is a real measured reward from the run's own evidence, and
each was reproduced identically across 10 fixed-input regrades.

| control | kind | reward | what it pins down |
| --- | --- | --- | --- |
| `pc1-primary-reference` | positive | **1.000** | the reference solution is exactly correct and under T |
| `pc3-alternate-solution` | positive | **1.000** | a different decomposition scores identically — not overfit to one derivation |
| `nc-empty-response` | negative | 0.000 | structural gate |
| `nc1-invalid-json` | negative | 0.000 | structural gate |
| `nc-fenced-json` | negative | 0.000 | structural gate — code fences are a *failure*, not a convenience |
| `nc-input-redefinition` | negative | 0.000 | structural gate |
| `nc6-shift-off-by-one` | negative | 0.071 | `q = 4/15` — a wrong shift wrecks most of the mantissa |
| `nc7-zero-flag-omission` | negative | 0.538 | `q = 11/15` — dropping the zero override is a real but partial loss |
| `nc8-enumerated-rom` | negative | **0.850** | functionally exact, not compact: `q = 1`, `c = 0` |

`nc8` is the row that proves the aggregation formula behaves as documented:
0.85 is exactly `1² × (0.85 + 0.15 × 0)`.

---

## Files in the export

| file | bytes | contents |
| --- | --- | --- |
| `specification.json` | 71,165 | the TaskSpec, including the three embedded verifier resources |
| `instruction.md` | 14 K | the solver-facing prompt, reproduced above |
| `binding.json` | 42 | `{"environment":{"kind":"none"},"tools":[]}` |
| `manifest.json` | 1.4 K | Harbor lowering manifest, renderings, verifier runtimes |
| `renderings.json` | 134 | the single `plain` rendering |
| `task.toml` | 54 | `allow_internet = false` |

Verified: `export_matches_harbor_bytes: true`, `taskspec_valid: true`,
zero schema errors, item state `quality_accepted`.
