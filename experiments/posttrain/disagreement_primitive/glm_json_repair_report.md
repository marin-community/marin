# GLM-5.1 phase_4 JSON repair pass — design and offline test results

## Context

GLM-5.1 (zai-org/GLM-5.1, Together-hosted) is one of three judges in the
spec-alignment phase_4 / `rubric_plus_spec` condition. Of 2,758 expected
phase_4 calls, only 2,443 returned content that `json.loads` accepts —
**315 rows (≈11.4%) were lost to JSON parse failures** even though the
request set `response_format={"type": "json_object"}`. Neither GPT-5.1
nor Gemini-3-Flash exhibits this rate.

Coverage matters because the per-statement κ-by-condition table
(`per_statement_kappa_by_condition.jsonl`) treats GLM phase_4 as a
required signal in the dual-condition (var_A + phase_4) repair-loop
diagnostic. Missing 11% of GLM phase_4 records biases per-statement
counts and weakens the upper-tail κ comparisons that gate operator
dispatch in the spec repair loop.

## Failure-pattern distribution

Counted across the 79 documented failures in
`claude_subagents/lm_judge_rubric_plus_spec/glm.md` (the high-disagreement-tuple
slice; the 315 full-corpus failures are inferred to follow the same
distribution because the high-disagreement slice is sampled uniformly):

| pattern | count | share |
|---|--:|--:|
| `Expecting ':' delimiter` at line 3, col N | 27 | 34% |
| `Expecting ',' delimiter` at line 3, col N | 22 | 28% |
| `Expecting value: line 1 column 1` | 15 | 19% |
| `Unterminated string starting at` | 12 | 15% |
| `Expecting property name enclosed in double quotes` | 3 | 4% |

Column position for line-3 errors: min=45, max=15,759, mean=1,365,
median=367. The long upper tail (4 cases with col > 4,000) is consistent
with truncation at `max_tokens=4000`.

## Repair function design

`experiments/posttrain/disagreement_primitive/e9_glm_json_repair.py`
exposes `repair_glm_json(raw_text) -> RepairResult`. The result carries
`data` (parsed dict or None), `strategy` (one of: `valid`,
`smart_quote_keys`, `escape_unescaped_quote_at_error`, `truncated_close`,
`empty_body`, `unrepairable`), and `partial` (True only when the close
strategy fired). Strategies run in order; the first hit wins.

### Strategy 0 — `valid` (the no-op for all other judges/conditions)

```
> json.loads(raw_text) succeeds AND result is dict
```

Returns immediately. **This is the load-bearing test for the
"don't change behavior on valid JSON" requirement.** A 200-row
spot-check on real `grounding/per_judgment.jsonl` records confirmed
no false positives across condition / judge.

### Strategy 1 — `smart_quote_keys`

```
before: …, “example_refs”: [], …
after:  …, "example_refs": [], …
```

Two regexes target curly double-quotes adjacent to JSON structural
punctuation. A curly-quote inside a string VALUE (e.g., `"reasoning":
"the user asked “why”"`) is left alone — the repair is conservative
and only normalizes quotes that look like keys. Targets the rare 3
"property name enclosed in double quotes" failures.

### Strategy 2 — `escape_unescaped_quote_at_error`

```
before: "reasoning": "the user said "this is fine" but..."
after:  "reasoning": "the user said \"this is fine\" but..."
```

Reads `JSONDecodeError.pos` and walks left looking for unescaped `"`
characters. For each candidate it inserts a `\` and either:
1. accepts and continues if the parse position advances (more pairs to
   escape); or
2. accepts and returns if the full text now parses.

The loop runs at most 8 passes × 5 candidates per pass — bounded so a
pathological input cannot wedge the repair. The strategy refuses to
fire on non-delimiter error messages (e.g., `Extra data`) so unrelated
parse failures fall through to `unrepairable` instead of being
"helpfully" mis-interpreted. Targets the 27 + 22 = 49 delimiter cases
(62% of failures).

### Strategy 3 — `truncated_close`

```
before: {\n  "score": 4,\n  "reasoning": "this is cut
after:  {\n  "score": 4,\n  "reasoning": "this is cut"}
```

Walks the input tracking depth of `{`/`[` while respecting strings,
appends `"` if mid-string, then appends balanced closers in reverse
stack order. Sets `partial=True` so downstream consumers can treat
these records differently if the missing fields matter. Targets the
12 `Unterminated string` cases plus the long-tail late-column delimiter
cases (estimated 12 + ~5 = 17 cases, 22% of failures).

### Strategy 4 — `empty_body`

```
before: ""    after: data=None, strategy="empty_body"
```

Returns `None` cleanly. The 15 `Expecting value: line 1 column 1`
failures are GLM finishing with `finish_reason=length` after consuming
the entire token budget on hidden chain-of-thought before emitting
any visible JSON — there is genuinely nothing to repair. Targets the
15 empty-body cases (19% of failures).

## Estimated coverage gain

If each strategy works for the documented pattern with ≥ 95% reliability
(plausible for the targeted patterns; the unescaped-quote fix is the
weakest, since it can only handle quote pairs not other escape errors):

| strategy | targets | estimated success | recovered |
|---|--:|--:|--:|
| `escape_unescaped_quote_at_error` | 49 | 90% | 44 |
| `truncated_close` | 17 (12 unterminated + 5 long-col) | 95% | 16 |
| `smart_quote_keys` | 3 | 100% | 3 |
| `empty_body` (no recovery) | 15 | 0% | 0 |
| **Total recovered** | | | **63 of 79** (~80%) |

Extrapolated to the full corpus of 315 failures, this is ≈ 250 recovered
records (around 9% of the 2,758-row condition baseline) and ≈ 65 still
unrecoverable (the empty-body class is irreducible without re-running
GLM with a larger token budget).

## Test pass rates

`experiments/posttrain/disagreement_primitive/test_glm_json_repair.py`:
**18 of 18 tests pass** (run with `.venv/bin/python -m pytest`).

| family | tests | result |
|---|--:|---|
| Regression (valid JSON unchanged) | 3 | pass |
| Repair on documented failure patterns | 4 | pass |
| Negative tests (empty / garbled / wrong shape) | 4 | pass |
| Ambiguity / non-delimiter rejection | 3 | pass |
| Wrapper enable/disable / partial-marker | 3 | pass |
| Coverage meta-check | 1 | pass |

The regression family runs against 60 randomly-sampled valid `phase_4`
records from `grounding/per_judgment.jsonl`. A separate ad-hoc spot
check on 200 records (mixed conditions and judges) returned strategy
`valid` for 200/200 — no false positives.

Pre-commit (`./infra/pre-commit.py --fix`) passes clean on all three
new files.

## Limitations

1. **Empty-body cases (15 of 79, ~19%) are not repairable.** They
   correspond to `finish_reason=length` with all tokens consumed by GLM's
   hidden reasoning. The structural fix is to re-run those calls with a
   higher `max_tokens` (already bumped 1500 → 4000 once; may need 8000+).
2. **Single-pass quote-escape** can fail when the unescaped quotes are
   not adjacent to the parse position — the algorithm walks back at most
   5 candidates, which covers the common "open + close" pattern but not
   pathological inputs with > 5 quote pairs distributed across the
   string. Estimated < 5% of the 49 delimiter cases.
3. **Truncated-close repairs return partial dicts.** A repaired record
   may be missing `spec_quotes`, `rubric_quotes`, `tension_description`,
   etc. Callers that need these fields must check `_partial_parse` and
   either drop or treat the record specially. Out-of-the-box, the κ
   pipeline only needs `score`, which is the first emitted field, so
   most truncated repairs will recover the score successfully.
4. **No schema enforcement.** The repair only ensures parseability; type
   checks (e.g., `score` is an int 1-5) remain the caller's job. This is
   intentional — the same repair must work for any future schema change.
5. **No coverage of the `decision`-shape phase_3 schema.** Phase 3
   judgments have a different shape (`decision: compliant|...`); this
   repair was designed against phase 4 only, though the strategies are
   schema-agnostic and would likely work there too — just not tested.

## What's needed to actually run the retry

The retry script
`experiments/posttrain/disagreement_primitive/e9_repair_glm_phase4.py`
expects EITHER:

- `--raw-dir <path>` pointing at a `RawAPILogger` `judge_*` directory
  containing per-call `.json` SDK dumps (the `e8_paired_indirection.py`
  / `e8_phase2_cross_model.py` runtime layout), OR
- `--raw-jsonl <path>` to a flat jsonl where each row carries `raw_text`
  / `content` / a nested `choices[0].message.content`.

**Inputs not currently on disk in this worktree.** The phase_4 raw
responses (`results/raw/e8_phase2_glm/<ts>/judge_*` or wherever phase 4
landed) were not included in the bundle. Per the task brief, the per-
candidate `repair_v0/round_*/<sid>/<cand>/phase4_judgments.jsonl` files
exist but use the e9 verifier code path — they only have parsed records,
not the raw GLM strings.

**To execute the retry**:

1. Locate the original raw GLM phase_4 SDK dump directory (could be on
   another machine, GCS bucket, or cold storage). Check the project
   logbook for the run timestamp; the path will look like
   `results/raw/e8_phase4_glm/<UTC-ts>/judge_phase4_glm/`.
2. Run:
   ```bash
   .venv/bin/python experiments/posttrain/disagreement_primitive/e9_repair_glm_phase4.py \
       --raw-dir <path-to-raw-dir> \
       --out-dir experiments/posttrain/disagreement_primitive/phase4_glm_repaired/
   ```
3. Inspect `repair_summary.json` — confirm valid + repaired ≥ 90% of
   input rows; drop the `repaired_judgments.jsonl` next to the existing
   `phase4_glm/judgments.jsonl` and re-run `e9_kappa_diagnostic.py` to
   see whether per-statement κ moves.

## Wiring (conservative)

- `e8_paired_indirection.py:parse_json` is **unchanged**.
- `e8_phase2_cross_model.py:call_glm_json` is **unchanged**.
- The repair is **opt-in** via two paths:
  1. Direct call: `from e9_glm_json_repair import repair_glm_json`.
  2. Drop-in wrapper: `parse_json_with_glm_repair(raw_text, enabled=True)`
     mirrors `parse_json` semantics (strips ``` fences) and adds repair
     when enabled. The env var `MARIN_E9_GLM_REPAIR=1` is the implicit
     enabler for any future caller that follows the convention.

The retry script is the primary intended caller. It does not modify any
production data — it writes to a new directory.

## Next-step suggestion

Locate the raw GLM phase_4 SDK dumps (the bundle missed them; they live
either on the original run host or in `results/raw/e8_phase4_glm/...`
on whichever machine ran phase 4). Run `e9_repair_glm_phase4.py` against
that directory, then re-run `e9_kappa_diagnostic.py` and confirm
per-statement κ_phase_4 sample sizes are restored to within ~3% of GPT
and Gemini coverage.

## Files

- `experiments/posttrain/disagreement_primitive/e9_glm_json_repair.py` — the repair function
- `experiments/posttrain/disagreement_primitive/test_glm_json_repair.py` — 18-test suite
- `experiments/posttrain/disagreement_primitive/e9_repair_glm_phase4.py` — offline retry CLI
- `experiments/posttrain/disagreement_primitive/glm_json_repair_report.md` — this report
