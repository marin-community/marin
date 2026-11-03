# Strict grading: decide each choice once, state it in the contract, implement exactly that

Evidence: the held-out evaluation of the campaign's 467 accepted tasks recorded 1,551 weakness clusters
(1,173 false accept, 378 false reject). Three independent readings of all of them sorted them into the classes
below, and the counts are theirs (approximate, ±10%; one cluster can fall in two classes). In the grader
scripts themselves:
- a duplicate-key hook appears in 85-89% of the clean graders and 31% of the witnessed ones;
- case folding (`lower`/`casefold`) appears in 0-4% of the clean graders and 61% of the witnessed ones;
- regex extraction appears in 11-17% of the clean graders and 39% of the witnessed ones.
The gaps hold within single seeds. For example, case folding appears in 7% of clean versus 88% of witnessed
graders for d40, and 7% versus 79% for d41.

## The rule

Every leniency is a choice: whether a fence is allowed around the JSON, whether `6.0` counts as an integer,
whether "Counterflow" means "counterflow", how close a number must be. The same task failed in both directions
when the choice was left implicit. It folded case in one field (a false accept) and refused an equivalent form
in another (a false reject), because neither choice was written down. So:

1. Decide each choice once, from the proposal.
2. Write it in the answer contract in plain words: "one JSON object and nothing else", "labels exactly as
   listed", "at least four significant figures", "a JSON integer".
3. Implement exactly that, and nothing more lenient or more strict. `grade_kit.py` makes the strict version
   the default and every leniency a named argument, so the code shows the choice.

## The nine false-accept classes and the kit call that closes each

| class | clusters | what went wrong | do this (`grade_kit`) |
|---|---:|---|---|
| label normalisation | ~187 | `strip`, `lower`, punctuation removal, Unicode folding before comparing, so `COUNTERFLOW` and ` s3 ` pass | `as_label(v, allowed)`: exact membership. Synonyms the contract grants go in an explicit `as_alias(v, {spelling: canonical})` map. Never fold implicitly |
| tolerance looser than the stated precision | ~165 | a fixed ±0.5 kW or ±4 min band wider than "four significant figures" or than the gap to the nearest wrong answer | `close_to(v, truth, sig=4)` (or `places=`, `rel=`, `abs_=`): the tolerance comes from the truth and the stated precision (`sig=4`: half a unit in the truth's fourth significant digit; `places=2`: 0.005), never from the digits the answer wrote. An expected 0 needs an explicit `abs_` |
| open schema | ~163 | extra keys, extra list entries and extra sections ignored, so an answer can carry alternatives | `as_object(v, required, optional)` at every level, and `as_list(v, length=...)` |
| type coercion | ~166 | `"12.3"`, `"yes"`, `1`, unit-suffixed strings and integral floats accepted where the contract says a number, a boolean or an integer | `as_number`, `as_bool`, `as_int` (bool is never a number; `integral_float=True` only when the contract says "a JSON number"). Formats: `as_hhmm`, `as_iso_date`, `as_rational` (ASCII digits only) |
| text around the JSON | ~169 | fences, prose and first-`{`-to-last-`}` slicing tolerated when the contract said "one JSON object and nothing else"; first or last object graded when several were given | `parse_json(text, framing)`, where framing is `EXACT`, `FENCED` or `LAST`. `LAST` refuses a reply with two answer objects. Never slice braces or strip fences by hand |
| list handling | ~139 | `set()` hides duplicates, subset or superset passes "exactly these", only the first match per key is checked | `same_items(got, expected, ordered=...)` (a multiset), `unique_by(items, key)` (no silent overwrite), `within_universe(items, universe)`. Validate every entry, not the first |
| derived fields not recomputed | ~70 | a residual, total or argmax checked only against the truth, so a self-contradictory answer passes | recompute each derived field from the answer's own primaries and check both relations: it equals f(submitted inputs), and the inputs match the truth |
| answer channel | ~50 (fa ~12, fr ~38) | falling back from the required file to the reply (fa), or reading only a file while the contract asked for the reply (fr), or reading an absolute path when the contract gave a relative one | one channel, named in the contract: `read_reply()` or `read_answer_file("/workspace/answer.json")`. No fallback |
| duplicate JSON keys | ~26 | `json.loads` keeps the last value, so a wrong value then a right one passes | `parse_json` refuses repeated keys at every depth |

Also from the readings:
- **Free-text fields never validated (~57).** The grader checks only that the field is non-empty. Either drop
  the field from a code grader's contract, or derive what it must say from computed facts (the reason must
  name the computed cause).
- **Only aggregates checked (~92).** Mostly ledgers, checked by per-account nets alone. Validate every row,
  then the totals.
- **Prompt or config leaks the truth (~8).**

## The false-reject side

| class | clusters | rule |
|---|---:|---|
| equivalent notation refused (fractions, a leading `+`, units with SI prefixes, key order) | ~76 | every form the contract allows gets a parser (`as_rational`, an alias map), and the contract allows nothing the grader cannot parse |
| one canonical decomposition or layout hard-coded | ~75 | check the stated property (balanced, nets tie, role covered); every cap or minimum the grader enforces must appear in the instruction |
| tolerance tighter than the permitted rounding | ~20 | the same `close_to` rule: extra written digits never tighten it, so the full float `json.dump` writes and trailing zeros pass. A consistency check gets an `abs_` window propagated from its inputs' stated precision, never one scaled from their written digits (`10 ** -decimals(v)` shrinks below float rounding error on a 15-digit answer). Check written digits (`decimals`, `sig_figs`) only when the contract requires them shown, as their own check |
| rejected for the wrong channel or path | ~38 | as above: one channel, the path the contract names |

## Rewards and the ceiling

`rejection_ceiling` is the most a substantively wrong answer may earn. Validation's negative controls demand it.

- **Clean tasks.** 44% of the clean script graders are all-or-nothing (ceiling 0), against 29% of the
  witnessed ones.
- **Weighted partial credit.** A wrong answer that misses one light check can earn far more than the ceiling.
  One witnessed film task gave 0.9 to an answer with a single mislabel, against a ceiling of 0.75.
- **What a qualified grader did.** It returned `1.0 if every check passed else 0.49 * share` for a ceiling of 0.5.
  `Checks.reward(partial_cap=...)` does this. Keep `partial_cap` at or below the ceiling, and no failing answer
  can exceed it. `Checks.worst_single_miss()` gives the plain weighted share of the best one-miss answer, to
  compare with the ceiling.

## A grader in this shape

```python
# grade.py; tests/grade_kit.py ships next to it as a private grader file
from grade_kit import (EXACT, Checks, as_int, as_label, as_list, as_number, as_object, close_to,
                       parse_json, read_private_json, read_reply, run, same_items)

def grade():
    truth = read_private_json("config.json")
    answer = as_object(parse_json(read_reply(), EXACT), required=("mode", "duty_kw", "units"))
    checks = Checks()
    checks.check("mode", 1, lambda: as_label(answer["mode"], {"counterflow", "parallel"}) == truth["mode"])
    checks.check("duty_kw", 1, lambda: close_to(as_number(answer["duty_kw"]), truth["duty_kw"], sig=4))
    checks.check("units", 1, lambda: same_items(as_list(answer["units"]), truth["units"], ordered=False))
    return checks.reward(partial_cap=0.0), checks

run(grade)
```

- A contract violation inside a check (`GradeError`) fails that check, with its reason in the verdict.
- A violation outside every check (the reply does not parse) is a verdict of 0 with the reason.
- Any other exception is a verdict of 0 with a `grader_error` check carrying the traceback. Treat it as a bug
  to fix, not a score.

Ship the kit with the grader:
- `draft.python_grader(script, config, ..., files=(draft.file("grade_kit.py", grade_kit),))`;
- for a `GraderDraft`, its `private_files`.

The text comes from the embedded copy: `embed_file` (see `probe-battery.md`).
