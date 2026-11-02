Close each finding above by its class, not by its instance. A witness is one example of a class of defect; a
repair that refuses only the quoted answer leaves the class open, and the next search finds another instance.
For every finding: name its class from this table, fix that class at every field and level of the answer the
grader reads, then build and run `run_probe_battery` (it replays these witnesses too) before you submit.

| class | the kit call that closes it (reference/helpers/grade_kit.py) |
|---|---|
| label normalisation | `as_label(v, allowed)`; synonyms only through an explicit `as_alias` map |
| tolerance looser than the stated precision | `close_to(v, truth, sig=N)` or `places=`; an expected 0 needs `abs_` |
| open schema | `as_object(v, required, optional)` at every level; `as_list(v, length=...)` |
| type coercion | `as_number`, `as_int`, `as_bool`, `as_rational`; never `float(str)` |
| text around the JSON, several answer objects | `parse_json(text, EXACT/FENCED/LAST)`; never slice braces |
| list handling (sets, subsets, first match only) | `same_items(got, expected, ordered=...)`, `unique_by`, `within_universe` |
| derived fields not recomputed | recompute each derived field from the answer's own primaries and check both relations |
| answer channel fallback | one channel: `read_reply()` or `read_answer_file(path)`, no fallback |
| duplicate JSON keys | `parse_json` refuses repeated keys at every depth |
| equivalent notation refused (false reject) | a parser for every form the contract allows (`as_rational`, an alias map) |
| one canonical layout hard-coded (false reject) | check the stated property; every cap the grader enforces appears in the instruction |
| wrong channel or path (false reject) | the path the contract names, read once |
