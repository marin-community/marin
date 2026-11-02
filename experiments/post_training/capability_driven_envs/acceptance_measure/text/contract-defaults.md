# The answer contract: state every choice, then grade exactly that

Every task you build states its answer contract in the solver-facing instruction, in plain words, and its grader
implements exactly that contract, nothing more lenient and nothing more strict. Decide each of these once, from
the proposal, and write the decision into the instruction (as a format requirement, never as grader behaviour):

1. **Channel.** The answer is either the final reply or one file at an exact path (for example
   `/workspace/answer.json`). Name one. The grader reads only that channel, with no fallback to the other.
2. **Framing** (a JSON answer). `EXACT`: the reply is one JSON value and nothing else. `FENCED`: one ```json fence
   or the bare value. `LAST`: prose may come first and the reply ends with one JSON object. Say which.
   A plain-text answer states its exact lines and literals instead.
3. **Surrounding prose.** Allowed or not, stated, and consistent with the framing.
4. **Transport noise.** Default: a leading byte-order mark, CRLF line endings and trailing whitespace are
   stripped before parsing, and the instruction need not mention them. If the task rejects them instead, the
   instruction says so.
5. **Numbers.** The precision required (significant figures or decimal places) and that more digits are
   allowed; whether an integer field accepts `6.0`; that a number given as a string is wrong.
6. **Labels.** Exactly as listed, or the explicit aliases the task grants. Never fold case or punctuation
   implicitly.
7. **Schema.** Closed by default: no keys or list entries beyond those named. State it.
8. **Empty answer.** An empty or missing answer earns 0 (a submission failure), never partial credit.

`grade_kit.py` (reference/helpers/) implements each default strictly: `parse_json(text, EXACT|FENCED|LAST)`,
`as_object`, `as_list`, `as_label`, `as_alias`, `as_int`, `as_number`, `close_to(v, truth, sig=...)`, `Checks`,
`run`. Ship it as the private grader file `grade_kit.py` next to `grade.py`. The grader may be written in the
grader step directly (`draft.python_grader(script, config, ..., files=(draft.file("grade_kit.py", grade_kit),))`,
with `script` and `grade_kit` filled by `embed_file`), or by the step's model call, in which case the step's
request states these contract choices and the kit's API, and the step adds the kit to the private files.

Before you submit, call `run_probe_battery` with the contract you stated (the same fields, as arguments). It
grades a fixed battery of near-miss and equivalent candidates, generated from your reference answer and your
contract, against the last draft you built. `submit` refuses a program whose last build has not passed the
battery. A battery failure costs no revision: fix the grader (or the contract you stated) and run it again.
