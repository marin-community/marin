# verifyit

`verifyit` executes the grader contract in a converted TaskTrove task. The task supplies a
flat `tests/verifier.toml`; `tests/test.sh` contains this shim:

```sh
exec verifyit /tests/verifier.toml
```

The command writes `/logs/verifier/verdict.json`:

```json
{"reward": 1.0, "status": "scored", "detail": {"extracted": "C"}}
```

Statuses are `scored`, `invalid_task`, and `infra_error`. A scored result also writes Harbor's
`reward.json` and `reward.txt`. Invalid tasks and infrastructure failures omit those reward files,
so the trial can be masked instead of recorded as a zero. Candidate output never causes a nonzero
process exit after a verdict has been written.

## Modes

| mode | contract |
|---|---|
| `unsolvable` | private classification: normal answers score zero; a validated terminal ejection scores one |
| `predicted_action` | unordered function calls with exact JSON types and optional float tolerance |
| `mcq` | expected option letter |
| `math` | expression equality through math-verify |
| `numeric` | numeric equality with explicit tolerances |
| `exact` | normalized string equality |
| `structure_closure` | continuation of a pinned prefix matching the first completed outer reference region |
| `json-schema` | JSON, YAML, or TOML checked against JSON Schema |
| `xml-elements` | required XML elements and attributes |
| `csv-columns` | required CSV header columns |
| `ifeval` | deterministic instruction-following constraints |
| `reasoning-gym` | the named reasoning-gym scorer and entry |
| `stdio` | program stdout over hidden cases |
| `pytest` | pytest JSON report with required and protected tests |
| `junit` | JUnit XML report |
| `gotest` | `go test -json` events |
| `judge` | reference, checklist, labels or pairwise rubric through a configured model endpoint |
| `script` | legacy `test.sh` fallback with normalized reward files and fail-closed errors |

`structure_closure` preserves trailing-output diagnostics separately from its first-region reward.
Callers with generation stop metadata remove configured terminal markers before candidate grading.

The `pytest` mode scores startup and collection errors attributed to editable candidate code as failed attempts.
`PytestSpec.candidate_modules` names modules the task asks the candidate to implement.
Missing imports of those modules score zero, including an empty submission.
Only the modules the spec names in ``candidate_modules`` (and their submodules) count as
candidate code; imports are never guessed from distribution metadata.
Missing modules with an installed provider remain invalid tasks.
Workspace-qualified candidate packages add the workspace parent to Python's import path.
Missing dependencies imported by implementation code, errors in restored task files, and genuine interruptions remain unscored.
`protected_paths_files` lists tests-directory manifests of trusted workspace paths restored by setup.
Every batch is checked before returning a candidate startup or collection failure.

For the `math` and `numeric` grading modes, the last `\boxed{...}` occurrence determines the
candidate when the output contains a box marker. Its braces must be balanced and its content must be
nonempty. Otherwise, the candidate receives reward `0.0`, even when an earlier marker contains the
expected answer. Without a box marker, both modes read the last nonempty line. Numeric mode
extracts exactly one integer, decimal, scientific-notation value or integer fraction from that line.
Surrounding prose is ignored, including negation: `Definitely not 42` extracts `42`.
When a box is present, its entire content must be a numeric literal, optionally wrapped in math delimiters.
Thousands separators require groups of three digits. Multiple literals such as `12 or 13` or
`2 + 2`, malformed numbers and nonfinite values are malformed submissions and receive reward `0.0`.

For a single multiple-choice reference such as `\textbf{(D)}\ 7`, the default grader accepts
the correct letter or an equivalent value. A reply that gives both must match both.
When symbolic parsing yields no expression, a partial expression, or an undefined value,
the grader compares the choice value as text after removing outer math delimiters and a
trailing period.

Other default math references, including lists of choice labels such as `(A) and (C)`,
must parse as one whole LaTeX expression. Each member of a LIST reference must parse separately.
TaskCompendium rejects partial parses and undefined expressions during conversion as
`invalid_reference`.

Scalar `MathSpec` contracts may set `positive_symbols = ["a_1", "a_2"]` when the public problem
explicitly states those symbols are positive. If ordinary comparison fails, the grader checks
whether the parsed scalar expressions simplify to exactly the same value under those assumptions.
The default is an empty list, so unrestricted variables keep their usual comparison.

Numeric `expected` is a required literal string; `tolerance_abs` and `tolerance_rel`
are required finite nonnegative floats. For example, `expected = "1/2"`, `tolerance_abs = 0.0`,
`tolerance_rel = 0.0` accepts both `1/2` and `0.5` without losing integer or decimal precision.
The effective tolerance is `max(tolerance_abs, tolerance_rel * abs(expected))`. Float tolerances
are converted to exact rational values through their decimal spelling before comparison;
`0.01` permits an exact difference of `1/100`. Native float expected values are rejected
because rounding may already have changed their meaning. `grade_numeric_candidate_float`
compares already parsed floats directly with an explicit absolute tolerance; MMMU uses
`0.0` and JEEBench uses `0.01` to retain their source scoring rules.
Literal components and expanded decimal powers are limited to 4096 digits before parsing.

[`spec.py`](src/verifyit/spec.py) owns the frozen mode dataclasses plus `parse_spec` and
`render_spec`. Spec paths are relative to the directory containing `verifier.toml`. `grade.py`
owns dispatch, output handling, verdict writing, and the CLI. Executable graders live in
`modes/grade_*.py`; `file_ops/` owns bounded reads and restoration, and `execution/` owns
command execution and worker deadlines.

Answer specs expose `empty_output = "zero"` by default. Explicit `"grade"` passes present empty
text to the mode's contract; missing files still score zero. Rewards must be finite numbers
in `[0, 1]`. Malformed verdicts, incomplete judge replies, and failed structured script producers
become unscored infrastructure errors. Interrupted test runs cannot retain positive credit.

Candidate-attributed pytest collection errors score zero. When pytest fails before producing a report, VerifyIT
checks the interpreter and JSON-report plugin from a temporary directory isolated from task
imports. A healthy baseline makes the failed task run a scored startup error; an unavailable
baseline remains an infrastructure error. The task's original import paths and tests still
run unchanged, so a candidate named after a standard-library module is graded as submitted.

VerifyIT itself requires Python 3.10 or newer. A `pytest` task may use a Python 3.9 or
newer subprocess with pytest and pytest-json-report installed in that interpreter.
Startup errors retain their available traceback and module provenance: editable
candidate failures score zero, while failures in protected code or with unknown
provenance remain unscored.

The mode modules expose direct candidate graders for callers holding extracted values.
`aggregate_rewards` combines required components with ALL, MEAN, MAX, MIN, or PRODUCT;
invalid tasks and infrastructure errors discard partial credit. The judge and Reasoning Gym modes
also expose direct candidate APIs for decoded context and trusted entries. Judge connections carry
runtime credentials separately from serializable specs. Reasoning Gym's optional `params` file
configures its scorer; callers own isolation when invoking its direct API.

`adapters/` prepares framework observations for the shared modes. Frameworks retain task
execution, dispatch, and dependency pins; installing this package does not enable an adapter.
`preparation/` retains raw inputs and named normalization policies. Preparation failures carry
Verifyit's own attribution category, which determines whether an incomplete attempt can be scored.

`json_comparison.json_values_equal` compares decoded JSON values with strict types and an optional
float tolerance.

`structure_closure` classifies partial JSON, delimited JSON, paired markers, and required code
fences. `closure_scoring` compares a completed outer region with a known reference and reports
syntax, content preservation, and trailing output separately. `tool_call_markup` classifies
completed `<tool_call>` markup in decoded assistant text. `reasoning_markup` classifies the
reasoning span of a decoded reply; the caller states whether the prompt opened a span that a
leading closer may close. The `reasoning_markup` grading mode scores one when the reply holds at
most one closed span and no stray closer, taking the prompt to open no span and excusing a span
left open by a length stop; its verdict names the violation otherwise.
These helpers inspect text; the caller retains the sampled tokens, supplies any generation stop
markers, and decides when a completed attempt can be scored.

`candidate_spec(mode, parameters)` builds and validates the specification of an in-process mode
from a JSON table. `grade_candidate(spec, candidate, resources)` scores an answer the caller
already extracted, with no filesystem or harness operations. `IN_PROCESS_MODES` lists the modes it
accepts: `unsolvable`, `exact`, `numeric`, `mcq`, `math`, `ifeval`, `json-schema`, `xml-elements`, `csv-columns`,
`structured_exact`, `structure_closure`, `reasoning_markup` and `predicted_action`. Text modes take a string, `structured_exact` takes a
decoded JSON value, `predicted_action` takes a tuple of `FunctionCall`, and `reasoning_markup`
takes a `ReasoningReply` of the reply text and its stop reason. A file the specification
names, such as the `json-schema` schema, comes from `resources`, keyed by its path relative to the
tests directory. A specification the mode rejects yields an `invalid_task` reward; a candidate of
the wrong type raises `TypeError`. Judge, Reasoning Gym, execution and script modes run only
through the file grader. Predicted-action matching preserves duplicate calls and requires all calls
to match one to one. Argument objects stay decoded in JSON tables; `render_spec` encodes each
argument object as a JSON string in TOML so nested JSON null values survive `parse_spec`.

`grade_ejection_candidate(spec)` scores a validated terminal ejection using the private
classification: `unsolvable` earns one and every other mode earns zero. Callers validate the
terminal action and its provenance before invoking this function. Refusal text and answer strings
never count as ejections. In-process task validation retains invalid-task and infrastructure-error
verdicts. File grading of `unsolvable` always scores zero because a file cannot establish an ejection.

### Judge

A `judge` spec prepares the answer, applies its deterministic gates, and then asks an
OpenAI-compatible endpoint for a verdict.

`candidate_steps` rewrite the answer, in order, before the IFEval `constraints`, the exact gates
and the prompt. They apply to every rubric:

- `{kind = "strip_reasoning_blocks"}` removes reasoning blocks, any text before the last closing
  delimiter, outer whitespace and a final `<|eot_id|>`. An unclosed block leaves an empty answer.
- `{kind = "strip_outer_whitespace"}` removes outer whitespace.
- `{kind = "extract_group", pattern = "..."}` replaces the answer with the first group of the first
  `re.search` match. Without a match, the answer is unchanged.

The rubric decides how the prepared answer is scored:

- `reference`: an answer equal to a reference after normalization (of its `\boxed{}` content, when
  present) scores 1 without a model call, unless `exact_gate = false`. Otherwise the judge compares
  the answer with `references` and replies `SCORE: 0`, `0.5` or `1`.
- `checklist`: the judge answers one yes/no question per entry of `criteria`, with the `context`
  file shown alongside the answer. The score is the fraction answered yes.
- `labels`: an answer equal to one of `exact_gate_answers` scores as `exact_gate_label` without a
  model call. Otherwise `system_prompt` and `prompt_template` are filled with `{question}`,
  `{reference}` (the single entry of `references`) and `{candidate}`, and the score is the
  `label_scores` entry for the label the judge replies with.
- `answers` requests one JSON verdict per reference in a single reply. It scores 1 when every
  reference is matched and reports the fraction matched as `detail.partial_credit`.
- `pairwise`: ports Arena-Hard's comparison with the single baseline answer in `references`. The
  judge receives two requests, with the baseline as assistant A and then as assistant B;
  `prompt_template` places the answers with `{answer_a}` and `{answer_b}`. Each reply's verdict is
  the last match of the first `verdict_patterns` entry that matches. `label_scores` gives assistant
  A's score for each verdict and `label_weights` counts a verdict as several games. The reward is
  the candidate's weighted mean score over both games. `detail.metrics.win_rate` repeats it and
  `detail.metric_weights.win_rate` holds the game count, so a weighted mean over rows equals
  Arena-Hard's mean over games.

`label_scan` sets how the `labels` rubric reads the label from the reply:

- `literal`: the last line is the label, and no other declared label appears anywhere in the reply.
- `lines`: the last line is the label, and no other declared label fills a whole line.
- `whole`: the whole reply is the label.
- `json`: the reply is a JSON object, optionally in a Markdown code fence, and its `label_field`
  member is the label. A boolean member is read as the label `true` or `false`.
- `pattern`: the last match of the first `verdict_patterns` entry that matches is the label, so the
  judge may name other labels while explaining its verdict.

`label_case = "upper"` matches labels and pairwise verdicts case-insensitively.
`strip_reasoning_blocks` removes reasoning blocks from the judge's reply before the label or verdict
is read. A `labels` reply that matches no label or more than one, and a `pairwise` reply without a
declared verdict, are scored `infra_error`.

The request options are:

- `system_role` sends `system_prompt` as a `system` (default) or `developer` message.
- `response_format` is sent with each `labels` request over Chat Completions.
- `sampling` sets the judge request's temperature. It defaults to `{ temperature = 0.0 }`.
  An empty table uses the endpoint defaults, as the `answers` rubric always does.
- `api = "responses"` sends the `labels` rubric through the Responses API.
- `reasoning_effort`, when set, is sent with each request.

`max_completion_tokens` sets the initial request budget of every rubric (default `8192`).
A positive `incomplete_retry_tokens` must exceed it and permits one larger
request when a reply ends with `finish_reason="length"`. Reference verdicts record `attempt_count`
and `attempts` in `detail`; checklist verdicts record them under each entry in `detail.criteria`.
Each attempt contains `finish_reason` and `completion_tokens` (`null` when the endpoint omits
usage). An exhausted retry remains `infra_error`, with the available attempt diagnostics in
`detail`.

For each model reply, the judge allows up to three API attempts (one request plus two retries)
for transient HTTP or connection failures and shares a 180-second budget across them, including
retry delays. Set `transport_attempts` and `transport_budget` (seconds) to change those limits.
`parse_attempts` limits requests for another reply after a malformed score or label.

## Install and use

```bash
uv tool install --python ">=3.10" "marin-verifyit[math]"
```

The `math` extra installs `math-verify` and `sympy` for symbolic expression grading. Exact,
numeric, and multiple-choice modes work with the base package. Other extras are `schema`,
`judge`, `reasoning-gym`, and `all`. Execution modes use the task image's toolchain.

```python
from pathlib import Path

from verifyit.grade import grade
from verifyit.spec import parse_spec

tests_dir = Path("/tests")
spec = parse_spec((tests_dir / "verifier.toml").read_text())
reward = grade(spec, tests_dir=tests_dir, workspace=Path("/app"))
```

Run the package tests from the repository root:

```bash
uv run --group test pytest lib/verifyit/tests
```

## Candidate scoring

Callers that already extracted an answer use `verifyit.candidate.grade_candidate` for the in-process modes, or the per-mode candidate functions in `verifyit.modes`. Standard specs and script graders share the `Reward` and `Status` contract. Dataset declarations select the grader and lower source fields into its inputs. Shared comparisons belong in VerifyIT; source-specific reward functions stay with the dataset, in its grade scripts and the vendored source scorers they ship.

For text candidates stored in a file, supply the contract and resource locations explicitly:

```bash
python -m verifyit.candidate_file \
  --spec /tests/contract.toml --answer /app/answer.txt \
  --workspace /app --logs-dir /logs/verifier \
  --resources-manifest /tests/resources.json
```

The resource manifest is a JSON list of paths relative to its containing directory; use `[]`
when there are no resources. The command passes the entire file text to `grade_candidate`,
preserving its extraction rules, and writes the same verdict and reward files as `verifyit`.
A missing answer scores zero. This entrypoint accepts text candidates; callers holding decoded
JSON or function calls use the Python API.

A `ScriptSpec` runs an ordinary grading script. The script may compose VerifyIT comparisons or implement its own scoring, and can declare `verdict_file` to distinguish scored results, invalid tasks and infrastructure failures. Fixtures are relative to the tests directory; candidate evidence belongs to the workspace.

`StructuredExactSpec` compares acquired JSON values through `grade_structured_exact_candidate`.
Its TOML reference is encoded as a JSON string so null and nested JSON types survive
`render_spec` and `parse_spec`; JSON tables keep decoded values. Its `numeric_types`
defaults to `"value"`; set `"strict"` to distinguish integers from floats. Structured and predicted-action
candidate JSON rejects duplicate object keys.
