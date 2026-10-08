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
| `predicted_action` | unordered function calls with exact JSON types and optional float tolerance |
| `mcq` | expected option letter |
| `math` | expression equality through math-verify |
| `numeric` | numeric equality with explicit tolerances |
| `exact` | normalized string equality |
| `json-schema` | JSON, YAML, or TOML checked against JSON Schema |
| `xml-elements` | required XML elements and attributes |
| `csv-columns` | required CSV header columns |
| `ifeval` | deterministic instruction-following constraints |
| `reasoning-gym` | the named reasoning-gym scorer and entry |
| `stdio` | program stdout over hidden cases |
| `pytest` | pytest JSON report with required and protected tests |
| `junit` | JUnit XML report |
| `gotest` | `go test -json` events |
| `judge` | reference-answer or checklist rubric through a configured model endpoint |
| `script` | legacy `test.sh` fallback with normalized reward files and fail-closed errors |

For the `math` and `numeric` grading modes, the last `\boxed{...}` occurrence determines the
candidate when the output contains a box marker. Its braces must be balanced and its content must be
nonempty. Otherwise, the candidate receives reward `0.0`, even when an earlier marker contains the
expected answer. Without a box marker, both modes read the last nonempty line. Numeric mode
extracts exactly one integer, decimal, scientific-notation value or integer fraction from that line.
Surrounding prose is ignored, including negation: `Definitely not 42` extracts `42`.
When a box is present, its entire content must be a numeric literal, optionally wrapped in math delimiters.
Thousands separators require groups of three digits. Multiple literals such as `12 or 13` or
`2 + 2`, malformed numbers and nonfinite values are malformed submissions and receive reward `0.0`.

Numeric private `expected` is a required literal string; `tolerance_abs` and `tolerance_rel`
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

For `judge` reference and checklist rubrics, `max_completion_tokens` sets the initial chat request
budget (default `8192`). A positive `incomplete_retry_tokens` must exceed it and permits one larger
request when a reply ends with `finish_reason="length"`. `reasoning_effort`, when set, is sent with
each request. Reference verdicts record `attempt_count` and `attempts` in `detail`; checklist
verdicts record them under each entry in `detail.criteria`. Each attempt contains `finish_reason`
and `completion_tokens` (`null` when the endpoint omits usage). An exhausted retry remains
`infra_error`, with the available attempt diagnostics in `detail`.

The checklist rubric judges each criterion `samples` times (default 1). With
`sample_resolution = "majority"` the count must be odd and the majority verdict wins. With
`"two_then_third"` and `samples = 3`, the judge is asked twice and a third time only when the
first two verdicts disagree. Single judgments go out at temperature 0. Repeated judgments use
`sample_temperature` (default 0, finite and nonnegative); at 0 a deterministic judge repeats its
first verdict, so set it above 0 to reduce judge noise. Each checklist entry keeps every verdict
under `samples`, with its own `reasoning`, `attempt_count` and `attempts`; the entry's
`attempt_count` and `attempts` cover every request for that criterion, and its `reasoning` is the
first sample's that agrees with the resolved verdict. A sample that fails leaves the whole grade
an infrastructure error. Repeated judgments are rejected outside the checklist rubric.

The mode modules expose direct candidate graders for callers holding extracted values.
`aggregate_rewards` combines required components with ALL, MEAN, MAX, MIN, or PRODUCT;
invalid tasks and infrastructure errors discard partial credit. The judge and Reasoning Gym modes
also expose direct candidate APIs for decoded context and trusted entries. Judge connections carry
runtime credentials separately from serializable specs. Reasoning Gym's optional `params` file
configures its scorer; callers own isolation when invoking its direct API.

`adapters/` prepares framework observations for the shared modes. Frameworks retain task
execution, dispatch, and dependency pins; installing this package does not enable an adapter.
`preparation/` retains raw inputs and named normalization policies. Preparation failures carry
Harbor's error categories from the pinned config-only `harbor-config` dependency.

`json_comparison.json_values_equal` compares decoded JSON values with strict types and an optional
float tolerance. `modes.grade_nl2bash` compares shell-output records as a multiset, preserving
repeated records and rejecting unexpected errors.

`candidate_spec(mode, parameters)` validates the shared exact, numeric, MCQ and predicted-action
contracts for callers that already extracted a submission. `grade_text_candidate` scores extracted
text; `grade_predicted_action_candidate` scores decoded function calls. These APIs perform no
filesystem or harness operations. Predicted-action matching preserves duplicate calls and requires
all calls to match one to one. Argument objects stay decoded in JSON descriptors; `render_spec`
encodes each argument object as a JSON string in TOML so nested JSON null values survive
`parse_spec`.

## Install and use

```bash
uv tool install --python ">=3.11" \
  "verifyit[answer] @ git+https://github.com/marin-community/marin@<sha>#subdirectory=lib/verifyit"
```

Extras are `answer`, `schema`, `judge`, `reasoning-gym`, and `all`. Execution modes use the task
image's toolchain.

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

Callers that already extracted an answer can use generic candidate scorers in `verifyit.candidate` and `verifyit.modes`. Standard specs and script graders share the `Reward` and `Status` contract. Dataset policy belongs to the converter that emits a grader: source parsing conventions, calendar postconditions and abstention rules should be packaged as task-owned scripts.

A `ScriptSpec` runs an ordinary grading script. The script may compose VerifyIT comparisons or implement its own scoring, and can declare `verdict_file` to distinguish scored results, invalid tasks and infrastructure failures. Private fixtures are relative to the tests directory; candidate evidence belongs to the workspace.

`StructuredExactSpec` compares acquired JSON values through `grade_structured_exact_candidate`.
Its TOML reference is encoded as a JSON string so null and nested JSON types survive
`render_spec` and `parse_spec`; private JSON descriptors keep decoded values. Its `numeric_types`
defaults to `"value"`; set `"strict"` to distinguish integers from floats. Structured and predicted-action
candidate JSON rejects duplicate object keys.
