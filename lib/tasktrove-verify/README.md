# tasktrove-verify

`tasktrove-verify` executes the grader contract in a converted TaskTrove task. The task supplies a
flat `tests/verifier.toml`; `tests/test.sh` contains this shim:

```sh
exec tasktrove-verify /tests/verifier.toml
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
| `structured_exact` | exact JSON values with distinct scalar types and ordered arrays |
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
expected answer. Without a box marker, `math` grades the last nonempty line and `numeric` grades the
last number. Numeric expected values and absolute and relative tolerances must be finite. Tolerances
must also be nonnegative. The effective tolerance,
`max(tolerance_abs, tolerance_rel * abs(expected))`, must be finite.

[`spec.py`](src/tasktrove_verify/spec.py) owns the frozen mode dataclasses plus `parse_spec` and
`render_spec`. Spec paths are relative to the directory containing `verifier.toml`. `grade.py`
owns dispatch, output handling, verdict writing, and the CLI. Executable graders live in
`modes/grade_*.py`; shared parsers and process runners remain separate.

`candidate_spec(mode, parameters)` validates the shared exact, numeric, MCQ, structured-JSON and predicted-action
contracts for callers that already extracted a submission. `grade_text_candidate` scores extracted
text; `grade_predicted_action_candidate` scores decoded function calls, and
`grade_structured_exact_candidate` scores acquired JSON values. These APIs perform no
filesystem or harness operations. Predicted-action matching preserves duplicate calls and requires
all calls to match one to one. Argument objects stay decoded in JSON descriptors; `render_spec`
encodes each argument object and each structured reference as a JSON string in TOML so nested
JSON null values survive `parse_spec`. Structured references must contain finite JSON values;
invalid references omit reward files instead of receiving a zero.

## Install and use

```bash
uv tool install --python ">=3.11" \
  "tasktrove-verify[answer] @ git+https://github.com/marin-community/marin@<sha>#subdirectory=lib/tasktrove-verify"
```

Extras are `answer`, `schema`, `judge`, `reasoning-gym`, and `all`. Execution modes use the task
image's toolchain.

```python
from pathlib import Path

from tasktrove_verify.grade import grade
from tasktrove_verify.spec import parse_spec

tests_dir = Path("/tests")
spec = parse_spec((tests_dir / "verifier.toml").read_text())
reward = grade(spec, tests_dir=tests_dir, workspace=Path("/app"))
```

Run the package tests from the repository root:

```bash
uv run --group test pytest lib/tasktrove-verify/tests
```
