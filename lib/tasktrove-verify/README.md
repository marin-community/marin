# tasktrove-verify

The grader for TaskTrove Clean tasks. Every task ships `tests/verifier.toml`, a flat table
naming one `mode` and that mode's parameters, and a three-line `tests/test.sh` that runs

```
exec tasktrove-verify /tests/verifier.toml
```

The tool grades the agent's work and writes `/logs/verifier/reward.json`:

```json
{"reward": 1.0, "status": "scored", "detail": {"extracted": "C"}}
```

plus `reward.txt` beside it for Harbor. `status` is `scored`, `invalid_task` (the task itself is
malformed) or `infra_error` (the grader crashed). A trainer keeps scored rewards and drops the rest.
The process exits 0 whenever it managed to write a reward; nothing about the candidate can make
it exit non-zero.

## Modes

| mode | candidate | compared against |
|---|---|---|
| `mcq` | output file | expected option letter |
| `math` | output file, boxed or last expression | expected expression via math-verify |
| `numeric` | output file | expected number with tolerance |
| `exact` | output file | expected string or list after normalization |
| `json-schema` | output file parsed as JSON or YAML | a JSON Schema under tests/ |
| `ifeval` | output file | a list of IFEval constraints |
| `reasoning-gym` | output file | the reasoning-gym scorer for the entry |
| `stdio` | a program in the workspace, run per case | expected stdout per case |
| `pytest` | the workspace | pytest JSON report, must_pass / must_not_break |
| `junit` | the workspace | JUnit XML report |
| `gotest` | the workspace | `go test -json` |
| `judge` | output file | `reference` rubric: reference answers, model call only after the exact gate misses; `checklist` rubric: yes/no criteria scored as a fraction, optional context file and IFEval gate |
| `script` | whatever the script reads | whatever the script decides; the fallback |

`spec.py` is the contract: one frozen dataclass per mode, `parse_spec` and `render_spec`.
Paths in a spec are relative to the directory holding `verifier.toml`.

## Install

```
uv tool install --python ">=3.11" "tasktrove-verify[answer] @ git+https://github.com/marin-community/marin@<sha>#subdirectory=lib/tasktrove-verify"
```

Extras: `answer`, `schema`, `judge`, `reasoning-gym`, `all`. Execution modes use the task image's
own toolchain and need no extra.

## Library use

```python
from tasktrove_verify.grade import grade
from tasktrove_verify.spec import parse_spec

reward = grade(parse_spec(text), tests_dir=Path("/tests"), workspace=Path("/app"))
```

## Tests

```
uv run --group test pytest lib/tasktrove-verify/tests
```
