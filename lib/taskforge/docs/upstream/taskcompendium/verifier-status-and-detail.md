# [taskcompendium] Keep verifier status and detail in GradeResult

A TaskCompendium grade cannot say that the task itself is broken, and it drops the evidence the verifier produced. `verifyit` returns a `Reward` with `status` (`scored`, `invalid_task`, `infra_error`) and a JSON `detail` object. `taskcompendium.grading.grade_answer` keeps only `Reward.reward`. A task whose own grading contract is inconsistent (a reference that fails its own verifier, a malformed rubric, a missing private file) is therefore reported as `infra_error` or as an exception. Callers treat both as transient and retry. A partial-credit grade also loses its per-criterion results, so a caller cannot check which criteria passed.

Today, on `origin/main`:

- `lib/taskcompendium/src/taskcompendium/grading.py`: `Outcome` is `graded | extraction_error | infra_error`. `GradeResult` is `status, reward, error`. `grade_answer` returns `GradeResult(Outcome.GRADED, grade_text_candidate(verifier, candidate).reward)`, so `Reward.status` and `Reward.detail` are discarded.
- `lib/taskcompendium/src/taskcompendium/harbor/adapter.py`: `SemanticVerifier._write_result` writes only `status`, `reward`, and `error` to `taskcompendium-result.json`.
- `lib/verifyit/src/verifyit/grade.py` already has the target contract: `Status.INVALID_TASK`, `Reward.detail`, and `write_reward`, which writes `verdict.json` as `{"reward", "status", "detail"}` and omits `reward.json` for unscored verdicts.

At the head of #9623 (schema `0.22`), `GradeResult` gains `passed`, `diagnostics`, `failure`, `score_min`, `score_max`, and `rewards` (numeric components). `Outcome` gains `unavailable` and `skipped`, but still has no invalid-task value. `grade_answer` is unchanged. A `ShellVerifierSpec` that runs `verifyit` inside the grading machine has no reward source that reads `verdict.json`: an `invalid_task` verdict writes no `reward.json`, so `rolloutengine.grading._file_grade` reports `infra_error` with `failure=missing_reward`. The gap is open on main and at the PR head.

Proposed:

- Add `Outcome.INVALID_TASK = "invalid_task"`, with `reward=None`. Callers must not retry it.
- Map `verifyit.grade.Status` to `Outcome` in `grade_answer` (`scored` to `graded`, `invalid_task` to `invalid_task`, `infra_error` to `infra_error`) and copy `Reward.detail` into `GradeResult.diagnostics`. Catch `verifyit.grade.InvalidTask` raised by `candidate_spec` and return `Outcome.INVALID_TASK`.
- Add a reward source to `taskcompendium.environment` for verifiers that write verifyit's verdict file:

  ```python
  class VerdictReward(BaseModel):
      """A verifyit verdict.json: status, reward in [0, 1], and JSON detail."""
      kind: Literal["verdict"] = "verdict"
      path: str  # absolute, for example /logs/verifier/verdict.json
  ```

  `ShellVerifierSpec.reward` accepts it beside `StdoutReward`, `FileReward`, and `ExitCodeReward`. The grader maps `status` as above and stores `detail` in `diagnostics`.
- Write `diagnostics` to `taskcompendium-result.json` in the Harbor adapter so a Harbor trial and a direct grade produce the same record.

Usage: a task ships a private `grade.py` that calls verifyit and writes `verdict.json` with `write_reward`. Its `ShellVerifierSpec` sets `reward=VerdictReward(path="/logs/verifier/verdict.json")`. A caller filters `grade.status == Outcome.INVALID_TASK` into task repair, retries only `infra_error`, and checks partial credit with `grade.diagnostics["criteria"]`.

Evidence:

- In the last production run of a task-generation pipeline (construct-003), 126 of 150 failed judge calibration reports had no misgraded case, only ungraded ones. Task defects and infrastructure failures were counted together.
- Fixed partial-credit controls assert per-criterion results, for example "C1 to C6 pass and C7 fails". With only a scalar reward, these controls cannot be checked. Assertion format: `experiments/post_training/capability_env_gen/docs/partial_controls.md` on branch `mark/autoenv`.
- `experiments/post_training/capability_env_gen/docs/task_contract.md` on branch `mark/autoenv` (outcome table with four statuses).

Not in scope for TaskCompendium: reading the verdict file at grading time is implemented in `lib/rolloutengine/src/rolloutengine/grading.py` (#9623), next to `_file_grade`. The detail schema for each verifier mode belongs to verifyit.
