## What `grade_candidate` returns

`grade_candidate` returns a `GradeResult` (`taskcompendium.grading_result`):

- `status`, an `Outcome`: `GRADED` when the grader ran and gave a reward; `SUBMISSION_FAILURE`, with reward 0.0, when the answer was refused before any grader ran, such as a reply `answer_format` cannot read; `INVALID_TASK` when the task or its grader is misconfigured; `INFRA_ERROR` when the grader crashed, timed out or wrote no usable reward, with `failure` saying which; `UNAVAILABLE` when no grade was produced at all.
- `reward`: a float for `GRADED` and `SUBMISSION_FAILURE`, `None` for every other status.
- `error`: the message of a grade that is not clean, or `None`.
- `detail`: a dict or `None`. For a grader program it is the `detail` it wrote beside its reward in `verdict.GRADER_VERDICT_PATH`, with its `checks`; for an in-process verifyit grader it is verifyit's own account (a matched answer, a parse error).
- `passed`: a bool for a reward with a pass mark or an exit-code reward, otherwise `None`; read the reward instead.
- `diagnostics`: a dict with the grader command's stdout, stderr and exit code. `grade_diagnostics(grade)` prints the error, the detail and these as one bounded text.
- `failure`: a `GradingFailure` for an `INFRA_ERROR` (`TIMEOUT` means it timed out twice, on fresh machines), otherwise `None`.
- `score_min` and `score_max`: the grader's declared reward range, 0.0 to 1.0 unless it declares another. `verdict.reward_in_range(grade)` checks the reward against it.

`verdict.grader_checks(grade)` parses `detail["checks"]` into `GraderCheck`s, or returns `None` for a grade without checks, such as an in-process verifyit grade. A `GradedCandidate` (`graded_candidate`, kept in `Grader.prototyped`) holds the reply, the files, the status and the reward only; read the score range, the detail and the checks from the `GradeResult`. The task's `answer_type` is an `AnswerType`; the template grades its candidates as `AnswerType.TEXT` with `ANSWER_FORMAT`.

`grade_candidate` passes a task machine exactly when its `requirements` have the shell capability, as validation does: pass the `requirements` step's output as it is, `draft.NO_MACHINE` included, and `workspace` files only for a task with a machine.

A judge grader (`draft.answer_grader(JudgeSpec(...))`) asks a model about the extracted answer and scores by its `rubric`. An empty answer, or one that fails any of its `constraints` (IFEval checks, run first), gets 0.0 without a model call. `rubric="reference"` gives 1.0 to an answer that matches a reference once normalized (`exact_gate`) and otherwise the judge's 0.0, 0.5 or 1.0; `"checklist"` asks one yes/no question per `criteria` entry and gives the fraction answered yes; `"labels"` reads the judge's last line as one of the `label_scores` labels and gives that label's score. The judge reads the answer rather than measuring it, so a length or format rule belongs in `constraints` or in a grader program. `grade_candidate` calls no model, so a build cannot grade a judge (see the rules).

