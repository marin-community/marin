# [verifyit] Apply output budgets to judge reference and checklist rubrics

The `judge` mode's `reference` and `checklist` rubrics send each judge request without an output token limit or reasoning setting, even though `JudgeSpec` has `max_completion_tokens`, `incomplete_retry_tokens`, and `reasoning_effort`. Those fields apply only to the `labels` rubric. With a reasoning model, the endpoint default decides the budget. A reply cut off at that limit (`finish_reason == "length"`) raises an error and becomes `infra_error`, and a grade costs a full request timeout. A checklist with N criteria makes N such calls, so one truncated reply leaves the whole response ungraded.

Today, on `origin/main`, in `lib/verifyit/src/verifyit/modes/grade_judge.py`:

- `_complete` calls `chat.completions.create(model, messages, temperature=0.0, timeout)` and sends no `max_completion_tokens` and no `reasoning_effort`. `_judge_reference` and `_judge_checklist` use it through `_ask`.
- `_completed_text` raises when `finish_reason != "stop"`. `_ask` retries once (`ATTEMPTS = 2`) only when the `SCORE:` line is missing, not when the reply is truncated.
- `_judge_labels` and `_label_completion` send `max_completion_tokens=budget`, retry once with `incomplete_retry_tokens` when the reply is truncated, and pass `reasoning_effort`.
- Raw replies and attempt counts are not kept in `Reward.detail`. Only a 400-character `reasoning` excerpt is kept.

Proposed:

- Use `max_completion_tokens`, `incomplete_retry_tokens`, and `reasoning_effort` in `_complete` for every rubric, with the same truncation handling as `_label_completion`.
- Record each judge call in `detail`: attempt count, `finish_reason`, and completion token count from `usage`. For `checklist`, record these per criterion.
- Keep the current outcomes: a reply that is truncated after the retry budget, or that has no valid score, stays an unscored `infra_error` and never becomes reward 0.

Usage: a task that sets `max_completion_tokens = 131072` and `incomplete_retry_tokens = 0` in its `judge` spec gets that budget for a checklist rubric, and its verdict shows how many calls each criterion needed.

Evidence:

- In the construct-003 production run, judge calibration rejected 167 items, and 126 of the 150 failed calibration reports contained only ungraded cases. Judge request timeouts affected 99 items and malformed verdicts 29. That judge client also sent no output budget. `experiments/post_training/capability_env_gen/docs/task_contract.md` ("Known limitations") on branch `mark/autoenv`.
- `experiments/post_training/capability_env_gen/capability_pipeline/native_judge_protocol.py` on branch `mark/autoenv`: the re-ask repair that pipeline added around the old judge client.

Not in scope for verifyit: the endpoint, credentials, and concurrency (`JudgeConnection` and the caller), and which model judges a given task (the task's spec).
