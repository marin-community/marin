# [taskcompendium] JSON submissions fail on a fenced reply or numeric answer

With the `json` submission convention, GLM-5.3 answered a number task correctly in 3 of 3 trials, and all 3 were recorded as `extraction_error`. Each reply was

````
```json
{"answer": 42}
```
````

`extract_answer` calls `json.loads` on the whole message, so the Markdown fence fails parsing (`Expecting value: line 1 column 1 (char 0)`). Without the fence the reply would still fail, because the convention requires `answer` to be a string and the model sent the integer `42`. On the same task and endpoint, the `plain` and `answer_call` conventions graded 3 of 3 at reward 1.0. As written, the JSON convention measures whether the model avoids a code fence, and training on it would teach fence avoidance.

Today: `lib/taskcompendium/src/taskcompendium/submission.py` on `origin/main`. `extract_answer` requires `json.loads(response.content)` to be an object whose `answer` is a nonempty string. `submission_instruction` says `Give your answer as a JSON object with an "answer" field.` and does not mention fences or the string type. The head of #9623 moves request building into `submission_request` but leaves `extract_answer` unchanged, and `rolloutengine` grades answer tasks through the same `grade_answer`, so the gap is open on main and at the PR head.

Proposed: pick one.

- (a) Before parsing, strip one fenced code block that encloses the whole reply. `verifyit.modes.extract.unwrap_fence` already does this. For `answer_type=number`, accept a finite JSON number and convert it to its string form. Any other non-string `answer` stays an `extraction_error`.
- (b) Keep extraction strict and state the contract in the instruction: `Reply with only a JSON object, without a code fence. "answer" must be a string.`

Option (a) changes what counts as a valid submission. Option (b) only changes the instruction text.

Usage: no caller change. Lowering and rollouts that select `SubmissionConvention(id="json", answer_format=AnswerFormat.JSON)` grade the fenced reply above as a number answer of `42` under (a), or receive the explicit instruction under (b).

Reproduce: lower a `numeric` task (expected `42`) under `SubmissionConvention(id="json", answer_format=AnswerFormat.JSON)` and run `taskcompendium.harbor.runner.run_trial` against GLM-5.3.

Evidence: two runs on 2026-10-05 (GLM-5.3, interactive tier), 9 Harbor trials each. plain: 3/3 graded 1.0 (reply `42`). json: 0/3 (`extraction_error`, every reply the fenced object above). answer_call: 3/3 graded 1.0 (`submit_answer` with `{"answer": "42"}`). A direct call with the identical JSON-convention request returned the same fenced reply with `finish_reason: "stop"` and 53 completion tokens (41 reasoning), so the reply was complete.

Not in scope for TaskCompendium: answer normalization after extraction (numeric tolerance, boxed answers) stays in verifyit.
