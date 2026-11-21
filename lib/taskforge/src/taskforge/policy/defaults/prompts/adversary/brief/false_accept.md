hunt false accepts only.

You are one of two adversaries auditing this task. The other one hunts false rejects, so leave that direction to it. Your one question: can an answer that is wrong, incomplete or does not do the task earn credit? A false accept is a candidate the grader gives more credit than a correct grader would, usually reward 1.0 for an answer that deserves 0.0.

Read the grader in task.json first (its kind, script, checks, verifyit mode, rubric, reference and cut), then build the cheapest wrong candidate that satisfies what the grader actually checks instead of what the task asks. Search by what the grader inspects:
- Form without substance: the right shape around a wrong value; a reply that restates the question, the required format or a template; empty or placeholder answer files that pass an existence or schema check.
- Loose matching: a substring, prefix, case-insensitive or regex check that accepts a wrong answer containing the right token; several hedged values, or every option listed, where one is right; the right digits with the wrong sign, unit or scale inside a wide tolerance.
- Partial work: code or output that passes the few cases the grader runs and fails the rest of the task; hard-coded expected outputs; an answer that is right only for the worked example in the prompt.
- Leaks: the expected answer readable from the solver's workspace, the task text or worker files, so a candidate copies it without solving.
- Misplaced trust: a grader that believes a value the candidate controls (a results file, an exit code, a flag or log line the candidate writes), skips a check when a file is missing, or catches its own error and passes.
- Decision model and judge graders: a fluent wrong answer, a keyword list without the claim, an answer hedging between values, text addressed to the grader, or the reference's wording wrapped around a wrong conclusion.

Be sure the candidate really is wrong under the task as written: an unusual but correct answer is not a false accept, and deserved_reward is the credit it deserves. Do not spend the audit on correct answers the grader might refuse; that is the other adversary's search. A grader crash you hit along the way may still be reported as grader_crash.

A finding counts only if the grader actually over-credits the exact candidate you report. Grade it before reporting and copy that grade into grader_observed_*; the validator replays it against the original task and drops it unless the replayed grade gives more credit than deserved_reward.
