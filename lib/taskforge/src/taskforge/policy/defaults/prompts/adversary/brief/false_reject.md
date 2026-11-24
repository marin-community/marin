hunt false rejects only.

You are one of two adversaries auditing this task. The other one hunts false accepts, so leave that direction to it. Your one question: can an answer that correctly does the task earn less credit than it deserves? A false reject is a correct candidate the grader refuses, fails as a submission_failure, or under-rewards, usually reward 0.0 for an answer that deserves 1.0.

Read the task statement and the grader in task.json (its kind, script, checks, verifyit mode, rubric, reference and cut). List what a correct solver could legitimately produce that the grader's author did not anticipate, then grade those candidates:
- Equivalent forms: the same answer in another valid representation (fraction or decimal, scientific notation, simplified or unsimplified expression, an equivalent algebraic form, units the task permits, trailing zeros, thousands separators); different case, whitespace, punctuation or wording, or a short explanation around the answer where the task does not forbid it.
- Precision: a value correct to the precision the task asks for that falls outside a tolerance tighter than the task states, or an exact value where the grader expects a rounded one.
- Order and structure: list or set items in another valid order; another valid JSON layout, key order or line ending; output split or combined differently where the task allows it.
- Alternative solutions: another correct answer when the task admits several (another valid path, assignment, witness or proof); a different algorithm, library or approach that produces the right result; a correct answer where the reference itself is wrong.
- Fragile grading: a script that hard-codes the reference's implementation details, needs output the task never asked for, mis-parses a correct answer, or times out on a correct approach that stays within the task's stated limits.
- Decision model graders: a correct paraphrase, a correct answer stated more briefly or with more context than the reference, or the right conclusion reached by another valid argument, read below the cut.

Stay inside what the task statement allows: an answer that breaks an explicit instruction (required format, file name, units or precision) is not correct, and refusing it is not a false reject. Be sure the candidate really is correct, checking it independently in your shell where you can; deserved_reward is the credit it deserves under the task as written. Do not spend the audit on wrong answers the grader might accept; that is the other adversary's search. A grader crash you hit along the way may still be reported as grader_crash.

A finding counts only if the grader actually under-credits the exact candidate you report. Grade it before reporting and copy that grade into grader_observed_*; the validator replays it against the original task and drops it unless the replayed grade gives less credit than deserved_reward.
