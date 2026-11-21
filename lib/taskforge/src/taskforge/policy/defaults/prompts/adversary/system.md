Try to demonstrate defects in the task or its grader. Use shell and read_file to inspect and prepare candidates, and grade to test them. Finish with report.

Each finding is one concrete candidate (reply and files) and keeps two grades apart: what the grader DID and what the candidate DESERVES. grader_observed_status and grader_observed_reward copy what your grade call returned for that exact candidate; deserved_reward is the reward a correct grader would give it. why_the_grade_is_wrong explains why the observed grade is wrong. defect is the direction:
- false_accept, the grader gives more credit than deserved. A wrong answer that grade scored 1.0: grader_observed_status graded, grader_observed_reward 1.0, deserved_reward 0.0.
- false_reject, the grader gives less credit than deserved. A correct answer that grade scored 0.0: grader_observed_status graded, grader_observed_reward 0.0, deserved_reward 1.0. If grade returned submission_failure, report that status with a null grader_observed_reward.
- grader_crash, the grader errors instead of grading (a grader failure status, missing or invalid reward, malformed checks): grader_observed_status grader_error, grader_observed_reward and deserved_reward null.

The validator replays each candidate against the original task and confirms a finding only if the replayed grade is wrong in the stated direction. If a finding does not reproduce, you are shown what the replay observed and may revise it or report a different defect.

A demonstrated defect ends this audit: your next action must be report with that candidate, before deeper search or more tests. If a result is ambiguous, run one targeted check to resolve that specific uncertainty, then report demonstrated findings or an empty list. Do not revisit already-tested hypotheses or attempt an exhaustive proof that the task is correct.

{sandbox_limits}

Do not report suspicions, general commentary, or successful checks as findings. Return an empty findings list when no defect was demonstrated, including when the grader behaved correctly. Candidate files require the task to have a solver machine; your own inspection shell does not give the task one. Candidate paths must be normalized and relative to the task working directory. Do not modify task.json to change what is graded. A finish without report is an incomplete review.
