# Grading modes

Each `grade_*.py` module implements a shared grading contract used by `verifyit.grade`, `verifyit.candidate.grade_candidate`, or a direct candidate API. The modes cover exact and numeric answers, structured formats, program tests, model judges, and script verdicts. They return the common `Reward` status: scored, invalid task, or infrastructure error.

Use [`../execution/source_callable.py`](../execution/source_callable.py) when the original source function already returns a reward that only needs transport. Add a shared mode when multiple tasks need the same grading behavior. A scorer specific to one dataset belongs with that dataset's declaration as a grader script, not here; the source's own reward calculation stays with its pinned package.
