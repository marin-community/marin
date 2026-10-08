# Grading modes

Each `grade_*.py` module implements a shared grading contract used by `verifyit.grade`, `verifyit.candidate.grade_candidate`, or a direct candidate API. The modes cover exact and numeric answers, structured formats, program tests, model judges, and script verdicts. They return the common `Reward` status: scored, invalid task, or infrastructure error.

Add a shared mode when multiple tasks need the same grading behavior. A scorer specific to one dataset belongs with that dataset's declaration as a grade script that ships the source's vendored scorer, not here.
