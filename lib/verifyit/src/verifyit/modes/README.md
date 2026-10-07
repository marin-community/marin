# Grading modes

Each `grade_*.py` module implements a shared grading contract used by `verifyit.grade` or a direct candidate API. The modes cover exact and numeric answers, structured formats, program tests, model judges, and script verdicts. They return the common `Reward` status: scored, invalid task, or infrastructure error.

The APPS, LiveCodeBench, and SQL entry points call the original evaluator modules installed in the selected task image. They adapt evaluator outputs to VerifyIT's reward file and retain source diagnostics. Dataset declarations prepare source-specific inputs and choose the image. These modes do not store copies of the source scorers.

Use [`../execution/source_callable.py`](../execution/source_callable.py) when the original source function already returns a reward that only needs transport. Add a shared mode when multiple tasks need the same grading behavior. The source's own reward calculation stays with its pinned package.
