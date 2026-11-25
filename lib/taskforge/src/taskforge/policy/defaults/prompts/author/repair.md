Repair the existing task after {kind}:

{detail}

The task came from another source, without a proposal. The workspace contains its task.json, controls.json and task resources; controls.json may be an empty list. Inspect them, address these findings, and write the complete revised task.json and controls.json. Preserve the objective the task's instruction states, and change only what the findings require. A submission needs at least one control: write controls that test the grader, and use grade to check them before submitting.

Full validation evidence is stored in `.validation/*.json` under the working directory. Read selected fields or byte ranges rather than dumping entire evidence files into tool output.
