# Task execution and grading

[task_grading.py](task_grading.py) grades a captured attempt with the task's grader.
`grade_task` returns `unavailable` for a `NoGrader`, grades an in-process
`VerifyitGrader` with `taskcompendium.grading.grade_answer`, and sends a grader
with an environment to [grading.py](grading.py). There, `grade_in_sandbox`
creates a fresh Shellbox machine from the grader's environment, stages the
grading inputs, runs the verifyit command or the script grader's command, and
parses its reward.

One archive stages the inputs, in order; later entries replace earlier ones:
verifier resources under `/tests`, shared and worker resources, captured output
files, the extracted answer (the verifyit mode's `output` file or the script
grader's `answer_path`), captured state at `/app/state.json`,
`/tests/verifier.toml` for verifyit, the conversation at the script grader's
`conversation_path`, and artifacts copied from the agent's machine. The grading
machine's working directory is the grader's workspace, normally `/app`. The
environment's setup commands run as root before grading.

The serialized TaskSpec names the verifyit mode or the grader command, verifier
resources, compatible backends and immutable image. Original scorer packages
must be installed in that image, or their source must be supplied as task
resources. Graders run without importing the ingestion converter.

[models.py](models.py) holds captured `RuntimeEvidence`; `grading_attempt` pairs
it with the conversation. [shell.py](shell.py) and [episode.py](episode.py)
provide an optional actor episode harness. [output_capture.py](output_capture.py)
captures declared files and bounded directory selections;
[resources.py](resources.py) decodes task resource bytes. Actor and grading
environments have separate requirements and lifetimes. QEMU runs inside its
caller, including a Zephyr worker. An Iris gVisor factory creates a separate
isolated Iris job. See the [task contract](../../../README.md) and
[pipeline verification contract](../pipeline/README.md).
