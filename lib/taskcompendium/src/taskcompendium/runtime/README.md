# Task execution and private grading

[task_grading.py](task_grading.py) grades conversation evidence and acquired files.
Pure answer modes delegate to common candidate grading; isolated modes use
[grading.py](grading.py) to create a fresh Shellbox environment, stage private
resources and candidate files, invoke the declared grader, and parse its result.

Private verifier files are staged under `/tests`; the grading workspace is
normally `/app`. The serialized TaskSpec names the command or VerifyIT descriptor,
private resources, compatible backend and immutable image. Original scorer
packages must be installed in that image, or their source must be supplied as
task resources. Native commands declare their working directory and result path.
They run without importing the ingestion converter.

[shell.py](shell.py) and [episode.py](episode.py) provide an optional actor episode
harness. [output_capture.py](output_capture.py) captures declared files and bounded
directory selections; [resources.py](resources.py) decodes task resource bytes.
Actor and private grading environments have separate requirements and lifetimes.
QEMU runs inside its caller, including a Zephyr worker. An Iris gVisor factory
creates a separate isolated Iris job.

The current native terminal-answer path uses `TaskSpec.output_paths` to stage the
candidate. Separating that private input from actor file capture is proposed in
[GOAL.md](../../../../../GOAL.md), along with RolloutEngine support for native and
unavailable grader contracts. See the [task contract](../../../README.md) and
[pipeline verification contract](../pipeline/README.md).
