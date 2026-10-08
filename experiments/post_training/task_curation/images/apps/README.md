# APPS grader image

The [Dockerfile](Dockerfile) pins Python 3.10 and the dependencies needed by the original APPS `testing_util.py`, then downloads that evaluator from a pinned upstream commit and verifies its hash. The [SkyRL code declarations](../../datasets/README.md) supply each task's test cases and a runner script; candidate execution and scoring stay in the installed evaluator.

See the [image overview](../README.md).
