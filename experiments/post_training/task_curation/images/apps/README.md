# APPS grader image

The [Dockerfile](Dockerfile) pins Python 3.10 and the dependencies needed by the original APPS `testing_util.py`, then downloads that evaluator from a pinned upstream commit and verifies its hash. The curation [code binding](../../datasets/README.md) supplies task cases and an invocation adapter; candidate execution and scoring remain in the installed source evaluator.

Build and image-selection instructions are in the [image overview](../README.md).
