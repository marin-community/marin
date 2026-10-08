# SkyRL code and SQL grader image

The [Dockerfile](Dockerfile) installs the pinned LiveCodeBench and text-to-SQL scorer modules under `/opt/skyrl_gym`. The [SkyRL code declarations](../../datasets/README.md) call those modules with each task's hidden test cases and read their reward. The APPS evaluator has a [separate image](../apps/README.md).

See the [image overview](../README.md).
