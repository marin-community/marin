# SkyRL code and SQL grader image

The [Dockerfile](Dockerfile) installs the pinned LiveCodeBench and text-to-SQL scorer modules under `/opt/skyrl_gym`. The [code/SQL binding](../../datasets/README.md) invokes those original modules with private task cases and reads their reward. The APPS evaluator has a [separate image](../apps/README.md).

The campaign runtime manifest supplies the published image by digest. See the [image overview](../README.md).
