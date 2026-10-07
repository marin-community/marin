# Reasoning Gym grader image

The [Dockerfile](Dockerfile) starts from a pinned Iris task image and adds hash-checked SkyRL modules used by the original Reasoning Gym runner. The [source binding](../../datasets/README.md) selects the TaskTrove script or installed package scorer and passes its private task data.

The campaign runtime manifest supplies the published image by digest. See the [image overview](../README.md).
