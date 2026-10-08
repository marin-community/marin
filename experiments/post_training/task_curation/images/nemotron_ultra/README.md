# Nemotron Ultra grader image

The [Dockerfile](Dockerfile) installs pinned Python dependencies and hash-checked SkyRL scoring modules under `/opt/skyrl_gym`, along with NeMo Skills support used by some source graders. The [Nemotron Ultra declarations](../../datasets/README.md) call those installed scorer functions on each task's grading data; the image does not contain the curation converters.

See the [image overview](../README.md).
