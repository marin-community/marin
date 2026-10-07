# Nemotron Ultra grader image

The [Dockerfile](Dockerfile) installs pinned Python dependencies and hash-checked SkyRL scoring modules under `/opt/skyrl_gym`, along with NeMo Skills support used by some source graders. [Ultra grader bindings](../../datasets/README.md) select installed source functions and supply private task contracts; this image does not contain the curation converter.

The campaign runtime manifest supplies the published image by digest. See the [image overview](../README.md).
