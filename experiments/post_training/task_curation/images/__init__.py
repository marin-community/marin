# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned grader and agent images used by the dataset declarations.

Each constant is the digest the previous campaign verified with. ``qemu_bundle`` names the guest
bundle that the campaign worker image carries for the QEMU backend; images without a bundle run on
gVisor or Docker only.
"""

from shellbox.machine import Backend

from experiments.post_training.task_curation.pipeline import Image

GVISOR_ONLY = (Backend.GVISOR, Backend.DOCKER)

NEMOTRON_ULTRA_IMAGE = Image(
    "ghcr.io/marin-community/iris-task@sha256:e64e5bb74faddb161739ce11b79e1c35e5cbdfd06b70463751c73e91bccb8896",
    qemu_bundle="/opt/task-curation-qemu-ultra-instructions",
)
# One combined build carries the APPS, LiveCodeBench, text-to-SQL and IFEval scorers with the
# Nemotron Ultra modules (see the Dockerfiles in this directory).
APPS_IMAGE = NEMOTRON_ULTRA_IMAGE
SKYRL_CODE_SQL_IMAGE = NEMOTRON_ULTRA_IMAGE
IFEVAL_IMAGE = NEMOTRON_ULTRA_IMAGE
ULTRA_MCQA_IMAGE = Image(
    "ghcr.io/marin-community/iris-task@sha256:f3f2b4b710a3ad266184bd4ef124ff8a37ce8f1db45773969ae942c34bb6a2c1",
    qemu_bundle="/opt/task-curation-qemu-ifeval",
)
EXECUTABLE_MATH_IMAGE = Image(
    "ghcr.io/marin-community/task-curation-executable@sha256:"
    "e9eac86838e0d7217fb4048d8eba46e69956e73a32b1b42f9019c1d2a5842d73",
    qemu_bundle="/opt/task-curation-qemu-math",
)
ARC_IMAGE = Image(
    "ghcr.io/marin-community/iris-task@sha256:f311383b9caadee19f009d62126a354135f6fc4aba3151e4dffa3a548ee062cc",
    backends=GVISOR_ONLY,
)
REASONING_GYM_IMAGE = Image(
    "ghcr.io/marin-community/iris-task@sha256:aa369fcb572bbd33e18ff31d9e7f618b3a76dd6ca7a834a42c49eb45d42fa727",
    backends=GVISOR_ONLY,
)
TASKTROVE_EXECUTABLE_IMAGE = Image(
    "ghcr.io/marin-community/task-curation-executable@sha256:"
    "d6af0d198b29650fea0eaccb27d01cb9bbfb0aec9e1d5959a74c2ae055b6f305",
    qemu_bundle="/opt/task-curation-qemu",
)
TASKTROVE_PYTHON_TESTS_IMAGE = Image(
    "ghcr.io/marin-community/iris-task@sha256:d15747080ff81dbbec4a1dcbc7cd651d3b935054d4b67e4b55dc517e1b2cfc56",
    qemu_bundle="/opt/task-curation-qemu-generic-bootstrap",
)
TASKTROVE_NL2BASH_IMAGE = Image(
    "ghcr.io/marin-community/task-curation-executable@sha256:"
    "66cba7cb3eb682f9a53e444876ef2468670336a71e03559de85b5b2b5d4cdde6",
    qemu_bundle="/opt/task-curation-qemu-nl2bash",
)
TASKTROVE_STACK_PYTEST_IMAGE = Image(
    "ghcr.io/marin-community/iris-task@sha256:97528e23c249c993641b0b5c6e7d05a978588f4be276f0fe61b1f3c3bcc6c622",
    qemu_bundle="/opt/task-curation-qemu-stack-bootstrap",
)
# Python 3.12 with verifyit, harbor-rewardkit 0.1.4 and litellm: the environment the TaskTrove
# source judges ran in.
REWARDKIT_IMAGE = Image(
    "ghcr.io/marin-community/iris-task@sha256:ee7a6cd6e5b536a296da1931556fa0becb8b24746415b47f934eaaa3df32b0f1",
    backends=(Backend.GVISOR,),
)

IMAGES = (
    NEMOTRON_ULTRA_IMAGE,
    ULTRA_MCQA_IMAGE,
    EXECUTABLE_MATH_IMAGE,
    ARC_IMAGE,
    REASONING_GYM_IMAGE,
    TASKTROVE_EXECUTABLE_IMAGE,
    TASKTROVE_PYTHON_TESTS_IMAGE,
    TASKTROVE_NL2BASH_IMAGE,
    TASKTROVE_STACK_PYTEST_IMAGE,
    REWARDKIT_IMAGE,
)
"""Every distinct image, for looking up QEMU bundles by reference."""
