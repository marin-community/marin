# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind each IFEval input format to its original scorer."""

from collections.abc import Mapping
from functools import partial
from pathlib import Path

from rigging.secrets import SecretSpec
from taskcompendium.pipeline.execution_binding import VerificationRuntime, bind_private_grader
from taskcompendium.pipeline.models import DatasetRecipe

from experiments.post_training.task_curation.datasets.skyrl import ifeval_native_binding as native_binding


def bind_rlvr_ifeval(
    recipe: DatasetRecipe,
    *,
    image: str,
    verification_runtime: VerificationRuntime,
    controller_url: str | None,
    qemu_bundle: Path | None,
    worker_image: str | None,
    verifier_secret_env: Mapping[str, SecretSpec] | None,
) -> DatasetRecipe:
    return bind_private_grader(
        recipe,
        binder=partial(native_binding.bind, input_format="rlvr"),
        image=image,
        verification_runtime=verification_runtime,
        controller_url=controller_url,
        qemu_bundle=qemu_bundle,
        worker_image=worker_image,
        verifier_secret_env=verifier_secret_env,
    )


def bind_nemotron_ifeval(
    recipe: DatasetRecipe,
    *,
    image: str,
    verification_runtime: VerificationRuntime,
    controller_url: str | None,
    qemu_bundle: Path | None,
    worker_image: str | None,
    verifier_secret_env: Mapping[str, SecretSpec] | None,
) -> DatasetRecipe:
    return bind_private_grader(
        recipe,
        binder=partial(native_binding.bind, input_format="nemotron"),
        image=image,
        verification_runtime=verification_runtime,
        controller_url=controller_url,
        qemu_bundle=qemu_bundle,
        worker_image=worker_image,
        verifier_secret_env=verifier_secret_env,
    )
