# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Region-local inputs shared by the Grug science curriculum runs.

Run this module once in the target region before source tokenization. The
worker's ``MARIN_PREFIX`` must point at the user-owned pool root.
"""

from marin.datakit.download.huggingface import download_hf_step
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import prefix_join
from rigging.log_setup import configure_logging

SNOWBALL_HF_MODEL = "open-athena/snowball-67b-a2b-base-262k-qk175-skew8"
SNOWBALL_VERIFIED_WEIGHTS_REVISION = "df89c02d713cca6758a0bdccf4184c8d3ee0a1c5"
SNOWBALL_RELEASE_REVISION = "058ecaf27b9e4f37219df221a51e7d490d58ec3d"
SNOWBALL_MODEL_KEY = f"models/snowball-67b-a2b-base-262k-qk175-skew8/{SNOWBALL_VERIFIED_WEIGHTS_REVISION}"
POOL_VERSION = "2026.09.21-v1"


def snowball_model_path() -> str:
    """Return the pinned Snowball HF export in the active regional pool."""
    return prefix_join(marin_prefix(), SNOWBALL_MODEL_KEY)


def snowball_release_manifest_path() -> str:
    """Return the separately pinned release manifest for the verified weights."""
    return prefix_join(marin_prefix(), f"models/snowball-release/{SNOWBALL_RELEASE_REVISION}")


def snowball_model_download_step() -> StepSpec:
    """Stream the immutable Snowball HF export into the active regional pool."""
    return download_hf_step(
        "raw-model/snowball-67b-a2b-base-262k-qk175-skew8",
        hf_dataset_id=SNOWBALL_HF_MODEL,
        revision=SNOWBALL_VERIFIED_WEIGHTS_REVISION,
        hf_urls_glob=[
            "config.json",
            "model-*.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        ],
        hf_repo_type_prefix="",
        override_output_path=snowball_model_path(),
        zephyr_max_parallelism=16,
    )


def snowball_release_manifest_download_step() -> StepSpec:
    """Stage the release metadata that names and verifies the weights revision."""
    return download_hf_step(
        "raw-model/snowball-67b-a2b-base-262k-qk175-skew8-release-manifest",
        hf_dataset_id=SNOWBALL_HF_MODEL,
        revision=SNOWBALL_RELEASE_REVISION,
        hf_urls_glob=["export-manifest.json"],
        hf_repo_type_prefix="",
        override_output_path=snowball_release_manifest_path(),
        zephyr_max_parallelism=1,
    )


if __name__ == "__main__":
    configure_logging()
    StepRunner().run([snowball_model_download_step(), snowball_release_manifest_download_step()])
