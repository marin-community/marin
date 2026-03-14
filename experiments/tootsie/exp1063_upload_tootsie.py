# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions to upload model checkpoints to Hugging Face.

This module provides utilities to create executor steps for uploading model checkpoints
from Google Cloud Storage (GCS) to Hugging Face repositories.

Usage:
1. Define a ModelUploadConfig with the HF repo ID and GCS paths
2. Create an upload step using upload_model_to_hf_step()
3. Execute the step with your executor

Example:
```python
upload_config = ModelUploadConfig(
    hf_repo_id="organization/model-name",
    gcs_directories=["gs://bucket/checkpoints/model-name/hf/"]
)
upload_step = upload_model_to_hf_step(upload_config)
executor_main([upload_step])
```
"""

from dataclasses import dataclass, field

from marin.download.huggingface.upload_gcs_to_hf import UploadConfig, upload_gcs_to_hf
from marin.execution.executor import ExecutorStep, executor_main


@dataclass(frozen=True)
class ModelUploadConfig:
    """Configuration for uploading model checkpoints to Hugging Face."""

    hf_repo_id: str
    gcs_directories: list[str] = field(default_factory=list)
    dry_run: bool = False


def upload_model_to_hf_step(model_config: ModelUploadConfig) -> ExecutorStep:
    """Create an ExecutorStep to upload model checkpoints to Hugging Face."""
    upload_step = ExecutorStep(
        name=f"upload_to_hf_{model_config.hf_repo_id.replace('/', '_')}",
        fn=upload_gcs_to_hf,
        config=UploadConfig(
            hf_repo_id=model_config.hf_repo_id,
            gcs_directories=model_config.gcs_directories,
            dry_run=model_config.dry_run,
            wait_for_completion=True,
        ),
    )

    return upload_step


# Predefined upload steps organized by region
# EU West4 region uploads
eu_west4_uploads = upload_model_to_hf_step(
    ModelUploadConfig(
        hf_repo_id="WillHeld/mystery-model",
        gcs_directories=[
            "gs://marin-eu-west4/checkpoints/llama-8b-tootsie-0.001-19ad63/hf/",
        ],
    )
)

# US Central2 region uploads
us_central2_uploads = upload_model_to_hf_step(
    ModelUploadConfig(
        hf_repo_id="WillHeld/mystery-model",
        gcs_directories=[
            "gs://marin-us-central2/checkpoints/llama-8b-tootsie-phase2/hf/",
            "gs://marin-us-central2/checkpoints/llama-8b-tootsie-phase3/hf/",
            "gs://marin-us-central2/checkpoints/tootsie-8b-soft-raccoon-3/hf/",
            "gs://marin-us-central2/checkpoints/llama-8b-tootsie-adept-phoenix/hf/",
            "gs://marin-us-central2/checkpoints/tootsie-8b-sensible-starling/hf/",
        ],
    )
)

# US Central1 region uploads
us_central1_uploads = upload_model_to_hf_step(
    ModelUploadConfig(
        hf_repo_id="WillHeld/mystery-model",
        gcs_directories=[
            "gs://marin-us-central1/checkpoints/tootsie-8b-deeper-starling/hf/",
        ],
    )
)


if __name__ == "__main__":
    # Default to running region-based upload steps when script is executed directly
    executor_main(
        [eu_west4_uploads, us_central2_uploads, us_central1_uploads],
        description="Upload model checkpoints to Hugging Face by region",
    )
