#!/usr/bin/env python3
"""
Downloads the fineweb-edu dataset
(https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) to GCS.
"""

from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from operations.download.huggingface.download import DownloadConfig, download

############################################################
# download fineweb-edu dataset
fineweb_edu_raw = ExecutorStep(
    name="raw/fineweb-edu",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        revision=versioned("651a648"),
        gcs_output_path=this_output_path(),
        wait_for_completion=False,
    ),
    override_output_path="gs://marin-us-central2/raw/fineweb-edu-651a648",
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            fineweb_edu_raw,
        ]
    )
