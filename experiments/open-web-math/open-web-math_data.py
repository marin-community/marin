#!/usr/bin/env python3
"""
Downloads the open-web-math dataset
(https://huggingface.co/datasets/open-web-math/open-web-math) to GCS.
"""

from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from operations.download.huggingface.download import DownloadConfig, download

############################################################
# download open-web-math dataset
open_web_math_raw = ExecutorStep(
    name="raw/open-web-math",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="open-web-math/open-web-math",
        revision=versioned("fde8ef8"),
        gcs_output_path=this_output_path(),
        wait_for_completion=False,
    ),
    override_output_path="gs://marin-us-central2/raw/open-web-math-fde8ef8",
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            open_web_math_raw,
        ]
    )
