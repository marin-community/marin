#!/usr/bin/env python3
"""
Downloads the open-web-math dataset
(https://huggingface.co/datasets/open-web-math/open-web-math) to GCS.
"""

from experiments.defaults import default_download
from marin.execution.executor import executor_main, versioned
from operations.download.huggingface.download import download

############################################################
# download open-web-math dataset

open_web_math_raw = default_download(
    name="raw/open-web-math",
    hf_dataset_id="open-web-math/open-web-math",
    revision=versioned("fde8ef8"),
    output_path="raw/open-web-math-fde8ef8",
    download_fn=download,
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            open_web_math_raw,
        ]
    )
