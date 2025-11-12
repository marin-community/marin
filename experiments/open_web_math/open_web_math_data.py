# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Downloads the open-web-math dataset
(https://huggingface.co/datasets/open-web-math/open-web-math) to GCS.
"""

from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

############################################################
# download open-web-math dataset
open_web_math_raw = ExecutorStep(
    name="raw/open-web-math",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="open-web-math/open-web-math",
        revision=versioned("fde8ef8"),
        gcs_output_path=this_output_path(),
        wait_for_completion=False,
    ),
    override_output_path="raw/open-web-math-fde8ef8",
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            open_web_math_raw,
        ]
    )
