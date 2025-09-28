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
This experiment is used to convert the DCLM HQ dump to markdown. DCLM HQ does not provide the WARC paths
for each document, so we use the Common Crawl index to get the WARC paths and then use the `download_dclm_hq_html`
operation to download the HTML content.

Note: At this moment we use a low spec machine to do this which causes the process to take a long time of ~3 months.

Reference Issue: https://github.com/stanford-crfm/marin/issues/813
"""

from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, executor_main, this_output_path

html_extracted_dclm_hq = ExecutorStep(
    name="raw/dolmino-dclm-hq-html-extracted",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="mlfoundations/dclm-baseline-1.0",
        revision="b428a12",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    pip_dependency_groups=["download_transform"],
)


if __name__ == "__main__":
    executor_main(
        steps=[
            html_extracted_dclm_hq,
        ]
    )
