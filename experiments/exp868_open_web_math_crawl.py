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
This script crawls the open-web-math dataset, which executed the `default_crawl` step for the dataset.

Link to issue: https://github.com/marin-community/marin/issues/868
"""

# nodryrun

import json

from experiments.crawl.default import default_crawl
from marin.crawl.common.schemas import HtmlExtractionConfig
from marin.crawl.get_open_web_math_crawl_yield import filter_and_yield
from marin.execution.executor import executor_main, this_output_path

open_web_math_crawling_steps = default_crawl(
    config=HtmlExtractionConfig(
        input_path="gs://marin-us-central2/raw/open-web-math-fde8ef8/fde8ef8/huggingface.co/datasets/open-web-math/open-web-math/resolve/fde8ef8/data/",
        output_path=this_output_path(),
        source_name="openwebmath",
        columns=["url", "date", "metadata"],
        warc_path_extractor=lambda x: json.loads(x)["warc_path"],
        max_files=1,
    ),
    yield_fn=filter_and_yield,
)

if __name__ == "__main__":
    executor_main(
        steps=open_web_math_crawling_steps,
    )
