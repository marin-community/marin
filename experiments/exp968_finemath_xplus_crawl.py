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
This script crawls the finemath 3 plus and 4 plus datasets, which executed the `default_crawl`
step for each dump in the dataset.

Link to issue: https://github.com/stanford-crfm/marin/issues/968
"""

# nodryrun

from experiments.crawl.default import default_crawl
from marin.crawl.common.schemas import HtmlExtractionConfig
from marin.crawl.get_finemath_crawl_yield import filter_and_yield
from marin.execution.executor import ExecutorMainConfig, executor_main, this_output_path

executor_config = ExecutorMainConfig(
    force_run_failed=True,
)


def url_modifier(url: str) -> str:
    return f"s3://commoncrawl/{url}"


finemath_crawling_steps = default_crawl(
    config=HtmlExtractionConfig(
        input_path="gs://marin-us-central2/raw/finemath-7090a5/finemath-3plus",
        output_path=this_output_path(),
        source_name="finemath-3plus",
        columns=[
            "url",
            "fetch_time",
            "content_mime_type",
            "warc_filename",
            "warc_record_offset",
            "warc_record_length",
            "text",
            "token_count",
            "char_count",
            "html_text",
            "html_token_count",
            "html_char_count",
        ],
        url_column="url",
        file_path_column="warc_filename",
        s3_url_modifier=url_modifier,
    ),
    yield_fn=filter_and_yield,
    input_pattern="*.jsonl.gz",
)


if __name__ == "__main__":
    executor_main(
        steps=finemath_crawling_steps,
    )
