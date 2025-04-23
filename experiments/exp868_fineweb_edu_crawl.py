"""
This script crawls the fineweb-edu dataset, which executed the `default_crawl` step for each dump in the dataset.

Link to issue: https://github.com/stanford-crfm/marin/issues/868
"""

# nodryrun

import re

from experiments.crawl.default import default_crawl
from marin.crawl.common.schemas import HtmlExtractionConfig
from marin.crawl.get_fineweb_edu_crawl_yield import filter_and_yield
from marin.execution.executor import executor_main, this_output_path
from marin.utils import fsspec_glob


def url_modifier(url: str) -> str:
    pattern = r"^/fsx/guilherme/cc2023-50/r\d+/input/"
    if re.match(pattern, url):
        url = re.sub(pattern, "s3://commoncrawl/crawl-data/CC-MAIN-2023-50/segments/", url)
    return url


fineweb_crawling_steps = []
fineweb_dumps = fsspec_glob("gs://marin-us-central2/raw/fineweb-edu/*")

for dump in fineweb_dumps:
    step = default_crawl(
        config=HtmlExtractionConfig(
            input_path=dump,
            output_path=this_output_path(),
            source_name="open-web-math",
            columns=["url", "file_path", "date"],
            s3_url_modifier=url_modifier,
        ),
        yield_fn=filter_and_yield,
        input_pattern="CC-MAIN-*/*_links.jsonl.gz",
    )
    fineweb_crawling_steps.append(step)

if __name__ == "__main__":
    executor_main(
        steps=fineweb_crawling_steps,
    )
