"""
This script crawls the finemath 3 plus and 4 plus datasets, which executed the `default_crawl` step for each dump in the dataset.

Link to issue: https://github.com/stanford-crfm/marin/issues/968
"""

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
        columns=["url", "fetch_time", "content_mime_type", "warc_filename", "warc_record_offset", "warc_record_length", "text", "token_count", "char_count"],
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
