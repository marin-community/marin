"""
Default crawl pipeline for extracting HTML, getting outlinks, deduplicating outlinks against CC,
and then fetching the outlinks. The complete pipeline is as follows:
    - Extract HTML from Dataset Parquet files
    - Get outlinks from HTML
    - Deduplicate outlinks against CC bloom filter for 2013-2018
    - Deduplicate outlinks against CC bloom filter for 2019-2024
    - Fetch outlinks
    - Convert outlinks to WARC
    - Yield outlinks
    - Minhash deduplicate outlinks
"""

from collections.abc import Callable

from marin.crawl.common.convert_to_html import process_parquet
from marin.crawl.common.schemas import HtmlExtractionConfig
from marin.crawl.convert_responses_parquet_to_warc import ConvertResponsesToWARCConfig, convert_shards_to_warc
from marin.crawl.deduplicate_outlinks_against_cc import (
    DeduplicateOutlinksAgainstCCConfig,
    deduplicate_outlinks_against_cc_driver,
)
from marin.crawl.fetch_links import FetchLinksConfig, process_shard_links
from marin.crawl.get_fineweb_edu_crawl_yield import GetCrawlYieldConfig
from marin.crawl.get_outlinks_from_html import OutlinksExtractionConfig, get_outlinks_from_html
from marin.crawl.minhash.deduplicate_against_index import (
    MinhashDeduplicateAgainstIndexConfig,
    minhash_deduplicate_against_index_driver,
)
from marin.execution.executor import ExecutorStep, output_path_of, this_output_path

# path to bloom filter for links. The year range corresponds to time period designations for CC
BLOOM_FILTER_2013_2018 = (
    "gs://marin-us-central2/gcsfuse_mount/nfliu/deduplicate_outlinks/cc-urls-partitioned_2013_2018.bloom"
)
BLOOM_FILTER_2019_2024 = (
    "gs://marin-us-central2/gcsfuse_mount/nfliu/deduplicate_outlinks/cc-urls-partitioned_2019_2024.bloom"
)


def default_crawl(
    config: HtmlExtractionConfig,
    yield_fn: Callable,
    input_pattern: str = "*_links.jsonl.gz",
) -> list[ExecutorStep]:
    """
    Crawls over a given parquet to extract the outlinks and populate a new dataset based on them.
    Args:
        config (HtmlExtractionConfig): Configuration for extracting HTML from Dataset Parquet files
        yield_fn (Callable): Function to apply processing to the crawled content
        input_pattern (str, optional): Pattern to match input files for deduplicating outlinks.
                                      Defaults to "*_links.jsonl.gz".
    Returns:
        steps[list[ExecutorStep]]: List of steps in the crawl pipeline
    """

    # Extracted HTML: gs://marin-us-central2/documents/open-web-math-fde8ef8/html/openwebmath_0.jsonl.gz
    extracted_html = ExecutorStep(
        name=f"crawl/{config.source_name}/html",
        fn=process_parquet,
        config=config,
    )

    # Extracted outlinks: gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8/1000_links.jsonl.gz
    extracted_outlinks = ExecutorStep(
        name=f"crawl/{config.source_name}/outlinks/raw_{config.source_name}",
        fn=get_outlinks_from_html,
        config=OutlinksExtractionConfig(
            html_input_path=output_path_of(extracted_html),
            outlinks_output_path=this_output_path(),
            prefix=config.source_name,
        ),
    )

    outlinks_deduplicated_2013_2018 = ExecutorStep(
        name=f"crawl/{config.source_name}/outlinks/{config.source_name}-deduplicated-2013-2018",
        fn=deduplicate_outlinks_against_cc_driver,
        config=DeduplicateOutlinksAgainstCCConfig(
            input_pattern=output_path_of(extracted_outlinks, input_pattern),
            bloom_filter_path=BLOOM_FILTER_2013_2018,
            output_path=this_output_path(),
            shards_per_batch=100,
        ),
    )

    # Outlinks deduplicated against 2013-2018 CC bloom filter: gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-cc-deduplicated/1000_links.jsonl.gz
    outlinks_deduplicated = ExecutorStep(
        name=f"crawl/{config.source_name}/outlinks/{config.source_name}-deduplicated",
        fn=deduplicate_outlinks_against_cc_driver,
        config=DeduplicateOutlinksAgainstCCConfig(
            input_pattern=output_path_of(outlinks_deduplicated_2013_2018, input_pattern),
            bloom_filter_path=BLOOM_FILTER_2019_2024,
            output_path=this_output_path(),
            shards_per_batch=100,
        ),
    )

    # Fetched outlinks: gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-10M/links.0_robots.json.gz
    links_fetched_parquet = ExecutorStep(
        name=f"crawl/{config.source_name}/fetched_outlinks/{config.source_name}",
        fn=process_shard_links,
        config=FetchLinksConfig(
            urls_input_directory=output_path_of(outlinks_deduplicated),
            output_path=this_output_path(),
            threads_per_shard=160,
            max_concurrent_shards=40,
        ),
    )

    # Fetched outlinks in WARC format: gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-10M/links.0.warc.gz
    links_fetched_warc = ExecutorStep(
        name=f"crawl/{config.source_name}/fetched_outlinks/{config.source_name}-cc-deduplicated",
        fn=convert_shards_to_warc,
        config=ConvertResponsesToWARCConfig(
            input_directory=output_path_of(links_fetched_parquet),
            output_path=this_output_path(),
        ),
    )

    # Yield outlinks: gs://marin-us-central2/scratch/nfliu/text/fineweb-edu-10M/links.0_extracted_text.parquet
    links_fetched_warc_yield = ExecutorStep(
        name=f"crawl/{config.source_name}",
        fn=yield_fn,
        config=GetCrawlYieldConfig(
            urls_input_directory=output_path_of(outlinks_deduplicated),
            crawl_input_directory=output_path_of(links_fetched_warc),
            data_source=config.source_name,
            text_output_directory=this_output_path(f"text/{config.source_name}-cc-deduplicated"),
            statistics_output_path=output_path_of(links_fetched_warc, "yield_statistics.json.gz"),
            urls_and_scores_output_directory=this_output_path(f"urls_and_scores/{config.source_name}-cc-deduplicated"),
        ),
        override_output_path=f"crawl/{config.source_name}",
    )

    # Passing paths: gs://marin-us-central2/scratch/nfliu/fineweb_edu_10M_passing_paths.txt
    links_minhash_deduplicated = ExecutorStep(
        name=f"crawl/{config.source_name}",
        fn=minhash_deduplicate_against_index_driver,
        config=MinhashDeduplicateAgainstIndexConfig(
            index_path=this_output_path(f"{config.source_name}_minhash_index/index"),
            input_patterns=[
                output_path_of(
                    links_fetched_warc_yield, f"{config.source_name}-cc-deduplicated/*_text_and_scores.passing.parquet"
                )
            ],
            parquets_paths_file=this_output_path(f"{config.source_name}-cc-deduplicated-passing_paths.txt"),
            minhash_base_path=this_output_path(
                f"minhash/{config.source_name}_cc_deduplicated_passing_minhash_against_{config.source_name}"
            ),
            minhash_logs_path=this_output_path(
                f"minhash/{config.source_name}_cc_deduplicated_passing_minhash_against_{config.source_name}_logs"
            ),
        ),
        override_output_path=f"crawl/{config.source_name}",
    )

    return [
        extracted_html,
        extracted_outlinks,
        outlinks_deduplicated_2013_2018,
        outlinks_deduplicated,
        links_fetched_parquet,
        links_fetched_warc,
        links_fetched_warc_yield,
        links_minhash_deduplicated,
    ]
