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
from marin.crawl.deduplicate_outlinks import DeduplicateOutlinksConfig, deduplicate_and_shuffle_with_bq_driver
from marin.crawl.deduplicate_outlinks_against_cc import (
    DeduplicateOutlinksAgainstCCConfig,
    deduplicate_outlinks_against_cc_driver,
)
from marin.crawl.fetch_links import FetchLinksConfig, process_shard_links
from marin.crawl.get_finemath_crawl_yield import GetCrawlYieldConfig
from marin.crawl.get_outlinks_from_html import OutlinksExtractionConfig, get_outlinks_from_html
from marin.crawl.minhash.deduplicate_against_index import (
    MinhashDeduplicateAgainstIndexConfig,
    minhash_deduplicate_against_index_driver,
)
from marin.crawl.sample_from_unique_outlinks import OutlinksSamplingConfig, sample_outlinks
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
    deduplicate_from_cc: bool = False,
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
            prefix=config.source_name.split("/")[-1],
        ),
    )

    if deduplicate_from_cc:
        outlinks_cc_deduplicated_2013_2018 = ExecutorStep(
            name=f"crawl/{config.source_name}/outlinks/{config.source_name}-cc-deduplicated-2013-2018",
            fn=deduplicate_outlinks_against_cc_driver,
            config=DeduplicateOutlinksAgainstCCConfig(
                input_pattern=output_path_of(extracted_outlinks, input_pattern),
                bloom_filter_path=BLOOM_FILTER_2013_2018,
                output_path=this_output_path(),
                shards_per_batch=100,
            ),
        )

        # Outlinks deduplicated against 2013-2018 CC bloom filter: gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-cc-deduplicated/1000_links.jsonl.gz
        outlinks_cc_deduplicated_2019_2024 = ExecutorStep(
            name=f"crawl/{config.source_name}/outlinks/{config.source_name}-cc-deduplicated-2019-2024",
            fn=deduplicate_outlinks_against_cc_driver,
            config=DeduplicateOutlinksAgainstCCConfig(
                input_pattern=output_path_of(outlinks_cc_deduplicated_2013_2018, input_pattern),
                bloom_filter_path=BLOOM_FILTER_2019_2024,
                output_path=this_output_path(),
                shards_per_batch=100,
            ),
        )

    inp_pat = None
    if not deduplicate_from_cc:
        inp_pat = output_path_of(extracted_outlinks, input_pattern)
    else:
        inp_pat = output_path_of(outlinks_cc_deduplicated_2013_2018, input_pattern)

    outlinks_deduplicated = ExecutorStep(
        name=f"crawl/{config.source_name}/outlinks/{config.source_name}-deduplicated",
        fn=deduplicate_and_shuffle_with_bq_driver,
        config=DeduplicateOutlinksConfig(
            gcs_input_pattern=inp_pat,
            gcs_output_prefix=this_output_path("links"),
            bq_table_id=f"{config.source_name.split('/')[-1]}",
        ),
    )

    sampled_outlinks = ExecutorStep(
        name=f"crawl/{config.source_name}/outlinks/{config.source_name}-unique",
        fn=sample_outlinks,
        config=OutlinksSamplingConfig(
            input_pattern=output_path_of(outlinks_deduplicated, input_pattern),
            num_to_sample=10000000,
            shard_size=10000,
            output_prefix=this_output_path("links"),
        ),
    )

    # Fetched outlinks: gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-10M/links.0_robots.json.gz
    links_fetched_parquet = ExecutorStep(
        name=f"crawl/{config.source_name}/fetched_outlinks/{config.source_name}",
        fn=process_shard_links,
        config=FetchLinksConfig(
            urls_input_directory=output_path_of(sampled_outlinks),
            output_path=this_output_path(),
            threads_per_shard=160,
            max_concurrent_shards=40,
        ),
    )

    # Fetched outlinks in WARC format: gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-10M/links.0.warc.gz
    links_fetched_warc = ExecutorStep(
        name=f"crawl/{config.source_name}/fetched_outlinks/{config.source_name}-warc",
        fn=convert_shards_to_warc,
        config=ConvertResponsesToWARCConfig(
            input_directory=output_path_of(links_fetched_parquet),
            output_path=this_output_path(),
        ),
    )

    # Yield outlinks: gs://marin-us-central2/scratch/nfliu/text/fineweb-edu-10M/links.0_extracted_text.parquet
    links_fetched_warc_yield = ExecutorStep(
        name=f"crawl/{config.source_name}/text",
        fn=yield_fn,
        config=GetCrawlYieldConfig(
            urls_input_directory=output_path_of(sampled_outlinks),
            crawl_input_directory=output_path_of(links_fetched_warc),
            data_source=config.source_name.split("/")[-1],
            text_output_directory=this_output_path(),
            statistics_output_path=output_path_of(links_fetched_warc, "yield_statistics.json.gz"),
        ),
    )

    # Passing paths: gs://marin-us-central2/scratch/nfliu/fineweb_edu_10M_passing_paths.txt
    links_minhash_deduplicated = ExecutorStep(
        name=f"crawl/{config.source_name}/minhash",
        fn=minhash_deduplicate_against_index_driver,
        config=MinhashDeduplicateAgainstIndexConfig(
            index_path=this_output_path("index"),
            input_patterns=[output_path_of(links_fetched_warc_yield, "*_text_and_scores.passing.parquet")],
            parquets_paths_file=this_output_path(f"{config.source_name}-passing_paths.txt"),
            minhash_base_path=this_output_path(f"{config.source_name}_passing_minhash_against_{config.source_name}"),
            minhash_logs_path=this_output_path(
                f"{config.source_name}_passing_minhash_against_{config.source_name}_logs"
            ),
        ),
    )

    steps = [
        extracted_html,
        extracted_outlinks,
        outlinks_deduplicated,
        sampled_outlinks,
        links_fetched_parquet,
        links_fetched_warc,
        links_fetched_warc_yield,
        links_minhash_deduplicated,
    ]

    if deduplicate_from_cc:
        steps.extend(
            [
                outlinks_cc_deduplicated_2013_2018,
                outlinks_cc_deduplicated_2019_2024,
            ]
        )

    return steps
