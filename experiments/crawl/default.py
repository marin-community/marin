from typing import Callable

from marin.crawl.common.convert_to_html import process_parquet
from marin.crawl.common.schemas import HtmlExtractionConfig
from marin.crawl.fetch_links import FetchLinksConfig, process_shard_links
from marin.crawl.deduplicate_outlinks_against_cc import DeduplicateOutlinksAgainstCCConfig, deduplicate_outlinks_against_cc_driver
from marin.crawl.get_fineweb_edu_crawl_yield import GetCrawlYieldConfig
from marin.crawl.get_outlinks_from_html import OutlinksExtractionConfig, get_outlinks_from_html
from marin.execution.executor import ExecutorStep, output_path_of, this_output_path
from marin.crawl.convert_responses_parquet_to_warc import ConvertResponsesToWARCConfig, convert_shards_to_warc
from marin.crawl.minhash.deduplicate_against_index import MinhashDeduplicateAgainstIndexConfig, minhash_deduplicate_against_index_driver

BLOOM_FILTER_2013_2018 = "gs://marin-us-central2/gcsfuse_mount/nfliu/deduplicate_outlinks/cc-urls-partitioned_2013_2018.bloom"
BLOOM_FILTER_2019_2024 = "gs://marin-us-central2/gcsfuse_mount/nfliu/deduplicate_outlinks/cc-urls-partitioned_2019_2024.bloom"

def default_crawl(
    config: HtmlExtractionConfig,
    yield_fn: Callable,
) -> list[ExecutorStep]:
    extracted_html = ExecutorStep(
        name=f"crawl/{config.source_name}/html",
        fn=process_parquet,
        config=config,
    )

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
            input_pattern=output_path_of(extracted_outlinks, "CC-MAIN-*/*_links.jsonl.gz"),
            bloom_filter_path=BLOOM_FILTER_2013_2018,
            output_path=this_output_path(),
            shards_per_batch=100,
        ),
    )

    outlinks_deduplicated = ExecutorStep(
        name=f"crawl/{config.source_name}/outlinks/{config.source_name}-deduplicated",
        fn=deduplicate_outlinks_against_cc_driver,
        config=DeduplicateOutlinksAgainstCCConfig(
            input_pattern=output_path_of(outlinks_deduplicated_2013_2018, "CC-MAIN-*/*_links.jsonl.gz"),
            bloom_filter_path=BLOOM_FILTER_2019_2024,
            output_path=this_output_path(),
            shards_per_batch=100,
        ),
    )

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

    links_fetched_warc = ExecutorStep(
        name=f"crawl/{config.source_name}/fetched_outlinks/{config.source_name}-cc-deduplicated",
        fn=convert_shards_to_warc,
        config=ConvertResponsesToWARCConfig(
            input_directory=output_path_of(links_fetched_parquet),
            output_path=this_output_path(),
        ),
    )

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
        override_output_path=f"crawl/{config.source_name}"
    )

    links_minhash_deduplicated = ExecutorStep(
        name=f"crawl/{config.source_name}",
        fn=minhash_deduplicate_against_index_driver,
        config=MinhashDeduplicateAgainstIndexConfig(
            index_path=this_output_path(f"{config.source_name}_minhash_index/index"),
            input_patterns=[output_path_of(links_fetched_warc_yield, f"{config.source_name}-cc-deduplicated/*_text_and_scores.passing.parquet")],
            parquets_paths_file=this_output_path(f"{config.source_name}-cc-deduplicated-passing_paths.txt"),
            minhash_base_path=this_output_path(f"minhash/{config.source_name}_cc_deduplicated_passing_minhash_against_{config.source_name}"),
            minhash_logs_path=this_output_path(f"minhash/{config.source_name}_cc_deduplicated_passing_minhash_against_{config.source_name}_logs"),
        ),
        override_output_path=f"crawl/{config.source_name}"
    )

    return [
        extracted_html, 
        extracted_outlinks, 
        outlinks_deduplicated_2013_2018, 
        outlinks_deduplicated, 
        links_fetched_parquet, 
        links_fetched_warc,
        links_fetched_warc_yield,
        links_minhash_deduplicated
    ]