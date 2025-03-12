from marin.crawl.get_open_web_math_crawl_yield import GetCrawlYieldConfig, main as get_open_web_math_crawl_yield_main
from marin.crawl.fetch_links import FetchLinksConfig, main as fetch_links_main
from marin.crawl.deduplicate_outlinks_against_cc import DeduplicateOutlinksAgainstCCConfig, deduplicate_outlinks_against_cc_driver
from marin.crawl.convert_responses_parquet_to_warc import ConvertResponsesToWARCConfig, main as convert_responses_parquet_to_warc_main
from marin.crawl.get_outlinks_from_html import OutlinksExtractionConfig, get_outlinks_from_html
from marin.crawl.minhash.deduplicate_against_index import MinhashDeduplicateAgainstIndexConfig, minhash_deduplicate_against_index_driver
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.crawl.open_web_math.convert_open_web_math_to_html import ParquetOpenWebMathConfig, process_open_web_math

BLOOM_FILTER_2013_2018 = "gs://marin-us-central2/gcsfuse_mount/nfliu/deduplicate_outlinks/cc-urls-partitioned_2013_2018.bloom"
BLOOM_FILTER_2019_2024 = "gs://marin-us-central2/gcsfuse_mount/nfliu/deduplicate_outlinks/cc-urls-partitioned_2019_2024.bloom"

open_web_math_html = ExecutorStep(
    name="documents/open-web-math/html-test",
    fn=process_open_web_math,
    config=ParquetOpenWebMathConfig(
        input_path="gs://marin-us-central2/raw/open-web-math-fde8ef8/fde8ef8/huggingface.co/datasets/open-web-math/open-web-math/resolve/fde8ef8/data/",
        html_output_path=this_output_path(),
    ),
)

open_web_math_outlinks = ExecutorStep(
    name="scratch/nfliu-test/outlinks/open-web-math",
    fn=get_outlinks_from_html,
    config=OutlinksExtractionConfig(
        input_path=output_path_of(open_web_math_html),
        output_path=this_output_path(),
        prefix="openwebmath",
    ),
)

open_web_math_outlinks_deduplicated_2013_2018 = ExecutorStep(
    name="scratch/nfliu-test/outlinks/open-web-math-cc-deduplicated-2013-2018",
    fn=deduplicate_outlinks_against_cc_driver,
    config=DeduplicateOutlinksAgainstCCConfig(
        input_pattern=output_path_of(open_web_math_outlinks, "*_links.jsonl.gz"),
        bloom_filter_path=BLOOM_FILTER_2013_2018,
        output_path=this_output_path(),
        shards_per_batch=100,
    ),
)

open_web_math_outlinks_deduplicated = ExecutorStep(
    name="scratch/nfliu-test/outlinks/open-web-math-cc-deduplicated",
    fn=deduplicate_outlinks_against_cc_driver,
    config=DeduplicateOutlinksAgainstCCConfig(
        input_pattern=output_path_of(open_web_math_outlinks_deduplicated_2013_2018, "*_links.jsonl.gz"),
        bloom_filter_path=BLOOM_FILTER_2019_2024,
        output_path=this_output_path(),
        shards_per_batch=100,
    ),
)

open_web_math_links_fetched_parquet = ExecutorStep(
    name="scratch/nfliu-test/fetched_outlinks/open-web-math-10M-cc-deduplicated/",
    fn=fetch_links_main,
    config=FetchLinksConfig(
        urls_input_directory="gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-10M-cc-deduplicated/",
        output_path=this_output_path(),
        threads_per_shard=160,
        max_concurrent_shards=40,
    ),
)

open_web_math_links_fetched_warc = ExecutorStep(
    name="scratch/nfliu-test/fetched_outlinks/open-web-math-10M-cc-deduplicated",
    fn=convert_responses_parquet_to_warc_main,
    config=ConvertResponsesToWARCConfig(
        input_directory=output_path_of(open_web_math_links_fetched_parquet),
        output_path=this_output_path(),
    ),
)

open_web_math_links_fetched_warc_yield = ExecutorStep(
    name="scratch/nfliu-test",
    fn=get_open_web_math_crawl_yield_main,
    config=GetCrawlYieldConfig(
        urls_input_directory=this_output_path("outlinks/open-web-math-10M-cc-deduplicated/"),
        crawl_input_directory=output_path_of(open_web_math_links_fetched_warc),
        data_source="open-web-math-10M-cc-deduplicated",
        text_output_directory=this_output_path("text/open-web-math-10M-cc-deduplicated/"),
        statistics_output_path=this_output_path("urls_and_scores/open-web-math-10M-cc-deduplicated/"),
        urls_and_scores_output_directory=output_path_of(open_web_math_links_fetched_parquet, "yield_statistics.json.gz"),
    ),
)

open_web_math_links_minhash_deduplicated = ExecutorStep(
    name="scratch/nfliu-test/minhash",
    fn=minhash_deduplicate_against_index_driver,
    config=MinhashDeduplicateAgainstIndexConfig(
        index_path=this_output_path("open_web_math_minhash_index/index"),
        input_patterns=[output_path_of(open_web_math_links_fetched_warc_yield, "text/open-web-math-10M/*_text_and_scores.passing.parquet")],
        parquets_paths_file=output_path_of(open_web_math_links_fetched_warc_yield, "open-web-math-10M-cc-deduplicated-passing_paths.txt"),
        minhash_base_path=this_output_path("open_web_math_10M_cc_deduplicated_passing_minhash_against_open_web_math"),
        minhash_logs_path=this_output_path("open_web_math_10M_cc_deduplicated_passing_minhash_against_open_web_math_logs"),
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[
            open_web_math_html,
            open_web_math_outlinks,
            open_web_math_outlinks_deduplicated_2013_2018,
            open_web_math_outlinks_deduplicated,
            open_web_math_links_fetched_parquet,
            open_web_math_links_fetched_warc,
            open_web_math_links_fetched_warc_yield,
            open_web_math_links_minhash_deduplicated,
        ]
    )
