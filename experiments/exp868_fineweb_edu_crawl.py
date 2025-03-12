from marin.crawl.get_fineweb_edu_crawl_yield import GetCrawlYieldConfig, main as get_fineweb_edu_crawl_yield_main
from marin.crawl.fetch_links import FetchLinksConfig, main as fetch_links_main
from marin.crawl.deduplicate_outlinks_against_cc import DeduplicateOutlinksAgainstCCConfig, deduplicate_outlinks_against_cc_driver
from marin.crawl.convert_responses_parquet_to_warc import ConvertResponsesParquetToWarcConfig, main as convert_responses_parquet_to_warc_main
from marin.crawl.get_outlinks_from_html import OutlinksExtractionConfig, get_outlinks_from_html
from marin.crawl.minhash.deduplicate_against_index import MinhashDeduplicateAgainstIndexConfig, minhash_deduplicate_against_index_driver
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.crawl.fineweb_edu.convert_fineweb_edu_to_html import ParquetFineWebEduConfig, process_fineweb_edu

BLOOM_FILTER_2013_2018 = "gs://marin-us-central2/gcsfuse_mount/nfliu/deduplicate_outlinks/cc-urls-partitioned_2013_2018.bloom"
BLOOM_FILTER_2019_2024 = "gs://marin-us-central2/gcsfuse_mount/nfliu/deduplicate_outlinks/cc-urls-partitioned_2019_2024.bloom"

fineweb_edu_html = ExecutorStep(
    name="documents/fineweb-edu/html",
    fn=process_fineweb_edu,
    config=ParquetFineWebEduConfig(
        input_path="gs://marin-us-central2/raw/fineweb-edu",
        output_path=this_output_path(),
    ),
)

fineweb_edu_outlinks = ExecutorStep(
    name="scratch/nfliu/outlinks/fineweb-edu",
    fn=get_outlinks_from_html,
    config=OutlinksExtractionConfig(
        input_path=output_path_of(fineweb_edu_html),
        output_path=this_output_path(),
        prefix="fineweb_edu",
    ),
)

fineweb_edu_outlinks_deduplicated_2013_2018 = ExecutorStep(
    name="scratch/nfliu/outlinks/fineweb-edu-cc-deduplicated-2013-2018",
    fn=deduplicate_outlinks_against_cc_driver,
    config=DeduplicateOutlinksAgainstCCConfig(
        input_pattern=output_path_of(fineweb_edu_outlinks, "CC-MAIN-*/*_links.jsonl.gz"),
        bloom_filter_path=BLOOM_FILTER_2013_2018,
        output_path=this_output_path(),
        shards_per_batch=100,
    ),
)

fineweb_edu_outlinks_deduplicated = ExecutorStep(
    name="scratch/nfliu/outlinks/fineweb-edu-cc-deduplicated",
    fn=deduplicate_outlinks_against_cc_driver,
    config=DeduplicateOutlinksAgainstCCConfig(
        input_pattern=output_path_of(fineweb_edu_outlinks_deduplicated_2013_2018, "CC-MAIN-*/*_links.jsonl.gz"),
        bloom_filter_path=BLOOM_FILTER_2019_2024,
        output_path=this_output_path(),
        shards_per_batch=100,
    ),
)

fineweb_edu_links_fetched_parquet = ExecutorStep(
    name="scratch/nfliu/fetched_outlinks/fineweb-edu-10M/",
    fn=fetch_links_main,
    config=FetchLinksConfig(
        urls_input_directory="gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-10M/",
        output_path=this_output_path(),
        threads_per_shard=160,
        max_concurrent_shards=40,
    ),
)

fineweb_edu_links_fetched_warc = ExecutorStep(
    name="scratch/nfliu/fetched_outlinks/fineweb-edu-10M",
    fn=convert_responses_parquet_to_warc_main,
    config=ConvertResponsesParquetToWarcConfig(
        input_directory=output_path_of(fineweb_edu_links_fetched_parquet),
        output_path=this_output_path(),
    ),
)

fineweb_edu_links_fetched_warc_yield = ExecutorStep(
    name="scratch/nfliu",
    fn=get_fineweb_edu_crawl_yield_main,
    config=GetCrawlYieldConfig(
        urls_input_directory=this_output_path("outlinks/fineweb-edu-10M/"),
        crawl_input_directory=output_path_of(fineweb_edu_links_fetched_warc),
        data_source="fineweb-edu",
        text_output_directory=this_output_path("text/fineweb-edu-10M/"),
        statistics_output_path=this_output_path("urls_and_scores/fineweb-edu-10M/"),
        urls_and_scores_output_directory=output_path_of(fineweb_edu_links_fetched_warc, "yield_statistics.json.gz"),
    ),
)

fineweb_edu_links_minhash_deduplicated = ExecutorStep(
    name="scratch/nfliu/minhash",
    fn=minhash_deduplicate_against_index_driver,
    config=MinhashDeduplicateAgainstIndexConfig(
        index_path=this_output_path("fineweb_edu_minhash_index/index"),
        input_patterns=[output_path_of(fineweb_edu_links_fetched_warc_yield, "*.json.gz")],
        parquets_paths_file=this_output_path("fineweb_edu_minhash_index/parquets_paths.txt"),
        minhash_base_path=this_output_path("fineweb_edu_minhash_index/minhash_base"),
        minhash_logs_path=this_output_path("fineweb_edu_minhash_index/minhash_logs"),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[
            fineweb_edu_html,
            fineweb_edu_outlinks,
            fineweb_edu_outlinks_deduplicated_2013_2018,
            fineweb_edu_outlinks_deduplicated,
            fineweb_edu_links_fetched_parquet,
            fineweb_edu_links_fetched_warc,
            fineweb_edu_links_fetched_warc_yield,
            fineweb_edu_links_minhash_deduplicated,
        ]
    )

"""
python marin/run/ray_run.py --no_wait --pip_deps '--find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html,resiliparse_dom @ git+https://github.com/nelson-liu/chatnoir-resiliparse@58247de82b4d881223435113f1a07a86ad66494c#egg=resiliparse_dom&subdirectory=resiliparse_dom,courlan,w3lib,cchardet,beautifulsoup4,lxml,rbloom-gcs==1.5.6,orjson,fastparquet,warcio,w3lib,trafilatura,jax[tpu],flax,transformers,requests,warcio[all],resiliparse,datatrove[processing] @ git+https://github.com/nelson-liu/datatrove@ray_executor_dedup_logging,spacy,cupy-cuda12x==13.3.0,datatrove[io] @ git+https://github.com/nelson-liu/datatrove@ray_executor_dedup_logging,spacy,cupy-cuda12x==13.3.0,orjson,scipy==1.13.1' -- python experiments/exp868_open_web_math_crawl.py
"""