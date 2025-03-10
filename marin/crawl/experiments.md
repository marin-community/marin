# Experiment Log

## Summary of the Overall Workflow

1. Download HTML content for Open-Web-Math and FineWeb-Edu examples.

2. Construct the crawl frontier for each dataset by extracting outlinks from the
   HTML.

3. Randomly sample links (e.g., 10M) from the crawl frontier.

4. Fetch sampled URLs at scale; store responses in Parquet format.

5. Convert Parquet responses to WARC files.

6. Process the fetched WARC-formatted responses with each dataset's original
   filtering pipeline. Collect the documents that pass the filtering pipeline
   and count the number of tokens.

7. Use MinHash deduplication to remove near-duplicates in the fetched pages that
   also occur in the original dataset.

8. Count tokens after deduplication to measure the final number of crawled tokens.

## Extracting HTML for all datasets examples

1. Fetch the HTML for each open-web-math example:

```
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- \
    python marin/crawl/open-web-math/convert_open_web_math_to_html.py \
    --input_path gs://marin-us-central2/raw/open-web-math-fde8ef8/fde8ef8/huggingface.co/datasets/open-web-math/open-web-math/resolve/fde8ef8/data/ \
    --html_output_path gs://marin-us-central2/documents/open-web-math-fde8ef8/html/
```

2. Fetch the HTML for each FineWeb-Edu:

```
for fineweb_edu_dump_path in $(gcloud storage ls gs://marin-us-central2/raw/fineweb-edu); do
    dump_name=$(basename -- ${fineweb_edu_dump_path})
    ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- \
    python marin/crawl/fineweb-edu/convert_fineweb_edu_to_html.py \
    --input_path ${fineweb_edu_dump_path} \
    --html_output_path gs://marin-us-central2/documents/fineweb-edu/html/${dump_name}/
done
```

## Get outlinks from HTML pages

1. Extracting outlinks from Dolma-format open-web-math examples containing HTML:

```
python marin/run/ray_run.py \
    --pip_deps 'resiliparse_dom @ git+https://github.com/nelson-liu/chatnoir-resiliparse@58247de82b4d881223435113f1a07a86ad66494c#egg=resiliparse_dom&subdirectory=resiliparse_dom,courlan,w3lib,cchardet,beautifulsoup4,lxml' \
    --no_wait -- \
    python marin/crawl/get_outlinks_from_html.py \
    --html_input_path gs://marin-us-central2/documents/open-web-math-fde8ef8/html/ \
    --prefix openwebmath \
    --outlinks_output_path gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8/
```

2. Extracting outlinks from Dolma-format FineWeb-Edu examples containing HTML:

```
for fineweb_edu_dump_html_path in $(gcloud storage ls gs://marin-us-central2/documents/fineweb-edu/html); do
    dump_name=$(basename -- ${fineweb_edu_dump_html_path})

    python marin/run/ray_run.py \
        --pip_deps 'resiliparse_dom @ git+https://github.com/nelson-liu/chatnoir-resiliparse@58247de82b4d881223435113f1a07a86ad66494c#egg=resiliparse_dom&subdirectory=resiliparse_dom,courlan,w3lib,cchardet,beautifulsoup4,lxml' \
        --no_wait -- \
        python marin/crawl/get_outlinks_from_html.py \
        --html_input_path ${fineweb_edu_dump_html_path} \
        --prefix fineweb_edu \
        --outlinks_output_path gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu/${dump_name}
done
```

## Optional: remove outlinks that already appear in Common Crawl:

1. Set the GCS authentication key in the environment variable:

```
export AUTHENTICATION_JSON="$(jq -c . data_browser/gcs-key.json)"
```

2. Running on OpenWebMath. We first deduplicate against a bloom filter
   containing the 2013-2018 successful URLs, and then another bloom filter
   containing the 2019-2024 successful URLs. We do this in two batches for
   robustness to pre-emption because the TPU machines are not capable of holding
   both bloom filters in memory at once.

```
# First, deduplicate with 2013-2018 bloom filter
python marin/run/ray_run.py \
    --pip_deps 'rbloom-gcs==1.5.6,orjson' \
    -e "GOOGLE_APPLICATION_CREDENTIALS_JSON" "$AUTHENTICATION_JSON" \
    --no_wait -- \
    python marin/crawl/deduplicate_outlinks_against_cc.py \
        --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8/*_links.jsonl.gz' \
        --bloom_filter_path 'gs://marin-us-central2/gcsfuse_mount/nfliu/deduplicate_outlinks/cc-urls-partitioned_2013_2018.bloom' \
        --output_path gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-cc-deduplicated-2013_2018/ \
        --shards_per_batch 100

# Then, deduplicate with 2019-2024 bloom filter
python marin/run/ray_run.py \
    --pip_deps 'rbloom-gcs==1.5.6,orjson' \
    -e "GOOGLE_APPLICATION_CREDENTIALS_JSON" "$AUTHENTICATION_JSON" \
    --no_wait -- \
    python marin/crawl/deduplicate_outlinks_against_cc.py \
        --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-cc-deduplicated-2013_2018/*_links.jsonl.gz' \
        --bloom_filter_path 'gs://marin-us-central2/gcsfuse_mount/nfliu/deduplicate_outlinks/cc-urls-partitioned_2019_2024.bloom' \
        --output_path gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-cc-deduplicated/ \
        --shards_per_batch 100
```

3. Running on FineWeb-Edu. We first deduplicate against a bloom filter
   containing the 2013-2018 successful URLs, and then another bloom filter
   containing the 2019-2024 successful URLs. We do this in two batches for
   robustness to pre-emption because the TPU machines are not capable of holding
   both bloom filters in memory at once.

```
# First, deduplicate with 2013-2018 bloom filter
python marin/run/ray_run.py \
    --pip_deps 'rbloom-gcs==1.5.6,orjson' \
    -e "GOOGLE_APPLICATION_CREDENTIALS_JSON" "$AUTHENTICATION_JSON" \
    --no_wait -- \
    python marin/crawl/deduplicate_outlinks_against_cc.py \
        --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu/CC-MAIN-*/*_links.jsonl.gz' \
        --bloom_filter_path 'gs://marin-us-central2/gcsfuse_mount/nfliu/deduplicate_outlinks/cc-urls-partitioned_2013_2018.bloom' \
        --output_path gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-cc-deduplicated-2013_2018/ \
        --shards_per_batch 100

# Then, deduplicate with 2019-2024 bloom filter
python marin/run/ray_run.py \
    --pip_deps 'rbloom-gcs==1.5.6,orjson' \
    -e "GOOGLE_APPLICATION_CREDENTIALS_JSON" "$AUTHENTICATION_JSON" \
    --no_wait -- \
    python marin/crawl/deduplicate_outlinks_against_cc.py \
        --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-cc-deduplicated-2013_2018/CC-MAIN-*/*_links.jsonl.gz' \
        --bloom_filter_path 'gs://marin-us-central2/gcsfuse_mount/nfliu/deduplicate_outlinks/cc-urls-partitioned_2019_2024.bloom' \
        --output_path gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-cc-deduplicated/ \
        --shards_per_batch 100
```

## Sample, fetch, and extract text from outlinks

### FineWeb-Edu-10M

This setting looks at 10M unique sampled outlinks from the original FineWeb-Edu
crawl frontier.

1. Fetch links:

```
python marin/run/ray_run.py \
    --pip_deps 'fastparquet' \
    --no_wait -- \
    python marin/crawl/fetch_links.py \
    --urls_input_directory gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-10M/ \
    --output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-10M/ \
    --threads_per_shard 160 \
    --max_concurrent_shards 40
```

2. Convert fetched parquet to WARC:

```
python marin/run/ray_run.py \
    --pip_deps 'warcio' \
    --no_wait -- \
    python marin/crawl/convert_responses_parquet_to_warc.py \
    --input_directory gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-10M/ \
    --output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-10M/
```

3. Run full FineWeb-Edu pipeline to get yield from the fetched WARC pages:

```
python marin/run/ray_run.py \
    --pip_deps '--find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html,w3lib,trafilatura,jax[tpu],flax,transformers,requests,warcio[all],resiliparse,datatrove[processing] @ git+https://github.com/nelson-liu/datatrove@ray_executor_dedup_logging,spacy,cupy-cuda12x==13.3.0' \
    --no_wait -- \
    python marin/crawl/get_fineweb_edu_crawl_yield.py \
    --urls_input_directory gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-10M/ \
    --crawl_input_directory gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-10M/ \
    --data_source fineweb-edu-10M \
    --text_output_directory gs://marin-us-central2/scratch/nfliu/text/fineweb-edu-10M/ \
    --urls_and_scores_output_directory gs://marin-us-central2/scratch/nfliu/urls_and_scores/fineweb-edu-10M/ \
    --statistics_output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-10M/yield_statistics.json.gz
```

```
Total URLs: 10000000
Total URLs fetched: 5,600,465
Total URLs passing: 381,453
```

4. Count the number of tokens in the fetched pages that pass the filtering
   pipeline:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/count_tokens.py \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/text/fineweb-edu-10M/*.passing.parquet"]' \
    --output_path "gs://marin-us-central2/scratch/nfliu/count_tokens/fineweb-edu-10M_passing/"
```

Results

```
Total number of tokens: 535,294,684
Total number of documents: 381,453
```

5. Take the fetched pages that pass the filtering pipeline and run MinHash
   deduplication against FineWeb-Edu:

```
python marin/run/ray_run.py \
    --pip_deps 'datatrove[io] @ git+https://github.com/nelson-liu/datatrove@ray_executor_dedup_logging,spacy,cupy-cuda12x==13.3.0,orjson,scipy==1.13.1' \
    --no_wait -- \
    python marin/crawl/minhash/deduplicate_against_index.py \
    --index_path 'gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_minhash_index/index' \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/text/fineweb-edu-10M/*_text_and_scores.passing.parquet"]' \
    --parquets_paths_file 'gs://marin-us-central2/scratch/nfliu/fineweb_edu_10M_passing_paths.txt' \
    --minhash_base_path 'gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_10M_passing_minhash_against_fineweb_edu' \
    --minhash_logs_path 'gs://marin-us-central2/scratch/nfliu/minhash/logs/fineweb_edu_10M_passing_minhash_against_fineweb_edu_logs'

# Move the deduplicated content
gcloud storage mv gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_10M_passing_minhash_against_fineweb_edu/deduplicated_output/* gs://marin-us-central2/scratch/nfliu/text/fineweb_edu_10M_passing_minhash_against_fineweb_edu/
# Remove the logs and intermediate output
gcloud storage rm --recursive gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_10M_passing_minhash_against_fineweb_edu
gcloud storage rm --recursive gs://marin-us-central2/scratch/nfliu/minhash/logs/fineweb_edu_10M_passing_minhash_against_fineweb_edu_logs
```

6. Count the number of tokens in the fetched pages that pass the filtering
   pipeline after minhash deduplication:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/count_tokens.py \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/text/fineweb_edu_10M_passing_minhash_against_fineweb_edu/*.jsonl.gz"]' \
    --output_path "gs://marin-us-central2/scratch/nfliu/count_tokens/fineweb_edu_10M_passing_minhash_against_fineweb_edu/"
```

Results

```
Total number of tokens: 78,752,251
Total number of documents: 52,727
```

### open-web-math-10M

This setting looks at 10M unique sampled outlinks from the original open-web-math
crawl frontier.

1. Fetch links:

```
python marin/run/ray_run.py \
    --pip_deps 'fastparquet' \
    --no_wait -- \
    python marin/crawl/fetch_links.py \
    --urls_input_directory gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-10M/ \
    --output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/open-web-math-fde8ef8-10M/ \
    --threads_per_shard 160 \
    --max_concurrent_shards 40
```

2. Convert fetched parquet to WARC:

```
python marin/run/ray_run.py \
    --pip_deps 'warcio' \
    --no_wait -- \
    python marin/crawl/convert_responses_parquet_to_warc.py \
    --input_directory gs://marin-us-central2/scratch/nfliu/fetched_outlinks/open-web-math-fde8ef8-10M/ \
    --output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/open-web-math-fde8ef8-10M/
```

3. Run full open-web-math pipeline to get yield from the fetched WARC pages:

```
python marin/run/ray_run.py \
    --pip_deps 'resiliparse,fasttext,lxml,py-asciimath,tabulate,warcio[all],w3lib,cchardet,kenlm' \
    --no_wait -- \
    python marin/crawl/get_open_web_math_crawl_yield.py \
    --urls_input_directory gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-10M/ \
    --crawl_input_directory gs://marin-us-central2/scratch/nfliu/fetched_outlinks/open-web-math-fde8ef8-10M/ \
    --data_source open-web-math-fde8ef8-10M \
    --text_output_directory gs://marin-us-central2/scratch/nfliu/text/open-web-math-fde8ef8-10M/ \
    --urls_and_scores_output_directory gs://marin-us-central2/scratch/nfliu/urls_and_scores/open-web-math-fde8ef8-10M/ \
    --statistics_output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/open-web-math-fde8ef8-10M/yield_statistics.json.gz
```

Results:

```
Total URLs: 10000000
Total URLs fetched: 4338570
Total URLs passing: 839650
```

4. Count the number of tokens in the fetched pages that pass the filtering
   pipeline:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/count_tokens.py \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/text/open-web-math-fde8ef8-10M/*.passing.parquet"]' \
    --output_path "gs://marin-us-central2/scratch/nfliu/count_tokens/open_web_math_10M_passing/"
```

Results:

```
Total number of tokens: 3,027,494,680
Total number of documents: 839,650
```

5. Take the fetched pages that pass the filtering pipeline and run MinHash
   deduplication against open-web-math:

```
python marin/run/ray_run.py \
    --pip_deps 'datatrove[io] @ git+https://github.com/nelson-liu/datatrove@ray_executor_dedup_logging,spacy,cupy-cuda12x==13.3.0,orjson,scipy==1.13.1' \
    --no_wait -- \
    python marin/crawl/minhash/deduplicate_against_index.py \
    --index_path 'gs://marin-us-central2/scratch/nfliu/minhash/open_web_math_minhash_index/index' \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/text/open-web-math-fde8ef8-10M/*_text_and_scores.passing.parquet"]' \
    --parquets_paths_file 'gs://marin-us-central2/scratch/nfliu/open-web-math-fde8ef8-10M-passing_paths.txt' \
    --minhash_base_path 'gs://marin-us-central2/scratch/nfliu/minhash/open_web_math_10M_passing_minhash_against_open_web_math' \
    --minhash_logs_path 'gs://marin-us-central2/scratch/nfliu/minhash/open_web_math_10M_passing_minhash_against_open_web_math_logs'

# Move the deduplicated content
gcloud storage mv gs://marin-us-central2/scratch/nfliu/minhash/open_web_math_10M_passing_minhash_against_open_web_math/deduplicated_output/* gs://marin-us-central2/scratch/nfliu/text/open_web_math_10M_passing_minhash_against_open_web_math/
# Remove the logs and intermediate output
gcloud storage rm --recursive gs://marin-us-central2/scratch/nfliu/minhash/open_web_math_10M_passing_minhash_against_open_web_math
gcloud storage rm --recursive gs://marin-us-central2/scratch/nfliu/minhash/open_web_math_10M_passing_minhash_against_open_web_math_logs
```

6. Count the number of tokens in the fetched pages that pass the filtering
   pipeline after minhash deduplication:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/count_tokens.py \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/text/open_web_math_10M_passing_minhash_against_open_web_math/*.jsonl.gz"]' \
    --output_path "gs://marin-us-central2/scratch/nfliu/count_tokens/open_web_math_10M_passing_minhash_against_open_web_math/"
```

Results:

```
Total number of tokens: 706,833,188
Total number of documents: 240,784
```

### fineweb-edu-10M-cc-deduplicated

This setting looks at 10M unique sampled outlinks from the a CC-deduplicated
version of the FineWeb-Edu crawl frontier, where links that already occur in CC
are removed.

1. Fetch links:

```
python marin/run/ray_run.py \
    --pip_deps 'fastparquet' \
    --no_wait -- \
    python marin/crawl/fetch_links.py \
    --urls_input_directory gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-10M-cc-deduplicated/ \
    --output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-10M-cc-deduplicated/ \
    --threads_per_shard 160 \
    --max_concurrent_shards 40
```

2. Convert fetched parquet to WARC:

```
python marin/run/ray_run.py \
    --pip_deps 'warcio' \
    --no_wait -- \
    python marin/crawl/convert_responses_parquet_to_warc.py \
    --input_directory gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-10M-cc-deduplicated/ \
    --output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-10M-cc-deduplicated/
```

3. Run full FineWeb-Edu pipeline to get yield from the fetched WARC pages:

```
python marin/run/ray_run.py \
    --pip_deps '--find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html,w3lib,trafilatura,jax[tpu],flax,transformers,requests,warcio[all],resiliparse,datatrove[processing] @ git+https://github.com/nelson-liu/datatrove@ray_executor_dedup_logging,spacy,cupy-cuda12x==13.3.0' \
    --no_wait -- \
    python marin/crawl/get_fineweb_edu_crawl_yield.py \
    --urls_input_directory gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-10M-cc-deduplicated/ \
    --crawl_input_directory gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-10M-cc-deduplicated/ \
    --data_source fineweb-edu-10M-cc-deduplicated \
    --text_output_directory gs://marin-us-central2/scratch/nfliu/text/fineweb-edu-10M-cc-deduplicated/ \
    --urls_and_scores_output_directory gs://marin-us-central2/scratch/nfliu/urls_and_scores/fineweb-edu-10M-cc-deduplicated/ \
    --statistics_output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-10M-cc-deduplicated/yield_statistics.json.gz
```

Results:

```
Total URLs: 10,000,000
Total URLs fetched: 5,286,057
Total URLs passing: 212,817
```

4. Count the number of tokens in the fetched pages that pass the filtering
   pipeline:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/count_tokens.py \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/text/fineweb-edu-10M-cc-deduplicated/*.passing.parquet"]' \
    --output_path "gs://marin-us-central2/scratch/nfliu/count_tokens/fineweb_edu_10M_cc_deduplicated/"
```

Results:

```
Total number of tokens: 314,264,543 (314M)
Total number of documents: 212,817
```

5. Take the fetched pages that pass the filtering pipeline and run MinHash
   deduplication against FineWeb-Edu:

```
python marin/run/ray_run.py \
    --pip_deps 'datatrove[io] @ git+https://github.com/nelson-liu/datatrove@ray_executor_dedup_logging,spacy,cupy-cuda12x==13.3.0,orjson,scipy==1.13.1' \
    --no_wait -- \
    python marin/crawl/minhash/deduplicate_against_index.py \
    --index_path 'gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_minhash_index/index' \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/text/fineweb-edu-10M-cc-deduplicated/*_text_and_scores.passing.parquet"]' \
    --parquets_paths_file 'gs://marin-us-central2/scratch/nfliu/fineweb_edu_10M_cc_deduplicated_passing_paths.txt' \
    --minhash_base_path 'gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_10M_cc_deduplicated_passing_minhash_against_fineweb_edu' \
    --minhash_logs_path 'gs://marin-us-central2/scratch/nfliu/minhash/logs/fineweb_edu_10M_cc_deduplicated_passing_minhash_against_fineweb_edu_logs'

# Move the deduplicated content
gcloud storage mv gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_10M_cc_deduplicated_passing_minhash_against_fineweb_edu/deduplicated_output/* gs://marin-us-central2/scratch/nfliu/text/fineweb_edu_10M_cc_deduplicated_passing_minhash_against_fineweb_edu/
# Remove the logs and intermediate output
gcloud storage rm --recursive gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_10M_cc_deduplicated_passing_minhash_against_fineweb_edu
gcloud storage rm --recursive gs://marin-us-central2/scratch/nfliu/minhash/logs/fineweb_edu_10M_cc_deduplicated_passing_minhash_against_fineweb_edu_logs
```

6. Count the number of tokens in the fetched pages that pass the filtering
   pipeline after minhash deduplication:


```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/count_tokens.py \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/text/fineweb_edu_10M_cc_deduplicated_passing_minhash_against_fineweb_edu/*.jsonl.gz"]' \
    --output_path "gs://marin-us-central2/scratch/nfliu/count_tokens/fineweb_edu_10M_cc_deduplicated_passing_minhash_against_fineweb_edu/"
```

Results:

```
Total number of tokens: 67,532,848
Total number of documents: 48,658
```

### open-web-math-10M-cc-deduplicated

This setting looks at 10M unique sampled outlinks from the a CC-deduplicated
version of the open-web-math crawl frontier, where links that already occur in CC
are removed.

1. Fetch links:

```
python marin/run/ray_run.py \
    --pip_deps 'fastparquet' \
    --no_wait -- \
    python marin/crawl/fetch_links.py \
    --urls_input_directory gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-10M-cc-deduplicated/ \
    --output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/open-web-math-fde8ef8-10M-cc-deduplicated/ \
    --threads_per_shard 160 \
    --max_concurrent_shards 40
```

2. Convert fetched parquet to WARC:

```
python marin/run/ray_run.py \
    --pip_deps 'warcio' \
    --no_wait -- \
    python marin/crawl/convert_responses_parquet_to_warc.py \
    --input_directory gs://marin-us-central2/scratch/nfliu/fetched_outlinks/open-web-math-fde8ef8-10M-cc-deduplicated/ \
    --output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/open-web-math-fde8ef8-10M-cc-deduplicated/
```

3. Run full open-web-math pipeline to get yield from the fetched WARC pages:

```
python marin/run/ray_run.py \
    --pip_deps 'resiliparse,fasttext,lxml,py-asciimath,tabulate,warcio[all],w3lib,cchardet,kenlm' \
    --no_wait -- \
    python marin/crawl/get_open_web_math_crawl_yield.py \
    --urls_input_directory gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-10M-cc-deduplicated/ \
    --crawl_input_directory gs://marin-us-central2/scratch/nfliu/fetched_outlinks/open-web-math-fde8ef8-10M-cc-deduplicated/ \
    --data_source open-web-math-fde8ef8-10M-cc-deduplicated \
    --text_output_directory gs://marin-us-central2/scratch/nfliu/text/open-web-math-fde8ef8-10M-cc-deduplicated/ \
    --urls_and_scores_output_directory gs://marin-us-central2/scratch/nfliu/urls_and_scores/open-web-math-fde8ef8-10M-cc-deduplicated/ \
    --statistics_output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/open-web-math-fde8ef8-10M-cc-deduplicated/yield_statistics.json.gz
```

Results:

```
Total URLs: 10,000,000
Total URLs fetched: 3,344,030
Total URLs passing: 563,213
```

4. Count the number of tokens in the fetched pages that pass the filtering
   pipeline:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/count_tokens.py \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/text/open-web-math-fde8ef8-10M-cc-deduplicated/*.passing.parquet"]' \
    --output_path "gs://marin-us-central2/scratch/nfliu/count_tokens/open_web_math_10M_cc_deduplicated_passing/"
```

Results:

```
Total number of tokens: 2,634,194,474
Total number of documents: 563,213
```

5. Take the fetched pages that pass the filtering pipeline and run MinHash
   deduplication against open-web-math:

```
python marin/run/ray_run.py \
    --pip_deps 'datatrove[io] @ git+https://github.com/nelson-liu/datatrove@ray_executor_dedup_logging,spacy,cupy-cuda12x==13.3.0,orjson,scipy==1.13.1' \
    --no_wait -- \
    python marin/crawl/minhash/deduplicate_against_index.py \
    --index_path 'gs://marin-us-central2/scratch/nfliu/minhash/open_web_math_minhash_index/index' \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/text/open-web-math-fde8ef8-10M-cc-deduplicated/*_text_and_scores.passing.parquet"]' \
    --parquets_paths_file 'gs://marin-us-central2/scratch/nfliu/open-web-math-fde8ef8-10M-cc-deduplicated-passing_paths.txt' \
    --minhash_base_path 'gs://marin-us-central2/scratch/nfliu/minhash/open_web_math_10M_cc_deduplicated_passing_minhash_against_open_web_math' \
    --minhash_logs_path 'gs://marin-us-central2/scratch/nfliu/minhash/open_web_math_10M_cc_deduplicated_passing_minhash_against_open_web_math_logs'

# Move the deduplicated content
gcloud storage mv gs://marin-us-central2/scratch/nfliu/minhash/open_web_math_10M_cc_deduplicated_passing_minhash_against_open_web_math/deduplicated_output/* gs://marin-us-central2/scratch/nfliu/text/open_web_math_10M_cc_deduplicated_passing_minhash_against_open_web_math/
# Remove the logs and intermediate output
gcloud storage rm --recursive gs://marin-us-central2/scratch/nfliu/minhash/open_web_math_10M_cc_deduplicated_passing_minhash_against_open_web_math
gcloud storage rm --recursive gs://marin-us-central2/scratch/nfliu/minhash/open_web_math_10M_cc_deduplicated_passing_minhash_against_open_web_math_logs
```

6. Count the number of tokens in the fetched pages that pass the filtering
   pipeline after minhash deduplication:


```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/count_tokens.py \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/text/open_web_math_10M_cc_deduplicated_passing_minhash_against_open_web_math/*.jsonl.gz"]' \
    --output_path "gs://marin-us-central2/scratch/nfliu/count_tokens/open_web_math_10M_cc_deduplicated_passing_minhash_against_open_web_math/"
```

Results:

```
Total number of tokens: 574,590,766
Total number of documents: 171,665
```
