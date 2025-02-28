# Experiment Log

## FineWeb-Edu-10M

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

## open-web-math-10M

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
Total URLs passing: 885352
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
Total number of tokens: 3,210,247,949
Total number of documents: 885,352
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
Total number of tokens: 830,539,923
Total number of documents: 275,387
```

## fineweb-edu-10M-cc-deduplicated

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

## open-web-math-10M-cc-deduplicated

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
Total URLs passing: 587,403
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
Total number of tokens: 2,752,544,528
Total number of documents: 587,403
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
Total number of tokens: 649,396,489
Total number of documents: 191,393
```
