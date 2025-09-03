# Dolma Deduplication Tool

To run deduplication, we use the dolma deduplication tool. You can find in-depth documentation here: https://github.com/allenai/dolma/blob/main/docs/deduplication.md.

For our purposes, here is a quickstart:

```bash
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- python marin/processing/classification/dedupe.py --input_path gs://marin-us-central2/documents/hello_world_fw/v1.0/quickstart/ --output_path gs://marin-us-central2/attributes/hello_world_fw/v1.0/quickstart_duplicates/
```

Please note that deduplication assumes the input path is of the form `gs://{$BUCKET_PATH}/documents/` and will write attributes to `gs://{$BUCKET_PATH}/attributes/`

## Parameters

### Required Parameters:
- `--input_path`: GCP input directory path (required)
    - Example: `gs://marin-us-central2/scratch/documents/dummy_dedupe_data/`
- `--output_path`: GCP output directory path to save deduplication attributes (required)
    - Example: `gs://marin-us-central2/scratch/attributes/dummy_dedupe_data/text_dup/`

### Optional Parameters with Defaults:
- `--dedupe_key`: JSON path for the deduplication key (default: `$.id`)
- `--attribute_name`: Name of the attribute to set if the document is a duplicate (default: `duplicate_documents`)
- `--skip_empty`: Skip empty documents (default: `True`)
- `--min_length`: Minimum length of documents to be deduplicated (default: `0`)
- `--min_words`: Minimum number of uniseg word units in documents to be deduplicated (default: `0`)
- `--bloom_filter_size`: Size of the Bloom filter in bytes (default: `None`)
- `--estimated_doc_count`: Estimated number of documents to dedupe (default: `1000000`)
- `--false_positive_rate`: Desired false positive rate for the Bloom filter (default: `0.00001`)
- `--processes`: Number of processes to use for deduplication (default: `1`)

## Important Parameters Explained

### Estimated Document Count (`estimated_doc_count`):
The `estimated_doc_count` parameter directly affects the size of the Bloom filter and the accuracy of deduplication. Underestimating this value results in a smaller Bloom filter, using less memory but potentially leading to over-deduplication as the false positive rate increases beyond the specified rate. Conversely, overestimating creates a larger Bloom filter, improving accuracy by maintaining a lower-than-specified false positive rate, but at the cost of increased memory usage. The trade-off is between memory efficiency and deduplication accuracy.

### False Positive Rate (`false_positive_rate`):
The `false_positive_rate` parameter inversely affects the Bloom filter size and deduplication accuracy. Increasing this rate allows for a smaller Bloom filter, reducing memory usage and potentially speeding up processing, but at the cost of increased incorrect deduplication. Decreasing the rate requires a larger Bloom filter, using more memory and potentially slowing processing, but it improves accuracy by reducing incorrect deduplication. This parameter balances between resource efficiency and the precision of the deduplication process.

## Example Commands

### Example command on dummy data:
```bash
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- python marin/processing/classification/dedupe.py --input_path gs://marin-us-central2/scratch/documents/dummy_dedupe_data/text/ --output_path gs://marin-us-central2/scratch/attributes/dummy_dedupe_data/text_dup/
```

### Another example:
```bash
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- python marin/processing/classification/dedupe.py --input_path gs://marin-us-central2/scratch/documents/dedupe_data/v1/testdedupe/ --output_path gs://marin-us-central2/scratch/attribute/dedupe_data/v1/testdedupe/
```

## Decontamination Mode

When you run with `--decontaminate` and specify `--decontaminate_path`, the script executes in two phases:
1. It first builds a Bloom filter from the decontamination dataset.
2. It then applies this Bloom filter in read-only mode to your main input path, tagging any spans or documents seen in the decontamination set.

Under the hood, the script stages the decontamination data, runs the deduper to create `decontaminated_bloom_filter.bin`, deletes staged JSONL files, stages the target documents, and runs the deduper again with `--bloom_filter.read_only` enabled.

For full details on the underlying flags, see the upstream Dolma docs: https://github.com/allenai/dolma/blob/main/docs/deduplication.md#bloom-filter-read-only

### Example for this repo

```bash
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- \
  python marin/processing/classification/dedupe.py \
  --input_path gs://marin-us-central2/scratch/documents/my_corpus/ \
  --output_path gs://marin-us-central2/scratch/attributes/my_corpus/decontaminated_dup/ \
  --decontaminate \
  --decontaminate_path gs://marin-us-central2/dolma/mmlu_dev/ \
  --processes 4 \
  --estimated_doc_count 1000000 \
  --false_positive_rate 0.001
```
```
