# Train-Test Overlap Detection

This tutorial will teach you how to use Marin's train-test overlap script (which is a wrapper around the dolma deduplication toolkit) to identify overlap between your training datasets and evaluation benchmarks. This is crucial for calibrating the results of your model evaluations. Under the hood, this uses the same library for deduplication and decontamination.

## What is Train-Test Overlap?

**Train-test overlap** occurs when your training data contains examples that are too similar to your evaluation data. This can lead to:

- **Inflated evaluation scores**: Your model may have "memorized" answers rather than learned to reason
- **Invalid comparisons**: Results become incomparable to other models trained on clean data
- **Poor generalization**: Performance doesn't reflect real-world capability

Marin's system detects overlap by finding shared n-grams (sequences of words) between training and evaluation datasets using efficient Bloom filters.

## How It Works

The system uses a two-step process:

1. **Deduplication Step** (`train_test_total.py`):
   - Automatically discovers and resolves all evaluation dataset paths using the executor framework
   - Builds Bloom filters from training data n-grams
   - Scans all evaluation datasets for matching n-grams
   - Outputs contaminated examples with overlap statistics

2. **Aggregation Step** (`aggregate_total.py`):
   - Combines results across all datasets
   - Generates overlap matrices and CSV summaries
   - Produces detailed overlap maps that tell you which documents between training and test datasets have overlap

## Quick Start

### Prerequisites

- Make sure you've followed the [installation guide](installation.md) to do the basic installation.
- Access to your training and evaluation datasets
- Ray cluster set up (if running distributed)

### Running on Existing Datasets

The easiest way to get started is to run overlap detection on the pre-configured datasets:

```bash
# Step 1: Run N-gram overlap between each train shard for selected training dataset and selected test datasets
python experiments/train_test_overlap/train_test_total.py --prefix gs://${BUCKET}

# Step 2: Aggregate results across datasets
python experiments/train_test_overlap/aggregate_total.py --prefix gs://${BUCKET}
```

**Concretely, this process will:**

1. **Generate a Bloom filter for each training shard using the configured N-gram settings**
2. **Read all specified test datasets and check them against the Bloom filter in read-only mode**
3. **Output attribute files that specify the overlap between each training shard and all test datasets**

This will process these datasets:
- **Training**: FineMath, DCLM, StarCoder, ProofPile, Dolmino, Nemotron-CC
- **Evaluation**: All datasets defined in `eval_datasets_overlap.py` includes: GSM8K, MATH, TruthfulQA, BBH, MMLU, HumanEval, and more

### Understanding the Output

After running both scripts, you'll find:

```
${PREFIX}/
├── train_test_overlap/total/
│   ├── finemath-abc123/          # Per-dataset results
│   ├── dclm-def456/
│   └── ...
└── train_test_overlap/aggregate_total/
    ├── union/15/summary.csv      # Overlap summary for 15-grams with a union over all training datasets
    ├── overlap_matrix.csv  # Training vs eval matrix
    └── individual_datasets/      # Per-dataset breakdowns
```

Key files to examine:
- **`overlap_matrix.csv`**: Shows overlap percentages between each training-eval pair
- **`summary.csv`**: Overall overlap statistics
- **`*_overlap_map.jsonl.gz`**: Detailed per-example overlap data - for each test dataset, shows every individual document that has overlap with training data, including the exact file paths where the overlap occurs in both training and test files

## Adding Your Own Dataset

### Step 1: Define Your Dataset

First, create a dataset configuration. Your dataset needs to be in one of these formats:

```python
# Supported formats
SUPPORTED_FORMATS = [
    ".parquet",
    ".jsonl.gz",
    ".jsonl.zst",
    ".jsonl",
    ".json.gz",
    ".json.zst"
]
```

### Step 2: Add Dataset to Configuration

Edit `experiments/train_test_overlap/train_test_total.py` and add your dataset:

```python
# Import the DatasetConfig class
from experiments.train_test_overlap.utils import DatasetConfig, ShardedDedupeConfig, run_all_shards

# Add your dataset to DATASET_CONFIGS
DATASET_CONFIGS = [
    # ... existing datasets ...
    DatasetConfig(
        name="${DATASET_NAME}",
        path="gs://${BUCKET}/${DATASET_NAME}",
        max_in_flight=64
    ),
]
```

**Parameters explained**:
- **name**: Identifier for your dataset (used in output paths)
- **path**: GCS/S3/local path to your dataset directory
- **max_in_flight**: Number of parallel tasks (adjust based on your cluster size)

### Step 3: Run Detection

```bash
python experiments/train_test_overlap/train_test_total.py --prefix gs://${BUCKET}
```

The system will automatically:
- Discover all shards in your dataset directory
- Process them in parallel with the configured n-gram settings
- Output overlap results for each evaluation dataset

## Configuration Options

### N-gram Settings

Edit the `BASE_DEDUPE_CONFIG` in `experiments/train_test_overlap/utils.py`:

```python
BASE_DEDUPE_CONFIG = DedupeConfig(
    # ... other settings ...
    ngram=NGramConfig(
        ngram_length=[15],        # List of n-gram sizes to check
        overlap_threshold=1e-6,   # Minimum overlap to report
        stride=0,                 # Stride between n-grams (0 = every position)
    ),
    processes=16,                 # Parallel processes per shard (see explanation below)
    false_positive_rate=1e-12,    # Bloom filter false positive rate
)
```

**Understanding the `processes` Parameter:**

The `processes` parameter controls how many parallel processes dolma uses **per training shard**. Here's how it works:

- **What it does**: Each training shard gets split into `processes` number of sub-shards for parallel processing
- **Performance trade-off**: More processes = faster Bloom filter creation but higher CPU/memory usage
- **Resource impact**: Each process loads a portion of the shard into memory simultaneously
- **Typical values**: 8-32 processes depending on your cluster's CPU and memory capacity

Example: If you have a 1GB training shard and set `processes=16`, the system will:
1. Split the shard into 16 smaller chunks
2. Process each chunk in parallel to build the Bloom filter
3. Use 16x more CPU cores but complete ~16x faster

**Common configurations**:
- **Fast screening**: `ngram_length=[15]`, `false_positive_rate=1e-9`, `processes=8`
- **Thorough analysis**: `ngram_length=[10, 15, 20]`, `false_positive_rate=1e-12`, `processes=16`
- **Memory-constrained**: `false_positive_rate=1e-6`, `processes=4` (fewer processes = less memory)

### Parallel Processing

The system has two levels of parallelism that you can adjust:

**1. Dataset-level parallelism (`max_in_flight`):**
```python
# In DATASET_CONFIGS - how many shards to process simultaneously
DatasetConfig(
    name="dataset_name",
    path="path",
    max_in_flight=128  # Process 128 shards at once
),
```

**2. Shard-level parallelism (`processes`):**
```python
# In BASE_DEDUPE_CONFIG - how many processes per shard. This will make generating the bloom filter for each training shard on a node run faster.
processes=32,  # Split each shard into 32 parallel processes
```

### Custom Evaluation Datasets

To add a new evaluation dataset, create a conversion step in `experiments/train_test_overlap/eval_datasets_overlap.py`:

```python
# Download the dataset
my_eval_raw = ExecutorStep(
    name="raw/${DATASET_NAME}",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="org/dataset-name",
        revision=versioned("commit-hash"),
        gcs_output_path=this_output_path(),
    ),
)

# Convert to decontamination format
my_eval_convert_dolma = ExecutorStep(
    name="decontamination/my_evaluation_dataset",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="org/dataset-name",
        input_path=my_eval_raw,
        output_path=this_output_path(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="question",  # Key containing the text to check
        # ... other conversion options ...
    ),
)
```

Then add it to the `EVAL_DATASET_STEPS` list in both `eval_datasets_overlap.py` and `utils.py`. The train-test overlap detection will automatically include your new evaluation dataset in all future runs.

## Advanced Usage

### Custom N-gram Analysis

For specialized analysis, you can run single n-gram sizes:

```python
# Create custom config
custom_config = dataclasses.replace(
    BASE_DEDUPE_CONFIG,
    ngram=NGramConfig(ngram_length=[10]),  # Only 10-grams
    false_positive_rate=1e-8,
)

# Use in your ExecutorStep
step = ExecutorStep(
    name="custom_overlap_analysis",
    fn=run_all_shards,
    config=ShardedDedupeConfig(
        dataset_dir="gs://${BUCKET}/${DATASET_NAME}",
        output_path=this_output_path(),
        max_in_flight=32,
    ),
)
```

### Memory Optimization

For very large datasets or memory-constrained environments:

```python
# Reduce memory usage at the shard level
BASE_DEDUPE_CONFIG = dataclasses.replace(
    BASE_DEDUPE_CONFIG,
    false_positive_rate=1e-6,     # Higher false positive rate = smaller Bloom filters
    processes=8,                   # Fewer processes per shard = less concurrent memory usage
)

# Reduce memory usage at the dataset level
DATASET_CONFIGS = [
    DatasetConfig(
        name="${DATASET_NAME}",
        path="gs://${BUCKET}/${DATASET_NAME}",
        max_in_flight=16  # Process fewer shards simultaneously
    ),
]
```

**Memory calculation**: Each process loads part of a shard into memory, so:
- `max_in_flight=16` × `processes=8` = 128 total processes
- Compare to default: `max_in_flight=64` × `processes=16` = 1,024 total processes
- **Result**: ~8x less memory usage but proportionally slower processing

## Gotchas and Troubleshooting

### Common Issues

1. **No files found error**
   ```
   FileNotFoundError: No shard files with extensions (...) found under gs://path
   ```

   **Solution**: Verify your dataset path contains files with supported extensions:
   ```bash
   gsutil ls -r gs://${BUCKET}/${DATASET_NAME}/**/*.{parquet,jsonl.gz,json.gz}
   ```

2. **Empty dataset early exit**
   ```
   INFO: Pipeline exited early - empty input detected
   ```

   **Solution**: Check that your files contain actual data:
   ```bash
   gsutil du -s gs://${BUCKET}/${DATASET_NAME}/*
   ```

3. **Out of memory errors**

   **Solution**: Reduce memory usage by lowering parallelism:
   ```python
   # Increase false positive rate (uses less memory for Bloom filters)
   false_positive_rate=1e-6,  # Instead of 1e-12

   # Reduce shard-level parallelism (fewer processes per shard)
   processes=8,  # Instead of 16

   # Reduce dataset-level parallelism (fewer concurrent shards)
   max_in_flight=16,  # Instead of 64
   ```

4. **Slow processing**

   **Solution**: Increase parallelism (if you have resources). Be careful because this can crash the head node!
   ```python
   # More concurrent shards
   max_in_flight=128,

   # More processes per shard (splits each shard into more parallel chunks)
   processes=32,
   ```

### Performance Tips

- **Start small**: Test with a subset of your data first
- **Monitor resources**: Watch memory and CPU usage during runs
- **Use appropriate n-gram sizes**: 15-grams are a good default for most use cases
- **Batch processing**: Process datasets separately if you hit resource limits


### Example Workflow

Here's a complete workflow for checking a new training dataset against all evaluation benchmarks:

**1. Add your dataset to train_test_total.py**
```python
DATASET_CONFIGS = [
    # ... existing datasets ...
    DatasetConfig(
        name="${DATASET_NAME}",
        path="gs://${BUCKET}/${DATASET_NAME}",
        max_in_flight=64
    ),
]
```

**2. Run deduplication**
```bash
python experiments/train_test_overlap/train_test_total.py
```
**3. Run aggregation**
```bash
python experiments/train_test_overlap/aggregate_total.py
```
**4. Analyze results**
```bash
gsutil cat gs://${BUCKET}/train_test_overlap/dolma/aggregate_total_final-*/overlap_matrix.csv
```

This will give you overlap percentages for your training data against all standard evaluation benchmarks, helping you make informed decisions about data quality and evaluation validity.

## Gotchas

- If the provided dataset path contains no files or only empty files, the
  pipeline will exit early to avoid hanging on empty shards.
- The Bloom filter false positive rate defaults to `1e-12` in
  `train_test_total.py`. For very large datasets you may need to adjust this
  value to manage memory usage.
- Ensure the dataset path you provide actually contains files with one of
  the supported extensions; otherwise no shards will be discovered.
