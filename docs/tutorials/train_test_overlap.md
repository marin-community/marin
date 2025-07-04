# Train-Test Overlap Detection

This tutorial will teach you how to use Marin's train-test overlap detection system to identify contamination between your training datasets and evaluation benchmarks. This is crucial for ensuring the validity of your model evaluations.

## What is Train-Test Overlap?

**Train-test overlap** (also called data contamination) occurs when your training data contains examples that are too similar to your evaluation data. This can lead to:

- **Inflated evaluation scores**: Your model may have "memorized" answers rather than learned to reason
- **Invalid comparisons**: Results become incomparable to other models trained on clean data
- **Poor generalization**: Performance doesn't reflect real-world capability

Marin's system detects overlap by finding shared n-grams (sequences of words) between training and evaluation datasets using efficient Bloom filters.

## How It Works

The system uses a two-step process:

1. **Deduplication Step** (`train_test_total.py`):
   - Builds Bloom filters from training data n-grams
   - Scans evaluation datasets for matching n-grams
   - Outputs contaminated examples with overlap statistics

2. **Aggregation Step** (`aggregate_total.py`):
   - Combines results across all datasets
   - Generates contamination matrices and CSV summaries
   - Produces detailed contamination maps

## Quick Start

### Prerequisites

Make sure you have:
- Marin installed and configured
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
- **Evaluation **: GSM8K, MATH, TruthfulQA, BBH, MMLU, HumanEval, and more

### Understanding the Output

After running both scripts, you'll find:

```
${PREFIX}/
├── train_test_overlap/dolma/total/
│   ├── finemath-abc123/          # Per-dataset results
│   ├── dclm-def456/
│   └── ...
└── train_test_overlap/dolma/aggregate_total_final-xyz789/
    ├── union/15/summary.csv      # Overall contamination summary
    ├── contamination_matrix.csv  # Training vs eval matrix
    └── individual_datasets/      # Per-dataset breakdowns
```

Key files to examine:
- **`contamination_matrix.csv`**: Shows overlap percentages between each training-eval pair
- **`summary.csv`**: Overall contamination statistics
- **`*_contamination_map.jsonl.gz`**: Detailed per-example contamination data

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
        path="gs://${BUCKET}/path/to/${DATASET_NAME}",
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
- Output contamination results for each evaluation dataset

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
    processes=16,                 # Parallel processes per shard
    false_positive_rate=1e-12,    # Bloom filter false positive rate
)
```

**Common configurations**:
- **Fast screening**: `ngram_length=[15]`, `false_positive_rate=1e-9`
- **Thorough analysis**: `ngram_length=[10, 15, 20]`, `false_positive_rate=1e-12`
- **Memory-constrained**: Increase `false_positive_rate` to `1e-8` or `1e-6`

### Parallel Processing

Adjust parallelism based on your resources:

```python
# In DATASET_CONFIGS - max tasks per dataset
DatasetConfig(
    name="dataset_name",
    path="path",
    max_in_flight=128  # High parallelism for large datasets
),

# In BASE_DEDUPE_CONFIG - processes per task
processes=32,  # More processes = more memory usage
```

### Custom Evaluation Datasets

To add a new evaluation dataset, create a conversion step in `experiments/train_test_overlap/eval_datasets_overlap.py`:

```python
# Download the dataset
my_eval_raw = ExecutorStep(
    name="raw/my_evaluation_dataset",
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

Then add it to the `EVAL_DATASET_STEPS` list in `aggregate_total.py`.

## Advanced Usage

### Custom N-gram Analysis

For specialized analysis, you can run single n-gram sizes:

```python
# Create custom config
custom_config = dataclasses.replace(
    BASE_DEDUPE_CONFIG,
    ngram=NGramConfig(ngram_length=[10]),  # Only 10-grams
    false_positive_rate=1e-8,              # Lower memory usage
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

For very large datasets:

```python
# Reduce memory usage
BASE_DEDUPE_CONFIG = dataclasses.replace(
    BASE_DEDUPE_CONFIG,
    false_positive_rate=1e-6,     # Higher false positive rate = less memory
    processes=8,                   # Fewer parallel processes
)

# Use smaller batch sizes
DATASET_CONFIGS = [
    DatasetConfig(
        name="${DATASET_NAME}",
        path="gs://${BUCKET}/${DATASET_NAME}",
        max_in_flight=16  # Lower max_in_flight
    ),
]
```

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

   **Solution**: Reduce memory usage:
   ```python
   # Increase false positive rate
   false_positive_rate=1e-6,  # Instead of 1e-12

   # Reduce parallel processes
   processes=8,  # Instead of 16

   # Lower max_in_flight
   max_in_flight=16,  # Instead of 64
   ```

4. **Slow processing**

   **Solution**: Increase parallelism (if you have resources):
   ```python
   # More parallel tasks
   max_in_flight=128,

   # More processes per task
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
python experiments/train_test_overlap/train_test_total.py --prefix gs://${BUCKET}
```
**3. Run aggregation**
```bash
python experiments/train_test_overlap/aggregate_total.py --prefix gs://${BUCKET}
```
**4. Analyze results**
```bash
gsutil cat gs://${BUCKET}/train_test_overlap/dolma/aggregate_total_final-*/contamination_matrix.csv
```

This will give you contamination percentages for your training data against all standard evaluation benchmarks, helping you make informed decisions about data quality and evaluation validity.
