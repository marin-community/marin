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
# Add your dataset to DATASET_CONFIGS in train_test_total.py
DATASET_CONFIGS = [
    # ... existing datasets ...
    DatasetConfig(
        name="${DATASET_NAME}",
        path="gs://${BUCKET}/${DATASET_NAME}",
        text_field="text"  # Or "content" if using different field name
    ),
]
```

**Parameters explained**:
- **name**: Identifier for your dataset (used in output paths)
- **path**: GCS/S3/local path to your dataset directory
- **text_field**: Name of the text field in your data files (default: "text")

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

The default `train_test_total.py` defines helpers for building dedupliation
steps. You can define your own steps as well with custom N-gram or matching options:

```python
# experiments/my_dedupe.py

from experiments.train_test_overlap.train_test_total import EVAL_DATASET_STEPS

NGRAM_CONFIG = NGramConfig(
    ngram_length=[5, 10, 15],
    overlap_threshold=1e-6,
    stride=5,
)


def build_step(dataset_config: DatasetConfig) -> ExecutorStep:
    dedupe_config = DedupeConfig(
        input_path=dataset_config.path,
        output_path=this_output_path(),
        decontaminate_source=EVAL_DATASET_STEPS,
        attribute_name="ngram_overlap",
        false_positive_rate=1e-20,
        ngram=NGRAM_CONFIG,
        processes=1024,
        mode=DedupMode.TRAIN_TEST_OVERLAP,
        text_field=dataset_config.text_field,
    )

    return ExecutorStep(
        name=f"tmp/power/train_test_overlap/dolma/total/{dataset_config.name}",
        fn=run_train_test_overlap,
        config=dedupe_config,
        description=f"Run dedupe train-test overlap on {dataset_config.name}",
        pip_dependency_groups=["quality_dedup_consolidate"],
    )
```

**Understanding the `processes` Parameter:**

The `processes` parameter controls how many parallel processes Zephyr uses when processing your dataset:

- **What it does**: Controls the degree of parallelism for processing files in the dataset
- **Performance trade-off**: More processes = faster processing but higher CPU/memory usage
- **Typical values**: 8-4096 processes depending on your cluster's CPU and memory capacity

**Common configurations**:
- **Fast screening**: `ngram_length=[15]`, `processes=1024`
- **Balanced**: `ngram_length=[15]`, `processes=1024`
- **Memory-constrained**: `processes=32`

### Parallel Processing

Zephyr handles file discovery and parallelism automatically. You can control the degree of parallelism using the `processes` parameter:

```python
# In build_step function - adjust processes for parallelism
def build_step(dataset_config: DatasetConfig) -> ExecutorStep:
    dedupe_config = DedupeConfig(
        # ...
        processes=128,  # Control parallelism level
    )
    return ExecutorStep(
        # ...
        config=dedupe_config,
    )
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

Then add it to the `EVAL_DATASET_STEPS` list in `eval_datasets_overlap.py`. The train-test overlap detection will automatically include your new evaluation dataset in all future runs.

## Advanced Usage

### Custom N-gram Analysis

For specialized analysis, you can edit the constants in `train_test_total.py`:

```python
# Edit DEFAULT_NGRAM_CONFIG for different n-gram sizes
DEFAULT_NGRAM_CONFIG = NGramConfig(
    ngram_length=[10],        # Change from [15] to [10] for 10-grams
    overlap_threshold=1e-6,
    stride=0,
)

# Edit the processes parameter in build_step
def build_step(dataset_config: DatasetConfig) -> ExecutorStep:
    dedupe_config = DedupeConfig(
        # ...
        processes=32,  # Increase for more parallelism
    )
    return ExecutorStep(
        # ...
        config=dedupe_config,
    )
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

   **Solution**: Reduce the `processes` parameter to lower memory usage:
   ```python
   processes=32,  # Reduce from default 128
   ```

4. **Slow processing**

   **Solution**: Increase the `processes` parameter (if you have resources):
   ```python
   processes=256,  # Increase from default 128
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
        text_field="text"  # Or "content" depending on your data
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
