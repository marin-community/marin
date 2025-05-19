# Train-Test Overlap Detection

This directory tracks efforts for measuring train-test overlap in Marin datasets.

## Directory Structure

- `format/`: Contains scripts for converting datasets to JSONL format, which is required by the Dolma deduplication pipeline
- `train_test/`: Contains scripts to run train-test overlap detection for each dataset of interest

## Overview

The process involves two main steps:

1. Converting datasets to JSONL format using scripts in the `format/` directory, since this is the required format for Dolma's deduplication functionality

2. Running train-test overlap detection using scripts in `train_test/` directory to identify and measure overlap between training and test sets for each dataset

The overlap detection helps ensure data quality by identifying potential contamination between training and evaluation data.

## Workflow Process

This section outlines the complete workflow for running train-test overlap detection:

### 1. Convert Raw Data to Dolma Format

First, run `eval_datasets_overlap.py` to convert raw Hugging Face datasets to Dolma format suitable for deduplication:

```bash
python experiments/train_test_overlap/eval_datasets_overlap.py
```

This script:
- Downloads evaluation datasets (MMLU, GSM8K, Math, TruthfulQA, BBH) from Hugging Face
- Converts each dataset to the "decontamination" format with proper text fields for overlap detection

### 2. Run Overlap Pipeline to Generate N-grams

First, run a sharded overlap pipeline script to generate n-gram overlap data. For example, to use the DCLM pipeline:

```bash
python experiments/train_test_overlap/train_test/overlap_pipeline_dclm_sharded.py
```

You can also use other pipeline scripts in the `train_test/` directory for different datasets (e.g., `overlap_pipeline_starcoder.py`).

### 3. Aggregate N-gram Results

Once the n-gram overlap data has been generated, run the aggregation script to combine results and compute summaries:

```bash
python experiments/train_test_overlap/train_test/aggregate_test_overlap.py
```

Optionally, filter which n-gram sizes to process by setting the `N_VALUES` environment variable:

```bash
export N_VALUES="10,15"
python experiments/train_test_overlap/train_test/aggregate_test_overlap.py
```

### 4. Analysis of Results

The overlap detection pipeline outputs several types of files:
- Overlap statistics showing which test instances have overlapping inputs or references
- Raw n-gram data showing the specific overlapping content
- Aggregated metrics for different n-gram sizes (5, 10, 15, etc.)
- Instance mapping files to help trace overlaps back to specific examples

These files can be used to assess the extent of data contamination and make informed decisions about dataset quality.
