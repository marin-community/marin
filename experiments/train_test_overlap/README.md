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

### 2. Convert to Scenario Format and Consolidate

Next, run `test_scenario_conversation.py` to convert the Dolma-formatted data to scenario files, and then use `consolidate_scenario_jsonl.py` to combine them into a single file:

```bash
# Run conversion only
python experiments/train_test_overlap/format/test_scenario_conversation.py

# Run consolidation (this will automatically run conversion step first if needed)
python experiments/train_test_overlap/format/consolidate_scenario_jsonl.py
```

Alternatively, you can run the entire pipeline at once:

```bash
python experiments/train_test_overlap/format/run_scenario_pipeline.py
```

These steps:
1. Convert each dataset to standardized scenario JSONL files with input and reference fields
2. Find all individual scenario files across datasets
3. Consolidate them into a single file for easier processing by the overlap detection system
4. The consolidated file can be directly passed to the overlap pipeline scripts

### 3. Run Overlap Detection

Finally, run the appropriate overlap pipeline script for the dataset you want to analyze. For example, to check StarCoder data:

```bash
python experiments/train_test_overlap/train_test/overlap_pipeline_starcoder.py
```

Each pipeline script:
- Processes the training data at n-gram level (with configurable N values)
- Compares against the scenario data to identify overlaps
- Calculates overlap metrics and statistics
- Outputs detailed reports and raw overlap data for analysis

For StarCoder data specifically, you may need to first convert from Parquet to JSONL format using:

```bash
python experiments/train_test_overlap/format/convert_starcoder_parquet2jsonl.py
```

### 4. Analysis of Results

The overlap detection pipeline outputs several types of files:
- Overlap statistics showing which test instances have overlapping inputs or references
- Raw n-gram data showing the specific overlapping content
- Aggregated metrics for different n-gram sizes (5, 10, 15, etc.)
- Instance mapping files to help trace overlaps back to specific examples

These files can be used to assess the extent of data contamination and make informed decisions about dataset quality.
