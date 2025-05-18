# Train-Test Overlap Detection System

## Overview

This system is designed to detect and measure overlap between training data and test datasets for machine learning models. This is critical for ensuring the validity of model evaluation, as test data that appears in the training data can lead to artificially inflated performance metrics. The system identifies specific instances in test sets that have n-gram overlaps with training data, providing a detailed analysis of potential data contamination.

## Key Components

### Data Structures

The system uses several key data structures to track and quantify overlap:

- **N-grams**: Sequences of n consecutive words used to compare text. The system supports configurable n-gram sizes (default: 5, 9, 13).
- **EntryOverlapNgrams**: Records which specific n-grams from test data appear in training data and their frequencies.
- **DataOverlapStats**: Tracks which test instances have any overlapping n-grams, separately for inputs and reference answers.

### Core Files and Functionality

1. **data_overlap_spec.py**
   - Defines the data structures for representing overlap information
   - Contains dataclasses for tracking overlap at various levels (instance, n-gram, metric)
   - Provides keys and containers for organizing overlap statistics by dataset and configuration

2. **compute_data_overlap_metrics.py**
   - Implements core functions for n-gram extraction and comparison
   - Creates n-gram indices from test datasets
   - Processes training data to identify overlapping n-grams
   - Tracks which specific test instances have overlaps

3. **run_data_overlap.py**
   - Manages the end-to-end pipeline for overlap detection
   - Handles file I/O, data decompression, and parallel processing
   - Writes overlap statistics to disk for further analysis
   - Computes and aggregates metrics from raw overlap data

4. **compute_metrics_from_ngrams.py and metrics.py**
   - Computes various metrics based on raw n-gram overlap data
   - Aggregates metrics across datasets and instances
   - Provides different ways to quantify overlap severity (binary, Jaccard, token-level)

### Utilities

- **utils.py**
   - Contains a tokenizer that splits text into words based on whitespace and punctuation
   - Provides utility functions for dataclass serialization

## How It Works

1. **Initialization**:
   - The system loads test data (scenarios) and creates n-gram indices for specified n-gram sizes
   - It prepares data structures to track overlapping instances and n-gram counts

2. **Processing Training Data**:
   - For each training data file, the system:
     - Extracts n-grams from the training text
     - Checks if these n-grams appear in any test instances
     - When an overlap is found, records the affected test instance and increments the count for that n-gram

3. **Output Generation**:
   - Writes raw n-gram overlap data to disk
   - Creates summary statistics showing which test instances have overlapping inputs or references
   - Computes various metrics to quantify the severity of overlap

4. **Metric Aggregation**:
   - Aggregates metrics across instances and datasets
   - Provides dataset-level statistics about contamination

## Usage

The system is configured through the `DataOverlapPipelineConfig` class, which allows specifying:
- Paths to input (training) data
- Path to scenario (test) data
- Output path for results
- N-gram sizes to analyze
- Number of parallel processes

The system uses Ray for distributed processing to handle large datasets efficiently.

## Output Data and Metrics

The system aggregates overlap data into metrics for each dataset (scenario) using different overlap calculation methods. Each output record contains:

1. **aggregate_data_overlap_key**: Identifies the dataset, split, n-gram size, and part (input or reference)
2. **metric_scores**: Array of overlap scores, one per test instance in the dataset (each score corresponds to one test question)
3. **metric_protocol_spec**: Specifies the calculation method used

The length of the metric_scores array matches the number of instances in the specified split (e.g., test) that have any overlapping n-grams. This may be a subset of all instances if some have no overlap at all, or it may be filtered by other criteria in the analysis pipeline.

For example, in the output where 11 scores are shown for MMLU Anatomy, this represents 11 test questions that were analyzed and found to have overlaps, though the full test set may contain many more questions.

### Relating Scores to Specific Questions

The scores in the metric_scores array don't directly identify which instance IDs they correspond to. To determine exactly which questions have concerning levels of overlap, you would need to look at:

1. The raw overlap data files (EntryOverlapNgrams)
2. The detailed metric files before aggregation

For instance-level analysis, these detailed files allow you to map specific scores to specific question IDs (e.g., id19, id20, etc. in the MMLU dataset).

### Overlap Calculation Methods

The system uses three methods to quantify overlap, each providing different insights into data contamination:

#### 1. Binary Overlap (partial_overlap_spec: 0)

A simple yes/no indicator of whether any overlap exists. Scores are either 0 (no overlap) or 1 (overlap exists).

**How it's calculated**: Checks if any n-gram in the test instance appears in the training data. If at least one match is found, the score is 1.0.

**Interpretation**: A score of 1.0 means there is at least some overlap, but doesn't tell you the extent. Useful for quickly identifying which instances have any overlap at all.

**Example (MMLU Anatomy, 5-grams, input texts):**
```json
{
  "aggregate_data_overlap_key": {
    "stats_key": {
      "light_scenario_key": {
        "scenario_spec": {
          "class_name": "helm.benchmark.scenarios.mmlu_scenario.MMLUScenario",
          "args": {"subject": "anatomy"}
        },
        "split": "test"
      },
      "overlap_protocol_spec": {"n": 5}
    },
    "part": "input"
  },
  "metric_scores": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
  "metric_protocol_spec": {
    "partial_overlap_spec": 0,
    "frequency_spec": {"filter_value": 0, "weighting": false}
  }
}
```
This shows that 11 test instances in the MMLU Anatomy dataset have at least one 5-gram that overlaps with the training data. Each 1.0 in the array corresponds to a specific test question, though we'd need the detailed files to map which 11 out of the full test set these represent.

#### 2. Jaccard Overlap (partial_overlap_spec: 1)

for each train shard:
  track test instances detected:
    jaccrd_overlap

for top level summary, go through each shard, aggregate all test instance
keys, keep the highest value for overlap.

So each shard will have highest training sample (somewhere) that has overlap w test instance

Then total shards will record max overlap per instance over all training shards.

Measures the proportion of n-grams in the test instance that also appear in the training data.

**How it's calculated**:
1. Count the number of n-grams in the test instance that match training data
2. Divide by the total number of n-grams in the test instance
3. Result is a value between 0 and 1

**Interpretation**:
- 0.0 means no overlap
- Values closer to 1.0 indicate more extensive overlap
- A score of 0.4 means 40% of the n-grams in the test instance appear in training data
- Higher values potentially suggest more significant contamination

**Example (MMLU Anatomy, 5-grams, input texts):**
```json
{
  "aggregate_data_overlap_key": {
    "stats_key": {
      "light_scenario_key": {
        "scenario_spec": {
          "class_name": "helm.benchmark.scenarios.mmlu_scenario.MMLUScenario",
          "args": {"subject": "anatomy"}
        },
        "split": "test"
      },
      "overlap_protocol_spec": {"n": 5}
    },
    "part": "input"
  },
  "metric_scores": [0.043, 0.045, 0.125, 0.04, 0.077, 0.154, 0.154, 0.4, 0.059, 0.1, 0.083],
  "metric_protocol_spec": {
    "partial_overlap_spec": 1,
    "frequency_spec": {"filter_value": 0, "weighting": false}
  }
}
```
Here, each score represents the Jaccard overlap for one specific test question. For example, the 8th question in the dataset has a high overlap score of 0.4, meaning 40% of its n-grams appear in the training data.

#### 3. Token-level Overlap (partial_overlap_spec: 2)

Measures the proportion of individual tokens (words) in the test instance that are part of overlapping n-grams.

**How it's calculated**:
1. Identify all tokens that are part of any overlapping n-gram
2. Count these tokens (handling continuous sequences properly)
3. Divide by the total number of tokens in the test instance

**Interpretation**:
- More sensitive than Jaccard for detecting substantial overlaps
- Accounts for overlapping sequences rather than just individual n-grams
- Higher values indicate more of the actual text content is present in training data
- A value of 0.667 means ~67% of the words in the test instance appear in overlapping sequences in training data

**Example (MMLU Anatomy, 5-grams, input texts):**
```json
{
  "aggregate_data_overlap_key": {
    "stats_key": {
      "light_scenario_key": {
        "scenario_spec": {
          "class_name": "helm.benchmark.scenarios.mmlu_scenario.MMLUScenario",
          "args": {"subject": "anatomy"}
        },
        "split": "test"
      },
      "overlap_protocol_spec": {"n": 5}
    },
    "part": "input"
  },
  "metric_scores": [0.185, 0.192, 0.4, 0.172, 0.294, 0.353, 0.353, 0.667, 0.238, 0.357, 0.313],
  "metric_protocol_spec": {
    "partial_overlap_spec": 2,
    "frequency_spec": {"filter_value": 0, "weighting": false}
  }
}
```
Again, each score corresponds to one test question's token-level overlap. The 8th question has 66.7% of its tokens appearing in overlapping sequences in the training data, which is particularly concerning.

### Understanding Frequency Specifications

The system provides additional controls for measuring overlap:

- **filter_value**: When set to a value > 0 (e.g., 10), the system only counts overlaps from n-grams that appear infrequently in the training data (≤ specified count). This helps identify overlaps that are more likely to be meaningful rather than common phrases.

- **weighting**: When true, applies inverse frequency weighting, giving more importance to rare overlaps. This means n-grams that appear infrequently in training data contribute more to the overlap score than common phrases.

### Comparing Metrics for Better Interpretation

For a complete picture of contamination:

1. **Binary overlap (Is there any overlap?)**:
   - Start here to identify which instances have any overlap
   - If all scores are 1.0, every instance has some overlap with training data

2. **Jaccard overlap (What proportion of n-grams overlap?)**:
   - Provides a simple ratio of overlapping n-grams
   - Values > 0.2 warrant closer examination
   - Values > 0.4 suggest significant overlap

3. **Token-level overlap (What proportion of the content overlaps?)**:
   - Most sensitive to substantial content overlap
   - Values > 0.3 are concerning
   - Values > 0.5 strongly suggest the instance was likely seen during training

4. **Comparing input vs. reference parts**:
   - If reference answers have high overlap but inputs don't, models might have memorized answers
   - If both have high overlap, the entire instance might have been in training data

### Taking Action Based on Overlap Scores

- **Low overlap (Jaccard < 0.1, Token < 0.2)**: Likely just common phrases or coincidental matches
- **Medium overlap (Jaccard 0.1-0.3, Token 0.2-0.4)**: Possible contamination, review carefully
- **High overlap (Jaccard > 0.3, Token > 0.4)**: Strong evidence of contamination, consider excluding these instances

The system doesn't directly show the specific overlapping text in the aggregated metrics. To see the actual overlapping n-grams, you need to refer to the raw ngram output files that contain the detailed `EntryOverlapNgrams` records.

## Example Interpretation

When the system finds an overlap like:
```
(('is', 'most', 'likely', 'to', 'be'), 16)
```

This means the 5-gram "is most likely to be" appears 16 times in the training data, and also appears in at least one test instance. High frequency counts can indicate more significant data contamination.
