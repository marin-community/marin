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
