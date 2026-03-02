# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Plan: Add swebm-minimax-m2.5 teacher to exp2956

## Context
Add SWE-bench-multilingual minimax-m2.5 as a new "teacher" entry (`swebm-minimax-m2.5`) for sanity-check fine-tuning. 299 samples. Minimal changes.

## Changes

### 1. `experiments/posttrain/instruction_datasets.py` (after line 712)
Add dataset config for `AlienKevin/SWE-bench-multilingual-minimax-m2.5-trajectories` (revision `479f2e1`).

### 2. `experiments/exp2956_sft_swe_smith_qwen3_8b.py`
- Add `"swe...

### Prompt 2

Great, can you commit the GLM changes first?

