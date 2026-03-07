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

### Prompt 3

push

### Prompt 4

commit swebm related

### Prompt 5

Great, can you also reduce epochs down to 1?

### Prompt 6

Great, can you launch an SFT on swebm minimax? Show me the command first before launch

### Prompt 7

Go ahead

### Prompt 8

monitor the job status, resubmit if the job gets pre-empted, let me know if it encounters other errors

### Prompt 9

hi

### Prompt 10

[Request interrupted by user]

### Prompt 11

Rename @experiments/exp2956_sft_swe_smith_qwen3_8b.py to @experiments/exp2956_sft_swe_smith.py and allow --student selection among qwen3-8b and qwen25-coder-32b-instruct

### Prompt 12

[Request interrupted by user for tool use]

