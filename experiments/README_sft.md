# SFT (Supervised Fine-Tuning) Quickstart Guide

## Overview
Guide for running supervised fine-tuning (SFT) experiments using Marin. Assumes Ray dashboard is running per setup docs.
The default doc reproduces OLMO SFT

## Key Steps

### 1. Basic Commands
# run Olmo sft with
```
python marin/run/ray_run.py --env_vars HF_TOKEN -- python experiments/227_sft.py
```

### 2. Configure Dataset
```python
instruction_dataset = get_instruction_dataset("your/dataset")  # e.g., "allenai/tulu-v2-sft-mixture"
```

Please look at `instruction_datasets.py` for examples on how to process SFT datasets.

### 3. Tokenization Configuration
In `tokenize_step`, key parameters to customize:
- `name`: `"tokenized/your_experiment_name"`
- `train_paths`: Location of training data
- `cache_path`: `"gs://your-bucket/tokenized/sft_cache/experiment-name"`
- `tokenizer`: Must match base model tokenizer
- `seq_len`: Maximum sequence length (typically 2048)
- `input_field`/`output_field`: Match your data format (default "user"/"assistant")

### 4. Training Configuration
In `train_step`, essential parameters:
- `name`: `"checkpoints/your_experiment_name"`
- `tpu_type`: e.g., "v4-8", "v4-128" based on needs
- `epoch`: Number of training epochs
- `train_batch_size`: Batch size (adjust for your TPU)
- `initialize_from`: Path to base model checkpoint
- Cache directory should match tokenization step's output

### 5. Running

```bash
# Basic run
python marin/run/ray_run.py --env_vars HF_TOKEN -- python experiments/my_sft.py

# Force specific steps
python marin/run/ray_run.py --env_vars HF_TOKEN -- python experiments/my_sft.py --force_run '["your_step_name"]'
```

### Storage Paths

All experiment names should follow the convention below, which is specifying the top level
directory undernead `marin-us-central2`. For example, the name for the `tokenize_step` should
start with the prefix `tokenized/`

- Base models: `gs://levanter-checkpoints/`
- Your experiments: `gs://marin-us-central2/checkpoints/`
- Tokenized cache: `gs://marin-us-central2/tokenized/sft_cache/`

### Common TPU Configurations
- Small experiments: `"v4-64"`
- Production runs: `"v4-128"`
- Batch sizes should be scaled accordingly, I get 57 MFU with batch size 1024 on v4-128

## Tips
- Match tokenizer to base model
- Monitor via Ray dashboard and W&B
- Use `--force_run` during development to re-run jobs or just delete executor state
- Adjust batch size based on TPU size

## Troubleshooting
- Verify TPU availability before large runs
- Check tokenizer/model compatibility
- Ensure GCS paths are accessible
- Monitor memory usage with large batch sizes

Reach out to Ahmed with questions
