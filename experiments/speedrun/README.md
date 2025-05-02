# Marin Speedrun

TODO (Nikil): README is currently a work in progress- will be refined as we make changes.

## Overview and Motivation

Marin Speedrun is a community-driven model training contest and framework that challenges participants to train language models under specific compute budgets. Inspired by [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt), the goal is to achieve the best possible model performance while staying within the allocated compute budget for each track.

## Tracks

The speedrun is divided into three tracks based on compute budget: **TINY**, **SMALL**, and **MEDIUM**.Each track has a specific FLOPs budget that your training run must stay within to qualify. We fix the dataset to the [FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and you can experiment with any model architecture and training approach you'd like, as long as it stays within the corresponding compute budget.

## Setup and Usage

### Prerequisites
Follow the steps in [Marin's README](../../README.md) to set up your development environment. You will need to use Weights and Biases to track your runs, as we use it to track eval scores and relevant metrics. TODO (Nikil): add a guide or link for setting up levanter's `.levanter.yaml`.

### Running a Speedrun
Create a configuration for your run and use `default_speedrun` to train your model. This will train the model and generate an analysis file containing metrics and metadata.

An example is provided in [llama_75m_fineweb_edu/llama_75m_fineweb_edu.py](llama_75m_fineweb_edu/llama_75m_fineweb_edu.py).

### Viewing the Leaderboard

```bash
# Start the leaderboard server
python -m marin.speedrun.update_leaderboard --storage-path <path-to-run-directory>

# [Example for GCS storage]:
python -m marin.speedrun.update_leaderboard --storage-path gs://marin-us-central2/checkpoints/speedrun
```

The leaderboard will be viewable at http://localhost:8000

## Submitting Your Run

1. Create a directory for your run at `experiments/speedrun/<your_run_name>/` and add your training script as `<your_run_name>.py`.

2. Train your model (both GPUs and TPUs are supported), and add the `speedrun_results.json` file that's generated to [`experiments/speedrun/<your_run_name>/speedrun_results.json`]. Examples of how to define a run and how the results file looks are available in [llama_75m_fineweb_edu/llama_75m_fineweb_edu.py](llama_75m_fineweb_edu/llama_75m_fineweb_edu.py) and [llama_75m_fineweb_edu/speedrun_results.json](llama_75m_fineweb_edu/speedrun_results.json) respectively.

3. Create a pull request with a brief explanation of your approach.

4. We will then review it manually and merge it if it satisfies the requirements.

5. Once merged, your run will appear on the public leaderboard at https://crfm.stanford.edu/marin/speedrun/

## Evaluation Metrics

- **C4-EN BPB (Bits per Byte)**: Compression performance on the C4 dataset (TODO: decide a concrete metric between these)
- **FLOPs Used**: Estimate of total compute used during training
