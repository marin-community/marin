# Marin Speedrun

TODO (Nikil): README is currently a work in progress- will be refined as we make changes.

## Overview and Motivation

Marin Speedrun is a community-driven model training contest and framework where participants can train language models and submit them to Marin. Inspired by [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt), the goal is to further the mission of collaborative model development. We also maintain a leaderboard that tracks different models submitted along with the Pareto frontier of performance vs FLOPs.

## Tracks

**FineWeb-Edu track:** We fix the dataset to the [FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) by default and you can experiment with any model architecture and training approach (optimizers, learning rate schedules, or hyperparameters) you'd like.
**Varying datasets**: Varying the dataset will put the run in a different track on the leaderboard, but other than that we encourage you to try experimenting with datasets.


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
