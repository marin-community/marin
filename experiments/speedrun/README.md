# Marin Speedrun

TODO (Nikil): README is currently a work in progress- refine as we make changes.

## Overview and Motivation

Marin Speedrun is a community-driven model training contest and framework that challenges participants to train language models under specific compute budgets. Inspired by NanoGPT's speedrun, the goal is to achieve the best possible model performance while staying within the allocated compute budget for each track.

## Tracks

The speedrun is divided into three tracks based on compute budget: **TINY**, **SMALL**, and **MEDIUM**.Each track has a specific FLOPs budget that your training run must stay within to qualify. We fix the dataset to the first <TBD> and you can experiment with any model architecture and training approach you'd like, as long as it stays within the corresponding compute budget.

## Setup and Usage

### Prerequisites
Follow the steps in [Marin's README](../../README.md) to set up your development environment. You will need to use Weights and Biases to track your runs, as we use it to track eval scores and relevant metrics.

### Running a Speedrun
Create a configuration for your run and use `default_speedrun` to train your model. This will train the model and generate an analysis file containing metrics and metadata.

An example is provided in [sample_run.py](experiments/speedrun/sample_run.py).

### Viewing the Leaderboard

```bash
# Start the leaderboard server
python experiments/speedrun/update_leaderboard.py --storage-path <path-to-run-directory>

# [Example for GCS storage]:
python experiments/speedrun/update_leaderboard.py --storage-path gs://marin-us-central2/checkpoints/speedrun
```

The leaderboard will be viewable at http://localhost:8000

## Submitting Your Run

1. Fork the Marin repository

2. Create a new branch for your run:
   ```bash
   git checkout -b speedrun/your-run-name
   ```

3. Add your run files to `experiments/speedrun/<your_run_name.py>`, containing the training script. Feel free to add docstrings describing your approach to the file.

4. Train your model on your compute resources (both GPUs and TPUs are supported), and add the `speedrun_analysis.json` file that's generated to [`data/runs.json`](data/runs.json).

5. Create a pull request with:
   - Title: `[Speedrun] Your Run Name - Track (e.g., TINY)`
   - Description:
     - Brief explanation of your approach
     - Key metrics (LM Eval Acc, C4-EN BPB)
     - Discussion of your approach

6. The CI workflow will then:
   - Validate your analysis file
   - Verify compute budget compliance
   - Update the leaderboard

7. Once merged, your run will appear on the public leaderboard at https://crfm.stanford.edu/marin/speedrun/

## Evaluation Metrics

- **LM Eval Accuracy**: Overall accuracy on language model evaluation benchmarks
- **C4-EN BPB (Bits per Byte)**: Compression performance on the C4 dataset (TODO: decide a concrete metric between these)
- **FLOPs Used**: Total compute used during training