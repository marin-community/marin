# Marin Speedrun

## Overview and Motivation

Marin Speedrun is a community-driven model training contest and framework that challenges participants to train language models under specific compute budgets. Inspired by NanoGPT's speedrun, the goal is to achieve the best possible model performance while staying within the allocated compute budget for each track. 

Key features:
- Compute budget tracking with FLOPs estimation
- Automated evaluation metrics (LM Eval accuracy, C4 BPB)
- Real-time leaderboard with track-specific rankings
- Support for various model architectures and training approaches

## Tracks

The speedrun is divided into three tracks based on compute budget: **TINY**, **SMALL**, and **MEDIUM**.Each track has a specific FLOPs budget that your training run must stay within to qualify.

## Setup and Usage

### Prerequisites

Follow the steps in [README.md](README.md) to set up your development environment. 

### Running a Speedrun
Create a configuration for your run and use `default_speedrun` to train your model. This will train the model and generate an analysis file containing metrics and metadata.

An example is provided in [sample_run.py](experiments/speedrun/sample_run.py).

### Viewing the Leaderboard

```bash
# Start the leaderboard server
python experiments/speedrun/update_leaderboard.py --storage-path <path-to-run-directory>

# [Example for GCS storage]:
--storage-path gs://marin-us-central2/checkpoints/speedrun
```

The leaderboard will be available at http://localhost:8000

## Submitting Your Run

1. Fork the Marin repository

2. Create a new branch for your run:
   ```bash
   git checkout -b speedrun/your-run-name
   ```

3. Add your run files to `experiments/speedrun/runs/your-run-name/`:
   - `run.py`: Your training script
   - `speedrun_analysis.json`: Generated analysis file
   - `README.md`: (Optional) Description of your approach

4. Create a pull request with:
   - Title: `[Speedrun] Your Run Name - Track (e.g., TINY)`
   - Description:
     - Brief explanation of your approach
     - Key metrics (LM Eval Acc, C4-EN BPB)
     - Discussion of findings and approach

5. The CI workflow will:
   - Validate your analysis file
   - Verify compute budget compliance
   - Update the leaderboard

Once merged, your run will appear on the public leaderboard at https://crfm.stanford.edu/marin/speedrun/

## Evaluation Metrics

- **LM Eval Accuracy**: Overall accuracy on language model evaluation benchmarks
- **C4-EN BPB (Bits per Byte)**: Compression performance on the C4 dataset
- **FLOPs Used**: Total compute used during training
