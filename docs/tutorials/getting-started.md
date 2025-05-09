# Getting Started with Marin

This tutorial will guide you through setting up Marin and running your first experiment.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.10 or higher
- pip (Python package manager)
- Git
- A [Weights & Biases](https://wandb.ai) account for experiment tracking (optional but recommended)


This document focuses on basic setup and usage of Marin. If you're on a GPU, see [local-gpu.md](local-gpu.md) for a GPU-specific walkthrough for getting started. If you want to set up a TPU cluster, see [TPU Setup](../how-to-guides/tpu-cluster-setup.md).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/marin-community/marin.git
   cd marin
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package:
   ```bash
   pip install -e .
   ```

## Hardware Setup

Marin supports multiple hardware configurations:

- **CPU**: Works out of the box, suitable for small experiments
- **GPU**: See [Local GPU Setup](local-gpu.md) for CUDA configuration and multi-GPU support
- **TPU**: See [TPU Setup](../how-to-guides/tpu-setup.md) for Google Cloud TPU configuration

## Running Your First Experiment

The easiest way to get started is to run our example experiment that trains a tiny language model on TinyStories. We defer explanation to the [First Experiment](first-experiment.md) tutorial.

```bash
python experiments/tutorial/train_tiny_model_cpu.py --prefix /tmp/marin-tutorial
```

This will:

1. Download and tokenize the TinyStories dataset to `/tmp/marin-tutorial/`
2. Train a tiny language model for 100 steps
3. Save the model checkpoint to `/tmp/marin-tutorial`

The experiment uses CPU by default, but you can modify it to use GPU or TPU by following the hardware setup guides above.

## Next Steps

Now that you have Marin set up and running, here are the recommended next steps:

1. Follow our [First Experiment](first-experiment.md) tutorial to understand how to create and run your own experiments
2. Learn about [Training Language Models](../how-to-guides/train-an-lm.md) - A comprehensive guide to training models with Marin
3. Read our [Language Modeling Overview](../lm/overview.md) to understand Marin's approach to language models
4. Explore Marin's key concepts in [Concepts](../explanation/concepts.md)
5. Learn about the [Executor framework](../explanation/executor.md) for managing experiments
6. Read about [Experiments](../explanation/experiments.md) to understand how we structure ML experiments
