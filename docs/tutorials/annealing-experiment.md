# How to Run an Annealing Experiment

This guide explains how to evaluate dataset quality using microannealing experiments, similar to techniques used in the Llama-3, Olmo-2, and Mosaic papers.
The process involves cooldown training a pretrained model on a specific mixture of datasets and evaluating its impact on benchmark performance.
For example, we could do a cooldown on FineMath, and expect improvements on tasks like MATH and GSM8K.

## Prerequisites

- A pretrained model (default is a Marin Tootsie 8b model trained for ~800B tokens)
- A dataset or mixture of datasets you want to try

## Overview

1. **Configure the Dataset Mixture**: Use `LmDataConfig` to set up your dataset mixture.
2. **Set Up a Control Model** (optional): Create a control model for baseline comparison.
3. **Run the Annealing Experiment**: Use the `default_anneal` function to execute your experiment.


This tutorial is based on the [FineMath annealing experiment](https://github.com/marin-community/marin/blob/main/experiments/exp722_anneal.py).

## Steps

### 1. Configure the Dataset Mixture

First, create an [`AnnealConfig`][experiments.anneal_config.AnnealConfig] with your desired dataset mixture
using [`LmDataConfig`][levanter.data.text.LmDataConfig].
For example, to evaluate the impact of Finemath:

```python
finemath_anneal_config = AnnealConfig(
    dataset_config=lm_mixture_data_config(
        components={
            "finemath": finemath_3_plus_tokenized,
            "dolmino/dclm": dolmino_dclm,
        },
        weights={"finemath": 0.3, "dolmino/dclm": 0.7},
        permutation_type="linear",
    ),
)
```

### 2. Set Up a Control Model (Optional)

To establish a baseline, you can create a control model using only one dataset:

```python
control_dataset_config=lm_mixture_data_config(
    components={
        "dolmino/dclm": dolmino_dclm,
    },
    weights={"dolmino/dclm": 1.0},
    permutation_type="linear",
)

control_dclm_anneal_config = AnnealConfig(
    dataset_config=control_dataset_config,
)
```

### 3. Run the Annealing Experiment

Import the `default_anneal` function and run your experiment:

```python
from experiments.defaults import default_anneal

control_model = default_anneal(
    name="llama-8b-anneal-dclm",
    anneal_config=control_dclm_anneal_config
)
```

## Configuration Options

### Training Duration

You can adjust the number of tokens for annealing by modifying the `num_anneal_training_tokens` parameter in `AnnealConfig`.

### Model Checkpoint

The default checkpoint path points to a Marin Tootsie 8b model trained for approximately 800B tokens.
You can change this to use your own model or a different checkpoint.

## Example Experiments

For reference, see these example files:

- FineMath: [`experiments.exp722_anneal.py`](https://github.com/marin-community/marin/blob/main/experiments/exp722_anneal.py)
- Dolmino: [`experiments.dolmino.dolmino_anneal.py`](https://github.com/marin-community/marin/blob/main/experiments/dolmino/dolmino_anneal.py)
- Dolma: [`experiments.dolma.dolma_anneal.py`](https://github.com/marin-community/marin/blob/main/experiments/dolma/dolma_anneal.py)
- Cooldown: [`experiments.cooldown_anneal.py`](https://github.com/marin-community/marin/blob/main/experiments/cooldown_anneal.py)


## Troubleshooting

If you encounter issues:

1. Verify your dataset configurations are correct
2. Check that your model checkpoint path is valid
3. Ensure you have sufficient computational resources for the experiment

## Next Steps

After running your annealing experiment:

1. Evaluate the model on relevant benchmarks
2. Compare results with your control model
3. Analyze the impact of different dataset mixtures
