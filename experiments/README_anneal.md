# Evaluating a dataset quickstart guide

This is a guide that goes through how to evaluate the quality of a dataset. We use a similar technique to the Llama-3, Olmo-2, and Mosaic papers, which is to cooldown a pretrained
model on a desired mixture of datasets. Then, we evaluate the effect of the mixture on a set of benchmarks. For example, we can cooldown a model on Finemath, and we would
expect the model to show improvements on GSM8K and MMLU mathematics benchmarks.

## How to run an anneal experiment

Currently, the annealing process requires just one change, which is to provide a `LMMixtureDatasetConfig` to the `AnnealConfig`.
For example, if I wanted to do an ablation of the effects of Finemath on the model, I would do the following:
```python
finemath_anneal_config = AnnealConfig(
    dataset_config=lm_mixture_data_config(
        components={
            "finemath": finemath_3_plus_tokenized,
            "dolmino": dolmino_dclm,
        },
        weights={"finemath": 0.3, "dolmino": 0.7},
    ),
)
```

And, if I wanted to set a control model that is just 100% Dolmino DCLM, I would do the following:
```python
control_dataset_config=lm_mixture_data_config(
        components={
            "dolmino": dolmino_dclm,
        },
        weights={"dolmino": 1.0},
    )
```

Then, you can simply run the experiment by importing the `default_anneal` function from `experiments.defaults` and running the following command:
```
control_model = default_anneal(name="llama-8b-anneal-dclm", anneal_config=control_dclm_anneal_config)
```

The other thing that could be changed is the number of tokens that you want to anneal for. This can be changed in the `AnnealConfig` by changing the `num_anneal_training_tokens` parameter.
Some other things to note is we have a `DEFAULT_CHECKPOINT_PATH` that is an llama-8b model trained for roughly 800B tokens. You may want to change this path to your own model or a future checkpoint.


### Examples
See more examples of how to run an anneal experiment in the following files:
- Finemath: experiments.exp722_anneal.py
- Dolmino: experiments.dolmino.dolmino_anneal.py
- Dolma: experiments.dolma.dolma_anneal.py
- Cooldown: experiments.cooldown_anneal.py
