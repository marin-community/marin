# Default Pipeline Steps

Marin provides a set of standard builders for the common stages of an LM experiment:
data download, tokenization, mixture assembly, training, and evaluation. Reach for these
before writing custom step code.

All builders return lazy artifact handles (`Dataset`, `Checkpoint`, or a step spec) that
`StepRunner` materializes on demand.

## Tokenization

::: marin.experiment.data.tokenized

::: marin.experiment.data.hf_download

::: marin.experiment.data.derived

::: marin.experiment.data.pretokenized

## Mixture assembly

::: marin.experiment.data.mixture

## Training

::: marin.experiment.train.train_lm

## Evaluation

::: experiments.evals.evals.default_eval

::: experiments.evals.evals.default_key_evals
