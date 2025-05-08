# Default Pipeline Steps

Marin comes with a set of default pipeline steps that can be used to build experiments.
These steps are defined in `experiments.defaults` and are intended to be used as building blocks for experiments.

In general, you should reach for the default steps before writing your own.

## Downloading

TODO

## Tokenization

::: experiments.defaults.default_tokenize

## Training

::: experiments.defaults.default_train

::: experiments.defaults.default_anneal

::: experiments.defaults.default_sft

::: experiments.defaults.simulated_epoching_train

## Scaling Law Prediction

::: marin.scaling_laws.create_ladder_suite.scaling_law_suite

::: experiments.defaults.default_scaling_law_pred

## Evaluation

::: experiments.evals.evals.default_eval
