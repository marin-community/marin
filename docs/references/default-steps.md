# Default Pipeline Steps

Marin comes with a set of default pipeline steps that can be used to build experiments.
These steps are defined in `experiments.defaults` and are intended to be used as building blocks for experiments.

In general, you should reach for the default steps before writing your own.

## Downloading

::: experiments.defaults.default_download

## Exporting and Uploading

::: marin.export.upload_dir_to_hf

## Tokenization

::: experiments.defaults.default_tokenize

## Training

::: experiments.defaults.default_train

::: experiments.defaults.default_sft

::: experiments.defaults.simulated_epoching_train

## Evaluation

::: experiments.evals.evals.default_eval
