# Default Pipeline Steps

Marin provides a set of standard builders for the common stages of an LM experiment:
data download, tokenization, mixture assembly, training, and evaluation. Reach for these
before writing custom step code.

All builders return lazy `ArtifactStep[T]` handles (e.g. `ArtifactStep[TokenizedCache]` or
`ArtifactStep[LevanterCheckpoint]`) that `StepRunner` materializes on demand.

## Download

::: marin.experiment.data.hf_download

::: marin.experiment.data.raw_download

## Tokenization

::: marin.experiment.data.tokenized

::: marin.experiment.data.pretokenized

## Mixture assembly

::: marin.experiment.data.mixture

## Training

::: marin.experiment.train.train_lm

## Evaluation

::: marin.experiment.evaluation.EvalGroup

::: marin.experiment.evaluation.eval_step

::: marin.experiment.evaluation.eval_steps

::: marin.experiment.evaluation.eval_report
