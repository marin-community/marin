# Experiments

At the [infrastructure level](../reference/executor.md), an experiment is simply a DAG of steps to be executed.
However, conceptually, an **experiment** represents a unit of inquiry with a
particular hypothesis or goal.
Each such experiment is captured by a GitHub issue with the `experiments` tag
(e.g., [#72](https://github.com/stanford-crfm/marin/issues/72)).

An experiment might involve testing whether one optimizer is better than another
in a controlled setting, or trying out different tokenizers or data quality
filtering schemes.  Regardless, an experiment consists of a sequence of steps.

To promote the reproducibility of experiments,
we record all experiments in the [experiments](https://github.com/stanford-crfm/marin/tree/main/experiments) directory.
Each file in that directory (e.g., [exp72_baselines.py](https://github.com/stanford-crfm/marin/blob/main/experiments/exp72_baselines.py)) corresponds to one experiment,
where the naming convention contains the GitHub issue number.

Running each experiment produces an experiment JSON file (see the
[executor documentation](../reference/executor.md)), which can be visualized specially
in the data browser.  From this experiments page in the data browser,
you can follow links to the Ray dashboard and wandb (for training steps).
