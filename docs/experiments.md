# Experiments

At the [infrastructure level](docs/executor.md), an experiment is simply a DAG of steps to be executed.
However, conceptually, an **experiment** represents a unit of inquiry with a
particular hypothesis or goal.
This might involve testing whether one optimizer is better
than another in a controlled setting,
and include a series of steps corresponding to training different models on the same dataset.

Each such experiment is captured by a GitHub issue with the `experiments` tag
(e.g., [#72](https://github.com/stanford-crfm/marin/issues/72)).

To promote the reproducibility of experiments,
we created the `experiments` directory.
Each file corresponds to one experiment (e.g., [exp72_baselines.py](experiments/exp72_baselines.py)).
The GitHub issue should link to the experiment page,
which further links to the actual results.
