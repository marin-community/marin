# Experiments

At the [infrastructure level](../explanations/lazy-artifacts.md), an experiment is a DAG
of typed artifact handles to be materialized. Conceptually, an **experiment** represents a
unit of inquiry with a particular hypothesis or goal. Each such experiment is captured by a
GitHub issue with the `experiments` tag
(e.g., [#72](https://github.com/marin-community/marin/issues/72)).

An experiment might involve testing whether one optimizer is better than another in a
controlled setting, or trying out different tokenizers or data quality filtering schemes.
Regardless, an experiment consists of a DAG of steps.

To promote the reproducibility of experiments, we record all experiments in the
[experiments](https://github.com/marin-community/marin/tree/main/experiments) directory.
Each file in that directory (e.g.,
[tutorials/dclm_1b_1x_inline.py](https://github.com/marin-community/marin/blob/main/experiments/tutorials/dclm_1b_1x_inline.py))
corresponds to one experiment, where the naming convention contains the GitHub issue number.

Running each experiment produces provenance records for every artifact it builds. From
the data browser you can follow links to the Iris dashboard and W&B (for training steps).
