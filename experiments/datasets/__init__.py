# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dataset catalogs for experiments.

Each leaf module is one dataset (or dataset family) and exposes lazy
``ArtifactStep[TokenizedCache]`` handles via :mod:`marin.experiment.data`:

- ``<name>_dataset()`` returns a single handle (a single-corpus dataset).
- ``<name>_datasets()`` returns ``dict[str, ArtifactStep[TokenizedCache]]`` keyed by
  subset, for a family. Mixture weights stay separate (``<NAME>_MIXTURE_WEIGHTS``);
  an experiment chooses weights and assembles them with ``mixture()``.

Each module is also runnable: ``python -m experiments.datasets.<name>`` prints its
build plan, and ``--run`` builds it (see :func:`marin.experiment.data.dataset_main`).

Import the catalog you need directly, e.g.
``from experiments.datasets.nemotron import nemotron_datasets``.
"""
