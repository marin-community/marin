# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for authoring experiments.

Lazy artifact builders live in :mod:`marin.experiment.data` (datasets and mixtures)
and :mod:`marin.experiment.sweep` (hyperparameter sweeps); :func:`train_lm` assembles a
training run as a lazy checkpoint; :class:`EvalSuite` pairs a set of harness tasks with
how often to run them. The marin-specific *content* (which eval suite, which validation
sets) lives in ``experiments/`` (see ``experiments.recipes``).
"""

from marin.experiment.evals import EvalSuite as EvalSuite
from marin.experiment.train import train_lm as train_lm
