# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for authoring experiments.

Lazy artifact builders live in :mod:`marin.experiment.data` (datasets and mixtures)
and :mod:`marin.experiment.sweep` (hyperparameter sweeps); :class:`EvalSuite` pairs a
set of harness tasks with how often to run them. The marin-specific *content* (which
eval suite, which validation sets) lives in ``experiments/`` (see ``experiments.recipes``).
"""

from marin.experiment.evals import EvalSuite as EvalSuite
