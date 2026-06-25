# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Composable helpers for writing experiments as readable protocols.

The experiment file states the decisions inline (optimizer, evals, precision,
token budget, init source); these helpers carry the framework invariants (mesh
mapping, shuffle policy, Levanter plumbing) so they do not clutter the protocol.
This is the mechanism layer; the marin-specific *content* (which eval suite,
which validation sets) lives in ``experiments/``.
"""

from marin.experiment.evals import EvalSuite
from marin.experiment.train import Parallelism, WandbTracker, adam, mixture, train_lm, wandb

__all__ = ["EvalSuite", "Parallelism", "WandbTracker", "adam", "mixture", "train_lm", "wandb"]
