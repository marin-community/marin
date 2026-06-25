# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Composable helpers for writing experiments as readable protocols.

The experiment file states the decisions inline (optimizer, evals, precision,
token budget, init source); these helpers carry the framework invariants (mesh
mapping, shuffle policy, Levanter plumbing) so they do not clutter the protocol.
This is the mechanism layer; the marin-specific *content* (which eval suite,
which validation sets) lives in ``experiments/``.
"""

from marin.experiment.evals import EvalSuite as EvalSuite
from marin.experiment.train import Parallelism as Parallelism
from marin.experiment.train import WandbTracker as WandbTracker
from marin.experiment.train import adam as adam
from marin.experiment.train import mixture as mixture
from marin.experiment.train import train_lm as train_lm
from marin.experiment.train import wandb as wandb
