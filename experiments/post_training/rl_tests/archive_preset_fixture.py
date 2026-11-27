# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve the archived Snowball preset against the test's local selection artifacts."""

import os

from experiments.post_training import rl, rl_compose
from experiments.post_training.snowball_rl import SNOWBALL as TABLE

rl_compose.SELECTION_ROOT = os.environ["MARIN_RL_TEST_SELECTION_ROOT"]


if __name__ == "__main__":
    raise SystemExit(rl.main(TABLE))
