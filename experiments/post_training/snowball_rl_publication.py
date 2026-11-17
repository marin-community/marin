# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish pinned Snowball controls to the RL progress dashboard."""

from experiments.post_training import rl_publication
from experiments.post_training.snowball_rl import SNOWBALL

if __name__ == "__main__":
    rl_publication.control_command(SNOWBALL)()
