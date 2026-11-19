# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train Snowball on the supervised mixture, then use its export to initialize RL."""

from experiments.post_training import rl
from experiments.post_training.snowball_rl import SNOWBALL

if __name__ == "__main__":
    # Spawned RL workers import this module without loading Levanter.
    from experiments.post_training.snowball_sft import MIXTURE_SFT

    raise SystemExit(rl.main(SNOWBALL, init=MIXTURE_SFT.model()))
