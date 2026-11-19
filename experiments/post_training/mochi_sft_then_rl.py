# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the production Mochi SFT export and use it to initialize RL."""

from experiments.post_training import rl
from experiments.post_training.mochi_rl import MOCHI

if __name__ == "__main__":
    from rigging.filesystem.s3_compat import configure_coreweave_s3

    from experiments.post_training.mochi_sft import MIXTURE_262K_176NODE

    configure_coreweave_s3()
    raise SystemExit(rl.main(MOCHI, init=MIXTURE_262K_176NODE.model()))
