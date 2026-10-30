# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch one immutable Snowball SFT configuration through Iris."""

import draccus

from experiments.grug.moe.optimizer import GrugMoeMuonHConfig  # noqa: F401 - registers the optimizer choice
from experiments.grug.snowball_sft.train import GrugRunConfig, run_grug


@draccus.wrap()
def main(config: GrugRunConfig) -> None:
    run_grug(config)


if __name__ == "__main__":
    main()
