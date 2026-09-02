# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.experiment.cli import experiment_main

from experiments.grug.moe_pallas_sconv.launch import grug_moe_pallas_sconv_gate1

if __name__ == "__main__":
    experiment_main(grug_moe_pallas_sconv_gate1)()
