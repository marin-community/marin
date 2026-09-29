# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the 234.382B-token A/B SFT mix outside the Snowball pretraining mix.

The allocation uses the scored SFT source list and corrected Terminus token counts.
The Terminus entries in the store manifest must use stores built after the reasoning parser fix.
"""

import argparse
import logging
from pathlib import Path

from rigging.log_setup import configure_logging

from experiments.grug_sft.special_token_lr import train

RUN_ID = "grug-67b-sft-20260928-special-token-lr-full-ab-4366"
MIX_PATH = Path(__file__).with_name("special_token_full_ab_mix.json")
STEPS = 4366


if __name__ == "__main__":
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--stores-manifest", required=True)
    args = parser.parse_args()
    train(STEPS, args.stores_manifest, mix_path=MIX_PATH, run_id=RUN_ID)
