# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train one packed epoch of the A/B SFT mix outside the Snowball pretraining mix.

The allocation includes scored A/B sources, science tool-use chats, and corrected Terminus counts.
The Terminus entries in the store manifest must use stores built after the reasoning parser fix.
The 234.421B source tokens occupy 968,596 packed examples (253.912B positions).
All are scheduled exactly once across 19 complete mixture blocks. Sources with fewer
than 19 examples share a shuffled component; larger sources get a block allocation
of their own. allocations_tokens in the mix denotes packed positions.
"""

import argparse
import logging
from pathlib import Path

from rigging.log_setup import configure_logging

from experiments.grug_sft.special_token_lr import train

RUN_ID = "grug-67b-sft-20260929-special-token-lr-full-ab-4750-grouped"
MIX_PATH = Path(__file__).with_name("special_token_full_ab_grouped_mix.json")
STORE_MANIFEST = "gs://marin-us-central2/grug_sft/stores/full-ab-packed-2026.09.29.json"
STEPS = 4750


if __name__ == "__main__":
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--stores-manifest", default=STORE_MANIFEST)
    args = parser.parse_args()
    train(STEPS, args.stores_manifest, mix_path=MIX_PATH, run_id=RUN_ID)
