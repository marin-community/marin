# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a masked Snowball SFT token store from audited Harmony Parquet."""

import argparse
import logging

from marin.datakit.sft import SftInput, SftTokenStore, build_sft_store

from experiments.datasets.science_forward_converted import SOURCE_NAME

CONTEXT = 32_768
SHUFFLE_SEED = 0
MODEL_REPO = "open-athena/Snowball-67B-A2B-5.7T-Mixed-RLVR-Step38"
MODEL_REVISION = "cfc1d845dae89b067cdc7250d0164abefa5a69cf"
TOKENIZER = f"{MODEL_REPO}@{MODEL_REVISION}"


def prepare_store(input_path: str, output_path: str, tokenizer: str, num_shards: int, max_workers: int) -> SftTokenStore:
    """Build the packed store and reject any excluded overlength conversations."""
    if tokenizer != TOKENIZER:
        raise ValueError(f"Expected the pinned Step38 tokenizer {TOKENIZER}")
    result = build_sft_store(
        [SftInput(SOURCE_NAME, input_path)],
        output_path=output_path,
        tokenizer=tokenizer,
        max_length=CONTEXT,
        seed=SHUFFLE_SEED,
        num_shards=num_shards,
        max_workers=max_workers,
    )
    if any(count.overlength_conversations for count in result.sources.values()):
        raise ValueError("Converted conversations exceed the 32K training context; inspect the source counts")
    logging.info("Built Snowball SFT store: %s", result.model_dump_json())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True, help="Audited Harmony Parquet outputs/main directory")
    parser.add_argument("--output-path", required=True, help="Versioned destination for the token store")
    parser.add_argument("--tokenizer", required=True, help="Pinned Snowball tokenizer URL")
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    prepare_store(args.input_path, args.output_path, args.tokenizer, args.num_shards, args.max_workers)


if __name__ == "__main__":
    main()
