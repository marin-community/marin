# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned private CONSUMER receipts produced by the real RL runtime."""

import argparse
import json
from functools import cache
from pathlib import Path

from finestore.rl.curation import ConsumerReceipt

from experiments.post_training.task_curation.receipt_shards import read_receipt_shards, write_receipt_shards
from experiments.post_training.task_curation.registry import read_registry_json


@cache
def consumer_receipts() -> dict[str, ConsumerReceipt]:
    return read_receipt_shards(
        Path(__file__).with_name("consumer_receipt_refs"), ConsumerReceipt, "consumer", load_json=read_registry_json
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_bytes())
    receipts = {}
    for result in audit["results"]:
        receipt = ConsumerReceipt.model_validate(
            {key: value for key, value in result.items() if key in ConsumerReceipt.model_fields}
        )
        if receipt.runtime_commit != audit["config"]["runtime_commit"] or receipt.source_id in receipts:
            raise ValueError("Audit runtime or source identities differ")
        receipts[receipt.source_id] = receipt.model_dump(mode="json")
    directory = args.staged_root / "experiments/post_training/task_curation/datasets/consumer_receipt_refs"
    directory.mkdir(parents=True, exist_ok=False)
    write_receipt_shards(directory, receipts)


if __name__ == "__main__":
    main()
