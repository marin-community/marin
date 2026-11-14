# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind complete derived contract censuses to the measured source population."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from experiments.post_training.task_curation.derived_contract_census import DerivedContractReceipt
from experiments.post_training.task_curation.receipt_shards import write_receipt_shards
from experiments.post_training.task_curation.source import RlDataSource


class DerivedContractShard(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    receipt: DerivedContractReceipt


def bind_derived_contract_tags(source: RlDataSource, receipt: DerivedContractReceipt | None) -> RlDataSource:
    """Expose the derived contract shape only while its measured population binding stays current.

    A declared shape tag that contradicts the derived shape fails the source: the tag lies about
    the tasks it describes, and the Atlas must never display the contradiction. Unmeasured or
    stale populations keep their declared tags and stay pending instead.
    """
    dataset = source.dataset
    if (
        receipt is None
        or dataset is None
        or (source.info.id, dataset.name, dataset.revision, source.version)
        != (receipt.source_id, receipt.input.dataset_id, receipt.input.dataset_revision, receipt.input.converter_version)
    ):
        return replace(source, info=replace(source.info, derived_contract=None))
    assert receipt is not None
    summary = receipt.contract_summary()
    info = replace(source.info, derived_contract=summary)
    contradictions = receipt.contradictions(frozenset(source.info.tags))
    if contradictions:
        raise ValueError(
            f"Source {source.info.id} declares shape tags that contradict its derived contract: "
            + ", ".join(sorted(contradictions))
            + f". Derived shape: {summary.grader_type_counts}, agentic {summary.agentic_count}/"
            f"{summary.task_count}, single-turn {summary.single_turn_count}/{summary.task_count}, "
            f"reply-only {summary.reply_only_count}/{summary.task_count}. Remove or correct the "
            "declared shape tags to match the final task contracts."
        )
    return replace(source, info=info)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipts = {}
    for raw in json.loads(args.receipts.read_text()):
        receipt = DerivedContractReceipt.model_validate(raw)
        if receipt.source_id in receipts:
            raise ValueError(f"Duplicate derived contract census: {receipt.source_id}")
        receipts[receipt.source_id] = receipt.model_dump(mode="json")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_receipt_shards(args.output_dir, receipts)


if __name__ == "__main__":
    main()
