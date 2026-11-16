# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover recorded-pass AgentTrove rows from the pinned upstream source."""

import argparse
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from huggingface_hub import hf_hub_download
from levanter.store.cache import write_levanter_cache
from marin.datakit.chat_normalize import normalize_chat_record
from marin.datakit.chat_render import render_chat_record
from marin.datakit.download.agenttrove import result_to_tag, row_to_chat_doc
from marin.datakit.download.rollout_transforms import TRAJECTORY_SOLVED_TAG
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.sft_new_mix.exp3_build import process_shard
from experiments.sft_new_mix.exp3_runtime import (
    BUILD_IMPLEMENTATION_FILES,
    configure_storage,
    implementation_identity,
    tokenizer_for,
)


def materialize_file(job: dict) -> dict:
    configure_storage()
    identity = job["raw_file"]["sha256"]
    root = StoragePath(job["output_prefix"]) / "raw-selections" / identity
    receipt_path = root / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["job"] != job:
            raise ValueError("AgentTrove source selection changed")
        return {"receipt_uri": str(receipt_path)}
    if (root / "recorded-pass-input").exists():
        raise ValueError("Unreceipted AgentTrove input preserved; choose a new rebuild version")
    raw_path = hf_hub_download(job["dataset"], job["raw_file"]["path"], revision=job["revision"], repo_type="dataset")
    with open(raw_path, "rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != identity:
            raise ValueError("Pinned AgentTrove source file digest changed")
    tokenizer = tokenizer_for(job["tokenizer_uri"], job["tokenizer_sha256"])
    counts = Counter()
    source_digest = hashlib.sha256()
    input_cache = str(root / "recorded-pass-input")

    def records() -> Iterator[dict[str, np.ndarray]]:
        with (root / "source-rows.jsonl").open("wt") as audit:
            index = 0
            for batch in pq.ParquetFile(raw_path).iter_batches(batch_size=16):
                for raw in batch.to_pylist():
                    index += 1
                    counts["raw_documents"] += 1
                    if result_to_tag(raw.get("result")) != TRAJECTORY_SOLVED_TAG:
                        counts["missing_or_failed_recorded_outcome"] += 1
                        continue
                    counts["recorded_pass_documents"] += 1
                    converted = row_to_chat_doc(raw)
                    if not converted:
                        counts["existing_adapter_rejected"] += 1
                        continue
                    if len(converted) != 1:
                        raise ValueError("AgentTrove adapter emitted multiple documents for one source row")
                    try:
                        normalized = normalize_chat_record(converted[0], "messages", "id")
                    except ValueError:
                        counts["existing_chat_normalizer_rejected"] += 1
                        continue
                    rendered = render_chat_record(normalized)
                    ids = np.asarray(tokenizer.encode(rendered["text"], add_special_tokens=False).ids, dtype=np.int32)
                    counts["eligible_documents"] += 1
                    counts["eligible_tokens"] += len(ids)
                    source_digest.update(ids.astype("<i4", copy=False).tobytes())
                    audit.write(
                        json.dumps(
                            {
                                "source_row": index - 1,
                                "source_id": rendered["id"],
                                "result": raw["result"],
                                "output_row": counts["eligible_documents"] - 1,
                            }
                        )
                        + "\n"
                    )
                    yield {"input_ids": ids}

    write_levanter_cache(
        records(),
        input_cache,
        metadata={
            "source": "agenttrove",
            "source_revision": job["revision"],
            "raw_file_sha256": identity,
            "selection": "recorded pass and existing source adapter",
        },
        batch_size=16,
        max_pending_batches=2,
    )
    selection = {
        "dataset": job["dataset"],
        "revision": job["revision"],
        "raw_file": job["raw_file"],
        "counts": counts,
        "eligible_token_ids_sha256": source_digest.hexdigest(),
        "input_token_measurement": "canonical rendered recorded-pass rows only; raw upstream token counts unavailable",
    }
    selection_bytes = json.dumps(selection, sort_keys=True).encode()
    (root / "selection.json").write_bytes(selection_bytes)
    built = None
    if counts["eligible_documents"]:
        build_job = {
            "source": "agenttrove",
            "input_path": input_cache,
            "expected_rows": counts["eligible_documents"],
            "expected_tokens": counts["eligible_tokens"],
            "row_start": 0,
            "row_stop": counts["eligible_documents"],
            "loss_mode": "chat_assistant_only",
            "source_revision": job["revision"],
            "passed": True,
            "pass_receipt_sha256": hashlib.sha256(selection_bytes).hexdigest(),
            "output_prefix": str(StoragePath(job["output_prefix"]) / "candidate"),
            "implementation_sha256": implementation_identity(BUILD_IMPLEMENTATION_FILES),
            **{key: job[key] for key in ("config_uri", "config_sha256", "tokenizer_uri", "tokenizer_sha256")},
        }
        built = process_shard(build_job)
        built.pop("lengths", None)
    receipt = {"job": job, "complete": True, "selection": selection, "candidate_shard": built}
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    return {"receipt_uri": str(receipt_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()
    content = StoragePath(args.plan).read_bytes()
    plan = json.loads(content)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != plan["implementation_sha256"]:
        raise ValueError("AgentTrove rebuild implementation changed")
    with ZephyrContext(
        name="exp3-recorded-pass-agenttrove",
        max_workers=args.workers,
        resources=ResourceConfig(cpu=2, ram="16g", disk="20g"),
        chunk_storage_prefix=prefix_join(plan["output_prefix"], "zephyr"),
        max_execution_retries=2,
    ) as context:
        result = context.execute(Dataset.from_list(plan["jobs"]).map(materialize_file))
    receipts = [json.loads(StoragePath(value["receipt_uri"]).read_text()) for value in result.results]
    (StoragePath(plan["output_prefix"]) / "manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "plan_uri": args.plan,
                "plan_sha256": hashlib.sha256(content).hexdigest(),
                "source": "agenttrove",
                "shards": receipts,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
