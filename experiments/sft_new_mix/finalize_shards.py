# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply verified replacements, the real-call gate, and pinned Atlas row drops."""

import argparse
import hashlib
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum

import numpy as np
from fray.types import ResourceConfig
from levanter.store.cache import write_levanter_cache
from levanter.store.jagged_array import JaggedArrayStore
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.sft_new_mix.exp3_filters import MAX_TOKENS, Decision, Document, Stage, apply_stage
from experiments.sft_new_mix.exp3_runtime import (
    build_config,
    configure_storage,
    count_stage,
    implementation_identity,
    read_pinned,
    tokenizer_for,
)
from experiments.sft_new_mix.job_manifests import read_job_manifest, stage_job_manifest
from experiments.sft_new_mix.loss_modes import LossMode, prediction_weights
from experiments.sft_new_mix.run_tool_payloads import JOB_MANIFEST_WRITE_WORKERS

BATCH_ROWS = 128

logger = logging.getLogger(__name__)

# The files whose bytes determine final shard output; the shared runtime module
# is included so helper drift changes the pin.
FINALIZE_IMPLEMENTATION_FILES = ("exp3_runtime.py", "finalize_shards.py", "exp3_filters.py", "loss_modes.py")


class DropStage(StrEnum):
    LEXICAL = "atlas_lexical"
    EXACT_DUPLICATE = "atlas_exact_dedup"


def transform_shard(job: dict) -> dict:
    """Retain complete documents and preserve original rows without duplicate appends."""
    if job["implementation_sha256"] != implementation_identity(FINALIZE_IMPLEMENTATION_FILES):
        raise ValueError("Final candidate transformation implementation changed")
    identity = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    root = StoragePath(job["output_prefix"]) / "shards" / identity
    receipt_path = root / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["job"] != job or receipt["complete"] is not True:
            raise ValueError("Final shard receipt identity mismatch")
        return {"receipt_uri": str(receipt_path)}
    if (root / "cache").exists():
        # A preempted worker can leave the cache in place before its receipt
        # lands; output is deterministic from the job identity, so the retry
        # deletes it and rebuilds identical bytes.
        logger.warning("Deleting unreceipted final cache from a preempted worker: %s", root / "cache")
        (root / "cache").rmtree()
    source = job["source"]
    base = job["base_shard"]
    mode = LossMode(job["loss_mode"])
    if mode == LossMode.FULL_TOKEN and (job["overlays"] or job["teacher_only"]):
        raise ValueError("Tool replacements and teacher flags require chat format")
    enabled_drop_stages = {DropStage(stage) for stage in job["enabled_drop_stages"]}
    active_drops = {}
    for index, dispositions in job["drop_rows"].items():
        if not 0 <= int(index) < base["rows"]:
            raise ValueError("Atlas drop index lies outside the bound source cache")
        for stage in (DropStage.EXACT_DUPLICATE, DropStage.LEXICAL):
            if stage in dispositions and stage in enabled_drop_stages:
                active_drops[index] = {"stage": stage, "reason": dispositions[stage]}
                break
    if not active_drops and not job["overlays"] and not job["real_tools_enabled"]:
        original = json.loads(read_pinned(base["receipt_uri"], base["receipt_sha256"]))
        if (original["output_rows"], original["output_tokens"]) != (base["rows"], base["tokens"]):
            raise ValueError("Unchanged source population disagrees with its receipt")
        stats = defaultdict(Counter)
        for stage in enabled_drop_stages:
            for length in original["lengths"]:
                count_stage(stats, stage, length, length)
            for direction in ("input", "output"):
                stats[stage][direction + "_supervised_prediction_positions"] = original[
                    "supervised_prediction_positions"
                ]
        receipt = {
            "job": job,
            "complete": True,
            "source": source,
            "output_path": base["cache_uri"],
            "input_rows": base["rows"],
            "input_tokens": base["tokens"],
            "output_rows": base["rows"],
            "output_tokens": base["tokens"],
            "lengths": original["lengths"],
            "supervised_prediction_positions": original["supervised_prediction_positions"],
            "output_tokens_and_weights_sha256": original["output_tokens_and_weights_sha256"],
            "changed_documents": 0,
            "verified_modified_rows_selected": 0,
            "verified_modified_rows_previously_filtered": 0,
            "stats": stats,
        }
        receipt_path.write_text(json.dumps(receipt, sort_keys=True))
        return {"receipt_uri": str(receipt_path)}
    token_store = JaggedArrayStore.open(prefix_join(base["cache_uri"], "input_ids"), mode="r", dtype=np.int32)
    weight_store = JaggedArrayStore.open(prefix_join(base["cache_uri"], "loss_weights"), mode="r", dtype=np.float32)
    if len(token_store) != base["rows"] or len(weight_store) != base["rows"]:
        raise ValueError("Base cache row count changed")
    overlays = {}
    for row in job["overlays"]:
        if not job["overlay_enabled"]:
            break
        if row["source_shard"] != base["original_input_path"] or row["source"] != source:
            raise ValueError("Verified overlay was routed to a different source shard")
        if (row["status"], row["tool_calls_modified"], row["diff"]["modified"]) != ("verified", True, True):
            raise ValueError("Only verified, actually modified rows can replace original traces")
        if row["verified_original_sha256"] != row["input_ids_sha256"]:
            raise ValueError("Overlay lacks a verified binding to original token IDs")
        index = int(row["source_row"])
        if index in overlays:
            raise ValueError("Multiple replacements for one original row")
        overlays[index] = row
    if job["teacher_only"] and overlays:
        raise ValueError("Relay teacher boundaries cannot be rewritten by tool normalization")
    retained_map = {}
    if overlays:
        with StoragePath(base["decisions_uri"]).open("rt") as stream:
            for line in stream:
                value = json.loads(line)
                if value["stage"] == "retained" and value["row"] in overlays:
                    retained_map[int(value["output_row"])] = overlays[value["row"]]
    if mode == LossMode.FULL_TOKEN and (retained_map or job["teacher_only"]):
        raise ValueError("Tool replacements and teacher flags require chat format")
    tokenizer = tokenizer_for(job["tokenizer_uri"], job["tokenizer_sha256"])
    _, config = build_config(job["config_uri"], job["config_sha256"])
    prefix = tuple(tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n", add_special_tokens=False).ids)
    message_start = tokenizer.token_to_id("<|start_header_id|>")
    stats = defaultdict(Counter)
    selected = []
    lengths = []
    supervised = changed_rows = 0
    expected_digest = hashlib.sha256()
    with (root / "decisions.jsonl").open("wt") as audit:
        for start in range(0, len(token_store), BATCH_ROWS):
            indices = list(range(start, min(start + BATCH_ROWS, len(token_store))))
            token_rows = token_store.get_batch_sync(indices)
            weight_rows = weight_store.get_batch_sync(indices)
            for index, original, original_weights in zip(indices, token_rows, weight_rows, strict=True):
                if len(original) != len(original_weights) or original_weights[-1] != 0:
                    raise ValueError("Base prediction weights cross a document boundary or have unequal lengths")
                token_ids, weights = original, original_weights
                before_supervised = int(weights.sum())
                row_drop = active_drops.get(str(index))
                for stage in (DropStage.EXACT_DUPLICATE, DropStage.LEXICAL):
                    if stage not in enabled_drop_stages:
                        continue
                    reason = row_drop["reason"] if row_drop is not None and row_drop["stage"] == stage else ""
                    count_stage(stats, stage, len(token_ids), 0 if reason else len(token_ids), reason)
                    stats[stage]["input_supervised_prediction_positions"] += before_supervised
                    stats[stage]["output_supervised_prediction_positions"] += 0 if reason else before_supervised
                    if reason:
                        break
                if row_drop is not None:
                    stage, reason = row_drop["stage"], row_drop["reason"]
                    audit.write(json.dumps({"base_row": index, "stage": stage, "reason": reason}) + "\n")
                    continue
                overlay = retained_map.get(index)
                replacement_uri = None
                if overlay is not None:
                    token_ids = np.frombuffer(
                        read_pinned(overlay["payload_uri"], overlay["export_manifest"]["files"]["input_ids.i32"]),
                        dtype="<i4",
                    )
                    document = Document(
                        source, f"{identity}:{index}", tokenizer.decode(token_ids.tolist(), False), job["passed"]
                    )
                    for stage in Stage:
                        if stage not in config.enabled or stage == Stage.REAL_TOOLS:
                            continue
                        before = len(token_ids)
                        try:
                            decision = apply_stage(
                                document,
                                stage,
                                config,
                                lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids),
                            )
                        except (ValueError, json.JSONDecodeError) as error:
                            decision = Decision(document, "invalid_replacement_structure:" + type(error).__name__)
                        if decision.document.text != document.text:
                            token_ids = np.asarray(
                                tokenizer.encode(decision.document.text, add_special_tokens=False).ids, dtype=np.int32
                            )
                        count_stage(
                            stats,
                            "replacement_" + stage,
                            before,
                            len(token_ids) if decision.keep else 0,
                            decision.reason,
                            changed=decision.document.text != document.text,
                        )
                        document = decision.document
                        if not decision.keep:
                            audit.write(
                                json.dumps({"base_row": index, "stage": stage, "reason": decision.reason}) + "\n"
                            )
                            break
                    else:
                        weights = prediction_weights(
                            token_ids,
                            loss_mode=mode,
                            message_start_id=message_start,
                            assistant_prefix=prefix,
                            trained_assistant_turns=None,
                        )
                        replacement_uri = str(root / "edits" / f"{index:012d}.i32")
                        StoragePath(replacement_uri).write_bytes(token_ids.astype("<i4", copy=False).tobytes())
                    if replacement_uri is None:
                        continue
                    count_stage(stats, "verified_tool_replacement", len(original), len(token_ids), changed=True)
                if mode == LossMode.CHAT_ASSISTANT_ONLY and job["real_tools_enabled"]:
                    document = Document(
                        source, f"{identity}:{index}", tokenizer.decode(token_ids.tolist(), False), job["passed"]
                    )
                    try:
                        decision = apply_stage(
                            document,
                            Stage.REAL_TOOLS,
                            config,
                            lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids),
                        )
                    except (ValueError, json.JSONDecodeError):
                        decision = Decision(document, "invalid_actual_tool_call_structure")
                    count_stage(
                        stats, Stage.REAL_TOOLS, len(token_ids), len(token_ids) if decision.keep else 0, decision.reason
                    )
                    stats[Stage.REAL_TOOLS]["input_supervised_prediction_positions"] += int(weights.sum())
                    if not decision.keep:
                        audit.write(
                            json.dumps({"base_row": index, "stage": Stage.REAL_TOOLS, "reason": decision.reason}) + "\n"
                        )
                        continue
                    stats[Stage.REAL_TOOLS]["output_supervised_prediction_positions"] += int(weights.sum())
                if len(token_ids) > MAX_TOKENS or len(weights) != len(token_ids) or not weights.any():
                    raise ValueError("Final row violates context or prediction-weight invariants")
                changed_rows += not np.array_equal(original, token_ids) or not np.array_equal(original_weights, weights)
                selected.append((index, replacement_uri))
                lengths.append(len(token_ids))
                supervised += int(weights.sum())
                expected_digest.update(token_ids.astype("<i4", copy=False).tobytes())
                expected_digest.update(weights.astype("<f4", copy=False).tobytes())
                audit.write(
                    json.dumps(
                        {
                            "base_row": index,
                            "stage": "retained",
                            "output_row": len(selected) - 1,
                            "verified_replacement": overlay is not None,
                        }
                    )
                    + "\n"
                )
    output = base["cache_uri"]
    if len(selected) != base["rows"] or changed_rows:
        actual_digest = hashlib.sha256()

        def records() -> Iterator[dict[str, np.ndarray]]:
            for start in range(0, len(selected), BATCH_ROWS):
                batch = selected[start : start + BATCH_ROWS]
                indices = [index for index, _ in batch]
                for (_index, replacement), tokens, weights in zip(
                    batch, token_store.get_batch_sync(indices), weight_store.get_batch_sync(indices), strict=True
                ):
                    if replacement is not None:
                        tokens = np.frombuffer(StoragePath(replacement).read_bytes(), dtype="<i4")
                        weights = prediction_weights(
                            tokens,
                            loss_mode=mode,
                            message_start_id=message_start,
                            assistant_prefix=prefix,
                            trained_assistant_turns=None,
                        )
                    actual_digest.update(tokens.astype("<i4", copy=False).tobytes())
                    actual_digest.update(weights.astype("<f4", copy=False).tobytes())
                    yield {"input_ids": tokens, "loss_weights": weights}

        output = str(root / "cache")
        result = write_levanter_cache(
            records(),
            output,
            metadata={
                "source": source,
                "loss_mode": mode,
                "loss_weights_indexing": "prediction",
                "final_shard_job_sha256": identity,
            },
            batch_size=BATCH_ROWS,
            max_pending_batches=2,
        )
        if result["count"] != len(selected) or actual_digest.digest() != expected_digest.digest():
            raise ValueError("Final cache differs from inspected row selection")
    receipt = {
        "job": job,
        "complete": True,
        "source": source,
        "output_path": output,
        "input_rows": base["rows"],
        "input_tokens": base["tokens"],
        "output_rows": len(selected),
        "output_tokens": sum(lengths),
        "lengths": lengths,
        "supervised_prediction_positions": supervised,
        "changed_documents": int(changed_rows),
        "verified_modified_rows_selected": sum(replacement is not None for _, replacement in selected),
        "verified_modified_rows_previously_filtered": len(overlays) - len(retained_map),
        "output_tokens_and_weights_sha256": expected_digest.hexdigest(),
        "stats": stats,
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    return {"receipt_uri": str(receipt_path)}


def process_shard_reference(reference: dict) -> dict:
    configure_storage()
    return transform_shard(read_job_manifest(reference))


def execute_plan(plan: dict, workers: int) -> None:
    """Dispatch a frozen final-shard plan through bounded job references."""
    with ThreadPoolExecutor(max_workers=JOB_MANIFEST_WRITE_WORKERS) as pool:
        references = list(pool.map(stage_job_manifest, plan["jobs"]))
    (StoragePath(plan["output_prefix"]) / "job-references.json").write_text(json.dumps(references, sort_keys=True))
    with ZephyrContext(
        name="exp3-final-shards",
        max_workers=workers,
        resources=ResourceConfig(cpu=2, ram="8g"),
        chunk_storage_prefix=prefix_join(plan["output_prefix"], "zephyr"),
        max_execution_retries=2,
    ) as context:
        result = context.execute(Dataset.from_list(references).map(process_shard_reference))
    (StoragePath(plan["output_prefix"]) / "execution.json").write_text(json.dumps({"shards": result.results}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()
    execute_plan(json.loads(StoragePath(args.plan).read_text()), args.workers)


if __name__ == "__main__":
    main()
