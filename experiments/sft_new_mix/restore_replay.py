# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restore only replay rows rejected by the superseded standalone line gate."""

import argparse
import hashlib
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fray.types import ResourceConfig
from levanter.store.cache import write_levanter_cache
from levanter.store.jagged_array import JaggedArrayStore
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.sft_new_mix.candidate_receipts import collect_inventory, expected_receipt, read_receipt
from experiments.sft_new_mix.exp3_build import BATCH_ROWS, build_shard
from experiments.sft_new_mix.exp3_filters import Stage, replay_reason
from experiments.sft_new_mix.exp3_runtime import (
    BUILD_IMPLEMENTATION_FILES,
    configure_storage,
    implementation_identity,
    tokenizer_for,
)
from experiments.sft_new_mix.job_manifests import read_job_manifest, stage_job_manifest
from experiments.sft_new_mix.prepare_overlay_gates import pinned_json

logger = logging.getLogger(__name__)


def restoration_identity() -> str:
    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update(implementation_identity(BUILD_IMPLEMENTATION_FILES).encode())
    return digest.hexdigest()


def restoration_path(job: dict) -> StoragePath:
    identity = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    return StoragePath(job["output_prefix"]) / "shards" / identity


def restore_shard(job: dict) -> dict:
    """Prove source-row disjointness, then rerun remaining gates on recovered rows."""
    if job["implementation_sha256"] != restoration_identity():
        raise ValueError("Replay restoration implementation changed")
    root = restoration_path(job)
    if (root / "receipt.json").exists():
        result = json.loads((root / "receipt.json").read_text())
        if result["job"] != job:
            raise ValueError("Restoration inputs changed")
        return {"receipt_uri": str(root / "receipt.json")}
    previous = pinned_json(job["base_receipt"])
    original = job["base_job"]
    if previous["job"] != original or previous["complete"] is not True or original["loss_mode"] != "full_token":
        raise ValueError("Restoration requires a complete original replay receipt")
    decisions = StoragePath(job["base_receipt"]["uri"]).parent / "decisions.jsonl"
    selected, retained = set(), set()
    decision_digest = hashlib.sha256()
    with decisions.open("rb") as stream:
        for line in stream:
            decision_digest.update(line)
            row = json.loads(line)
            if row["stage"] == "retained":
                retained.add(row["row"])
            if row["stage"] == Stage.REPLAY_HEURISTICS and row.get("reason") == "mean_line_length":
                index = row["row"]
                if index in selected or not job["source_row_start"] <= index < job["source_row_stop"]:
                    raise ValueError("Restoration row is repeated or outside its original source range")
                selected.add(index)
    expected = previous["stats"][Stage.REPLAY_HEURISTICS]["drop_reason:mean_line_length"]
    if len(selected) != expected or selected & retained:
        raise ValueError("Restored rows disagree with historical drops or duplicate existing retained rows")
    indices = np.asarray(sorted(selected), dtype="<i8")
    tokenizer = tokenizer_for(original["tokenizer_uri"], original["tokenizer_sha256"])
    store = JaggedArrayStore.open(prefix_join(original["input_path"], "input_ids"), mode="r", dtype=np.int32)
    raw_tokens = 0
    raw_digest = hashlib.sha256()

    def records():
        nonlocal raw_tokens
        for start in range(0, len(indices), BATCH_ROWS):
            batch = indices[start : start + BATCH_ROWS]
            for tokens in store.get_batch_sync(batch.tolist()):
                text = tokenizer.decode(tokens.tolist(), skip_special_tokens=False)
                lines = text.splitlines()
                if not lines or sum(map(len, lines)) / len(lines) <= 400:
                    raise ValueError("Recovered source bytes do not satisfy the historical line rejection")
                if replay_reason(text) not in ("", "templated_lines"):
                    raise ValueError("Recovered source bytes disagree with the historical rule order")
                raw_tokens += len(tokens)
                raw_digest.update(tokens.astype("<i4", copy=False).tobytes())
                yield {"input_ids": tokens}

    raw_path = str(root / "selected-original-rows")
    if StoragePath(raw_path).exists():
        # A preempted worker can leave this cache before the restoration receipt
        # lands; the selection is deterministic from the base receipt pin.
        logger.warning("Deleting unreceipted restoration cache: %s", raw_path)
        StoragePath(raw_path).rmtree()
    result = write_levanter_cache(
        records(),
        raw_path,
        metadata={"original_cache_uri": original["input_path"]},
        batch_size=BATCH_ROWS,
        max_pending_batches=2,
    )
    if result["count"] != len(indices):
        raise ValueError("Raw restoration lost selected original rows")
    row_bytes = indices.tobytes()
    (root / "original-row-indices.i64").write_bytes(row_bytes)
    candidate_job = {
        **original,
        "input_path": raw_path,
        "row_start": 0,
        "row_stop": len(indices),
        "expected_rows": len(indices),
        "expected_tokens": raw_tokens,
        "config_uri": job["config"]["uri"],
        "config_sha256": job["config"]["sha256"],
        "implementation_sha256": implementation_identity(BUILD_IMPLEMENTATION_FILES),
        "output_prefix": str(root / "candidate"),
    }
    build_shard(candidate_job)
    result = {
        "job": job,
        "candidate_job": candidate_job,
        "candidate_receipt_uri": expected_receipt(candidate_job),
        "selected_original_rows": len(indices),
        "selected_original_tokens": raw_tokens,
        "original_row_indices_uri": str(root / "original-row-indices.i64"),
        "original_row_indices_sha256": hashlib.sha256(row_bytes).hexdigest(),
        "original_selected_token_ids_sha256": raw_digest.hexdigest(),
        "historical_decisions_uri": str(decisions),
        "historical_decisions_sha256": decision_digest.hexdigest(),
        "disjoint_from_previous_retained_rows": True,
    }
    (root / "receipt.json").write_text(json.dumps(result, sort_keys=True))
    return {"receipt_uri": str(root / "receipt.json")}


def process_reference(reference: dict) -> dict:
    configure_storage()
    return restore_shard(read_job_manifest(reference))


def merge_restoration(base: dict, restored: dict) -> dict:
    """Keep unique original input counts while incorporating disjoint restored rows."""
    if not base["complete"] or not restored["complete"]:
        raise ValueError("Restoration requires complete populations")
    additions = {source["source"]: source for source in restored["sources"]}
    required = {
        row["source"]
        for row in base["transforms"]
        if row["transform"] == Stage.REPLAY_HEURISTICS and row.get("drop_reason:mean_line_length", 0)
    }
    if set(additions) != required:
        raise ValueError("Restoration omits sources with historical line rejections")
    transforms = {(row["source"], row["transform"]): row for row in restored["transforms"]}
    sources = []
    for source in base["sources"]:
        extra = additions.get(source["source"])
        if extra is None:
            sources.append(source)
            continue
        if source["loss_mode"] != "full_token" or source["source_revision"] != extra["source_revision"]:
            raise ValueError("Restoration changed source identity or supervision")
        counts = dict(source["counts"])
        for key in ("output_documents", "output_tokens", "supervised_prediction_positions"):
            counts[key] += extra["counts"][key]
        sources.append(
            {
                **source,
                "counts": counts,
                "shards": source["shards"] + extra["shards"],
                "restoration_counts": extra["counts"],
            }
        )
    current = []
    for row in base["transforms"]:
        extra = additions.get(row["source"])
        addition = transforms.get((row["source"], row["transform"]))
        if (
            extra is None
            or addition is None
            or row["transform"] not in (Stage.REPLAY_HEURISTICS, Stage.REPLAY_EMAILS, "junkgate2")
        ):
            current.append(row)
            continue
        numbers = Counter({key: value for key, value in row.items() if isinstance(value, int) and key != "complete"})
        numbers.update({key: value for key, value in addition.items() if isinstance(value, int) and key != "complete"})
        if row["transform"] == Stage.REPLAY_HEURISTICS:
            selected, tokens = extra["counts"]["input_documents"], extra["counts"]["input_tokens"]
            if row["drop_reason:mean_line_length"] != selected:
                raise ValueError("Restoration does not cover every superseded line rejection")
            numbers.update(
                input_documents=-selected, input_tokens=-tokens, dropped_documents=-selected, dropped_tokens=-tokens
            )
            numbers["drop_reason:mean_line_length"] = 0
            if (
                numbers["input_documents"] != numbers["output_documents"] + numbers["dropped_documents"]
                or numbers["input_tokens"] != numbers["output_tokens"] + numbers["dropped_tokens"]
            ):
                raise ValueError("Corrected replay rule accounting does not conserve original documents and tokens")
        current.append({**row, **numbers, "line_length_rejection_enabled": False})
    existing = {(row["source"], row["transform"]) for row in base["transforms"]}
    current.extend(
        {**row, "line_length_rejection_enabled": False}
        for key, row in transforms.items()
        if key not in existing and row["transform"] in (Stage.REPLAY_EMAILS, "junkgate2")
    )
    if additions.keys() - {source["source"] for source in base["sources"]}:
        raise ValueError("Restoration introduced an unknown source")
    return {
        **base,
        "sources": sources,
        "transforms": current,
        "training_ready": False,
        "replay_line_length_rejection_enabled": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--recipe-sha256", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()
    recipe = pinned_json({"uri": args.recipe, "sha256": args.recipe_sha256})
    plan, base = pinned_json(recipe["scripted_plan"]), pinned_json(recipe["base_inventory"])
    config = pinned_json(recipe["config"])
    if Stage.REPLAY_LINE_LENGTH in config["enabled"]:
        raise ValueError("Restoration must disable the standalone line rejection")
    originals = [job for job in plan["jobs"] if job["loss_mode"] == "full_token"]
    with ThreadPoolExecutor(max_workers=recipe["receipt_workers"]) as pool:
        receipts = list(pool.map(read_receipt, originals))
    jobs = []
    for original, receipt in zip(originals, receipts, strict=True):
        if receipt["complete"] is not True:
            raise ValueError("Original replay population is incomplete")
        if not receipt["stats"].get(Stage.REPLAY_HEURISTICS, {}).get("drop_reason:mean_line_length", 0):
            continue
        start, stop = (
            (original["row_start"], original["row_stop"]) if "row_start" in original else (0, original["expected_rows"])
        )
        jobs.append(
            {
                "base_job": original,
                "base_receipt": {"uri": receipt["receipt_uri"], "sha256": receipt["receipt_sha256"]},
                "config": recipe["config"],
                "output_prefix": recipe["output_prefix"],
                "implementation_sha256": restoration_identity(),
                "source_row_start": start,
                "source_row_stop": stop,
            }
        )
    root = StoragePath(recipe["output_prefix"])
    (root / "plan.json").write_text(json.dumps({"jobs": jobs}, sort_keys=True))
    with ZephyrContext(
        name="exp3-replay-line-restoration",
        max_workers=recipe["workers"],
        resources=ResourceConfig(cpu=2, ram="16g"),
        chunk_storage_prefix=str(root / "zephyr"),
        max_execution_retries=2,
    ) as context:
        references = [stage_job_manifest(job) for job in jobs]
        context.execute(Dataset.from_list(references).map(process_reference))
    candidate_jobs = [json.loads((restoration_path(job) / "receipt.json").read_text())["candidate_job"] for job in jobs]
    restored = collect_inventory(
        {"jobs": candidate_jobs, "source_loss": plan["source_loss"], "source_replacements": plan["source_replacements"]},
        recipe["receipt_workers"],
    )
    (root / "restored-only-inventory.json").write_text(json.dumps(restored, sort_keys=True))
    merged = merge_restoration(base, restored)
    merged["replay_restoration_recipe"] = {"uri": args.recipe, "sha256": args.recipe_sha256}
    (root / "inventory.json").write_text(json.dumps(merged, sort_keys=True))


if __name__ == "__main__":
    main()
