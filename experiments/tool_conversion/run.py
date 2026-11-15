# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run conversion over this campaign's pinned source shards.

Read README.md, "Scope and limits — read before reuse", before using this entry point.
"""

import argparse
import hashlib
import json
import logging
import os
import time
from collections import Counter
from dataclasses import asdict

import numpy as np
from levanter.store.tree_store import TreeStore
from rigging.filesystem.cluster_config import StoreType, data_buckets
from rigging.filesystem.s3_compat import configure_coreweave_s3, s3_credentials
from rigging.filesystem.storage_path import StoragePath, prefix_join
from tokenizers import Tokenizer

from experiments.tool_conversion.archive import commit_batch, json_bytes, provenance, read_json, write_immutable
from experiments.tool_conversion.convert import (
    SCOPE_GUIDE,
    CampaignScopeError,
    Conversion,
    Disposition,
    convert_trace,
    excluded_source,
    parse_trace,
    visible_calls,
)
from experiments.tool_conversion.export import digest, export_conversion

LOGGER = logging.getLogger(__name__)
HEADER_ID = 128006
HEADER_END_ID = 128007
CALL_ID = 128005
ASSISTANT_ID = 78191
TOKENIZER_SHA256 = "6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45"
SELECTION_SHA256 = frozenset(
    {
        "a3a614704c0af8baf5c20de4d57e9071ef80a77a7479a0223be8966335356058",
        "64e8bdbe70b2f50d99281e7c0dfb420e7a664659d1b09aa2b33700db544af342",
        "8fd6c1959ec3970e1841a5ec3e8ec59f9b8fc73afe078a1a562d50d667e2f846",
    }
)


def validate_campaign(selection: dict, tokenizer: Tokenizer, tokenizer_sha: str) -> None:
    """Reject unreviewed catalogs and tokenizers before reading shards or writing outputs."""
    if digest(json_bytes(selection)) not in SELECTION_SHA256:
        raise CampaignScopeError(f"Unsupported pinned selection or selection schema; see {SCOPE_GUIDE}")
    if tokenizer_sha != TOKENIZER_SHA256 or selection["tokenizer_sha256"] != tokenizer_sha:
        raise CampaignScopeError(f"Unsupported tokenizer; see {SCOPE_GUIDE}")
    for token, expected in [
        ("<|start_header_id|>", HEADER_ID),
        ("<|end_header_id|>", HEADER_END_ID),
        ("<tool_call>", CALL_ID),
        ("assistant", ASSISTANT_ID),
    ]:
        if tokenizer.token_to_id(token) != expected:
            raise CampaignScopeError(f"Unsupported tokenizer special-token IDs; see {SCOPE_GUIDE}")


def has_native_calls(ids: np.ndarray) -> bool:
    """Cheap selection: declarations in system turns do not select a document."""
    heads = np.flatnonzero(ids == HEADER_ID)
    for pos in np.flatnonzero(ids == CALL_ID):
        h = int(np.searchsorted(heads, pos)) - 1
        if (
            h >= 0
            and heads[h] + 2 < len(ids)
            and ids[heads[h] + 1] == ASSISTANT_ID
            and ids[heads[h] + 2] == HEADER_END_ID
        ):
            return True
    return False


def convert_record(
    row: dict, ids: np.ndarray, tokenizer: Tokenizer, tokenizer_sha: str, name: str
) -> tuple[dict, dict[str, bytes]]:
    identity = hashlib.sha256(f'{row["source"]}\n{row["input_path"]}\n{row["row"]}'.encode()).hexdigest()
    item = {
        "id": identity,
        "source": row["source"],
        "shard": row["input_path"],
        "row": row["row"],
        "revision": row.get("source_revision"),
        "input_ids_sha256": digest(ids.tobytes()),
        "tokens": len(ids),
    }
    if excluded_source(row["source"]):
        return {**item, "status": Disposition.EXCLUDED, "reason": "dropped source or Terminus"}, {}
    if tokenizer_sha != TOKENIZER_SHA256:
        raise CampaignScopeError(f"Unsupported tokenizer; see {SCOPE_GUIDE}")
    if not np.any(ids == HEADER_ID):
        raise CampaignScopeError(f"Unsupported chat format; see {SCOPE_GUIDE}")
    if not has_native_calls(ids):
        return {**item, "status": Disposition.NO_CALLS}, {}
    text = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
    item["text_sha256"] = digest(text.encode())
    converted = convert_trace(row["source"], text, identity)
    try:
        _, original_turns = parse_trace(text)
        item["observed_tools"] = sorted(
            {
                json.loads(match[1])["name"]
                for turn in original_turns
                if turn.role == "assistant"
                for match in visible_calls(turn.body)
            }
        )
    except (ValueError, KeyError, TypeError) as error:
        item["tool_parse_error"] = str(error)
    files = {}
    record_prefix = f"{name}/run/records/{identity}/"
    if converted.disposition == Disposition.VERIFIED:
        try:
            exported = export_conversion(converted, tokenizer, tokenizer_sha)
        except ValueError as error:
            converted = Conversion(Disposition.RESIDUAL, str(error), text)
        else:
            source_name = f"{name}/sources/{identity}/source.json"
            source = {**item, "text": text}
            files[source_name] = json_bytes(source)
            files[f"{name}/sources/{identity}/input_ids.i32"] = ids.tobytes()
            files[record_prefix + "source-reference.json"] = json_bytes(
                {"id": identity, "path": source_name, "sha256": digest(files[source_name])}
            )
            files[record_prefix + "candidate.json"] = json_bytes(converted.candidate)
            files[record_prefix + "changes.json"] = json_bytes([asdict(change) for change in converted.changes])
            for filename, content in exported.files.items():
                files[record_prefix + "export/" + filename] = content
            result = {
                "status": Disposition.VERIFIED,
                "rule_version": "deterministic-v1",
                "rule_ids": converted.rules,
                "verification": {
                    "kind": "deterministic_adapter",
                    "schema_parse_roundtrip": True,
                    "original_task_reexecuted": False,
                    "tool_calls_modified": True,
                    "observation_policy": (
                        "historical source bytes retained; control acknowledgements recorded in changes"
                    ),
                },
                "reason": converted.reason,
            }
            files[record_prefix + "result.json"] = json_bytes(result)
            return {**item, **result}, files
    return {**item, "status": converted.disposition, "reason": converted.reason}, files


def run(
    selection: dict,
    tokenizer: Tokenizer,
    tokenizer_sha: str,
    prefix: str,
    *,
    shard_start: int,
    shard_stop: int,
    batch_size: int,
    row_limit: int | None,
) -> dict:
    validate_campaign(selection, tokenizer, tokenizer_sha)
    catalog_sha = digest(json_bytes(selection))
    config = {
        "catalog_sha256": catalog_sha,
        "tokenizer_sha256": tokenizer_sha,
        "shard_start": shard_start,
        "shard_stop": shard_stop,
        "batch_size": batch_size,
        "row_limit": row_limit,
        "converter": "deterministic-v1",
    }
    write_immutable(prefix_join(prefix, "config.json"), json_bytes(config))
    write_immutable(prefix_join(prefix, "selection.json"), json_bytes(selection))
    path = StoragePath(prefix_join(prefix, "provenance.json"))
    current = provenance(catalog_sha, tokenizer_sha)
    if path.exists():
        current_run = read_json(prefix_join(prefix, "provenance.json"))
        if current_run["files_sha256"] != current["files_sha256"]:
            raise ValueError("Converter changed; use a new output prefix")
    else:
        current_run = current
        write_immutable(prefix_join(prefix, "provenance.json"), json_bytes(current_run))
    if StoragePath(prefix_join(prefix, "completed.json")).exists():
        return read_json(prefix_join(prefix, "completed.json"))
    counts = Counter()
    batches = 0
    artifact_bytes = 0
    processed = 0
    started = time.monotonic()
    for shard_index in range(shard_start, min(shard_stop, len(selection["sources"]))):
        shard = selection["sources"][shard_index]
        if excluded_source(shard["source"]):
            counts[Disposition.EXCLUDED] += shard["expected_rows"]
            continue
        store = TreeStore.open(
            {"input_ids": np.zeros(0, dtype=np.int32)}, shard["input_path"], mode="r", cache_metadata=True
        ).tree["input_ids"]
        if store.num_rows != shard["expected_rows"]:
            raise ValueError("Pinned shard row count changed")
        limit = min(store.num_rows, row_limit) if row_limit is not None else store.num_rows
        for start in range(0, limit, batch_size):
            stop = min(start + batch_size, limit)
            name = f"shard-{shard_index:05d}-rows-{start:08d}-{stop:08d}"
            receipt_url = prefix_join(prefix, f"receipts/{name}.json")
            if StoragePath(receipt_url).exists():
                receipt = read_json(receipt_url)
                if (
                    receipt["catalog_sha256"] != catalog_sha
                    or receipt["provenance"]["files_sha256"] != current_run["files_sha256"]
                ):
                    raise ValueError("Receipt provenance differs")
                counts.update(receipt["post_commit_cursor"]["batch_counts"])
                batches += 1
                artifact_bytes += receipt["archive_bytes"]
                continue
            batch = store.get_batch_sync(range(start, stop))
            dispositions = []
            batch_counts = Counter()
            files = {
                "config.json": json_bytes(config),
                "provenance.json": json_bytes(current_run),
                "lineage.json": json_bytes(
                    {"source": "pinned independent deterministic selection", "catalog_sha256": catalog_sha}
                ),
            }
            for offset, array in enumerate(batch):
                item, artifacts = convert_record(
                    {**shard, "row": start + offset}, np.asarray(array, dtype="<i4"), tokenizer, tokenizer_sha, name
                )
                dispositions.append(item)
                files.update(artifacts)
                counts[item["status"]] += 1
                batch_counts[item["status"]] += 1
                processed += 1
            files[name + "/dispositions.json"] = json_bytes(dispositions)
            cursor = {
                "shard": shard_index,
                "row": stop,
                "batches": batches + 1,
                "counts": dict(counts),
                "batch_counts": dict(batch_counts),
                "artifact_bytes_before_batch": artifact_bytes,
                "updated_at": time.time(),
                "catalog_sha256": catalog_sha,
            }
            receipt = commit_batch(prefix, name, files, catalog_sha, cursor, current_run)
            batches += 1
            artifact_bytes += receipt["archive_bytes"]
            LOGGER.info(
                "batch_committed %s",
                json.dumps(
                    {
                        "batch": name,
                        "counts": dict(counts),
                        "rows_per_second": processed / (time.monotonic() - started),
                        "archive_bytes": receipt["archive_bytes"],
                    }
                ),
            )
    summary = {
        "counts": dict(counts),
        "elapsed_seconds": time.monotonic() - started,
        "rows_processed_this_run": processed,
        "rows": sum(counts.values()),
    }
    write_immutable(prefix_join(prefix, "completed.json"), json_bytes(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-start", type=int, required=True)
    parser.add_argument("--shard-stop", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--row-limit", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    credentials = s3_credentials(StoreType.COREWEAVE)
    region = data_buckets()["marin-us-east-02a"].signing_region
    if credentials is None or region is None:
        raise ValueError("CoreWeave credentials and signing region are required")
    os.environ["AWS_ACCESS_KEY_ID"], os.environ["AWS_SECRET_ACCESS_KEY"] = credentials
    os.environ["AWS_REGION"] = os.environ["AWS_DEFAULT_REGION"] = region
    selection = read_json(args.selection)
    tokenizer_bytes = StoragePath(args.tokenizer).read_bytes()
    tokenizer_sha = digest(tokenizer_bytes)
    if tokenizer_sha != TOKENIZER_SHA256:
        raise CampaignScopeError(f"Unsupported tokenizer; see {SCOPE_GUIDE}")
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode())
    LOGGER.info(
        "complete %s",
        run(
            selection,
            tokenizer,
            tokenizer_sha,
            args.output,
            shard_start=args.shard_start,
            shard_stop=args.shard_stop,
            batch_size=args.batch_size,
            row_limit=args.row_limit,
        ),
    )


if __name__ == "__main__":
    main()
