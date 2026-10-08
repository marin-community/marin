# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve fresh curated pairs in a non-repeating native DPO canary batch."""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from levanter.store.cache import CacheLedger, TreeCache
from marin.execution.artifact import Artifact
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.bfcl_rl.collect import MODELS
from experiments.post_training.bfcl_rl.data import PARTITION_MANIFEST_SHA256
from experiments.post_training.bfcl_rl.preference_union import PREFERENCE_COLUMNS
from experiments.post_training.bfcl_rl.recovery_data import RecoveryPreferenceCache, load_audited_partition

SMOKE_PAIRS = 64


@dataclass(frozen=True)
class FreshSmokeDataConfig:
    fresh_cache_path: str
    frozen_inputs_path: str
    complement_path: str
    output_path: str


class FreshSmokeData(Artifact):
    num_pairs: int
    fresh_pairs: int


def token_mask_hash(row: dict) -> str:
    digest = hashlib.sha256()
    for column in PREFERENCE_COLUMNS:
        values = np.asarray(row[column], dtype="<i4")
        digest.update(column.encode() + b"\0")
        digest.update(len(values).to_bytes(8, "little"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def fresh_smoke_rows(fresh_rows: list[dict], frozen_rows: list[dict]) -> list[dict]:
    """Keep each newly curated pair once, filling the remaining canary with distinct frozen pairs."""
    if not 0 < len(fresh_rows) <= SMOKE_PAIRS:
        raise ValueError("The collection smoke must produce between one and 64 accepted fresh pairs")
    rows = []
    seen = set()
    for row in (*fresh_rows, *frozen_rows):
        digest = token_mask_hash(row)
        if digest in seen:
            continue
        seen.add(digest)
        rows.append(row)
        if len(rows) == SMOKE_PAIRS:
            break
    if len(rows) != SMOKE_PAIRS:
        raise ValueError("Not enough distinct pairs for four complete canary updates")
    if len({token_mask_hash(row) for row in fresh_rows}) != len(fresh_rows):
        raise ValueError("Duplicate freshly curated pairs")
    return rows


def run_fresh_smoke_data(config: FreshSmokeDataConfig) -> FreshSmokeData:
    fresh = RecoveryPreferenceCache.load(config.fresh_cache_path)
    raw = StoragePath(fresh.selection_manifest_uri).read_bytes()
    manifest = json.loads(raw)
    cache_path = str(StoragePath(fresh.path) / "train")
    ledger = CacheLedger.load(cache_path)
    metadata = ledger.metadata.preprocessor_metadata
    if not ledger.is_finished or not (StoragePath(cache_path) / ".success").exists():
        raise ValueError("Fresh canary requires a finished curated preference cache")
    if metadata is None or metadata["preference_provenance_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("Fresh preference provenance differs from its cache ledger")
    if metadata["preference_provenance_uri"] != fresh.selection_manifest_uri:
        raise ValueError("Fresh preference provenance locator differs from its cache ledger")
    if len(manifest["preferences"]) != ledger.total_num_rows or fresh.num_preferences != ledger.total_num_rows:
        raise ValueError("Fresh preference counts disagree")
    frozen = StoragePath(config.frozen_inputs_path)
    selection = json.loads((frozen / "selection.json").read_text())
    partition = load_audited_partition(config.complement_path)
    tokenizer = MODELS["student"]
    if manifest["dataset_commit"] != partition.dataset_commit:
        raise ValueError("Fresh preferences use a different dataset revision")
    if manifest["partition_manifest_sha256"] != PARTITION_MANIFEST_SHA256:
        raise ValueError("Fresh preferences use a different audited partition")
    if manifest["student_tokenizer"] != f"{tokenizer.model}@{tokenizer.revision}":
        raise ValueError("Fresh preferences use a different training tokenizer")
    if (fresh.tokenizer_uri, fresh.tokenizer_revision, fresh.max_length) != (
        tokenizer.model,
        tokenizer.revision,
        manifest["max_length"],
    ) or fresh.max_length != 40960:
        raise ValueError("Fresh preference artifact metadata differs from the training contract")
    complement = {task.source_id: task.digest for task in partition.complement}
    seen_tasks = set(selection["seen_task_ids"])
    with (frozen / "full-batches.parquet").open("rb") as source:
        table = pq.read_table(source)
    if table.num_rows != 1712:
        raise ValueError("Frozen final training subset must contain 1,712 pairs")
    cache = TreeCache.load_from_ledger(cache_path, {name: np.zeros(0, np.int32) for name in PREFERENCE_COLUMNS}, ledger)
    fresh_rows = []
    for index, row in enumerate(cache.get_batch_sync(list(range(fresh.num_preferences)))):
        provenance = manifest["preferences"][index]
        chosen, rejected = provenance["chosen"], provenance["rejected"]
        task = chosen["task_source_id"]
        if task in seen_tasks or complement.get(task) != chosen["task_digest"]:
            raise ValueError("Fresh canary pair is outside the unseen-task complement")
        if (
            rejected["task_source_id"] != task
            or rejected["task_digest"] != chosen["task_digest"]
            or rejected["harness"] != chosen["harness"]
            or (chosen["outcome"], rejected["outcome"]) != ("correct", "incorrect")
        ):
            raise ValueError("Fresh canary pair does not distinguish correctness on one task")
        values = {name: row[name].tolist() for name in PREFERENCE_COLUMNS}
        for role in ("chosen", "rejected"):
            ids, masks = values[f"{role}_input_ids"], values[f"{role}_assistant_masks"]
            if (
                not 0 < len(ids) <= 40960
                or min(ids) < 0
                or len(ids) != len(masks)
                or not any(masks)
                or not set(masks) <= {0, 1}
            ):
                raise ValueError("Fresh pair has invalid token/mask lengths or supervision")
        values.update(
            source_row_index=-1 - index,
            task_source_id=task,
            task_digest=chosen["task_digest"],
            token_mask_sha256=token_mask_hash(values),
            provenance_json=json.dumps(provenance, sort_keys=True),
        )
        fresh_rows.append(values)
    rows = fresh_smoke_rows(fresh_rows, table.to_pylist())
    root = StoragePath(config.output_path)
    root.mkdirs()
    with (root / "smoke.parquet").open("wb") as destination:
        pq.write_table(
            pa.Table.from_pylist(rows, schema=table.schema), destination, compression="zstd", row_group_size=16
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "fresh_cache": fresh.path,
                "fresh_selection_sha256": hashlib.sha256(raw).hexdigest(),
                "frozen_inputs": config.frozen_inputs_path,
                "pairs": len(rows),
                "fresh_pairs": len(fresh_rows),
                "token_mask_hashes": [token_mask_hash(row) for row in rows],
                "unseen_tasks_only": True,
                "order": "Fresh curated pairs, then nonduplicate frozen rows in their original order",
            },
            indent=2,
        )
        + "\n"
    )
    return FreshSmokeData(path=config.output_path, num_pairs=len(rows), fresh_pairs=len(fresh_rows))
