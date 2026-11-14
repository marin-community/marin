# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Atomic per-observation execution grades, reusable only for identical inputs."""

import json

from rigging.filesystem.buckets import filesystem_for

from experiments.weight_merging.playground.cloud_io import require_coreweave
from experiments.weight_merging.playground.remote.durable import digest


def canonical_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class GradeStore:
    def __init__(self, remote: str, contract: dict):
        if remote.startswith("s3://"):
            require_coreweave(remote)
        self.fs, self.prefix = filesystem_for(remote)
        self.prefix = self.prefix.rstrip("/") + "/items"
        self.fs.makedirs(self.prefix, exist_ok=True)
        self.contract = contract

    def identity(self, row: dict, item: dict) -> dict:
        return {
            "contract": self.contract,
            "candidate": row["candidate"],
            "item": item,
            "item_id": row["item_id"],
            "mode": row["mode"],
            "seed": row["seed"],
            "completion_sha256": digest(row["completion"].encode()),
        }

    def get_record(self, identity: dict) -> dict | None:
        path = self.prefix + "/" + digest(canonical_bytes(identity)) + ".done.json"
        if not self.fs.exists(path):
            return None
        result = json.loads(self.fs.cat(path))
        if result["identity"] != identity or result["grade_sha256"] != digest(canonical_bytes(result["grade"])):
            raise ValueError("Durable grade input or payload changed")
        return result

    def get(self, identity: dict) -> dict | None:
        result = self.get_record(identity)
        return result["grade"] if result is not None else None

    def put(self, identity: dict, grade: dict, *, elapsed_seconds: float | None = None) -> None:
        path = self.prefix + "/" + digest(canonical_bytes(identity)) + ".done.json"
        payload = {"identity": identity, "grade": grade, "grade_sha256": digest(canonical_bytes(grade))}
        if elapsed_seconds is not None:
            payload["elapsed_seconds"] = elapsed_seconds
        self.fs.pipe_file(path, canonical_bytes(payload))
        self.fs.invalidate_cache(path)
        if self.get(identity) != grade:
            raise ValueError("Durable grade readback differs")
