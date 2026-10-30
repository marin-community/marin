"""Durable per-request Evalchemy resume state in the evaluation's FineStore archive."""

from __future__ import annotations

import json
import re
from collections.abc import Generator
from contextlib import closing
from dataclasses import dataclass, field
from typing import Literal

import pyarrow as pa
from finestore.layout import OnConflict
from finestore.reader import ReadView
from finestore.store import DataStore
from rigging.filesystem.storage_path import prefix_join

from eval.contracts.finestore_output import read_finestore_output
from eval.contracts.resume_values import decode_resume_value, encode_resume_value
from eval.resume.fingerprint import RunFingerprint
from eval.runtime.checkpoints import CheckpointRow

RESUME_TABLE = "evalchemy_operations"
OPERATION_PROTOCOL = 2
_FLUSH_EVERY = 32
_READ_WINDOW = 256
_NAMESPACE = "namespace"
_POSITION = "position"
_IDENTITY = "identity"
_OPERATION = "operation"
_PAYLOAD = "payload"
_RESUME_SCHEMA = pa.schema(
    [pa.field(_NAMESPACE, pa.string()), pa.field(_POSITION, pa.int64()), pa.field(_IDENTITY, pa.string()),
     pa.field(_OPERATION, pa.string()), pa.field(_PAYLOAD, pa.string())]
)
ResumeMode = Literal["auto", "force-fresh"]


class ResumeRefused(RuntimeError):
    """Raised when committed FineStore state cannot safely be resumed."""


def _path_component(value: str) -> str:
    return re.sub(r"[^\w.-]", "_", value) or "task"


@dataclass
class FineStoreResumeManager:
    """Stream committed operations in admission order and buffer only new writes."""

    store: DataStore
    source_prefix: str
    task_name: str
    fingerprint: RunFingerprint
    mode: ResumeMode = "auto"
    world_size: int = 1
    rank: int = 0
    _decision: str | None = field(default=None, init=False, repr=False)
    _pending_count: int = field(default=0, init=False, repr=False)

    @property
    def root(self) -> str:
        return self.store.root

    @property
    def _namespace(self) -> str:
        return prefix_join(_path_component(self.source_prefix), _path_component(self.task_name))

    @property
    def _fingerprint_blob(self) -> str:
        return prefix_join(prefix_join("evalchemy", self._namespace), "resume/fingerprint.json")

    def decide(self) -> str:
        """Return fresh or resume after validating the stored fingerprint; refuse material changes."""
        if self._decision is not None:
            return self._decision
        view = ReadView(self.root)
        stored = view.read_blob(self._fingerprint_blob)
        if stored is None:
            if read_finestore_output(self.root, self.source_prefix) is not None:
                raise ResumeRefused(f"FineStore result exists without a resume fingerprint for {self._namespace}")
            document = {
                **self.fingerprint.to_json(),
                "world_size": self.world_size,
                "operation_protocol": OPERATION_PROTOCOL,
                "provenance": {name: self.fingerprint.inputs[name] for name in ("implementation", "dependencies")
                               if name in self.fingerprint.inputs},
            }
            with self.store.transaction() as transaction:
                transaction.write_object(self._fingerprint_blob, json.dumps(document, sort_keys=True).encode())
            self._decision = "fresh"
            return self._decision

        if self.mode == "force-fresh":
            raise ResumeRefused("FineStore output already exists; use a new output path for a fresh evaluation")
        try:
            document = json.loads(stored)
            if document.get("operation_protocol") != OPERATION_PROTOCOL:
                raise ResumeRefused("checkpoint operation protocol changed; use a new FineStore output path")
            prior = RunFingerprint(inputs=document["canonical_payload"])
            prior_world_size = int(document["world_size"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ResumeRefused(f"invalid FineStore resume fingerprint for {self._namespace}") from exc
        if not self.fingerprint.matches(prior) or prior_world_size != self.world_size:
            raise ResumeRefused(f"FineStore resume fingerprint changed for {self._namespace}")
        self._decision = "resume"
        return self._decision

    def _table(self):
        return self.store.table(
            RESUME_TABLE, primary_key=(_NAMESPACE, _POSITION, _OPERATION),
            schema=_RESUME_SCHEMA, on_conflict=OnConflict.ERROR,
        )

    def records(self) -> Generator[CheckpointRow, None, None]:
        """Read one ordered snapshot; decoding stays bounded by admitted items."""
        if self._decision is None:
            self.decide()
        namespace = prefix_join(self._namespace, f"rank{self.rank}")
        view = ReadView(self.root)
        where = [(_NAMESPACE, "==", namespace)]
        # FineStore merges one batch per committed shard. Project the high-water
        # scan to keys, then restrict payload scans to a fixed position window;
        # otherwise many small shards can retain every completion in the merge.
        with closing(view.iter_rows(RESUME_TABLE, columns=[_POSITION], where=where)) as keys:
            highest = max((row[_POSITION] for row in keys), default=-1)
        for start in range(0, highest + 1, _READ_WINDOW):
            selected = [*where, (_POSITION, "in", list(range(start, start + _READ_WINDOW)))]
            with closing(view.iter_rows(RESUME_TABLE, where=selected)) as rows:
                for row in rows:
                    yield CheckpointRow(row[_POSITION], row[_IDENTITY], json.loads(row[_OPERATION]),
                                        decode_resume_value(json.loads(row[_PAYLOAD])))

    def record(self, row: CheckpointRow) -> None:
        if self._decision is None:
            self.decide()
        table = self._table()
        table.append(
            {
                _NAMESPACE: prefix_join(self._namespace, f"rank{self.rank}"),
                _POSITION: row.position,
                _IDENTITY: row.identity,
                _OPERATION: json.dumps(row.unit, sort_keys=True),
                _PAYLOAD: json.dumps(encode_resume_value(row.payload), separators=(",", ":")),
            }
        )
        self._pending_count += 1
        if self._pending_count >= _FLUSH_EVERY:
            self.store.flush_table(table)
            self._pending_count = 0

    def finalize(self) -> None:
        self.store.flush_table(self._table())
        self._pending_count = 0
