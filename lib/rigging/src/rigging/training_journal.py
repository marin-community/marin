# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Producer-owned metric history and native phase evidence, independent of providers."""

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from rigging.event_log import EventLog
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join


class TrainingJournal:
    """Fsync each metric call locally; publish immutable files and a completion receipt last."""

    def __init__(
        self,
        root: str,
        run: str,
        stage: str,
        producer: str,
        *,
        json_default: Callable[[object], object] | None = None,
        local_directory: str | None = None,
    ):
        self.json_default = json_default
        self.attempt = uuid4().hex
        self.destination = StoragePath(root) / producer / self.attempt
        self.filesystem, self.destination_key = filesystem_for(str(self.destination))
        self.directory = Path(tempfile.mkdtemp(prefix="training-journal-", dir=local_directory))
        self.history = self.directory / "history.jsonl"
        self.history.touch()
        self.events = EventLog(self.directory / "phases.jsonl", run, stage, self.attempt)
        self.events.path.touch()
        self.producer = producer
        self.run = run
        self.stage = stage
        self.rows = 0
        self.steps: set[int] = set()
        self.lock = threading.RLock()
        self.closed = False
        self.destination.mkdirs(exist_ok=True)
        started = {
            "schema": 1,
            "run": run,
            "stage": stage,
            "producer": producer,
            "attempt": self.attempt,
            "started_utc": datetime.now(UTC).isoformat(),
        }
        content = json.dumps(started, sort_keys=True).encode()
        (self.directory / "started.json").write_bytes(content)
        with self.filesystem.open(prefix_join(self.destination_key, "started.json"), "xb") as output:
            output.write(content)

    def log(self, metrics: Mapping, *, step: int | None, commit: bool | None) -> None:
        """Preserve every call, including repeated step IDs and uncommitted metrics."""
        with self.lock:
            if self.closed:
                raise RuntimeError("Metric history is already sealed")
            row = {
                "sequence": self.rows,
                "step": step,
                "commit": commit,
                "utc": datetime.now(UTC).isoformat(),
                "metrics": dict(metrics),
            }
            encoded = json.dumps(row, sort_keys=True, default=self.json_default) + "\n"
            with self.history.open("a") as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            self.rows += 1
            if step is not None:
                self.steps.add(step)

    def finish(self, *, status: str) -> None:
        """Publish before external tracker shutdown; unsuccessful attempts remain explicit."""
        with self.lock:
            if self.closed:
                return
            files = {}
            for name in ("history.jsonl", "phases.jsonl"):
                content = (self.directory / name).read_bytes()
                target = self.destination / name
                target.parent.mkdirs(exist_ok=True)
                # Each writer owns a UUID namespace. Exclusive creation rejects collisions.
                with self.filesystem.open(prefix_join(self.destination_key, name), "xb") as output:
                    output.write(content)
                files[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            receipt = {
                "schema": 1,
                "run": self.run,
                "stage": self.stage,
                "producer": self.producer,
                "attempt": self.attempt,
                "status": status,
                "rows": self.rows,
                "steps": sorted(self.steps),
                "files": files,
            }
            content = json.dumps(receipt, sort_keys=True).encode()
            (self.directory / "receipt.json").write_bytes(content)
            with self.filesystem.open(prefix_join(self.destination_key, "receipt.json"), "xb") as output:
                output.write(content)
            self.closed = True


@contextmanager
def phase(journal: TrainingJournal | None, kind: str, *, step: int | None = None) -> Iterator[None]:
    """Record a native phase only when this producer has explicit durable telemetry enabled."""
    if journal is None:
        yield
        return
    with journal.events.span(kind, step=step):
        yield


def verify_training_journals(root: str, producers: set[str], required_phases: set[str]) -> list[dict]:
    """Verify hashes, every logged row, complete phase pairs and the exact producer set."""
    fs, prefix = filesystem_for(root)
    receipts = []
    observed = set()
    starts = fs.glob(prefix_join(prefix, "*/*/started.json"))
    for started in starts:
        directory = str(StoragePath(started).parent)
        if not fs.isfile(prefix_join(directory, "receipt.json")):
            raise ValueError(f"Telemetry attempt has no completion receipt: {fs.unstrip_protocol(directory)}")
    for key in sorted(fs.glob(prefix_join(prefix, "*/*/receipt.json"))):
        path = StoragePath(fs.unstrip_protocol(key))
        directory = str(StoragePath(key).parent)
        receipt = json.loads(fs.cat_file(key))
        if receipt["status"] not in {"completed", "failed"}:
            raise ValueError("Unknown telemetry completion status")
        started = json.loads(fs.cat_file(prefix_join(directory, "started.json")))
        if any(started[key] != receipt[key] for key in ("run", "stage", "producer", "attempt")):
            raise ValueError("Telemetry completion differs from the started attempt")
        producer = receipt["producer"]
        if producer not in producers:
            raise ValueError(f"Unexpected telemetry producer: {producer}")
        if receipt["status"] == "completed":
            if producer in observed:
                raise ValueError(f"Multiple completed telemetry attempts for {producer}")
            observed.add(producer)
        if path.parent.parent.name != producer or path.parent.name != receipt["attempt"]:
            raise ValueError("Telemetry identity does not match its storage namespace")
        for name, identity in receipt["files"].items():
            if name not in {"history.jsonl", "phases.jsonl"}:
                raise ValueError(f"Unexpected telemetry file: {name}")
            content = fs.cat_file(prefix_join(directory, name))
            if len(content) != identity["bytes"] or hashlib.sha256(content).hexdigest() != identity["sha256"]:
                raise ValueError(f"Telemetry file changed: {path.parent / name}")
        if set(receipt["files"]) != {"history.jsonl", "phases.jsonl"}:
            raise ValueError("Telemetry receipt must include history and phases")
        rows = [json.loads(line) for line in fs.cat_file(prefix_join(directory, "history.jsonl")).decode().splitlines()]
        if [row["sequence"] for row in rows] != list(range(receipt["rows"])):
            raise ValueError("Metric history has missing or reordered rows")
        if (not rows and receipt["status"] == "completed") or sorted(
            {row["step"] for row in rows if row["step"] is not None}
        ) != receipt["steps"]:
            raise ValueError("Metric history step coverage differs from the producer receipt")
        spans: dict[str, list[dict]] = {}
        for line in fs.cat_file(prefix_join(directory, "phases.jsonl")).decode().splitlines():
            event = json.loads(line)
            if (
                event["run"] != receipt["run"]
                or event["stage"] != receipt["stage"]
                or event["attempt"] != receipt["attempt"]
            ):
                raise ValueError("Phase journal identity differs from its producer receipt")
            spans.setdefault(event["span"], []).append(event)
        completed = set()
        for pair in spans.values():
            if len(pair) != 2 or [event["event"] for event in pair] != ["start", "end"]:
                if receipt["status"] == "failed":
                    continue
                raise ValueError("Phase journal contains an incomplete span")
            start, end = pair
            if start["clock"] != end["clock"] or start["kind"] != end["kind"] or end["monotonic"] < start["monotonic"]:
                raise ValueError("Phase journal contains inconsistent boundaries")
            if end["status"] != "completed" and receipt["status"] == "completed":
                raise ValueError("Completed producer contains a failed phase")
            completed.add(start["kind"])
        expected_phases = required_phases
        if receipt["status"] == "completed" and not expected_phases <= completed:
            raise ValueError(f"Missing native phases for {producer}: {expected_phases - completed}")
        receipts.append({**receipt, "source": str(path)})
    if observed != producers:
        raise ValueError(
            f"Telemetry producer coverage mismatch: expected {sorted(producers)}, observed {sorted(observed)}"
        )
    return receipts
