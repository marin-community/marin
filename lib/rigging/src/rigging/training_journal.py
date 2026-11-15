# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Producer-owned metric history and native phase evidence, independent of providers."""

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from rigging.event_log import EventLog
from rigging.filesystem.storage_path import StoragePath


class TrainingJournal:
    """Fsync each metric call locally; publish immutable files and a completion receipt last."""

    def __init__(self, root: str, run: str, stage: str, producer: str):
        self.attempt = uuid4().hex
        self.destination = StoragePath(root) / producer / self.attempt
        self.directory = Path(tempfile.mkdtemp(prefix="training-journal-", dir=os.environ.get("IRIS_OUTPUT_DIR")))
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
        with (self.destination / "started.json").open("xb") as output:
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
            encoded = json.dumps(row, sort_keys=True) + "\n"
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
                with target.open("xb") as output:
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
            with (self.destination / "receipt.json").open("xb") as output:
                output.write(content)
            self.closed = True


_current_journal: TrainingJournal | None = None


def get_training_journal() -> TrainingJournal | None:
    return _current_journal


def set_training_journal(journal: TrainingJournal | None) -> None:
    global _current_journal
    _current_journal = journal


@contextmanager
def phase(kind: str, *, step: int | None = None) -> Iterator[None]:
    """Record a native phase only when this producer has explicit durable telemetry enabled."""
    journal = get_training_journal()
    if journal is None:
        yield
        return
    with journal.events.span(kind, step=step):
        yield


def verify_training_journals(root: str, producers: set[str], required_phases: set[str]) -> list[dict]:
    """Verify hashes, every logged row, complete phase pairs and the exact producer set."""
    location = StoragePath(root)
    receipts = []
    observed = set()
    starts = (location / "*/*/started.json").glob()
    for started in starts:
        if not (started.parent / "receipt.json").exists():
            raise ValueError(f"Telemetry attempt has no completion receipt: {started.parent}")
    for path in sorted((location / "*/*/receipt.json").glob(), key=str):
        receipt = json.loads(path.read_bytes())
        if receipt["status"] not in {"completed", "failed"}:
            raise ValueError("Unknown telemetry completion status")
        started = json.loads((path.parent / "started.json").read_bytes())
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
            content = (path.parent / name).read_bytes()
            if len(content) != identity["bytes"] or hashlib.sha256(content).hexdigest() != identity["sha256"]:
                raise ValueError(f"Telemetry file changed: {path.parent / name}")
        if set(receipt["files"]) != {"history.jsonl", "phases.jsonl"}:
            raise ValueError("Telemetry receipt must include history and phases")
        rows = [json.loads(line) for line in (path.parent / "history.jsonl").read_text().splitlines()]
        if [row["sequence"] for row in rows] != list(range(receipt["rows"])):
            raise ValueError("Metric history has missing or reordered rows")
        if (not rows and receipt["status"] == "completed") or sorted(
            {row["step"] for row in rows if row["step"] is not None}
        ) != receipt["steps"]:
            raise ValueError("Metric history step coverage differs from the producer receipt")
        spans: dict[str, list[dict]] = {}
        for line in (path.parent / "phases.jsonl").read_text().splitlines():
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
