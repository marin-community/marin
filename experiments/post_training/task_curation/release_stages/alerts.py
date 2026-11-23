# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct stage-event emission to the ops feed (DESIGN.md §8).

Every stage emits ``start`` / ``finish`` / ``fail`` / ``stall`` events with evidence
straight to the job-watchdog ``alerts.jsonl`` (and any extra feed the release graph
passes). Failure detection never depends on a watchdog registry edit; registry
entries are proposals merged by the operator.
"""

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

WATCHDOG_FEED = Path.home() / "Documents/experiments/active/job-watchdog/alerts.jsonl"

KIND_START = "STAGE_START"
KIND_FINISH = "STAGE_FINISH"
KIND_FAIL = "STAGE_FAIL"
KIND_STALL = "STAGE_STALL"


@dataclasses.dataclass(frozen=True)
class StageAlert:
    """One stage event in the watchdog alert envelope."""

    key: str
    kind: str
    severity: str
    state: str
    message: str
    evidence: dict

    def to_row(self) -> dict:
        now = datetime.now(tz=UTC)
        return {
            "time_utc": now.isoformat(timespec="seconds"),
            "time_local": now.astimezone().isoformat(timespec="seconds"),
            "key": self.key,
            "kind": self.kind,
            "severity": self.severity,
            "entry": self.key,
            "state": self.state,
            "evidence": self.evidence,
            "message": self.message,
        }


def emit_stage_event(alert: StageAlert, feeds: tuple[Path, ...] = (WATCHDOG_FEED,)) -> None:
    """Append one stage event to each feed (created on first write)."""
    row = json.dumps(alert.to_row(), sort_keys=True) + "\n"
    for feed in feeds:
        feed.parent.mkdir(parents=True, exist_ok=True)
        with feed.open("a", encoding="utf-8") as handle:
            handle.write(row)


def stage_key(release_id: str, stage: str) -> str:
    """The stable alert key for one stage of one release."""
    return f"atlas-release:{release_id}:{stage}"


def stage_started(release_id: str, stage: str, evidence: dict, feeds: tuple[Path, ...] = (WATCHDOG_FEED,)) -> None:
    emit_stage_event(
        StageAlert(
            key=stage_key(release_id, stage),
            kind=KIND_START,
            severity="info",
            state="RUNNING",
            message=f"{stage} started for {release_id}",
            evidence=evidence,
        ),
        feeds,
    )


def stage_finished(release_id: str, stage: str, evidence: dict, feeds: tuple[Path, ...] = (WATCHDOG_FEED,)) -> None:
    emit_stage_event(
        StageAlert(
            key=stage_key(release_id, stage),
            kind=KIND_FINISH,
            severity="info",
            state="SUCCEEDED",
            message=f"{stage} finished for {release_id}",
            evidence=evidence,
        ),
        feeds,
    )


def stage_failed(release_id: str, stage: str, evidence: dict, feeds: tuple[Path, ...] = (WATCHDOG_FEED,)) -> None:
    emit_stage_event(
        StageAlert(
            key=stage_key(release_id, stage),
            kind=KIND_FAIL,
            severity="critical",
            state="FAILED",
            message=f"{stage} failed for {release_id}",
            evidence=evidence,
        ),
        feeds,
    )
