# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the accepted tasks of a Taskforge run into a pinned release.

A Taskforge run root holds one ledger per item (``ledger/<item_id>.jsonl``) and the item's rounds
(``items/<item_id>/rounds/<round>/``). An item is accepted when its last terminal event says
``accepted``; that event's round holds the accepted draft. The round's authoritative evidence is
``evidence-<hash>``, where ``hash`` is the first 12 characters of the input hash of the round's last
``built`` event; earlier builds and rewinds can leave stale evidence beside it.

The release groups accepted tasks by subject and acceptance band, one JSONL file per group, each row
carrying the item's unmodified Taskforge artifacts. ``manifest.json`` records the run root, the
snapshot time, counts and file digests. The run root is only read. Rerunning the command on a
later ledger snapshot and pushing produces a new dataset revision::

    uv run python -m experiments.post_training.task_curation.datasets.taskforge.release \\
        --run-root s3://marin-us-east-02a/marin/taskforge/gen1000/gen1000-20261013 \\
        --run-label testgen20261013 --output /tmp/taskforge-release \\
        --repo open-athena/taskforge-testgen20261013 --push
"""

import hashlib
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import click
from huggingface_hub import HfApi
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

logger = logging.getLogger(__name__)

SUBJECTS = {
    "d01": "Mathematics & Statistics",
    "d03": "Physics",
    "d04": "Chemistry",
    "d06": "Biology",
}
IDEA_PREFIX = "idea--"
ITEM_SEPARATOR = "--"
EVIDENCE_HASH_CHARS = 12
DRAFT_FILES = ("task.json", "lowered.json", "controls.json", "provenance.json")
EVIDENCE_FILES = ("calibration.json", "decision.json")
READ_WORKERS = 64
MANIFEST = "manifest.json"
README = "README.md"
TASKS_DIRECTORY = "tasks"
TASK_FILES = f"{TASKS_DIRECTORY}/*.jsonl"


class Band(StrEnum):
    """Where an accepted task's solve rate fell relative to the run's validation band."""

    CALIBRATED = "calibrated"
    TOO_EASY = "too-easy"


def group_name(subject: str, band: Band) -> str:
    """The release group of a subject's tasks in one band, which names its file and Atlas source."""
    return subject if band == Band.CALIBRATED else f"{subject}-{band}"


def group_file(group: str) -> str:
    return f"{TASKS_DIRECTORY}/{group}.jsonl"


@dataclass(frozen=True)
class AcceptedItem:
    item_id: str
    round: int
    reason: str
    evidence_hash: str

    @property
    def capability(self) -> str:
        return self.item_id.rpartition(ITEM_SEPARATOR)[0]

    @property
    def subject(self) -> str:
        subject = self.capability.partition(".")[0]
        if subject not in SUBJECTS:
            raise ValueError(f"{self.item_id}: unknown subject {subject!r}")
        return subject

    @property
    def band(self) -> Band:
        if self.reason == "calibrated":
            return Band.CALIBRATED
        if "too_easy" in self.reason:
            return Band.TOO_EASY
        raise ValueError(f"{self.item_id}: unrecognized acceptance reason {self.reason!r}")

    @property
    def group(self) -> str:
        return group_name(self.subject, self.band)


def ledger_rows(text: str) -> list[dict[str, Any]]:
    """Parse a ledger; the mirror cuts at a newline, so only a final partial line is tolerated."""
    lines = text.splitlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return rows


def accepted_item(rows: Iterable[Mapping[str, Any]]) -> AcceptedItem | None:
    """The item's accepted round and its authoritative evidence, or None when it is not accepted."""
    events = [row for row in rows if row["kind"] == "event"]
    terminals = [row for row in events if "terminal" in row["attrs"]]
    if not terminals or terminals[-1]["attrs"]["terminal"] != "accepted":
        return None
    terminal = terminals[-1]
    accepted_round = terminal["round"]
    builds = [row for row in events if row["step"] == "built" and row["round"] == accepted_round]
    if not builds:
        raise ValueError(f"{terminal['item_id']}: accepted round {accepted_round} has no built event")
    return AcceptedItem(
        item_id=terminal["item_id"],
        round=accepted_round,
        reason=terminal["attrs"].get("reason", ""),
        evidence_hash=builds[-1]["input_hash"][:EVIDENCE_HASH_CHARS],
    )


def read_json(path: StoragePath) -> Any:
    return json.loads(path.read_text())


def ledger_acceptance(path: StoragePath) -> AcceptedItem | None:
    return accepted_item(ledger_rows(path.read_text()))


def item_record(run_root: StoragePath, item: AcceptedItem, run_label: str) -> dict[str, Any]:
    """One release row: the accepted round's draft and evidence, unmodified."""
    round_root = run_root / f"items/{item.item_id}/rounds/{item.round}"
    draft = {name: read_json(round_root / f"draft/{name}") for name in DRAFT_FILES}
    evidence = {name: read_json(round_root / f"evidence-{item.evidence_hash}/{name}") for name in EVIDENCE_FILES}
    if draft["provenance.json"]["round"] != item.round:
        raise ValueError(f"{item.item_id}: draft provenance round differs from the accepted round {item.round}")
    if not evidence["calibration.json"]["task_digest"].startswith(item.evidence_hash):
        raise ValueError(f"{item.item_id}: calibration digest does not match evidence-{item.evidence_hash}")
    number = item.item_id.rpartition(ITEM_SEPARATOR)[2]
    return {
        "id": f"taskforge:{item.capability}:{run_label}/{number}",
        "item_id": item.item_id,
        "capability": item.capability,
        "subject": item.subject,
        "subject_name": SUBJECTS[item.subject],
        "band": str(item.band),
        "acceptance_reason": item.reason,
        "round": item.round,
        "task": draft["task.json"],
        "lowered": draft["lowered.json"],
        "controls": draft["controls.json"],
        "provenance": draft["provenance.json"],
        "calibration": evidence["calibration.json"],
        "decision": evidence["decision.json"],
    }


def accepted_items(run_root: StoragePath) -> list[AcceptedItem]:
    ledgers = [path for path in (run_root / "ledger").ls() if path.name.endswith(".jsonl")]
    ledgers = [path for path in ledgers if not path.name.startswith(IDEA_PREFIX)]
    with ThreadPoolExecutor(READ_WORKERS) as pool:
        items = [item for item in pool.map(ledger_acceptance, ledgers) if item is not None]
    return sorted(items, key=lambda item: item.item_id)


def write_release(
    records: list[tuple[AcceptedItem, dict[str, Any]]], output: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Write one JSONL per group plus the manifest and dataset card; returns the completed manifest."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item, record in records:
        groups[item.group].append(record)
    files = {}
    (output / TASKS_DIRECTORY).mkdir(parents=True, exist_ok=True)
    for group, rows in sorted(groups.items()):
        relative = group_file(group)
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
        (output / relative).write_bytes(payload)
        files[relative] = {"rows": len(rows), "sha256": hashlib.sha256(payload).hexdigest()}
    manifest = {**manifest, "files": files}
    (output / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / README).write_text(dataset_card(manifest))
    return manifest


def dataset_card(manifest: Mapping[str, Any]) -> str:
    rows = "\n".join(f"| `{name}` | {entry['rows']} |" for name, entry in sorted(manifest["files"].items()))
    return f"""---
license: other
configs:
- config_name: default
  data_files: {TASK_FILES}
---

# Taskforge {manifest["run_label"]}

Accepted tasks from the Taskforge generation run `{manifest["run_root"]}`, frozen from its ledger at
{manifest["snapshot"]}. Each row holds the accepted round's TaskSpec (`task`), its lowered form, the
author's grader controls, build provenance, and the validation evidence (`calibration`, `decision`).

Tasks are grouped by catalog subject and acceptance band. `calibrated` tasks fell inside the solve-rate
band of the run's validation policy; `too-easy` tasks were solved by every validation attempt and were
accepted by this run's policy.

| File | Tasks |
| --- | ---: |
{rows}
"""


def release_manifest(run_root: StoragePath, run_label: str, items: list[AcceptedItem]) -> dict[str, Any]:
    groups = Counter(f"{item.subject}/{item.band}" for item in items)
    return {
        "run_root": str(run_root),
        "run_label": run_label,
        "snapshot": datetime.now(UTC).isoformat(timespec="seconds"),
        "accepted": len(items),
        "groups": dict(sorted(groups.items())),
        "policy": read_json(run_root / "policy.json"),
    }


@click.command()
@click.option("--run-root", required=True, help="Taskforge run root, read only.")
@click.option("--run-label", required=True, help="Label in task IDs, for example testgen20261013.")
@click.option("--output", required=True, type=click.Path(path_type=Path), help="Empty local release directory.")
@click.option("--repo", help="Hugging Face dataset repository for --push.")
@click.option("--push", is_flag=True, help="Upload the release to --repo as a private dataset and print its revision.")
def main(run_root: str, run_label: str, output: Path, repo: str | None, push: bool) -> None:
    logging.basicConfig(level=logging.INFO)
    if push and repo is None:
        raise click.UsageError("--push requires --repo")
    if output.exists() and any(output.iterdir()):
        raise click.UsageError(f"{output} is not empty")
    configure_coreweave_s3()
    root = StoragePath.parse(run_root.rstrip("/"))
    items = accepted_items(root)
    logger.info("%d accepted items in %s", len(items), root)
    with ThreadPoolExecutor(READ_WORKERS) as pool:
        records = list(pool.map(lambda item: (item, item_record(root, item, run_label)), items))
    manifest = write_release(records, output, release_manifest(root, run_label, items))
    logger.info("Wrote %s: %s", output, manifest["groups"])
    if not push:
        return
    assert repo is not None
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=output,
        delete_patterns=[TASK_FILES],
        commit_message=f"Taskforge {run_label}: {manifest['accepted']} accepted tasks at {manifest['snapshot']}",
    )
    click.echo(commit.oid)


if __name__ == "__main__":
    main()
