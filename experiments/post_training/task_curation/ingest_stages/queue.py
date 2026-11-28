# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture a unique queue row and claim it with a row-level compare-and-swap."""

import fcntl
import hashlib
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field

from experiments.post_training.task_curation.ingest_stages.ingest_config import IngestConfig
from experiments.post_training.task_curation.ingest_stages.receipts import SHA256
from experiments.post_training.task_curation.release_stages.main_sources import CANDIDATES_LOCK_NAME
from infra.marina.apps.rl_data_catalog.server.source_identity import SourceId

URL = re.compile(r"https?://[^\s|<>`]+")
CLAIM_MARKER = re.compile(r"<!-- atlas-ingest:claim:([0-9a-f]+) -->")


class QueueSelection(BaseModel):
    """The reviewed row and full-file digest captured before a queue mutation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId
    policy_sha256: str = Field(pattern=SHA256)
    queue_sha256: str = Field(pattern=SHA256)
    row: str
    requester_issue: str | None


class QueueClaim(BaseModel):
    """Portable evidence that the host claimed the selected row under the queue lock."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    selection: QueueSelection
    claimed_row_sha256: str = Field(pattern=SHA256)


def row_cells(row: str) -> list[str]:
    return [value.strip() for value in row.strip().strip("|").split("|")]


def row_urls(row: str) -> list[str]:
    return [value.rstrip(".,;)") for value in URL.findall(row)]


def upstream_links(row: str, policy: IngestConfig) -> list[list[str]]:
    links = []
    for value in row_urls(row):
        url = urlsplit(value)
        parts = url.path.strip("/").split("/")
        if url.hostname == "huggingface.co" and parts[:3] == ["datasets", *policy.upstream.repository.split("/")]:
            links.append(parts[3:])
    return links


def select_queue(payload: bytes, policy: IngestConfig) -> QueueSelection:
    """Refuse missing, ambiguous or differently pinned rows before any side effect."""
    rows = [row for row in payload.decode("utf-8").splitlines() if row.lstrip().startswith("|")]
    exact = [row for row in rows if row_cells(row)[0] == policy.source_id]
    matching = exact or [row for row in rows if upstream_links(row, policy)]
    if not exact and len(matching) > 1:
        # Legacy repackager rows have human labels; require every folder token,
        # including version digits, rather than selecting the first repository hit.
        tokens = set(re.findall(r"[a-z]+|[0-9]+", policy.upstream.path.lower()))
        matching = [row for row in matching if tokens <= set(re.findall(r"[a-z]+|[0-9]+", row_cells(row)[0].lower()))]
    if len(matching) != 1:
        raise ValueError("the ingest policy must select exactly one queue row")
    row = matching[0]
    links = upstream_links(row, policy)
    if not links:
        raise ValueError("the selected queue row does not name the configured upstream")
    for parts in links:
        if parts and (len(parts) < 2 or parts[0] != "tree" or parts[1] != policy.upstream.revision):
            raise ValueError("the selected queue row names a different upstream pin")
        if len(parts) > 2 and "/".join(parts[2:]) != policy.upstream.path:
            raise ValueError("the selected queue row names a different upstream folder")
    issues = set()
    for value in row_urls(row):
        url = urlsplit(value)
        if url.hostname == "github.com" and re.fullmatch(r"/[^/]+/[^/]+/issues/[0-9]+", url.path):
            if url.scheme != "https" or url.username is not None or url.port is not None:
                raise ValueError("requester issue must be a canonical HTTPS GitHub issue")
            issues.add(urlunsplit((url.scheme, url.netloc, url.path, "", "")))
    if len(issues) > 1:
        raise ValueError("the selected queue row names several requester issues")
    return QueueSelection(
        source_id=policy.source_id,
        policy_sha256=policy.digest(),
        queue_sha256=hashlib.sha256(payload).hexdigest(),
        row=row,
        requester_issue=next(iter(issues), None),
    )


def claimed_row(selection: QueueSelection) -> str:
    cells = row_cells(selection.row)
    if len(cells) < 3:
        raise ValueError("queue row has no status column")
    markers = CLAIM_MARKER.findall(selection.row)
    if markers and markers != [selection.policy_sha256]:
        raise ValueError("queue row has a conflicting ingest claim")
    if markers:
        return selection.row
    cells[2] += f"; atlas-ingest claimed <!-- atlas-ingest:claim:{selection.policy_sha256} -->"
    return "| " + " | ".join(cells) + " |"


def claim_queue(path: Path, policy: IngestConfig, selection: QueueSelection) -> QueueClaim:
    """Claim once while preserving unrelated concurrent edits and refusing row changes."""
    if selection.source_id != policy.source_id or selection.policy_sha256 != policy.digest():
        raise ValueError("queue selection belongs to another policy")
    # macOS /usr/bin/lockf uses BSD flock locks, as does main-source sync.
    # Refuse a busy queue immediately; a command retry is finite and explicit.
    with (path.parent / CANDIDATES_LOCK_NAME).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        payload = path.read_bytes()
        current = select_queue(payload, policy)
        replacement = claimed_row(selection)
        if current.row not in (selection.row, replacement):
            raise ValueError("selected queue row changed; recapture and review before claiming")
        if current.row != replacement:
            lines = payload.decode("utf-8").splitlines(keepends=True)
            index = next(i for i, line in enumerate(lines) if line.rstrip("\r\n") == current.row)
            newline = "\r\n" if lines[index].endswith("\r\n") else "\n" if lines[index].endswith("\n") else ""
            lines[index] = replacement + newline
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                try:
                    os.fchmod(handle.fileno(), path.stat().st_mode & 0o777)
                    handle.write("".join(lines))
                    handle.flush()
                    os.fsync(handle.fileno())
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
        if select_queue(path.read_bytes(), policy).row != replacement:
            raise ValueError("claimed queue row failed readback")
    return QueueClaim(selection=selection, claimed_row_sha256=hashlib.sha256(replacement.encode()).hexdigest())


def verify_claim(claim: QueueClaim, policy: IngestConfig) -> None:
    selection = claim.selection
    if selection.source_id != policy.source_id or selection.policy_sha256 != policy.digest():
        raise ValueError("queue claim belongs to another policy")
    # Reparse the captured row so even a forged selection cannot inject an issue
    # destination that did not appear on the selected source's queue row.
    derived = select_queue(selection.row.encode(), policy)
    if derived.requester_issue != selection.requester_issue:
        raise ValueError("queue claim requester differs from its captured row")
    if hashlib.sha256(claimed_row(selection).encode()).hexdigest() != claim.claimed_row_sha256:
        raise ValueError("claimed queue row digest mismatch")
