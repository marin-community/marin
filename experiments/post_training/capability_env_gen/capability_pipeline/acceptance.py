"""Content-bound, resumable quality acceptance.

A quality-accepted item is terminal: a resumed controller (a relaunch under a
new run name restores the prior durable snapshot into a new work root) must not
re-run construction gates or semantic review for it.  The acceptance is keyed
on stable, content-derived identity only:

* the item key and proposal hash,
* ``contract/accepted.json`` equal to the current admitted item, and
* the exact bytes of the reviewed task artefacts (``contract/accepted.json``,
  ``workspace/task/**`` and ``harbor/**``), compared file by file against the
  accepted review's own input manifest.

Run names, job ids, absolute work roots, timestamps and snapshot ids are never
part of the identity; absolute paths in a restored status are rebased onto the
current root.  Any mismatch or unreadable evidence fails closed: the caller
falls back to the normal construction path, which re-reviews the item.

Every acceptance is also written to a write-once ledger,
``<root>/acceptances/<item>.json``, so a later overwrite of ``status.json``
cannot silently demote an accepted item whose content has not changed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .conveyor import write_status
from .inference import atomic_json, digest

SCHEMA = "capability-acceptance-v1"
TASK_PREFIXES = ("workspace/task/", "harbor/")
TASK_FILES = ("contract/accepted.json",)


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _is_task_artifact(relative: str) -> bool:
    return relative in TASK_FILES or relative.startswith(TASK_PREFIXES)


def task_artifacts(item_root: Path) -> dict[str, str]:
    """Current sha256 of every reviewed task artefact under ``item_root``."""
    files: dict[str, str] = {}
    for name in TASK_FILES:
        path = item_root / name
        if path.is_file():
            files[name] = _sha256(path)
    for prefix in TASK_PREFIXES:
        base = item_root / prefix.rstrip("/")
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file():
                files[path.relative_to(item_root).as_posix()] = _sha256(path)
    return files


def reviewed_task_artifacts(manifest: dict[str, Any]) -> dict[str, str]:
    """The task artefacts a quality review snapshot covered, from its manifest."""
    hashes = manifest.get("files")
    item_files = manifest.get("item_files")
    if not isinstance(hashes, dict) or not isinstance(item_files, list):
        raise ValueError("review manifest lacks file hashes")
    selected = {}
    for relative in item_files:
        if isinstance(relative, str) and _is_task_artifact(relative):
            value = hashes.get(relative)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"review manifest lacks a hash for {relative}")
            selected[relative] = value
    return selected


def ledger_path(root: Path, item_root: Path) -> Path:
    return root / "acceptances" / f"{item_root.name}.json"


def _review_dir_from_claim(
    claim: dict[str, Any], item_root: Path
) -> tuple[str, str | None]:
    """Return (review dir relative to the run root, claimed prior run root)."""
    record = claim.get("acceptance")
    if isinstance(record, dict) and isinstance(record.get("review_dir"), str):
        relative = record["review_dir"]
    else:
        review = claim.get("quality_review")
        artifact = review.get("artifact") if isinstance(review, dict) else None
        if not isinstance(artifact, str):
            raise ValueError("accepted status lacks a quality review artifact")
        parts = Path(artifact).parts
        if "quality" not in parts:
            raise ValueError("quality review artifact is outside quality/")
        index = len(parts) - 1 - parts[::-1].index("quality")
        relative = Path(*parts[index:]).parent.as_posix()
    rel = Path(relative)
    if (
        rel.is_absolute()
        or ".." in rel.parts
        or len(rel.parts) != 3
        or rel.parts[0] != "quality"
        or rel.parts[1] != item_root.name
    ):
        raise ValueError(f"quality review directory is not this item's: {relative}")
    prior_root = None
    prior_item_root = claim.get("item_root")
    if isinstance(prior_item_root, str):
        prior = Path(prior_item_root)
        if prior.name != item_root.name or prior.parent.name != "items":
            raise ValueError("accepted status belongs to another item")
        prior_root = str(prior.parents[1])
    return relative, prior_root


def verify_acceptance(
    item: dict[str, Any], key: str, root: Path, item_root: Path, claim: dict[str, Any]
) -> dict[str, Any]:
    """Re-derive a content-bound acceptance record, or raise ``ValueError``."""
    if not isinstance(claim, dict) or claim.get("state") != "quality_accepted":
        raise ValueError("claim is not a quality acceptance")
    if claim.get("key") != key or claim.get("proposal_hash") != item["proposal_hash"]:
        raise ValueError("accepted status identity differs from the admitted item")
    accepted = item_root / "contract" / "accepted.json"
    try:
        if json.loads(accepted.read_text()) != item:
            raise ValueError("contract/accepted.json differs from the admitted item")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"contract/accepted.json is unreadable: {error}") from error
    review_dir, _ = _review_dir_from_claim(claim, item_root)
    review_root = root / review_dir
    try:
        manifest = json.loads((review_root / "input-manifest.json").read_text())
        result_path = review_root / "result.json"
        verdict = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"accepted quality review is unreadable: {error}") from error
    if verdict.get("state") != "accept":
        raise ValueError("referenced quality review did not accept")
    snapshot = manifest.get("snapshot_hash")
    if not isinstance(snapshot, str) or verdict.get("snapshot_hash") != snapshot:
        raise ValueError("quality verdict is bound to another snapshot")
    result_sha = _sha256(result_path)
    review = claim.get("quality_review")
    if isinstance(review, dict):
        if review.get("state") not in (None, "accept"):
            raise ValueError("status records a non-accepting review")
        if review.get("artifact_sha256") not in (None, result_sha):
            raise ValueError("quality verdict bytes changed since acceptance")
        if review.get("snapshot_hash") not in (None, snapshot):
            raise ValueError("status is bound to another review snapshot")
    reviewed = reviewed_task_artifacts(manifest)
    if not any(name.startswith("workspace/task/") for name in reviewed):
        raise ValueError("review manifest covers no task bundle")
    current = task_artifacts(item_root)
    if current != reviewed:
        changed = sorted(
            name
            for name in set(current) | set(reviewed)
            if current.get(name) != reviewed.get(name)
        )
        raise ValueError(
            f"task artefacts changed since review ({len(changed)} files, e.g. "
            + ", ".join(changed[:3])
            + ")"
        )
    record = {
        "schema_version": SCHEMA,
        "key": key,
        "proposal_hash": item["proposal_hash"],
        "task_artifacts_sha256": digest(reviewed),
        "task_artifact_count": len(reviewed),
        "review_dir": review_dir,
        "review_snapshot_hash": snapshot,
        "review_result_sha256": result_sha,
    }
    prior = claim.get("acceptance")
    if isinstance(prior, dict) and prior.get("schema_version") == SCHEMA:
        for field in ("task_artifacts_sha256", "review_snapshot_hash", "review_result_sha256"):
            if prior.get(field) != record[field]:
                raise ValueError(f"acceptance record {field} differs")
    return record


def _rebase(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return new + value[len(old):] if value == old or value.startswith(old + "/") else value
    if isinstance(value, list):
        return [_rebase(entry, old, new) for entry in value]
    if isinstance(value, dict):
        return {name: _rebase(entry, old, new) for name, entry in value.items()}
    return value


def _export(root: Path, item_root: Path) -> Path:
    export = root / "validated" / item_root.name
    if export.exists():
        shutil.rmtree(export)
    shutil.copytree(item_root / "harbor", export)
    return export


def record_acceptance(
    item: dict[str, Any], key: str, root: Path, item_root: Path, result: dict[str, Any]
) -> None:
    """Attach the content-bound record to a fresh acceptance and write the ledger.

    Acceptance itself is unchanged; if the reviewed bytes cannot be re-derived
    the item is simply not made resumable (a later resume re-reviews it).
    """
    try:
        record = verify_acceptance(item, key, root, item_root, result)
    except ValueError as error:
        result["acceptance_issue"] = str(error)
        return
    result["acceptance"] = record
    atomic_json(ledger_path(root, item_root), {"record": record, "status": result})


_DEMOTION_FIELDS = ("issues", "terminal_disposition", "repair_issue", "acceptance_issue")
# Conveyor bookkeeping from the demoted (non-accepted) status: stale on an acceptance.
_STALE_FIELDS = (
    "wait", "wait_exhausted", "wait_hold", "unclassified_state", "failure_stage",
    "controller_exception", "conveyor_error",
)


def _attempt_number(path: Path) -> int:
    try:
        return int(path.name.removeprefix("attempt-"))
    except ValueError:
        return -1


GATE_RECEIPT = "controller-gate.json"


GATE_PROOF_RECEIPT = "receipt"
GATE_PROOF_LEGACY = "legacy_reconstruction"


def post_review_gate_proof(attempt: Path) -> tuple[str | None, dict[str, Any] | None]:
    """Which evidence proves the controller's post-review gate passed.

    An accepting semantic review is necessary but not sufficient: the old
    controller still required ``repeated diagnostics state == "ready"`` before
    writing ``quality_accepted`` ("semantic review accepted despite failed
    repeated runtime gates" otherwise).

    Returns ``(proof, reconstruction)``:

    * ``("receipt", None)`` when a ``controller-gate.json`` beside the review
      records ``"ready"``.  A receipt that exists is authoritative: a non-ready
      or unreadable receipt is never overridden by reconstruction.
    * ``("legacy_reconstruction", record)`` when there is no receipt (every
      review run before the receipt existed) and
      ``legacy_acceptance_gate.reconstruct`` recomputes ``"ready"`` from the
      review's own frozen input snapshot.
    * ``(None, record_or_None)`` otherwise: not an acceptance claim.
    """
    receipt = attempt / GATE_RECEIPT
    if receipt.exists() or receipt.is_symlink():
        try:
            gate = json.loads(receipt.read_text())
        except (OSError, json.JSONDecodeError):
            return None, None
        ready = isinstance(gate, dict) and gate.get("repeated_diagnostics_state") == "ready"
        return (GATE_PROOF_RECEIPT if ready else None), None
    from .legacy_acceptance_gate import READY, reconstruct

    record = reconstruct(attempt)
    return (GATE_PROOF_LEGACY if record.get("verdict") == READY else None), record


def post_review_gate_passed(attempt: Path) -> bool:
    """Whether a receipt or a legacy reconstruction proves the gate passed."""
    return post_review_gate_proof(attempt)[0] is not None


def _gate_summary(record: dict[str, Any]) -> dict[str, Any]:
    """The reconstruction facts worth keeping in a status (small, content-bound)."""
    return {
        "schema_version": record.get("schema_version"),
        "verdict": record.get("verdict"),
        "reason": record.get("reason"),
        "diagnostics": [
            {
                "attempt": entry.get("attempt"),
                "repeated": (entry.get("repeated") or {}).get("state"),
                "grading": (entry.get("grading") or {}).get("state")
                or (entry.get("grading") or {}).get("kind"),
                "reset": (entry.get("reset") or {}).get("state"),
            }
            for entry in record.get("candidates") or []
        ],
        "eliminated_environments": len(record.get("eliminated") or []),
    }


# Gate proofs are pure functions of a review's retained files (the legacy
# reconstruction reads its frozen snapshot), and every resume of a waiting item
# would otherwise recompute them: memoised per review attempt per process,
# keyed on the files that decide them.
_GATE_PROOF_CACHE: dict[tuple, tuple[str | None, dict[str, Any] | None]] = {}
_GATE_PROOF_LOCK = threading.Lock()


def _stat_key(path: Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size, stat.st_ino)


def _cached_gate_proof(attempt: Path) -> tuple[str | None, dict[str, Any] | None]:
    key = (
        str(attempt.resolve()),
        _stat_key(attempt / "result.json"),
        _stat_key(attempt / GATE_RECEIPT),
        _stat_key(attempt / "input-manifest.json"),
    )
    with _GATE_PROOF_LOCK:
        if key in _GATE_PROOF_CACHE:
            return _GATE_PROOF_CACHE[key]
    value = post_review_gate_proof(attempt)
    with _GATE_PROOF_LOCK:
        _GATE_PROOF_CACHE[key] = value
    return value


def _review_record_claims(
    item: dict[str, Any], key: str, root: Path, item_root: Path,
    prior_status: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Every acceptance claim from this item's own accepting quality reviews."""
    return list(_iter_review_record_claims(item, key, root, item_root, prior_status))


def _iter_review_record_claims(
    item: dict[str, Any], key: str, root: Path, item_root: Path,
    prior_status: dict[str, Any] | None,
) -> Iterator[dict[str, Any]]:
    """Acceptance claims rebuilt from this item's own accepting quality reviews.

    Jobs launched before the ledger existed re-ran every gate for an accepted
    item on resume and could overwrite ``status.json`` with a later failure
    (2026-09-29: shard-071 complexity-9 was accepted twice, then demoted to
    failed by a re-run control).  The quality review directory survives that
    overwrite, so an accepting verdict whose reviewed task bytes still equal the
    current ones is claim enough.  The newest accepting attempt is tried first;
    ``verify_acceptance`` still decides.  Lazy: an older attempt's gate proof
    is computed only when the newer claims were rejected.  A failure while
    judging one attempt skips that attempt; nothing escapes.
    """
    base = dict(prior_status) if isinstance(prior_status, dict) else {}
    superseded = {
        "state": base.get("state"),
        "issues": list(base.get("issues") or [])[:5],
    }
    for field in (*_DEMOTION_FIELDS, *_STALE_FIELDS, "quality_review", "acceptance",
                  "acceptance_resumed", "gate_proof", "gate_reconstruction"):
        base.pop(field, None)
    try:
        attempts = sorted((root / "quality" / item_root.name).glob("attempt-*"),
                          key=_attempt_number, reverse=True)
    except OSError as error:
        print(f"quality reviews of {key} are unreadable: {error}", file=sys.stderr, flush=True)
        return
    for attempt in attempts:
        try:
            verdict = json.loads((attempt / "result.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(verdict, dict) or verdict.get("state") != "accept":
            continue
        try:
            proof, reconstruction = _cached_gate_proof(attempt)
        except Exception as error:  # noqa: BLE001 -- one unjudgeable review is not a claim
            print(
                f"accepting review quality/{item_root.name}/{attempt.name} could not be "
                f"judged for {key}: {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
            continue
        if proof is None:
            why = (reconstruction or {}).get("reason") or "gate receipt is not ready"
            print(
                f"accepting review quality/{item_root.name}/{attempt.name} is not an "
                f"acceptance claim for {key}: {why}",
                file=sys.stderr,
                flush=True,
            )
            continue
        claim = {
            **base,
            "key": key,
            "proposal_hash": item["proposal_hash"],
            "state": "quality_accepted",
            "item_root": str(item_root),
            "acceptance": {"review_dir": f"quality/{item_root.name}/{attempt.name}"},
            "superseded_status": superseded,
            "gate_proof": proof,
        }
        if reconstruction is not None:
            claim["gate_reconstruction"] = _gate_summary(reconstruction)
        yield claim


def resume_accepted(
    item: dict[str, Any],
    key: str,
    root: Path,
    item_root: Path,
    prior_status: dict[str, Any] | None,
    seeds: list[dict[str, Any]] = (),
) -> dict[str, Any] | None:
    """Return the preserved accepted result for an unchanged item, else ``None``.

    Claims are tried in order: the restored ``status.json``, the durable ledger,
    explicit recovery seeds (prior accepted statuses supplied by an operator),
    then the item's own accepting quality reviews (evaluated only when every
    earlier claim failed).  Every claim is re-verified against the local
    evidence.  Nothing raises: a claim that cannot be evaluated or restored is
    skipped, and with no claim left the caller takes the construction path.
    """
    ledger = ledger_path(root, item_root)

    def claims() -> Iterator[tuple[str, dict[str, Any]]]:
        if isinstance(prior_status, dict) and prior_status.get("state") == "quality_accepted":
            yield "status", prior_status
        try:
            if ledger.is_file():
                yield "ledger", json.loads(ledger.read_text())["status"]
        except Exception:  # noqa: BLE001 -- an unreadable ledger is no claim
            print(f"acceptance ledger unreadable for {key}", file=sys.stderr, flush=True)
        for seed in seeds:
            yield "seed", seed
        for claim in _iter_review_record_claims(item, key, root, item_root, prior_status):
            yield "quality_review", claim

    for source, claim in claims():
        try:
            record = verify_acceptance(item, key, root, item_root, claim)
            _, prior_root = _review_dir_from_claim(claim, item_root)
        except Exception as error:  # noqa: BLE001 -- any unverifiable claim fails closed
            print(
                f"not resuming acceptance of {key} from {source}: {error}",
                file=sys.stderr,
                flush=True,
            )
            continue
        try:
            result = dict(claim)
            if prior_root is not None and prior_root != str(root):
                result = _rebase(result, prior_root, str(root))
            for field in (*_DEMOTION_FIELDS, *_STALE_FIELDS):
                result.pop(field, None)
            result["issues"] = []
            result["state"] = "quality_accepted"
            result["item_root"] = str(item_root)
            result["acceptance"] = record
            result["quality_review"] = {
                **(result.get("quality_review") or {}),
                "state": "accept",
                "artifact": str(root / record["review_dir"] / "result.json"),
                "artifact_sha256": record["review_result_sha256"],
                "snapshot_hash": record["review_snapshot_hash"],
            }
            result["acceptance_resumed"] = {
                "source": source,
                "prior_root": prior_root,
                "policy": "content_bound_acceptance_not_rereviewed",
                **({"gate_proof": claim["gate_proof"]} if source == "quality_review" else {}),
            }
            result["export"] = str(_export(root, item_root))
            # The stamping helper: a recovered acceptance gets its own transition
            # and state_since instead of the demoted status's timing.
            write_status(item_root, result)
        except Exception as error:  # noqa: BLE001 -- a verified claim we could not restore
            print(
                f"not resuming acceptance of {key} from {source}: restore failed: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
            continue
        try:
            atomic_json(ledger, {"record": record, "status": result})
        except Exception as error:  # noqa: BLE001 -- status is already restored
            print(f"acceptance ledger write failed for {key}: {error}", file=sys.stderr, flush=True)
        return result
    return None


def load_seeds(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    """Load operator-supplied prior accepted statuses, grouped by item key.

    ``path`` is a status.json, a JSON list of statuses, or a directory searched
    recursively for ``status.json`` files.  Seeds are claims only: each is
    re-verified against the restored evidence before it has any effect.
    """
    if path is None:
        return {}
    files = sorted(path.rglob("status.json")) if path.is_dir() else [path]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for file in files:
        value = json.loads(file.read_text())
        for status in value if isinstance(value, list) else [value]:
            if isinstance(status, dict) and status.get("state") == "quality_accepted":
                grouped.setdefault(str(status.get("key")), []).append(status)
    return grouped
