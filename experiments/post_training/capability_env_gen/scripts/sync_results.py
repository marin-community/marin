#!/usr/bin/env python3
"""Resume and atomically snapshot a capability-pipeline result tree on CoreWeave S3."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath

import fsspec
from rigging.filesystem.s3_compat import configure_coreweave_s3

# These two reports are declared, content-free c07 mutation artifacts.  The
# uploader normally excludes every component with a credential-like prefix;
# retain that broad protection and exempt only these exact durable paths.
_DECLARED_BENIGN_ARTIFACTS = frozenset(
    {
        "items/c07.experiments-health.experimental-design.ab-analysis-3-7ab4ffa1a034/workspace/build/mutants/reports/token_case_preserved.json",
        "quality/c07.experiments-health.experimental-design.ab-analysis-3-7ab4ffa1a034/attempt-1/input/workspace/build/mutants/reports/token_case_preserved.json",
    }
)

# Pygments 2.21.0 is an input to the c02 build.  Its wheel provenance was
# checked against https://pypi.org/pypi/Pygments/2.21.0/json: the authoritative
# ``pygments-2.21.0-py3-none-any.whl`` sha256 is
# 2363c69b61c4a97c838da3b130dcd6468f4848992b21a82f2a63ec34377137d9.
# Do not use a mutable RECORD file as authority.  This is deliberately the
# exact extracted source member observed in that wheel, not a general token*
# exception.
_PYGMENTS_221_TOKEN_SHA256 = (
    "0d58a5e53da5b479138104d4632d5d32d8328a08dabc2b35ca913841fd82ac6a"
)
_PYGMENTS_TOKEN_SUFFIX = (
    "workspace",
    "build",
    "venv",
    "lib",
    "python3.11",
    "site-packages",
    "pygments",
    "token.py",
)
_PYGMENTS_CLEAN_TOOLCHAIN_TOKEN_SUFFIX = (
    "workspace",
    "clean-inputs",
    "toolchain",
    "lib",
    "pygments",
    "token.py",
)
_PYGMENTS_TOKEN_CACHE_SUFFIX = (
    "workspace",
    "build",
    "venv",
    "lib",
    "python3.11",
    "site-packages",
    "pygments",
    "__pycache__",
    "token.cpython-311.pyc",
)


def _sha256(*, path: Path | None = None, data: bytes | None = None) -> str | None:
    if data is not None:
        return hashlib.sha256(data).hexdigest()
    if path is not None and path.is_file():
        return digest(path)
    return None


def _items_root_prefix(p: PurePosixPath) -> tuple[str, ...]:
    """Return the ``items`` root prefix for a result-relative path.

    Each phase writes items under its own stage root -- ``items/`` for a
    standalone synthesize run, ``construction/items/`` for a generate run,
    ``checkpoint-revalidation/items/`` for a revalidation.  Naming the roots
    individually meant a new phase silently lost the content-bound exemption
    below and published an incomplete snapshot instead.
    """
    if p.parts[:1] == ("items",):
        return ("items",)
    if len(p.parts) > 1 and p.parts[1] == "items":
        return p.parts[:2]
    return ()


def _is_verified_pygments_token(
    relative: str, *, path: Path | None = None, data: bytes | None = None
) -> bool:
    p = PurePosixPath(relative)
    item_prefix = _items_root_prefix(p)
    return (
        bool(item_prefix)
        and any(
            len(p.parts) > len(item_prefix) + len(suffix)
            and p.parts[-len(suffix) :] == suffix
            for suffix in (
                _PYGMENTS_TOKEN_SUFFIX,
                _PYGMENTS_CLEAN_TOOLCHAIN_TOKEN_SUFFIX,
            )
        )
        and _sha256(path=path, data=data) == _PYGMENTS_221_TOKEN_SHA256
    )


def _is_verified_pygments_token_cache(
    relative: str, *, path: Path | None = None
) -> bool:
    p = PurePosixPath(relative)
    if (
        path is None
        or len(p.parts) <= len(_PYGMENTS_TOKEN_CACHE_SUFFIX)
        or not _items_root_prefix(p)
        or p.parts[-len(_PYGMENTS_TOKEN_CACHE_SUFFIX) :] != _PYGMENTS_TOKEN_CACHE_SUFFIX
    ):
        return False
    source_relative = (p.parent.parent / "token.py").as_posix()
    return _is_verified_pygments_token(
        source_relative, path=path.parent.parent / "token.py"
    )


# Exact credential names only: ``.env*`` and a component whose stem (the text
# before its first dot) is token/tokens/secret/secrets.  A prefix match also
# dropped reviewed task content such as ``evidence/battery/token_governance/``
# (42 files of one accepted healthcare task, 2026-09-29).
_CREDENTIAL_STEMS = {"token", "tokens", "secret", "secrets"}


def _credential_like(component: str) -> bool:
    lowered = component.lower()
    return lowered.startswith(".env") or lowered.split(".", 1)[0] in _CREDENTIAL_STEMS


def omission_reason(
    relative: str, *, path: Path | None = None, data: bytes | None = None
) -> str | None:
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts:
        return "unsafe_path"
    if ".tmp" in p.parts or relative.endswith(".tmp"):
        return "temporary"
    if relative in _DECLARED_BENIGN_ARTIFACTS:
        return None
    if _is_verified_pygments_token(relative, path=path, data=data):
        return None
    if _is_verified_pygments_token_cache(relative, path=path):
        return "regenerable_verified_pygments_cache"
    if any(_credential_like(component) for component in p.parts):
        return "credential_like_component"
    if p.suffix.lower() in {".pem", ".key"}:
        return "private_key_suffix"
    return None


def safe(relative: str, *, data: bytes | None = None) -> bool:
    return omission_reason(relative, data=data) is None


def digest(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def resolve(uri: str):
    configure_coreweave_s3()
    return fsspec.core.url_to_fs(uri.rstrip("/"))


_UPLOAD_WORKERS = 16
_RESTORE_WORKERS = 32


def _spool_snapshot(source: Path, cache: Path) -> tuple[Path, str]:
    """Capture file bytes privately, then address that immutable copy by hash."""
    target = cache / os.urandom(16).hex()
    with source.open("rb") as input_file, target.open("xb") as output:
        shutil.copyfileobj(input_file, output, 1024 * 1024)
    return target, digest(target)


class _LocalReadError(Exception):
    """The local file vanished or became unreadable while being captured.

    Kept distinct from upload errors: network failures (often OSError subclasses)
    must still fail the sync, but a workspace file an agent deleted or locked between
    inventory and capture must not block every future snapshot of the job.
    """


def _upload_one(fs, prefix: str, path: Path, expected: str, cache: Path, relative: str) -> str:
    obj = f"{prefix}/_objects/{expected}"
    if fs.exists(obj):
        return expected
    try:
        staged, captured = _spool_snapshot(path, cache)
    except OSError as error:
        raise _LocalReadError(type(error).__name__) from error
    try:
        # Logs may grow between inventory and upload. Publish the actual captured
        # bytes, without requiring a quiet worker. Content-based exceptions to
        # omission rules must still hold for those bytes, not an earlier read.
        if omission_reason(relative, path=staged) is not None:
            raise ValueError("captured bytes no longer satisfy publication policy")
        # fsspec's put streams from a filename.  This avoids a one-file bytes
        # allocation while retaining a byte-stable input for the object key.
        fs.put(str(staged), f"{prefix}/_objects/{captured}")
        return captured
    finally:
        staged.unlink(missing_ok=True)


# Paths a results tree never loses once written: one status per item, one verdict per
# quality attempt.  A tree holding fewer of them than the published snapshot is partial
# (2026-09-29: shard-071/075 were preempted mid-restore and a final sync replaced their
# snapshot with the half-restored tree, dropping every status and verdict).
_NEVER_SHRINK = (
    ("item statuses", re.compile(r"^items/[^/]+/status\.json$")),
    ("quality verdicts", re.compile(r"^quality/[^/]+/attempt-[0-9]+/result\.json$")),
)


def _shrink_refusal(fs, prefix: str, files: dict[str, str]) -> str | None:
    """Why publishing ``files`` would shrink the durable snapshot, or None."""
    try:
        remote = json.loads(fs.cat(f"{prefix}/_manifests/latest.json"))
    except FileNotFoundError:
        return None
    except Exception as error:  # noqa: BLE001 - unreadable: refuse this round, retry next.
        return f"published manifest unreadable ({type(error).__name__})"
    published = remote.get("files") if isinstance(remote, dict) else None
    if not isinstance(published, dict):
        return "published manifest is invalid"
    for label, pattern in _NEVER_SHRINK:
        before = sum(1 for rel in published if pattern.match(rel))
        after = sum(1 for rel in files if pattern.match(rel))
        if after < before:
            return f"{label} would drop from {before} to {after}"
    return None


def _keep_history(fs, prefix: str, manifest_path: Path, created: str, state: Path) -> None:
    """Keep one published manifest per hour so a bad publish can be rolled back."""
    hour = created[:13].replace(":", "")
    marker = state / "history-hour"
    try:
        if marker.exists() and marker.read_text() == hour:
            return
        fs.put(str(manifest_path), f"{prefix}/_manifests/history/{hour}.json")
        marker.write_text(hour)
    except Exception as error:  # noqa: BLE001 - history is best-effort; latest.json is not.
        print(json.dumps({"history": type(error).__name__}), file=sys.stderr)


# Small run-level views also published at fixed keys, so an observer can read where every
# item is without downloading the (multi-MB) manifest first.
_LIVE_FILES = ("conveyor.json", "report.json")


def _publish_live(fs, prefix: str, source: Path, current: dict[str, str], state: Path) -> None:
    record = state / "live.json"
    try:
        published = json.loads(record.read_text()) if record.exists() else {}
    except (OSError, json.JSONDecodeError):
        published = {}
    for name in _LIVE_FILES:
        sha = current.get(name)
        if sha is None or published.get(name) == sha:
            continue
        try:
            fs.put(str(source / name), f"{prefix}/_live/{name}")
            published[name] = sha
        except Exception as error:  # noqa: BLE001 - a view; the manifest is authoritative
            print(json.dumps({"live": name, "error": type(error).__name__}), file=sys.stderr)
    try:
        record.write_text(json.dumps(published))
    except OSError:
        pass


def sync(a: argparse.Namespace) -> int:
    source, state = Path(a.source), Path(a.state)
    state.mkdir(parents=True, exist_ok=True)
    fs, prefix = resolve(a.destination)
    record = state / "uploaded.json"
    previous = json.loads(record.read_text()) if record.exists() else {}
    current = {}
    pending: list[tuple[Path, str]] = []
    failures = []
    omitted = {}
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source).as_posix()
        try:
            if not path.is_file():
                continue
            reason = omission_reason(rel, path=path)
            if reason:
                omitted[rel] = reason
                continue
            current[rel] = digest(path)
        except FileNotFoundError:
            # Deleted between listing and hashing: transient, not part of this snapshot.
            current.pop(rel, None)
            continue
        except OSError as error:
            # One unreadable member (a chmod-000 file or socket an agent left in its
            # workspace) used to fail every sync forever, silently stranding the job's
            # results. Record it as an omission instead; under items/ that makes the
            # snapshot non-final, which a final sync still reports.
            current.pop(rel, None)
            omitted[rel] = f"unreadable:{type(error).__name__}"
            continue
        if previous.get(rel) == current[rel]:
            continue
        pending.append((path, current[rel]))
    if pending and not previous:
        # A fresh state dir (first sync after a relaunch restored the tree) would
        # otherwise HEAD every object one by one; 270k agent-venv files ran past the
        # sync deadline on every attempt. One paged listing answers them all.
        try:
            stored = {key.rsplit("/", 1)[-1] for key in fs.find(f"{prefix}/_objects")}
        except FileNotFoundError:
            stored = set()
        pending = [(path, expected) for path, expected in pending if expected not in stored]
    with tempfile.TemporaryDirectory(prefix="capability-sync-spool-", dir=state) as raw_cache:
        cache = Path(raw_cache)
        with concurrent.futures.ThreadPoolExecutor(max_workers=_UPLOAD_WORKERS) as pool:
            futures = {
                pool.submit(_upload_one, fs, prefix, path, expected, cache, path.relative_to(source).as_posix()): path.relative_to(source).as_posix()
                for path, expected in pending
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    current[futures[future]] = future.result()
                except _LocalReadError as error:
                    current.pop(futures[future], None)
                    omitted[futures[future]] = f"unreadable_at_capture:{error}"
                except Exception as error:  # noqa: BLE001
                    failures.append(f"{futures[future]}: {type(error).__name__}")
    required_omissions = {
        rel: reason
        for rel, reason in omitted.items()
        if rel.startswith(("items/", "construction/items/"))
        and reason != "regenerable_verified_pygments_cache"
    }
    if failures:
        print(
            json.dumps(
                {"ok": False, "failed": failures[:20], "omitted": omitted},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    final = bool(a.final) and not required_omissions
    if not previous or a.final:
        # A fresh state dir means the tree was just restored (or a relaunch); a final
        # sync may run after a killed restore.  Either may be partial: never let it
        # replace a fuller published snapshot.  The record is not written, so every
        # later round re-checks until the tree is whole again.
        refusal = _shrink_refusal(fs, prefix, current)
        if refusal:
            print(json.dumps({"ok": False, "error": "PartialTreeRefusedError", "refused": refusal}), file=sys.stderr)
            return 1
    record.write_text(json.dumps(current, sort_keys=True))
    manifest = {
        "snapshot_id": uuid.uuid4().hex,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": current,
        "omitted": omitted,
        "final": final,
    }
    mp = state / "manifest.json"
    mp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        fs.put(str(mp), f"{prefix}/_manifests/latest.json")
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "manifest": type(e).__name__}), file=sys.stderr)
        return 1
    _keep_history(fs, prefix, mp, manifest["created_utc"], state)
    _publish_live(fs, prefix, source, current, state)
    if required_omissions and a.final:
        print(
            json.dumps(
                {
                    "ok": False,
                    "files": len(current),
                    "final": False,
                    "required_omissions": required_omissions,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps({"ok": True, "files": len(current), "final": final}))
    return 0


def _writer_still_publishing(fs, prefix: str, manifest: dict) -> bool:
    """Whether the job that wrote this snapshot had not yet written its final one.

    Every snapshot carries its writer's submission.json (started) and, once the writer
    ran its EXIT trap, terminal.json (finished).  A writer that started after the last
    recorded finish may still be running its final sync: a relaunch that cancels the old
    job and restores at once used to miss the old job's last minutes (2026-09-29
    shard-065 lost an acceptance made at 21:05 that way).
    """
    files = manifest.get("files") or {}

    def stamp(name: str, field: str) -> str | None:
        sha = files.get(name)
        if not isinstance(sha, str):
            return None
        try:
            value = json.loads(fs.cat(f"{prefix}/_objects/{sha}")).get(field)
        except Exception:  # noqa: BLE001 - unreadable stamps never block a restore
            return None
        return value if isinstance(value, str) else None

    started = stamp("submission.json", "started_utc")
    finished = stamp("terminal.json", "finished_utc")
    return bool(started) and (finished is None or finished < started)


def _settled_manifest(fs, prefix: str, manifest_key: str) -> dict:
    """The snapshot to restore, after giving a still-publishing writer time to finish."""
    manifest = json.loads(fs.cat(manifest_key))
    wait = float(os.environ.get("CAPABILITY_RESTORE_WRITER_WAIT_SECONDS", "600"))
    deadline = time.monotonic() + wait
    while _writer_still_publishing(fs, prefix, manifest) and time.monotonic() < deadline:
        time.sleep(15)
        manifest = json.loads(fs.cat(manifest_key))
    if _writer_still_publishing(fs, prefix, manifest):
        print("restoring a snapshot whose writer never recorded its final sync "
              "(killed hard, or still running)", file=sys.stderr)
    return manifest


def restore(a: argparse.Namespace) -> int:
    dest = Path(a.destination)
    fs, prefix = resolve(a.source)
    try:
        objects = fs.find(prefix)
    except FileNotFoundError:
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"restore listing failed: {type(e).__name__}", file=sys.stderr)
        return 1
    manifest_key = prefix.rstrip("/") + "/_manifests/latest.json"
    if manifest_key not in objects:
        # Submission writes a reproducibility archive before the worker starts.
        # It is immutable metadata, not resumable result state, and therefore
        # must not make a fresh result prefix look like a partial prior run.
        result_objects = [
            key
            for key in objects
            if not key.removeprefix(prefix.rstrip("/") + "/").startswith(
                "_source_snapshot/"
            )
        ]
        if a.new_run and not result_objects:
            return 0
        print(
            "no verified snapshot manifest; use a new output prefix or explicitly migrate",
            file=sys.stderr,
        )
        return 1
    try:
        manifest = _settled_manifest(fs, prefix, manifest_key)
    except Exception as e:  # noqa: BLE001
        print(f"manifest read failed: {type(e).__name__}", file=sys.stderr)
        return 1
    files = manifest.get("files")
    if not isinstance(files, dict):
        print("invalid snapshot manifest", file=sys.stderr)
        return 1
    for rel, expected in files.items():
        if (
            not isinstance(rel, str)
            or not isinstance(expected, str)
            or len(expected) != 64
        ):
            print("invalid snapshot member", file=sys.stderr)
            return 1
    # Fetch each distinct object once, in parallel (a serial restore of a ~100K-file
    # snapshot took most of an hour before the job did any work). Any fetch,
    # checksum or safety failure still fails the whole restore closed.
    by_digest: dict[str, list[str]] = {}
    for rel, expected in files.items():
        by_digest.setdefault(expected, []).append(rel)

    def fetch(expected: str) -> str | None:
        rels = by_digest[expected]
        # Retry transient object-store errors (throttling under many parallel
        # restores) with backoff; still fail closed once the attempts run out.
        for attempt in range(4):
            try:
                data = fs.cat(f"{prefix}/_objects/{expected}")
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    return f"restore failed: {rels[0]}: {type(e).__name__}"
                time.sleep(0.5 * 2**attempt)
        if hashlib.sha256(data).hexdigest() != expected:
            return f"restore checksum failed: {rels[0]}"
        for rel in rels:
            if not safe(rel, data=data):
                return "invalid snapshot member"
            local = dest / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            temporary = local.with_name(local.name + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(local)
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=_RESTORE_WORKERS) as pool:
        for error in pool.map(fetch, list(by_digest)):
            if error is not None:
                print(error, file=sys.stderr)
                pool.shutdown(wait=True, cancel_futures=True)
                return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    subs = p.add_subparsers(dest="cmd", required=True)
    s = subs.add_parser("sync")
    s.add_argument("--source", required=True)
    s.add_argument("--destination", required=True)
    s.add_argument("--state", required=True)
    s.add_argument("--final", action="store_true")
    r = subs.add_parser("restore")
    r.add_argument("--source", required=True)
    r.add_argument("--destination", required=True)
    r.add_argument("--new-run", action="store_true")
    a = p.parse_args()
    return sync(a) if a.cmd == "sync" else restore(a)


if __name__ == "__main__":
    raise SystemExit(main())
