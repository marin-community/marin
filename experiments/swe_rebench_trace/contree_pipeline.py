# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Zephyr pipeline that traces SWE-rebench-V2 instances via Nebius ConTree.

For each Python row in ``nebius/SWE-rebench-V2`` this opens a ConTree session,
applies the row's ``test_patch`` (pre-patch trace) and then ``patch`` (post-
patch trace), runs the row's ``test_cmd`` under the marin tracer, and finally
runs the *full* repo test suite (broad phase) so we capture every test, not
just the PR-affected ones.

The actual per-row work lives in :func:`contree_trace_one` which yields one
dict per (instance_id, test_id) combination — Zephyr writes the resulting
stream to sharded parquet via ``write_parquet``.

Compared to the runsc-based ``pipeline.py`` in the same directory, this
variant uses the ConTree HTTP service for sandboxing — no local container
runtime needed on the worker.

Configuration is internal (LIMIT, MAX_WORKERS, WORKER_RAM near the bottom of
this file). Output goes to ``$MARIN_PREFIX/raw/swe-rebench-contree-traces``.

Submit on Iris::

    uv run iris --cluster=marin job run --extra contree \\
        -e CONTREE_BASE_URL "$CONTREE_BASE_URL" \\
        -e CONTREE_TOKEN "$CONTREE_TOKEN" \\
        -- python -m experiments.swe_rebench_trace.contree_pipeline
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import tempfile
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
from contree_sdk._internals.utils.wrapper import coro_sync
from fray import ResourceConfig
from rigging.filesystem import marin_prefix
from datasets import load_dataset
from zephyr import Dataset, ZephyrContext, counters

logger = logging.getLogger(__name__)

PYTRACER_DIR = Path(__file__).resolve().parent / "pytracer"
TRACER_MOUNT = "/pytracer"
TRACE_META_MARKER = "::TRACE_META::"
RATE_LIMIT_BACKOFFS = (3, 6, 12, 24, 48)
BROAD_CHUNK_SIZE = 20
BROAD_PYTEST_FLAGS = (
    "--continue-on-collection-errors --no-header -rA --tb=line --color=no "
    "-p no:cacheprovider -W ignore::DeprecationWarning"
)
DEFAULT_TIMEOUT_S = 600.0  # First-pass: cut heavy-tail shards short, recover via deferred-status follow-up.
# Output goes under MARIN_PREFIX (set on Iris workers; falls back to /tmp/marin
# locally). Convention follows datakit: <prefix>/raw/<dataset-name>/.
DEFAULT_OUTPUT_NAME = "raw/swe-rebench-contree-traces"

OUTPUT_SCHEMA = pa.schema(
    [
        ("instance_id", pa.string()),
        ("test_id", pa.string()),
        ("file", pa.string()),
        ("function", pa.string()),
        ("affected", pa.bool_()),
        ("text", pa.string()),
        ("pre_event_count", pa.int64()),
        ("post_event_count", pa.int64()),
        ("pre_depth_cap", pa.int64()),
        ("post_depth_cap", pa.int64()),
        ("post_patch_applied", pa.bool_()),
        ("broad_complete", pa.bool_()),
        ("status", pa.string()),
    ]
)


def _deferred_row(instance_id: str, status: str) -> dict:
    """Sentinel row emitted when a phase fails before producing any traces.

    Without this, an instance that times out on pre-phase would just disappear
    from the output (the early-return yields nothing). With it, a follow-up
    pipeline can recover the deferred instance_ids by querying the parquet
    for ``status IN ('pre_failed', 'session_failed', ...)``.
    """
    return {
        "instance_id": instance_id,
        "test_id": "",
        "file": "",
        "function": "",
        "affected": False,
        "text": "",
        "pre_event_count": 0,
        "post_event_count": 0,
        "pre_depth_cap": -1,
        "post_depth_cap": -1,
        "post_patch_applied": False,
        "broad_complete": False,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Surrogate scrubbing — pyarrow can't encode lone surrogates that come from
# surrogateescape decoding of non-UTF-8 filesystem bytes.
# ---------------------------------------------------------------------------


def _scrub_surrogates(s: str) -> str:
    if not s:
        return s
    return s.encode("utf-8", "replace").decode("utf-8")


def _split_trace(trace: str) -> tuple[str, str]:
    """Return (test_source, execution_trace) given a per-test trace string."""
    marker = "# --- execution trace ---"
    if marker in trace:
        head, _, tail = trace.partition(marker)
        head = head.replace("# --- test source ---\n", "", 1).rstrip("\n")
        return head, tail.lstrip("\n")
    return "", trace


def _format_affected_row(test_code: str, pre_trace: str, patch: str, post_trace: str) -> str:
    return (
        f"{test_code}\n"
        f"\n# --- pre-patch trace ---\n{pre_trace}\n"
        f"\n# --- patch ---\n{patch}\n"
        f"\n# --- post-patch trace ---\n{post_trace}\n"
    )


def _format_broad_row(test_code: str, trace: str) -> str:
    return f"{test_code}\n\n# --- trace ---\n{trace}\n"


def _jsonl_to_rows(jsonl_path: Path) -> dict[str, dict]:
    """Parse a phase's trace jsonl into {test_id: row_dict} with surrogate scrub."""
    out: dict[str, dict] = {}
    if not jsonl_path.exists():
        return out
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            tid = _scrub_surrogates(rec.get("test_id") or "")
            if not tid:
                continue
            rec["test_id"] = tid
            rec["trace"] = _scrub_surrogates(rec.get("trace") or "")
            rec["file"] = _scrub_surrogates(rec.get("file") or "")
            rec["function"] = _scrub_surrogates(rec.get("function") or "")
            out[tid] = rec
    return out


# ---------------------------------------------------------------------------
# ContreeSync client — cached per-worker (per-process) so the auth token + HTTP
# pool persists across map calls on the same worker actor.
# ---------------------------------------------------------------------------

_CLIENT_CACHE: dict[int, object] = {}


def _client():
    from contree_sdk import ContreeSync

    pid = os.getpid()
    client = _CLIENT_CACHE.get(pid)
    if client is None:
        base_url = os.environ.get("CONTREE_BASE_URL")
        token = os.environ.get("CONTREE_TOKEN")
        if not base_url or not token:
            raise RuntimeError("CONTREE_BASE_URL and CONTREE_TOKEN must be set in the worker env")
        client = ContreeSync(base_url=base_url, token=token)
        _CLIENT_CACHE[pid] = client
    return client


# ---------------------------------------------------------------------------
# Retry-on-429 wrapper (ConTree caps concurrent instances per token).
# ---------------------------------------------------------------------------


def _run_with_retry(session, **kwargs):
    """session.run(**kwargs).wait() with exponential backoff on 429s."""
    for sleep_s in (*RATE_LIMIT_BACKOFFS, None):
        try:
            return session.run(**kwargs).wait()
        except Exception as e:
            if "TooManyRequests" in type(e).__name__ and sleep_s is not None:
                time.sleep(sleep_s)
                continue
            raise


# ---------------------------------------------------------------------------
# OCI ref normalization (Docker Hub doesn't allow underscores in repo names).
# ---------------------------------------------------------------------------


def _oci_ref(image_name: str) -> str:
    ref = image_name.replace("docker.io/", "docker://registry-1.docker.io/", 1)
    host_repo, sep, tag = ref.rpartition(":")
    if sep and tag and "/" not in tag:
        ref = f"{host_repo.replace('_', '-')}:{tag}"
    return ref


def _tracer_injection() -> tuple[dict[str, str], dict[str, str]]:
    return (
        {"PYTHONPATH": TRACER_MOUNT},
        {f"{TRACER_MOUNT}/sitecustomize.py": str(PYTRACER_DIR / "sitecustomize.py")},
    )


# ---------------------------------------------------------------------------
# Phase helpers — each runs one of the three trace phases and returns the
# parsed-row dict plus per-phase status flags. ``contree_trace_one`` chains
# them together, threading the same session.
# ---------------------------------------------------------------------------


def _phase_cmd(test_cmd: str, trace_path: str, apply_patch_path: str | None) -> str:
    """Build a shell line that optionally applies a patch, runs ``test_cmd`` traced, and prints a trace-meta marker."""
    apply_line = (
        f"git apply --allow-empty {apply_patch_path} || " "{ echo '::PATCH_FAILED::'; exit 97; }; "
        if apply_patch_path
        else ""
    )
    injected = test_cmd.replace("pytest", "pytest --continue-on-collection-errors", 1)
    return (
        f"{apply_line}{injected}; rc=$?; "
        f"echo '{TRACE_META_MARKER}'; "
        f"wc -lc {trace_path} 2>/dev/null || echo 'no-trace'; "
        f"exit $rc"
    )


def _run_pre_phase(
    session,
    *,
    instance_id: str,
    test_cmd: str,
    test_patch: str,
    test_patch_path: str,
    pre_trace_path: str,
    common_env: dict[str, str],
    files: dict[str, str],
    workdir_path: Path,
    timeout: float,
) -> tuple[dict[str, dict], str, bool] | None:
    """Phase 1: apply ``test_patch`` and run ``test_cmd`` traced.

    Returns ``(rows, stdout, download_failed)`` or ``None`` on a hard failure
    that means we should give up on this instance entirely.
    """
    try:
        _run_with_retry(
            session,
            shell=_phase_cmd(test_cmd, pre_trace_path, test_patch_path if test_patch else None),
            env={**common_env, "TRACER_OUTPUT": pre_trace_path},
            files=files,
            timeout=timeout,
            disposable=False,
        )
    except Exception as e:
        logger.warning("pre-phase failed for %s: %s", instance_id, e)
        counters.increment("instances_failed_pre")
        return None

    stdout = session.stdout or ""
    pre_jsonl = workdir_path / "pre.jsonl"
    try:
        session.download(pre_trace_path, pre_jsonl)
        rows = _jsonl_to_rows(pre_jsonl)
        pre_jsonl.unlink(missing_ok=True)
        return rows, stdout, False
    except Exception as e:
        logger.warning("pre download failed for %s: %s", instance_id, e)
        counters.increment("pre_download_failed")
        return {}, stdout, True


def _run_post_phase(
    session,
    *,
    instance_id: str,
    test_cmd: str,
    fix_patch: str,
    fix_patch_path: str,
    post_trace_path: str,
    common_env: dict[str, str],
    workdir_path: Path,
    timeout: float,
) -> tuple[dict[str, dict], bool, bool]:
    """Phase 2: apply ``fix_patch`` and rerun ``test_cmd`` traced.

    Returns ``(rows, post_patch_ok, download_failed)``. ``post_patch_ok=False``
    means the fix patch didn't apply cleanly — broad phase is then skipped.
    """
    post_patch_ok = False
    try:
        _run_with_retry(
            session,
            shell=_phase_cmd(test_cmd, post_trace_path, fix_patch_path if fix_patch else None),
            env={**common_env, "TRACER_OUTPUT": post_trace_path},
            files=None,
            timeout=timeout,
            disposable=False,
        )
        post_patch_ok = "::PATCH_FAILED::" not in (session.stdout or "")
    except Exception as e:
        logger.warning("post-phase failed for %s: %s", instance_id, e)

    if not post_patch_ok:
        return {}, False, False

    post_jsonl = workdir_path / "post.jsonl"
    try:
        session.download(post_trace_path, post_jsonl)
        rows = _jsonl_to_rows(post_jsonl)
        post_jsonl.unlink(missing_ok=True)
        return rows, True, False
    except Exception as e:
        logger.warning("post download failed for %s: %s", instance_id, e)
        counters.increment("post_download_failed")
        return {}, True, True


def _download_broad_chunk(
    session, *, chunk_trace: str, workdir_path: Path, chunk_index: int, instance_id: str
) -> dict[str, dict] | None:
    """Pull and parse one broad-phase chunk's trace. ``None`` = download failed."""
    chunk_local = workdir_path / f"broad_{chunk_index}.jsonl"
    try:
        session.download(chunk_trace, chunk_local)
        rows = _jsonl_to_rows(chunk_local)
        chunk_local.unlink(missing_ok=True)
        return rows
    except Exception as e:
        logger.warning("broad chunk %d download failed for %s: %s", chunk_index, instance_id, e)
        counters.increment("broad_chunk_download_failed")
        return None


def _recover_broad_session(
    image,
    client,
    error: Exception,
    *,
    instance_id: str,
    test_patch: str,
    fix_patch: str,
    test_patch_path: str,
    fix_patch_path: str,
    files: dict[str, str],
):
    """Resurrect a fresh session after a failed broad chunk.

    Cancels the failed op, sleeps, opens a new session, and reapplies both
    patches. Returns the new session, or ``None`` if recovery fails.
    """
    try:
        m = re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", str(error))
        if m:
            # _api is the async client; ContreeSync runs coroutines on a
            # daemon-thread loop via coro_sync. Calling _api.* directly returns
            # an unawaited coroutine, so we reuse the SDK's wrapper here.
            coro_sync(client._api.cancel_operation(uuid.UUID(m.group(0))))
    except Exception as cancel_err:
        logger.info("cancel_operation best-effort failed for %s: %s", instance_id, cancel_err)
    time.sleep(5)
    try:
        new_session = image.session()
        reapply = ""
        if test_patch:
            reapply += f"git apply --allow-empty {test_patch_path} || " "{ echo '::PATCH_FAILED::'; exit 97; }; "
        if fix_patch:
            reapply += f"git apply --allow-empty {fix_patch_path} || " "{ echo '::PATCH_FAILED::'; exit 97; }; "
        new_session.run(
            shell=reapply + "echo RECOVERED",
            files=files,
            timeout=120,
            disposable=False,
        ).wait()
        if "::PATCH_FAILED::" in (new_session.stdout or "") or new_session.exit_code:
            raise RuntimeError("patch reapply failed during recovery")
        counters.increment("broad_session_recovered")
        return new_session
    except Exception as e2:
        logger.warning("broad recovery failed for %s: %s", instance_id, e2)
        counters.increment("broad_recovery_failed")
        return None


def _run_broad_phase(
    session,
    image,
    client,
    *,
    instance_id: str,
    test_patch: str,
    fix_patch: str,
    test_patch_path: str,
    fix_patch_path: str,
    files: dict[str, str],
    common_env: dict[str, str],
    broad_trace_prefix: str,
    workdir_path: Path,
    timeout: float,
) -> tuple[dict[str, dict], bool, bool]:
    """Phase 3: discover all repo tests and trace them in fixed-size chunks.

    Returns ``(rows, complete, download_failed)``. ``complete=True`` means
    every chunk both ran and downloaded successfully.
    """
    try:
        session.run(
            shell="find . -name 'test_*.py' -o -name '*_test.py' | sort",
            timeout=60,
            disposable=False,
        ).wait()
        all_test_files = [f.strip() for f in (session.stdout or "").strip().split("\n") if f.strip().endswith(".py")]
        chunks = [all_test_files[i : i + BROAD_CHUNK_SIZE] for i in range(0, len(all_test_files), BROAD_CHUNK_SIZE)]
    except Exception as e:
        logger.warning("broad discovery failed for %s: %s", instance_id, e)
        counters.increment("broad_discovery_failed")
        return {}, False, False

    rows: dict[str, dict] = {}
    download_failed = False
    completed = 0
    for ci, chunk in enumerate(chunks):
        chunk_trace = f"{broad_trace_prefix}_{ci}.jsonl"
        chunk_files = " ".join(shlex.quote(f) for f in chunk)
        chunk_shell = (
            f"pytest {BROAD_PYTEST_FLAGS} {chunk_files}; rc=$?; "
            f"echo '{TRACE_META_MARKER}'; "
            f"wc -lc {chunk_trace} 2>/dev/null || echo 'no-trace'; "
            f"exit $rc"
        )
        try:
            _run_with_retry(
                session,
                shell=chunk_shell,
                env={**common_env, "TRACER_OUTPUT": chunk_trace},
                files=None,
                timeout=timeout,
                disposable=False,
            )
        except Exception as e:
            err_str = str(e)
            # 409 'Operation already completed' = chunk finished server-side,
            # we just lost the cancel race. Try the download to see if the
            # trace is there; session is NOT poisoned, no recovery needed.
            if "status=409" in err_str and "already completed" in err_str.lower():
                counters.increment("broad_chunk_409_race")
                chunk_rows = _download_broad_chunk(
                    session,
                    chunk_trace=chunk_trace,
                    workdir_path=workdir_path,
                    chunk_index=ci,
                    instance_id=instance_id,
                )
                if chunk_rows is not None:
                    rows.update(chunk_rows)
                    completed += 1
                    counters.increment("broad_chunk_409_recovered")
                else:
                    download_failed = True
                continue
            logger.warning("broad chunk %d failed for %s: %s", ci, instance_id, e)
            counters.increment("broad_chunk_failed")
            if "TimedOut" not in type(e).__name__ and "StateError" not in type(e).__name__:
                continue
            session = _recover_broad_session(
                image,
                client,
                e,
                instance_id=instance_id,
                test_patch=test_patch,
                fix_patch=fix_patch,
                test_patch_path=test_patch_path,
                fix_patch_path=fix_patch_path,
                files=files,
            )
            if session is None:
                break
            continue

        chunk_rows = _download_broad_chunk(
            session, chunk_trace=chunk_trace, workdir_path=workdir_path, chunk_index=ci, instance_id=instance_id
        )
        if chunk_rows is not None:
            rows.update(chunk_rows)
            completed += 1
        else:
            download_failed = True

    complete = completed == len(chunks) and not download_failed
    return rows, complete, download_failed


def _determine_status(
    *,
    post_patch_ok: bool,
    pre_download_failed: bool,
    post_download_failed: bool,
    broad_download_failed: bool,
    broad_complete: bool,
) -> str:
    if not post_patch_ok:
        return "post_patch_failed"
    if pre_download_failed or post_download_failed or broad_download_failed:
        return "trace_download_failed"
    if not broad_complete:
        return "broad_partial"
    return "ok"


def contree_trace_one(img: dict, *, timeout: float = DEFAULT_TIMEOUT_S) -> Iterator[dict]:
    """Pull, sandbox, and trace one SWE-rebench-V2 row via ConTree.

    Yields one dict per (instance_id, test_id) combination. Calls
    ``zephyr.counters.increment`` for visibility into the live run.
    """
    instance_id = img["instance_id"]
    image_name = img["image_name"]
    test_cmd = img["test_cmd"]
    test_patch = img.get("test_patch") or ""
    fix_patch = img.get("patch") or ""

    counters.increment("instances_started")

    if not test_cmd or not image_name:
        counters.increment("instances_failed_missing_input")
        yield _deferred_row(instance_id, "missing_input")
        return

    client = _client()

    try:
        image = client.images.oci(_oci_ref(image_name), timeout=timeout)
    except Exception as e:
        logger.warning("pull failed for %s: %s", instance_id, e)
        counters.increment("instances_failed_pull")
        yield _deferred_row(instance_id, "image_pull_failed")
        return

    pre_trace_path = "/tmp/trace_pre.jsonl"
    post_trace_path = "/tmp/trace_post.jsonl"
    test_patch_path = "/tmp/_test.patch"
    fix_patch_path = "/tmp/_fix.patch"
    broad_trace_prefix = "/tmp/trace_broad"

    common_env, files = _tracer_injection()

    tp_tmp = tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False)
    tp_tmp.write(test_patch)
    tp_tmp.close()
    fp_tmp = tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False)
    fp_tmp.write(fix_patch)
    fp_tmp.close()
    files[test_patch_path] = tp_tmp.name
    files[fix_patch_path] = fp_tmp.name

    def cleanup_temp_files() -> None:
        Path(tp_tmp.name).unlink(missing_ok=True)
        Path(fp_tmp.name).unlink(missing_ok=True)

    try:
        session = image.session()
    except Exception as e:
        logger.warning("session failed for %s: %s", instance_id, e)
        counters.increment("instances_failed_session")
        cleanup_temp_files()
        yield _deferred_row(instance_id, "session_failed")
        return

    workdir = tempfile.mkdtemp(prefix=f"contree_{instance_id.replace('/', '_')}_")
    workdir_path = Path(workdir)

    try:
        pre_result = _run_pre_phase(
            session,
            instance_id=instance_id,
            test_cmd=test_cmd,
            test_patch=test_patch,
            test_patch_path=test_patch_path,
            pre_trace_path=pre_trace_path,
            common_env=common_env,
            files=files,
            workdir_path=workdir_path,
            timeout=timeout,
        )
        if pre_result is None:
            yield _deferred_row(instance_id, "pre_failed")
            return
        pre_rows, pre_stdout, pre_download_failed = pre_result
        if "::PATCH_FAILED::" in pre_stdout:
            counters.increment("instances_failed_test_patch")
            yield _deferred_row(instance_id, "test_patch_failed")
            return

        post_rows, post_patch_ok, post_download_failed = _run_post_phase(
            session,
            instance_id=instance_id,
            test_cmd=test_cmd,
            fix_patch=fix_patch,
            fix_patch_path=fix_patch_path,
            post_trace_path=post_trace_path,
            common_env=common_env,
            workdir_path=workdir_path,
            timeout=timeout,
        )

        broad_rows: dict[str, dict] = {}
        broad_complete = False
        broad_download_failed = False
        if post_patch_ok:
            broad_rows, broad_complete, broad_download_failed = _run_broad_phase(
                session,
                image,
                client,
                instance_id=instance_id,
                test_patch=test_patch,
                fix_patch=fix_patch,
                test_patch_path=test_patch_path,
                fix_patch_path=fix_patch_path,
                files=files,
                common_env=common_env,
                broad_trace_prefix=broad_trace_prefix,
                workdir_path=workdir_path,
                timeout=timeout,
            )

        status = _determine_status(
            post_patch_ok=post_patch_ok,
            pre_download_failed=pre_download_failed,
            post_download_failed=post_download_failed,
            broad_download_failed=broad_download_failed,
            broad_complete=broad_complete,
        )
        counters.increment(f"status_{status}")

        # Affected rows: tests touched by either pre or post phase.
        affected_ids = sorted(set(pre_rows) | set(post_rows))
        for tid in affected_ids:
            pr = pre_rows.get(tid) or {}
            po = post_rows.get(tid) or {}
            any_row = pr or po
            test_source, _ = _split_trace(any_row.get("trace", ""))
            _, pre_trace = _split_trace(pr.get("trace", "")) if pr else ("", "")
            _, post_trace = _split_trace(po.get("trace", "")) if po else ("", "")
            text = _format_affected_row(test_source, pre_trace, fix_patch, post_trace)
            counters.increment("affected_rows")
            counters.increment("chars", len(text))
            yield {
                "instance_id": instance_id,
                "test_id": tid,
                "file": any_row.get("file", ""),
                "function": any_row.get("function", ""),
                "affected": True,
                "text": text,
                "pre_event_count": int(pr.get("event_count", 0)),
                "post_event_count": int(po.get("event_count", 0)),
                "pre_depth_cap": int(pr.get("final_depth_cap", -1)),
                "post_depth_cap": int(po.get("final_depth_cap", -1)),
                "post_patch_applied": post_patch_ok,
                "broad_complete": broad_complete,
                "status": status,
            }

        # Broad rows: every other test in the repo, post-patch only.
        affected_set = set(affected_ids)
        for tid, rec in sorted(broad_rows.items()):
            if tid in affected_set:
                continue
            test_source, trace = _split_trace(rec.get("trace", ""))
            text = _format_broad_row(test_source, trace)
            counters.increment("broad_rows")
            counters.increment("chars", len(text))
            yield {
                "instance_id": instance_id,
                "test_id": tid,
                "file": rec.get("file", ""),
                "function": rec.get("function", ""),
                "affected": False,
                "text": text,
                "pre_event_count": 0,
                "post_event_count": int(rec.get("event_count", 0)),
                "pre_depth_cap": -1,
                "post_depth_cap": int(rec.get("final_depth_cap", -1)),
                "post_patch_applied": post_patch_ok,
                "broad_complete": broad_complete,
                "status": status,
            }

        counters.increment("instances_completed")
    finally:
        cleanup_temp_files()
        try:
            for p in workdir_path.iterdir():
                p.unlink(missing_ok=True)
            workdir_path.rmdir()
        except Exception as cleanup_err:
            logger.info("workdir cleanup failed for %s: %s", workdir_path, cleanup_err)


def _load_images(language: str = "python", limit: int | None = None) -> list[dict]:
    """Stream Python rows out of the HF dataset and shape them for the map fn."""
    ds = load_dataset("nebius/SWE-rebench-V2", split="train", streaming=True)
    out: list[dict] = []
    for row in ds:
        if row.get("language") != language:
            continue
        install_cfg = row.get("install_config") or {}
        test_cmd = install_cfg.get("test_cmd")
        if not test_cmd or not row.get("image_name"):
            continue
        out.append(
            {
                "instance_id": row["instance_id"],
                "image_name": row["image_name"],
                "test_cmd": test_cmd,
                "test_patch": row.get("test_patch") or "",
                "patch": row.get("patch") or "",
            }
        )
        if limit is not None and len(out) >= limit:
            break
    return out


LIMIT: int | None = None
MAX_WORKERS: int = 10  # ConTree's instance_max_concurrency
WORKER_RAM: str = "4g"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    output = f"{marin_prefix().rstrip('/')}/{DEFAULT_OUTPUT_NAME}"
    output_pattern = f"{output}/traces-{{shard:05d}}.parquet"

    images = _load_images(limit=LIMIT)
    logger.info("Loaded %d Python rows from nebius/SWE-rebench-V2", len(images))

    pipeline: Dataset = (
        Dataset.from_list(images)
        .flat_map(contree_trace_one)
        .write_parquet(output_pattern, schema=OUTPUT_SCHEMA, skip_existing=True)
    )

    ctx = ZephyrContext(
        max_workers=MAX_WORKERS,
        resources=ResourceConfig(cpu=1, ram=WORKER_RAM, preemptible=True),
        coordinator_resources=ResourceConfig(cpu=1, ram="2g", preemptible=True),
        name="swe-rebench-contree",
    )

    logger.info(
        "Submitting pipeline (output=%s, limit=%s, max_workers=%d, worker_ram=%s)",
        output,
        LIMIT,
        MAX_WORKERS,
        WORKER_RAM,
    )

    result = ctx.execute(pipeline)
    logger.info("Pipeline complete: wrote %d shard files", len(result.results))
    print("\nFinal counters:")
    for k, v in sorted(result.counters.items()):
        print(f"  {k:35s}  {v:,}")


if __name__ == "__main__":
    main()
