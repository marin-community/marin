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
from fray.v2 import ResourceConfig
from rigging.filesystem import marin_prefix
from zephyr import Dataset, ZephyrContext, counters

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — keep in sync with scripts/contree_bench.py
# ---------------------------------------------------------------------------

PYTRACER_DIR = Path(__file__).resolve().parents[2] / "scripts" / "contree_pytracer"
TRACER_MOUNT = "/pytracer"
TRACE_META_MARKER = "::TRACE_META::"
RATE_LIMIT_BACKOFFS = (3, 6, 12, 24, 48)
BROAD_CHUNK_SIZE = 20
BROAD_PYTEST_FLAGS = (
    "--continue-on-collection-errors --no-header -rA --tb=line --color=no "
    "-p no:cacheprovider -W ignore::DeprecationWarning"
)
DEFAULT_TIMEOUT_S = 3600.0
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


# ---------------------------------------------------------------------------
# Surrogate scrubbing — pyarrow can't encode lone surrogates that come from
# surrogateescape decoding of non-UTF-8 filesystem bytes.
# ---------------------------------------------------------------------------


def _scrub_surrogates(s: str) -> str:
    if not s:
        return s
    return s.encode("utf-8", "replace").decode("utf-8")


# ---------------------------------------------------------------------------
# Trace splitting / row formatting
# ---------------------------------------------------------------------------


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
# The per-row map function
# ---------------------------------------------------------------------------


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
        return

    client = _client()

    # Pull image.
    try:
        image = client.images.oci(_oci_ref(image_name), timeout=timeout)
    except Exception as e:
        logger.warning("pull failed for %s: %s", instance_id, e)
        counters.increment("instances_failed_pull")
        return

    pre_trace_path = "/tmp/trace_pre.jsonl"
    post_trace_path = "/tmp/trace_post.jsonl"
    test_patch_path = "/tmp/_test.patch"
    fix_patch_path = "/tmp/_fix.patch"
    broad_trace_prefix = "/tmp/trace_broad"

    def phase_cmd(trace_path: str, apply_patch: str | None) -> str:
        apply_line = (
            f"git apply --allow-empty {apply_patch} || " "{ echo '::PATCH_FAILED::'; exit 97; }; " if apply_patch else ""
        )
        injected = test_cmd.replace("pytest", "pytest --continue-on-collection-errors", 1)
        return (
            f"{apply_line}{injected}; rc=$?; "
            f"echo '{TRACE_META_MARKER}'; "
            f"wc -lc {trace_path} 2>/dev/null || echo 'no-trace'; "
            f"exit $rc"
        )

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
        return

    workdir = tempfile.mkdtemp(prefix=f"contree_{instance_id.replace('/', '_')}_")
    workdir_path = Path(workdir)

    try:
        # ----- Phase 1: pre-patch trace ------------------------------------
        try:
            _run_with_retry(
                session,
                shell=phase_cmd(pre_trace_path, test_patch_path if test_patch else None),
                env={**common_env, "TRACER_OUTPUT": pre_trace_path},
                files=files,
                timeout=timeout,
                disposable=False,
            )
        except Exception as e:
            logger.warning("pre-phase failed for %s: %s", instance_id, e)
            counters.increment("instances_failed_pre")
            return
        pre_stdout = session.stdout or ""

        pre_jsonl = workdir_path / "pre.jsonl"
        pre_rows: dict[str, dict] = {}
        try:
            session.download(pre_trace_path, pre_jsonl)
            pre_rows = _jsonl_to_rows(pre_jsonl)
            pre_jsonl.unlink(missing_ok=True)
        except Exception as e:
            logger.warning("pre download failed for %s: %s", instance_id, e)
            counters.increment("pre_download_failed")
            pre_download_failed = True
        else:
            pre_download_failed = False

        if "::PATCH_FAILED::" in pre_stdout:
            counters.increment("instances_failed_test_patch")
            return

        # ----- Phase 2: post-patch trace -----------------------------------
        post_stdout = ""
        post_patch_ok = False
        try:
            _run_with_retry(
                session,
                shell=phase_cmd(post_trace_path, fix_patch_path if fix_patch else None),
                env={**common_env, "TRACER_OUTPUT": post_trace_path},
                files=None,
                timeout=timeout,
                disposable=False,
            )
            post_stdout = session.stdout or ""
            post_patch_ok = "::PATCH_FAILED::" not in post_stdout
        except Exception as e:
            logger.warning("post-phase failed for %s: %s", instance_id, e)

        post_jsonl = workdir_path / "post.jsonl"
        post_rows: dict[str, dict] = {}
        post_download_failed = False
        if post_patch_ok:
            try:
                session.download(post_trace_path, post_jsonl)
                post_rows = _jsonl_to_rows(post_jsonl)
                post_jsonl.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("post download failed for %s: %s", instance_id, e)
                counters.increment("post_download_failed")
                post_download_failed = True

        # ----- Phase 3: broad ----------------------------------------------
        broad_rows: dict[str, dict] = {}
        broad_complete = False
        broad_download_failed = False
        chunks: list[list[str]] = []
        if post_patch_ok:
            try:
                session.run(
                    shell="find . -name 'test_*.py' -o -name '*_test.py' | sort",
                    timeout=60,
                    disposable=False,
                ).wait()
                all_test_files = [
                    f.strip() for f in (session.stdout or "").strip().split("\n") if f.strip().endswith(".py")
                ]
                chunks = [
                    all_test_files[i : i + BROAD_CHUNK_SIZE] for i in range(0, len(all_test_files), BROAD_CHUNK_SIZE)
                ]
                broad_chunks_completed = 0
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
                        chunk_local = workdir_path / f"broad_{ci}.jsonl"
                        try:
                            session.download(chunk_trace, chunk_local)
                            broad_rows.update(_jsonl_to_rows(chunk_local))
                            chunk_local.unlink(missing_ok=True)
                        except Exception as dl_e:
                            logger.warning(
                                "broad chunk %d download failed for %s: %s",
                                ci,
                                instance_id,
                                dl_e,
                            )
                            counters.increment("broad_chunk_download_failed")
                            broad_download_failed = True
                        broad_chunks_completed += 1
                    except Exception as e:
                        err_str = str(e)
                        # 409 'Operation already completed' = chunk finished
                        # server-side, we just lost the race. Try the download;
                        # session is NOT poisoned, no recovery needed.
                        if "status=409" in err_str and "already completed" in err_str.lower():
                            # Cancel raced with completion. Op is in terminal
                            # state — could be succeeded OR failed. Try the
                            # download to find out: if trace data is there,
                            # treat as success; if not, count as a download
                            # failure and skip recovery (session not poisoned).
                            counters.increment("broad_chunk_409_race")
                            chunk_local = workdir_path / f"broad_{ci}.jsonl"
                            try:
                                session.download(chunk_trace, chunk_local)
                                broad_rows.update(_jsonl_to_rows(chunk_local))
                                chunk_local.unlink(missing_ok=True)
                                broad_chunks_completed += 1
                                counters.increment("broad_chunk_409_recovered")
                            except Exception as dl_e:
                                logger.warning(
                                    "broad chunk %d 409 with no trace for %s: %s",
                                    ci,
                                    instance_id,
                                    dl_e,
                                )
                                counters.increment("broad_chunk_download_failed")
                                broad_download_failed = True
                            continue
                        logger.warning("broad chunk %d failed for %s: %s", ci, instance_id, e)
                        counters.increment("broad_chunk_failed")
                        if "TimedOut" not in type(e).__name__ and "StateError" not in type(e).__name__:
                            continue
                        # Cancel + sleep + fresh session + reapply patches.
                        try:
                            m = re.search(
                                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                                str(e),
                            )
                            if m:
                                client._api.cancel_operation(uuid.UUID(m.group(0)))
                        except Exception:
                            pass
                        time.sleep(5)
                        try:
                            session = image.session()
                            reapply = ""
                            if test_patch:
                                reapply += (
                                    f"git apply --allow-empty {test_patch_path} || "
                                    "{ echo '::PATCH_FAILED::'; exit 97; }; "
                                )
                            if fix_patch:
                                reapply += (
                                    f"git apply --allow-empty {fix_patch_path} || "
                                    "{ echo '::PATCH_FAILED::'; exit 97; }; "
                                )
                            session.run(
                                shell=reapply + "echo RECOVERED",
                                files=files,
                                timeout=120,
                                disposable=False,
                            ).wait()
                            if "::PATCH_FAILED::" in (session.stdout or "") or session.exit_code:
                                raise RuntimeError("patch reapply failed during recovery")
                            counters.increment("broad_session_recovered")
                        except Exception as e2:
                            logger.warning("broad recovery failed for %s: %s", instance_id, e2)
                            counters.increment("broad_recovery_failed")
                            break
                broad_complete = broad_chunks_completed == len(chunks) and not broad_download_failed
            except Exception as e:
                logger.warning("broad discovery failed for %s: %s", instance_id, e)
                counters.increment("broad_discovery_failed")

        # ----- Determine status ---------------------------------------------
        if not post_patch_ok:
            status = "post_patch_failed"
        elif pre_download_failed or post_download_failed or broad_download_failed:
            status = "trace_download_failed"
        elif not broad_complete:
            status = "broad_partial"
        else:
            status = "ok"
        counters.increment(f"status_{status}")

        # ----- Yield rows ---------------------------------------------------
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
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------


def _load_images(language: str = "python", limit: int | None = None) -> list[dict]:
    """Stream Python rows out of the HF dataset and shape them for the map fn."""
    from datasets import load_dataset

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


# Pipeline configuration — adjust here, not via CLI.
LIMIT: int | None = None  # None = all 7,243 Python rows
MAX_WORKERS: int = 10  # Matches ConTree's instance_max_concurrency
WORKER_RAM: str = "4g"  # Per-worker RAM budget


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
        resources=ResourceConfig(cpu=1, ram=WORKER_RAM),
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
