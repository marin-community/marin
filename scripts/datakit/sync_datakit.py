# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sync every Datakit source's raw downloads between two MARIN-shaped prefixes.

Direction-agnostic: copies the raw downloads of each
:class:`marin.datakit.sources.DatakitSource` from ``--src-prefix`` to
``--dest-prefix`` while preserving the relative path layout below the prefix.
Typical use is GCS -> R2 (e.g. ``gs://marin-us-central1`` to
``s3://marin-na/marin``); the reverse works too.

For each source, this script builds a single :class:`StepSpec`. The step's
``fn`` runs a Zephyr pipeline that:

* Lists every object under each leaf download path on the source side
  (including files whose names start with ``.`` or ``_`` — S3/GCS treat
  these like any other key).
* Sorts the file list and groups it into deterministic shards of
  ``files_per_shard`` files (default 64). Same input set => same shards.
* For each shard, hashes the sorted ``(rel_path, fingerprint)`` pairs —
  where ``fingerprint`` is the source object's ETag/size/mtime — and uses
  that hash as the filename of the shard's JSONL output under
  ``status_prefix``. Zephyr's ``skip_existing=True`` short-circuits any
  shard whose output already exists. If a source object is rewritten, its
  fingerprint flips, the shard hash changes, the JSONL key doesn't match,
  and the shard is re-copied.
* Otherwise, streams every file in the shard from source to a sibling
  ``.tmp.<uuid>`` key at the destination and publishes it via
  :func:`zephyr.writers.atomic_rename` (the per-file copies fan out across
  ``copy_threads`` threads), then Zephyr writes the per-shard JSONL — that
  output IS the resume marker. When the destination is ``s3://``, the dst
  ``S3FileSystem`` is constructed with ``fixed_upload_size=True`` so R2
  accepts the multipart upload (AWS S3 accepts uniform-size parts too, so
  the flag is harmless there).
* After all shards complete, copies the leaf's root ``.executor_status``
  marker as a single follow-up op. Its presence on dst is what whole-source
  skip keys off — re-runs of an already-synced source short-circuit before
  any StepSpec is even built.
* Finally, sweeps any ``.tmp.<uuid.hex>`` orphans under the dst leaf —
  leftovers from prior interrupted ``atomic_rename`` runs (ours or
  upstream). They're never legitimate content; deleting them after the
  marker lands keeps the dst byte-identical to src.

Per-shard JSONL and per-source executor status all live under
``status_prefix`` (default ``gs://marin-us-central1/data/datakit/sync``)
namespaced by a hash of ``(src_prefix, dest_prefix)`` so to-R2 and from-R2
runs of the same source don't share markers (their fingerprints aren't
comparable anyway — GCS ETag vs R2 ETag are different schemes for the same
bytes). The status prefix is the permanent record of what has been copied;
there's no TTL to worry about.

Usage::

    AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \\
    AWS_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com \\
    uv run python scripts/datakit/sync_datakit.py \\
        --src-prefix gs://marin-us-central1 \\
        --dest-prefix s3://marin-na/marin
"""

from __future__ import annotations

import argparse
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec
from fray import ResourceConfig
from marin.datakit.sources import DatakitSource, all_sources
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem import marin_prefix
from rigging.log_setup import configure_logging
from zephyr import Dataset, ZephyrContext, counters
from zephyr.plan import deterministic_hash
from zephyr.writers import atomic_rename

logger = logging.getLogger(__name__)

DEFAULT_FILES_PER_SHARD = 64
DEFAULT_COPY_THREADS_PER_SHARD = 16
DEFAULT_STATUS_PREFIX = "gs://marin-us-central1/data/datakit/sync"
COPY_CHUNK_BYTES = 8 * 1024 * 1024

# Matches the ``.tmp.<uuid.hex>`` suffix that ``zephyr.writers.atomic_rename``
# uses for its intermediate write key. Used to filter orphan leftovers out
# of source listings (see ``_list_relative_files``).
_ATOMIC_RENAME_TMP_RE = re.compile(r"\.tmp\.[0-9a-f]{32}$")

# Name of the per-source completion marker that ``marin.execution`` writes
# at the root of every download leaf. We deliberately exclude it from the
# batch shards and copy it as the very last step so its presence on dst
# means "this leaf is fully synced" — enabling whole-source skip on resume.
_EXECUTOR_STATUS_FILENAME = ".executor_status"


# ----------------------------------------------------------------------
# DAG walk + path helpers
# ----------------------------------------------------------------------


def _leaf_downloads(step: StepSpec) -> list[StepSpec]:
    """Return the leaf StepSpecs (no ``deps``) reachable from ``step``.

    May yield the same leaf twice if the DAG has a diamond; the call site
    dedupes by ``output_path``.
    """
    if not step.deps:
        return [step]
    return [leaf for dep in step.deps for leaf in _leaf_downloads(dep)]


def _safe_name(source_name: str) -> str:
    """Filesystem-safe form of ``source_name`` (e.g. ``cp/biodiversity`` -> ``cp__biodiversity``)."""
    return source_name.replace("/", "__")


def _rebase(src_path: str, src_prefix: str, dst_prefix: str) -> str:
    """Replace the leading ``src_prefix`` of ``src_path`` with ``dst_prefix``."""
    s = src_prefix.rstrip("/")
    d = dst_prefix.rstrip("/")
    if not src_path.startswith(s):
        raise ValueError(f"path {src_path!r} does not start with prefix {s!r}")
    return d + src_path[len(s) :]


# ----------------------------------------------------------------------
# Zephyr worker: one shard = up to N files
# ----------------------------------------------------------------------


def _object_fingerprint(info: dict) -> str:
    """Stable per-object fingerprint that changes when the source object changes.

    Prefers the storage-layer content hash (GCS ``etag``/``md5Hash``, S3 ``ETag``)
    since that flips on every object overwrite. Falls back to ``size`` + the
    last-modified timestamp so we always include *something* metadata-shaped.
    """
    etag = info.get("etag") or info.get("ETag") or info.get("md5Hash")
    size = info.get("size") or info.get("Size")
    mtime = info.get("mtime") or info.get("LastModified") or info.get("updated")
    parts: list[str] = []
    if etag:
        parts.append(f"etag={etag}")
    if size is not None:
        parts.append(f"size={size}")
    if mtime is not None:
        parts.append(f"mtime={mtime}")
    if not parts:
        raise RuntimeError(f"no usable metadata in info dict (keys={sorted(info)})")
    return "|".join(parts)


def _shard_hash(entries: list[tuple[str, str]]) -> str:
    """Stable hash of ``[(rel_path, fingerprint)]`` — identifies the shard for resume.

    Including the per-file fingerprint means the marker invalidates whenever
    a source object is rewritten (different ETag/size/mtime), so a re-sync
    picks up the change instead of trusting a stale marker.
    """
    return format(deterministic_hash(entries), "016x")


def _copy_one(src_fs, dst_fs, src_path: str, dst_path: str) -> int:
    """Stream-copy one file from ``src_fs`` to ``dst_path`` via atomic_rename.

    Passes ``dst_fs`` into ``atomic_rename`` so the rename (server-side
    ``CopyObject`` on S3) reuses the same S3 client + connection pool as
    the upload, instead of fsspec instantiating a second default-kwargs
    instance just for the rename.
    """
    total = 0
    with atomic_rename(dst_path, fs=dst_fs) as temp_path:
        with src_fs.open(src_path, "rb") as fin, dst_fs.open(temp_path, "wb") as fout:
            while True:
                chunk = fin.read(COPY_CHUNK_BYTES)
                if not chunk:
                    break
                fout.write(chunk)
                total += len(chunk)
    return total


def _copy_shard(
    entries: list[tuple[str, str]],
    *,
    src_dir: str,
    dst_dir: str,
    copy_threads: int,
) -> dict:
    """Worker entry point — copy one shard's files in parallel.

    ``entries`` is a list of ``(rel_path, fingerprint)`` pairs; the fingerprint
    is folded into the shard hash that names the downstream JSONL output,
    so a re-listed source with updated objects yields a different filename
    and bypasses Zephyr's ``skip_existing`` short-circuit.

    Files within a shard are streamed in parallel across ``copy_threads``
    threads. Sharing one ``S3FileSystem``/``GCSFileSystem`` instance per side
    is safe — fsspec funnels async ops onto a single dedicated event loop,
    and synchronous APIs are reentrant.
    """
    src_fs, _ = fsspec.core.url_to_fs(src_dir)
    # ``fixed_upload_size=True`` (set inside ``_open_kwargs_for`` for s3 dests)
    # is required for Cloudflare R2's multipart PUT and harmless on AWS S3.
    # It only matters for the write through ``dst_fs.open(..., "wb")``; the
    # ``fs.mv`` inside atomic_rename is a single ``CopyObject`` and can use
    # any S3 client config.
    dst_fs, _ = fsspec.core.url_to_fs(dst_dir, **_open_kwargs_for(dst_dir))
    src_base = src_dir.rstrip("/")
    dst_base = dst_dir.rstrip("/")

    bytes_copied = 0
    with ThreadPoolExecutor(max_workers=copy_threads) as pool:
        futures = [
            pool.submit(_copy_one, src_fs, dst_fs, f"{src_base}/{rel}", f"{dst_base}/{rel}") for rel, _fp in entries
        ]
        # Surface the first failure immediately — a partial shard must NOT
        # produce a JSONL output, since that's what marks the shard "done".
        for fut in as_completed(futures):
            bytes_copied += fut.result()

    counters.increment("upload/shards_copied")
    counters.increment("upload/files_copied", len(entries))
    counters.increment("upload/bytes_copied", bytes_copied)
    return {"shard_hash": _shard_hash(entries), "files": len(entries), "bytes": bytes_copied}


# ----------------------------------------------------------------------
# Source listing + sharding
# ----------------------------------------------------------------------


def _list_relative_files(src_dir: str) -> list[tuple[str, str]]:
    """List ``src_dir`` recursively as ``(rel_path, fingerprint)`` pairs.

    Pulls per-file metadata in the same listing call (``detail=True``) so the
    shard hash can incorporate it without a second round trip. Includes files
    whose names start with ``.`` or ``_`` — ``fs.find`` lists all leaf keys;
    object stores have no concept of hidden files. Two intentional exclusions:

    * ``.tmp.<uuid.hex>`` orphans from prior interrupted ``atomic_rename`` runs.
    * The root-level ``.executor_status`` marker — copied separately as the
      last step (see ``_finalize_executor_status``) so its presence on dst
      cleanly signals "this leaf is fully synced".
    """
    src_fs, _ = fsspec.core.url_to_fs(src_dir)
    stripped_root = src_fs._strip_protocol(src_dir).rstrip("/")
    entries: list[tuple[str, str]] = []
    skipped_tmp = 0
    for full, info in src_fs.find(src_dir, detail=True).items():
        # ``find`` returns scheme-stripped keys (``bucket/key/...``).
        if full == stripped_root:
            continue
        if not full.startswith(stripped_root + "/"):
            raise RuntimeError(f"unexpected find result {full!r} under {stripped_root!r}")
        rel = full[len(stripped_root) + 1 :]
        if rel == _EXECUTOR_STATUS_FILENAME:
            # Copied separately at the end so its presence is the signal that
            # the whole leaf is done.
            continue
        if _ATOMIC_RENAME_TMP_RE.search(rel):
            # Orphans from a prior interrupted sync: ``zephyr.writers.atomic_rename``
            # writes ``<dst>.tmp.<uuid.hex>`` then ``fs.mv``s; if the worker is
            # SIGKILLed between the two, the temp lingers. They're never
            # legitimate datakit content — skip them so they don't propagate.
            skipped_tmp += 1
            continue
        entries.append((rel, _object_fingerprint(info)))
    if skipped_tmp:
        logger.warning("Skipped %d atomic_rename temp leftover(s) under %s", skipped_tmp, src_dir)
    entries.sort(key=lambda e: e[0])
    return entries


def _executor_status_path(dir_path: str) -> str:
    """Return ``<dir_path>/.executor_status``."""
    return f"{dir_path.rstrip('/')}/{_EXECUTOR_STATUS_FILENAME}"


def _open_kwargs_for(path: str) -> dict:
    """Constructor kwargs for the fs at ``path`` — ``fixed_upload_size=True`` for s3 dests."""
    return {"fixed_upload_size": True} if path.startswith("s3://") else {}


def _finalize_executor_status(src_dir: str, dst_dir: str) -> None:
    """Copy ``.executor_status`` from ``src_dir`` to ``dst_dir`` as a single op.

    Runs after the batch Zephyr pipeline completes successfully. The marker's
    presence on dst is what subsequent runs use to short-circuit the whole
    leaf (see ``_leaf_already_synced``).
    """
    src = _executor_status_path(src_dir)
    dst = _executor_status_path(dst_dir)
    src_fs, _ = fsspec.core.url_to_fs(src)
    if not src_fs.exists(src):
        logger.warning("source has no %s — not finalizing %s", _EXECUTOR_STATUS_FILENAME, dst_dir)
        return
    dst_fs, _ = fsspec.core.url_to_fs(dst, **_open_kwargs_for(dst))
    _copy_one(src_fs, dst_fs, src, dst)
    logger.info("Finalized %s", dst)


def _leaf_already_synced(dst_dir: str) -> bool:
    """True if ``<dst_dir>/.executor_status`` already exists on the destination."""
    dst = _executor_status_path(dst_dir)
    fs, _ = fsspec.core.url_to_fs(dst)
    return fs.exists(dst)


def _remove_tmp_orphans(dst_dir: str) -> None:
    """Delete ``.tmp.<uuid.hex>`` orphans anywhere under ``dst_dir``.

    Runs once per leaf after ``.executor_status`` is published. The orphans
    are interrupted ``atomic_rename`` writes from earlier runs (ours or
    other producers); they never become legitimate content, and leaving
    them around would make a byte-for-byte src/dst comparison fail despite
    the real data matching.
    """
    dst_fs, _ = fsspec.core.url_to_fs(dst_dir)
    if not dst_fs.exists(dst_dir):
        return
    orphans = [full for full in dst_fs.find(dst_dir) if _ATOMIC_RENAME_TMP_RE.search(full)]
    if not orphans:
        return
    logger.info("Removing %d .tmp.<uuid> orphan(s) under %s", len(orphans), dst_dir)
    dst_fs.rm(orphans)
    counters.increment("upload/tmp_orphans_removed", len(orphans))


def _shard(items: list[tuple[str, str]], shard_size: int) -> list[list[tuple[str, str]]]:
    return [items[i : i + shard_size] for i in range(0, len(items), shard_size)]


def _upload_dir(
    *,
    src_dir: str,
    dst_dir: str,
    shard_prefix: str,
    files_per_shard: int,
    copy_threads: int,
    job_name: str,
) -> None:
    """Drive the Zephyr pipeline that copies one source directory."""
    rels = _list_relative_files(src_dir)
    if not rels:
        logger.warning("No files under %s — nothing to upload.", src_dir)
        return
    shards = _shard(rels, files_per_shard)
    logger.info(
        "Uploading %d files in %d shards (x%d copy threads/shard): %s -> %s",
        len(rels),
        len(shards),
        copy_threads,
        src_dir,
        dst_dir,
    )

    # Each shard's JSONL output is named after its content hash, so
    # ``skip_existing=True`` doubles as the resume marker: if the source
    # files haven't changed the hash matches an existing key and Zephyr
    # short-circuits the whole shard. If any source file's fingerprint
    # changes, the hash changes and the shard re-runs.
    shard_paths = [f"{shard_prefix.rstrip('/')}/{_shard_hash(shard)}.jsonl.gz" for shard in shards]

    pipeline = (
        Dataset.from_list(shards)
        .map(
            lambda shard: _copy_shard(
                shard,
                src_dir=src_dir,
                dst_dir=dst_dir,
                copy_threads=copy_threads,
            )
        )
        .write_jsonl(lambda shard_idx, _total: shard_paths[shard_idx], skip_existing=True)
    )
    # Each shard fans out into ``copy_threads`` blocking I/O threads, so the
    # worker needs at least that many cores' worth of headroom.
    ctx = ZephyrContext(
        name=job_name,
        resources=ResourceConfig(cpu=max(2, copy_threads // 4), ram="4g"),
    )
    ctx.execute(pipeline)
    # Only now that every other file is at dst, publish ``.executor_status``
    # — its presence is the canonical "this leaf is fully synced" signal that
    # lets future runs skip the source entirely.
    _finalize_executor_status(src_dir, dst_dir)
    # With the marker in place, the leaf is canonical; any ``.tmp.<uuid>``
    # debris under it is orphaned and can be removed.
    _remove_tmp_orphans(dst_dir)


# ----------------------------------------------------------------------
# Public factory: one StepSpec per source
# ----------------------------------------------------------------------


def sync_source_step(
    source: DatakitSource,
    *,
    src_prefix: str,
    dest_prefix: str,
    files_per_shard: int = DEFAULT_FILES_PER_SHARD,
    copy_threads: int = DEFAULT_COPY_THREADS_PER_SHARD,
    status_prefix: str = DEFAULT_STATUS_PREFIX,
) -> StepSpec:
    """Build a StepSpec that copies ``source``'s raw downloads between two prefixes.

    Leaf download paths are listed (canonically) against ``marin_prefix()``,
    then re-anchored to both ``src_prefix`` and ``dest_prefix`` so the same
    relative-under-prefix layout is preserved on either side. Pick the
    direction by what you pass: ``src_prefix=gs://…`` + ``dest_prefix=s3://…``
    pushes to R2, the swap pulls back.

    Args:
        source: The DatakitSource whose raw downloads we want to copy.
        src_prefix: Source root to read from (e.g. ``gs://marin-us-central1``).
        dest_prefix: Destination root to write to (e.g. ``s3://marin-na/marin``).
        files_per_shard: Files per Zephyr shard (default 64). The shard
            boundaries are deterministic; ``Dataset.from_list`` preserves
            ordering so the same input set always produces the same shards.
        copy_threads: Per-shard fan-out for parallel file copies (default 16).
        status_prefix: Where per-shard JSONLs and the executor status marker
            live (default ``gs://marin-us-central1/data/datakit/sync``). The
            shard and step namespaces are further qualified by a hash of
            ``(src_prefix, dest_prefix)`` so different directions never
            share markers (their fingerprints aren't comparable anyway).

    Returns:
        A StepSpec that, when run, copies every leaf download of ``source``
        from ``src_prefix`` to ``dest_prefix``.
    """
    canonical_prefix = marin_prefix()
    canonical_leaves = sorted({leaf.output_path for leaf in _leaf_downloads(source.normalized)})
    if not canonical_leaves:
        raise ValueError(f"source {source.name!r} has no leaf downloads")
    pairs = [
        (
            _rebase(leaf, canonical_prefix, src_prefix),
            _rebase(leaf, canonical_prefix, dest_prefix),
        )
        for leaf in canonical_leaves
    ]

    safe = _safe_name(source.name)
    status_root = status_prefix.rstrip("/")
    # Encode the direction so to-R2 and from-R2 of the same source can't
    # collide on shard markers or step status.
    direction_key = format(deterministic_hash((src_prefix.rstrip("/"), dest_prefix.rstrip("/"))), "016x")

    def _run(output_path: str) -> None:
        del output_path  # the upload writes to ``dest_prefix``; status lives at status_prefix
        for src, dst in pairs:
            # Namespace the shard markers per src leaf so two leaves whose
            # ``(rel_path, fingerprint)`` shards happen to collide can't
            # cause the second leaf to be silently skipped by ``skip_existing``.
            leaf_key = format(deterministic_hash(src), "016x")
            _upload_dir(
                src_dir=src,
                dst_dir=dst,
                shard_prefix=f"{status_root}/shards/{direction_key}/{safe}/{leaf_key}",
                files_per_shard=files_per_shard,
                copy_threads=copy_threads,
                job_name=f"sync-{safe}",
            )

    # No deps: the source data already exists in ``src_prefix`` (R2 or GCS) and
    # we want to *copy* it, not re-execute the upstream download leaves. Listing
    # them as ``deps`` would make StepRunner re-materialize each leaf at
    # ``dest_prefix`` from its original origin (Amazon S3, HF, etc.) before
    # the sync step's ``fn`` even runs — which is both wasteful and defeats
    # the point of pulling from the mirror.
    return StepSpec(
        name=f"sync/{safe}",
        fn=_run,
        # Step status sits next to the shard sentinels so a single
        # bucket-lifecycle rule manages the whole sync prefix.
        override_output_path=f"{status_root}/step_status/{direction_key}/{safe}",
        hash_attrs={
            "version": "v1",
            "src_prefix": src_prefix.rstrip("/"),
            "dest_prefix": dest_prefix.rstrip("/"),
            "src_paths": canonical_leaves,
        },
    )


# ----------------------------------------------------------------------
# CLI driver
# ----------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-prefix",
        default=None,
        help="Source root to read from (e.g. 'gs://marin-us-central1'). Defaults to ``marin_prefix()``.",
    )
    parser.add_argument(
        "--dest-prefix",
        required=True,
        help="Destination root to write to, e.g. 's3://marin-na/marin'.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Source name to sync (repeatable). Default: all registered sources.",
    )
    parser.add_argument(
        "--files-per-shard",
        type=int,
        default=DEFAULT_FILES_PER_SHARD,
    )
    parser.add_argument(
        "--copy-threads",
        type=int,
        default=DEFAULT_COPY_THREADS_PER_SHARD,
        help=f"Parallel per-shard copies (default {DEFAULT_COPY_THREADS_PER_SHARD}).",
    )
    parser.add_argument(
        "--status-prefix",
        default=DEFAULT_STATUS_PREFIX,
        help=f"Where shard sentinels + step status live (default: {DEFAULT_STATUS_PREFIX!r}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build StepSpecs and list them, but don't run.",
    )
    return parser.parse_args()


def _select_sources(names: list[str] | None) -> list[DatakitSource]:
    all_src = all_sources()
    if not names:
        return list(all_src.values())
    missing = [n for n in names if n not in all_src]
    if missing:
        raise SystemExit(f"unknown source(s): {', '.join(missing)}")
    return [all_src[n] for n in names]


def _source_already_synced(source: DatakitSource, canonical_prefix: str, dest_prefix: str) -> bool:
    """True if every leaf of ``source`` already has ``.executor_status`` on dst.

    Whole-source skip is the right granularity for our pipeline because (a)
    ``.executor_status`` is only written by ``_finalize_executor_status``
    after the entire batch completed, and (b) any partial-progress shard
    markers still live under ``status_prefix`` and would handle in-leaf
    resume on their own — but if all leaves have the final marker, there's
    nothing to do for this source.
    """
    canonical_leaves = sorted({leaf.output_path for leaf in _leaf_downloads(source.normalized)})
    if not canonical_leaves:
        return False
    return all(_leaf_already_synced(_rebase(leaf, canonical_prefix, dest_prefix)) for leaf in canonical_leaves)


def main() -> None:
    args = _parse_args()
    configure_logging()
    # Boto's credential resolver and config provider log at INFO on every
    # request; with ~5 S3 API calls per file x thousands of files, that's
    # tens of thousands of log lines per shard. The information ("credentials
    # in environment variables", "endpoint via environment_global") is the
    # same on every call, so demote to WARNING.
    for name in ("aiobotocore.credentials", "botocore.configprovider"):
        logging.getLogger(name).setLevel(logging.WARNING)

    src_prefix = args.src_prefix or marin_prefix()
    logger.info("Syncing %s -> %s", src_prefix, args.dest_prefix)

    sources = _select_sources(args.source)

    # Pre-flight: drop sources whose dst already has ``.executor_status`` on
    # every leaf. Parallelized because each check is one ``fs.exists`` and
    # the listing can have 100+ sources.
    canonical_prefix = marin_prefix()
    with ThreadPoolExecutor(max_workers=32) as pool:
        already_synced_flags = list(
            pool.map(
                lambda s: _source_already_synced(s, canonical_prefix, args.dest_prefix),
                sources,
            )
        )
    todo: list[DatakitSource] = []
    for src, done in zip(sources, already_synced_flags, strict=True):
        if done:
            logger.info("Skipping %s: dst already has .executor_status on every leaf", src.name)
        else:
            todo.append(src)
    if len(todo) < len(sources):
        logger.info("Pre-flight: skipped %d/%d already-synced source(s)", len(sources) - len(todo), len(sources))

    steps = [
        sync_source_step(
            src,
            src_prefix=src_prefix,
            dest_prefix=args.dest_prefix,
            files_per_shard=args.files_per_shard,
            copy_threads=args.copy_threads,
            status_prefix=args.status_prefix,
        )
        for src in todo
    ]

    logger.info("Built %d sync StepSpec(s)", len(steps))
    for s in steps:
        logger.info("  %s -> %s", s.name, s.output_path)

    if args.dry_run:
        print(f"{len(steps)} sync StepSpec(s) built (dry run).")
        return

    StepRunner().run(steps)
    print(f"Synced {len(steps)} source(s): {src_prefix} -> {args.dest_prefix}.")


if __name__ == "__main__":
    main()
