# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marina backend of the Forge tasks app: one task's record and its solver transcripts, by snapshot.

Everything the overview and the task browser need ships in the package (``dist/data``). This backend serves only
what a task page adds, from the files ``build_data.py`` wrote under the export's ``app/<snapshot>/``:

- ``/detail?snap=S&id=<capability>/<slot>``: one task. Its subject's shard is read once per process and kept; each
  task is a gzip member sliced out of it, so serving costs no compression work in the shared Marina process.
- ``/trials?snap=S&cap=<capability>``: the trimmed transcripts of every task of one capability, passed through as
  stored (gzip). The page shows these at once.
- ``/transcript?id=<capability>/<slot>&trial=N``: one trial's complete transcript, read from the export record when
  the reader expands it.
- ``/warm?snap=S``: called once by every page load; the first call for a snapshot loads all its subject shards in the
  background, so a reader's first task opens from memory instead of waiting on object storage.

Snapshots are immutable, so responses carry a long public cache lifetime; a new snapshot is a new URL.
"""

import concurrent.futures
import functools
import gzip
import json
import logging
import re
import threading

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from marina.applets import AppletServices
from rigging.filesystem.buckets import filesystem_for

APP_ROOT = "s3://marin-us-east-02a/marin/forge-viz/v1/app"
# Exported records of accepted workpieces; written once per workpiece and never rewritten, so they cache like a snapshot.
RECORDS_ROOT = "s3://marin-us-east-02a/marin/forge-viz/v1/records"
FULL_TASKS = 64
SNAPSHOT = re.compile(r"\d{8}T\d{4}Z")
CAPABILITY = re.compile(r"d\d{2}(\.[\w-]+)+")
TASK = re.compile(r"(d\d{2}(?:\.[\w-]+)+)/(\d{1,3})")
CACHE_CONTROL = "public, max-age=604800, immutable"
TRIAL_SHARDS = 512
WARM_THREADS = 2
PACK_HEADER_BYTES = 8

logger = logging.getLogger(__name__)


def gzip_json(body: bytes) -> Response:
    return Response(
        body,
        media_type="application/json",
        headers={"Content-Encoding": "gzip", "Cache-Control": CACHE_CONTROL, "Vary": "Accept-Encoding"},
    )


def create_api(services: AppletServices) -> FastAPI:
    api = FastAPI()
    fs, root = filesystem_for(APP_ROOT)
    records_fs, records_root = filesystem_for(RECORDS_ROOT)

    def read(path: str) -> bytes:
        try:
            return fs.cat_file(f"{root}/{path}")
        except FileNotFoundError:
            raise HTTPException(404, f"{path} is not in this snapshot") from None

    # One load per (snapshot, subject): a request that arrives while the warmer is reading its shard waits for that
    # read instead of starting a second one. A snapshot holds 45 shards and is replaced rarely, so this is not evicted.
    shard_loads: dict[tuple[str, str], concurrent.futures.Future] = {}
    shard_lock = threading.Lock()

    def load_subject(snapshot: str, subject: str) -> dict[str, bytes]:
        # build_data.packed_shard: an index length, the {id: [offset, length]} index, then one gzip member per task.
        body = read(f"{snapshot}/detail/{subject}.pack")
        start = PACK_HEADER_BYTES + int.from_bytes(body[:PACK_HEADER_BYTES], "big")
        index = json.loads(body[PACK_HEADER_BYTES:start])
        return {identity: body[start + offset : start + offset + length] for identity, (offset, length) in index.items()}

    def subject_tasks(snapshot: str, subject: str) -> dict[str, bytes]:
        key = (snapshot, subject)
        with shard_lock:
            load = shard_loads.get(key)
            owner = load is None
            if owner:
                load = shard_loads[key] = concurrent.futures.Future()
        if owner:
            try:
                load.set_result(load_subject(snapshot, subject))
            except Exception as error:
                load.set_exception(error)
                with shard_lock:
                    del shard_loads[key]
        return load.result()

    @functools.lru_cache(maxsize=TRIAL_SHARDS)
    def capability_trials(snapshot: str, capability: str) -> bytes:
        return read(f"{snapshot}/trials/{capability}.json.gz")

    def checked_snapshot(snap: str) -> str:
        if not SNAPSHOT.fullmatch(snap):
            raise HTTPException(400, "unknown snapshot")
        return snap

    @api.get("/detail")
    def detail(snap: str = Query(...), task: str = Query(..., alias="id")) -> Response:
        match = TASK.fullmatch(task)
        if match is None:
            raise HTTPException(400, "a task id is <capability>/<slot>")
        identity = f"{match.group(1)}/{int(match.group(2)):02d}"
        tasks = subject_tasks(checked_snapshot(snap), match.group(1).split(".", 1)[0].upper())
        if identity not in tasks:
            raise HTTPException(404, f"{identity} is not in this snapshot")
        return gzip_json(tasks[identity])

    @api.get("/trials")
    def trials(snap: str = Query(...), cap: str = Query(...)) -> Response:
        if not CAPABILITY.fullmatch(cap):
            raise HTTPException(400, "not a capability id")
        return gzip_json(capability_trials(checked_snapshot(snap), cap))

    @functools.lru_cache(maxsize=FULL_TASKS)
    def full_trials(identity: str) -> dict[int, bytes]:
        try:
            body = records_fs.cat_file(f"{records_root}/{identity}/trials.json.gz")
        except FileNotFoundError:
            raise HTTPException(404, f"{identity} has no exported transcripts") from None
        return {
            trial["trial"]: gzip.compress(json.dumps(trial, separators=(",", ":")).encode(), mtime=0)
            for trial in json.loads(gzip.decompress(body))["trials"]
        }

    @api.get("/transcript")
    def transcript(task: str = Query(..., alias="id"), trial: int = Query(..., ge=0, le=63)) -> Response:
        match = TASK.fullmatch(task)
        if match is None:
            raise HTTPException(400, "a task id is <capability>/<slot>")
        found = full_trials(f"{match.group(1)}/{int(match.group(2)):02d}")
        if trial not in found:
            raise HTTPException(404, f"trial {trial} is not recorded")
        return gzip_json(found[trial])

    warmed: set[str] = set()
    warm_lock = threading.Lock()

    def warm_snapshot(snapshot: str) -> None:
        subjects = [path.rsplit("/", 1)[1].removesuffix(".pack") for path in fs.ls(f"{root}/{snapshot}/detail")]
        with concurrent.futures.ThreadPoolExecutor(WARM_THREADS) as pool:
            for subject, outcome in zip(subjects, pool.map(lambda s: subject_tasks(snapshot, s), subjects), strict=True):
                logger.info("warmed %s/%s: %d tasks", snapshot, subject, len(outcome))

    def warm_in_background(snapshot: str) -> None:
        try:
            warm_snapshot(snapshot)
        except Exception:
            # Warming is an optimisation: a failure leaves each shard to load on its first request.
            logger.warning("warming %s failed", snapshot, exc_info=True)
            with warm_lock:
                warmed.discard(snapshot)

    @api.get("/warm")
    def warm(snap: str = Query(...)) -> Response:
        snapshot = checked_snapshot(snap)
        with warm_lock:
            started = snapshot not in warmed
            warmed.add(snapshot)
        if started:
            threading.Thread(target=warm_in_background, args=(snapshot,), daemon=True).start()
        return Response(status_code=202 if started else 204, headers={"Cache-Control": "no-store"})

    return api
