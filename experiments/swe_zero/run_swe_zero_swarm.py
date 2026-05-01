# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Swarm worker for SWE-ZERO 140B synthesis.

Loops claiming unclaimed PR-batches from a 1260-way partition (~94 PRs each),
running the existing run_swe_zero_multilang.py rollout logic for each batch.

Atomic claim is held at the SWARM level via `_swarm_claim.json` files using
google.cloud.storage's `if_generation_match=0` precondition. The inner worker
is invoked with `--disable-shard-lease` so it doesn't fight our outer claim;
this is the same pattern the legacy `monitor_140b_pipeline.py` uses (it
disables the inner lease because the batch structure prevents collisions
externally).

Hard region invariant: fails fast if the host VM is not in us-east5-* zone or
if the configured output_root is not in gs://marin-us-east5/. Cross-region
transfer cost a $1.5K/day incident — these checks are non-negotiable.

Output layout: gs://marin-us-east5/experiments/swe_zero_100b/per_pr/shard_NNN_of_1260/
(distinct from the legacy shard_NNN_of_126/ outputs so existing data is preserved.)

Designed to be invoked by launch_adaptive_swe_zero.py.
"""

import argparse
import datetime as dt
import json
import logging
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request

logger = logging.getLogger(__name__)

# 10x finer than the legacy 126-shard partition — ~94 PRs per batch instead of ~944.
# Reduces work-loss on worker death from ~hours to ~minutes.
TOTAL_PR_BATCHES = 1260
PINNED_REGION = "us-east5"
PINNED_BUCKET_PREFIX = "gs://marin-us-east5/"
LEGACY_OUTPUT_ROOT = "gs://marin-us-east5/experiments/swe_zero_100b"

# Inner worker — existing rollout pipeline. Untouched here; we just call it.
INNER_WORKER = "experiments/swe_zero/run_swe_zero_multilang.py"

# Swarm-level claim freshness window. A claim is considered stale (and may be
# stolen by another worker) if its blob mtime is older than this. Inner rollouts
# can take hours, so a heartbeat thread refreshes the claim every CLAIM_REFRESH_SEC.
CLAIM_STALE_SECONDS = 4 * 3600  # 4 hours
CLAIM_REFRESH_SEC = 600  # 10 minutes
CLAIM_FILENAME = "_swarm_claim.json"
DONE_FILENAME = "_done"


def _gcs_blob(path: str):
    """Return a google.cloud.storage Blob for a `gs://bucket/key` path."""
    from google.cloud import storage

    assert path.startswith("gs://"), path
    bucket_name, _, key = path[len("gs://") :].partition("/")
    client = storage.Client()
    return client.bucket(bucket_name).blob(key)


def _claim_path(output_root: str, batch_idx: int) -> str:
    return f"{_shard_path(output_root, batch_idx)}/{CLAIM_FILENAME}"


def _claim_payload(worker_seed: int) -> str:
    return json.dumps(
        {
            "worker_seed": worker_seed,
            "hostname": os.environ.get("HOSTNAME", ""),
            "iris_task_id": os.environ.get("IRIS_TASK_ID", ""),
            "pid": os.getpid(),
            "updated_at": time.time(),
        }
    )


def _try_acquire_claim(output_root: str, batch_idx: int, worker_seed: int) -> bool:
    """Atomically create the claim blob; returns True if we won.

    Uses GCS `if_generation_match=0` precondition: only the first writer
    succeeds, others get 412 PreconditionFailed (raised as PreconditionFailed).
    If a stale claim exists (mtime older than CLAIM_STALE_SECONDS), we delete
    it and retry once.
    """
    from google.api_core.exceptions import PreconditionFailed

    path = _claim_path(output_root, batch_idx)
    blob = _gcs_blob(path)
    try:
        blob.upload_from_string(_claim_payload(worker_seed), if_generation_match=0)
        return True
    except PreconditionFailed:
        # Someone holds it. Check freshness — if stale, steal.
        try:
            blob.reload()
            updated = blob.updated  # datetime in UTC
            age = (dt.datetime.now(dt.timezone.utc) - updated).total_seconds()
        except Exception as e:
            logger.warning("could not read claim mtime for batch %d: %s", batch_idx, e)
            return False
        if age <= CLAIM_STALE_SECONDS:
            return False
        logger.info("stealing stale claim for batch %d (age=%.0fs)", batch_idx, age)
        try:
            blob.delete()
        except Exception:
            return False
        try:
            blob.upload_from_string(_claim_payload(worker_seed), if_generation_match=0)
            return True
        except PreconditionFailed:
            return False


def _refresh_claim(output_root: str, batch_idx: int, worker_seed: int) -> None:
    """Overwrite the claim blob to bump its mtime. Best-effort, swallows errors."""
    try:
        _gcs_blob(_claim_path(output_root, batch_idx)).upload_from_string(_claim_payload(worker_seed))
    except Exception as e:
        logger.warning("claim refresh failed for batch %d: %s", batch_idx, e)


def _release_claim(output_root: str, batch_idx: int) -> None:
    try:
        _gcs_blob(_claim_path(output_root, batch_idx)).delete()
    except Exception:
        pass


def _is_done(output_root: str, batch_idx: int) -> bool:
    """Check whether the shard's _done marker exists."""
    path = f"{_shard_path(output_root, batch_idx)}/{DONE_FILENAME}"
    try:
        return _gcs_blob(path).exists()
    except Exception:
        return False


def _purge_legacy_locks(output_root: str) -> None:
    """One-time cleanup of `_active.lock.json` files from previous swarm-debug runs.

    The inner worker's ShardLease used these; we now disable it via
    --disable-shard-lease. Leftover lock files don't block our SwarmClaim but
    keep them around can confuse readers.
    """
    import fsspec

    fs, _ = fsspec.core.url_to_fs(f"{output_root}/per_pr/")
    pattern = f"{output_root}/per_pr/shard_*_of_*/_active.lock.json"
    try:
        matches = fs.glob(pattern)
    except Exception:
        return
    if not matches:
        return
    logger.info("purging %d legacy _active.lock.json files...", len(matches))
    for path in matches:
        try:
            fs.rm(path)
        except Exception:
            pass


def _prefetch_hf_dataset(dataset_id: str) -> None:
    """Populate the HF dataset cache once per worker pod.

    Without this, each inner subprocess invocation calls load_dataset() which
    hits HF's API. After this prefetch we set HF_HUB_OFFLINE=1 in subprocess
    env so further iterations use the local cache.
    """
    logger.info("prefetching HF dataset %s into local cache...", dataset_id)
    from datasets import load_dataset

    load_dataset(dataset_id, split="train")
    logger.info("prefetch complete")


def _assert_us_east5_zone() -> None:
    try:
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/zone",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            zone = resp.read().decode().strip()
    except Exception:
        logger.warning("Could not query GCE metadata for zone; skipping zone assert")
        return
    zone_short = zone.rsplit("/", 1)[-1]
    if not zone_short.startswith(f"{PINNED_REGION}-"):
        raise SystemExit(f"REGION INVARIANT VIOLATED: VM in {zone_short!r}, expected {PINNED_REGION}-*")
    logger.info("zone check passed: %s", zone_short)


def _assert_bucket_pinned(output_root: str) -> None:
    if not output_root.startswith(PINNED_BUCKET_PREFIX):
        raise SystemExit(f"REGION INVARIANT VIOLATED: output_root={output_root!r} not under {PINNED_BUCKET_PREFIX}")


def _shard_path(output_root: str, batch_idx: int) -> str:
    return f"{output_root}/per_pr/shard_{batch_idx:03d}_of_{TOTAL_PR_BATCHES:03d}"


def _claim_unclaimed_batch(output_root: str, worker_seed: int, iteration: int) -> int | None:
    """Walk a shuffled index list, atomically claim the first unclaimed shard.

    Returns the batch_idx we acquired, or None if every shard is either done or
    has a fresh claim.
    """
    rng = random.Random(worker_seed * 100_003 + iteration)
    indices = list(range(TOTAL_PR_BATCHES))
    rng.shuffle(indices)
    for idx in indices:
        if _is_done(output_root, idx):
            continue
        if _try_acquire_claim(output_root, idx, worker_seed):
            logger.info("acquired swarm claim for batch %d", idx)
            return idx
    return None


def _heartbeat(stop_event: threading.Event, output_root: str, batch_idx: int, worker_seed: int) -> None:
    """Refresh the claim every CLAIM_REFRESH_SEC until stop_event is set."""
    while not stop_event.wait(CLAIM_REFRESH_SEC):
        _refresh_claim(output_root, batch_idx, worker_seed)
        logger.info("heartbeat: refreshed claim for batch %d", batch_idx)


def _run_inner(output_root: str, batch_idx: int, tensor_parallel: int, worker_seed: int, iteration: int) -> int:
    shard_dir = _shard_path(output_root, batch_idx)
    cmd = [
        sys.executable,
        INNER_WORKER,
        "--local",
        "--all-prs",
        "--dataset",
        "nebius/SWE-rebench-V2-PRs",
        "--shard-index",
        str(batch_idx),
        "--total-shards",
        str(TOTAL_PR_BATCHES),
        "--output_dir",
        f"{output_root}/per_pr",
        "--n-rollouts",
        "100",
        "--max-num-seqs",
        "256",
        "--max-model-len",
        "32768",
        "--max-total-tokens",
        "32768",
        "--tensor-parallel-size",
        str(tensor_parallel),
        "--concurrency",
        "64",
        # We hold the claim at the swarm level — disable the inner ShardLease.
        "--disable-shard-lease",
    ]
    # Only pass --resume-from when the file actually exists; the inner worker
    # raises FileNotFoundError when given a path that doesn't resolve, even on
    # a brand-new shard with nothing to resume from.
    resume_from = f"{shard_dir}/rollouts.json"
    if _gcs_blob(resume_from).exists():
        cmd.extend(["--resume-from", resume_from])
    logger.info("running inner worker: shard %d/%d -> %s", batch_idx, TOTAL_PR_BATCHES, shard_dir)
    log_path = f"{shard_dir}/_swarm_worker_{worker_seed:04d}_iter_{iteration:03d}.log"
    # HF_HUB_OFFLINE=1 was too aggressive — it also blocks vLLM's model snapshot
    # download. With the resume-from fix preventing rapid claim-fail cycling, HF
    # API call rate stays low enough to stay under the 1000/5min quota even with
    # many workers (each call is ~1/shard rather than 100/min during retry loops).
    inner_env = {**os.environ}
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=inner_env)
    log_blob = (
        f"=== cmd ===\n{' '.join(cmd)}\n"
        f"=== rc={proc.returncode} ===\n"
        f"=== stderr (last 200 lines) ===\n"
        + "\n".join(proc.stderr.splitlines()[-200:])
        + "\n=== stdout (last 50 lines) ===\n"
        + "\n".join(proc.stdout.splitlines()[-50:])
    )
    try:
        import fsspec

        with fsspec.open(log_path, "w") as f:
            f.write(log_blob)
    except Exception as e:
        logger.error("failed to upload inner log to %s: %s", log_path, e)
    return proc.returncode


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=LEGACY_OUTPUT_ROOT)
    parser.add_argument("--worker-seed", type=int, default=0)
    parser.add_argument("--tensor-parallel", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--idle-sleep", type=int, default=120)
    parser.add_argument("--idle-max-cycles", type=int, default=10)
    args = parser.parse_args()

    _assert_us_east5_zone()
    _assert_bucket_pinned(args.output_root)
    _purge_legacy_locks(args.output_root)
    _prefetch_hf_dataset("nebius/SWE-rebench-V2-PRs")

    iterations = 0
    idle_cycles = 0
    while iterations < args.max_iterations:
        batch_idx = _claim_unclaimed_batch(args.output_root, args.worker_seed, iterations)
        if batch_idx is None:
            idle_cycles += 1
            if idle_cycles >= args.idle_max_cycles:
                logger.info("no claimable batches after %d cycles — swarm done; exiting", idle_cycles)
                return 0
            logger.info(
                "no claimable batches (cycle %d/%d); sleeping %ds",
                idle_cycles,
                args.idle_max_cycles,
                args.idle_sleep,
            )
            time.sleep(args.idle_sleep)
            continue

        idle_cycles = 0
        # Run inner with heartbeat thread refreshing the claim.
        stop_event = threading.Event()
        hb = threading.Thread(
            target=_heartbeat,
            args=(stop_event, args.output_root, batch_idx, args.worker_seed),
            daemon=True,
        )
        hb.start()
        try:
            rc = _run_inner(args.output_root, batch_idx, args.tensor_parallel, args.worker_seed, iterations)
        finally:
            stop_event.set()
            # Don't release the claim if we just produced a `_done` marker — leave the
            # claim in place as a tombstone (it'll expire). If inner failed without
            # writing _done, release so another worker can try.
            if not _is_done(args.output_root, batch_idx):
                _release_claim(args.output_root, batch_idx)
                logger.info("released claim for batch %d (inner rc=%d, no _done)", batch_idx, rc)
            else:
                logger.info("inner produced _done for batch %d; leaving claim as tombstone", batch_idx)

        if rc != 0:
            logger.warning("inner worker exited rc=%d for batch %d", rc, batch_idx)
        iterations += 1

    logger.info("hit max_iterations=%d; exiting", args.max_iterations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
