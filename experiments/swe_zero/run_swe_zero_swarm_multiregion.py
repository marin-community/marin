# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-region swarm worker for SWE-ZERO synthesis.

Forked from `run_swe_zero_swarm.py`. The single-region invariant is preserved
*per worker* - each worker fails fast if its host VM is not in its assigned
region, and reads/writes only `gs://marin-{region}/experiments/swe_zero_100b/`
(its own regional bucket). Cross-region transfer is still architecturally
impossible.

The 1260-shard partition is divided deterministically across regions:
``shard_idx % len(REGIONS) == REGIONS.index(my_region)``.

A worker only attempts to claim shards in its partition. The
`_swarm_claim.json` and `_done` markers live alongside the rollouts.json files
in the worker's regional bucket; workers in different regions can never see
each other's claims because they never look outside their own bucket.

Output layout (per region):
    gs://marin-{region}/experiments/swe_zero_100b/per_pr/shard_NNN_of_1260/

Designed to be invoked by ``launch_adaptive_swe_zero_multiregion.py``, which
sets ``IRIS_WORKER_REGION`` indirectly via the iris ``Constraint(REGION, EQ, ...)``
attribute and an EnvironmentSpec entry. The worker reads ``IRIS_WORKER_REGION``
to determine its assigned region; all bucket paths are derived from that.
"""

import argparse
import datetime as dt
import json
import logging
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.request

logger = logging.getLogger(__name__)

# Regions this multi-region pipeline writes to. Order matters - it determines
# the static shard-to-region partition. Adding a region requires running an
# audit and merge step at upload time.
PINNED_REGIONS = ["us-east5", "us-east1", "us-west4", "us-central1"]

TOTAL_PR_BATCHES = 1260
INNER_WORKER = "experiments/swe_zero/run_swe_zero_multilang.py"

CLAIM_STALE_SECONDS = 4 * 3600
CLAIM_REFRESH_SEC = 600
CLAIM_FILENAME = "_swarm_claim.json"
DONE_FILENAME = "_done"
# Target rollouts/PR for a shard to be considered complete. The inner worker
# is invoked with --n-rollouts=100 (see _run_inner). A shard is _done only
# when every PR in sampling_plan.json has at least this many rollouts in
# rollouts.json. Previously _done was written on any rc=0 from the inner,
# which marked partial-completion shards as done and prevented re-claim.
TARGET_ROLLOUTS_PER_PR = 100


def _detect_region() -> str:
    """Read the worker's region from iris-supplied env vars; fail closed if absent.

    Iris exposes ``IRIS_WORKER_REGION`` (see lib/iris/.../task_attempt.py).
    We refuse to run if it's missing or not in the pipeline's region whitelist
    - running in an unexpected region would mean writing to a same-region
    bucket that the upload script doesn't know about.
    """
    region = os.environ.get("IRIS_WORKER_REGION", "").strip()
    if not region:
        raise SystemExit(
            "REGION INVARIANT: IRIS_WORKER_REGION env var not set - "
            "this worker requires iris to surface the placement region."
        )
    if region not in PINNED_REGIONS:
        raise SystemExit(
            f"REGION INVARIANT: detected region {region!r} not in pipeline whitelist {PINNED_REGIONS!r}. "
            "Add it to PINNED_REGIONS *and* the upload merge script before re-running."
        )
    return region


def _output_root_for_region(region: str) -> str:
    return f"gs://marin-{region}/experiments/swe_zero_100b"


def _bucket_prefix_for_region(region: str) -> str:
    return f"gs://marin-{region}/"


def _partition_for_region(region: str) -> tuple[int, int]:
    """Static partition: (mod, idx). Worker claims shards where i % mod == idx."""
    return len(PINNED_REGIONS), PINNED_REGIONS.index(region)


def _gcs_blob(path: str):
    """google.cloud.storage Blob for a `gs://bucket/key` path."""
    from google.cloud import storage

    assert path.startswith("gs://"), path
    bucket_name, _, key = path[len("gs://") :].partition("/")
    client = storage.Client()
    return client.bucket(bucket_name).blob(key)


def _shard_path(output_root: str, batch_idx: int) -> str:
    return f"{output_root}/per_pr/shard_{batch_idx:03d}_of_{TOTAL_PR_BATCHES:03d}"


def _claim_path(output_root: str, batch_idx: int) -> str:
    return f"{_shard_path(output_root, batch_idx)}/{CLAIM_FILENAME}"


def _claim_payload(worker_seed: int, region: str) -> str:
    return json.dumps(
        {
            "worker_seed": worker_seed,
            "region": region,
            "hostname": os.environ.get("HOSTNAME", ""),
            "iris_task_id": os.environ.get("IRIS_TASK_ID", ""),
            "pid": os.getpid(),
            "updated_at": time.time(),
        }
    )


def _try_acquire_claim(output_root: str, batch_idx: int, worker_seed: int, region: str) -> bool:
    """Atomic claim via GCS if_generation_match=0; steal stale claims (>4 h)."""
    from google.api_core.exceptions import PreconditionFailed

    path = _claim_path(output_root, batch_idx)
    blob = _gcs_blob(path)
    try:
        blob.upload_from_string(_claim_payload(worker_seed, region), if_generation_match=0)
        return True
    except PreconditionFailed:
        pass

    try:
        blob.reload()
        age = (dt.datetime.now(dt.timezone.utc) - blob.updated).total_seconds()
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
        blob.upload_from_string(_claim_payload(worker_seed, region), if_generation_match=0)
        return True
    except PreconditionFailed:
        return False


def _refresh_claim(output_root: str, batch_idx: int, worker_seed: int, region: str) -> None:
    try:
        _gcs_blob(_claim_path(output_root, batch_idx)).upload_from_string(_claim_payload(worker_seed, region))
    except Exception as e:
        logger.warning("claim refresh failed for batch %d: %s", batch_idx, e)


def _release_claim(output_root: str, batch_idx: int) -> None:
    try:
        _gcs_blob(_claim_path(output_root, batch_idx)).delete()
    except Exception:
        pass


def _is_done(output_root: str, batch_idx: int) -> bool:
    path = f"{_shard_path(output_root, batch_idx)}/{DONE_FILENAME}"
    try:
        return _gcs_blob(path).exists()
    except Exception:
        return False


def _shard_is_complete(output_root: str, batch_idx: int) -> tuple[bool, str]:
    """Check if a shard has >=TARGET_ROLLOUTS_PER_PR for every PR in its
    sampling_plan.json. Returns (is_complete, summary_message).

    Used to gate the swarm-level _done marker: a shard should NOT be marked
    done just because the inner worker exited rc=0 - the inner can exit clean
    with partial rollouts when individual rollouts hit `_is_error_rollout`
    filters or when it processed only a subset of PRs before exiting. We
    require true per-PR completion before sealing the shard.
    """
    shard_dir = _shard_path(output_root, batch_idx)
    rollouts_path = f"{shard_dir}/rollouts.json"
    resume_path = f"{shard_dir}/rollouts_resume.json"
    plan_path = f"{shard_dir}/sampling_plan.json"
    try:
        plan_blob = _gcs_blob(plan_path)
        if not plan_blob.exists():
            return False, "no sampling_plan.json"
        plan = json.loads(plan_blob.download_as_bytes())
        # Read both rollouts.json and rollouts_resume.json. Inner worker writes
        # new rollouts to rollouts_resume.json (when --resume-from is set), so
        # checking rollouts.json alone misses the multi-region/post-migration
        # work entirely.
        rollouts_blob = _gcs_blob(rollouts_path)
        resume_blob = _gcs_blob(resume_path)
        rollouts = []
        if rollouts_blob.exists():
            rollouts.extend(json.loads(rollouts_blob.download_as_bytes()) or [])
        if resume_blob.exists():
            rollouts.extend(json.loads(resume_blob.download_as_bytes()) or [])
        if not rollouts:
            return False, "no rollouts.json or rollouts_resume.json"
    except Exception as e:
        return False, f"read error: {e}"

    expected_ids: set[str] = set()
    languages = plan.get("languages", {}) if isinstance(plan, dict) else {}
    for lang_data in languages.values():
        if isinstance(lang_data, dict):
            for iid in lang_data.get("instance_ids", []) or []:
                expected_ids.add(str(iid))
    if not expected_ids:
        return False, "sampling_plan.json has no instance_ids"

    counts: dict[str, int] = {}
    for r in rollouts:
        iid = r.get("instance_id") if isinstance(r, dict) else None
        if iid is not None:
            counts[str(iid)] = counts.get(str(iid), 0) + 1

    short = [iid for iid in expected_ids if counts.get(iid, 0) < TARGET_ROLLOUTS_PER_PR]
    total_expected = len(expected_ids)
    complete_prs = total_expected - len(short)
    msg = f"{complete_prs}/{total_expected} PRs at >={TARGET_ROLLOUTS_PER_PR} rollouts"
    return len(short) == 0, msg


def _purge_legacy_locks(output_root: str) -> None:
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
    """Prefetch an HF dataset into the same /tmp/hf-cache that holds the model.

    The inner worker subprocess runs with HF_HUB_OFFLINE=1 + HF_HOME=/tmp/hf-cache
    (set in _run_inner). Aligning the outer prefetch's cache to the same root
    ensures the inner can find the dataset locally without an HF round-trip.
    """
    cache_root = os.environ.get("MARIN_HF_CACHE_ROOT", "/tmp/hf-cache")
    os.environ["HF_HOME"] = cache_root  # propagates to load_dataset
    logger.info("prefetching HF dataset %s into local cache (HF_HOME=%s)...", dataset_id, cache_root)
    from datasets import load_dataset

    load_dataset(dataset_id, split="train")
    logger.info("prefetch complete")


def _prefetch_model_from_gcs(region: str) -> bool:
    """Copy the mini-coder-1.7b HF cache from regional GCS into /tmp/hf-cache.

    Populates the local cache only; does NOT set HF_HUB_OFFLINE in os.environ
    (that would block the outer worker's _prefetch_hf_dataset call to
    nebius/SWE-rebench-V2-PRs). The inner worker subprocess gets HF_HUB_OFFLINE=1
    + HF_HOME=/tmp/hf-cache via _run_inner's inner_env, scoping offline-mode
    to where it matters: the vLLM model load that triggers the 429 race.

    Returns True if the cache was successfully populated; False on any failure
    (inner worker falls back to online HF, which may 429 but launcher retries).
    """
    import fsspec

    src = f"gs://marin-{region}/hf-cache/models--ricdomolm--mini-coder-1.7b/"
    dst_root = "/tmp/hf-cache"
    dst = f"{dst_root}/models--ricdomolm--mini-coder-1.7b/"
    if os.path.exists(dst) and os.environ.get("MARIN_HF_CACHE_ROOT") == dst_root:
        logger.info("local hf-cache already populated at %s, skipping", dst)
        return True
    try:
        fs = fsspec.filesystem("gs")
        if not fs.exists(src):
            logger.warning("GCS hf-cache not found at %s; falling back to HF online", src)
            return False
        os.makedirs(dst, exist_ok=True)
        files = [f for f in fs.find(src) if not f.endswith("/")]
        logger.info("prefetching %d files from %s to %s", len(files), src, dst)
        prefix = src[:-1] if src.endswith("/") else src
        prefix_no_proto = prefix[len("gs://") :]
        for f in files:
            f_no_proto = f[len("gs://") :] if f.startswith("gs://") else f
            rel = (
                f_no_proto[len(prefix_no_proto) + 1 :]
                if f_no_proto.startswith(prefix_no_proto + "/")
                else os.path.basename(f)
            )
            local_path = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with fs.open(f, "rb") as src_f, open(local_path, "wb") as out:
                while True:
                    chunk = src_f.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
        # Tag the cache root for _run_inner to pass into subprocess env.
        os.environ["MARIN_HF_CACHE_ROOT"] = dst_root
        logger.info(
            "model prefetch complete (%d files, ~3.5 GB) at %s; inner workers will run with HF_HUB_OFFLINE=1",
            len(files),
            dst_root,
        )
        return True
    except Exception as e:
        logger.warning("model prefetch from GCS failed (%s); falling back to HF online", e)
        return False


def _assert_zone_in_region(region: str) -> None:
    """Verify the host VM's GCE zone is actually inside `region`. Cross-region writes are forbidden."""
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
    if not zone_short.startswith(f"{region}-"):
        raise SystemExit(
            f"REGION INVARIANT VIOLATED: VM in zone {zone_short!r}, expected {region}-* "
            f"(IRIS_WORKER_REGION reported {region!r})"
        )
    logger.info("zone check passed: %s in region %s", zone_short, region)


def _assert_bucket_pinned(output_root: str, region: str) -> None:
    expected = _bucket_prefix_for_region(region)
    if not output_root.startswith(expected):
        raise SystemExit(f"REGION INVARIANT VIOLATED: output_root={output_root!r} not under {expected}")


_AUDIT_PATH = "gs://marin-us-east5/datasets/swe-zero-stale-done-shards.json"
_MISSING_BY_SHARD: dict[int, int] | None = None


def _load_missing_counts() -> dict[int, int]:
    """Load per-shard `n_missing` from the audit JSON, once. Returns {} on failure."""
    global _MISSING_BY_SHARD
    if _MISSING_BY_SHARD is not None:
        return _MISSING_BY_SHARD
    import fsspec

    try:
        with fsspec.open(_AUDIT_PATH, "rb") as f:
            audit = json.load(f)
        result: dict[int, int] = {}
        for entry in audit.get("shards", []):
            result[int(entry["shard"])] = int(entry.get("n_missing", 0))
        _MISSING_BY_SHARD = result
        logger.info("Loaded missing-PR counts from %s for %d shards", _AUDIT_PATH, len(result))
    except Exception as e:
        logger.warning("Failed to load %s (%s) - falling back to random claim order", _AUDIT_PATH, e)
        _MISSING_BY_SHARD = {}
    return _MISSING_BY_SHARD


def _claim_unclaimed_batch(
    output_root: str,
    worker_seed: int,
    iteration: int,
    region: str,
    partition_mod: int,
    partition_idx: int,
    claim_order: str = "missing-first",
) -> int | None:
    """Walk THIS REGION's partition; claim the first free shard.

    claim_order:
      - "missing-first": prefer shards with the highest `n_missing` count (per
        the audit JSON). Tie-broken by random shuffle. Falls back to random if
        the audit JSON is unavailable.
      - "random": uniform shuffle (legacy behavior).
    """
    indices = [i for i in range(TOTAL_PR_BATCHES) if i % partition_mod == partition_idx]
    rng = random.Random(worker_seed * 100_003 + iteration)
    if claim_order == "missing-first":
        missing = _load_missing_counts()
        if missing:
            rng.shuffle(indices)  # break ties randomly
            indices.sort(key=lambda i: missing.get(i, 0), reverse=True)
        else:
            rng.shuffle(indices)  # fallback if audit JSON missing
    else:
        rng.shuffle(indices)
    for idx in indices:
        if _is_done(output_root, idx):
            continue
        if _try_acquire_claim(output_root, idx, worker_seed, region):
            logger.info(
                "acquired swarm claim for batch %d (region %s, n_missing=%d)",
                idx,
                region,
                _load_missing_counts().get(idx, -1),
            )
            return idx
    return None


def _heartbeat(stop_event, output_root, batch_idx, worker_seed, region):
    while not stop_event.wait(CLAIM_REFRESH_SEC):
        _refresh_claim(output_root, batch_idx, worker_seed, region)
        logger.info("heartbeat: refreshed claim for batch %d", batch_idx)


_METRICS_LINE_RE = re.compile(
    r"\[METRICS\] rollouts_produced=(?P<rollouts>\d+) completion_tokens=(?P<completion>\d+) "
    r"submission_rate=(?P<sub>[\d.]+) total_after=(?P<total_after>\d+)"
)


def _parse_inner_metrics(stderr: str) -> dict | None:
    """Find the [METRICS] line emitted by run_swe_zero_multilang.py at end of iter."""
    m = _METRICS_LINE_RE.search(stderr or "")
    if not m:
        return None
    return {
        "rollouts_produced": int(m.group("rollouts")),
        "completion_tokens": int(m.group("completion")),
        "submission_rate": float(m.group("sub")),
        "total_after": int(m.group("total_after")),
    }


def _emit_throughput_metric(
    output_root: str,
    region: str,
    batch_idx: int,
    worker_seed: int,
    iteration: int,
    inner_metrics: dict | None,
    duration_s: float,
    rc: int,
) -> None:
    """Write per-iter throughput metric to gs://marin-{region}/swe_zero_100b/throughput/.

    Uses the [METRICS] ground-truth line emitted by run_swe_zero_multilang.py.
    Falls back to all-zeros when the inner crashed before emitting (rc != 0).

    scripts/throughput_stats.py aggregates these files to compute per-region
    tokens/sec over arbitrary windows. One file per (worker, iter); ~250 B JSON.
    """
    if inner_metrics:
        rollouts_produced = inner_metrics["rollouts_produced"]
        completion_tokens = inner_metrics["completion_tokens"]
        submission_rate = inner_metrics["submission_rate"]
        total_after = inner_metrics["total_after"]
        # Trajectory tokens (what SFT trains on) ~ 9,050/rollout per round-3 stats.
        trajectory_tokens_estimate = rollouts_produced * 9050
        source = "inner_metrics"
    else:
        rollouts_produced = completion_tokens = total_after = 0
        submission_rate = 0.0
        trajectory_tokens_estimate = 0
        source = "missing_inner_metrics"
    metric = {
        "ts": time.time(),
        "region": region,
        "shard_idx": batch_idx,
        "worker_id": worker_seed,
        "iter": iteration,
        "rollouts_produced": rollouts_produced,
        "completion_tokens": completion_tokens,
        "trajectory_tokens_estimate": trajectory_tokens_estimate,
        "submission_rate": submission_rate,
        "total_after": total_after,
        "duration_s": round(duration_s, 1),
        "rc": rc,
        "source": source,
    }
    metric_path = f"{output_root}/throughput/{worker_seed:04d}_{iteration:03d}.json"
    try:
        _gcs_blob(metric_path).upload_from_string(json.dumps(metric))
    except Exception as e:
        logger.warning("failed to write throughput metric to %s: %s", metric_path, e)


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
        "--disable-shard-lease",
    ]
    resume_from = f"{shard_dir}/rollouts.json"
    if _gcs_blob(resume_from).exists():
        cmd.extend(["--resume-from", resume_from])
    logger.info("running inner worker: shard %d/%d -> %s", batch_idx, TOTAL_PR_BATCHES, shard_dir)
    log_path = f"{shard_dir}/_swarm_worker_{worker_seed:04d}_iter_{iteration:03d}.log"
    inner_env = {**os.environ}
    # Scope HF_HUB_OFFLINE=1 to the inner subprocess only (where vLLM loads
    # the model). MARIN_HF_CACHE_ROOT is set by _prefetch_model_from_gcs after
    # successfully populating the local cache; if absent, fall through to
    # online HF (with retry-loop risk on 429).
    cache_root = os.environ.get("MARIN_HF_CACHE_ROOT")
    if cache_root:
        inner_env["HF_HOME"] = cache_root
        inner_env["HF_HUB_OFFLINE"] = "1"
    inner_start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=inner_env)
    inner_duration = time.time() - inner_start
    region = os.environ.get("IRIS_WORKER_REGION", "unknown")
    inner_metrics = _parse_inner_metrics(proc.stderr)
    _emit_throughput_metric(
        output_root,
        region,
        batch_idx,
        worker_seed,
        iteration,
        inner_metrics,
        inner_duration,
        proc.returncode,
    )
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
    # Region is detected from env, NOT a CLI flag - reduces config drift between
    # what the iris constraint pinned and what the worker thinks it's writing.
    parser.add_argument("--worker-seed", type=int, default=0)
    parser.add_argument("--tensor-parallel", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--idle-sleep", type=int, default=120)
    parser.add_argument("--idle-max-cycles", type=int, default=10)
    parser.add_argument(
        "--claim-order",
        choices=["missing-first", "random"],
        default="missing-first",
        help="Shard claim order. 'missing-first' prioritizes shards with the highest "
        "n_missing PR count (per gs://marin-us-east5/datasets/swe-zero-stale-done-shards.json), "
        "falling back to random if the audit JSON is unavailable. 'random' uses uniform shuffle.",
    )
    args = parser.parse_args()

    region = _detect_region()
    output_root = _output_root_for_region(region)
    partition_mod, partition_idx = _partition_for_region(region)
    n_my_shards = sum(1 for i in range(TOTAL_PR_BATCHES) if i % partition_mod == partition_idx)
    logger.info(
        "MULTI-REGION SWARM WORKER region=%s output_root=%s partition=(%d/%d, %d shards)",
        region,
        output_root,
        partition_idx,
        partition_mod,
        n_my_shards,
    )

    _assert_zone_in_region(region)
    _assert_bucket_pinned(output_root, region)
    _purge_legacy_locks(output_root)
    _prefetch_model_from_gcs(region)
    _prefetch_hf_dataset("nebius/SWE-rebench-V2-PRs")

    iterations = 0
    idle_cycles = 0
    while iterations < args.max_iterations:
        batch_idx = _claim_unclaimed_batch(
            output_root,
            args.worker_seed,
            iterations,
            region,
            partition_mod,
            partition_idx,
            claim_order=args.claim_order,
        )
        if batch_idx is None:
            idle_cycles += 1
            logger.info(
                "no unclaimed shard in partition %d/%d (idle cycle %d/%d) - sleep %ds",
                partition_idx,
                partition_mod,
                idle_cycles,
                args.idle_max_cycles,
                args.idle_sleep,
            )
            if idle_cycles >= args.idle_max_cycles:
                logger.info("partition saturated for region %s - exiting", region)
                return 0
            time.sleep(args.idle_sleep)
            iterations += 1
            continue

        idle_cycles = 0
        stop = threading.Event()
        hb = threading.Thread(
            target=_heartbeat,
            args=(stop, output_root, batch_idx, args.worker_seed, region),
            daemon=True,
        )
        hb.start()
        try:
            rc = _run_inner(output_root, batch_idx, args.tensor_parallel, args.worker_seed, iterations)
            if rc == 0:
                # rc=0 means the inner exited cleanly, but that's not enough.
                # Inner can exit clean with partial rollouts (e.g., only a
                # subset of PRs reached n-rollouts; others had every rollout
                # filtered as an error). Verify true per-PR completion before
                # sealing the shard; otherwise release the claim so a future
                # worker can top it up via --resume-from.
                complete, msg = _shard_is_complete(output_root, batch_idx)
                if complete:
                    try:
                        _gcs_blob(f"{_shard_path(output_root, batch_idx)}/{DONE_FILENAME}").upload_from_string(
                            json.dumps(
                                {
                                    "completed_at": time.time(),
                                    "worker_seed": args.worker_seed,
                                    "region": region,
                                    "iteration": iterations,
                                    "completion_check": msg,
                                }
                            )
                        )
                        logger.info("marked _done for batch %d (%s)", batch_idx, msg)
                    except Exception as e:
                        logger.error("failed to write _done for batch %d: %s", batch_idx, e)
                else:
                    logger.info(
                        "inner rc=0 but shard %d incomplete (%s) - releasing claim for top-up",
                        batch_idx,
                        msg,
                    )
                _release_claim(output_root, batch_idx)
            else:
                logger.warning("inner worker rc=%d for batch %d - releasing claim for retry", rc, batch_idx)
                _release_claim(output_root, batch_idx)
        finally:
            stop.set()
            hb.join(timeout=5)

        iterations += 1

    logger.info("max-iterations reached (%d), exiting", args.max_iterations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
