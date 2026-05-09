# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-region swarm worker for SWE-ZERO synthesis.

Forked from `run_swe_zero_swarm.py`. Two operating modes:

1. **Native mode** (``--claim-region`` matches detected region, the default):
   Worker claims shards in its own partition via the canonical
   ``shard_idx % 4 == region_partition`` mapping. Reads and writes go to the
   region's own bucket ``gs://marin-{region}/experiments/swe_zero_100b/per_pr/``.
   No cross-region traffic; identical to the original single-region behaviour.

2. **Imported mode** (``--claim-region`` differs from detected region):
   Worker claims shards belonging to ``claim_region``'s partition (e.g., a
   us-east5 worker can absorb us-central1 P3 work). The claim file and
   per-shard state (``rollouts.json``, ``rollouts_resume.json``,
   ``sampling_plan.json``) live in the **source** region's bucket; the worker
   reads them cross-region once at iter start and writes new rollouts to its
   own bucket under ``imported/shard_NNN_of_1260/``. On clean iter completion
   the local ``rollouts_resume.json`` is copied back to the source bucket
   (one-time cross-region write of ~250-400 MB), and ``_done`` is marked in
   the source bucket. This lets idle compute in fully-done regions absorb
   work from saturated regions while keeping the source region's bucket as
   the canonical store for downstream aggregation.

Output layout, native mode:
    gs://marin-{region}/experiments/swe_zero_100b/per_pr/shard_NNN_of_1260/

Output layout, imported mode (compute region's local writes):
    gs://marin-{compute}/experiments/swe_zero_100b/imported/shard_NNN_of_1260/
After iter completion the contents land back in:
    gs://marin-{claim}/experiments/swe_zero_100b/per_pr/shard_NNN_of_1260/

Designed to be invoked by ``launch_adaptive_swe_zero_multiregion.py``, which
sets ``IRIS_WORKER_REGION`` indirectly via the iris ``Constraint(REGION, EQ, ...)``
attribute and an EnvironmentSpec entry. The worker reads ``IRIS_WORKER_REGION``
to determine its compute region; ``--claim-region`` (defaulting to the compute
region) selects which partition's shards to work on.
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

    # Fold in any imported_summary_{region}.json files. Each summary's
    # per_pr_counts reflects that region's local imported file, which (post-
    # seed) contains source's prior state + that region's new rollouts. So
    # imported count >= source count whenever imported is active. Take MAX
    # per PR to dedup the source overlap; the resulting count is at most the
    # union (under-counts when 2+ regions worked the same shard, which is
    # rare given exclusive claim files).
    try:
        import fsspec

        fs = fsspec.filesystem("gs")
        summary_glob = f"{_shard_path(output_root, batch_idx)}/imported_summary_*.json"
        for hit in fs.glob(summary_glob):
            try:
                with fsspec.open(f"gs://{hit}" if not hit.startswith("gs://") else hit, "rb") as f:
                    summary = json.load(f)
                for iid, n in (summary.get("per_pr_counts", {}) or {}).items():
                    counts[str(iid)] = max(counts.get(str(iid), 0), int(n))
            except Exception as ie:
                logger.warning("failed to read imported summary %s: %s", hit, ie)
    except Exception as e:
        logger.warning("imported summary scan failed for batch %d: %s", batch_idx, e)

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


def _imported_shard_path(local_output_root: str, batch_idx: int) -> str:
    """Local writes go under imported/ to avoid colliding with the compute
    region's natural per_pr/ partition (different shard indices)."""
    return f"{local_output_root}/imported/shard_{batch_idx:03d}_of_{TOTAL_PR_BATCHES:03d}"


def _run_inner(
    *,
    local_output_root: str,
    claim_output_root: str,
    is_imported: bool,
    batch_idx: int,
    tensor_parallel: int,
    worker_seed: int,
    iteration: int,
) -> int:
    # Source-bucket dir holds canonical state (rollouts.json, sampling_plan,
    # claim file). For native mode it's the same as compute-region; for
    # imported mode it's the source region's bucket.
    source_shard_dir = _shard_path(claim_output_root, batch_idx)
    # The inner worker auto-suffixes output_dir with shard_NNN_of_TTT/ when
    # --all-prs + --total-shards>1 are set, so we pass the parent dir here.
    if is_imported:
        # Local writes to imported/ to avoid colliding with the compute
        # region's natural per_pr/ partition (different shard index space).
        inner_output_dir = f"{local_output_root}/imported"
        # Seed local imported/rollouts_resume.json with source's accumulated
        # work so the inner's prior_dicts includes source state. Without this,
        # the migration step at iter end would overwrite source's resume with
        # only the new rollouts — losing source's prior work.
        _seed_imported_resume(local_output_root, claim_output_root, batch_idx)
    else:
        inner_output_dir = f"{local_output_root}/per_pr"

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
        inner_output_dir,
        "--n-rollouts",
        "100",
        "--max-num-seqs",
        # Scale concurrent vLLM batch with chip count: 256 at TP=4 (current
        # baseline) up to 4096 at TP=64. Mini-coder-1.7b's KV cache footprint
        # is small at 32K context, so larger slices have headroom for a wider
        # batch — needed to keep big slices saturated.
        str(max(256, 64 * tensor_parallel)),
        "--max-model-len",
        "32768",
        "--max-total-tokens",
        "32768",
        "--tensor-parallel-size",
        str(tensor_parallel),
        "--concurrency",
        # Rollout-loop concurrency tracks max_num_seqs / 4 — each in-flight
        # rollout uses ~4 simultaneous vLLM seqs across its multi-turn loop.
        str(max(64, 16 * tensor_parallel)),
        "--disable-shard-lease",
    ]
    # Resume-from selection.
    #   Native mode: point at source's rollouts.json (legacy behaviour). The
    #     inner's auto-resume + line 517 merge handles source rollouts_resume.
    #   Imported mode: point at the LOCAL seeded file. Cross-region reads
    #     already happened in _seed_imported_resume; the inner reads
    #     existing_counts from this same path it'll write to (the line 518
    #     `resume_path != resume_from` guard skips double-counting).
    if is_imported:
        seeded_resume = f"{_imported_shard_path(local_output_root, batch_idx)}/rollouts_resume.json"
        # If seed found nothing, the file is absent — that's fine, inner will
        # treat the shard as fresh and write to rollouts.json. Migration step
        # then has nothing to copy and the post-iter completeness check sees
        # only the new rollouts on source side. (This is acceptable because
        # truly-fresh shards have no prior work to preserve.)
        import fsspec

        fs = fsspec.filesystem("gs")
        if fs.exists(seeded_resume):
            cmd.extend(["--resume-from", seeded_resume])
    else:
        src_resume = f"{source_shard_dir}/rollouts_resume.json"
        src_rollouts = f"{source_shard_dir}/rollouts.json"
        if _gcs_blob(src_resume).exists():
            cmd.extend(["--resume-from", src_resume])
        elif _gcs_blob(src_rollouts).exists():
            cmd.extend(["--resume-from", src_rollouts])
    logger.info(
        "running inner worker: shard %d/%d (mode=%s) source=%s output=%s",
        batch_idx,
        TOTAL_PR_BATCHES,
        "imported" if is_imported else "native",
        source_shard_dir,
        inner_output_dir,
    )
    log_path = f"{source_shard_dir}/_swarm_worker_{worker_seed:04d}_iter_{iteration:03d}.log"
    inner_env = {**os.environ}
    cache_root = os.environ.get("MARIN_HF_CACHE_ROOT")
    if cache_root:
        # Prefetch lays out files at {cache_root}/models--<repo>/, which is the
        # raw HF Hub cache layout. HF_HOME would make huggingface_hub look at
        # {cache_root}/hub/models--<repo>/ (one level too deep), so use
        # HUGGINGFACE_HUB_CACHE which points directly at the cache dir.
        inner_env["HUGGINGFACE_HUB_CACHE"] = cache_root
        inner_env["HF_HOME"] = cache_root
        inner_env["HF_HUB_OFFLINE"] = "1"
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


def _seed_imported_resume(
    local_output_root: str,
    claim_output_root: str,
    batch_idx: int,
) -> None:
    """Pre-populate local imported/shard_X/rollouts_resume.json with source's
    accumulated rollouts so the inner worker's prior_dicts includes source
    state. Without this, the inner reads its own (empty) local resume file
    as prior_dicts and writes only newly-generated rollouts back, which the
    later migration step would overwrite source's resume with — losing all
    source-region prior work. One-time cross-region read of ~250-400 MB per
    claim. No-op if source has nothing yet (fresh shard).
    """
    src_resume = f"{_shard_path(claim_output_root, batch_idx)}/rollouts_resume.json"
    src_rollouts = f"{_shard_path(claim_output_root, batch_idx)}/rollouts.json"
    dst = f"{_imported_shard_path(local_output_root, batch_idx)}/rollouts_resume.json"
    try:
        import fsspec

        fs = fsspec.filesystem("gs")
        # Preserve cross-iter accumulation: if local imported already has
        # content from a prior iter (same compute region), keep it. Without
        # this guard, every re-claim overwrites iter-N's hard-won rollouts
        # with source's stale state — fresh shards with no source rollouts
        # could never accumulate to 100/PR across multiple iters.
        if fs.exists(dst):
            dst_size = int(fs.info(dst).get("size", 0))
            if dst_size > 2:  # `[]` is 2 bytes; anything bigger has work
                logger.info(
                    "skipping seed for batch %d: local imported has %.1f MB already",
                    batch_idx,
                    dst_size / 1e6,
                )
                return
        # Prefer rollouts_resume.json (cumulative); fall back to rollouts.json
        # (early-state shards never resumed). If neither exists, write an
        # empty array so the inner still sees --resume-from existing and uses
        # output_filename=rollouts_resume.json (which the summary step reads).
        if fs.exists(src_resume):
            src = src_resume
        elif fs.exists(src_rollouts):
            src = src_rollouts
        else:
            with fsspec.open(dst, "wb") as w:
                w.write(b"[]")
            logger.info("seeded imported resume with [] for fresh batch %d", batch_idx)
            return
        size = int(fs.info(src).get("size", 0))
        with fsspec.open(src, "rb") as r, fsspec.open(dst, "wb") as w:
            while True:
                chunk = r.read(8 * 1024 * 1024)
                if not chunk:
                    break
                w.write(chunk)
        logger.info("seeded imported resume %.1f MB from %s -> %s", size / 1e6, src, dst)
    except Exception as e:
        logger.error("seed-imported failed for batch %d: %s", batch_idx, e)


def _imported_summary_path(claim_output_root: str, batch_idx: int, compute_region: str) -> str:
    """Per-(shard, compute-region) summary file in source bucket. Tiny (~2 KB)
    cross-region write that gives source-region workers + audit visibility
    into imported progress without copying full rollout data."""
    return f"{_shard_path(claim_output_root, batch_idx)}/imported_summary_{compute_region}.json"


def _write_imported_summary(
    local_output_root: str,
    claim_output_root: str,
    compute_region: str,
    batch_idx: int,
) -> bool:
    """Write a per-PR count summary of this region's local imported file to
    source bucket. ~2 KB cross-region write; replaces this region's prior
    summary (idempotent on re-runs).

    Called on rc=0 iter exit in imported mode. The local imported file
    contains seeded source state + newly generated rollouts, so the count
    we write is the cumulative rollout count visible from this region.
    `_shard_is_complete` takes max(source, any_imported_summary) per PR —
    correctly handling the source-overlap-with-seed case (imported >= source).

    Replaces the prior `_migrate_imported_to_source` design (which did a
    250-400 MB cross-region write per iter). Final rollout data stays in
    gs://marin-{compute}/imported/shard_X/ and is picked up by the round-4
    aggregator's content-hash dedup at upload time.
    """
    src = f"{_imported_shard_path(local_output_root, batch_idx)}/rollouts_resume.json"
    dst = _imported_summary_path(claim_output_root, batch_idx, compute_region)
    try:
        import fsspec

        fs = fsspec.filesystem("gs")
        if not fs.exists(src):
            logger.warning("local imported file missing, skipping summary: %s", src)
            return False
        with fsspec.open(src, "rb") as f:
            rollouts = json.load(f)
        counts: dict[str, int] = {}
        if isinstance(rollouts, list):
            for r in rollouts:
                if isinstance(r, dict):
                    iid = r.get("instance_id")
                    if iid:
                        counts[str(iid)] = counts.get(str(iid), 0) + 1
        summary = {
            "compute_region": compute_region,
            "shard_idx": batch_idx,
            "ts": time.time(),
            "imported_local_path": f"{_imported_shard_path(local_output_root, batch_idx)}/",
            "per_pr_counts": counts,
        }
        _gcs_blob(dst).upload_from_string(json.dumps(summary))
        n_at = sum(1 for v in counts.values() if v >= TARGET_ROLLOUTS_PER_PR)
        logger.info(
            "wrote imported summary for batch %d (%d PRs, %d at >=%d) -> %s",
            batch_idx,
            len(counts),
            n_at,
            TARGET_ROLLOUTS_PER_PR,
            dst,
        )
        return True
    except Exception as e:
        logger.error("write-imported-summary failed for batch %d: %s", batch_idx, e)
        return False


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
        "--claim-region",
        default=None,
        help="Source region whose partition we claim work from (default: same as compute "
        "region, i.e., native mode). When different from the detected compute region, the "
        "worker runs in 'imported mode': claim file and shard state read cross-region from "
        "the source bucket, new rollouts written to the compute region's bucket under "
        "imported/, and the resume file copied back to source on iter completion.",
    )
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
    local_output_root = _output_root_for_region(region)
    claim_region = args.claim_region or region
    if claim_region not in PINNED_REGIONS:
        raise SystemExit(f"--claim-region {claim_region!r} not in {PINNED_REGIONS!r}")
    claim_output_root = _output_root_for_region(claim_region)
    is_imported = claim_region != region
    partition_mod, partition_idx = _partition_for_region(claim_region)
    n_claim_shards = sum(1 for i in range(TOTAL_PR_BATCHES) if i % partition_mod == partition_idx)
    logger.info(
        "MULTI-REGION SWARM WORKER compute=%s claim=%s mode=%s claim_partition=(%d/%d, %d shards)",
        region,
        claim_region,
        "imported" if is_imported else "native",
        partition_idx,
        partition_mod,
        n_claim_shards,
    )

    _assert_zone_in_region(region)
    _assert_bucket_pinned(local_output_root, region)
    # Native mode: legacy-clean own per_pr/. Imported mode: skip — we don't own
    # the source bucket and shouldn't be deleting its files.
    if not is_imported:
        _purge_legacy_locks(local_output_root)
    _prefetch_model_from_gcs(region)
    _prefetch_hf_dataset("nebius/SWE-rebench-V2-PRs")

    iterations = 0
    idle_cycles = 0
    while iterations < args.max_iterations:
        batch_idx = _claim_unclaimed_batch(
            claim_output_root,
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
                "no unclaimed shard in claim-partition %d/%d (idle cycle %d/%d) - sleep %ds",
                partition_idx,
                partition_mod,
                idle_cycles,
                args.idle_max_cycles,
                args.idle_sleep,
            )
            if idle_cycles >= args.idle_max_cycles:
                logger.info(
                    "claim-partition saturated for compute=%s claim=%s - exiting",
                    region,
                    claim_region,
                )
                return 0
            time.sleep(args.idle_sleep)
            iterations += 1
            continue

        idle_cycles = 0
        stop = threading.Event()
        hb = threading.Thread(
            target=_heartbeat,
            args=(stop, claim_output_root, batch_idx, args.worker_seed, region),
            daemon=True,
        )
        hb.start()
        try:
            rc = _run_inner(
                local_output_root=local_output_root,
                claim_output_root=claim_output_root,
                is_imported=is_imported,
                batch_idx=batch_idx,
                tensor_parallel=args.tensor_parallel,
                worker_seed=args.worker_seed,
                iteration=iterations,
            )
            if rc == 0:
                # Imported mode: write a tiny per-PR summary JSON to source
                # bucket (~2 KB cross-region write). _shard_is_complete reads
                # all such summaries and merges via MAX so source-region
                # workers + audit see this region's contributions without a
                # full 250-400 MB migration. Final rollout data stays in
                # gs://marin-{compute}/imported/shard_X/ and is folded in by
                # the round-4 aggregator's content-hash dedup at upload time.
                if is_imported:
                    _write_imported_summary(local_output_root, claim_output_root, region, batch_idx)
                # rc=0 means the inner exited cleanly, but that's not enough.
                # Inner can exit clean with partial rollouts (e.g., only a
                # subset of PRs reached n-rollouts; others had every rollout
                # filtered as an error). Verify true per-PR completion before
                # sealing the shard; otherwise release the claim so a future
                # worker can top it up via --resume-from.
                complete, msg = _shard_is_complete(claim_output_root, batch_idx)
                if complete:
                    try:
                        _gcs_blob(f"{_shard_path(claim_output_root, batch_idx)}/{DONE_FILENAME}").upload_from_string(
                            json.dumps(
                                {
                                    "completed_at": time.time(),
                                    "worker_seed": args.worker_seed,
                                    "compute_region": region,
                                    "claim_region": claim_region,
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
                _release_claim(claim_output_root, batch_idx)
            else:
                logger.warning("inner worker rc=%d for batch %d - releasing claim for retry", rc, batch_idx)
                _release_claim(claim_output_root, batch_idx)
        finally:
            stop.set()
            hb.join(timeout=5)

        iterations += 1

    logger.info("max-iterations reached (%d), exiting", args.max_iterations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
