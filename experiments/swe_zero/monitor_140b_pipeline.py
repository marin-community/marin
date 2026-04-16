# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor and conservatively relaunch the 140B-token SWE-ZERO pipeline.

This replaces the ad hoc `/tmp/monitor_100b*.sh` scripts with a safer policy:

1. Discover the newest job for each batch from Iris.
2. Inspect per-batch shard freshness in GCS.
3. Optionally count valid rollouts exactly for incomplete/stalled batches.
4. Relaunch only the specific batches that are both incomplete and stale.

The key difference from the earlier shell scripts is that we do not kill whole
rounds just because a coarse "alive batch" threshold dropped. Pending batches
are left alone, and running batches are never replaced automatically.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import os
import re
import shlex
import subprocess
import sys
from collections import Counter
import concurrent.futures

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

JOB_RE = re.compile(r"(/[^/\s]+/swe-zero-100b-r(?P<round>\d+)-batch(?P<batch>\d+))")
GS_FILE_RE = re.compile(r"^\s*(?P<size>\d+)\s+(?P<ts>\S+)\s+(?P<path>gs://\S+)$")
SHARD_FILE_RE = re.compile(
    r"shard_(?P<shard>\d+)_of_(?P<total>\d+)/(?P<name>rollouts(?:_resume)?\.json|sampling_plan\.json)$"
)


@dataclasses.dataclass(frozen=True)
class BatchLayout:
    batch: int
    shard_start: int
    shard_end: int
    replicas: int


@dataclasses.dataclass
class JobRef:
    job_id: str
    round_idx: int
    batch: int
    list_state: str


@dataclasses.dataclass
class BatchFiles:
    latest_update: dt.datetime | None = None
    total_bytes: int = 0
    rollouts_files: list[str] = dataclasses.field(default_factory=list)
    sampling_plan_files: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class BatchProgress:
    valid_rollouts: int | None = None
    error_rollouts: int | None = None
    target_rollouts: int | None = None
    prs_complete_100: int | None = None


@dataclasses.dataclass
class BatchStatus:
    layout: BatchLayout
    latest_job: JobRef | None
    summary: dict | None
    files: BatchFiles
    progress: BatchProgress | None
    action: str


def _run(cmd: list[str], *, check: bool = True) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed rc={proc.returncode}: {' '.join(shlex.quote(c) for c in cmd)}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    return proc.stdout


def _batch_layouts(total_shards: int, batches: int) -> list[BatchLayout]:
    base = total_shards // batches
    remainder = total_shards % batches
    layouts: list[BatchLayout] = []
    shard = 0
    for batch in range(batches):
        replicas = base + (1 if batch < remainder else 0)
        layouts.append(BatchLayout(batch=batch, shard_start=shard, shard_end=shard + replicas - 1, replicas=replicas))
        shard += replicas
    return layouts


def _discover_jobs(iris_bin: str, cluster: str, job_prefix: str = "/kevin/swe-zero-100b-") -> dict[int, list[JobRef]]:
    raw = _run(
        [iris_bin, "--cluster", cluster, "job", "list", "--prefix", job_prefix],
        check=False,
    )
    jobs: dict[int, list[JobRef]] = {}
    for line in raw.splitlines():
        m = JOB_RE.search(line)
        if not m:
            continue
        state_match = re.search(r"\b(pending|running|failed|killed|succeeded|worker_failed|unschedulable)\b", line)
        state = state_match.group(1) if state_match else "unknown"
        ref = JobRef(
            job_id=m.group(1),
            round_idx=int(m.group("round")),
            batch=int(m.group("batch")),
            list_state=state,
        )
        jobs.setdefault(ref.batch, []).append(ref)
    for refs in jobs.values():
        refs.sort(key=lambda r: (r.round_idx, r.job_id), reverse=True)
    return jobs


def _job_summary(iris_bin: str, cluster: str, job_id: str) -> dict | None:
    raw = _run([iris_bin, "--cluster", cluster, "job", "summary", job_id, "--json"], check=False)
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _parse_ts(value: str) -> dt.datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return dt.datetime.fromisoformat(value)


def _load_batch_files(output_root: str, total_shards: int) -> dict[int, BatchFiles]:
    raw = _run(
        [
            "gsutil",
            "ls",
            "-l",
            f"{output_root.rstrip('/')}/shard_*/rollouts*.json",
            f"{output_root.rstrip('/')}/shard_*/sampling_plan.json",
        ]
    )
    files: dict[int, BatchFiles] = {}
    for line in raw.splitlines():
        m = GS_FILE_RE.match(line)
        if not m:
            continue
        path = m.group("path")
        shard_m = SHARD_FILE_RE.search(path)
        if not shard_m:
            continue
        shard = int(shard_m.group("shard"))
        batch_info = files.setdefault(shard, BatchFiles())
        ts = _parse_ts(m.group("ts"))
        if batch_info.latest_update is None or ts > batch_info.latest_update:
            batch_info.latest_update = ts
        batch_info.total_bytes += int(m.group("size"))
        name = shard_m.group("name")
        if name == "sampling_plan.json":
            batch_info.sampling_plan_files.append(path)
        else:
            batch_info.rollouts_files.append(path)
    missing = [s for s in range(total_shards) if s not in files]
    if missing:
        logger.warning("Missing GCS objects for %d shards; first few: %s", len(missing), missing[:10])
    return files


def _count_json_entries(path: str) -> tuple[int, int, Counter[str], Counter[str]]:
    raw = _run(["gsutil", "cat", path], check=False)
    if not raw.strip():
        return 0, 0, Counter(), Counter()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Skipping malformed rollout file %s: %s", path, exc)
        return 0, 0, Counter(), Counter()
    valid = 0
    errors = 0
    by_instance: Counter[str] = Counter()
    for item in data:
        status = ((item.get("info", {}) or {}).get("exit_status", "")) or ""
        if "API error" in status or "Connection error" in status:
            errors += 1
            continue
        valid += 1
        instance_id = item.get("instance_id")
        if instance_id:
            by_instance[instance_id] += 1
    return valid, errors, by_instance, Counter()


def _load_sampling_plan(path: str) -> int:
    raw = _run(["gsutil", "cat", path], check=False)
    if not raw.strip():
        return 0
    payload = json.loads(raw)
    return int(payload.get("total_prs", 0))


def _count_batch_progress(
    output_root: str,
    layout: BatchLayout,
) -> BatchProgress:
    valid_rollouts = 0
    error_rollouts = 0
    target_rollouts = 0
    prs_complete_100 = 0
    for shard in range(layout.shard_start, layout.shard_end + 1):
        prefix = f"{output_root.rstrip('/')}/shard_{shard:03d}_of_126"
        n_prs = _load_sampling_plan(f"{prefix}/sampling_plan.json")
        target_rollouts += 100 * n_prs
        shard_counts: Counter[str] = Counter()
        for filename in ("rollouts.json", "rollouts_resume.json"):
            valid, errors, by_instance, _ = _count_json_entries(f"{prefix}/{filename}")
            valid_rollouts += valid
            error_rollouts += errors
            shard_counts.update(by_instance)
        prs_complete_100 += sum(1 for v in shard_counts.values() if v >= 100)
    return BatchProgress(
        valid_rollouts=valid_rollouts,
        error_rollouts=error_rollouts,
        target_rollouts=target_rollouts,
        prs_complete_100=prs_complete_100,
    )


def _render_age(ts: dt.datetime | None, now: dt.datetime) -> str:
    if ts is None:
        return "-"
    delta = now - ts
    minutes = int(delta.total_seconds() // 60)
    return f"{minutes}m"


def _choose_action(
    *,
    summary: dict | None,
    latest_update: dt.datetime | None,
    now: dt.datetime,
    stale_minutes: int,
    progress: BatchProgress | None,
) -> str:
    if progress and progress.target_rollouts and progress.valid_rollouts is not None:
        if progress.valid_rollouts >= progress.target_rollouts:
            return "complete"
    if summary:
        state = summary.get("state", "unknown")
        if state in {"running", "pending", "building"}:
            return "keep"
        if latest_update and (now - latest_update) < dt.timedelta(minutes=stale_minutes):
            return "wait"
        return "relaunch"
    if latest_update and (now - latest_update) < dt.timedelta(minutes=stale_minutes):
        return "wait"
    return "relaunch"


def _is_live_state(summary: dict | None, latest_job: JobRef | None) -> bool:
    live_states = {"running", "pending", "building"}
    if summary and summary.get("state") in live_states:
        return True
    if latest_job and latest_job.list_state in live_states:
        return True
    return False


def _is_complete(progress: BatchProgress | None) -> bool:
    return bool(
        progress
        and progress.target_rollouts
        and progress.valid_rollouts is not None
        and progress.valid_rollouts >= progress.target_rollouts
    )


def _hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return ""


def _priority_for_batch(args, batch: int) -> str:
    interactive_batches = _interactive_batch_set(args)
    return args.interactive_priority if batch in interactive_batches else args.priority


def _interactive_batch_set(args) -> set[int]:
    if args.interactive_batch_ids:
        return {int(b) for b in args.interactive_batch_ids.split(",") if b.strip()}
    n = max(0, args.interactive_batches)
    if n == 0:
        return set()
    return set(range(min(n, args.batches)))


def _submit_batch(args, round_idx: int, layout: BatchLayout) -> None:
    hf_token = _hf_token()
    if not hf_token:
        raise RuntimeError("HF_TOKEN not found in env or ~/.cache/huggingface/token")
    priority = _priority_for_batch(args, layout.batch)
    cmd = [
        args.iris_bin,
        "--cluster",
        args.cluster,
        "job",
        "run",
        "--job-name",
        f"swe-zero-100b-r{round_idx}-batch{layout.batch}",
        "--tpu",
        args.tpu_types,
        "--enable-extra-resources",
        "--replicas",
        str(layout.replicas),
        "--cpu",
        str(args.cpu),
        "--memory",
        args.memory,
        "--disk",
        args.disk,
        "--priority",
        priority,
        "--max-retries",
        str(args.max_retries),
        "--extra",
        "vllm",
        "--extra",
        "tpu",
        "-e",
        "VLLM_TPU_SKIP_PRECOMPILE",
        "1",
        "-e",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
        "1",
        "-e",
        "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION",
        "1",
        "-e",
        "SHARD_OFFSET",
        str(layout.shard_start),
        "-e",
        "HF_TOKEN",
        hf_token,
        "--no-wait",
        "--",
        "python",
        "experiments/swe_zero/run_swe_zero_multilang.py",
        "--local",
        "--all-prs",
        "--dataset",
        args.dataset,
        "--total-shards",
        str(args.total_shards),
        "--n-rollouts",
        str(args.n_rollouts),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-model-len",
        str(args.max_model_len),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--concurrency",
        str(args.concurrency),
        "--seed",
        str(args.seed),
        "--output_dir",
        args.output_root,
        "--disable-shard-lease",
    ]
    logger.info("Submitting batch %02d as round %d (priority=%s)", layout.batch, round_idx, priority)
    _run(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor and conservatively relaunch the 140B SWE-ZERO pipeline")
    parser.add_argument("--iris-bin", default="/home/kevin/marin-iris-tpu-cli/.venv/bin/iris")
    parser.add_argument("--cluster", default="marin")
    parser.add_argument("--output-root", default="gs://marin-us-central2/experiments/swe_zero_100b")
    parser.add_argument("--dataset", default="nebius/SWE-rebench-V2-PRs")
    parser.add_argument("--total-shards", type=int, default=126)
    parser.add_argument("--batches", type=int, default=13)
    parser.add_argument("--stale-minutes", type=int, default=45)
    parser.add_argument("--count-rollouts", action="store_true", help="Exactly count valid/error rollouts per batch")
    parser.add_argument(
        "--min-live-batches",
        type=int,
        default=4,
        help="Ensure at least this many batches are live (running/pending/building). "
        "If the fleet drops below this floor, backfill non-live incomplete batches.",
    )
    parser.add_argument("--relaunch", action="store_true", help="Actually submit replacement jobs")
    parser.add_argument("--max-relaunches", type=int, default=2, help="Safety cap per invocation")
    parser.add_argument("--cpu", type=int, default=16)
    parser.add_argument("--memory", default="32GB")
    parser.add_argument("--disk", default="60GB")
    parser.add_argument("--priority", default="batch", help="Priority for the majority of batches")
    parser.add_argument(
        "--interactive-priority",
        default="interactive",
        help="Priority assigned to the interactive-anchor batches",
    )
    parser.add_argument(
        "--interactive-batches",
        type=int,
        default=5,
        help="Run this many batches (indexed from 0) at interactive priority.",
    )
    parser.add_argument(
        "--interactive-batch-ids",
        default="",
        help="Explicit batch indices for interactive priority (overrides --interactive-batches).",
    )
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument(
        "--tpu-types",
        default="v6e-4",
        help="TPU variant for replacement batches. Must match --tensor-parallel-size (v6e-4 = 4 chips).",
    )
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-total-tokens", type=int, default=32768)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    now = dt.datetime.now(dt.timezone.utc)
    layouts = _batch_layouts(args.total_shards, args.batches)
    jobs_by_batch = _discover_jobs(args.iris_bin, args.cluster)
    files_by_shard = _load_batch_files(args.output_root, args.total_shards)
    latest_jobs = {batch: refs[0] for batch, refs in jobs_by_batch.items() if refs}
    summaries: dict[int, dict | None] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, len(latest_jobs) or 1)) as ex:
        future_to_batch = {
            ex.submit(_job_summary, args.iris_bin, args.cluster, ref.job_id): batch for batch, ref in latest_jobs.items()
        }
        for future in concurrent.futures.as_completed(future_to_batch):
            summaries[future_to_batch[future]] = future.result()

    next_round = max((ref.round_idx for refs in jobs_by_batch.values() for ref in refs), default=0) + 1
    batch_statuses: list[BatchStatus] = []
    relaunch_candidates: list[BatchLayout] = []

    print("batch  shards      latest_job                      state     tasks                age   bytes")
    print("-----  ----------  ------------------------------  --------  -------------------  ----  --------")

    for layout in layouts:
        refs = jobs_by_batch.get(layout.batch, [])
        latest_job = refs[0] if refs else None
        summary = summaries.get(layout.batch) if latest_job else None

        batch_files = BatchFiles()
        for shard in range(layout.shard_start, layout.shard_end + 1):
            info = files_by_shard.get(shard)
            if not info:
                continue
            if info.latest_update and (
                batch_files.latest_update is None or info.latest_update > batch_files.latest_update
            ):
                batch_files.latest_update = info.latest_update
            batch_files.total_bytes += info.total_bytes

        progress = _count_batch_progress(args.output_root, layout) if args.count_rollouts else None
        action = _choose_action(
            summary=summary,
            latest_update=batch_files.latest_update,
            now=now,
            stale_minutes=args.stale_minutes,
            progress=progress,
        )
        status_obj = BatchStatus(
            layout=layout,
            latest_job=latest_job,
            summary=summary,
            files=batch_files,
            progress=progress,
            action=action,
        )
        batch_statuses.append(status_obj)
        if action == "relaunch":
            relaunch_candidates.append(layout)

        if summary:
            state = summary.get("state", "unknown")
        elif latest_job:
            state = latest_job.list_state
        else:
            state = "missing"
        task_counts = "-"
        if summary:
            parts = [f"{k}={v}" for k, v in sorted((summary.get("task_state_counts") or {}).items()) if v]
            task_counts = ",".join(parts) if parts else "-"
        label = latest_job.job_id.split("/")[-1] if latest_job else "-"
        print(
            f"{layout.batch:>5}  "
            f"{layout.shard_start:03d}-{layout.shard_end:03d}  "
            f"{label:<30}  "
            f"{state:<8}  "
            f"{task_counts:<19}  "
            f"{_render_age(batch_files.latest_update, now):>4}  "
            f"{batch_files.total_bytes / 1e9:>6.2f}G"
        )
        if progress:
            pct = 100 * progress.valid_rollouts / progress.target_rollouts if progress.target_rollouts else 0.0
            print(
                "       "
                f"valid={progress.valid_rollouts} target={progress.target_rollouts} "
                f"({pct:.2f}%) errors={progress.error_rollouts} prs_complete_100={progress.prs_complete_100} "
                f"action={action}"
            )
        else:
            print(f"       action={action}")

    live_batches = [s for s in batch_statuses if _is_live_state(s.summary, s.latest_job)]
    logger.info("Live batches: %d/%d", len(live_batches), len(batch_statuses))

    if len(live_batches) < args.min_live_batches:
        needed = args.min_live_batches - len(live_batches)
        logger.warning(
            "Live batch count %d is below floor %d; selecting up to %d backfill candidates",
            len(live_batches),
            args.min_live_batches,
            needed,
        )
        backfill_pool = [
            s
            for s in batch_statuses
            if not _is_live_state(s.summary, s.latest_job) and s.layout not in relaunch_candidates
        ]
        backfill_pool.sort(
            key=lambda s: (
                0 if s.files.latest_update is None else 1,
                s.files.latest_update or dt.datetime.min.replace(tzinfo=dt.timezone.utc),
                s.layout.batch,
            )
        )
        added = 0
        for status in backfill_pool:
            if added >= needed:
                break
            if _is_complete(status.progress):
                logger.info("Skipping batch %02d backfill; already complete", status.layout.batch)
                continue
            relaunch_candidates.append(status.layout)
            status.action = "backfill"
            added += 1

    if relaunch_candidates:
        logger.warning(
            "Relaunch candidates: %s (next round=%d)",
            [layout.batch for layout in relaunch_candidates],
            next_round,
        )
    else:
        logger.info("No relaunch candidates")

    if args.relaunch:
        launched = 0
        failures: list[tuple[int, str]] = []
        for layout in relaunch_candidates:
            if launched >= args.max_relaunches:
                logger.warning("Stopping after %d relaunches due to --max-relaunches", launched)
                break
            try:
                _submit_batch(args, next_round, layout)
                launched += 1
            except Exception as exc:
                failures.append((layout.batch, str(exc).splitlines()[0] if str(exc) else type(exc).__name__))
                logger.error("Submit failed for batch %02d (continuing): %s", layout.batch, exc)
        if failures:
            logger.warning("Relaunch summary: %d submitted, %d failed: %s", launched, len(failures), failures)

    return 0


if __name__ == "__main__":
    sys.exit(main())
