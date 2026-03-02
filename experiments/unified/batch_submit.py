"""Batch job submission for unified pre-training experiments.

Define jobs as a list of dicts, then run this script to submit them all.
After submitting each job via ray_run.py, polls the Ray cluster for job
status. If the job reaches FAILED (e.g. OOM-killed on the head node),
sleeps 10 minutes and retries. Each job is only counted as successful
once it reaches RUNNING status on the cluster.

Usage:
    # Submit all jobs
    uv run python experiments/unified/batch_submit.py

    # Dry run — print what would be submitted
    uv run python experiments/unified/batch_submit.py --dry_run
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Ray token auth setup (must happen before importing ray)
os.environ.setdefault("RAY_AUTH_MODE", "token")
_default_token_path = Path.home() / ".ray" / "auth_token"
if _default_token_path.exists() and "RAY_AUTH_TOKEN_PATH" not in os.environ:
    os.environ["RAY_AUTH_TOKEN_PATH"] = str(_default_token_path)

from ray.job_submission import JobSubmissionClient, JobStatus

from fray.v1.cluster.ray import DashboardConfig, ray_dashboard

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Common defaults (override per-job if needed)
# ──────────────────────────────────────────────────────────────────────
CLUSTER = "infra/marin-us-central2.yaml"
ENTRYPOINT = "python experiments/unified/unified_pretrain_demo.py"
TPU_TYPE = "v4-64"

# Retry settings
POLL_INTERVAL = 10  # seconds between status checks
POLL_DURATION = 180  # total seconds to poll before considering job stable
STABILIZE_DURATION = 180  # seconds to keep watching after RUNNING, to catch quick OOM crashes
RETRY_WAIT = 600  # seconds to sleep before retrying after failure (10 min)
MAX_RETRIES = 20

# ──────────────────────────────────────────────────────────────────────
# Job list — each dict is one job submission.
# Keys become env vars on the Ray cluster.
# EXP_NAME is required; everything else falls back to the defaults
# defined in unified_pretrain_demo.py.
# ──────────────────────────────────────────────────────────────────────
JOBS: list[dict] = [
    {
        "EXP_NAME": "unified-qwen3-1.7b-1-1-1-w0.5-3e4-demo4",
        "TEXT_WEIGHT": "1.0",
        "MULTIMODAL_WEIGHT": "2.0",
        "W_VISUAL": "0.5",
        "UND_GEN_RATIO": "1",
        "LEARNING_RATE": "3e-4",
    },
    {
        "EXP_NAME": "unified-qwen3-1.7b-1-1-1-w1-3e4-demo4",
        "TEXT_WEIGHT": "1.0",
        "MULTIMODAL_WEIGHT": "2.0",
        "W_VISUAL": "1",
        "UND_GEN_RATIO": "1",
        "LEARNING_RATE": "3e-4",
    },
    # {
    #     "EXP_NAME": "unified-qwen3-1.7b-1-1-1-w0.3-3e4-demo2",
    #     "TEXT_WEIGHT": "1.0",
    #     "MULTIMODAL_WEIGHT": "2.0",
    #     "W_VISUAL": "0.3",
    #     "UND_GEN_RATIO": "1",
    #     "LEARNING_RATE": "3e-4",
    # },
    # {
    #     "EXP_NAME": "unified-qwen3-1.7b-1-1-1-w0.2-1e3-demo2",
    #     "TEXT_WEIGHT": "1.0",
    #     "MULTIMODAL_WEIGHT": "2.0",
    #     "W_VISUAL": "0.2",
    #     "UND_GEN_RATIO": "1",
    #     "LEARNING_RATE": "1e-3",
    # },
    # {
    #     "EXP_NAME": "unified-qwen3-1.7b-1-1-1-w0.2-1e4-demo2",
    #     "TEXT_WEIGHT": "1.0",
    #     "MULTIMODAL_WEIGHT": "2.0",
    #     "W_VISUAL": "0.2",
    #     "UND_GEN_RATIO": "1",
    #     "LEARNING_RATE": "1e-4",
    # },
    # {
    #     "EXP_NAME": "unified-qwen3-1.7b-2-1-1-w0.2-1e3-demo2",
    #     "TEXT_WEIGHT": "2.0",
    #     "MULTIMODAL_WEIGHT": "2.0",
    #     "W_VISUAL": "0.2",
    #     "UND_GEN_RATIO": "1",
    #     "LEARNING_RATE": "1e-3",
    # },
    # ── Add more jobs below ──────────────────────────────────────────
]


def _build_ray_run_cmd(
    job: dict,
    cluster: str,
    entrypoint: str,
    tpu_type: str,
) -> list[str]:
    """Build the uv run ray_run command for a single job (with --no_wait)."""
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_key:
        raise RuntimeError("WANDB_API_KEY is not set in the environment")

    cmd = [
        "uv", "run", "python", "-m", "marin.run.ray_run",
        "--cluster", cluster,
        "--no_wait",
        "-e", "WANDB_API_KEY", wandb_key,
        "-e", "TPU_TYPE", tpu_type,
    ]
    for key, value in job.items():
        cmd.extend(["-e", key, str(value)])
    cmd.extend(["--", *entrypoint.split()])
    return cmd


def _submit_via_ray_run(cmd: list[str]) -> str | None:
    """Submit a job via ray_run.py --no_wait, return the submission ID."""
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print ray_run output for visibility
    if result.stdout:
        sys.stdout.write(result.stdout)
        sys.stdout.flush()
    if result.stderr:
        sys.stderr.write(result.stderr)
        sys.stderr.flush()

    # Parse job ID from output: "Job submitted with ID: <id>"
    # ray_run.py uses logger.info() which goes to stderr, so search both.
    combined_output = (result.stdout or "") + (result.stderr or "")
    match = re.search(r"Job submitted with ID:\s*(\S+)", combined_output)
    if match:
        return match.group(1)

    logger.error("Failed to parse job ID from ray_run output")
    return None


def _poll_until_decided(
    client: JobSubmissionClient,
    submission_id: str,
    exp_name: str,
) -> JobStatus:
    """Poll job status until it stabilizes or fails.

    Two phases:
      1. Wait up to POLL_DURATION for the job to leave PENDING.
      2. Once RUNNING, keep polling for STABILIZE_DURATION more seconds.
         If the job crashes back to FAILED during this window (e.g. OOM),
         we return FAILED so the retry logic kicks in.
    """
    start = time.time()
    last_status = None
    running_since: float | None = None

    while True:
        elapsed = time.time() - start
        status = client.get_job_status(submission_id)

        if status != last_status:
            logger.info("  %s status: %s (%.0fs)", exp_name, status, elapsed)
            last_status = status

        # Terminal failures — return immediately.
        if status in (JobStatus.FAILED, JobStatus.SUCCEEDED, JobStatus.STOPPED):
            return status

        if status == JobStatus.RUNNING:
            if running_since is None:
                running_since = time.time()
                logger.info("  %s entered RUNNING, stabilizing for %ds...", exp_name, STABILIZE_DURATION)
            elif time.time() - running_since >= STABILIZE_DURATION:
                logger.info("  %s has been RUNNING for %ds — looks stable.", exp_name, STABILIZE_DURATION)
                return status
        else:
            # Still PENDING — check overall timeout.
            running_since = None
            if elapsed >= POLL_DURATION:
                logger.info("  %s still %s after %ds — treating as stable enough.", exp_name, status, POLL_DURATION)
                return status

        time.sleep(POLL_INTERVAL)


def submit_one_job(
    client: JobSubmissionClient,
    job: dict,
    exp_name: str,
    cluster: str,
    entrypoint: str,
    tpu_type: str,
) -> bool:
    """Submit a single job with retry logic.

    1. Submits via ray_run.py --no_wait (creates its own short-lived SSH tunnel).
    2. Polls job status via the persistent Ray API client.
    3. If the job reaches RUNNING/SUCCEEDED, returns True.
    4. If the job FAILED (e.g. OOM), sleeps RETRY_WAIT and retries.
    """
    cmd = _build_ray_run_cmd(job, cluster, entrypoint, tpu_type)

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info("[Attempt %d/%d] Submitting %s", attempt, MAX_RETRIES, exp_name)

        submission_id = _submit_via_ray_run(cmd)
        if not submission_id:
            logger.error("  Submission failed for %s (no job ID returned)", exp_name)
            return False

        logger.info("  Polling status for job %s ...", submission_id)
        status = _poll_until_decided(client, submission_id, exp_name)

        if status in (JobStatus.RUNNING, JobStatus.SUCCEEDED, JobStatus.PENDING):
            logger.info("  Job %s is %s.", exp_name, status)
            return True

        # FAILED — retry after sleeping.
        if attempt < MAX_RETRIES:
            logger.warning(
                "  Job %s FAILED on cluster. Sleeping %d seconds before retry...",
                exp_name, RETRY_WAIT,
            )
            time.sleep(RETRY_WAIT)
        else:
            logger.error("  Job %s FAILED after %d retries. Giving up.", exp_name, MAX_RETRIES)
            return False

    return False


def main():
    parser = argparse.ArgumentParser(description="Batch submit unified pre-training jobs")
    parser.add_argument("--dry_run", action="store_true", help="Print what would be submitted")
    parser.add_argument("--cluster", default=CLUSTER, help=f"Cluster config (default: {CLUSTER})")
    parser.add_argument("--entrypoint", default=ENTRYPOINT, help=f"Entrypoint command (default: {ENTRYPOINT})")
    parser.add_argument("--tpu_type", default=TPU_TYPE, help=f"TPU type (default: {TPU_TYPE})")
    args = parser.parse_args()

    if not JOBS:
        logger.error("JOBS list is empty — nothing to submit.")
        sys.exit(1)

    if args.dry_run:
        logger.info("DRY RUN — %d job(s):", len(JOBS))
        for i, job in enumerate(JOBS):
            logger.info("  [%d] %s: %s", i + 1, job.get("EXP_NAME", f"job-{i}"), job)
        return

    logger.info("Submitting %d job(s) to cluster %s", len(JOBS), args.cluster)

    # Open a persistent SSH tunnel for status polling.
    # ray_run.py will create its own short-lived tunnel for each submission
    # (on a different local port since this one occupies 8265).
    dashboard_config = DashboardConfig.from_cluster(args.cluster)
    succeeded = []
    failed = []

    with ray_dashboard(dashboard_config):
        client = JobSubmissionClient(os.environ["RAY_ADDRESS"])

        for i, job in enumerate(JOBS):
            exp_name = job.get("EXP_NAME", f"job-{i}")
            logger.info("==== [%d/%d] %s ======", i + 1, len(JOBS), exp_name)

            if submit_one_job(client, job, exp_name, args.cluster, args.entrypoint, args.tpu_type):
                succeeded.append(exp_name)
            else:
                failed.append(exp_name)
                logger.error("Job %s FAILED.", exp_name)

    logger.info("==============================================")
    logger.info(
        "Results: %d succeeded, %d failed out of %d total",
        len(succeeded), len(failed), len(JOBS),
    )
    if succeeded:
        logger.info("  Succeeded: %s", ", ".join(succeeded))
    if failed:
        logger.error("  Failed: %s", ", ".join(failed))


if __name__ == "__main__":
    main()
