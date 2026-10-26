# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Learning curve of MARINER's one-threshold-per-bucket ablation, on the pinned-protocol study's subsets and folds.

Preempts the question raised by Table 1 and Table 5, where the per-bucket-threshold ablation ties MARINER at the full
swarm: does an extra nonlinear parameter per bucket cost sample efficiency? This wrapper adds the ablation
(`weibull_softplus_unscaled@kappa_floor_link_flat15_nocap_per_bucket_threshold`) to
`learning_curve_mariner_fits_20260908.py` at import time and fits it on exactly that study's subsets, folds and seeds,
writing records to its own directory so the 2026-09-08 study's records (whose protocol hash covers the model list) stay
valid. Compare with the study's MARINER records through `learning_curve_metrics_20260905.py --study mariner_pbt`.

usage: uv run --offline --no-sync python learning_curve_pbt_fits_20260929.py run --draws 10 --workers 14 [--dry-run]
"""

import argparse
import importlib
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from joblib import Parallel, delayed  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    learning_curve_mariner_fits_20260908 as harness,
)

MODULE_NAME = "experiments.domain_phase_mix.exploratory.two_phase_many.learning_curve_pbt_fits_20260929"
MODEL = "mariner_pbt"
MODEL_ID = "weibull_softplus_unscaled@kappa_floor_link_flat15_nocap_per_bucket_threshold"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "learning_curve_pbt_delphi_3e18_20260929"
RECORD_DIR = OUTPUT_DIR / "records"
LOGGER = logging.getLogger("learning_curve_pbt")

# Applied at import, so loky workers that unpickle ``run_job`` by reference see the same model table.
harness.MODEL_IDS[MODEL] = MODEL_ID
harness.MODEL_KEYS = tuple(harness.MODEL_IDS)
if MODEL not in harness.FLOORED_MODELS:
    harness.FLOORED_MODELS = (*harness.FLOORED_MODELS, MODEL)

# The study as `learning_curve_metrics_20260905.py --study mariner_pbt` reads it: the ablation beside the 2026-09-08
# study's MARINER records, which `records/<target>/mariner` links to (their protocol hash predates this model, so the
# metrics run needs --allow-stale).
TARGETS = harness.TARGETS
MODEL_KEYS = ("mariner", MODEL)
SUBSET_SIZES = harness.SUBSET_SIZES
OUTER_FOLDS = harness.OUTER_FOLDS
PANEL = harness.PANEL
Job = harness.Job
component_chunks = harness.component_chunks
protocol_hash = harness.protocol_hash


def job_path(job: harness.Job, record_dir: Path = RECORD_DIR) -> Path:
    return harness.job_path(job, record_dir)


def run_job(job: harness.Job, record_dir: Path) -> str:
    return harness.run_job(job, record_dir)


def run_jobs(jobs: list, workers: int, record_dir: Path, progress: Path) -> dict[str, int]:
    counts = {"cached": 0, "fitted": 0, "failed": 0}
    pending = [job for job in jobs if not harness.record_valid(harness.job_path(job, record_dir), job)]
    counts["cached"] = len(jobs) - len(pending)
    LOGGER.info("%d jobs, %d pending, %d workers", len(jobs), len(pending), workers)
    started = time.monotonic()
    parallel = Parallel(n_jobs=workers, backend="loky", batch_size=1, return_as="generator_unordered")
    for done, status in enumerate(parallel(delayed(run_job)(job, record_dir) for job in pending), start=1):
        counts[status] += 1
        elapsed = time.monotonic() - started
        if done % 10 == 0 or done == len(pending):
            message = (
                f"{done}/{len(pending)} pending jobs done in {elapsed / 60:.1f} min; "
                f"projected total {elapsed / 60 * len(pending) / done:.1f} min; failed {counts['failed']}"
            )
            LOGGER.info(message)
            progress.write_text(message + "\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="fit every pending job of the ablation")
    run.add_argument("--draws", type=int, required=True)
    run.add_argument("--workers", type=int, default=os.cpu_count())
    run.add_argument("--sizes", type=int, nargs="*", default=list(harness.SUBSET_SIZES))
    run.add_argument("--targets", nargs="*", default=list(harness.TARGETS))
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    panel = harness.benchmark.load_panel(harness.PANEL)
    jobs = harness.enumerate_jobs(panel, args.draws, tuple(args.sizes), tuple(args.targets), (MODEL,))
    if args.dry_run:
        pending = sum(not harness.record_valid(harness.job_path(job, RECORD_DIR), job) for job in jobs)
        print(f"{len(jobs)} jobs, {pending} pending; protocol {harness.protocol_hash()[:12]}")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "protocol_hash.txt").write_text(harness.protocol_hash() + "\n")
    counts = run_jobs(jobs, args.workers, RECORD_DIR, OUTPUT_DIR / "progress.txt")
    print(f"completed: {counts}")


if __name__ == "__main__":
    importable = importlib.import_module(MODULE_NAME)
    importable.main()
