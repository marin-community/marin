# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Learning curves of the remaining trained-proposal models of the paper's Table 11 on the pinned-protocol subsets.

The 2026-09-08 study and its extensions already cover MARINER, Olmix, the quadratic, the spline, both RegMix variants
(`learning_curve_regmix_oof_20260924.py`) and the per-bucket-threshold ablation (`learning_curve_pbt_fits_20260929.py`).
This wrapper adds the three models still missing, on exactly the same subsets, folds and seeds, so the appendix can show
one learning-curve figure for MARINER's ablations and one for the baselines:

- ``mariner_cvx``: the fully convex objective (unsquared softplus of raw epochs), the trained ablation of Table 11;
- ``mariner_add``: the additive response (no floor, no exponential link);
- ``krr``: the Hellinger kernel ridge baseline.

The harness's non-floored path fits Olmix's log-linear head, so all three join ``FLOORED_MODELS``, the generic
per-component path that calls each registry model's own ``fit``. Records go to this study's directory so the earlier
studies' protocol hashes stay valid; ``records/<target>/mariner`` links to the 2026-09-08 study's MARINER records.

usage: uv run --offline --no-sync python learning_curve_extra_fits_20261006.py run --draws 10 --workers 13 [--dry-run]
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

MODULE_NAME = "experiments.domain_phase_mix.exploratory.two_phase_many.learning_curve_extra_fits_20261006"
EXTRA_MODELS = {
    "mariner_cvx": f"{harness.MODEL_IDS['mariner']}_raw_epoch_hinge",
    "mariner_add": "weibull_softplus_unscaled",
    "krr": "hellinger_krr",
}
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "learning_curve_extra_delphi_3e18_20261006"
RECORD_DIR = OUTPUT_DIR / "records"
LOGGER = logging.getLogger("learning_curve_extra")

# Applied at import, so loky workers that unpickle ``run_job`` by reference see the same model table.
for _model, _model_id in EXTRA_MODELS.items():
    harness.MODEL_IDS[_model] = _model_id
harness.MODEL_KEYS = tuple(harness.MODEL_IDS)
harness.FLOORED_MODELS = (*harness.FLOORED_MODELS, *(m for m in EXTRA_MODELS if m not in harness.FLOORED_MODELS))

# The study as `learning_curve_metrics_20260905.py --study mariner_extra` reads it: the three models beside the
# 2026-09-08 study's MARINER records (their protocol hash predates these models, so the metrics run needs --allow-stale).
TARGETS = harness.TARGETS
MODEL_KEYS = ("mariner", *EXTRA_MODELS)
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


def link_mariner_records() -> None:
    """Point ``records/<target>/mariner`` at the 2026-09-08 study's MARINER records, as the pbt study does."""
    for target in TARGETS:
        link = RECORD_DIR / target / "mariner"
        if link.exists() or link.is_symlink():
            continue
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(Path("../../../learning_curve_mariner_delphi_3e18_20260908/records") / target / "mariner")


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
        message = (
            f"{done}/{len(pending)} pending jobs done in {elapsed / 60:.1f} min; "
            f"projected total {elapsed / 60 * len(pending) / done:.1f} min; failed {counts['failed']}"
        )
        if done % 10 == 0 or done == len(pending):
            LOGGER.info(message)
        progress.write_text(message + "\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="fit every pending job of the three models")
    run.add_argument("--draws", type=int, required=True)
    run.add_argument("--workers", type=int, default=os.cpu_count())
    run.add_argument("--sizes", type=int, nargs="*", default=list(harness.SUBSET_SIZES))
    run.add_argument("--targets", nargs="*", default=list(harness.TARGETS))
    run.add_argument("--models", nargs="*", default=list(EXTRA_MODELS), choices=list(EXTRA_MODELS))
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    panel = harness.benchmark.load_panel(harness.PANEL)
    jobs = harness.enumerate_jobs(panel, args.draws, tuple(args.sizes), tuple(args.targets), tuple(args.models))
    if args.dry_run:
        pending = sum(not harness.record_valid(harness.job_path(job, RECORD_DIR), job) for job in jobs)
        print(f"{len(jobs)} jobs, {pending} pending; protocol {harness.protocol_hash()[:12]}")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    link_mariner_records()
    (OUTPUT_DIR / "protocol_hash.txt").write_text(harness.protocol_hash() + "\n")
    counts = run_jobs(jobs, args.workers, RECORD_DIR, OUTPUT_DIR / "progress.txt")
    print(f"completed: {counts}")


if __name__ == "__main__":
    importable = importlib.import_module(MODULE_NAME)
    importable.main()
