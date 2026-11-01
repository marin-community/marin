# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The torch reference job: checks, one load, the queued ``router_rounding`` spec, a result check.

Steps, each fatal on failure, before any GPU work:

0. For an ``s3://`` source, the pod's S3 settings (``POD_S3_ENV``, set by Iris on CoreWeave task pods) must be
   present: the source audit reads through plain fsspec, which without them reports a present file as missing.
1. ``preflight.source_preflight`` on ``--expect`` (small files: metadata hashes, config fields, EOS, index size,
   the pinned positions.pt hash and its 465 sequences): ``<queue>/preflight.json``, exit 2.
2. The full source audit (``precision_source_audit.audit_precision_source``: every header, all 502 tensors streamed,
   native and FP32 hashes, finite values) against the ``files`` hashes of ``--expect``:
   ``<queue>/source-tensor-hashes.json``, or ``<queue>/source-audit-error.json`` and exit 2.

Then ``worker.serve`` loads the model once and runs the queued specs. Afterwards every ``router_rounding`` result is
checked with ``preflight.check_router_rounding_result`` (``<queue>/final-check.json``). Exit 0 means every step
passed; 1 means a spec or a result check failed; 2 means a check before the load failed.
"""

import argparse
import json
import logging
import os
import traceback

from rigging.filesystem.storage_path import prefix_join

from experiments.interp.snowball_identity import preflight
from experiments.interp.snowball_identity.worker import Queue, serve
from experiments.post_training.identity_context_distill.precision_source_audit import audit_precision_source

logger = logging.getLogger(__name__)

POD_S3_ENV = ("AWS_ENDPOINT_URL", "FSSPEC_S3", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")


def source_audit(weights: str, expected_hashes: dict[str, str], receipt_path: str) -> None:
    audit_precision_source(weights, expected_hashes=expected_hashes, receipt_path=receipt_path)


def final_check(queue: Queue, layers: int, k: int) -> dict:
    """Every ``router_rounding`` result in ``queue``, checked; a queue with none fails."""
    checks = {}
    fs, results = queue.fs, queue.path("results")
    for entry in fs.ls(results, detail=False) if fs.exists(results) else []:
        job_id = entry.rstrip("/").rsplit("/", 1)[-1]
        result = prefix_join(queue.url, f"results/{job_id}")
        done = queue.read("results", job_id, "done.json") or queue.read("results", job_id, "error.json") or {}
        spec = done.get("spec", {})
        if spec.get("command") != "router_rounding":
            continue
        checks[job_id] = preflight.check_router_rounding_result(
            result, spec["positions"], spec["positions_sha256"], layers, k
        )
    failures = [f"{job_id}: {f}" for job_id, found in checks.items() for f in found]
    if not checks:
        failures.append("no router_rounding result")
    return {"checks": checks, "failures": failures}


def run(queue: Queue, weights: str, expect: dict, idle_minutes: float, layers: int, k: int) -> int:
    missing = [name for name in POD_S3_ENV if not os.environ.get(name)] if weights.startswith("s3://") else []
    if missing:
        queue.write({"error": f"S3 settings missing on this pod: {missing}"}, "source-audit-error.json")
        logger.error("S3 settings missing on this pod: %s", missing)
        return 2
    failures = preflight.source_preflight(weights, expect)
    queue.write({"weights": weights, "expect": expect, "failures": failures}, "preflight.json")
    if failures:
        logger.error("preflight failed: %s", failures)
        return 2
    try:
        source_audit(weights, expect["files"], prefix_join(queue.url, "source-tensor-hashes.json"))
    except Exception as error:
        logger.exception("source audit failed")
        queue.write({"error": repr(error), "traceback": traceback.format_exc()}, "source-audit-error.json")
        return 2
    failed = serve(queue, weights, idle_minutes)
    report = final_check(queue, layers, k)
    queue.write(report, "final-check.json")
    for failure in report["failures"]:
        logger.error("final check: %s", failure)
    return 1 if failed or report["failures"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--expect", required=True, help="preflight expectations JSON (s3:// or local)")
    parser.add_argument("--idle-minutes", type=float, default=2)
    parser.add_argument("--layers", type=int, default=26)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()
    expect = json.loads(preflight.read(args.expect))
    raise SystemExit(run(Queue(args.queue), args.weights, expect, args.idle_minutes, args.layers, args.k))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
