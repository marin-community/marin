# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect the seed-panel eval results into one blob and print it to the job log.

Runs as a CPU job on the cluster (the results live on the CW object store,
which the submitting VM cannot read). Reads every
``evaluation/grug_logprob/<run>/<task>/results.json``, keeps the lm-eval
``results`` section per (run, task), writes the combined JSON next to the
results tree, and prints it gzip+base64 between markers so the submitter can
reconstruct it from ``iris job logs``.
"""

import argparse
import base64
import gzip
import json
import logging

from rigging.filesystem import StoragePath, marin_prefix, prefix_join

logger = logging.getLogger(__name__)

MARK_BEGIN = "===SEEDPANEL_EVALS_B64_BEGIN==="
MARK_END = "===SEEDPANEL_EVALS_B64_END==="


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="run names, e.g. rav_mve_seedpanel_h100_00 ...")
    parser.add_argument("--out-name", default="seedpanel_eval_results_combined.json")
    args = parser.parse_args()

    # Flatten so a single comma/space-joined token works too (iris can pass a
    # multi-value arg as one argv element).
    runs = [r for chunk in args.runs for r in chunk.replace(",", " ").split()]

    combined: dict[str, dict] = {}
    missing: list[str] = []
    for run in runs:
        run_prefix = StoragePath(prefix_join(marin_prefix(), f"evaluation/grug_logprob/{run}"))
        per_task: dict[str, dict] = {}
        for task_dir in sorted(run_prefix.ls(), key=str):
            results_path = StoragePath(str(task_dir).rstrip("/")) / "results.json"
            if not results_path.exists():
                missing.append(str(results_path))
                continue
            payload = json.loads(results_path.read_text())
            per_task[results_path.parent.name] = payload.get("results", {})
        combined[run] = per_task
        logger.info("%s: %d tasks", run, len(per_task))

    blob = json.dumps({"runs": combined, "missing": missing}, sort_keys=True).encode()
    out_path = StoragePath(prefix_join(marin_prefix(), f"evaluation/grug_logprob/{args.out_name}"))
    out_path.write_text(blob.decode())
    logger.info("wrote %s (%d bytes, %d missing)", out_path, len(blob), len(missing))

    encoded = base64.b64encode(gzip.compress(blob)).decode()
    print(MARK_BEGIN)
    for i in range(0, len(encoded), 3000):
        print(encoded[i : i + 3000])
    print(MARK_END)


if __name__ == "__main__":
    main()
