# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["joblib", "numpy", "pandas", "scipy"]
# ///

"""Resume the frozen fitting screen after a prior controller exits.

Only local worker allocation changes. The fit source, protocol, seeds, split
membership, candidate identities, and atomic shard keys are verified unchanged.
"""

import argparse
import datetime
import hashlib
import json
import os
import time
from pathlib import Path

import joblib

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    screen_mariner_convex_fitting_round2_20260917 as screen,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=screen.DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, choices=(4, 8, 12), required=True)
    args = parser.parse_args()
    screen.register_candidates()
    manifest = screen.prepare(args.output)
    tasks = screen.planned_tasks("full", args.output)
    execution = {
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "controller_pid": os.getpid(),
        "workers": args.workers,
        "inner_max_num_threads": 1,
        "batch_size": 1,
        "orchestrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256((args.output / "screen_manifest.json").read_bytes()).hexdigest(),
        "scope": "Resume unchanged fitting after the prior local controller and workers exited",
    }
    record = args.output / f"execution_{time.time_ns()}.json"
    record.write_text(json.dumps(execution, indent=2) + "\n")
    counts: dict[str, int] = {}
    started = time.monotonic()
    print(json.dumps({"tasks": len(tasks), **execution}), flush=True)
    with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
        results = joblib.Parallel(n_jobs=args.workers, batch_size=1, return_as="generator_unordered")(
            joblib.delayed(screen.fit_task)(task, args.output, manifest["split_hash"]) for task in tasks
        )
        for index, result in enumerate(results, 1):
            counts[result] = counts.get(result, 0) + 1
            if index % 60 == 0 or index == len(tasks):
                print(
                    json.dumps(
                        {"completed": index, "total": len(tasks), "elapsed": round(time.monotonic() - started), **counts}
                    ),
                    flush=True,
                )
    screen.report(tasks, args.output, manifest["split_hash"], "full")
    record.write_text(
        json.dumps(
            execution
            | {"completed_utc": datetime.datetime.now(datetime.UTC).isoformat(), "status": "complete", "counts": counts},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
