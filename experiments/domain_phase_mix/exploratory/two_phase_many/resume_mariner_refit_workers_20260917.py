# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["joblib", "numpy", "pandas", "scipy"]
# ///

"""Resume frozen MARINER refits with newly available local CPU workers.

Worker-count changes do not alter the existing fit protocol, seeds, candidate
grids or atomic shard identities. The caller must first stop the prior local
controller and verify its workers have exited.
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
    validate_mariner_refit_robustness_20260917 as validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=validation.DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, choices=(4, 8, 12), required=True)
    args = parser.parse_args()
    if not (args.output / "protocol.json").is_file():
        raise ValueError("This entry point resumes an already frozen validation")
    validation.register_models()
    manifest = validation.prepare(args.output)
    tasks = validation.planned_tasks("all")
    execution = {
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "controller_pid": os.getpid(),
        "workers": args.workers,
        "inner_max_num_threads": 1,
        "batch_size": 1,
        "orchestrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256((args.output / "protocol.json").read_bytes()).hexdigest(),
        "scope": "Resume unchanged frozen fitting functions after the prior controller has exited",
    }
    (args.output / f"execution_{time.time_ns()}.json").write_text(json.dumps(execution, indent=2) + "\n")
    counts: dict[str, int] = {}
    started = time.monotonic()
    print(json.dumps({"tasks": len(tasks), **execution}), flush=True)
    with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
        results = joblib.Parallel(n_jobs=args.workers, batch_size=1, return_as="generator_unordered")(
            joblib.delayed(validation.fit_task)(task, args.output) for task in tasks
        )
        for index, result in enumerate(results, 1):
            counts[result] = counts.get(result, 0) + 1
            if index % 60 == 0 or index == len(tasks):
                print(
                    json.dumps(
                        {
                            "completed": index,
                            "total": len(tasks),
                            "elapsed": round(time.monotonic() - started),
                            **counts,
                        }
                    ),
                    flush=True,
                )
    validation.report(tasks, args.output, manifest, "all")


if __name__ == "__main__":
    main()
