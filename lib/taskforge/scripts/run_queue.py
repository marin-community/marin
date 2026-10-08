# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run an unattended Taskforge queue from a run config, on a laptop or inside an Iris task.

``CONFIG`` is a ``queue.config.RunConfig`` file (``docs/policy.example.json`` is the committed
example). ``--inputs MODULE:FUNCTION`` names a ``queue.job.InputsFactory``: a function of the run's
``GlmClient`` and run root that returns the run's ideas, proposal source, idea record, adversary context
and triage checks (``queue.job.RunInputs``); the band rules and the adversary submission budget and
repair threshold are policy fields of ``CONFIG``. The run writes ``summary.json`` into the run root and
exits non-zero when any item ended ``FAILED``.

Laptop, through the GLM port-forward on the interactive pool (``"glm": {"kind": "laptop", ...,
"pool": "high"}``, ``"host": "laptop"``; ``root``, ``image_cache`` and ``token_file`` absolute paths)::

    cd lib/taskforge && uv run python scripts/run_queue.py run.json --inputs my_inputs:inputs

Iris, with ``run.json`` a copy of the committed example whose ``relay_job`` names the cluster's GLM
relay (relay, bulk pool, root under ``$IRIS_OUTPUT_DIR``). The token and the Parallel key travel in
the job environment; ``queue.job`` removes them from what sandboxes inherit::

    lib/taskforge/.venv/bin/iris --cluster=marin job run --no-wait --no-sync \\
      --job-name taskforge-queue-NN --target-cluster cw-rno2a --priority batch \\
      --cpu 8 --memory 32GB --timeout 172800 \\
      -e GLM_API_TOKEN "$GLM_BULK_TOKEN" -e PARALLEL_KEY "$PARALLEL_KEY" -- \\
      bash -c 'cd lib/taskforge && uv run --frozen python scripts/run_queue.py run.json \\
        --inputs <module>:<function>'

A relaunch on the same root resumes every item from its event log; on Iris, set ``restore_from`` to
the previous attempt's archived run root. ``--retry-failed`` re-enters items that ended ``FAILED``.
"""

import argparse
import asyncio
import importlib
import json
import logging
import sys
from pathlib import Path

from taskforge.loop.events import Terminal
from taskforge.queue.config import load_run_config
from taskforge.queue.job import InputsFactory, run_job
from taskforge.queue.run import FailedItems


def inputs_factory(name: str) -> InputsFactory:
    module, sep, attr = name.partition(":")
    if not sep:
        raise SystemExit(f"--inputs takes MODULE:FUNCTION, got {name!r}")
    return getattr(importlib.import_module(module), attr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--inputs", required=True, help="MODULE:FUNCTION returning the run's RunInputs")
    parser.add_argument("--retry-failed", action="store_true", help="re-enter items that ended FAILED")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = load_run_config(args.config)
    failed = FailedItems.RETRY if args.retry_failed else FailedItems.SKIP
    summary = asyncio.run(run_job(config, inputs_factory(args.inputs), failed))
    print("RUN_SUMMARY " + json.dumps(summary.summary_json()), flush=True)
    if Terminal.FAILED in summary.items.values() or summary.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
