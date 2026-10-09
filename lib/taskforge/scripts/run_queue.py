# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run an unattended Taskforge queue from a run config on a laptop.

``CONFIG`` is a ``queue.config.RunConfig`` file (``docs/policy.example.json`` is the committed
example; its ``root`` and ``image_cache`` are absolute paths). ``--inputs MODULE:FUNCTION`` names a
``queue.job.InputsFactory``: a function of the run's model client and run root that returns the run's
ideas, proposal source and idea record (``queue.job.RunInputs``). ``--model MODULE:FUNCTION`` names a
function of no arguments that returns the run's ``queue.job.RunModel``. The band rules are policy
fields of ``CONFIG``. The run writes ``summary.json`` into the run root and exits non-zero when any
item ended ``FAILED``::

    uv run --project lib/taskforge --frozen python lib/taskforge/scripts/run_queue.py run.json \\
      --inputs my_inputs:inputs --model my_inputs:model

A relaunch on the same root resumes every item from its event log. ``--retry-failed`` re-enters items
that ended ``FAILED``.
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
from taskforge.queue.job import run_job
from taskforge.queue.run import FailedItems


def named(name: str, flag: str) -> object:
    module, sep, attr = name.partition(":")
    if not sep:
        raise SystemExit(f"{flag} takes MODULE:FUNCTION, got {name!r}")
    return getattr(importlib.import_module(module), attr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--inputs", required=True, help="MODULE:FUNCTION returning the run's RunInputs")
    parser.add_argument("--model", required=True, help="MODULE:FUNCTION returning the run's RunModel")
    parser.add_argument("--retry-failed", action="store_true", help="re-enter items that ended FAILED")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = load_run_config(args.config)
    failed = FailedItems.RETRY if args.retry_failed else FailedItems.SKIP
    model = named(args.model, "--model")()
    summary = asyncio.run(run_job(config, named(args.inputs, "--inputs"), failed, model))
    print("RUN_SUMMARY " + json.dumps(summary.summary_json()), flush=True)
    if Terminal.FAILED in summary.items.values() or summary.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
