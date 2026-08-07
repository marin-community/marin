# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic trainer-subprocess entry point spawned by ``GPUHangSupervisor``.

The supervisor passes a pickled spec naming an importable entrypoint function
and its (pickled) config. This module imports the entrypoint, runs it, and maps
any escaping exception onto the exit-code contract in
``levanter.recovery.types`` so the supervisor can classify the death. Fatal
signals (XLA ``LOG(FATAL)`` → SIGABRT) and the progress watchdog's own
``os._exit(124)`` bypass this handler by design and are read from the returncode.

Environment (``XLA_FLAGS`` etc.) is already set by the supervisor at spawn time,
so JAX reads the intended flags regardless of import order here.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import pickle
import sys

from levanter.recovery.detection import classify_exception
from levanter.recovery.types import EXIT_STALL, EXIT_STICKY_FAULT, FaultClass


logger = logging.getLogger(__name__)


def _load_spec(spec_path: str):
    with open(spec_path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()

    spec = _load_spec(args.spec)
    module = importlib.import_module(spec["entry_module"])
    entrypoint = getattr(module, spec["entry_qualname"])
    config = spec["config"]

    try:
        entrypoint(config)
    except BaseException as exc:  # noqa: BLE001 - map to the exit-code contract, then re-signal
        fault_class = classify_exception(exc)
        logger.exception("trainer subprocess raised; classified as %s", fault_class)
        if fault_class is FaultClass.STICKY:
            sys.exit(EXIT_STICKY_FAULT)
        if fault_class is FaultClass.STALL:
            sys.exit(EXIT_STALL)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
