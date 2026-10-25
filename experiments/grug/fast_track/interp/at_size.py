# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run an interp module with the instrumented forward built for another study tag's dimensions.

Usage: ``python -m experiments.grug.fast_track.interp.at_size <study tag> <module> <args...>``. ``forward`` reads its
dimensions from ``common.STUDY_TAG`` when first imported, so this sets it before importing ``<module>`` (e.g. the d512
``tn-d512-ckpt1-rno`` from the d1280 branch).
"""

import importlib
import sys

from experiments.grug.fast_track.interp import common


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: at_size <study tag> <module> <args...>")
    tag, module = sys.argv[1], sys.argv[2]
    if tag not in common.CHECKPOINTS:
        raise SystemExit(f"unknown study tag {tag}")
    assert "experiments.grug.fast_track.interp.forward" not in sys.modules, "forward was imported before at_size"
    common.STUDY_TAG = tag
    sys.argv = [module, *sys.argv[3:]]
    importlib.import_module(f"experiments.grug.fast_track.interp.{module}").main()


if __name__ == "__main__":
    main()
