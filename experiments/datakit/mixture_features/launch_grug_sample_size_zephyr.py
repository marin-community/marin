# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Thin launcher for the sample-size sensitivity zephyr pipeline.

Separate from ``grug_sample_size_zephyr`` for the same reason as
``launch_grug_histograms_zephyr``: running the pipeline module as ``__main__`` would make
cloudpickle stamp the map callable with ``__module__='__main__'`` and the coordinator pod
would fail to unpickle it. Importing ``run`` here keeps the real qualified module name.
"""

import argparse
import logging
import os

from experiments.datakit.mixture_features.grug_histograms_zephyr import INPUTS_DIR
from experiments.datakit.mixture_features.grug_sample_size_zephyr import run


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-meta", default=os.path.join(INPUTS_DIR, "store_meta.json"))
    ap.add_argument("--readout-only", action="store_true", help="skip the map stage; recompute the summary")
    args = ap.parse_args()
    run(args.store_meta, args.readout_only)


if __name__ == "__main__":
    main()
