# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Thin launcher for the distributed grug-histogram zephyr pipeline.

Kept separate from ``grug_histograms_zephyr`` so the pipeline's map function and its worker-side
helpers keep their real qualified ``__module__`` (``experiments.datakit.mixture_features.
grug_histograms_zephyr``). Running the pipeline module itself via ``python -m`` would execute it as
``__main__``, and cloudpickle would then stamp the map callable with ``__module__='__main__'`` — the
zephyr coordinator pod unpickles under a different ``__main__`` and fails with
``AttributeError: Can't get attribute '_worker_tokenizer'``. Importing ``run`` here avoids that.
"""

import argparse
import logging

from experiments.datakit.mixture_features.grug_histograms_zephyr import run


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets-json", required=True)
    ap.add_argument("--store-meta", required=True)
    ap.add_argument("--max-shards", type=int, default=None, help="process at most N buckets (smoke test)")
    args = ap.parse_args()
    run(args.buckets_json, args.store_meta, args.max_shards)


if __name__ == "__main__":
    main()
