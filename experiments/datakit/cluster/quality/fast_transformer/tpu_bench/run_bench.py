# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launcher for the TPU throughput benchmark stages.

Runs as ``__main__`` and dispatches into a stage module's ``run()``. This indirection
matters: the Zephyr pipeline closures (map_shard writers) are cloudpickled to the
coordinator job, and cloudpickle references module-level functions *by import path*. If a
stage were launched with ``python -m ...fast_stage`` it would be ``__main__`` and its
functions would pickle as ``__main__.<name>`` -- unimportable on the coordinator. Importing
the stage module here keeps every referenced function under its real module path.
"""

import argparse
import logging
import os

# The fast stage tokenizes in a fork process pool (one core per proc), so the tokenizers-lib
# internal rayon parallelism is disabled to avoid oversubscription. Set it before any import
# that pulls the tokenizers lib.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from experiments.datakit.cluster.quality.fast_transformer.tpu_bench import fast_stage, fasttext_stage
from experiments.datakit.cluster.quality.fast_transformer.tpu_bench.common import MODEL_CALIB


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["fast", "fasttext"])
    p.add_argument("--corpus", default=None, help="text parquet glob")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-files", type=int, default=24)
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--device-batch", type=int, default=4096)
    p.add_argument("--tok-procs", type=int, default=96, help="fork processes for tokenization (off the GIL)")
    p.add_argument("--read-threads", type=int, default=12, help="host threads for row-group parquet reads")
    p.add_argument("--cpu", type=int, default=8, help="vCPUs per worker for the fasttext stage")
    p.add_argument("--calib-file", default=MODEL_CALIB)
    p.add_argument(
        "--fasttext-model", default="gs://marin-eu-west4/datakit/llm-quality-classifier/model/sonnet46-thr05/model.bin"
    )
    p.add_argument("--result-json", default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.stage == "fast":
        fast_stage.run(
            corpus_glob=args.corpus,
            model_dir=args.model_dir,
            output_path=args.out_dir,
            max_files=args.max_files,
            max_workers=args.max_workers,
            device_batch=args.device_batch,
            tok_procs=args.tok_procs,
            read_threads=args.read_threads,
            calib_file=args.calib_file,
            result_json=args.result_json,
        )
    elif args.stage == "fasttext":
        fasttext_stage.run(
            corpus_glob=args.corpus,
            model_path=args.fasttext_model,
            output_path=args.out_dir,
            max_files=args.max_files,
            max_workers=args.max_workers,
            cpu=args.cpu,
            result_json=args.result_json,
        )


if __name__ == "__main__":
    main()
