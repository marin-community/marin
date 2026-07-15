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

# The marin worker image ships TOKENIZERS_PARALLELISM=false, which pins the HF fast
# tokenizer to a single core -- fatal here, since tokenization is the throughput bound.
# Force it on before any transformers import so the tokenizer's rayon pool uses the host.
os.environ["TOKENIZERS_PARALLELISM"] = "true"

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
    p.add_argument(
        "--tok-procs", type=int, default=0, help="tokenizer processes on the v6e host (0/1 = main-thread rayon)"
    )
    p.add_argument("--cpu", type=int, default=8)
    p.add_argument(
        "--cpu-only", action="store_true", help="fast stage: no TPU, forward on host CPUs (the 'before' baseline)"
    )
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
            calib_file=args.calib_file,
            result_json=args.result_json,
            cpu_only=args.cpu_only,
            cpu=args.cpu,
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
