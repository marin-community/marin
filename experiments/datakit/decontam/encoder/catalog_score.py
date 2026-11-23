# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the catalog run's pairs (``catalog.py scan``) on one GPU, for one part of the pair files.

Each pair file ``pairs/<shard>.parquet`` gets ``scores/<shard>.parquet`` with the question score of every
model and their mean (``score``). A file that has scores is skipped, so a restarted part resumes. Parts
take every ``--parts``-th file, so the parts can run as separate jobs.
"""

import argparse
import logging
import tempfile
import time
from pathlib import Path

import torch
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from experiments.datakit.decontam.encoder.score import fetch_model
from experiments.datakit.decontam.encoder.tables import item_texts, parquet_files, read_frame, write_frame
from experiments.datakit.decontam.encoder.train import OUTPUTS, predict, with_pair_text

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="the catalog run's output, with pairs/*.parquet")
    parser.add_argument("--models", nargs="+", required=True, help="storage prefixes of the saved models")
    parser.add_argument("--items", required=True, help="the item table directory (items/*.parquet), for item texts")
    parser.add_argument("--part", type=int, required=True)
    parser.add_argument("--parts", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    files = parquet_files(prefix_join(args.output, "pairs"))[args.part :: args.parts]
    models = []
    for prefix in args.models:
        with tempfile.TemporaryDirectory() as directory:
            fetch_model(prefix, directory)
            model = AutoModelForSequenceClassification.from_pretrained(directory).to("cuda")
            models.append((AutoTokenizer.from_pretrained(directory), model))
    texts = item_texts(args.items)
    head = OUTPUTS.index("label_q")
    started, scored = time.time(), 0
    for path in files:
        target = prefix_join(args.output, f"scores/{Path(path).name}")
        if StoragePath(target).exists():
            continue
        pairs = with_pair_text(read_frame(path), texts)
        frame = pairs[["path", "record", "item", "eval", "record_chars"]].copy()
        for number, (tokenizer, model) in enumerate(models):
            frame[f"score_{number}"] = predict(model, tokenizer, pairs, args.max_length, "cuda")[:, head]
        frame["score"] = frame[[f"score_{number}" for number in range(len(models))]].mean(axis=1)
        write_frame(frame, target)
        scored += len(frame)
    seconds = time.time() - started
    logger.info("part %d of %d: %d files, %d pairs in %.0f s", args.part, args.parts, len(files), scored, seconds)


if __name__ == "__main__":
    main()
