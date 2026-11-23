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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from experiments.datakit.decontam.encoder.score import fetch_model
from experiments.datakit.decontam.encoder.tables import item_texts, parquet_files, read_frame, write_frame
from experiments.datakit.decontam.encoder.train import OUTPUTS, batches, with_pair_text

logger = logging.getLogger(__name__)


# Pair files scored together: one group is read and prepared while the GPU scores the group before it.
GROUP_FILES = 16
BATCH = 64


def _prepared(paths: list[str], output: str, texts: dict, pool: ThreadPoolExecutor) -> tuple[list[str], pd.DataFrame]:
    """The pair files of *paths* that have no scores yet, and their pairs with the pair text, in file order."""
    targets = [prefix_join(output, f"scores/{Path(path).name}") for path in paths]
    todo = [
        path for path, done in zip(paths, pool.map(lambda t: StoragePath(t).exists(), targets), strict=True) if not done
    ]
    if not todo:
        return [], pd.DataFrame()
    frames = list(pool.map(read_frame, todo))
    for number, frame in enumerate(frames):
        frame["file"] = number
    return todo, with_pair_text(pd.concat(frames, ignore_index=True), texts)


@torch.no_grad()
def _scores(models: list, tokenizer, pairs: pd.DataFrame, max_length: int) -> np.ndarray:
    """The question score of each model for each pair; each batch is tokenized once for all models."""
    for model in models:
        model.eval()
    head = OUTPUTS.index("label_q")
    scores = np.zeros((len(pairs), len(models)))
    for group, encoded in batches(pairs, tokenizer, BATCH, max_length, False):
        inputs = {key: value.to("cuda") for key, value in encoded.items()}
        for number, model in enumerate(models):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**inputs).logits.float()
            scores[group, number] = torch.sigmoid(logits[:, head]).cpu().numpy()
    return scores


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
    models, tokenizers = [], []
    for prefix in args.models:
        with tempfile.TemporaryDirectory() as directory:
            fetch_model(prefix, directory)
            models.append(AutoModelForSequenceClassification.from_pretrained(directory).to("cuda"))
            tokenizers.append(AutoTokenizer.from_pretrained(directory))
    assert all(t.get_vocab() == tokenizers[0].get_vocab() for t in tokenizers), "the models must share a tokenizer"
    texts = item_texts(args.items)
    groups = [files[i : i + GROUP_FILES] for i in range(0, len(files), GROUP_FILES)]
    started, scored = time.time(), 0
    with ThreadPoolExecutor(1) as prepare, ThreadPoolExecutor(GROUP_FILES) as io:
        writes = []
        upcoming = prepare.submit(_prepared, groups[0], args.output, texts, io) if groups else None
        for number in range(len(groups)):
            todo, pairs = upcoming.result()
            if number + 1 < len(groups):
                upcoming = prepare.submit(_prepared, groups[number + 1], args.output, texts, io)
            if not todo:
                continue
            scores = _scores(models, tokenizers[0], pairs, args.max_length)
            frame = pairs[["path", "record", "item", "eval", "record_chars", "file"]].copy()
            for model_number in range(len(models)):
                frame[f"score_{model_number}"] = scores[:, model_number]
            frame["score"] = scores.mean(axis=1)
            for file_number, part in frame.groupby("file"):
                target = prefix_join(args.output, f"scores/{Path(todo[file_number]).name}")
                writes.append(io.submit(write_frame, part.drop(columns="file"), target))
            scored += len(frame)
        for write in writes:
            write.result()
    seconds = time.time() - started
    logger.info(
        "part %d of %d: %d files, %d pairs in %.0f s (%.0f pairs/s)",
        args.part,
        args.parts,
        len(files),
        scored,
        seconds,
        scored / max(seconds, 1),
    )


if __name__ == "__main__":
    main()
