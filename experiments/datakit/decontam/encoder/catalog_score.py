# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the catalog run's pairs (``catalog.py scan``) on one GPU, for one part of the pair files.

Each pair file ``pairs/<shard>.parquet`` gets ``scores/<shard>.parquet`` with the question score of every
model and their mean (``score``). A file that has scores is skipped, so a restarted part resumes. Parts
take every ``--parts``-th file, so the parts can run as separate jobs.
"""

import argparse
import json
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
# Later models score only the pairs that the first model scores at least this high; the score of the others is
# the first model's. On 1,673,046 scored pairs, 96.7% score below it, and 6 of 14,075 pairs with a two-model mean
# of at least 0.5 had a first score below it (both models' scores near 0 and 1).
LATER_MODELS_FLOOR = 0.01


def _prepared(
    paths: list[str], scores_dir: str, texts: dict, tokenizer, max_length: int, pool: ThreadPoolExecutor
) -> tuple[list[str], pd.DataFrame, list]:
    """The pair files of *paths* that have no scores yet, their pairs with the pair text in file order, and the
    pairs' tokenized batches (row positions, encoding)."""
    targets = [prefix_join(scores_dir, Path(path).name) for path in paths]
    todo = [
        path for path, done in zip(paths, pool.map(lambda t: StoragePath(t).exists(), targets), strict=True) if not done
    ]
    if not todo:
        return [], pd.DataFrame(), []
    frames = list(pool.map(read_frame, todo))
    for number, frame in enumerate(frames):
        frame["file"] = number
    pairs = with_pair_text(pd.concat(frames, ignore_index=True), texts)
    return todo, pairs, list(batches(pairs, tokenizer, BATCH, max_length, False))


@torch.no_grad()
def _scores(models: list, encoded_batches: list, rows: int) -> np.ndarray:
    """The question score of each model for each pair (NaN where a later model is skipped, see
    ``LATER_MODELS_FLOOR``); each batch goes to the GPU once, and the scores come back once per group."""
    for model in models:
        model.eval()
    head = OUTPUTS.index("label_q")
    parts = []
    for group, encoded in encoded_batches:
        inputs = {key: value.to("cuda", non_blocking=True) for key, value in encoded.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            first = torch.sigmoid(models[0](**inputs).logits[:, head].float())
            later = torch.full((len(group), len(models) - 1), float("nan"), device="cuda")
            keep = (first >= LATER_MODELS_FLOOR).nonzero(as_tuple=True)[0]
            if len(keep):
                subset = {key: value[keep] for key, value in inputs.items()}
                for number, model in enumerate(models[1:]):
                    later[keep, number] = torch.sigmoid(model(**subset).logits[:, head].float())
        parts.append((group, torch.cat([first[:, None], later], dim=1)))
    scores = np.zeros((rows, len(models)))
    for group, values in parts:
        scores[group] = values.cpu().numpy()
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
    parser.add_argument("--scores", help="where the scores go; default <output>/scores")
    parser.add_argument("--files", help="JSON list of pair file names under <output>/pairs to score, in place of all")
    args = parser.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    scores_dir = args.scores or prefix_join(args.output, "scores")
    if args.files:
        names = json.loads(StoragePath(args.files).read_text())
        files = [prefix_join(args.output, f"pairs/{name}") for name in names][args.part :: args.parts]
    else:
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

        def submit(group):
            return prepare.submit(_prepared, group, scores_dir, texts, tokenizers[0], args.max_length, io)

        upcoming = submit(groups[0]) if groups else None
        for number in range(len(groups)):
            todo, pairs, encoded = upcoming.result()
            if number + 1 < len(groups):
                upcoming = submit(groups[number + 1])
            if not todo:
                continue
            scores = _scores(models, encoded, len(pairs))
            frame = pairs[["path", "record", "item", "eval", "record_chars", "file"]].copy()
            for model_number in range(len(models)):
                frame[f"score_{model_number}"] = scores[:, model_number]
            frame["score"] = np.nanmean(scores, axis=1)
            for file_number, part in frame.groupby("file"):
                target = prefix_join(scores_dir, Path(todo[file_number]).name)
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
