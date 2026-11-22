# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score (left, right) pairs with a trained classifier on one GPU.

Reads every ``pairs/*.parquet`` under ``--pairs``, builds ``left`` from the item (question,
then answer) and scores each pair with the model saved under ``--model``. Writes one parquet
with a ``score_<head>`` column per output and logs the rate.
"""

import argparse
import io
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from experiments.datakit.decontam.encoder.train import RIGHT_CHARS, pair_left, text_normalize

logger = logging.getLogger(__name__)


def fetch_model(prefix: str) -> str:
    local = Path("/tmp/model")
    local.mkdir(exist_ok=True)
    for path in StoragePath(f"{prefix}/*").glob():
        with path.open("rb") as source:
            (local / str(path).rsplit("/", 1)[1]).write_bytes(source.read())
    return str(local)


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="storage prefix of the saved model")
    parser.add_argument("--heads", nargs="+", required=True, help="names of the model outputs, in order")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    frames = []
    for path in StoragePath(f"{args.pairs}/pairs/*.parquet").glob():
        with path.open("rb") as source:
            frames.append(pq.read_table(io.BytesIO(source.read())).to_pandas())
    pairs = pd.concat(frames, ignore_index=True)
    rights = [text_normalize(text)[:RIGHT_CHARS] for text in pairs["right"]]
    lefts = [
        pair_left(question, list(choices), answer, right)
        for question, choices, answer, right in zip(
            pairs["question"], pairs["choices"], pairs["answer"], rights, strict=True
        )
    ]
    directory = fetch_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForSequenceClassification.from_pretrained(directory).to("cuda").eval()
    order = np.argsort([len(text) for text in rights])
    scores = np.zeros((len(pairs), len(args.heads)))
    started = time.time()
    with torch.no_grad():
        for start in range(0, len(order), 64):
            group = order[start : start + 64]
            encoded = tokenizer(
                [lefts[i] for i in group],
                [rights[i] for i in group],
                truncation="longest_first",
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**encoded).logits.float()
            scores[group] = torch.sigmoid(logits).cpu().numpy()
    seconds = time.time() - started
    logger.info("scored %d pairs in %.0f s: %.0f pairs/s", len(pairs), seconds, len(pairs) / seconds)
    for number, head in enumerate(args.heads):
        pairs[f"score_{head}"] = scores[:, number]
    buffer = io.BytesIO()
    pairs.to_parquet(buffer)
    with StoragePath(args.output).open("wb") as sink:
        sink.write(buffer.getvalue())


if __name__ == "__main__":
    main()
