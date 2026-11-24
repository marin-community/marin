# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score labeled or sampled pairs with one trained classifier on one GPU, for the research evaluations.

Reads every ``pairs/*.parquet`` under ``--pairs``, builds each pair's text as the pipeline does
(``pair_text.with_pair_text``), and scores it with the model saved under ``--model``. Writes one parquet with a
``score_<head>`` column per output and logs the rate. The pipeline itself scores with
``encoder_models.EncoderModels``.
"""

import argparse
import logging
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch
from marin.datakit.decon_tables import parquet_files, read_frame, write_frame
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from experiments.datakit.decontam.encoder.train import predict
from experiments.datakit.decontam.pair_text import OUTPUTS, item_texts, with_pair_text

logger = logging.getLogger(__name__)


def fetch_model(prefix: str, directory: str) -> None:
    for path in StoragePath(prefix_join(prefix, "*")).glob():
        (Path(directory) / path.name).write_bytes(path.read_bytes())


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="storage prefix of the saved model")
    parser.add_argument("--pairs", required=True, help="pairs output with pairs/*.parquet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--items", required=True, help="the item table directory (items/*.parquet), for item texts")
    args = parser.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    pairs = pd.concat([read_frame(path) for path in parquet_files(prefix_join(args.pairs, "pairs"))], ignore_index=True)
    with tempfile.TemporaryDirectory() as directory:
        fetch_model(args.model, directory)
        tokenizer = AutoTokenizer.from_pretrained(directory)
        model = AutoModelForSequenceClassification.from_pretrained(directory).to("cuda")
    assert model.config.num_labels == len(OUTPUTS), "the model must have the trainer's outputs"
    started = time.time()
    scores = predict(model, tokenizer, with_pair_text(pairs, item_texts(args.items)), args.max_length, "cuda")
    seconds = time.time() - started
    logger.info("scored %d pairs in %.0f s: %.0f pairs/s", len(pairs), seconds, len(pairs) / seconds)
    for number, head in enumerate(OUTPUTS):
        pairs[f"score_{head}"] = scores[:, number]
    write_frame(pairs, args.output)


if __name__ == "__main__":
    main()
