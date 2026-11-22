# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune a pretrained multilingual encoder to say whether `right` contains `left`.

`left` is an eval item and `right` is record text (see ``pair_left``). The model has two outputs
(``OUTPUTS``): the record holds the question, and the record holds the question with its answer.
Results (metrics, test scores, model) go to ``--output``.
"""

import argparse
import json
import math
import random
import re
import tempfile
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rigging.filesystem.storage_path import StoragePath, prefix_join
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from experiments.datakit.decontam.encoder.labels import TRAINING_COLUMNS
from experiments.datakit.decontam.encoder.tables import item_passages, read_frame, write_frame

_SPACE = re.compile(r"\s+")
TARGET_PRECISION = 0.95
# The model's outputs, in order: the record holds the question; it holds the question with its answer.
OUTPUTS = ("label_q", "label_qa")
# One in this many train items of the real label sets goes to the dev split.
DEV_EVERY = 4
# Judged pairs of the earlier block-rule candidates, in --data.
JUDGED_PAIRS = "judged_pairs_v2.parquet"


def text_normalize(text: str) -> str:
    """Light cleanup for both sides: Unicode compatibility forms folded, whitespace runs collapsed."""
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


_TOKEN = re.compile(r"\w+")
# Most missing words that the hint lists.
MISSING_SHOWN = 32
LEFT_CHARS = 1500
RIGHT_CHARS = 6000
# Characters of a passage that `left` shows; 9 in 10 passages of the required evals are shorter.
PASSAGE_CHARS = 700


def _missing(text: str, present: set[str]) -> str:
    missing = list(dict.fromkeys(token for token in _TOKEN.findall(text.lower()) if token not in present))
    return " ".join(missing[:MISSING_SHOWN]) if missing else "none"


def pair_left(question: str, choices: list[str], answer: str, passage: str, right: str) -> str:
    """The `left` text of a pair: the item in marked sections, then the words of each section that
    *right* lacks.

    The sections are ``[passage]`` (only for an item with a passage, cut to ``PASSAGE_CHARS``),
    ``[question]``, ``[choices]`` and ``[answer]``, normalized and cut to ``LEFT_CHARS``. The hints
    list the missing passage words (``[passage missing]``), question and choice words
    (``[missing]``) and answer words (``[answer missing]``) separately: an item with a passage is
    held only by a record with both its passage and its question, and an answer that the record
    does not show does not count against the question.
    """
    answer, passage = answer or "", text_normalize(passage or "")[:PASSAGE_CHARS]
    sections = [f"[passage] {passage}"] if passage else []
    sections.append(f"[question] {text_normalize(question)}")
    if choices:
        sections.append("[choices] " + " | ".join(map(text_normalize, choices)))
    if answer:
        sections.append(f"[answer] {text_normalize(answer)}")
    left = "\n".join(sections)[:LEFT_CHARS]
    present = set(_TOKEN.findall(right.lower()))
    hints = [f"[passage missing] {_missing(passage, present)}"] if passage else []
    hints.append(f"[missing] {_missing(' '.join([question, *choices]), present)}")
    hints.append(f"[answer missing] {_missing(answer, present) if answer else 'none'}")
    return left + "\n" + "\n".join(hints)


def with_pair_text(frame: pd.DataFrame, passages: dict[str, str]) -> pd.DataFrame:
    """*frame* with `right` normalized and cut to ``RIGHT_CHARS``, and `left` built by ``pair_left``
    with each item's passage from *passages* (see ``item_passages``)."""
    frame = frame.copy()
    frame["right"] = frame["right"].map(text_normalize).str.slice(0, RIGHT_CHARS)
    frame["left"] = [
        pair_left(question, list(choices), answer, passages.get(item, ""), right)
        for question, choices, answer, item, right in zip(
            frame["question"], frame["choices"], frame["answer"], frame["item"], frame["right"], strict=True
        )
    ]
    return frame


def load(
    data: str, synthetic_name: str, extra: list[str], passages: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    judged = read_frame(prefix_join(data, JUDGED_PAIRS))
    judged["source"], judged["kind"], judged["eval"] = "judged", judged["judges"], ""
    synthetic = read_frame(prefix_join(data, synthetic_name))
    synthetic["source"] = "synthetic"
    frames = [judged[TRAINING_COLUMNS], synthetic[TRAINING_COLUMNS]]
    for path in extra:
        # More labeled pairs, for example judged stage-1 candidates, with the same columns.
        frames.append(read_frame(path)[TRAINING_COLUMNS])
    frame = with_pair_text(pd.concat(frames, ignore_index=True), passages)
    # The threshold comes from judge-labeled stage-1 candidates, not synthetic, silver, or older pairs.
    real = frame["source"].str.startswith("set")
    real_train = sorted(frame.loc[(frame.split == "train") & real, "item"].unique())
    dev_items = set(random.Random(0).sample(real_train, len(real_train) // DEV_EVERY))
    dev = frame[real & frame["item"].isin(dev_items)]
    train = frame[(frame.split == "train") & ~frame.index.isin(dev.index)]
    return train, dev, frame[frame.split == "test"]


def batches(frame: pd.DataFrame, tokenizer, size: int, max_length: int, shuffle: bool):
    """Row positions and encoded (left, right) pairs, in batches of records of similar length."""
    order = list(range(len(frame)))
    if shuffle:
        random.shuffle(order)
    lengths = frame["right"].str.len().to_numpy()
    chunks = [sorted(order[i : i + size * 50], key=lambda k: lengths[k]) for i in range(0, len(order), size * 50)]
    groups = [chunk[i : i + size] for chunk in chunks for i in range(0, len(chunk), size)]
    if shuffle:
        random.shuffle(groups)
    for group in groups:
        rows = frame.iloc[group]
        encoded = tokenizer(
            list(rows["left"]),
            list(rows["right"]),
            truncation="longest_first",
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        yield group, encoded


@torch.no_grad()
def predict(model, tokenizer, frame: pd.DataFrame, max_length: int, device: str) -> np.ndarray:
    """Sigmoid scores of every output of *model* for every pair of *frame* (see ``with_pair_text``)."""
    was_training = model.training
    model.eval()
    scores = np.zeros((len(frame), model.config.num_labels))
    for group, encoded in batches(frame, tokenizer, 64, max_length, False):
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(**{k: v.to(device) for k, v in encoded.items()}).logits.float()
        scores[group] = torch.sigmoid(logits).cpu().numpy()
    model.train(was_training)
    return scores


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = labels == 1
    if positives.all() or not positives.any():
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):
        tied = scores == value
        ranks[tied] = ranks[tied].mean()
    n_pos, n_neg = positives.sum(), (~positives).sum()
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def threshold_for(scores: np.ndarray, labels: np.ndarray) -> float:
    """The lowest threshold meeting target precision, or 1.0 if none meets it."""
    best = 1.0
    for value in np.unique(scores)[::-1]:
        if labels[scores >= value].mean() >= TARGET_PRECISION:
            best = float(value)
    return best


def metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    marked = scores >= threshold
    best = threshold_for(scores, labels) if labels.sum() else 1.0
    return {
        "recall_at_target_on_this_set": float((scores[labels == 1] >= best).mean()) if labels.sum() else None,
        "n": len(labels),
        "positives": int(labels.sum()),
        "auc": roc_auc(labels, scores),
        "precision": float(labels[marked].mean()) if marked.any() else None,
        "recall": float(marked[labels == 1].mean()) if labels.sum() else None,
        "marked": int(marked.sum()),
    }


def evaluation(dev: pd.DataFrame, dev_scores: np.ndarray, test: pd.DataFrame, test_scores: np.ndarray) -> dict:
    """Per output: the dev threshold, and metrics on dev, on judged and synthetic test pairs, and per test source."""
    results: dict = {}
    for head, column in enumerate(OUTPUTS):
        threshold = threshold_for(dev_scores[:, head], dev[column].to_numpy())
        results[column] = {
            "threshold": threshold,
            "dev": metrics(dev_scores[:, head], dev[column].to_numpy(), threshold),
        }
        for name, part in (("judged", test.source != "synthetic"), ("synthetic", test.source == "synthetic")):
            results[column][name] = metrics(
                test_scores[part.to_numpy(), head], test.loc[part, column].to_numpy(), threshold
            )
        for source in sorted(set(test["source"]) - {"synthetic", "judged"}):
            part = (test["source"] == source).to_numpy()
            results[column][source] = metrics(test_scores[part, head], test.loc[part, column].to_numpy(), threshold)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="pretrained encoder, for example jhu-clsp/mmBERT-base")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--data", required=True, help="directory with judged_pairs_v2.parquet and the synthetic pairs")
    parser.add_argument("--synthetic", required=True, help="file name in --data of the generic synthetic pairs")
    parser.add_argument("--extra", action="append", default=[], help="parquet of more labeled pairs")
    parser.add_argument("--items", required=True, help="the item table directory (items/*.parquet), for passages")
    parser.add_argument("--seed", type=int, default=0, help="seed of the training order and the output layer")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # The cuDNN attention backend builds no plan for these padded bf16 batches; the others do.
        torch.backends.cuda.enable_cudnn_sdp(False)
    train, dev, test = load(args.data, args.synthetic, args.extra, item_passages(args.items))
    print("train", len(train), "dev", len(dev), "test", len(test), "device", device, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(OUTPUTS)).to(device)
    steps = math.ceil(len(train) / args.batch) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    schedule = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), int(steps))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    step, started = 0, time.time()
    while step < steps:
        for group, encoded in batches(train, tokenizer, args.batch, args.max_length, True):
            labels = torch.tensor(train.iloc[group][list(OUTPUTS)].to_numpy(), dtype=torch.float32)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(**{k: v.to(device) for k, v in encoded.items()}).logits
            loss = loss_fn(logits.float(), labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
            step += 1
            if step % 100 == 0 or step == 1:
                print(
                    f"step {step}/{int(steps)} loss {loss.item():.4f} {(time.time() - started) / step:.3f}s/step",
                    flush=True,
                )
            if step >= steps:
                break

    started = time.time()
    dev_scores = predict(model, tokenizer, dev, args.max_length, device)
    test_scores = predict(model, tokenizer, test, args.max_length, device)
    rate = (len(dev) + len(test)) / (time.time() - started)
    results: dict = {
        "seed": args.seed,
        "model": args.model,
        "max_length": args.max_length,
        "pairs_per_second": rate,
    }
    results.update(evaluation(dev, dev_scores, test, test_scores))
    for head, column in enumerate(OUTPUTS):
        test[f"score_{column}"] = test_scores[:, head]
    print(json.dumps(results, indent=1), flush=True)
    run = prefix_join(args.output, args.tag)
    write_frame(test, prefix_join(run, "test_scores.parquet"))
    StoragePath(prefix_join(run, "results.json")).write_text(json.dumps(results, indent=1))
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        tokenizer.save_pretrained(directory)
        for path in Path(directory).iterdir():
            StoragePath(prefix_join(run, f"model/{path.name}")).write_bytes(path.read_bytes())


if __name__ == "__main__":
    main()
