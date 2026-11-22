# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune a pretrained multilingual encoder to say whether `right` contains `left`.

`left` is an eval item and `right` is record text (see ``pair_left``). With ``--heads 2`` the
model has two outputs: the record holds the question, and the record holds the question with
its answer. With ``--heads 1`` it has one output: the judges' "contaminated" label.
Results (metrics, test scores, model) go to ``--output``.
"""

import argparse
import io
import json
import math
import random
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rigging.filesystem.storage_path import StoragePath
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from experiments.datakit.decontam.encoder.labels import TRAINING_COLUMNS

_SPACE = re.compile(r"\s+")
TARGET_PRECISION = 0.95
# One in this many train items of the real label sets goes to the dev split.
DEV_EVERY = 4


def text_normalize(text: str) -> str:
    """Light cleanup for both sides: Unicode compatibility forms folded, whitespace runs collapsed."""
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


_TOKEN = re.compile(r"\w+")
# Most missing words that the hint lists.
MISSING_SHOWN = 32
LEFT_CHARS = 1500
RIGHT_CHARS = 6000


def _missing(text: str, present: set[str]) -> str:
    missing = list(dict.fromkeys(token for token in _TOKEN.findall(text.lower()) if token not in present))
    return " ".join(missing[:MISSING_SHOWN]) if missing else "none"


def pair_left(question: str, choices: list[str], answer: str, right: str) -> str:
    """The `left` text of a pair: the item, then the words of the item that *right* lacks.

    The item is the question, a ``[choices]`` line and an ``[answer]`` line, normalized and cut to
    ``LEFT_CHARS``. The hint lists the missing question and choice words on a ``[missing]`` line and
    the missing answer words on an ``[answer missing]`` line, so that an answer the record does not
    show does not count against the question.
    """
    answer = answer or ""
    item = text_normalize(question) + ("\n[choices] " + " | ".join(map(text_normalize, choices)) if choices else "")
    left = (item + ("\n[answer] " + text_normalize(answer) if answer else ""))[:LEFT_CHARS]
    present = set(_TOKEN.findall(right.lower()))
    answer_missing = _missing(answer, present) if answer else "none"
    return f"{left}\n[missing] {_missing(' '.join([question, *choices]), present)}\n[answer missing] {answer_missing}"


def read_parquet(path: str) -> pd.DataFrame:
    if path.startswith("s3://"):
        with StoragePath(path).open("rb") as source:
            return pd.read_parquet(io.BytesIO(source.read()))
    return pd.read_parquet(path)


def load(data: str, synthetic_name: str, extra: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    judged = read_parquet(f"{data}/judged_pairs_v2.parquet")
    judged["source"], judged["kind"], judged["eval"] = "judged", judged["judges"], ""
    synthetic = read_parquet(f"{data}/{synthetic_name}")
    synthetic["source"] = "synthetic"
    frames = [judged[TRAINING_COLUMNS], synthetic[TRAINING_COLUMNS]]
    for path in extra:
        # More labeled pairs, for example judged stage-1 candidates, with the same columns.
        frames.append(read_parquet(path)[TRAINING_COLUMNS])
    frame = pd.concat(frames, ignore_index=True)
    frame["right"] = frame["right"].map(text_normalize).str.slice(0, RIGHT_CHARS)
    frame["left"] = [
        pair_left(question, list(choices), answer, right)
        for question, choices, answer, right in zip(
            frame["question"], frame["choices"], frame["answer"], frame["right"], strict=True
        )
    ]
    # The threshold comes from judge-labeled real stage-1 candidates (the label sets), whose share of
    # copies matches what stage 2 sees; never from silver, synthetic or the older judged pairs.
    real = frame["source"].str.startswith("set")
    real_train = sorted(frame.loc[(frame.split == "train") & real, "item"].unique())
    dev_items = set(random.Random(0).sample(real_train, len(real_train) // DEV_EVERY))
    dev = frame[real & frame["item"].isin(dev_items)]
    train = frame[(frame.split == "train") & ~frame.index.isin(dev.index)]
    return train, dev, frame[frame.split == "test"]


def batches(frame: pd.DataFrame, tokenizer, size: int, max_length: int, shuffle: bool, label_columns: list[str]):
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
        yield group, encoded, torch.tensor(rows[label_columns].to_numpy(), dtype=torch.float32)


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
    """The lowest threshold whose marks keep the target precision on this data."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="pretrained encoder, for example jhu-clsp/mmBERT-base")
    parser.add_argument("--heads", type=int, choices=[1, 2], default=2)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--data", required=True, help="directory with judged_pairs_v2.parquet and the synthetic pairs")
    parser.add_argument("--synthetic", required=True, help="file name in --data of the generic synthetic pairs")
    parser.add_argument("--extra", action="append", default=[], help="parquet of more labeled pairs")
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
    label_columns = ["label_q", "label_qa"] if args.heads == 2 else ["label"]
    train, dev, test = load(args.data, args.synthetic, args.extra)
    print("train", len(train), "dev", len(dev), "test", len(test), "device", device, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(label_columns)).to(device)
    steps = math.ceil(len(train) / args.batch) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    schedule = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), int(steps))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    step, started = 0, time.time()
    while step < steps:
        for _, encoded, labels in batches(train, tokenizer, args.batch, args.max_length, True, label_columns):
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

    @torch.no_grad()
    def predict(frame: pd.DataFrame) -> np.ndarray:
        model.eval()
        scores = np.zeros((len(frame), len(label_columns)))
        for group, encoded, _ in batches(frame, tokenizer, 64, args.max_length, False, label_columns):
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(**{k: v.to(device) for k, v in encoded.items()}).logits.float()
            scores[group] = torch.sigmoid(logits).cpu().numpy()
        model.train()
        return scores

    started = time.time()
    dev_scores, test_scores = predict(dev), predict(test)
    rate = (len(dev) + len(test)) / (time.time() - started)
    results: dict = {
        "seed": args.seed,
        "heads": args.heads,
        "model": args.model,
        "max_length": args.max_length,
        "pairs_per_second": rate,
    }
    for head, column in enumerate(label_columns):
        threshold = threshold_for(dev_scores[:, head], dev[column].to_numpy())
        results[column] = {
            "threshold": threshold,
            "dev": metrics(dev_scores[:, head], dev[column].to_numpy(), threshold),
        }
        for name, part in (("judged", test.source != "synthetic"), ("synthetic", test.source == "synthetic")):
            results[column][name] = metrics(
                test_scores[part.to_numpy(), head], test.loc[part, column].to_numpy(), threshold
            )
            # Every head is also scored against the judges' "contaminated" label.
            results[column][f"{name}_vs_label"] = metrics(
                test_scores[part.to_numpy(), head], test.loc[part, "label"].to_numpy(), threshold
            )
        for source in sorted(set(test["source"]) - {"synthetic", "judged"}):
            part = (test["source"] == source).to_numpy()
            results[column][source] = metrics(test_scores[part, head], test.loc[part, column].to_numpy(), threshold)
        test[f"score_{column}"] = test_scores[:, head]
    print(json.dumps(results, indent=1), flush=True)
    buffer = io.BytesIO()
    test.to_parquet(buffer)
    with StoragePath(f"{args.output}/{args.tag}/test_scores.parquet").open("wb") as sink:
        sink.write(buffer.getvalue())
    with StoragePath(f"{args.output}/{args.tag}/results.json").open("w") as sink:
        json.dump(results, sink, indent=1)
    model.save_pretrained(f"/tmp/model_{args.tag}")
    tokenizer.save_pretrained(f"/tmp/model_{args.tag}")
    for path in Path(f"/tmp/model_{args.tag}").iterdir():
        with StoragePath(f"{args.output}/{args.tag}/model/{path.name}").open("wb") as sink:
            sink.write(path.read_bytes())


if __name__ == "__main__":
    main()
