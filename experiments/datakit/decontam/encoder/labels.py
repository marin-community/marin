# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Judge label sets from real stage-1 candidates, and training rows from two judges' labels.

``sample``: choose pairs from scored candidates and write judge batches. Most pairs are taken
where the record holds at least 90% of the question's words, because copies are rare and all
judged copies so far were there; the rest are spread over the classifier's score range. Both
groups are spread over eval families.

``silver``: silver negatives, real candidates whose record holds less than 80% of the
question's words. About 1% of judged pairs in that range were copies (5 of 561 in sets 11
and 12), so these labels are noisy but cheap.

``merge``: join two judges' labels (see RUBRIC.md). A pair is contaminated when both judges say
``COPY`` or ``EDITED``, and clean when both say ``VARIANT``, ``GENERIC`` or ``UNRELATED``; other
pairs are dropped. ``label_qa`` is 1 when both judges say the answer is shown. Items are split
60/40 into train and test, so no test item is seen in training.
"""

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from marin.datakit.decon_tables import read_frame, write_frame
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)
EXCERPT = 3000
HIGH_OVERLAP = 0.9
SCORE_BINS = (0, 0.3, 0.9, 1.01)
CONTAMINATED = frozenset({"COPY", "EDITED"})
CLEAN = frozenset({"VARIANT", "GENERIC", "UNRELATED"})
# Every label a judge may give, in the order RUBRIC.md lists them; UNSURE is neither contaminated nor clean.
LABELS = ("COPY", "EDITED", "VARIANT", "GENERIC", "UNRELATED", "UNSURE")
TEST_SHARE = 0.4
SILVER_OVERLAP = 0.8
# The two LLM judges that judge.py runs; a pair is labeled when both agree.
JUDGE_MODELS = ("glm-5.3", "deepseek-v4.1-flash")
_WORD = re.compile(r"\w+")


def word_overlap(question: str, right: str) -> float:
    """The share of the question's words that appear anywhere in *right*."""
    words = _WORD.findall(question.lower())
    present = set(_WORD.findall(right.lower()))
    return float(np.mean([word in present for word in words])) if words else 0.0


def _round_robin(part: pd.DataFrame, count: int, seed: int) -> list:
    """Up to *count* row labels of *part*, taken in turn from each eval family."""
    rng = np.random.default_rng(seed)
    groups = [group.sample(frac=1, random_state=int(rng.integers(1 << 30))) for _, group in part.groupby("family")]
    taken, index = [], 0
    while len(taken) < min(count, len(part)):
        for group in groups:
            if index < len(group) and len(taken) < count:
                taken.append(group.index[index])
        index += 1
    return taken


def sample(pairs: pd.DataFrame, high: int, other: int) -> pd.DataFrame:
    pairs = pairs.reset_index(drop=True)
    pairs["family"] = pairs["eval"].str.split(r"[_ -]").str[0]
    pairs["overlap"] = [word_overlap(q, r) for q, r in zip(pairs["question"], pairs["right"], strict=True)]
    picked = _round_robin(pairs[pairs["overlap"] >= HIGH_OVERLAP], high, 1)
    low = pairs[pairs["overlap"] < HIGH_OVERLAP]
    bins = pd.cut(low["score_label_q"], SCORE_BINS, right=False, labels=False)
    for number, (_, part) in enumerate(low.groupby(bins)):
        picked += _round_robin(part, other // (len(SCORE_BINS) - 1), 10 + number)
    chosen = pairs.loc[picked].sample(frac=1, random_state=4).reset_index(drop=True)
    chosen["stratum"] = np.where(chosen["overlap"] >= HIGH_OVERLAP, "high", "other")
    chosen["uid"] = [
        hashlib.sha1(f"{record}|{item}".encode()).hexdigest()[:12]
        for record, item in zip(chosen["record"], chosen["item"], strict=True)
    ]
    return chosen


def write_batches(chosen: pd.DataFrame, directory: Path, batches: int) -> None:
    (directory / "batches").mkdir(parents=True, exist_ok=True)
    chosen.to_parquet(directory / "pairs.parquet")
    size = -(-len(chosen) // batches)
    for batch in range(batches):
        with open(directory / "batches" / f"batch_{batch:03d}.jsonl", "w") as sink:
            for _, row in chosen.iloc[batch * size : (batch + 1) * size].iterrows():
                task = {
                    "uid": row["uid"],
                    "kind": "question",
                    "eval_text": row["question"],
                    "eval_choices": list(row["choices"]),
                    "eval_answer": row["answer"],
                    "eval_passage": row.get("passage") or "",
                    "document_excerpt": row["right"][:EXCERPT],
                }
                sink.write(json.dumps(task, ensure_ascii=False) + "\n")


def judge_labels(prefix: str, judge: str) -> pd.DataFrame:
    """The labels of *judge* under *prefix* (``judge.py --output``): ``<judge>/batch_*.jsonl``."""
    rows = [
        json.loads(line)
        for path in sorted(StoragePath(prefix_join(prefix, f"{judge}/batch_*.jsonl")).glob(), key=str)
        for line in path.read_text().split("\n")
        if line.strip()
    ]
    return pd.DataFrame(rows).set_index("uid")[["label", "answer", "note"]].add_prefix(f"{judge}_")


# Columns of every training data file; the trainer builds `left` from question, choices and answer.
TRAINING_COLUMNS = [
    "question",
    "choices",
    "answer",
    "right",
    "label_q",
    "label_qa",
    "split",
    "source",
    "kind",
    "item",
    "eval",
]


def merge(pairs: pd.DataFrame, first: pd.DataFrame, second: pd.DataFrame, name: str) -> pd.DataFrame:
    """Training rows from the pairs both judges agree on; columns of the trainer's input."""
    frame = pairs.set_index("uid").join(first, how="inner").join(second, how="inner")
    a, b = (frame[column] for column in frame.columns if column.endswith("_label"))
    contaminated = a.isin(CONTAMINATED) & b.isin(CONTAMINATED)
    clean = a.isin(CLEAN) & b.isin(CLEAN)
    logger.info(
        "%s: %d labeled by both; agree contaminated %d, agree clean %d, disagree %d",
        name,
        len(frame),
        contaminated.sum(),
        clean.sum(),
        (~contaminated & ~clean).sum(),
    )
    answers = [column for column in frame.columns if column.endswith("_answer")]
    frame = frame[contaminated | clean].copy()
    frame["label_q"] = contaminated[contaminated | clean].astype(int)
    frame["label_qa"] = (frame[answers] == "YES").all(axis=1).astype(int)
    items = frame["item"].unique()
    test_items = set(np.random.default_rng(11).choice(items, int(len(items) * TEST_SHARE), replace=False))
    frame["split"] = np.where(frame["item"].isin(test_items), "test", "train")
    frame["source"] = name
    frame["kind"] = a[frame.index] + "/" + b[frame.index]
    return frame.reset_index()[["uid", *TRAINING_COLUMNS]]


def silver(pairs: pd.DataFrame, excluded: set[str], size: int, seed: int) -> pd.DataFrame:
    pairs = pairs[~pairs["item"].isin(excluded)]
    overlap = np.array([word_overlap(q, r) for q, r in zip(pairs["question"], pairs["right"], strict=True)])
    frame = pairs[overlap < SILVER_OVERLAP].sample(frac=1, random_state=seed).head(size).copy()
    frame["label_q"] = frame["label_qa"] = 0
    frame["split"], frame["source"], frame["kind"] = "train", "silver", f"overlap<{SILVER_OVERLAP}"
    return frame.reset_index(drop=True)[TRAINING_COLUMNS]


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["sample", "merge", "silver"])
    parser.add_argument("--set", help="sample, merge: local directory of the label set")
    parser.add_argument("--scored", help="sample, silver: scored stage-1 pairs (parquet)")
    parser.add_argument(
        "--exclude-items", nargs="*", default=[], help="sample, silver: parquet files with an item column"
    )
    parser.add_argument("--size", type=int, default=20000, help="silver: pairs kept")
    parser.add_argument("--high", type=int, default=600, help="sample: pairs with high word overlap")
    parser.add_argument("--other", type=int, default=300, help="sample: pairs spread over scores")
    parser.add_argument("--batches", type=int, default=6, help="sample: judge batches")
    parser.add_argument("--labels", help="merge: prefix of the judge outputs (judge.py --output)")
    parser.add_argument("--judges", nargs=2, default=list(JUDGE_MODELS), help="merge: the two judges")
    parser.add_argument("--output", help="merge, silver: training rows (parquet)")
    args = parser.parse_args()
    excluded = {item for path in args.exclude_items for item in read_frame(path)["item"]}
    if args.step == "silver":
        write_frame(silver(read_frame(args.scored), excluded, args.size, seed=0), args.output)
        return
    directory = Path(args.set)
    if args.step == "sample":
        pairs = read_frame(args.scored)
        pairs = pairs[~pairs["item"].isin(excluded)]
        chosen = sample(pairs, args.high, args.other)
        write_batches(chosen, directory, args.batches)
        logger.info(
            "%d pairs; %s; %d families",
            len(chosen),
            chosen["stratum"].value_counts().to_dict(),
            chosen["family"].nunique(),
        )
        return
    pairs = pd.read_parquet(directory / "pairs.parquet")
    rows = merge(pairs, *(judge_labels(args.labels, judge) for judge in args.judges), directory.name)
    write_frame(rows, args.output)
    logger.info("wrote %d rows: %s", len(rows), rows.groupby(["split", "label_q", "label_qa"]).size().to_dict())


if __name__ == "__main__":
    main()
