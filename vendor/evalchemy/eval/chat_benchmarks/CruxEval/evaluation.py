# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
from eval.contracts.aggregation import functional_pass_at_k
from eval.contracts.sample_results import PER_TASK_PASS_RATE_FIELD
from .execution import check_correctness
import json
import argparse
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from tqdm import tqdm
import os
import gzip
from typing import Dict, Iterable, List, Union
import itertools
import sys


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray], num_correct: Union[List[int], np.ndarray], k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def stream_jsonl_all(filename: str) -> Iterable[Dict]:
    """
    Streams a JSONL file.
    """
    results = []
    if filename.endswith(".gz"):
        fp = gzip.open(open(filename, "rb"), "rt")
    else:
        fp = open(filename, "r")
    for line in fp:
        if any(not x.isspace() for x in line):
            results.append(json.loads(line))
    fp.close()

    return results


def grade_generations(
    input_file,
    mode,
    examples,
    tmp_dir,
    timeout=10.0,
):
    # Load the samples
    references = [(doc["code"], doc["input"], doc["output"]) for doc in examples]
    n_workers = 8

    # Load the generations
    sample_jsonl = stream_jsonl_all(input_file)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for sample in tqdm(sample_jsonl):
            task_id = sample["task_id"]
            lang = "python"
            tmp_dir_ = os.path.join(tmp_dir, lang, "evaluation")
            sample["task_id"] = task_id
            c = sample["code"]
            o = sample["output"]
            g = sample["generation"]
            if mode == "input":
                g = re.search(r"f\(.*?\)", g)
                if g is not None:
                    g = g.group(0)
                else:
                    g = sample["generation"]
            else:
                func = re.search(r"f\(.*?\)", g)
                if func is not None:
                    g = g.replace(func.group(0), "").replace("=", "").replace("assert", "").strip()
                else:
                    g = sample["generation"]
            sample["test_code"] = f"{c}\nassert {o} == {g}"
            if sample["test_code"] is None:
                continue
            if "completion_id" in sample:
                completion_id_ = sample["completion_id"]
            else:
                completion_id_ = completion_id[task_id]
            args = (task_id, sample, lang, timeout, tmp_dir_, completion_id_)
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        print("Running test suites...")
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))
    # print("all_scores: ", all_scores)

    return {
        task_id: [verdict for _, verdict in sorted(completions, key=lambda row: row[0])]
        for task_id, completions in results.items()
    }


def evaluate_generations(
    input_file,
    mode,
    examples,
    tmp_dir,
    k: List[int] = [1, 5],
    timeout=10.0,
):

    verdicts = grade_generations(input_file, mode, examples, tmp_dir, timeout=timeout)
    return {
        **functional_pass_at_k(verdicts, k),
        PER_TASK_PASS_RATE_FIELD: {
            task_id: sum(row["passed"] for row in rows) / len(rows)
            for task_id, rows in verdicts.items()
        },
    }
