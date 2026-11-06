# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gold-answer BPB on the non-agentic Eval Policy benchmarks (marin#10197), as packed validation caches.

Small ladder models score near chance on these benchmarks, so the Data Belay measures how well a model predicts the
gold answer after the question: each example is ``BOS prompt answer EOS``, and the loss weight is 1 only where the
next token is an answer token. Examples are packed into 4,096-token sequences; the EOS between them starts a new
attention segment, so an example never sees the one before it. Each task is one cache, tagged with its group, so a
run logs the BPB of every task and the macro BPB of every group.

GPQA is left out: its dataset is gated, and the cluster's Hugging Face token has no access.
"""

import ast
import dataclasses
import json
import logging
import random
import re
from collections.abc import Callable, Iterable, Iterator
from enum import StrEnum

import numpy as np
from datasets import load_dataset
from levanter.data.text.formats import PrebuiltCacheProcessor
from levanter.store.cache import CacheLedger, write_levanter_cache
from levanter.tokenizers import MarinTokenizer
from marin.processing.tokenize.store_builder import write_stats_json
from rigging.filesystem.storage_path import StoragePath, prefix_join

logger = logging.getLogger(__name__)

ANSWER_BPB_VERSION = "eval-policy-answer-bpb-v1"
ANSWER_BPB_ROOT = f"s3://marin-us-east-02a/marin/data_belay/eval/{ANSWER_BPB_VERSION}"
MANIFEST_NAME = "manifest.json"
SEQ_LEN = 4096
# A fixed seeded sample of a larger task keeps one run's eval to a few minutes.
MAX_EXAMPLES_PER_TASK = 2000
SAMPLE_SEED = 0
INPUT_IDS_KEY = "input_ids"
LOSS_WEIGHTS_KEY = "loss_weights"


class TaskGroup(StrEnum):
    KNOWLEDGE = "knowledge"
    COMMONSENSE = "commonsense"
    MATH = "math"
    CODE = "code"


@dataclasses.dataclass(frozen=True)
class AnswerExample:
    prompt: str
    answer: str


@dataclasses.dataclass(frozen=True)
class AnswerTask:
    name: str
    group: TaskGroup
    repo: str
    revision: str
    config: str | None
    split: str
    to_example: Callable[[dict], AnswerExample | None]


def _mmlu_pro(row: dict) -> AnswerExample:
    return AnswerExample(f"Question: {row['question']}\nAnswer:", f" {row['options'][row['answer_index']]}")


def _trivia_qa(row: dict) -> AnswerExample:
    return AnswerExample(f"Question: {row['question']}\nAnswer:", f" {row['answer']['value']}")


def _truthful_qa(row: dict) -> AnswerExample:
    return AnswerExample(f"Q: {row['question']}\nA:", f" {row['best_answer']}")


def _boolq(row: dict) -> AnswerExample:
    answer = "yes" if row["answer"] else "no"
    return AnswerExample(f"{row['passage']}\nQuestion: {row['question']}?\nAnswer:", f" {answer}")


def _piqa(row: dict) -> AnswerExample:
    solution = row["sol1"] if int(row["label"]) == 0 else row["sol2"]
    return AnswerExample(f"Goal: {row['goal']}\nSolution:", f" {solution}")


def _winogrande(row: dict) -> AnswerExample:
    # Partial scoring: the gold option fills the blank, and the answer is the rest of the sentence after it.
    option = row["option1"] if row["answer"] == "1" else row["option2"]
    before, after = row["sentence"].split("_", 1)
    return AnswerExample(before + option, after)


def _gsm8k(row: dict) -> AnswerExample:
    # Calculator annotations (<<48/2=24>>) are not text a model should predict.
    solution = re.sub(r"<<[^>]*>>", "", row["answer"])
    reasoning, _, final = solution.rpartition("####")
    return AnswerExample(
        f"Question: {row['question']}\nAnswer:", f" {reasoning.strip()}\nThe answer is {final.strip()}."
    )


def _math500(row: dict) -> AnswerExample:
    return AnswerExample(f"Problem: {row['problem']}\nSolution:", f" {row['solution']}")


def _humaneval(row: dict) -> AnswerExample:
    return AnswerExample(row["prompt"], row["canonical_solution"])


def _mbpp(row: dict) -> AnswerExample:
    tests = "\n".join(ast.literal_eval(row["test_list"]) if isinstance(row["test_list"], str) else row["test_list"])
    # The gold solutions use \r\n line ends; a model trained on \n text spends most of its bits on the \r.
    code = row["code"].replace("\r\n", "\n").strip("\n")
    return AnswerExample(f'"""\n{row["prompt"]}\n{tests}\n"""\n', f"{code}\n")


def _cruxeval_output(row: dict) -> AnswerExample:
    return AnswerExample(f"{row['code']}\nassert f({row['input']}) ==", f" {row['output']}")


TASKS = (
    AnswerTask(
        "mmlu_pro",
        TaskGroup.KNOWLEDGE,
        "TIGER-Lab/MMLU-Pro",
        "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        None,
        "test",
        _mmlu_pro,
    ),
    AnswerTask(
        "trivia_qa",
        TaskGroup.KNOWLEDGE,
        "mandarjoshi/trivia_qa",
        "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
        "rc.nocontext",
        "validation",
        _trivia_qa,
    ),
    AnswerTask(
        "truthful_qa",
        TaskGroup.KNOWLEDGE,
        "truthfulqa/truthful_qa",
        "741b8276f2d1982aa3d5b832d3ee81ed3b896490",
        "generation",
        "validation",
        _truthful_qa,
    ),
    AnswerTask(
        "boolq",
        TaskGroup.COMMONSENSE,
        "google/boolq",
        "35b264d03638db9f4ce671b711558bf7ff0f80d5",
        None,
        "validation",
        _boolq,
    ),
    AnswerTask(
        "piqa",
        TaskGroup.COMMONSENSE,
        "baber/piqa",
        "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
        None,
        "validation",
        _piqa,
    ),
    AnswerTask(
        "winogrande",
        TaskGroup.COMMONSENSE,
        "allenai/winogrande",
        "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
        "winogrande_xl",
        "validation",
        _winogrande,
    ),
    AnswerTask(
        "gsm8k", TaskGroup.MATH, "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main", "test", _gsm8k
    ),
    AnswerTask(
        "math500",
        TaskGroup.MATH,
        "HuggingFaceH4/MATH-500",
        "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        None,
        "test",
        _math500,
    ),
    AnswerTask(
        "humaneval_plus",
        TaskGroup.CODE,
        "evalplus/humanevalplus",
        "d32357cf319e50e9c8d8dab5ea876c72b0fd321b",
        None,
        "test",
        _humaneval,
    ),
    AnswerTask(
        "mbpp_plus",
        TaskGroup.CODE,
        "evalplus/mbppplus",
        "b2d74c91837c3f2a20c1299ae98133cbe7cfa077",
        None,
        "test",
        _mbpp,
    ),
    AnswerTask(
        "cruxeval_o",
        TaskGroup.CODE,
        "cruxeval-org/cruxeval",
        "b96af0450242eb4da433032b90998f25588a5d0f",
        None,
        "test",
        _cruxeval_output,
    ),
)


def task_examples(task: AnswerTask) -> list[AnswerExample]:
    """The task's examples: all of them, or a seeded sample of ``MAX_EXAMPLES_PER_TASK`` in dataset order."""
    rows = load_dataset(task.repo, task.config, split=task.split, revision=task.revision)
    indices = list(range(len(rows)))
    if len(indices) > MAX_EXAMPLES_PER_TASK:
        indices = sorted(random.Random(SAMPLE_SEED).sample(indices, MAX_EXAMPLES_PER_TASK))
    return [task.to_example(rows[i]) for i in indices]


def encode_example(tokenizer: MarinTokenizer, example: AnswerExample) -> tuple[list[int], list[float]]:
    """Tokens ``BOS prompt answer EOS`` and weights that are 1 where the next token is an answer token.

    The prompt and the answer are encoded separately, so the answer's tokens do not depend on the prompt's last
    characters, as in lm-eval-harness.
    """
    prompt = tokenizer.encode(example.prompt)
    answer = tokenizer.encode(example.answer)
    if not answer:
        raise ValueError(f"empty answer for prompt {example.prompt[:80]!r}")
    tokens = [tokenizer.bos_token_id, *prompt, *answer, tokenizer.eos_token_id]
    weights = [0.0] * len(tokens)
    # weights[i] scores the prediction of tokens[i + 1]; the answer starts at index 1 + len(prompt).
    for i in range(len(prompt), len(prompt) + len(answer)):
        weights[i] = 1.0
    return tokens, weights


def pack_examples(
    encoded: Iterable[tuple[list[int], list[float]]], seq_len: int, pad_id: int
) -> Iterator[dict[str, np.ndarray]]:
    """Pack whole examples into ``seq_len`` rows, padding each row with ``pad_id`` at zero weight.

    An example longer than ``seq_len`` is dropped; the caller counts what it passes in against what comes out.
    """
    tokens: list[int] = []
    weights: list[float] = []

    def row() -> dict[str, np.ndarray]:
        pad = seq_len - len(tokens)
        return {
            INPUT_IDS_KEY: np.asarray(tokens + [pad_id] * pad, dtype=np.int32),
            LOSS_WEIGHTS_KEY: np.asarray(weights + [0.0] * pad, dtype=np.float32),
        }

    for example_tokens, example_weights in encoded:
        if len(example_tokens) > seq_len:
            continue
        if len(tokens) + len(example_tokens) > seq_len:
            yield row()
            tokens, weights = [], []
        tokens.extend(example_tokens)
        weights.extend(example_weights)
    if tokens:
        yield row()


@dataclasses.dataclass(frozen=True)
class TaskCache:
    """One task's validation cache, as the launcher's ``--eval-manifest`` reads it."""

    name: str
    """Component name ``eval_policy/<group>/<task>``: the run logs task BPB and group and overall macro BPB."""
    cache_dir: str
    """Component root; the cache itself is its ``validation`` split."""
    examples: int
    dropped: int
    answer_tokens: int


def build_task_cache(task: AnswerTask, tokenizer: MarinTokenizer, root: str) -> TaskCache:
    examples = task_examples(task)
    encoded = [encode_example(tokenizer, example) for example in examples]
    kept = [e for e in encoded if len(e[0]) <= SEQ_LEN]
    cache_dir = prefix_join(root, task.name)
    split_dir = prefix_join(cache_dir, "validation")
    processor = PrebuiltCacheProcessor(INPUT_IDS_KEY, LOSS_WEIGHTS_KEY)
    write_levanter_cache(pack_examples(kept, SEQ_LEN, tokenizer.eos_token_id), split_dir, metadata=processor.metadata)
    write_stats_json(split_dir, CacheLedger.load(split_dir))
    answer_tokens = int(sum(sum(weights) for _, weights in kept))
    dropped = len(encoded) - len(kept)
    logger.info("%s: %d examples (%d dropped), %d answer tokens", task.name, len(kept), dropped, answer_tokens)
    return TaskCache(f"eval_policy/{task.group}/{task.name}", cache_dir, len(kept), dropped, answer_tokens)


def build_answer_bpb(tokenizer: MarinTokenizer, root: str = ANSWER_BPB_ROOT) -> list[TaskCache]:
    """Write every task's cache under ``root`` and a manifest that the launcher reads."""
    caches = [build_task_cache(task, tokenizer, root) for task in TASKS]
    manifest = {
        "version": ANSWER_BPB_VERSION,
        "seq_len": SEQ_LEN,
        "input_ids_key": INPUT_IDS_KEY,
        "loss_weights_key": LOSS_WEIGHTS_KEY,
        "components": [dataclasses.asdict(c) for c in caches],
    }
    StoragePath(prefix_join(root, MANIFEST_NAME)).write_text(json.dumps(manifest, indent=2))
    return caches
