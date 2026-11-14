# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gold-answer BPB on the non-agentic Eval Policy benchmarks (marin#10197), as validation steps.

Small models score near chance on these benchmarks, so this measures how well a model predicts the gold answer after
the question: each example is ``BOS prompt answer EOS``, and the loss weight is 1 only where the next token is an
answer token. Examples are packed into ``seq_len``-token rows; the EOS between them starts a new attention segment,
so an example never sees the one before it. Each task is one cache, named ``eval_policy/<group>/<task>``, so a run
logs the BPB of every task and the macro BPB of every group.

GPQA is left out: its dataset is gated, and the cluster's Hugging Face token has no access.
"""

import ast
import dataclasses
import logging
import random
import re
from collections.abc import Callable, Iterable, Iterator
from enum import StrEnum

import numpy as np
from datasets import load_dataset
from fray.cluster import ResourceConfig
from levanter.data.text.datasets import DatasetComponent
from levanter.data.text.formats import PrebuiltCacheProcessor, PrebuiltLmDatasetFormat
from levanter.store.cache import CacheLedger, write_levanter_cache
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.execution.remote import remote
from marin.processing.tokenize.store_builder import write_stats_json
from rigging.filesystem.storage_path import prefix_join

logger = logging.getLogger(__name__)

# The step version for each tokenizer. The step names are the W&B metric names, so they cannot carry the tokenizer
# the way Paloma's names do; a new tokenizer needs its own version here.
ANSWER_BPB_VERSIONS = {"hero-bpe-v16384": "2026.11.06"}
# Downloads and tokenizes one task; TriviaQA's rc.nocontext split is the largest download.
BUILD_RESOURCES = ResourceConfig.with_cpu(cpu=2, ram="16g", disk="32g")
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
    to_example: Callable[[dict], AnswerExample]


def mmlu_pro_prompt(row: dict) -> str:
    return f"Question: {row['question']}\nAnswer:"


def truthful_qa_prompt(row: dict) -> str:
    return f"Q: {row['question']}\nA:"


def _mmlu_pro(row: dict) -> AnswerExample:
    return AnswerExample(mmlu_pro_prompt(row), f" {row['options'][row['answer_index']]}")


def _trivia_qa(row: dict) -> AnswerExample:
    return AnswerExample(f"Question: {row['question']}\nAnswer:", f" {row['answer']['value']}")


def _truthful_qa(row: dict) -> AnswerExample:
    return AnswerExample(truthful_qa_prompt(row), f" {row['best_answer']}")


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


TASKS_BY_NAME = {task.name: task for task in TASKS}


def task_component_name(task: AnswerTask) -> str:
    """Validation component name: the run logs the task's BPB and the macro BPB of its group and of all tasks."""
    return f"eval_policy/{task.group}/{task.name}"


def sampled_rows(repo: str, config: str | None, split: str, revision: str) -> list[dict]:
    """The split's rows: all of them, or a seeded sample of ``MAX_EXAMPLES_PER_TASK`` in dataset order."""
    rows = load_dataset(repo, config, split=split, revision=revision)
    indices = list(range(len(rows)))
    if len(indices) > MAX_EXAMPLES_PER_TASK:
        indices = sorted(random.Random(SAMPLE_SEED).sample(indices, MAX_EXAMPLES_PER_TASK))
    return [rows[i] for i in indices]


def task_examples(task: AnswerTask) -> list[AnswerExample]:
    return [task.to_example(row) for row in sampled_rows(task.repo, task.config, task.split, task.revision)]


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
class AnswerBpbCacheConfig:
    task: str
    """The task's name in ``TASKS``."""
    tokenizer: str
    seq_len: int
    output_path: str


class AnswerBpbCache(Artifact):
    """One task's packed cache: the component root, whose ``validation`` split holds the rows."""

    def as_component(self) -> DatasetComponent:
        """A validation component that scores each token with its loss weight."""
        if self.record is None:
            raise ValueError(f"{self.path}: answer-BPB cache has no record")
        task = TASKS_BY_NAME[self.record.config["task"]]
        fmt = PrebuiltLmDatasetFormat(input_ids_key=INPUT_IDS_KEY, loss_weights_key=LOSS_WEIGHTS_KEY)
        return DatasetComponent(source=None, cache_dir=self.path, format=fmt, tags=[task_component_name(task)])


def build_answer_bpb_cache(config: AnswerBpbCacheConfig) -> None:
    task = TASKS_BY_NAME[config.task]
    tokenizer = load_tokenizer(config.tokenizer)
    encoded = [encode_example(tokenizer, example) for example in task_examples(task)]
    kept = [e for e in encoded if len(e[0]) <= config.seq_len]
    split_dir = prefix_join(config.output_path, "validation")
    processor = PrebuiltCacheProcessor(INPUT_IDS_KEY, LOSS_WEIGHTS_KEY)
    rows = pack_examples(kept, config.seq_len, tokenizer.eos_token_id)
    write_levanter_cache(rows, split_dir, metadata=processor.metadata)
    write_stats_json(split_dir, CacheLedger.load(split_dir))
    answer_tokens = int(sum(sum(weights) for _, weights in kept))
    dropped = len(encoded) - len(kept)
    logger.info("%s: %d examples (%d dropped), %d answer tokens", task.name, len(kept), dropped, answer_tokens)


def eval_policy_answer_bpb_datasets(*, tokenizer: str, seq_len: int) -> dict[str, ArtifactStep[AnswerBpbCache]]:
    """One validation step per task, keyed by its component name."""
    if tokenizer not in ANSWER_BPB_VERSIONS:
        raise ValueError(f"no answer-BPB cache version for tokenizer {tokenizer!r}; add one to ANSWER_BPB_VERSIONS")
    run = remote(build_answer_bpb_cache, resources=BUILD_RESOURCES)
    steps = {}
    for task in TASKS:
        name = task_component_name(task)
        steps[name] = ArtifactStep(
            name=name,
            version=ANSWER_BPB_VERSIONS[tokenizer],
            artifact_type=AnswerBpbCache,
            run=run,
            build_config=lambda ctx, task=task: AnswerBpbCacheConfig(task.name, tokenizer, seq_len, ctx.output_path),
        )
    return steps
