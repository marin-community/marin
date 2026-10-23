"""Public LAB-Bench text components with shared multiple-choice scoring."""

import hashlib
import random
from dataclasses import dataclass
from string import ascii_uppercase
from types import MappingProxyType

import pyarrow.parquet as pq
from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.data_sources import fetch_hf
from eval.sources.common import SourceOptions, text_specs
from eval.task import BaseBenchmark
from rigging.filesystem.factory import open_url

DATASET = "futurehouse/lab-bench"
REVISION = "5c77cec648430f30611808808861eb86f81d5eaa"
PROTOCOL_VERSION = "marin-labbench-no-tools-v2"
PROMPT = """The following is a multiple choice question about biology.
Please answer by responding with the letter of the correct answer.
Think step by step.

Question: {question}

Options:
{answers}

End your response with "Answer: X", where X is the letter of your chosen answer."""


@dataclass(frozen=True)
class Component:
    subset: str
    count: int


COMPONENTS = MappingProxyType({
    "LABBenchProtocolQA": Component("ProtocolQA", 108),
    "LABBenchSeqQA": Component("SeqQA", 600),
    "LABBenchCloning": Component("CloningScenarios", 33),
})


def prepare_question(row: dict, subset: str, seed: int) -> dict:
    """Shuffle all choices per item, independently of execution order or run caps."""
    choices = [row["ideal"], *row["distractors"]]
    if len(choices) > len(ascii_uppercase):
        raise ValueError(f"Too many answer choices for {row['id']}")
    permutation = list(range(len(choices)))
    identity = f"{seed}:{subset}:{row['id']}"
    random.Random(int.from_bytes(hashlib.sha256(identity.encode()).digest())).shuffle(permutation)
    letters = ascii_uppercase[:len(choices)]
    answers = "\n".join(f"({letter}) {choices[index]}" for letter, index in zip(letters, permutation, strict=True))
    # The native ProtocolQA task concatenates the supplied protocol and question.
    question = row["protocol"] + row["question"] if subset == "ProtocolQA" else row["question"]
    return {
        "id": row["id"],
        "prompt": PROMPT.format(question=question, answers=answers),
        "answer": letters[permutation.index(0)],
        "subtask": row["subtask"],
    }


class LABBenchBenchmark(BaseBenchmark):
    SOURCE = SourceRef(DATASET, "train", REVISION)
    METRICS = ("accuracy",)
    PRIMARY_METRIC = "accuracy"
    REDUCER = ReducerSpec(trial_metric=None)

    def __init__(self, logger=None, system_instruction=None):
        if system_instruction is not None:
            raise ValueError("LAB-Bench uses its fixed user prompt without a system instruction")
        super().__init__(logger=logger)

    def benchmark_size(self):
        return COMPONENTS[self.benchmark_name].count

    def evaluation_metadata(self):
        return {"protocol": PROTOCOL_VERSION, "subset": COMPONENTS[self.benchmark_name].subset}

    def iter_specs(self, options: SourceOptions):
        if options.system_instruction is not None:
            raise ValueError("LAB-Bench uses its fixed user prompt without a system instruction")
        component = COMPONENTS[options.task]
        path = fetch_hf(
            DATASET, f"{component.subset}/train-00000-of-00001.parquet",
            revision=REVISION, cache_prefix=self.dataset_cache,
        )
        with open_url(path, "rb") as stream:
            rows = pq.read_table(stream).to_pylist()
        if len(rows) != component.count or len({row["id"] for row in rows}) != component.count:
            raise ValueError(f"LAB-Bench {component.subset} must contain {component.count} distinct items")
        records = [prepare_question(row, component.subset, options.seed) for row in rows]
        return text_specs(
            records, options,
            prompt=lambda row: row["prompt"], target=lambda row: row["answer"],
            grader="letter_choice", generation_kwargs={"temperature": 0.0, "do_sample": False},
            reference=lambda row: {"extractor": "hle"},
            tags=lambda row: {"subtask": row["subtask"]},
        )
