"""Public LAB-Bench text components with deterministic Marin answer extraction."""

import hashlib
import random
from dataclasses import dataclass
from string import ascii_uppercase

import pyarrow.parquet as pq
from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.data_sources import fetch_hf
from eval.limits import ensure_context_window, require_context_length
from eval.sources.common import SourceOptions, text_specs
from eval.task import BaseBenchmark
from rigging.filesystem.factory import open_url

DATASET = "futurehouse/lab-bench"
REVISION = "5c77cec648430f30611808808861eb86f81d5eaa"
PROTOCOL_VERSION = "marin-labbench-no-tools-v1"
ABSTENTION = "Insufficient information to answer the question"
PROMPT = """The following is a multiple choice question about biology.
Please answer by responding with the letter of the correct answer.
Think step by step.

Question: {question}

Options:
{answers}

You MUST include the letter of the correct answer within the following tags: [ANSWER] and [/ANSWER].
For example, '[ANSWER]<answer>[/ANSWER]', where <answer> is the correct letter.
Always answer in exactly this format of a single letter between the two tags, even if you are unsure.
We require this because we use automatic parsing."""


@dataclass(frozen=True)
class Component:
    subset: str
    count: int


COMPONENTS = {
    "LABBenchProtocolQA": Component("ProtocolQA", 108),
    "LABBenchSeqQA": Component("SeqQA", 600),
    "LABBenchCloning": Component("CloningScenarios", 33),
}


def prepare_question(row: dict, subset: str, seed: int) -> dict:
    """Shuffle all choices per item, independently of execution order or run caps."""
    choices = [row["ideal"], ABSTENTION, *row["distractors"]]
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
        "abstention": letters[permutation.index(1)],
        "letters": letters,
        "subtask": row["subtask"],
    }


class LABBenchBenchmark(BaseBenchmark):
    SOURCE = SourceRef(DATASET, "train", REVISION)
    METRICS = ("accuracy", "precision", "attempted_answer_coverage", "abstention_rate", "parse_failure_rate")
    PRIMARY_METRIC = "accuracy"
    REDUCER = ReducerSpec(name="labbench", trial_metric=None)

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
        specs = list(text_specs(
            records, options,
            prompt=lambda row: row["prompt"], target=lambda row: row["answer"],
            grader="labbench", generation_kwargs={"temperature": 0.0, "do_sample": False, "max_gen_toks": 8192},
            reference=lambda row: {"abstention": row["abstention"], "letters": row["letters"]},
            tags=lambda row: {"subtask": row["subtask"]},
        ))
        context = require_context_length(options.context_length, task_name=options.task)
        if options.prompt_token_count is None:
            raise ValueError("LAB-Bench requires the served tokenizer for context validation")
        # Validate the entire selected cohort before sending any requests. Do not silently
        # truncate questions, lower their output budgets, or score overflow as abstention.
        for spec in specs:
            request = spec.requests[0]
            ensure_context_window(
                context_length=context,
                prompt_tokens=options.prompt_token_count(request.prompt),
                output_tokens=request.generation["max_gen_toks"],
            )
        yield from specs
