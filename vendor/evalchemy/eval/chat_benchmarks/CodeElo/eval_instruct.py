from eval.contracts.eval_spec import ReducerSpec, SourceRef
"""CodeElo data, prompt formatting, and the shared program-grader reference."""

import logging
import os

from datasets import load_dataset

from eval.sources.common import text_specs
from eval.sources.programs import difficulty, exact_tests
from eval.task import BaseBenchmark

INSTRUCTION = '"You are a coding expert. Given a competition-level coding problem, you need to write a Python program to solve it. You may start by outlining your thought process. In the end, please provide the complete code in a code block enclosed with ``` ```. The code should take stdin as input and print the output.'




def prompt(problem):
    sample_input, sample_output = problem["examples"][0]
    text = (
        f'<html><body><h1>{problem["title"]}</h1>'
        f'<div>Time limit per test: {problem["time_limit_ms"]} ms</div>'
        f'<h2>Description</h2><div>{problem["description"]}</div>'
        f'<h2>Input</h2><div>{problem["input"]}</div>'
        f'<h2>Output</h2><div>{problem["output"]}</div>'
        f'<h2>Example</h2><h3>Input</h3><div>{sample_input}</div>'
        f'<h3>Output</h3><div>{sample_output}</div>'
    )
    if problem["interaction"]:
        text += f'<h2>Interaction</h2><div>{problem["interaction"]}</div>'
    if problem["note"]:
        text += f'<h2>Note</h2><div>{problem["note"]}</div>'
    return f"{INSTRUCTION}\n\n{text}</body></html>"


def test_reference(problem):
    examples = [{"input": sample_input, "output": output} for sample_input, output in problem["examples"]]
    return {"tests": exact_tests(examples), "stdin": True, "timeout": problem["time_limit_ms"] / 1000, "comparison": "exact"}


class CodeEloBenchmark(BaseBenchmark):
    SOURCE = SourceRef("Qwen/CodeElo")
    REDUCER = ReducerSpec(accuracy="accuracy_avg", stderr="accuracy_std_err", repeat_ddof=1, group_suffix="_avg")
    METRICS = ("accuracy_avg",)
    PRIMARY_METRIC = "accuracy_avg"

    def __init__(self, debug=False, seed=(0, 1234, 1234, 1234), max_tokens=32768,
                 logger: logging.Logger | None = None, system_instruction=None):
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens
        self.n_repeat = 3
        self.filter_interaction_questions = True

    def load_questions(self):
        rows = load_dataset("Qwen/CodeElo", cache_dir=os.environ.get("HF_HUB_CACHE"))["test"]
        if self.debug:
            rows = list(rows)[:10]
        return ({**row, "difficulty": difficulty(row["rating"])} for row in rows
                if not self.filter_interaction_questions or not row["interaction"])

    def iter_specs(self, options):
        return text_specs(
            self.load_questions(), options, prompt=prompt, target=lambda example: example["examples"],
            grader="program", reference=test_reference,
            generation_kwargs={"do_sample": False, "max_gen_toks": self.max_new_tokens}, repeats=self.n_repeat,
            tags=lambda example: {"difficulty": example["difficulty"]},
        )
