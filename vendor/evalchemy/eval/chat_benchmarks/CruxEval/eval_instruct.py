import json
import logging
import os
from typing import Any, Dict, Optional
import warnings

from rigging.filesystem.factory import open_url

from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.data_sources import fetch_hf
from eval.sources.common import text_specs
from eval.task import BaseBenchmark


warnings.filterwarnings("ignore")

CRUXEVAL_REPO = "cruxeval-org/cruxeval"
CRUXEVAL_REVISION = "b96af0450242eb4da433032b90998f25588a5d0f"




def make_cot_output_prompt(s):
    code, input = s
    return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(s):
    s = s + s
    return "b" + s + "a"
assert f("hi") == ??
[/PYTHON]
[THOUGHT]
Let's execute the code step by step:

1. The function f is defined, which takes a single argument s.
2. The function is called with the argument "hi", so within the function, s is initially "hi".
3. Inside the function, s is concatenated with itself, so s becomes "hihi".
4. The function then returns a new string that starts with "b", followed by the value of s (which is now "hihi"), and ends with "a".
5. The return value of the function is therefore "bhihia".
[/THOUGHT]
[ANSWER]
assert f("hi") == "bhihia"
[/ANSWER]

[PYTHON]
{code}
assert f({input}) == ??
[/PYTHON]
[THOUGHT]
"""


def make_cot_input_prompt(s):
    code, output = s
    return f"""You will be given a function f and an output in the form f(??) == output. Your task is to find any input such that executing f on the input leads to the given output. There may be multiple answers, but only output one. First, think step by step. You MUST surround the answer with [ANSWER] and [/ANSWER] tags. Express your answer as a passing assertion containing the input and the given output.

[PYTHON]
def f(x):
    return x + 1
assert f(??) == 17
[/PYTHON]
[THOUGHT]
To find an input such that executing f on the input leads to the given output, we can work backwards from the given assertion. We know that f(??) == 17. 

Since the function f(x) returns x + 1, for f(??) to be equal to 17, the value of ?? should be 16. 
[/THOUGHT]
[ANSWER]
assert f(16) == 17
[/ANSWER]

[PYTHON]
{code}
assert f(??) == {output}
[/PYTHON]
[THOUGHT]
"""




class CruxEvalBenchmark(BaseBenchmark):
    """
    CruxEval benchmark for evaluating code generation capabilities across different languages.
    """

    DO_SAMPLE = True
    TEMPERATURE = 0.2

    def __init__(
        self,
        data_dir: str | None = None,
        max_tokens: int = 2048,
        timeout: float = 120,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize CruxEval benchmark.

        Args:
            data_dir: Optional directory containing CruxEval JSON fixtures
            max_tokens: Maximum number of tokens for generation
            timeout: Timeout for code execution
            debug: If True, only evaluate the first 10 examples
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_dir = data_dir
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.debug = debug
        self.tasks = ["input", "output"]






    @staticmethod
    def _prompt_for_task(example: Dict[str, Any], task: str) -> str:
        if task == "input":
            return make_cot_input_prompt((example["code"], example["output"]))
        return make_cot_output_prompt((example["code"], example["input"]))


    SOURCE = SourceRef(CRUXEVAL_REPO, revision=CRUXEVAL_REVISION)
    METRICS = ("pass@1",)
    PRIMARY_METRIC = "pass@1"
    REDUCER = ReducerSpec(accuracy="pass@1", groups={"direction": "{group}_{metric}"})

    def iter_specs(self, options):
        path = os.path.join(self.data_dir, "cruxeval.jsonl") if self.data_dir is not None else fetch_hf(
            CRUXEVAL_REPO, "test.jsonl", revision=CRUXEVAL_REVISION, cache_prefix=self.dataset_cache,
        )
        with open_url(path, "rt") as data:
            rows = [json.loads(line) for line in data if line.strip()]
        if self.debug:
            rows = rows[:10]
        for direction in self.tasks:
            yield from text_specs(
                rows, options, namespace=direction, prompt=lambda row: self._prompt_for_task(row, direction),
                target=lambda row: row[direction], grader="crux", reference=lambda row: {"direction": direction},
                generation_kwargs={"do_sample": self.DO_SAMPLE, "temperature": self.TEMPERATURE, "max_gen_toks": self.max_tokens},
                tags=lambda row: {"direction": direction},
            )
