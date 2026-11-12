import json
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import pickle
import copy
import zlib
import base64
import re
from enum import Enum

from eval.chat_benchmarks.LiveBench.livebench.lcb_runner.utils.extraction_utils import extract_code
from eval.chat_benchmarks.LiveBench.livebench.lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

# from LiveCodebench, modified


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)
        # if self.testtype == TestType.FUNCTIONAL:
        #     self.input = json.loads(self.input)
        #     self.output = json.loads(self.output)


def LCB_generation_process_results(question: dict, llm_answer: str, debug=False) -> int:
    llm_answer = extract_code(
        model_output=llm_answer, lmstyle=None
    )  # Missing out only on some slightly different handling for CodeLlamaInstruct from the original LiveCodeBench

    # if this is a completion question, check that the completion is present.
    if (
        "partial_solution" in question
        and (not question["partial_solution"] is None)
        and (len(question["partial_solution"]) > 0)
    ):
        # if len(llm_answer) < len(question['partial_solution']):
        #     return 0
        # if llm_answer[:len(question['partial_solution'])] != question['partial_solution']:
        #     return 0
        llm_answer = question["partial_solution"] + "\n" + llm_answer

    # code mostly from LiveCodeBench, with modifications.
    public_test_cases = json.loads(question["public_test_cases"])  # type: ignore
    public_test_cases = [Test(**t) for t in public_test_cases]

    try:
        private_test_cases = json.loads(question["private_test_cases"])  # type: ignore
    except:
        private_test_cases = json.loads(
            pickle.loads(
                zlib.decompress(base64.b64decode(question["private_test_cases"].encode("utf-8")))  # type: ignore
            )
        )  # type: ignore

    private_test_cases = [Test(**t) for t in private_test_cases]
    metadata = json.loads(question["original_json"]["metadata"])  # type: ignore
    eval_sample = {
        "input_output": json.dumps(
            {
                "inputs": [t.input for t in public_test_cases + private_test_cases],
                "outputs": [t.output for t in public_test_cases + private_test_cases],
                "fn_name": metadata.get("func_name", None),
            }
        )
    }

    metrics, results, metadata = codegen_metrics(
        [eval_sample],
        [[llm_answer]],
        k_list=[1],  # can't compute higher pass@ because we don't have more than one prediction.
        num_process_evaluate=1,  # multiprocessing is handled at a higher level to parallelize multiple questions at once, so we don't want to complicate this with forking here.
        timeout=6,  # default eval setting from livecodebench.
    )

    if metrics["pass@1"] == 1.0:
        return 1
    else:
        if debug:
            print("INCORRECT", question["question_title"], question["question_id"])
            print("llm answer", "\n", llm_answer)
            print("results", results)
            print("metadata", metadata)
        return 0
