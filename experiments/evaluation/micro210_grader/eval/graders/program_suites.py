"""Single-program adapters to the existing language-specific test protocols."""

import importlib
import re
import tempfile
import textwrap

from eval.contracts.eval_spec import Grade

SUITES = {
    "HumanEval": "eval.chat_benchmarks.HumanEval.human_eval",
    "HumanEvalPlus": "eval.chat_benchmarks.HumanEvalPlus.human_eval_plus",
    "MBPP": "eval.chat_benchmarks.HumanEval.human_eval",
    "MBPPPlus": "eval.chat_benchmarks.MBPPPlus.mbpp_plus",
    "MultiPLE": "eval.chat_benchmarks.MultiPLE.multiple",
    "BigCodeBench": "eval.chat_benchmarks.BigCodeBench",
}


def program_suite(spec, responses):
    reference = spec.grader.reference
    suite = reference["suite"]
    package = SUITES[suite]
    evaluation = importlib.import_module(package + ".evaluation")
    execution = importlib.import_module(package + ".execution")
    problem = reference["problem"]
    sample = dict(problem)
    output = str(responses[-1])
    language = reference["language"]
    if suite in {"MBPP", "MBPPPlus"}:
        blocks = re.findall(r"```python\n(.*?)```", output, re.DOTALL | re.IGNORECASE)
        code = blocks[0] if blocks else output
        sample["generation"] = textwrap.dedent(code).strip("\n") if suite == "MBPPPlus" else code
    elif suite == "BigCodeBench":
        from eval.chat_benchmarks.BigCodeBench.sanitize import code_extract

        sample["generation"] = code_extract(output)
    else:
        module = {
            "HumanEval": "eval.chat_benchmarks.HumanEval.utils.utils",
            "HumanEvalPlus": "eval.chat_benchmarks.HumanEvalPlus.utils.utils",
            "MultiPLE": "eval.chat_benchmarks.MultiPLE.utils",
        }[suite]
        sample["output"] = output
        sample = importlib.import_module(module).extract_generation_code(sample, lang_code=language)
    task_id = problem["task_id"]
    sample["test_code"] = evaluation.process_humaneval_test(
        sample, {task_id: problem}, is_mbpp=reference.get("mbpp", False), language=language,
    )
    with tempfile.TemporaryDirectory(prefix="evalchemy-program-") as directory:
        args = (task_id, sample, language, reference["timeout"], directory, 0)
        if suite == "MultiPLE":
            outcome = execution.check_code_correctness(*args)
        elif suite == "BigCodeBench":
            outcome = execution.check_correctness(*args, safe_mode=reference["safe_mode"])
        else:
            outcome = execution.check_correctness(*args)
    return Grade({"accuracy": float(outcome["passed"])}, filtered_resps=[sample["generation"]])
