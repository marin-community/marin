"""
Code for running the CORE evaluation benchmark from the DCLM paper.
"""

from evals import evaluate_lm_evaluation_harness
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main

# tasks to run (corresponding to lm_eval_harness tasks)
# from page 43 of the DCLM paper: https://arxiv.org/pdf/2406.11794
CORE_TASKS = [
    EvalTaskConfig("agieval_lsat_ar", 3),  # 3-shot tests in legal domain
    EvalTaskConfig("arc_easy", 10),  # 10-shot, four-way MCQ questions involving grade 3-9 basic science
    EvalTaskConfig("arc_challenge", 10),  # a (harder) version of arc_easy
    # these are all from BigBench
    # requires completing factual statements with the correct answer
    EvalTaskConfig("bigbench_qa_wikidata_generate_until", 10),
    # requires completing partially balanced expression consisting of parentheses & braces
    EvalTaskConfig("bigbench_dyck_languages_multiple_choice", 10),
    # compute the output from some expression with newly defined operators
    EvalTaskConfig("bigbench_operators_generate_until", 10),
    # differentiate instructions from text-to-copy & to perform sequence of operations
    EvalTaskConfig("bigbench_repeat_copy_logic_generate_until", 10),
    # requires executing algorithms such as recursion and dynamic programming
    EvalTaskConfig("bigbench_cs_algorithms_multiple_choice", 10),
    # identify the language of given text
    EvalTaskConfig("bigbench_language_identification_multiple_choice", 10),
    EvalTaskConfig("boolq", 10),  # answer yes/no questions based on a passage
    EvalTaskConfig("commonsense_qa", 10),  # 5-way multiple-choice questions based on common-sense, everyday scenarios
    EvalTaskConfig("copa", 0),  # use causal reasoning to predict the correct outcome of a given scenario
    EvalTaskConfig("coqa", 0),  # conversational question-answering based on a passage
    EvalTaskConfig("hellaswag", 0),  # 4-way multiple choice commonsense reasoning dataset
    EvalTaskConfig("hellaswag", 10),  # 4-way multiple choice commonsense reasoning dataset
    # ("jeopardy", 10, None) # not in lm_eval_harness right now :(
    EvalTaskConfig("lambada", 0),  # predict the endings of text passages
    EvalTaskConfig("openbookqa", 0),  # 4-way multiple choice question answering task that requires multi-step reasoning
    EvalTaskConfig("piqa", 10),  # answer questions based on a passage
    EvalTaskConfig("squadv2", 10),  # reading comprehension benchmark
    EvalTaskConfig("wsc273", 0),  # Winograd Schema Challenge
    EvalTaskConfig("winogrande", 0),  # Winograd challenge, extended to more domains
]


def run_core_evaluations(model_name: str, model_path: str) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using LM Evaluation Harness.

    Args:
        model_name (str): Name of the model.
        model_path (str): Path to the model.
        evals (list[EvalTaskConfig]): List of evaluations to run with LM Evaluation Harness.
    """

    core_evals = CORE_TASKS
    return evaluate_lm_evaluation_harness(model_name, model_path, core_evals)


steps = [
    run_core_evaluations(
        model_name="core-eval-benchmark/dclm_baseline_1b_1x_replication_nov12-b182e8",
        model_path="gs://marin-us-central2/checkpoints/dclm_baseline_1b_1x_replication_nov12-b182e8/hf/step-54930",
    ),
]

if __name__ == "__main__":
    executor_main_config = ExecutorMainConfig()
    executor_main(executor_main_config, steps=steps)
