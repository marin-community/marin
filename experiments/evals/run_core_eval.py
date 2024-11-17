"""
Code for running the CORE evaluation benchmark from the DCLM paper.
"""

from evals import evaluate_lm_evaluation_harness
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, ExecutorMainConfig, executor_main

# tasks to run (corresponding to lm_eval_harness tasks)
# from page 43 of the DCLM paper: https://arxiv.org/pdf/2406.11794
CORE_TASKS = [
    ("agieval_lsat_ar", 3, None),  # 3-shot tests in legal domain
    ("arc_easy", 10, None),  # 10-shot, four-way MCQ questions involving grade 3-9 basic science
    ("arc_challenge", 10, None),  # a (harder) version of arc_easy
    # these are all from BigBench
    ("bigbench_qa_wikidata_generate_until", 10, None),  # requires completing factual statements with the correct answer
    ("bigbench_dyck_languages_multiple_choice", 10, None),  # requires completing partially balanced expression consisting of parentheses & braces
    ("bigbench_operators_generate_until", 10, None),  # compute the output from some expression with newly defined operators
    ("bigbench_repeat_copy_logic_generate_until", 10, None),  # differentiate instructions from text-to-copy & to perform sequence of operations
    ("bigbench_cs_algorithms_multiple_choice", 10, None),  # requires executing algorithms such as recursion and dynamic programming
    ("bigbench_language_identification_multiple_choice", 10, None),  # identify the language of given text
    ("boolq", 10, None),  # answer yes/no questions based on a passage
    ("commonsense_qa", 10, None),  # 5-way multiple-choice questions based on common-sense, everyday scenarios
    ("copa", 0, None),  # use causal reasoning to predict the correct outcome of a given scenario
    ("coqa", 0, None),  # conversational question-answering based on a passage
    ("hellaswag", 0, None),  # 4-way multiple choice commonsense reasoning dataset
    ("hellaswag", 10, None),  # 4-way multiple choice commonsense reasoning dataset
    # ("jeopardy", 10, None) # not in lm_eval_harness right now :(
    ("lambada", 0, None),  # predict the endings of text passages
    ("openbookqa", 0, None),  # 4-way multiple choice question answering task that requires multi-step reasoning
    ("piqa", 10, None),  # answer questions based on a passage
    ("squadv2", 10, None),  # reading comprehension benchmark
    ("wsc273", 0, None),  # Winograd Schema Challenge
    ("winogrande", 0, None),  # Winograd challenge, extended to more domains
]


def create_core_eval_configs(core_tasks: list[tuple[str, int, int | None]]) -> list[EvalTaskConfig]:
    """
    Create a list of EvalTaskConfig objects for the CORE evaluation tasks.

    Args:
        core_tasks (list[tuple[str, int, int | None]]): List of CORE tasks to run.
    """
    return [EvalTaskConfig(name=task[0], num_fewshot=task[1], max_eval_instances=task[2]) for task in core_tasks]


def run_core_evaluations(model_name: str, model_path: str) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using LM Evaluation Harness.

    Args:
        model_name (str): Name of the model.
        model_path (str): Path to the model.
        evals (list[EvalTaskConfig]): List of evaluations to run with LM Evaluation Harness, e.g, ["mmlu"].
    """

    core_evals = create_core_eval_configs(CORE_TASKS)
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
