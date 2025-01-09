from collections.abc import Sequence

from levanter.eval_harness import TaskConfig

from marin.evaluation.evaluation_config import EvalTaskConfig

# tasks to run (corresponding to lm_eval_harness tasks)
# subset from from page 43 of the DCLM paper: https://arxiv.org/pdf/2406.11794
# TODO: add more once supported in lm-eval-harness and/or tested on our end
CORE_TASKS = (
    EvalTaskConfig("agieval_lsat_ar", 3),  # 3-shot tests in legal domain
    EvalTaskConfig("arc_easy", 10),  # 10-shot, four-way MCQ questions involving grade 3-9 basic science
    EvalTaskConfig("arc_challenge", 10),  # a (harder) version of arc_easy
    EvalTaskConfig("boolq", 10),  # answer yes/no questions based on a passage
    EvalTaskConfig("commonsense_qa", 10),  # 5-way multiple-choice questions based on common-sense, everyday scenarios
    EvalTaskConfig("copa", 0),  # use causal reasoning to predict the correct outcome of a given scenario
    EvalTaskConfig("hellaswag", 0, task_alias="hellaswag_0shot"),  # 4-way multiple choice commonsense reasoning dataset
    EvalTaskConfig("hellaswag", 10, task_alias="hellaswag_10shot"),  # 4-way MCQ commonsense reasoning dataset
    EvalTaskConfig("lambada_openai", 0),  # predict the endings of text passages
    EvalTaskConfig("openbookqa", 0),  # 4-way multiple choice question answering task that requires multi-step reasoning
    EvalTaskConfig("piqa", 10),  # answer questions based on a passage
    # (requires generation which is not supported in Levanter at the moment)
    # EvalTaskConfig("squadv2", 10),  # reading comprehension benchmark
    EvalTaskConfig("wsc273", 0),  # Winograd Schema Challenge
    EvalTaskConfig("winogrande", 0),  # Winograd challenge, extended to more domains
)


def convert_to_levanter_task_config(tasks: Sequence[EvalTaskConfig]) -> list[TaskConfig]:
    """
    Convert a list of EvalTaskConfig to a list of TaskConfig that Levanter's eval_harness expects.
    """
    return [
        TaskConfig(
            task=task.name,
            num_fewshot=task.num_fewshot,
            task_alias=task.task_alias,
        )
        for task in tasks
    ]
