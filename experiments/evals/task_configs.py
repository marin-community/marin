from levanter.eval_harness import TaskConfig

from marin.evaluation.evaluation_config import EvalTaskConfig

# tasks to run (corresponding to lm_eval_harness tasks)
# subset from from page 43 of the DCLM paper: https://arxiv.org/pdf/2406.11794
# TODO: add more once supported in lm-eval-harness and/or tested on our end
CORE_TASKS = [
    EvalTaskConfig("agieval_lsat_ar", 3),  # 3-shot tests in legal domain
    EvalTaskConfig("arc_easy", 10),  # 10-shot, four-way MCQ questions involving grade 3-9 basic science
    EvalTaskConfig("arc_challenge", 10),  # a (harder) version of arc_easy
    EvalTaskConfig("boolq", 10),  # answer yes/no questions based on a passage
    EvalTaskConfig("commonsense_qa", 10),  # 5-way multiple-choice questions based on common-sense, everyday scenarios
    EvalTaskConfig("copa", 0),  # use causal reasoning to predict the correct outcome of a given scenario
    EvalTaskConfig("hellaswag", 0, task_alias="hellaswag_0shot"),  # 4-way multiple choice commonsense reasoning dataset
    EvalTaskConfig("hellaswag", 10, task_alias="hellaswag_10shot"),  # 4-way MCQ commonsense reasoning dataset
    EvalTaskConfig("lambada", 0),  # predict the endings of text passages
    EvalTaskConfig("openbookqa", 0),  # 4-way multiple choice question answering task that requires multi-step reasoning
    EvalTaskConfig("piqa", 10),  # answer questions based on a passage
    # EvalTaskConfig("squadv2", 10),  # reading comprehension benchmark
    EvalTaskConfig("wsc273", 0),  # Winograd Schema Challenge
    EvalTaskConfig("winogrande", 0),  # Winograd challenge, extended to more domains
]


LEVANTER_LM_EVAL_CORE_TASKS = [
    TaskConfig(task="agieval_lsat_ar", num_fewshot=3),  # 3-shot tests in legal domain
    TaskConfig(task="arc_easy", num_fewshot=10),  # 10-shot, four-way MCQs involving grade 3-9 basic science
    TaskConfig(task="arc_challenge", num_fewshot=10),  # a (harder) version of arc_easy
    TaskConfig(task="boolq", num_fewshot=10),  # answer yes/no questions based on a passage
    TaskConfig(task="commonsense_qa", num_fewshot=10),  # 5-way MCQs based on common-sense, everyday scenarios
    TaskConfig(task="copa", num_fewshot=0),  # use causal reasoning to predict correct outcome of given scenario
    # 4-way multiple choice commonsense reasoning dataset
    TaskConfig(task="hellaswag", num_fewshot=0, task_alias="hellaswag_0shot"),
    TaskConfig(task="hellaswag", num_fewshot=10, task_alias="hellaswag_10shot"),
    TaskConfig(task="lambada", num_fewshot=0),  # predict the endings of text passages
    TaskConfig(task="openbookqa", num_fewshot=0),  # 4-way MCQ, QA task that requires multi-step reasoning
    TaskConfig(task="piqa", num_fewshot=10),  # answer questions based on a passage
    # TaskConfig(task="squadv2", num_fewshot=10),  # reading comprehension benchmark
    TaskConfig(task="wsc273", num_fewshot=0),  # Winograd Schema Challenge
    TaskConfig(task="winogrande", num_fewshot=0),  # Winograd challenge, extended to more domains
]
