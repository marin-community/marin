from marin.evaluation.evaluation_config import EvalTaskConfig
from experiments.evals.evals import default_eval
import os
from marin.execution.executor import executor_main

marin_prefix = os.environ["MARIN_PREFIX"]

EVAL_TASKS = (
    # EvalTaskConfig("arc_challenge", 10),  # a (harder) version of arc_easy
    # EvalTaskConfig("piqa", 10),  # answer questions based on a passage
    EvalTaskConfig("mathqa", 5, task_alias="mathqa_5shot"),
    # EvalTaskConfig("pubmedqa", 0, task_alias="pubmedqa"),
    EvalTaskConfig("pubmedqa", 5, task_alias="pubmedqa_5shot"),
    # EvalTaskConfig("medqa", 0, task_alias="medqa"),
    # EvalTaskConfig("medqa", 5, task_alias="medqa_5shot")
)

checkpoint_path = f"{marin_prefix}/checkpoints/suhas/pubmed-dclm-llama3.1-8b-0.1B-ra1.0-evaldebug3-1/hf/step-99"

print(checkpoint_path)

eval_step = default_eval(
    checkpoint_path,
    EVAL_TASKS,
)

if __name__ == "__main__":
    executor_main(
        steps=[eval_step],
        description="Evaluate models",
    )


