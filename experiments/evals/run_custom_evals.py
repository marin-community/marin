from experiments.evals.evals import evaluate_lm_evaluation_harness
from experiments.evals.task_configs import KEY_GENERATION_TASKS
from experiments.evals.resource_configs import SINGLE_TPU_V4_8
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main
from levanter.eval_harness import TaskConfig

BASE_TASKS = (
    EvalTaskConfig(name="arc_easy", num_fewshot=0),
    EvalTaskConfig(name="sciq", num_fewshot=0),
)

evaluate_step = evaluate_lm_evaluation_harness(
    None,
    "gs://marin-us-central2/checkpoints/two_stage/150m4k-4.3B-flanx0.001x32-c4-rr0.88-rs0.12-wsd-0.003-0.10-e/hf/step-1023",
    BASE_TASKS,
    resource_config=SINGLE_TPU_V4_8,
)

if __name__ == "__main__":
    executor_main(steps=[evaluate_step])