
from experiments.evals.task_configs import CORE_TASKS
from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.run import evaluate
from marin.execution.executor import ExecutorStep, executor_main, this_output_path

evaluate_resiliparse_custom_fork = ExecutorStep(
    name="evaluation/lm_evaluation_harness/fineweb-small-resiliparse-custom-fork-1.4b",
    fn=evaluate,
    config=EvaluationConfig(
        evaluator="levanter_lm_evaluation_harness",
        model_name=None,
        model_path="gs://marin-us-central2/checkpoints/fineweb-small-resiliparse-custom-fork-1.4b-9518f2",  # type: ignore
        evaluation_path=this_output_path(),
        evals=CORE_TASKS,
        discover_latest_checkpoint=True,
    ),
)

if __name__ == "__main__":
    executor_main(steps=[evaluate_resiliparse_custom_fork])
