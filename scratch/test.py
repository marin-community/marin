from marin.evaluation.run import evaluate
from experiments.evals.task_configs import CORE_TASKS
from marin.evaluation.evaluation_config import EvaluationConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path


evaluate_readability = ExecutorStep(
    name=f"evaluation/lm_evaluation_harness/fineweb-small-readability-1.4b",
    fn=evaluate,
    config=EvaluationConfig(
        evaluator="levanter_lm_evaluation_harness",
        model_name=None,
        model_path="gs://marin-us-central2/checkpoints/fineweb-small-1.4b-readability-5c0e2b/",  # type: ignore
        evaluation_path=this_output_path(),
        evals=CORE_TASKS,
        discover_latest_checkpoint=True,
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[
            evaluate_readability,
        ]
    )