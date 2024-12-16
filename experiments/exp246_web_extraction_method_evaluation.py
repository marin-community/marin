from marin.evaluation.run import evaluate
from experiments.evals.task_configs import CORE_TASKS
from marin.evaluation.evaluation_config import EvaluationConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path


evaluate_readability = ExecutorStep(
    name=f"evaluation/levanter_lm_evaluation_harness/fineweb-small-readability-1.4b",
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


evaluate_trafilatura_default = ExecutorStep(
    name=f"evaluation/levanter_lm_evaluation_harness/fineweb-small-trafilatura-default-1.4b",
    fn=evaluate,
    config=EvaluationConfig(
        evaluator="levanter_lm_evaluation_harness",
        model_name=None,
        model_path="gs://marin-us-central2/checkpoints/fineweb-small-1.4b-trafilatura-6ba941/",  # type: ignore
        evaluation_path=this_output_path(),
        evals=CORE_TASKS,
        discover_latest_checkpoint=True,
    ),
)


evaluate_trafilatura_favor_precision = ExecutorStep(
    name=f"evaluation/levanter_lm_evaluation_harness/fineweb-small-trafilatura-favor-precision-1.4b",
    fn=evaluate,
    config=EvaluationConfig(
        evaluator="levanter_lm_evaluation_harness",
        model_name=None,
        model_path="gs://marin-us-central2/checkpoints/fineweb-small-1.4b-trafilatura-favor-precision-b4f367/",  # type: ignore
        evaluation_path=this_output_path(),
        evals=CORE_TASKS,
        discover_latest_checkpoint=True,
    ),
)


evaluate_resiliparse_default = ExecutorStep(
    name=f"evaluation/levanter_lm_evaluation_harness/fineweb-small-resiliparse-default-1.4b",
    fn=evaluate,
    config=EvaluationConfig(
        evaluator="levanter_lm_evaluation_harness",
        model_name=None,
        model_path="gs://marin-us-central2/checkpoints/fineweb-small-1.4b-resiliparse-49c4d6/",  # type: ignore
        evaluation_path=this_output_path(),
        evals=CORE_TASKS,
        discover_latest_checkpoint=True,
    ),
)


evaluate_resiliparse_preserve_formatting = ExecutorStep(
    name=f"evaluation/levanter_lm_evaluation_harness/fineweb-small-resiliparse-preserve-formatting-1.4b",
    fn=evaluate,
    config=EvaluationConfig(
        evaluator="levanter_lm_evaluation_harness",
        model_name=None,
        model_path="gs://marin-us-central2/checkpoints/fineweb-small-1.4b-resiliparse-preserve-formatting-792c36/",  # type: ignore
        evaluation_path=this_output_path(),
        evals=CORE_TASKS,
        discover_latest_checkpoint=True,
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[
            evaluate_readability,
            # evaluate_trafilatura_default,
            # evaluate_trafilatura_favor_precision,
            # evaluate_resiliparse_default,
            # evaluate_resiliparse_preserve_formatting,
        ]
    )
