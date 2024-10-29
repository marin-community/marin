from marin.execution.executor import ExecutorMainConfig, executor_main
from .eval_executor_steps import create_helm_executor_step


def main():
    executor_main_config = ExecutorMainConfig()
    steps = [
        # TODO: make this configurable?
        create_helm_executor_step(
            model_name="pf5pe4ut/step-600",
            model_path="gs://marin-us-central2/checkpoints/quickstart_single_script_docker_test_09_18/"
            "pf5pe4ut/hf/pf5pe4ut/step-600",
            evals=["mmlu"],
        ),
    ]
    executor_main(executor_main_config, steps=steps)


if __name__ == "__main__":
    main()
