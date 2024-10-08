import dataclasses

import draccus

from marin.execution.executor import (
    ExecutorMainConfig,
    ExecutorStep,
    executor_main,
    this_output_path,
)

USER = "abhinavg"

############################################################
# Training

from marin.training import TrainLmOnPodConfig, run_levanter_train_lm  # noqa

training_config = draccus.load(TrainLmOnPodConfig, open("config/training/dclm_1b_1x.yaml"))

train_step = ExecutorStep(
    name=f"checkpoints/dclm-1b-1x-{USER}",
    fn=run_levanter_train_lm,
    config=dataclasses.replace(
        training_config,
        output_path=this_output_path(),
        tpu_type="v4-64",
    ),
)

############################################################
# Evaluate

# from marin.evaluation.evaluation_config import EvaluationConfig
# from marin.evaluation.run import evaluate
#
# evaluate_step = ExecutorStep(
#     name=f"evaluation/hello_world_fw-{USER}",
#     fn=evaluate,
#     config=EvaluationConfig(
#         evaluator="helm",
#         model_name="pf5pe4ut/step-600",
#         # TODO: replace this with `output_path_of(train_step)`
#         model_path="gs://marin-us-central2/checkpoints/quickstart_single_script_docker_test_09_18/pf5pe4ut/hf/pf5pe4ut/step-600",
#         evaluation_path=this_output_path(),
#         evals=["mmlu"],
#     ),
# )

############################################################

if __name__ == "__main__":
    executor_main(
        ExecutorMainConfig(force_run=[f"checkpoints/dclm-1b-1x-{USER}"]),
        steps=[
            # train_quality_step,  # Not used  (TODO: fails right now)
            train_step,
        ],
    )
