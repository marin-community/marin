from experiments.evals.engine_configs import DEFAULT_VLLM_ENGINE_KWARGS
from experiments.evals.evals import evaluate_lm_evaluation_harness
from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from experiments.evals.task_configs import EvalTaskConfig
from marin.execution.executor import executor_main

# Insert your model path here
# model_path = "gs://marin-us-central2/checkpoints/llama-8b-control-00f31b/hf/step-210388"
model_path = "gs://marin-us-central2/checkpoints/scaling-law-suite-default-512-9b1182/hf/step-49999"

# key_evals = default_key_evals(
#     step=model_path,
#     resource_config=SINGLE_TPU_V6E_8,
#     model_name="llama-8b-control-00f31b",
# )

step = evaluate_lm_evaluation_harness(
    model_name="llama-small",
    model_path=model_path,
    evals=[EvalTaskConfig(name="minerva_math", num_fewshot=4)],
    max_eval_instances=None,
    engine_kwargs=DEFAULT_VLLM_ENGINE_KWARGS,
    resource_config=SINGLE_TPU_V6E_8,
)

if __name__ == "__main__":
    executor_main(steps=[step])
