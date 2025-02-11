from experiments.evals.evals import evaluate_alpaca_eval
from experiments.evals.task_configs import EvalTaskConfig
from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.run import evaluate
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

model_name = "tulu3_sft_seed0_alldocs-7334"

# works
gsm8k_cot = ExecutorStep(
    name=f"evaluation/lm_evaluation_harness/{model_name}",
    fn=evaluate,
    config=EvaluationConfig(
        evaluator="lm_evaluation_harness",
        model_name=model_name,
        model_path="gs://marin-us-central2/checkpoints/llama3.1_8b_tulu_3_seed0_alldocs-379cf0/hf/step-7334",
        evaluation_path=this_output_path(),
        evals=versioned([EvalTaskConfig(name="gsm8k_cot", num_fewshot=8)]),
        max_eval_instances=None,
        launch_with_ray=True,
        engine_kwargs={"max_model_len": 8192},
    ),
)

# works
gsm8k = ExecutorStep(
    name=f"evaluation/lm_evaluation_harness/{model_name}",
    fn=evaluate,
    config=EvaluationConfig(
        evaluator="lm_evaluation_harness",
        model_name=model_name,
        model_path="gs://marin-us-central2/checkpoints/llama3.1_8b_tulu_3_seed0_alldocs-379cf0/hf/step-7334",
        evaluation_path=this_output_path(),
        evals=versioned([EvalTaskConfig(name="gsm8k", num_fewshot=5)]),
        max_eval_instances=None,
        launch_with_ray=True,
        engine_kwargs={"max_model_len": 8192},
    ),
)

# works
drop = ExecutorStep(
    name=f"evaluation/lm_evaluation_harness/{model_name}",
    fn=evaluate,
    config=EvaluationConfig(
        evaluator="lm_evaluation_harness",
        model_name=model_name,
        model_path="gs://marin-us-central2/checkpoints/llama3.1_8b_tulu_3_seed0_alldocs-379cf0/hf/step-7334",
        evaluation_path=this_output_path(),
        evals=versioned([EvalTaskConfig(name="drop", num_fewshot=3)]),
        max_eval_instances=None,
        launch_with_ray=True,
        engine_kwargs={"max_model_len": 8192},
    ),
)

ifeval = ExecutorStep(
    name=f"evaluation/lm_evaluation_harness/{model_name}",
    fn=evaluate,
    config=EvaluationConfig(
        evaluator="lm_evaluation_harness",
        model_name=model_name,
        model_path="gs://marin-us-central2/checkpoints/llama3.1_8b_tulu_3_seed0_alldocs-379cf0/hf/step-7334",
        evaluation_path=this_output_path(),
        evals=versioned([EvalTaskConfig(name="ifeval", num_fewshot=0)]),
        max_eval_instances=None,
        launch_with_ray=True,
        engine_kwargs={"max_model_len": 8192},
    ),
)

# humaneval_score = ExecutorStep(
#         name=f"evaluation/lm_evaluation_harness/{model_name}",
#         fn=evaluate,
#         config=EvaluationConfig(
#             evaluator="lm_evaluation_harness",
#             model_name=model_name,
#             model_path="gs://marin-us-central2/checkpoints/llama3.1_8b_tulu_3_seed0_alldocs-379cf0/hf/step-7334",
#             evaluation_path=this_output_path(),
#             evals=versioned([EvalTaskConfig(name="humaneval")]),
#             max_eval_instances=None,
#             launch_with_ray=True,
#             engine_kwargs={"max_model_len": 8192},
#         ),
#     )

alpaca_eval = evaluate_alpaca_eval(
    model_name="tulu3_sft_seed0_alldocs-7334",
    model_path="gs://marin-us-central2/checkpoints/llama3.1_8b_tulu_3_seed0_alldocs-379cf0/hf/step-7334",
)
steps = [
    # alpaca_eval,
    # gsm8k_cot,
    # drop,
    # ifeval,
    #     humaneval_score,
]

if __name__ == "__main__":
    executor_main(steps=steps)
