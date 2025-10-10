from experiments.evals.evals import evaluate_helm, evaluate_lm_evaluation_harness, evaluate_levanter_lm_evaluation_harness
import logging
from marin.evaluation.evaluation_config import EvalTaskConfig
# from experiments.models import llama_3_1_8b_instruct, tulu_3_1_8b_instruct
# from experiments.tootsie.exp1237_starling_sft import mixture_sft_deeper_starling
from marin.execution.executor import executor_main, ExecutorStep
from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.run import evaluate
from marin.execution.executor import this_output_path
from experiments.evals.resource_configs import ResourceConfig

resource_config = ResourceConfig(num_tpu=4, tpu_type="TPU-v5p-8", strategy="STRICT_PACK")

"""
Note for people trying to do evals:
- The difference between evaluate_lm_evaluation_harness and evaluate_levanter_lm_evaluation_harness is that the latter uses the vLLM engine and the former uses the Levanter engine.
- The levanter engine can only compute loglikelihoods and not completions. So, we have to use use lm_evaluation_harness for typical evals.

"""


if __name__ == "__main__":
    # Quiet ray logs for this experiment
    # logging.getLogger("ray").setLevel(logging.WARNING)

    model_name = "bison"
    model_path = "gs://marin-us-central1/checkpoints/tootsie-32b-cooldown-bison-adamc/hf/tootsie-32b-cooldown-bison-adamc/step-192000"
    
    # Run all evaluations on all models
    helm_eval = ExecutorStep(
        name="evaluation/helm/exp905c",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="helm",
            model_name=model_name,
            model_path=model_path,
            evaluation_path=this_output_path(),
            evals=[
                EvalTaskConfig(name="mmlu", num_fewshot=0),
                EvalTaskConfig(name="lite", num_fewshot=0)
            ],
            apply_chat_template=True,
            resource_config=resource_config
        ),
        pip_dependency_groups=["eval", "transformers", "tiktoken", "sentencepiece"],
    )

    # NOTE(chiheem 2025-10-01): We may want to run the lm-eval tasks as separate steps so that we can avoid
    # `out of pages` error.
    eval_tasks = [
        EvalTaskConfig(name="gsm8k", num_fewshot=5, task_alias="gsm8k_5shot"),
        EvalTaskConfig(name="humaneval", num_fewshot=0, task_alias="humaneval_0shot"),
        # # requires pip install lm-eval[math]
        EvalTaskConfig(name="minerva_math", num_fewshot=4, task_alias="math_500_4shot"),
    ]
    
    all_steps = []
    for eval_task in eval_tasks:
        lm_eval_task_step = evaluate_levanter_lm_evaluation_harness(
            model_name,
            model_path,
            evals = [eval_task],
            max_eval_instances=None,
            resource_config=resource_config,
            apply_chat_template=True,
            max_gen_toks=1024,  # Override max_gen_toks to 4096 tokens
        )
        all_steps.append(lm_eval_task_step)

    # Collect all steps properly
    # all_steps.extend(default_sft_eval(mixture_sft_deeper_starling))
    # all_steps.extend(default_sft_eval(llama_3_1_8b_instruct))
    # all_steps.extend(default_sft_eval(tulu_3_1_8b_instruct))
    # all_steps.append(helm_eval)  # Commented out to avoid TPU resource conflict

    executor_main(steps=all_steps)