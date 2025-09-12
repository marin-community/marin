from experiments.evals.evals import evaluate_helm, evaluate_lm_evaluation_harness, evaluate_levanter_lm_evaluation_harness
from marin.evaluation.evaluation_config import EvalTaskConfig
# from experiments.models import llama_3_1_8b_instruct, tulu_3_1_8b_instruct
# from experiments.tootsie.exp1237_starling_sft import mixture_sft_deeper_starling
from marin.execution.executor import executor_main, ExecutorStep
from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.run import evaluate
from marin.execution.executor import this_output_path
from experiments.evals.resource_configs import ResourceConfig

resource_config = ResourceConfig(num_tpu=1, tpu_type="TPU-v6e-4", strategy="STRICT_PACK")

"""
Note for people trying to do evals:
- The difference between evaluate_lm_evaluation_harness and evaluate_levanter_lm_evaluation_harness is that the latter uses the vLLM engine and the former uses the Levanter engine.
- The levanter engine can only compute loglikelihoods and not completions. So, we have to use use lm_evaluation_harness for typical evals.

"""


if __name__ == "__main__":
    model_name = "deeper_starling_sft_nemotron_and_openthoughts3"
    model_path = "gs://marin-us-central2/checkpoints/sft/deeper_starling_sft_nemotron_and_openthoughts3/hf/step-1540000"

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

    max_eval_instances = None
    # Configuration to fix rope_scaling original_max_position_embeddings issue
    # The error occurs because original_max_position_embeddings (8192) > max_position_embeddings (4096)
    engine_kwargs = {
        "max_model_len": 4096,
    }
    

    lm_evaluation_harness_eval = evaluate_lm_evaluation_harness(
        model_name,
        model_path,
        [
            # EvalTaskConfig("agieval_lsat_ar", num_fewshot=3, task_alias="agieval_lsat_ar_3shot"),  # 3-shot tests in legal domain
            # EvalTaskConfig("arc_easy", num_fewshot=10, task_alias="arc_easy_10shot"),  # 10-shot, four-way MCQ questions involving grade 3-9 basic science
            # EvalTaskConfig("arc_challenge", num_fewshot=10, task_alias="arc_challenge_10shot"),  # a (harder) version of arc_easy
            # EvalTaskConfig("boolq", num_fewshot=10, task_alias="boolq_10shot"),  # answer yes/no questions based on a passage
            # EvalTaskConfig("commonsense_qa", num_fewshot=10, task_alias="commonsense_qa_10shot"),  # 5-way multiple-choice questions based on common-sense, everyday scenarios
            # EvalTaskConfig("copa", num_fewshot=0, task_alias="copa_0shot"),  # use causal reasoning to predict the correct outcome of a given scenario
            EvalTaskConfig(name="drop", num_fewshot=0, task_alias="drop_0shot"),
            EvalTaskConfig(name="gsm8k_cot", num_fewshot=8, task_alias="gsm8k_cot"),
            # EvalTaskConfig("hellaswag", 0, task_alias="hellaswag_0shot"),  # 4-way multiple choice commonsense reasoning dataset
            # EvalTaskConfig("hellaswag", num_fewshot=10, task_alias="hellaswag_10shot"),  # 4-way MCQ commonsense reasoning dataset
            EvalTaskConfig(name="humaneval", num_fewshot=10, task_alias="humaneval_10shot"),
            EvalTaskConfig(name="bbh_cot_fewshot", num_fewshot=3, task_alias="bbh_3shot"),
            EvalTaskConfig(name="ifeval", num_fewshot=0, task_alias="ifeval_0shot"),
            # EvalTaskConfig("lambada_openai", num_fewshot=0, task_alias="lambada_openai_0shot"),  # predict the endings of text passages
            # EvalTaskConfig("leaderboard_gpqa", num_fewshot=0, task_alias="gpqa_0shot"),
            # EvalTaskConfig("leaderboard_ifeval", num_fewshot=0, task_alias="lb_ifeval_0shot"),
            EvalTaskConfig("leaderboard_math_hard", num_fewshot=4, task_alias="lb_math_4shot"),
            EvalTaskConfig("leaderboard_mmlu_pro", num_fewshot=5, task_alias="mmlu_5shot"),
            EvalTaskConfig(name="minerva_math", num_fewshot=4, task_alias="math_4shot"),
            EvalTaskConfig("mmlu", num_fewshot=0, task_alias="mmlu_0shot"),
            EvalTaskConfig("mmlu", num_fewshot=5, task_alias="mmlu_5shot"),
            EvalTaskConfig("openbookqa", num_fewshot=0, task_alias="openbookqa_0shot"),  # 4-way multiple choice question answering task that requires multi-step reasoning
            EvalTaskConfig("piqa", num_fewshot=10, task_alias="piqa_10shot"),  # answer questions based on a passage
            # EvalTaskConfig("truthfulqa_mc2", num_fewshot=6, task_alias="truthqa_6shot"),
            # EvalTaskConfig("wsc273", num_fewshot=0, task_alias="wsc273_0shot"),  # Winograd Schema Challenge
            # EvalTaskConfig("winogrande", num_fewshot=0, task_alias="winogrande_0shot"),  # Winograd challenge, extended to more domains
        ],
        max_eval_instances=max_eval_instances,
        engine_kwargs=engine_kwargs,
        resource_config=resource_config,
        apply_chat_template=True,
    )

    # Collect all steps properly
    all_steps = []
    # all_steps.extend(default_sft_eval(mixture_sft_deeper_starling))
    # all_steps.extend(default_sft_eval(llama_3_1_8b_instruct))
    # all_steps.extend(default_sft_eval(tulu_3_1_8b_instruct))
    all_steps.append(lm_evaluation_harness_eval)
    # all_steps.append(levanter_lm_evaluation_harness_eval)
    all_steps.append(helm_eval)

    executor_main(steps=all_steps)