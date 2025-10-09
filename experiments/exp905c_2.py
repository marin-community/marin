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

resource_config = ResourceConfig(num_tpu=4, tpu_type="TPU-v6e-4", strategy="STRICT_PACK")

"""
Note for people trying to do evals:
- The difference between evaluate_lm_evaluation_harness and evaluate_levanter_lm_evaluation_harness is that the latter uses the vLLM engine and the former uses the Levanter engine.
- The levanter engine can only compute loglikelihoods and not completions. So, we have to use use lm_evaluation_harness for typical evals.

"""


if __name__ == "__main__":
    # Quiet ray logs for this experiment
    # logging.getLogger("ray").setLevel(logging.WARNING)

    model_name = "deeper_starling_sft_nemotron_and_openthoughts3"
    model_path = "gs://marin-us-central2/checkpoints/sft/deeper_starling_sft_nemotron_and_openthoughts3/hf/step-1540000"
    # model_path = "gs://marin-us-central2/models/llama-3.1-8b"
    
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
        #>>> 3-shot tests in legal domain
        # EvalTaskConfig("agieval_lsat_ar", num_fewshot=3, task_alias="agieval_lsat_ar_3shot"),

        #>>> AIME dataset
        # EvalTaskConfig("aime", num_fewshot=0, task_alias="aime_0shot"),
        EvalTaskConfig("aime24", num_fewshot=0, task_alias="aime24_0shot_test1"),
        # EvalTaskConfig("aime25", num_fewshot=0, task_alias="aime25_0shot"),

        #>>> 10-shot, four-way MCQ questions involving grade 3-9 basic science
        # EvalTaskConfig("arc_easy", num_fewshot=10, task_alias="arc_easy_10shot"),

        #>>> Harder version of arc_easy
        # EvalTaskConfig("arc_challenge", num_fewshot=10, task_alias="arc_challenge_10shot"),

        #>>> answer yes/no questions based on a passage
        # EvalTaskConfig("boolq", num_fewshot=10, task_alias="boolq_10shot"),

        #>>> 5-way multiple-choice questions based on common-sense, everyday scenarios
        # EvalTaskConfig("commonsense_qa", num_fewshot=10, task_alias="commonsense_qa_10shot"),

        #>>> use causal reasoning to predict the correct outcome of a given scenario
        # EvalTaskConfig("copa", num_fewshot=0, task_alias="copa_0shot"),

        # EvalTaskConfig(name="drop", num_fewshot=0, task_alias="drop_0shot"),

        # EvalTaskConfig(name="gsm8k_cot", num_fewshot=8, task_alias="gsm8k_cot"),

        #>>> 4-way multiple choice commonsense reasoning dataset
        # EvalTaskConfig("hellaswag", 0, task_alias="hellaswag_0shot"),
        # EvalTaskConfig("hellaswag", num_fewshot=10, task_alias="hellaswag_10shot"),

        #>>> 4-way MCQ commonsense reasoning dataset
        # EvalTaskConfig(name="humaneval_instruct", num_fewshot=0, task_alias="humanevalInstruct_0shot"),

        # EvalTaskConfig(name="bbh_cot_fewshot", num_fewshot=3, task_alias="bbh_3shot"),

        # EvalTaskConfig(name="ifeval", num_fewshot=0, task_alias="ifeval_0shot"),

        #>>> predict the endings of text passages
        # EvalTaskConfig("lambada_openai", num_fewshot=0, task_alias="lambada_openai_0shot"),

        #>>> Leaderboard tasks
        # EvalTaskConfig("leaderboard_gpqa", num_fewshot=0, task_alias="gpqa_0shot"),
        # EvalTaskConfig("leaderboard_ifeval", num_fewshot=0, task_alias="lb_ifeval_0shot"),
        # EvalTaskConfig("leaderboard_math_hard", num_fewshot=4, task_alias="lb_math_4shot"),
        # EvalTaskConfig("leaderboard_mmlu_pro", num_fewshot=5, task_alias="mmlu_5shot"),

        # EvalTaskConfig(name="minerva_math", num_fewshot=4, task_alias="math_4shot"),

        # EvalTaskConfig("mmlu", num_fewshot=0, task_alias="mmlu_0shot"),
        # EvalTaskConfig("mmlu", num_fewshot=1, task_alias="mmlu_1shot"),

        #>>> 4-way multiple choice question answering task that requires multi-step reasoning
        # EvalTaskConfig("openbookqa", num_fewshot=0, task_alias="openbookqa_0shot"),

        #>>> answer questions based on a passage
        # EvalTaskConfig("piqa", num_fewshot=10, task_alias="piqa_10shot"),

        # EvalTaskConfig("truthfulqa_mc2", num_fewshot=6, task_alias="truthqa_6shot"),
        
        #>>> Winograd Schema Challenge
        # EvalTaskConfig("wsc273", num_fewshot=0, task_alias="wsc273_0shot"),
        
        #>>> Winograd challenge, extended to more domains
        # EvalTaskConfig("winogrande", num_fewshot=0, task_alias="winogrande_0shot"),
    ]
    
    all_steps = []
    for eval_task in eval_tasks:
        lm_eval_task_step = evaluate_levanter_lm_evaluation_harness(
            model_name,
            model_path,
            evals = [eval_task],
            max_eval_instances=5,
            resource_config=resource_config,
            apply_chat_template=True,
            max_gen_toks=4096,  # Override max_gen_toks to 4096 tokens
        )
        all_steps.append(lm_eval_task_step)

    # Collect all steps properly
    # all_steps.extend(default_sft_eval(mixture_sft_deeper_starling))
    # all_steps.extend(default_sft_eval(llama_3_1_8b_instruct))
    # all_steps.extend(default_sft_eval(tulu_3_1_8b_instruct))
    # all_steps.append(helm_eval)  # Commented out to avoid TPU resource conflict

    executor_main(steps=all_steps)