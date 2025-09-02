from experiments.evals.evals import default_sft_eval, evaluate_helm
from marin.evaluation.evaluation_config import EvalTaskConfig
from experiments.models import llama_3_1_8b_instruct, tulu_3_1_8b_instruct
from experiments.tootsie.exp1237_starling_sft import mixture_sft_deeper_starling
from marin.execution.executor import executor_main, ExecutorStep
from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.run import evaluate
from marin.execution.executor import this_output_path


if __name__ == "__main__":
    # Run all evaluations on all models
    helm_eval = ExecutorStep(
        name="evaluation/helm/deeper_starling_sft_nemotron_and_openthoughts3/step-1536000",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="helm",
            model_name="deeper_starling_sft_nemotron_and_openthoughts3/step-1536000",
            model_path="gs://marin-us-central2/checkpoints/sft/deeper_starling_sft_nemotron_and_openthoughts3/hf/step-1536000",
            evaluation_path=this_output_path(),
            evals=[EvalTaskConfig(name="mmlu", num_fewshot=0), EvalTaskConfig(name="lite", num_fewshot=0)],
        ),
    )

    # Collect all steps properly
    all_steps = []
    all_steps.extend(default_sft_eval(mixture_sft_deeper_starling))
    all_steps.extend(default_sft_eval(llama_3_1_8b_instruct))
    all_steps.extend(default_sft_eval(tulu_3_1_8b_instruct))
    all_steps.append(helm_eval)

    executor_main(steps=all_steps)