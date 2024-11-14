from experiments.evals.evals import evaluate_lm_evaluation_harness
from marin.execution.executor import executor_main

evaluate_readability = evaluate_lm_evaluation_harness(
    model_name="fineweb-small-readability-1.4b",
    model_path="gs://marin-us-central2/checkpoints/fineweb-small-1.4b-readability-5c0e2b/hf/step-9999",
    evals=["mmlu"],
)

evaluate_trafilatura_default = evaluate_lm_evaluation_harness(
    model_name="fineweb-small-trafilatura-default-1.4b",
    model_path="gs://marin-us-central2/checkpoints/fineweb-small-1.4b-trafilatura-6ba941/hf/step-9999",
    evals=["mmlu"],
)

evaluate_trafilatura_favor_precision = evaluate_lm_evaluation_harness(
    model_name="fineweb-small-trafilatura-favor-precision-1.4b",
    model_path="gs://marin-us-central2/checkpoints/fineweb-small-1.4b-trafilatura-favor-precision-b4f367/hf/step-9999",
    evals=["mmlu"],
)

evaluate_resiliparse_default = evaluate_lm_evaluation_harness(
    model_name="fineweb-small-resiliparse-default-1.4b",
    model_path="gs://marin-us-central2/checkpoints/fineweb-small-1.4b-resiliparse-49c4d6/hf/step-9999",
    evals=["mmlu"],
)

evaluate_resiliparse_preserve_formatting = evaluate_lm_evaluation_harness(
    model_name="fineweb-small-resiliparse-preserve-formatting-1.4b",
    model_path="gs://marin-us-central2/checkpoints/fineweb-small-1.4b-resiliparse-preserve-formatting-792c36/hf/step-9999",
    evals=["mmlu"],
)

if __name__ == "__main__":
    executor_main(steps=[
        evaluate_readability,
        evaluate_trafilatura_default,
        evaluate_trafilatura_favor_precision,
        evaluate_resiliparse_default,
        evaluate_resiliparse_preserve_formatting,
    ])