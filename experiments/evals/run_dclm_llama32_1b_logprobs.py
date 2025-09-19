from experiments.llama import llama_3_2_1b, llama3_tokenizer
from marin.evaluation.log_probs import default_lm_log_probs
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import mixture_for_evaluation
from experiments.midtraining_datasets import finemath_3_plus
from experiments.defaults import default_tokenize

# Evaluate log probabilities of meta-llama/Llama-3.2-1B on a subset of DCLM baseline
# Uses 1024 samples by default (adjust max_samples_per_dataset as needed)

finemath_3_plus_tokenized_validation = default_tokenize(
    name="finemath_3_plus",
    dataset=finemath_3_plus,
    tokenizer=llama3_tokenizer,
    is_validation=True,
)

# Build evaluation mixture containing just the DCLM baseline (evaluation-only)
eval_data = mixture_for_evaluation(
    {
        "finemath_3_plus": finemath_3_plus_tokenized_validation,
    }
)


llama_32_1b_on_finemath_3_plus_logprobs = default_lm_log_probs(
    checkpoint="meta-llama/Llama-3.2-1B",
    model=llama_3_2_1b,
    data=eval_data,
    checkpoint_is_hf=True,
    per_device_batch_size=4,
    max_samples_per_dataset=1024,
    name="llama-3.2-1B-logprobs-finemath_3_plus-1024",
)


if __name__ == "__main__":
    executor_main(steps=[llama_32_1b_on_finemath_3_plus_logprobs])
