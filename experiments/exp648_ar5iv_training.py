"""
Test different html->text transformation methods (on Ar5iv Dump, train 1.4B models).
https://github.com/stanford-crfm/marin/issues/648
"""

import logging

from experiments.defaults import default_train
from experiments.dolma.tokenize_dolma import DOLMA_OLMO_MIXTURE_WEIGHTS, tokenize_dolma_steps
from experiments.evals.evals import default_eval
from experiments.llama import llama_1_4b, llama_1_4b_train_config
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from marin.utils import fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ray")

EXPERIMENT_TAG = ["arxiv_no_problem_subbed_dolma"]

weights = DOLMA_OLMO_MIXTURE_WEIGHTS.pop("dolma/arxiv")
DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/arxiv-subbed-no-problem-readability-no-references"] = weights

arxiv_no_problem_readability_path = "gs://marin-us-central2/documents/ar5iv/ar5iv-04-2024-no-problem-53fc69/readability"
arxiv_no_problem_readability_files = fsspec_glob(f"{arxiv_no_problem_readability_path}/*.jsonl.gz")
logger.info(f"Found {len(arxiv_no_problem_readability_files)} files in {arxiv_no_problem_readability_files}")
arxiv_no_problem_readability_subbed_dolma_tokenized = tokenize_dolma_steps(
    substitute={"arxiv": arxiv_no_problem_readability_files},
    prefix="no-problem-readability-no-references",
)
arxiv_no_problem_readability_subbed_dolma_llama3_tokenized = lm_mixture_data_config(
    components=arxiv_no_problem_readability_subbed_dolma_tokenized,
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS,
)

arxiv_no_problem_readablity_1_4b_subbed_dolma_model = default_train(
    name="dolma-arxiv_no_problem-subbed-1.4b-readablity-no-references",
    tokenized=arxiv_no_problem_readability_subbed_dolma_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=EXPERIMENT_TAG,
)

arxiv_no_problem_readablity_1_4b_subbed_dolma_evals = default_eval(step=arxiv_no_problem_readablity_1_4b_subbed_dolma_model)


weights = DOLMA_OLMO_MIXTURE_WEIGHTS.pop("dolma/arxiv-subbed-no-problem-readability-no-references")
DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/arxiv-subbed-no-problem-resiliparse-with-preserving-formatting-no-references"] = weights

arxiv_no_problem_resiliparse_with_preserve_formatting_path = "gs://marin-us-central2/documents/ar5iv/ar5iv-04-2024-no-problem-8ca63a/resiliparse-with-preserving-formatting"
arxiv_no_problem_resiliparse_with_preserve_formatting_files = fsspec_glob(f"{arxiv_no_problem_resiliparse_with_preserve_formatting_path}/*.jsonl.gz")
arxiv_no_problem_resiliparse_with_preserve_formatting_subbed_dolma_tokenized = tokenize_dolma_steps(
    substitute={"arxiv": arxiv_no_problem_resiliparse_with_preserve_formatting_files},
    prefix="no-problem-resiliparse-with-preserving-formatting-no-references",
)
arxiv_no_problem_resiliparse_with_preserve_formatting_subbed_dolma_llama3_tokenized = lm_mixture_data_config(
    components=arxiv_no_problem_resiliparse_with_preserve_formatting_subbed_dolma_tokenized,
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS
)
arxiv_no_problem_resiliparse_with_preserve_formatting_subbed_dolma_1_4b_model = default_train(
    name="dolma-arxiv_no_problem-subbed-1.4b-resiliparse-with-preserving-formatting-no-references",
    tokenized=arxiv_no_problem_resiliparse_with_preserve_formatting_subbed_dolma_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=EXPERIMENT_TAG,
)

arxiv_no_problem_resiliparse_with_preserve_formatting_subbed_dolma_1_4b_evals = default_eval(step=arxiv_no_problem_resiliparse_with_preserve_formatting_subbed_dolma_1_4b_model)


if __name__ == "__main__":
    arxiv_no_problem_readability_subbed_dolma_tokenized.update(arxiv_no_problem_resiliparse_with_preserve_formatting_subbed_dolma_tokenized)
    tokenization_steps = arxiv_no_problem_readability_subbed_dolma_tokenized.values()

    executor_main(
        steps=[
            *tokenization_steps,
            arxiv_no_problem_readablity_1_4b_subbed_dolma_model,
            arxiv_no_problem_readablity_1_4b_subbed_dolma_evals,
            arxiv_no_problem_resiliparse_with_preserve_formatting_subbed_dolma_1_4b_model,
            arxiv_no_problem_resiliparse_with_preserve_formatting_subbed_dolma_1_4b_evals,
        ]
    )
