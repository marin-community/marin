"""
Test different html->text transformation methods (on Wikipedia Dump, train 1.4B models).
https://github.com/stanford-crfm/marin/issues/647
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

EXPERIMENT_TAG = ["wiki_subbed_dolma"]

weights = DOLMA_OLMO_MIXTURE_WEIGHTS.pop("dolma/wiki")
DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/wiki-subbed-readability-no-references"] = weights

wiki_readability_path = "gs://marin-us-central2/documents/wikipedia-readability-a03369/20241201"
wiki_readability_files = fsspec_glob(f"{wiki_readability_path}/*.jsonl.gz")
logger.info(f"Found {len(wiki_readability_files)} files in {wiki_readability_files}")
wiki_readability_subbed_dolma_tokenized = tokenize_dolma_steps(
    substitute={"wiki": wiki_readability_files},
    prefix="readability-no-references",
)
wiki_readability_subbed_dolma_llama3_tokenized = lm_mixture_data_config(
    components=wiki_readability_subbed_dolma_tokenized,
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS,
)

wiki_readablity_1_4b_subbed_dolma_model = default_train(
    name="dolma-wiki-subbed-1.4b-readablity-no-references",
    tokenized=wiki_readability_subbed_dolma_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=EXPERIMENT_TAG,
)

wiki_readablity_1_4b_subbed_dolma_evals = default_eval(step=wiki_readablity_1_4b_subbed_dolma_model)


weights = DOLMA_OLMO_MIXTURE_WEIGHTS.pop("dolma/wiki-subbed-readability-no-references")
DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/wiki-subbed-resiliparse-with-preserving-formatting-no-references"] = weights

wiki_resiliparse_with_preserve_formatting_path = "gs://marin-us-central2/documents/wikipedia-resiliparse-with-preserving-formatting-0cccb4/20241201"
wiki_resiliparse_with_preserve_formatting_files = fsspec_glob(f"{wiki_resiliparse_with_preserve_formatting_path}/*.jsonl.gz")
wiki_resiliparse_with_preserve_formatting_subbed_dolma_tokenized = tokenize_dolma_steps(
    substitute={"wiki": wiki_resiliparse_with_preserve_formatting_files},
    prefix="resiliparse-with-preserving-formatting-no-references",
)
wiki_resiliparse_with_preserve_formatting_subbed_dolma_llama3_tokenized = lm_mixture_data_config(
    components=wiki_resiliparse_with_preserve_formatting_subbed_dolma_tokenized,
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS
)
wiki_resiliparse_with_preserve_formatting_subbed_dolma_1_4b_model = default_train(
    name="dolma-wiki-subbed-1.4b-resiliparse-with-preserving-formatting-no-references",
    tokenized=wiki_resiliparse_with_preserve_formatting_subbed_dolma_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=EXPERIMENT_TAG,
)

wiki_resiliparse_with_preserve_formatting_subbed_dolma_1_4b_evals = default_eval(step=wiki_resiliparse_with_preserve_formatting_subbed_dolma_1_4b_model)


if __name__ == "__main__":
    wiki_readability_subbed_dolma_tokenized.update(wiki_resiliparse_with_preserve_formatting_subbed_dolma_tokenized)
    tokenization_steps = wiki_readability_subbed_dolma_tokenized.values()

    executor_main(
        steps=[
            *tokenization_steps,
            wiki_readablity_1_4b_subbed_dolma_model,
            wiki_readablity_1_4b_subbed_dolma_evals,
            wiki_resiliparse_with_preserve_formatting_subbed_dolma_1_4b_model,
            wiki_resiliparse_with_preserve_formatting_subbed_dolma_1_4b_evals,
        ]
    )
