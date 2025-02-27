"""
Test different html->text transformation methods (on Wikipedia Dump, train 1.4B models).
https://github.com/stanford-crfm/marin/issues/647
"""

import logging

from experiments.defaults import default_tokenize, default_train
from experiments.dolma.tokenize_dolma import DOLMA_OLMO_MIXTURE_WEIGHTS, tokenize_dolma_steps
from experiments.evals.evals import default_eval
from experiments.exp575_wikipedia_markdownify import wikipedia_readability, wikipedia_resiliparse_with_pf
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ray")

EXPERIMENT_TAG = ["wiki_subbed_dolma"]

tokenized_dolma_steps = tokenize_dolma_steps()

wiki_readability_tokenized = default_tokenize(
    name="dolma-wiki-readability-tokenized",
    dataset=wikipedia_readability,
    tokenizer=llama3_tokenizer,
)

dolma_wiki_readability_tokenization_steps = dict(tokenized_dolma_steps, {"wiki-readability": wiki_readability_tokenized})
wiki_readability_weights = dict(
    DOLMA_OLMO_MIXTURE_WEIGHTS, {"wiki-readability": DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/wiki"]}
)
wiki_readability_weights.pop("dolma/wiki")

wiki_readability_subbed_dolma_llama3_tokenized = lm_mixture_data_config(
    components=dolma_wiki_readability_tokenization_steps,
    weights=wiki_readability_weights,
)

wiki_readablity_1_4b_subbed_dolma_model = default_train(
    name="dolma-wiki-subbed-1.4b-readablity-no-references",
    tokenized=wiki_readability_subbed_dolma_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=EXPERIMENT_TAG,
)

wiki_readablity_1_4b_subbed_dolma_evals = default_eval(step=wiki_readablity_1_4b_subbed_dolma_model)

wiki_resiliparse_with_preserve_formatting_tokenized = default_tokenize(
    name="dolma-wiki-resiliparse-with-preserving-formatting-tokenized",
    dataset=wikipedia_resiliparse_with_pf,
    tokenizer=llama3_tokenizer,
)

dolma_wiki_resiliparse_with_preserve_formatting_tokenization_steps = dict(
    tokenized_dolma_steps,
    {"wiki-resiliparse-with-preserving-formatting": wiki_resiliparse_with_preserve_formatting_tokenized},
)
wiki_resiliparse_with_preserve_formatting_weights = dict(
    DOLMA_OLMO_MIXTURE_WEIGHTS, {"wiki-resiliparse-with-preserving-formatting": DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/wiki"]}
)
wiki_resiliparse_with_preserve_formatting_weights.pop("dolma/wiki")
wiki_resiliparse_with_preserve_formatting_subbed_dolma_llama3_tokenized = lm_mixture_data_config(
    components=dolma_wiki_resiliparse_with_preserve_formatting_tokenization_steps,
    weights=wiki_resiliparse_with_preserve_formatting_weights,
)
wiki_resiliparse_with_preserve_formatting_subbed_dolma_1_4b_model = default_train(
    name="dolma-wiki-subbed-1.4b-resiliparse-with-preserving-formatting-no-references",
    tokenized=wiki_resiliparse_with_preserve_formatting_subbed_dolma_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=EXPERIMENT_TAG,
)

wiki_resiliparse_with_preserve_formatting_subbed_dolma_1_4b_evals = default_eval(
    step=wiki_resiliparse_with_preserve_formatting_subbed_dolma_1_4b_model
)


if __name__ == "__main__":
    tokenization_step = (
        list(tokenized_dolma_steps.values())
        + list(dolma_wiki_readability_tokenization_steps.values())
        + list(dolma_wiki_resiliparse_with_preserve_formatting_tokenization_steps.values())
    )

    executor_main(
        steps=[
            *tokenization_step,
            wiki_readablity_1_4b_subbed_dolma_model,
            wiki_readablity_1_4b_subbed_dolma_evals,
            wiki_resiliparse_with_preserve_formatting_subbed_dolma_1_4b_model,
            wiki_resiliparse_with_preserve_formatting_subbed_dolma_1_4b_evals,
        ]
    )
