"""
Tokenizes the Dolma 1.7 datasets.
"""

from experiments.defaults import default_tokenize, default_train
from experiments.dolma.tokenize_dolma import DOLMA_OLMO_MIXTURE_WEIGHTS, tokenize_dolma_steps
from experiments.evals.evals import default_eval
from experiments.exp575_wikipedia_markdownify import wikipedia_resiliparse_custom_fork
from experiments.exp579_ar5iv_markdownify import ar5iv_no_problem_resiliparse_custom_fork
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config

EXPERIMENT_TAG = ["mixture-of-formats-training"]

tokenized_dolma_steps = tokenize_dolma_steps()

# BASELINE DOLMA TRAINING
no_mixture_llama3_tokenized = lm_mixture_data_config(
    components=tokenized_dolma_steps,
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS,
)

no_mixture_dolma_model = default_train(
    name="dolma-mixture-of-formats-1.4b-no-mixture",
    tokenized=no_mixture_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=[*EXPERIMENT_TAG, "no-mixture"],
)
no_mixture_dolma_evals = default_eval(step=no_mixture_dolma_model)


# ARXIV ONLY MIXING FOR MARKDOWNIFIED DATASETS
arxiv_markdownified_tokenized = default_tokenize(
    name="arxiv-resiliparse-custom-fork",
    dataset=ar5iv_no_problem_resiliparse_custom_fork,
    tokenizer=llama3_tokenizer,
)

dolma_arxiv_tokenization_steps = dict(
    tokenized_dolma_steps, {"arxiv-resiliparse-custom-fork": arxiv_markdownified_tokenized}
)
arxiv_weights = dict(
    DOLMA_OLMO_MIXTURE_WEIGHTS, {"arxiv-resiliparse-custom-fork": DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/arxiv"]}
)

arxiv_only_subbed_llama3_tokenized = lm_mixture_data_config(
    components=dolma_arxiv_tokenization_steps,
    weights=arxiv_weights,
)

arxiv_only_subbed_dolma_model = default_train(
    name="dolma-mixture-of-formats-1.4b-arxiv-only-subbed",
    tokenized=arxiv_only_subbed_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=[*EXPERIMENT_TAG, "arxiv-only-subbed"],
)

arxiv_only_subbed_dolma_evals = default_eval(step=arxiv_only_subbed_dolma_model)

# WIKI ONLY MIXING FOR MARKDOWNIFIED DATASETS
wiki_markdownified_tokenized = default_tokenize(
    name="wiki-resiliparse-custom-fork",
    dataset=wikipedia_resiliparse_custom_fork,
    tokenizer=llama3_tokenizer,
)

dolma_wiki_tokenization_steps = dict(
    tokenized_dolma_steps, {"wiki-resiliparse-custom-fork": wiki_markdownified_tokenized}
)
wiki_weights = dict(
    DOLMA_OLMO_MIXTURE_WEIGHTS, {"wiki-resiliparse-custom-fork": DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/wiki"]}
)

wiki_only_subbed_llama3_tokenized = lm_mixture_data_config(
    components=dolma_wiki_tokenization_steps,
    weights=wiki_weights,
)

wiki_only_subbed_dolma_model = default_train(
    name="dolma-mixture-of-formats-1.4b-wiki-only-subbed",
    tokenized=wiki_only_subbed_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=[*EXPERIMENT_TAG, "wiki-only-subbed"],
)

wiki_only_subbed_dolma_evals = default_eval(step=wiki_only_subbed_dolma_model)

# WIKI AND ARXIV MIXING FOR MARKDOWNIFIED DATASETS
wiki_and_arxiv_tokenization_steps = dict(dolma_wiki_tokenization_steps, dolma_arxiv_tokenization_steps)
wiki_and_arxiv_weights = dict(wiki_weights, arxiv_weights)

wiki_and_arxiv_subbed_llama3_tokenized = lm_mixture_data_config(
    components=wiki_and_arxiv_tokenization_steps,
    weights=wiki_and_arxiv_weights,
)

wiki_and_arxiv_subbed_dolma_model = default_train(
    name="dolma-mixture-of-formats-1.4b-wiki-and-arxiv-subbed",
    tokenized=wiki_and_arxiv_subbed_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=[*EXPERIMENT_TAG, "wiki-and-arxiv-subbed"],
)

wiki_and_arxiv_subbed_dolma_evals = default_eval(step=wiki_and_arxiv_subbed_dolma_model)

if __name__ == "__main__":
    tokenize_steps = (
        list(dolma_arxiv_tokenization_steps.values())
        + list(dolma_wiki_tokenization_steps.values())
        + list(wiki_and_arxiv_tokenization_steps.values())
    )

    executor_main(
        steps=[
            *tokenize_steps,
            no_mixture_dolma_model,
            no_mixture_dolma_evals,
            arxiv_only_subbed_dolma_model,
            arxiv_only_subbed_dolma_evals,
            wiki_only_subbed_dolma_model,
            wiki_only_subbed_dolma_evals,
            wiki_and_arxiv_subbed_dolma_model,
            wiki_and_arxiv_subbed_dolma_evals,
        ]
    )
