import dataclasses

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets import dclm_baseline, proofpile_2, starcoderdata
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config

DCLM_MIXTURE_WEIGHTS = {
    # token counts are for neox tokenizer
    "dclm_baseline": 3.8,  # 3.8 trillion tokens https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
    "starcoderdata": 0.25,  # 250 billion tokens https://huggingface.co/datasets/bigcode/starcoderdata
    "proofpile_2": 0.055,  # 55 billion tokens https://huggingface.co/datasets/EleutherAI/proof-pile-2
}


DCLM_BASELINE_ONLY_MIXTURE = {
    "dclm_baseline": 3.8,  # 3.8 trillion tokens https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
    "starcoderdata": 0,  # 250 billion tokens https://huggingface.co/datasets/bigcode/starcoderdata
    "proofpile_2": 0,  # 55 billion tokens https://huggingface.co/datasets/EleutherAI/proof-pile-2
}


dclm_components_llama3 = {
    "dclm_baseline": dataclasses.replace(
        default_tokenize(
            name="dclm_baseline",
            dataset=dclm_baseline,
            tokenizer=llama3_tokenizer,
        ),
        # make consistent with path in eu (they differ b/c of the cd)
        override_output_path="tokenized/dclm_baseline-0206f1/",
    ),
    "starcoderdata": default_tokenize(
        name="starcoderdata", dataset=starcoderdata, tokenizer=llama3_tokenizer, text_key="content"
    ),
    "proofpile_2": default_tokenize(
        name="proofpile_2",
        dataset=proofpile_2,
        tokenizer=llama3_tokenizer,
    ),
}
dclm_mixture_config_llama3 = lm_mixture_data_config(components=dclm_components_llama3, weights=DCLM_MIXTURE_WEIGHTS)


if __name__ == "__main__":
    executor_main(steps=list(dclm_components_llama3.values()))
