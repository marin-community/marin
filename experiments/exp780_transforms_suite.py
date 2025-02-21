"""
This script runs a suite of scaling laws on the Dolma mix.

Link to issue: https://github.com/stanford-crfm/marin/issues/780
"""

from experiments.dolma.exp442_dolma import dolma_llama3_tokenized
from marin.execution.executor import executor_main
from marin.scaling_laws.create_ladder_suite import scaling_law_suite

from experiments.exp246_web_extraction_method_training import (
    fineweb_trafilatura_tokenized,
    fineweb_trafilatura_favor_precision_tokenized,
    fineweb_resiliparse_tokenized,
    fineweb_resiliparse_preserve_formatting_tokenized,
    fineweb_readability_tokenized,
)

fineweb_trafilatura_tokenized_suite = scaling_law_suite(
    sweep_name="fineweb-trafilatura-suite",
    tokenized=fineweb_trafilatura_tokenized,
)

fineweb_trafilatura_favor_precision_tokenized_suite = scaling_law_suite(
    sweep_name="fineweb-trafilatura-favor-precision-suite",
    tokenized=fineweb_trafilatura_favor_precision_tokenized,
)

fineweb_resiliparse_tokenized_suite = scaling_law_suite(
    sweep_name="fineweb-resiliparse-suite",
    tokenized=fineweb_resiliparse_tokenized,
)

fineweb_resiliparse_preserve_formatting_tokenized_suite = scaling_law_suite(
    sweep_name="fineweb-resiliparse-preserve_formatting-suite",
    tokenized=fineweb_resiliparse_preserve_formatting_tokenized,
)

fineweb_readability_tokenized_suite = scaling_law_suite(
    sweep_name="fineweb-readability-suite",
    tokenized=fineweb_readability_tokenized,
)



if __name__ == "__main__":
    executor_main(
        steps=[
            *fineweb_trafilatura_tokenized_suite,
            #*fineweb_trafilatura_favor_precision_tokenized_suite,
            #*fineweb_resiliparse_tokenized_suite,
            #*fineweb_resiliparse_preserve_formatting_tokenized_suite,
            #*fineweb_readability_tokenized_suite,
        ],
        description="suite for scaling laws on Dolma mix",
    )
