from experiments.evals.evals import default_eval
from experiments.exp246_web_extraction_method_training import (
    fineweb_readability_1_4b_model,
    fineweb_resiliparse_preserve_formatting_1_4b_model,
    fineweb_trafilatura_1_4b_model,
    fineweb_trafilatura_favor_precision_1_4b_model,
    transform_resiliparse_default,
)
from marin.execution.executor import executor_main

readability_eval = default_eval(fineweb_readability_1_4b_model)

trafilatura_default_eval = default_eval(fineweb_trafilatura_1_4b_model)

trafilatura_favor_precision_eval = default_eval(fineweb_trafilatura_favor_precision_1_4b_model)

resiliparse_default_eval = default_eval(transform_resiliparse_default)

resiliparse_preserve_formatting_eval = default_eval(fineweb_resiliparse_preserve_formatting_1_4b_model)


if __name__ == "__main__":
    executor_main(
        steps=[
            readability_eval,
            trafilatura_default_eval,
            trafilatura_favor_precision_eval,
            resiliparse_default_eval,
            resiliparse_preserve_formatting_eval,
        ]
    )
