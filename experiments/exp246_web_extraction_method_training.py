"""
Test different html->text transformation methods (on FineWeb, train 1.4B models).
https://github.com/stanford-crfm/marin/issues/246
"""

import logging

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config
from experiments.pretraining_datasets import fineweb
from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.run import evaluate
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.schemas.web.convert import HtmlToMarkdownConfig, ResiliparseConfig, TrafilaturaConfig
from scripts.fineweb.process_parquet_fw import ParquetFWConfig, process_fw_dump

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ray")


def get_extraction_steps():
    transform_trafilatura_default = ExecutorStep(
        name="documents/fineweb-small-trafilatura",
        fn=process_fw_dump,
        config=ParquetFWConfig(
            input_path=output_path_of(fineweb),
            cc_dumps=versioned(["CC-MAIN-2024-18"]),
            md_output_path=this_output_path("md"),
            text_output_path=this_output_path("text"),
            extract_method=versioned("trafilatura"),
            config=TrafilaturaConfig(
                favor_precision=versioned(False),
                favor_recall=versioned(True),
                include_comments=versioned(False),
                deduplicate=versioned(False),
            ),
        ),
    )

    transform_trafilatura_favor_precision = ExecutorStep(
        name="documents/fineweb-small-trafilatura-favor-precision",
        fn=process_fw_dump,
        config=ParquetFWConfig(
            input_path=output_path_of(fineweb),
            cc_dumps=versioned(["CC-MAIN-2024-18"]),
            md_output_path=this_output_path("md"),
            text_output_path=this_output_path("text"),
            extract_method=versioned("trafilatura"),
            config=TrafilaturaConfig(
                favor_precision=versioned(True),
                favor_recall=versioned(False),
                include_comments=versioned(False),
                deduplicate=versioned(False),
            ),
        ),
    )

    transform_resiliparse_default = ExecutorStep(
        name="documents/fineweb-small-resiliparse-default",
        fn=process_fw_dump,
        config=ParquetFWConfig(
            input_path=output_path_of(fineweb),
            cc_dumps=versioned(["CC-MAIN-2024-18"]),
            md_output_path=this_output_path("md"),
            text_output_path=this_output_path("text"),
            extract_method=versioned("resiliparse"),
            config=ResiliparseConfig(
                preserve_formatting=versioned(False),
                main_content=versioned(True),
                links=versioned(False),
            ),
        ),
    )

    transform_resiliparse_preserve_formatting = ExecutorStep(
        name="documents/fineweb-small-resiliparse-preserve-formatting",
        fn=process_fw_dump,
        config=ParquetFWConfig(
            input_path=output_path_of(fineweb),
            cc_dumps=versioned(["CC-MAIN-2024-18"]),
            md_output_path=this_output_path("md"),
            text_output_path=this_output_path("text"),
            html_output_path=this_output_path("html"),
            extract_method=versioned("resiliparse"),
            config=ResiliparseConfig(
                preserve_formatting=versioned(True),
                main_content=versioned(True),
                links=versioned(False),
            ),
        ),
    )

    transform_readability = ExecutorStep(
        name="documents/fineweb-small-readability",
        fn=process_fw_dump,
        config=ParquetFWConfig(
            input_path=output_path_of(fineweb),
            cc_dumps=versioned(["CC-MAIN-2024-18"]),
            md_output_path=this_output_path("md"),
            text_output_path=this_output_path("text"),
            extract_method=versioned("readability"),
            config=HtmlToMarkdownConfig(
                include_images=versioned(False),
                include_links=versioned(False),
            ),
        ),
    )

    return [
        transform_trafilatura_default,
        transform_trafilatura_favor_precision,
        transform_resiliparse_default,
        transform_resiliparse_preserve_formatting,
        transform_readability,
    ]


def create_steps() -> list[ExecutorStep]:
    total_steps = []

    extraction_steps = get_extraction_steps()

    for step in extraction_steps:
        step_name = step.name.split("/", 1)[1]

        fineweb_tokenized = default_tokenize(
            name=f"fineweb-small-{step_name}",
            dataset=step,
            tokenizer=llama3_tokenizer,
        )
        fineweb_100b_model = default_train(
            name=f"fineweb-small-1.4b-{step_name}",
            tokenized=fineweb_tokenized,
            model_config=llama_1_4b,
            train_config=llama_1_4b_train_config,
        )

        evaluate_model = ExecutorStep(
            name=f"evaluation/fw-small-{step_name}",
            fn=evaluate,
            config=EvaluationConfig(
                evaluator="helm",
                model_name=versioned(step_name),
                model_path=output_path_of(fineweb_100b_model),
                evaluation_path=this_output_path(),
                evals=["mmlu"],
            ),
        )

        total_steps.extend([step, fineweb_tokenized, fineweb_100b_model, evaluate_model])

    return total_steps


if __name__ == "__main__":
    steps = create_steps()
    executor_main(steps=steps)
