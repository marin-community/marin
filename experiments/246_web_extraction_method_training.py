import logging

from experiments.defaults import default_tokenize, default_train, llama_1_4b_train_config
from experiments.llama import llama3_tokenizer, llama_1_4b
from experiments.pretraining_datasets import download_fineweb
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
    transform_trafilatura_default_step = ExecutorStep(
        name="documents/fineweb-small-trafilatura",
        fn=process_fw_dump,
        config=ParquetFWConfig(
            input_path=output_path_of(download_fineweb),
            cc_dumps=versioned(["CC-MAIN-2024-18"]),
            md_output_path=this_output_path("md"),
            text_output_path=this_output_path("text"),
            html_output_path=this_output_path("html"),
            extract_method=versioned("trafilatura"),
            config=TrafilaturaConfig(
                favor_precision=versioned(False),
                favor_recall=versioned(True),
                include_comments=versioned(False),
                deduplicate=versioned(False),
            ),
        ),
    )

    transform_trafilatura_favor_precision_step = ExecutorStep(
        name="documents/fineweb-small-trafilatura-favor-precision",
        fn=process_fw_dump,
        config=ParquetFWConfig(
            input_path=output_path_of(download_fineweb),
            cc_dumps=versioned(["CC-MAIN-2024-18"]),
            md_output_path=this_output_path("md"),
            text_output_path=this_output_path("text"),
            html_output_path=this_output_path("html"),
            extract_method=versioned("trafilatura"),
            config=TrafilaturaConfig(
                favor_precision=versioned(True),
                favor_recall=versioned(False),
                include_comments=versioned(False),
                deduplicate=versioned(False),
            ),
        ),
    )

    transform_resiliparse_default_step = ExecutorStep(
        name="documents/fineweb-small-resiliparse-default",
        fn=process_fw_dump,
        config=ParquetFWConfig(
            input_path=output_path_of(download_fineweb),
            cc_dumps=versioned(["CC-MAIN-2024-18"]),
            md_output_path=this_output_path("md"),
            text_output_path=this_output_path("text"),
            html_output_path=this_output_path("html"),
            extract_method=versioned("resiliparse"),
            config=ResiliparseConfig(
                preserve_formatting=versioned(False),
                main_content=versioned(True),
                links=versioned(False),
            ),
        ),
    )

    transform_resiliparse_preserve_formatting_step = ExecutorStep(
        name="documents/fineweb-small-resiliparse-preserve-formatting",
        fn=process_fw_dump,
        config=ParquetFWConfig(
            input_path=output_path_of(download_fineweb),
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

    transform_readability_step = ExecutorStep(
        name="documents/fineweb-small-readability",
        fn=process_fw_dump,
        config=ParquetFWConfig(
            input_path=output_path_of(download_fineweb),
            cc_dumps=versioned(["CC-MAIN-2024-18"]),
            md_output_path=this_output_path("md"),
            text_output_path=this_output_path("text"),
            html_output_path=this_output_path("html"),
            extract_method=versioned("readability"),
            config=HtmlToMarkdownConfig(
                include_images=versioned(False),
                include_links=versioned(False),
            ),
        ),
    )

    return [
        transform_trafilatura_default_step,
        transform_trafilatura_favor_precision_step,
        transform_resiliparse_default_step,
        transform_resiliparse_preserve_formatting_step,
        transform_readability_step,
    ]


def create_steps() -> list[ExecutorStep]:
    total_steps = []

    extraction_steps = get_extraction_steps()

    for step in extraction_steps:
        step_name = step.name.split("/", 1)[1]

        fw_tokenized = default_tokenize(
            name=f"fw-small-100B-{step_name}",
            dataset=output_path_of(step),
            tokenizer=llama3_tokenizer,
        )
        fw_100b_model = default_train(
            name=f"fw-small-100B-1.4b-{step_name}",
            tokenized=fw_tokenized,
            model_config=llama_1_4b,
            train_config=llama_1_4b_train_config,
        )

        evaluate_step = ExecutorStep(
            name=f"evaluation/fw-small-{step_name}",
            fn=evaluate,
            config=EvaluationConfig(
                evaluator="helm",
                model_name=versioned(step_name),
                model_path=output_path_of(fw_100b_model),
                evaluation_path=this_output_path(),
                evals=["mmlu"],
            ),
        )

        total_steps.extend([step, fw_tokenized, fw_100b_model, evaluate_step])

    return total_steps


if __name__ == "__main__":
    steps = create_steps()
    executor_main(steps=steps)
