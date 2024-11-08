"""
Test different html->text transformation methods (on FineWeb, train 1.4B models).
https://github.com/stanford-crfm/marin/issues/246
"""

import logging

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config
from experiments.pretraining_datasets import fineweb
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.schemas.web.convert import HtmlToMarkdownConfig, ResiliparseConfig
from scripts.fineweb.process_parquet_fw import ParquetFWConfig, process_fw_dump

logger = logging.getLogger("ray")


def create_steps() -> list[ExecutorStep]:
    transform_resiliparse_custom_fork = ExecutorStep(
        name="documents/fineweb-small-resiliparse-custom-fork",
        fn=process_fw_dump,
        config=ParquetFWConfig(
            input_path=output_path_of(fineweb),
            cc_dumps=versioned(["CC-MAIN-2024-18"]),
            md_output_path=this_output_path("md"),
            extract_method=versioned("resiliparse"),
            config=ResiliparseConfig(
                preserve_formatting=versioned(False),
                main_content=versioned(True),
                links=versioned(False),
                use_custom_variant=versioned(True),
                markdownify_config=HtmlToMarkdownConfig(
                    include_images=versioned(False),
                    include_links=versioned(False),
                ),
            ),
        ),
    )

    step_name = "fineweb-small-resiliparse-custom-fork"

    fw_tokenized = default_tokenize(
        name=f"fw-small-{step_name}",
        dataset=transform_resiliparse_custom_fork,
        tokenizer=llama3_tokenizer,
    )
    fw_100b_model = default_train(
        name=f"fw-small-1.4b-{step_name}",
        tokenized=fw_tokenized,
        model_config=llama_1_4b,
        train_config=llama_1_4b_train_config,
    )

    return [
        fineweb,
        transform_resiliparse_custom_fork,
        fw_tokenized,
        fw_100b_model,
    ]


if __name__ == "__main__":
    executor_main(steps=create_steps())
