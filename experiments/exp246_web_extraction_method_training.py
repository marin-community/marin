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


# Trafilatura Default
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

fineweb_trafilatura_tokenized = default_tokenize(
    name="fineweb-small-trafilatura",
    dataset=transform_trafilatura_default,
    tokenizer=llama3_tokenizer,
)

fineweb_trafilatura_1_4b_model = default_train(
    name="fineweb-small-1.4b-trafilatura",
    tokenized=fineweb_trafilatura_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)


# Trafilatura Favor Precision
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

fineweb_trafilatura_favor_precision_tokenized = default_tokenize(
    name="fineweb-small-trafilatura-favor-precision",
    dataset=transform_trafilatura_favor_precision,
    tokenizer=llama3_tokenizer,
)

fineweb_trafilatura_favor_precision_1_4b_model = default_train(
    name="fineweb-small-1.4b-trafilatura-favor-precision",
    tokenized=fineweb_trafilatura_favor_precision_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)


# Resiliparse Default
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

fineweb_resiliparse_tokenized = default_tokenize(
    name="fineweb-small-resiliparse",
    dataset=transform_resiliparse_default,
    tokenizer=llama3_tokenizer,
)

fineweb_resiliparse_1_4b_model = default_train(
    name="fineweb-small-1.4b-resiliparse",
    tokenized=fineweb_resiliparse_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)


# Resiliparse Preserve Formatting
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

fineweb_resiliparse_preserve_formatting_tokenized = default_tokenize(
    name="fineweb-small-resiliparse-preserve-formatting",
    dataset=transform_resiliparse_preserve_formatting,
    tokenizer=llama3_tokenizer,
)

fineweb_resiliparse_preserve_formatting_1_4b_model = default_train(
    name="fineweb-small-1.4b-resiliparse-preserve-formatting",
    tokenized=fineweb_resiliparse_preserve_formatting_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)


# Readability
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

fineweb_readability_tokenized = default_tokenize(
    name="fineweb-small-readability",
    dataset=transform_readability,
    tokenizer=llama3_tokenizer,
)

fineweb_readability_1_4b_model = default_train(
    name="fineweb-small-1.4b-readability",
    tokenized=fineweb_readability_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)


if __name__ == "__main__":
    steps = [
        fineweb,
        transform_trafilatura_default,
        fineweb_trafilatura_tokenized,
        fineweb_trafilatura_1_4b_model,
        transform_trafilatura_favor_precision,
        fineweb_trafilatura_favor_precision_tokenized,
        fineweb_trafilatura_favor_precision_1_4b_model,
        transform_resiliparse_default,
        fineweb_resiliparse_tokenized,
        fineweb_resiliparse_1_4b_model,
        transform_resiliparse_preserve_formatting,
        fineweb_resiliparse_preserve_formatting_tokenized,
        fineweb_resiliparse_preserve_formatting_1_4b_model,
        transform_readability,
        fineweb_readability_tokenized,
        fineweb_readability_1_4b_model,
    ]
    executor_main(steps=steps)
