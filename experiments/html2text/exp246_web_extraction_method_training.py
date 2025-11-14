# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test different html->text transformation methods (on FineWeb, train 1.4B models).
https://github.com/marin-community/marin/issues/246
"""

import logging

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config
from experiments.pretraining_datasets.simple import downloads
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.processing.tokenize import lm_data_config
from marin.convert import HtmlToMarkdownConfig, ResiliparseConfig, TrafilaturaConfig
from marin.transform.fineweb.process_parquet_fw import ParquetFWConfig, process_fw_dump

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ray")


# Trafilatura Default
transform_trafilatura_default = ExecutorStep(
    name="documents/fineweb-small-trafilatura",
    fn=process_fw_dump,
    config=ParquetFWConfig(
        input_path=output_path_of(downloads["fineweb"]),
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
    tokenized=lm_data_config(fineweb_trafilatura_tokenized, permutation_type="linear"),
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)


# Trafilatura Favor Precision
transform_trafilatura_favor_precision = ExecutorStep(
    name="documents/fineweb-small-trafilatura-favor-precision",
    fn=process_fw_dump,
    config=ParquetFWConfig(
        input_path=output_path_of(downloads["fineweb"]),
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
fineweb_trafilatura_favor_precision_data = lm_data_config(
    fineweb_trafilatura_favor_precision_tokenized,
    permutation_type="linear",
)

fineweb_trafilatura_favor_precision_1_4b_model = default_train(
    name="fineweb-small-1.4b-trafilatura-favor-precision",
    tokenized=fineweb_trafilatura_favor_precision_data,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)


# Resiliparse Default
transform_resiliparse_default = ExecutorStep(
    name="documents/fineweb-small-resiliparse-default",
    fn=process_fw_dump,
    config=ParquetFWConfig(
        input_path=output_path_of(downloads["fineweb"]),
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
fineweb_resiliparse_data = lm_data_config(fineweb_resiliparse_tokenized, permutation_type="linear")

fineweb_resiliparse_1_4b_model = default_train(
    name="fineweb-small-1.4b-resiliparse",
    tokenized=fineweb_resiliparse_data,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)


# Resiliparse Preserve Formatting
transform_resiliparse_preserve_formatting = ExecutorStep(
    name="documents/fineweb-small-resiliparse-preserve-formatting",
    fn=process_fw_dump,
    config=ParquetFWConfig(
        input_path=output_path_of(downloads["fineweb"]),
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
fineweb_resiliparse_preserve_formatting_data = lm_data_config(
    fineweb_resiliparse_preserve_formatting_tokenized,
    permutation_type="linear",
)

fineweb_resiliparse_preserve_formatting_1_4b_model = default_train(
    name="fineweb-small-1.4b-resiliparse-preserve-formatting",
    tokenized=fineweb_resiliparse_preserve_formatting_data,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)


# Readability
transform_readability = ExecutorStep(
    name="documents/fineweb-small-readability",
    fn=process_fw_dump,
    config=ParquetFWConfig(
        input_path=output_path_of(downloads["fineweb"]),
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
fineweb_readability_data = lm_data_config(fineweb_readability_tokenized, permutation_type="linear")

fineweb_readability_1_4b_model = default_train(
    name="fineweb-small-1.4b-readability",
    tokenized=fineweb_readability_data,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)


if __name__ == "__main__":
    steps = [
        downloads["fineweb"],
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
