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

from experiments.common_pile.tokenize_common_pile import stackv2_edu_filtered
from experiments.defaults import default_download, default_tokenize
from experiments.llama import llama3_tokenizer
from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution import versioned
from marin.execution.executor import ExecutorStep, this_output_path
from marin.processing.tokenize import lm_mixture_data_config
from marin.transform.common_pile.filter_by_extension import (
    FilterByMetadataExtensionConfig,
    filter_dataset_by_metadata_extension,
)
from marin.transform.medical.lavita_to_dolma import LavitaToDolmaConfig, convert_lavita_split_to_dolma

finemath_commit_hash = "8f233cf"
finemath = ExecutorStep(
    name="raw/finemath",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="HuggingFaceTB/finemath",
        revision=finemath_commit_hash,
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)


finemath_3_plus = finemath.cd("finemath-3plus")
finemath_3_plus_tokenized = default_tokenize(
    name="finemath_3_plus",
    dataset=finemath_3_plus,
    tokenizer=llama3_tokenizer,
).with_output_path("tokenized/finemath_3_plus-a26b0f/")

STACKV2_EDU_PYTHON_EXTENSIONS = (".py", ".pyw", ".pyi")

stackv2_edu_filtered_python = ExecutorStep(
    name="documents/common_pile/stackv2_edu_filtered_python",
    fn=filter_dataset_by_metadata_extension,
    config=FilterByMetadataExtensionConfig(
        input_path=stackv2_edu_filtered,
        output_path=this_output_path(),
        allowed_extensions=STACKV2_EDU_PYTHON_EXTENSIONS,
        input_glob="stack-edu-*.json.gz",
    ),
)

stackv2_edu_filtered_python_tokenized = default_tokenize(
    name="common_pile_stackv2_edu_filtered_python",
    dataset=stackv2_edu_filtered_python,
    tokenizer=llama3_tokenizer,
)

# Define MegaMath dataset source
megamath_source = default_download(
    name="raw/llm360/megamath",
    hf_dataset_id="llm360/MegaMath",
    revision=versioned("3cbc64616594d6bc8759abaa0b2a71858f880f0d"),
    override_output_path="raw/llm360/megamath",
    hf_urls_glob=["**/*.parquet", "*.md"],
)

# Megamath is partitioned into 6 sources. We expose each of them as a separate step.
megamath_split_paths = {
    # Code just seems to be metadata, not actual code files.
    # "megamath/code": megamath_source / "megamath-code/*.parquet",
    "megamath/qa": megamath_source / "megamath-qa/**/*.parquet",
    "megamath/text_code_block": megamath_source / "megamath-text-code-block/*.parquet",
    "megamath/translated_code": megamath_source / "megamath-translated-code/*.parquet",
    "megamath/web_pro": megamath_source / "megamath-web-pro/*.parquet",
    "megamath/web": megamath_source / "megamath-web/*/*.parquet",
}

megamath_tokenized = {
    name: default_tokenize(
        name=name,
        dataset=path,
        tokenizer=llama3_tokenizer,
    )
    for name, path in megamath_split_paths.items()
}

# source: https://huggingface.co/datasets/LLM360/MegaMath#detailed-statistics
# in teratokens
megamath_token_counts = {
    # "Real"
    "megamath/web": 0.2639,  # 263.9B
    "megamath/web_pro": 0.0151,  # 15.1B
    # Synthetic
    "megamath/text_code_block": 0.0503,  # 50.3B
    "megamath/translated_code": 0.0072,  # 7.2B
    "megamath/qa": 0.0070,  # 7.0B
}

megamath_mixture = lm_mixture_data_config(
    components=megamath_tokenized,
    weights=megamath_token_counts,
)

megamath_real_only = lm_mixture_data_config(
    components={
        "megamath/web": megamath_tokenized["megamath/web"],
        "megamath/web_pro": megamath_tokenized["megamath/web_pro"],
    },
    weights={
        "megamath/web": megamath_token_counts["megamath/web"],
        "megamath/web_pro": megamath_token_counts["megamath/web_pro"],
    },
    permutation_type="feistel",
)


pile_pubmed_abstracts_validation = default_download(
    name="raw/pile_pubmed_abstracts",
    hf_dataset_id="suolyer/pile_pubmed-abstracts",
    revision="139fdbf",
    override_output_path="raw/pile_pubmed_abstracts",
    hf_urls_glob=["val.json"],
)

pile_pubmed_central_validation = default_download(
    name="raw/pile_pubmed_central",
    hf_dataset_id="suolyer/pile_pubmed-central",
    revision="783dc95",
    override_output_path="raw/pile_pubmed_central",
    hf_urls_glob=["val.json"],
)

pile_pubmed_central_validation_tokenized = default_tokenize(
    name="pile_pubmed_central",
    dataset=pile_pubmed_central_validation,
    tokenizer=llama3_tokenizer,
    is_validation=True,
)

pile_pubmed_abstracts_validation_tokenized = default_tokenize(
    name="pile_pubmed_abstracts",
    dataset=pile_pubmed_abstracts_validation,
    tokenizer=llama3_tokenizer,
    is_validation=True,
)


# Medical QA datasets
lavita_medical_qa_datasets = default_download(
    name="raw/lavita_medical_qa",
    hf_dataset_id="lavita/medical-qa-datasets",
    revision="59d48e2",
    override_output_path="raw/lavita_medical_qa",
)

lavita_pubmed = ExecutorStep(
    name="documents/lavita_pubmed",
    fn=convert_lavita_split_to_dolma,
    config=LavitaToDolmaConfig(
        input_path=lavita_medical_qa_datasets, output_path=this_output_path(), subset="pubmed-qa", split="train"
    ),
)

lavita_medmcqa = ExecutorStep(
    name="documents/lavita_medmcqa",
    fn=convert_lavita_split_to_dolma,
    config=LavitaToDolmaConfig(
        input_path=lavita_medical_qa_datasets, output_path=this_output_path(), subset="medmcqa", split="train"
    ),
)

lavita_allprocessed = ExecutorStep(
    name="documents/lavita_allprocessed",
    fn=convert_lavita_split_to_dolma,
    config=LavitaToDolmaConfig(
        input_path=lavita_medical_qa_datasets,
        output_path=this_output_path(),
        subset="all-processed",
        split="train",
    ),
)

lavita_pubmed_validation = ExecutorStep(
    name="documents/lavita_pubmed_validation",
    fn=convert_lavita_split_to_dolma,
    config=LavitaToDolmaConfig(
        input_path=lavita_medical_qa_datasets, output_path=this_output_path(), subset="pubmed-qa", split="validation"
    ),
)

lavita_medmcqa_validation = ExecutorStep(
    name="documents/lavita_medmcqa_validation",
    fn=convert_lavita_split_to_dolma,
    config=LavitaToDolmaConfig(
        input_path=lavita_medical_qa_datasets, output_path=this_output_path(), subset="medmcqa", split="validation"
    ),
)

lavita_allprocessed_tokenized = default_tokenize(
    "tokenized/lavita_allprocessed",
    lavita_allprocessed,
    tokenizer=llama3_tokenizer,
)

lavita_medmcqa_tokenized = default_tokenize(
    "tokenized/lavita_medmcqa",
    lavita_medmcqa,
    tokenizer=llama3_tokenizer,
)

lavita_pubmedqa_tokenized = default_tokenize(
    "tokenized/lavita_pubmedqa",
    lavita_pubmed,
    tokenizer=llama3_tokenizer,
)

lavita_pubmedqa_validation_tokenized = default_tokenize(
    "tokenized/lavita_pubmedqa_validation",
    lavita_pubmed_validation,
    tokenizer=llama3_tokenizer,
    is_validation=True,
)

lavita_medmcqa_validation_tokenized = default_tokenize(
    "tokenized/lavita_medmcqa_validation",
    lavita_medmcqa_validation,
    tokenizer=llama3_tokenizer,
    is_validation=True,
)
