from experiments.defaults import default_download, default_tokenize
from experiments.llama import llama3_tokenizer
from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution import versioned
from marin.execution.executor import ExecutorStep, this_output_path
from marin.processing.tokenize import lm_mixture_data_config

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

lavita_medical_qa_datasets = default_download(
    name="raw/lavita_medical_qa",
    hf_dataset_id="lavita/medical-qa-datasets",
    revision="59d48e2",
    override_output_path="raw/lavita_medical_qa",
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
)
