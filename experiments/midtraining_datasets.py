from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from marin.execution.executor import ExecutorStep, this_output_path, executor_main
from operations.download.huggingface.download_hf import DownloadConfig, download_hf

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
)

pubmed_abstracts = ExecutorStep(
    name="raw/suhas/pubmed_abstracts",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="hwang2006/PUBMED_title_abstracts_2020_baseline",
        revision="2f25efe",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)

pubmed_abstracts_tokenized = default_tokenize(
    name="pubmed_abstracts",
    dataset=pubmed_abstracts,
    tokenizer=llama3_tokenizer,
)

open_web_math = ExecutorStep(
    name="raw/open-web-math",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="open-web-math/open-web-math",
        revision="fde8ef8",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)

open_web_math_tokenized = default_tokenize(
    name="open-web-math",
    dataset=open_web_math,
    tokenizer=llama3_tokenizer,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            open_web_math,
            open_web_math_tokenized,
        ],
        description="Download and tokenize pubmed abstracts and open web math",
    )
