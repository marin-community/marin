"""
Train 1.4B models on standard datasets.
https://github.com/stanford-crfm/marin/issues/72
"""

from experiments.defaults import llama_1_4b, default_tokenize, default_train
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from operations.download.huggingface.download import DownloadConfig, download

############################################################
# Download

download_fineweb = ExecutorStep(
    name="raw/fineweb",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="HuggingFaceFW/fineweb",
        revision="cd85054",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="gs://marin-us-central2/raw/fineweb",
)

download_fineweb_edu = ExecutorStep(
    name="raw/fineweb-edu",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        revision="3c452cb",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)
fineweb_edu_raw = output_path_of(download_fineweb_edu)

download_slimpajama = ExecutorStep(
    name="raw/SlimPajama-627B",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id=versioned("cerebras/SlimPajama-627B"),
        revision=versioned("2d0accd"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)
slimpajama_raw = output_path_of(
    download_slimpajama, "2d0accd/huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/2d0accd"
)

download_dolma = ExecutorStep(
    name="raw/dolma",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id=versioned("allenai/dolma"),
        revision=versioned("7f48140"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="gs://marin-us-central2/raw/dolma",
)

download_dclm_baseline = ExecutorStep(
    name="raw/dclm-baseline-1.0",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id=versioned("mlfoundations/dclm-baseline-1.0"),
        revision=versioned("a3b142c"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="gs://marin-us-central2/raw/dclm",
)

############################################################
# Train models

slimpajama_tokenized = default_tokenize(name="SlimPajama-627B", dataset=slimpajama_raw, tokenizer="llama2")
slimpajama_model = default_train(name="SlimPajama-627B-1.4b", tokenized=slimpajama_tokenized, model=llama_1_4b)

fineweb_edu_tokenized = default_tokenize(name="fineweb-edu", dataset=fineweb_edu_raw, tokenizer="llama2")
fineweb_edu_model = default_train(name="fineweb-edu-1.4b", tokenized=fineweb_edu_tokenized, model=llama_1_4b)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            download_fineweb_edu,
            #slimpajama_model,
        ]
    )
