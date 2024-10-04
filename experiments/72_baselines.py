"""
Train 1.4B models on standard datasets.
https://github.com/stanford-crfm/marin/issues/72
"""

from marin.execution.executor import ExecutorStep, this_output_path, versioned, executor_main

############################################################
# Download 

from operations.download.huggingface.download import DownloadConfig, download

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

download_slimpajama = ExecutorStep(
    name="raw/slimpajama",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id=versioned("cerebras/SlimPajama-627B"),
        revision=versioned("2d0accd"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)

download_dclm_baseline = ExecutorStep(
    name="raw/dclm-baseline",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id=versioned("mlfoundations/dclm-baseline-1.0"),
        revision=versioned("a3b142c"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
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
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            download_slimpajama
        ]
    )
