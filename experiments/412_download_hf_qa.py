from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from operations.download.huggingface.download import DownloadConfig, download

"""
Downloads the following datasets
- mmlu
"""
############################################################
# download mmlu dataset
mmlu_download_step = ExecutorStep(
    name="raw/cais/mmlu",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="cais/mmlu",
        revision=versioned("c30699e"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="gs://marin-us-central2/raw/cais/mmlu",  # no versioned path; this had already been downloaded
)
############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            mmlu_download_step,
        ]
    )
