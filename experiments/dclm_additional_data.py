from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from operations.download.huggingface.download import DownloadConfig, download

"""
Downloads the following two datasets:
- https://huggingface.co/datasets/bigcode/the-stack-dedup
- https://huggingface.co/datasets/EleutherAI/proof-pile-2
to GCS. These are the other datasets that were used in the DCLM paper (in addition to the actual DCLM-Baseline dataset).
"""

############################################################
# download The Stack dataset
stack_download_step = ExecutorStep(
    name="raw/the-stack-dedup",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="bigcode/the-stack-dedup",
        revision=versioned("17cad72"), 
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)

############################################################
# download Proof Pile 2 dataset
proofpile_download_step = ExecutorStep(
    name="raw/proof-pile-2",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="EleutherAI/proof-pile-2",
        revision=versioned("901a927"), 
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            stack_download_step,
            proofpile_download_step,
        ]
    )
