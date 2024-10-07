from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from operations.download.huggingface.download import DownloadConfig, download
from operations.download.huggingface.download_gated_manual import download_and_upload_to_gcs

"""
Downloads the following datasets:
- https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
- https://huggingface.co/datasets/bigcode/the-stack-dedup
- https://huggingface.co/datasets/EleutherAI/proof-pile-2
to GCS. These are the other datasets that were used in the DCLM paper (in addition to the actual DCLM-Baseline dataset).
"""

############################################################
# download DCLM-Baseline dataset
# dclm_baseline_download_step = ExecutorStep(
#     name="raw/dclm-baseline-1.0",
#     fn=download,
#     config=DownloadConfig(
#         hf_dataset_id="mlfoundations/dclm-baseline-1.0",
#         revision=versioned("a3b142c"),
#         gcs_output_path=this_output_path(),
#         wait_for_completion=True,
#     ),
# )

############################################################
# download The Stack dataset
the_stack_download_step = ExecutorStep(
    name="raw/the-stack-dedup",
    fn=download_and_upload_to_gcs,
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
    name="raw/proof-pile-2-dummy",
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
            # dclm_baseline_download_step,
            the_stack_download_step,
            # proofpile_download_step,
        ]
    )
