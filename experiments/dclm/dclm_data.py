"""
Downloads the following datasets:
- https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
- https://huggingface.co/datasets/EleutherAI/proof-pile-2
- https://huggingface.co/datasets/bigcode/starcoderdata

[Also has the step for
- https://huggingface.co/datasets/bigcode/the-stack-dedup (gated) which is not really needed
but related to StarCoder.]

to GCS. The 2nd and 3rd ones above are the other datasets that were used in the DCLM paper (in addition to
    the actual DCLM-Baseline dataset).
"""

from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from operations.download.huggingface.download import DownloadConfig, download
from operations.download.huggingface.download_gated_manual import download_and_upload_to_gcs

############################################################
# download DCLM-Baseline dataset
dclm_baseline = ExecutorStep(
    name="raw/dclm-baseline-1.0",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="mlfoundations/dclm-baseline-1.0",
        revision="a3b142c",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="gs://marin-us-central2/raw/dclm",  # no version in path; this had already been downloaded
)

############################################################
# download The Stack dataset- this is done manually via `download_and_upload_to_gcs` because the dataset is gated
the_stack_dedup = ExecutorStep(
    name="raw/the-stack-dedup",
    fn=download_and_upload_to_gcs,
    config=DownloadConfig(
        hf_dataset_id="bigcode/the-stack-dedup",
        revision="17cad72",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="gs://marin-us-central2/raw/the-stack-dedup-4ba450",
).cd("17cad72")

############################################################
# download Proof Pile 2 dataset
proofpile_2 = ExecutorStep(
    name="raw/proof-pile-2",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="EleutherAI/proof-pile-2",
        revision="901a927",
        gcs_output_path=this_output_path(),
        wait_for_completion=False,
    ),
    override_output_path="gs://marin-us-central2/raw/proof-pile-2-f1b1d8",
).cd("901a927/huggingface.co/datasets/EleutherAI/proof-pile-2/resolve/901a927")

############################################################
# download StarCoder data
starcoderdata = ExecutorStep(
    name="raw/starcoderdata",
    fn=download_and_upload_to_gcs,
    config=DownloadConfig(
        hf_dataset_id="bigcode/starcoderdata",
        revision="9fc30b5",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="gs://marin-us-central2/raw/starcoderdata-720c8c",
).cd("9fc30b5")

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            dclm_baseline,
            the_stack_dedup,
            proofpile_2,
            starcoderdata,
        ]
    )
