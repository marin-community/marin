from marin.execution.executor import ExecutorStep, this_output_path
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
