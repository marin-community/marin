from marin.execution.executor import ExecutorStep, this_output_path
from operations.download.huggingface.download import DownloadConfig, download

finemath_commit_hash = "8f233cf"
finemath = ExecutorStep(
    name="raw/finemath",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="HuggingFaceTB/finemath",
        revision=finemath_commit_hash,
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)

finemath_3_plus = finemath.cd(f"{finemath_commit_hash}/finemath-3plus")
