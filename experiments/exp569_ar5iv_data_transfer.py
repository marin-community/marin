from operations.download.ar5iv.download import DownloadConfig, download
from marin.execution.executor import ExecutorStep, executor_main, this_output_path

transfer_ar5iv_no_problem = ExecutorStep(
    name="raw/ar5iv/ar5iv-04-2024-no-problem",
    fn=download,
    config=DownloadConfig(
        input_path="https://data.fau.de/share/nSaD4SCQnWLR7xQfMRUXgEZY6CSQyXRFB5bnrYA9BSorE7RzynF84mto2hg6sC4A/ar5iv-04-2024-no-problem.zip",
        output_path=this_output_path(),
    ),
)

transfer_ar5iv_warnings = ExecutorStep(
    name="raw/ar5iv/ar5iv-04-2024-warning",
    fn=download,
    config=DownloadConfig(
        input_path="https://data.fau.de/share/nSaD4SCQnWLR7xQfMRUXgEZY6CSQyXRFB5bnrYA9BSorE7RzynF84mto2hg6sC4A/ar5iv-04-2024-warnings.zip",
        output_path=this_output_path(),
    ),
)

if __name__ == "__main__":
    executor_main(steps=[
        transfer_ar5iv_no_problem,
        transfer_ar5iv_warnings,
    ])