from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from operations.download.ar5iv.download import DownloadConfig, download

ar5iv_no_problem_raw = ExecutorStep(
    name="raw/ar5iv/ar5iv-04-2024-no-problem",
    fn=download,
    config=DownloadConfig(
        input_path="https://data.fau.de/share/nSaD4SCQnWLR7xQfMRUXgEZY6CSQyXRFB5bnrYA9BSorE7RzynF84mto2hg6sC4A/ar5iv-04-2024-no-problem.zip",
        output_path=this_output_path(),
    ),
)

ar5iv_warnings_raw = ExecutorStep(
    name="raw/ar5iv/ar5iv-04-2024-warning",
    fn=download,
    config=DownloadConfig(
        input_path="https://data.fau.de/share/nSaD4SCQnWLR7xQfMRUXgEZY6CSQyXRFB5bnrYA9BSorE7RzynF84mto2hg6sC4A/ar5iv-04-2024-warnings.zip",
        output_path=this_output_path(),
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[
            ar5iv_no_problem_raw,
            ar5iv_warnings_raw,
        ]
    )
