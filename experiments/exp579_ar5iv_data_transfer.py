from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.schemas.web.convert import ResiliparseConfig
from operations.download.ar5iv.download import DownloadConfig, download
from operations.transform.ar5iv.transform_ar5iv import Ar5ivExtractionConfig, process_ar5iv_dump

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

ar5iv_no_problem_extracted = ExecutorStep(
    name="documents/ar5iv/ar5iv-04-2024-no-problem",
    fn=process_ar5iv_dump,
    config=Ar5ivExtractionConfig(
        input_path=output_path_of(ar5iv_no_problem_raw),
        revision="042024",
        output_path=this_output_path("resiliparse-with-preserving-formatting"),
        extract_method="resiliparse",
        extract_config=ResiliparseConfig(
            preserve_formatting=versioned(True),
            main_content=versioned(True),
            links=versioned(True),
        ),
    ),
    pip_dependency_groups=["download_transform"],
)

ar5iv_warnings_extracted = ExecutorStep(
    name="documents/ar5iv/ar5iv-04-2024-warning",
    fn=process_ar5iv_dump,
    config=Ar5ivExtractionConfig(
        input_path=output_path_of(ar5iv_warnings_raw),
        revision="042024",
        output_path=this_output_path("resiliparse-with-preserving-formatting"),
        extract_method="resiliparse",
        extract_config=ResiliparseConfig(
            preserve_formatting=versioned(True),
            main_content=versioned(True),
            links=versioned(True),
        ),
    ),
    pip_dependency_groups=["download_transform"],
)

if __name__ == "__main__":
    executor_main(
        steps=[
            ar5iv_no_problem_raw,
            ar5iv_warnings_raw,
            ar5iv_no_problem_extracted,
            ar5iv_warnings_extracted,
        ]
    )
