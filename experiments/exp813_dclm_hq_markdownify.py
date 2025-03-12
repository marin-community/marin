from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from operations.download.dclm_hq.download_dclm_hq_html import DCLMHQDownloadConfig, extract_dclm_hq_dump

html_extracted_dclm_hq = ExecutorStep(
    name="raw/dolmino-dclm-hq-html-extracted",
    fn=extract_dclm_hq_dump,
    config=DCLMHQDownloadConfig(
        input_path=versioned("gs://marin-us-central2/raw/dolmino-mix-1124-157960/bb54cab/data/dclm"),
        output_path=this_output_path(),
    ),
    pip_dependency_groups=["download_transform"],
)


if __name__ == "__main__":
    executor_main(
        steps=[
            html_extracted_dclm_hq,
        ]
    )
