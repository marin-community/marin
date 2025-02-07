from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.schemas.web.convert import HtmlToMarkdownConfig, ResiliparseConfig
from operations.download.ar5iv.download import DownloadConfig, download
from operations.transform.ar5iv.transform_ar5iv import Ar5ivExtractionConfig, process_ar5iv_dump

ARXIV_BLACKLISTED_SELECTORS = [
    "h2.ltx_title_bibliography",
    "div.ltx_classification",
    "span.ltx_role_author"
]

ar5iv_no_problem_raw = ExecutorStep(
    name="raw/ar5iv/ar5iv-04-2024-no-problem",
    fn=download,
    config=DownloadConfig(
        input_path="gs://marin-us-central2/raw/ar5iv/v04.2024/ar5iv-04-2024-no-problem.zip",
        output_path=this_output_path(),
    ),
)

ar5iv_warnings_raw = ExecutorStep(
    name="raw/ar5iv/ar5iv-04-2024-warning-with-references",
    fn=download,
    config=DownloadConfig(
        input_path="gs://marin-us-central2/raw/ar5iv/v04.2024/ar5iv-04-2024-warnings.zip",
        output_path=this_output_path(),
    ),
)

ar5iv_no_problem_resiliparse_custom_fork = ExecutorStep(
    name="documents/ar5iv/ar5iv-04-2024-no-problem",
    fn=process_ar5iv_dump,
    config=Ar5ivExtractionConfig(
        input_path=output_path_of(ar5iv_no_problem_raw),
        revision="042024",
        output_path=this_output_path("resiliparse-custom-fork"),
        extract_method="resiliparse",
        extract_config=ResiliparseConfig(
            preserve_formatting=True,
            main_content=True,
            links=versioned(False),
            prepend_title=True,
            skip_elements=ARXIV_BLACKLISTED_SELECTORS,
            use_custom_variant=True,
            markdownify_config=HtmlToMarkdownConfig(
                include_images=False,
                include_links=False,
            )
        ),
        remove_reference_section=versioned(True),
    ),
    pip_dependency_groups=["download_transform"],
)

ar5iv_warnings_resiliparse_custom_fork = ExecutorStep(
    name="documents/ar5iv/ar5iv-04-2024-warning",
    fn=process_ar5iv_dump,
    config=Ar5ivExtractionConfig(
        input_path=output_path_of(ar5iv_no_problem_raw),
        revision="042024",
        output_path=this_output_path("resiliparse-custom-fork"),
        extract_method="resiliparse",
        extract_config=ResiliparseConfig(
            preserve_formatting=True,
            main_content=True,
            links=versioned(False),
            prepend_title=True,
            skip_elements=ARXIV_BLACKLISTED_SELECTORS,
            use_custom_variant=True,
            markdownify_config=HtmlToMarkdownConfig(
                include_images=False,
                include_links=False,
            )
        ),
        remove_reference_section=versioned(True),
    ),
    pip_dependency_groups=["download_transform"],
)



if __name__ == "__main__":
    executor_main(
        steps=[
            ar5iv_no_problem_raw,
            ar5iv_warnings_raw,
            ar5iv_no_problem_resiliparse_custom_fork,
            ar5iv_warnings_resiliparse_custom_fork,
        ]
    )
