"""
Experiment to convert Arxiv HTML dumps to Markdown format.

This experiment downloads and processes the Arxiv HTML dumps to extract clean markdown/text.
We prepare the text/markdown for use as a training dataset for a language model, over 3 settings:

* Readability which extracts simplifies HTML DOM tree and is then combined with markdownify
  for markdown conversion. We generate four variants with combinations of references and links removed or kept.
* The default Resiliparse configuration, which removes boilerplate but does not support markdownification. We generate
  four variants with combinations of references and links removed or kept.
* Our custom fork of Resiliparse which provides a simplified DOM tree with removed boilerplate which can be
  passed to Markdownify to producing Markdown text that has less noise. We generate single variant for each without
  references and links.

Reference Issue: https://github.com/stanford-crfm/marin/issues/579
"""

from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.schemas.web.convert import ExtractionConfig, HtmlToMarkdownConfig, ResiliparseConfig
from operations.download.ar5iv.download import DownloadConfig, download
from operations.transform.ar5iv.transform_ar5iv import Ar5ivExtractionConfig, process_ar5iv_dump

# Selectors to remove from the DOM tree, these mostly contain reference sections, Authors,
# and Title sections[Prepended Manually to avoid duplication].
ARXIV_BLACKLISTED_SELECTORS = [
    "h2.ltx_title_bibliography",
    "div.ltx_classification",
    "span.ltx_role_author",
    "h1.ltx_title",
]


# Download the ARXIV HTML dumps for all 3 splits: no-problem, warnings, errors
# Dataset Source: https://sigmathling.kwarc.info/resources/ar5iv-dataset-2024/
#
# Note: The following steps download the ar5iv dataset and process it into sharded
# jsonl.gz files. The download function handles both GCS paths and direct zip URLs,
# we already have the dataset in GCS, so we use the GCS path.
ar5iv_no_problem_raw = ExecutorStep(
    name="raw/ar5iv/ar5iv-04-2024-no-problem",
    fn=download,
    config=DownloadConfig(
        input_path="gs://marin-us-central2/raw/ar5iv/v04.2024/ar5iv-04-2024-no-problem.zip",
        output_path=this_output_path(),
    ),
)

ar5iv_warnings_raw = ExecutorStep(
    name="raw/ar5iv/ar5iv-04-2024-warning",
    fn=download,
    config=DownloadConfig(
        input_path="gs://marin-us-central2/raw/ar5iv/v04.2024/ar5iv-04-2024-warnings.zip",
        output_path=this_output_path(),
    ),
)

ar5iv_errors_raw = ExecutorStep(
    name="raw/ar5iv/ar5iv-04-2024-errors",
    fn=download,
    config=DownloadConfig(
        input_path="gs://marin-us-central2/raw/ar5iv/v04.2024/ar5iv-04-2024-errors.zip",
        output_path=this_output_path(),
    ),
)

ar5iv_no_problem_raw_202404 = output_path_of(ar5iv_no_problem_raw, "202404")
ar5iv_warnings_raw_202404 = output_path_of(ar5iv_warnings_raw, "202404")
ar5iv_errors_raw_202404 = output_path_of(ar5iv_errors_raw, "202404")


def get_ar5iv_extraction_step(extraction_method: str, extraction_config: ExtractionConfig):
    """
    Returns a tuple of ExecutorSteps for the markdownification of the ar5iv dataset(no-problem, warnings, errors)
    for the given extraction method and configuration.

    Args:
        extraction_method: The method to use for the extraction.
        extraction_config: The configuration to use for the extraction.

    Returns:
        A tuple of ExecutorSteps for the markdownification of the ar5iv dataset.
    """
    output_path_basename = extraction_method
    if isinstance(extraction_config, ResiliparseConfig) and extraction_config.use_custom_variant:
        output_path_basename = "resiliparse-custom-fork"

    no_problem_step = ExecutorStep(
        name="documents/ar5iv/ar5iv-04-2024-no-problem",
        fn=process_ar5iv_dump,
        config=Ar5ivExtractionConfig(
            input_path=ar5iv_no_problem_raw_202404,
            revision="042024",
            output_path=this_output_path(output_path_basename),
            extract_method=versioned(extraction_method),
            extract_config=extraction_config,
            remove_reference_section=versioned(True),
        ),
        pip_dependency_groups=["download_transform"],
    )

    warnings_step = ExecutorStep(
        name="documents/ar5iv/ar5iv-04-2024-warnings",
        config=Ar5ivExtractionConfig(
            input_path=ar5iv_warnings_raw_202404,
            revision="042024",
            output_path=this_output_path(output_path_basename),
            extract_method=versioned(extraction_method),
            extract_config=extraction_config,
            remove_reference_section=versioned(True),
        ),
        pip_dependency_groups=["download_transform"],
    )

    errors_step = ExecutorStep(
        name="documents/ar5iv/ar5iv-04-2024-errors",
        config=Ar5ivExtractionConfig(
            input_path=ar5iv_errors_raw_202404,
            revision="042024",
            output_path=this_output_path(output_path_basename),
            extract_method=versioned(extraction_method),
            extract_config=extraction_config,
            remove_reference_section=versioned(True),
        ),
        pip_dependency_groups=["download_transform"],
    )

    return (no_problem_step, warnings_step, errors_step)


# Markdownification using readability without references and without links
ar5iv_no_problem_readability_no_references_no_links = ExecutorStep(
    name="documents/ar5iv/ar5iv-04-2024-no-problem-readability-no-references-no-links",
    fn=process_ar5iv_dump,
    config=Ar5ivExtractionConfig(
        input_path=ar5iv_no_problem_raw_202404,
        revision="042024",
        output_path=this_output_path("readability-no-references-no-links"),
        extract_method=versioned("readability"),
        extract_config=HtmlToMarkdownConfig(
            include_images=False,
            include_links=False,
        ),
        remove_reference_section=versioned(True),
    ),
)

# Markdownification using readability without references and with links
ar5iv_no_problem_readability_no_references_with_links = ExecutorStep(
    name="documents/ar5iv/ar5iv-04-2024-no-problem-readability-no-references-with-links",
    fn=process_ar5iv_dump,
    config=Ar5ivExtractionConfig(
        input_path=ar5iv_no_problem_raw_202404,
        revision="042024",
        output_path=this_output_path("readability-no-references-with-links"),
        extract_method=versioned("readability"),
        extract_config=HtmlToMarkdownConfig(
            include_images=False,
            include_links=True,
        ),
        remove_reference_section=versioned(True),
    ),
)

# Markdownification using readability with references and without links
ar5iv_no_problem_readability_with_references_no_links = ExecutorStep(
    name="documents/ar5iv/ar5iv-04-2024-no-problem-readability-with-references-no-links",
    fn=process_ar5iv_dump,
    config=Ar5ivExtractionConfig(
        input_path=ar5iv_no_problem_raw_202404,
        revision="042024",
        output_path=this_output_path("readability-with-references-no-links"),
        extract_method=versioned("readability"),
        extract_config=HtmlToMarkdownConfig(
            include_images=False,
            include_links=False,
        ),
        remove_reference_section=versioned(False),
    ),
)

# Markdownification using readability with references and with links
ar5iv_no_problem_readability_with_references_with_links = ExecutorStep(
    name="documents/ar5iv/ar5iv-04-2024-no-problem-readability-with-references-with-links",
    fn=process_ar5iv_dump,
    config=Ar5ivExtractionConfig(
        input_path=ar5iv_no_problem_raw_202404,
        revision="042024",
        output_path=this_output_path("readability-with-references-with-links"),
        extract_method=versioned("readability"),
        extract_config=HtmlToMarkdownConfig(
            include_images=False,
            include_links=True,
        ),
        remove_reference_section=versioned(False),
    ),
)

# Text extraction using Resiliparse without references and without links
ar5iv_no_problem_resiliparse_no_references_no_links = ExecutorStep(
    name="documents/ar5iv/ar5iv-04-2024-no-problem-resiliparse-no-references-no-links",
    fn=process_ar5iv_dump,
    config=Ar5ivExtractionConfig(
        input_path=ar5iv_no_problem_raw_202404,
        revision="042024",
        output_path=this_output_path("resiliparse-no-references-no-links"),
        extract_method=versioned("resiliparse"),
        extract_config=ResiliparseConfig(
            preserve_formatting=True,
            main_content=True,
            links=versioned(False),
            prepend_title=True,
            skip_elements=ARXIV_BLACKLISTED_SELECTORS,
            use_custom_variant=False,
        ),
        remove_reference_section=versioned(True),
    ),
)

# Text extraction using Resiliparse without references and with links
ar5iv_no_problem_resiliparse_no_references_with_links = ExecutorStep(
    name="documents/ar5iv/ar5iv-04-2024-no-problem-resiliparse-no-references-with-links",
    fn=process_ar5iv_dump,
    config=Ar5ivExtractionConfig(
        input_path=ar5iv_no_problem_raw_202404,
        revision="042024",
        output_path=this_output_path("resiliparse-no-references-with-links"),
        extract_method=versioned("resiliparse"),
        extract_config=ResiliparseConfig(
            preserve_formatting=True,
            main_content=True,
            links=versioned(True),
            prepend_title=True,
            skip_elements=ARXIV_BLACKLISTED_SELECTORS,
            use_custom_variant=False,
        ),
        remove_reference_section=versioned(True),
    ),
)

# Text extraction using Resiliparse with references and without links
ar5iv_no_problem_resiliparse_with_references_no_links = ExecutorStep(
    name="documents/ar5iv/ar5iv-04-2024-no-problem-resiliparse-with-references-no-links",
    fn=process_ar5iv_dump,
    config=Ar5ivExtractionConfig(
        input_path=ar5iv_no_problem_raw_202404,
        revision="042024",
        output_path=this_output_path("resiliparse-with-references-no-links"),
        extract_method=versioned("resiliparse"),
        extract_config=ResiliparseConfig(
            preserve_formatting=True,
            main_content=True,
            links=versioned(False),
            prepend_title=True,
            skip_elements=ARXIV_BLACKLISTED_SELECTORS,
            use_custom_variant=False,
        ),
        remove_reference_section=versioned(False),
    ),
)

# Text extraction using Resiliparse with references and with links
ar5iv_no_problem_resiliparse_with_references_with_links = ExecutorStep(
    name="documents/ar5iv/ar5iv-04-2024-no-problem-resiliparse-with-references-with-links",
    fn=process_ar5iv_dump,
    config=Ar5ivExtractionConfig(
        input_path=ar5iv_no_problem_raw_202404,
        revision="042024",
        output_path=this_output_path("resiliparse-with-references-with-links"),
        extract_method=versioned("resiliparse"),
        extract_config=ResiliparseConfig(
            preserve_formatting=True,
            main_content=True,
            links=versioned(True),
            prepend_title=True,
            skip_elements=ARXIV_BLACKLISTED_SELECTORS,
            use_custom_variant=False,
        ),
        remove_reference_section=versioned(False),
    ),
)

# Markdownification using Resiliparse custom fork without references and links
(
    ar5iv_no_problem_resiliparse_custom_fork,
    ar5iv_warnings_resiliparse_custom_fork,
    ar5iv_errors_resiliparse_custom_fork,
) = get_ar5iv_extraction_step(
    "resiliparse",
    ResiliparseConfig(
        preserve_formatting=True,
        main_content=True,
        links=versioned(False),
        prepend_title=True,
        skip_elements=ARXIV_BLACKLISTED_SELECTORS,
        use_custom_variant=True,
    ),
)

if __name__ == "__main__":
    # We decided to only run the custom fork of Resiliparse
    # without references and links. This was mostly intuition
    # driven based on the results of the previous experiments with wikipedia.
    executor_main(
        steps=[
            ar5iv_no_problem_raw,
            ar5iv_warnings_raw,
            ar5iv_errors_raw,
            ar5iv_no_problem_resiliparse_custom_fork,
            ar5iv_warnings_resiliparse_custom_fork,
            ar5iv_errors_resiliparse_custom_fork,
        ]
    )
