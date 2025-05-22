"""
This experiment is used to convert the DCLM HQ dump to markdown. DCLM HQ does not provide the WARC paths
for each document, so we use the Common Crawl index to get the WARC paths and then use the `download_dclm_hq_html`
operation to download the HTML content.

Note: At this moment we use a low spec machine to do this which causes the process to take a long time of ~3 months.

Reference Issue: https://github.com/stanford-crfm/marin/issues/813
"""

from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.download.dclm_hq.download_dclm_hq_html import DCLMHQDownloadConfig, extract_dclm_hq_dump

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
