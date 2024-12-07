from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.schemas.web.convert import ResiliparseConfig
from operations.download.wikipedia.download import DownloadConfig, download
from operations.transform.wikipedia.transform_wikipedia import WikiExtractionConfig, process_wiki_dump

wikipedia_dump_raw = ExecutorStep(
    name="raw/wikipedia",
    fn=download,
    config=DownloadConfig(
        input_urls=versioned(
            [
                "https://dumps.wikimedia.org/other/enterprise_html/runs/20241201/enwiki-NS0-20241201-ENTERPRISE-HTML.json.tar.gz",
            ]
        ),
        revision=versioned("20241201"),
        output_path=this_output_path(),
    ),
    pip_dependency_groups=["download_transform"],
)

wikipedia_text_extracted = ExecutorStep(
    name="documents/wikipedia-resiliparse-with-preserving-formatting",
    fn=process_wiki_dump,
    config=WikiExtractionConfig(
        input_path=output_path_of(wikipedia_dump_raw),
        revision="20241201",
        output_path=this_output_path(),
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
            wikipedia_dump_raw,
            wikipedia_text_extracted,
        ]
    )
