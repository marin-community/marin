from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.schemas.web.convert import HtmlToMarkdownConfig, ResiliparseConfig
from operations.download.wikipedia.download import DownloadConfig, download
from operations.transform.wikipedia.transform_wikipedia import WikiExtractionConfig, process_wiki_dump

WIKI_BLACKLISTED_SELECTORS = [
    "div.navbox",
    "span.portal-bar",
    "div#catlinks",
    "h2#External_links",
    "h2#See_also",
    "div#p-navigation",
    "span.mw-editsection",
    "h2.Further_reading",
    "header",
    "a.mw-jump-link",
    "div.printfooter",
    "div.vector-header-container",
    ".noprint",
    "span.mw-cite-backlink",
    "sup.reference",
    "div#mw-indicators",
    "span.portal-barion",
    "h2#Notes",
    "div#mw-indicator-coordinates",
]

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

wikipedia_text_resiliparse_custom_fork = ExecutorStep(
    name="documents/wikipedia-resiliparse-custom-fork",
    fn=process_wiki_dump,
    config=WikiExtractionConfig(
        input_path=output_path_of(wikipedia_dump_raw, "20241201"),
        revision=versioned("20241201"),
        output_path=this_output_path(),
        extract_method="resiliparse",
        extract_config=ResiliparseConfig(
            preserve_formatting=True,
            main_content=True,
            links=False,
            skip_elements=WIKI_BLACKLISTED_SELECTORS,
            use_custom_variant=True,
            markdownify_config=HtmlToMarkdownConfig(
                include_images=False,
                include_links=False,
            ),
        ),
        remove_reference_section=versioned(True),
        digit_threshold=versioned(50),
        word_threshold=versioned(70),
        special_char_threshold=versioned(50),
    ),
    pip_dependency_groups=["download_transform"],
)

if __name__ == "__main__":
    executor_main(
        steps=[
            wikipedia_dump_raw,
            wikipedia_text_resiliparse_custom_fork,
        ]
    )
