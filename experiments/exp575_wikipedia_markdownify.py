"""
Experiment to convert Wikipedia HTML dumps to Markdown format.

This experiment downloads and processes the Wikipedia HTML dumps to extract clean markdown/text.
We prepare the text/markdown for use as a training dataset for a language model, over 3 settings:

* The readability algorithm, which does not support text extraction implicitly but with markdownify.
* The default Resiliparse configuration, which removes boilerplate but does not support markdownification.
* Our custom fork of Resiliparse which provides a simplified DOM tree with removed boilerplate which can be
passed to Markdownify to producing Markdown text that has less noise.

Reference Issue: https://github.com/stanford-crfm/marin/issues/575
"""

from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.schemas.web.convert import HtmlToMarkdownConfig, ResiliparseConfig
from operations.download.wikipedia.download import DownloadConfig, download
from operations.transform.wikipedia.transform_wikipedia import WikiExtractionConfig, process_wiki_dump

# Selectors to remove from the DOM tree, these mostly contain footers, headers, navigation elements,
# reference sections, link clusters, filler sections, etc.
WIKI_BLACKLISTED_SELECTORS = [
    "div.navbox",
    "span.portal-bar",
    "div#catlinks",
    "h2#References",
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
    "h3#Sources",
    "ol.references",
    "div#mw-indicator-coordinates",
]

# Download the Wikipedia HTML dump
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

wikipedia_dump_raw_20241201 = output_path_of(wikipedia_dump_raw, "20241201")

# Markdownification using the readability algorithm
wikipedia_readability = ExecutorStep(
    name="documents/wikipedia-readability",
    fn=process_wiki_dump,
    config=WikiExtractionConfig(
        input_path=wikipedia_dump_raw_20241201,
        revision=versioned("20241201"),
        output_path=this_output_path(),
        extract_method="readability",
        extract_config=HtmlToMarkdownConfig(
            include_images=False,
            include_links=False,
        ),
        remove_reference_section=versioned(True),
        digit_threshold=versioned(50),
        word_threshold=versioned(70),
        special_char_threshold=versioned(50),
    ),
)

# Text extraction using the default Resiliparse configuration
wikipedia_resiliparse_with_pf = ExecutorStep(
    name="documents/wikipedia-resiliparse-with-preserve-formatting",
    fn=process_wiki_dump,
    config=WikiExtractionConfig(
        input_path=wikipedia_dump_raw_20241201,
        revision=versioned("20241201"),
        output_path=this_output_path(),
        extract_method="resiliparse",
        extract_config=ResiliparseConfig(
            preserve_formatting=True,
            main_content=True,
            links=False,
            skip_elements=WIKI_BLACKLISTED_SELECTORS,
            use_custom_variant=False,
        ),
        remove_reference_section=versioned(True),
        digit_threshold=versioned(50),
        word_threshold=versioned(70),
        special_char_threshold=versioned(50),
    ),
)

# Markdownification using our custom fork of Resiliparse
wikipedia_resiliparse_custom_fork = ExecutorStep(
    name="documents/wikipedia-resiliparse-custom-fork",
    fn=process_wiki_dump,
    config=WikiExtractionConfig(
        input_path=wikipedia_dump_raw_20241201,
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
    # We decided to only run the custom fork of Resiliparse
    executor_main(
        steps=[
            wikipedia_dump_raw,
            wikipedia_resiliparse_custom_fork,
        ]
    )
