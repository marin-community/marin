
import os

from tqdm import tqdm

from marin.schemas.web.convert import ExtractionConfig, HtmlToMarkdownConfig, ResiliparseConfig
from marin.web.convert import convert_page
from operations.transform.wikipedia.transform_wikipedia import clean_wiki_html


def prepare_expected_output(html: str, extract_method: str, extract_config: ExtractionConfig) -> str:
    remove_reference_section = True

    filtered_html = clean_wiki_html(html, remove_reference_section)
    content = convert_page(
        filtered_html, extract_method=extract_method, config=extract_config
    )["content"]

    return content


if __name__ == "__main__":
    input_path = "tests/snapshots/wiki/inputs"
    expected_path = "tests/snapshots/wiki/expected"

    if not os.path.exists(expected_path):
        os.makedirs(expected_path)

    input_files = os.listdir(input_path)

    for input_file in tqdm(input_files):
        with open(os.path.join(input_path, input_file), "r") as f:
            html = f.read()

            extract_method = "resiliparse"
            extract_config = ResiliparseConfig(
                preserve_formatting=True,
                main_content=True,
                links=False,
                skip_elements=["div.navbox", "span.portal-bar", "div#catlinks", "h2#References", "h2#External_links", "h2#See_also", "div#p-navigation", "span.mw-editsection", "h2.Further_reading", "header", "a.mw-jump-link", "div.printfooter", "div.vector-header-container", ".noprint", "span.mw-cite-backlink", "sup.reference", "div#mw-indicators", "span.portal-barion", "h2#Notes", "h3#Sources", "ol.references"],
                use_custom_variant=True,
                markdownify_config=HtmlToMarkdownConfig(
                    include_images=False,
                    include_links=False,
                )
            )
            expected = prepare_expected_output(html, extract_method, extract_config)

            with open(os.path.join(expected_path, input_file.replace("html", "md")), "w") as f:
                f.write(expected)
