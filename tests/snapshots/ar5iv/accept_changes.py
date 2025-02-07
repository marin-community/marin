
import os

from tqdm import tqdm

from marin.web.convert import convert_page
from operations.transform.ar5iv.transform_ar5iv import clean_html
from operations.transform.wikipedia.transform_wikipedia import clean_wiki_html
from marin.schemas.web.convert import ExtractionConfig, ResiliparseConfig, HtmlToMarkdownConfig


def prepare_expected_output(html: str, extract_method: str, extract_config: ExtractionConfig) -> str:
    remove_reference_section = False
    filtered_html = clean_html(html, remove_reference_section)

    content = convert_page(
        filtered_html, extract_method=extract_method, config=extract_config
    )["content"]
    
    return content


if __name__ == "__main__":
    input_path = "tests/snapshots/ar5iv/inputs"
    expected_path = "tests/snapshots/ar5iv/expected"

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
                prepend_title=True,
                skip_elements=["h2.ltx_title_bibliography", "div.ltx_classification", "span.ltx_role_author"],
                use_custom_variant=True,
                markdownify_config=HtmlToMarkdownConfig(
                    include_images=False,
                    include_links=False,
                )
            )
            expected = prepare_expected_output(html, extract_method, extract_config)

            with open(os.path.join(expected_path, input_file.replace("html", "md")), "w") as f:
                f.write(expected)
