import os
import re

from bs4 import BeautifulSoup
from tqdm import tqdm

from experiments.exp579_ar5iv_markdownify import ARXIV_BLACKLISTED_SELECTORS
from marin.schemas.web.convert import ExtractionConfig, HtmlToMarkdownConfig, ResiliparseConfig
from marin.transform.ar5iv.transform_ar5iv import clean_html
from marin.web.convert import convert_page


def prepare_expected_output(html: str, extract_method: str, extract_config: ExtractionConfig) -> str:
    remove_reference_section = True
    bs4_html = BeautifulSoup(html, "html.parser")

    html = str(bs4_html)
    filtered_html = clean_html(html, remove_reference_section)

    content = convert_page(filtered_html, extract_method=extract_method, config=extract_config)["content"]

    if remove_reference_section:
        content = re.sub(r"\s?\\\[(?:\d+(?:,\s*\d+)*)\\\]", "", content)

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
                skip_elements=ARXIV_BLACKLISTED_SELECTORS,
                use_custom_variant=True,
                markdownify_config=HtmlToMarkdownConfig(
                    include_images=False,
                    include_links=False,
                ),
            )
            expected = prepare_expected_output(html, extract_method, extract_config)

            with open(os.path.join(expected_path, input_file.replace("html", "md")), "w") as f:
                print(expected, file=f)
