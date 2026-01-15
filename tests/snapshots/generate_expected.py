"""Regenerate all expected outputs for snapshot tests.

Usage:
    uv run python tests/snapshots/generate_expected.py
"""

import json
import os
import re

from bs4 import BeautifulSoup
from marin.schemas.web.convert import HtmlToMarkdownConfig, ResiliparseConfig
from marin.schemas.web.selectors import ARXIV_BLACKLISTED_SELECTORS, WIKI_BLACKLISTED_SELECTORS
from marin.transform.ar5iv.transform_ar5iv import clean_html
from marin.transform.stackexchange.transform_stackexchange import prepare_md_template
from marin.transform.wikipedia.transform_wikipedia import clean_wiki_html
from marin.web.convert import convert_page

MY_DIR = os.path.dirname(os.path.realpath(__file__))


def transform_web(html: str) -> str:
    """Transform web HTML to markdown using resiliparse."""
    output = convert_page(html, extract_method="resiliparse", config=ResiliparseConfig())
    return output["content"]


def transform_wiki(html: str) -> str:
    """Transform Wikipedia HTML to markdown."""
    filtered_html = clean_wiki_html(html, remove_reference_section=True)
    output = convert_page(
        filtered_html,
        extract_method="resiliparse",
        config=ResiliparseConfig(
            links=False,
            skip_elements=WIKI_BLACKLISTED_SELECTORS,
            markdownify_config=HtmlToMarkdownConfig(
                include_images=False,
                include_links=False,
            ),
        ),
    )
    return output["content"]


def transform_ar5iv(html: str) -> str:
    """Transform ar5iv HTML to markdown."""
    bs4_html = BeautifulSoup(html, "html.parser")
    html = str(bs4_html)
    filtered_html = clean_html(html, remove_reference_section=True)

    output = convert_page(
        filtered_html,
        extract_method="resiliparse",
        config=ResiliparseConfig(
            links=False,
            prepend_title=True,
            skip_elements=ARXIV_BLACKLISTED_SELECTORS,
            markdownify_config=HtmlToMarkdownConfig(
                include_images=False,
                include_links=False,
            ),
        ),
    )

    content = output["content"]
    # Remove citation references
    content = re.sub(r"\s?\\\[(?:\d+(?:,\s*\d+)*)\\\]", "", content)
    return content


def transform_stackexchange(json_str: str) -> str:
    """Transform StackExchange JSON to markdown."""
    row = json.loads(json_str)

    title = row["metadata"]["title"] if "title" in row["metadata"] else row["title"]
    question = row["metadata"]["question"] if "question" in row["metadata"] else row["question"]
    answers = row["metadata"]["answers"]
    tags = row["metadata"]["tags"] if "tags" in row["metadata"] else row["tags"]

    config = ResiliparseConfig(
        links=False,
        prepend_title=True,
        skip_elements=[],
        markdownify_config=HtmlToMarkdownConfig(
            include_images=False,
            include_links=False,
        ),
    )
    content = prepare_md_template(title, question, answers, tags, "resiliparse", config, prepend_vote_count=False)
    return content.strip()


def transform_dclm_hq(html: str) -> str:
    """Transform DCLM HQ HTML to markdown."""
    output = convert_page(
        html,
        extract_method="resiliparse",
        config=ResiliparseConfig(
            links=False,
            skip_elements=[],
            markdownify_config=HtmlToMarkdownConfig(
                include_images=False,
                include_links=False,
            ),
        ),
    )
    return output["content"]


DATASETS = {
    "web/resiliparse": (".html", transform_web),
    "wiki": (".html", transform_wiki),
    "ar5iv": (".html", transform_ar5iv),
    "stackexchange": (".json", transform_stackexchange),
    "dclm_hq": (".html", transform_dclm_hq),
}


def regenerate_dataset(name: str, input_ext: str, transform_fn) -> None:
    """Regenerate expected outputs for a dataset."""
    input_dir = os.path.join(MY_DIR, name.split("/")[0], "inputs")
    expected_dir = os.path.join(MY_DIR, name, "expected")

    # Handle web/resiliparse special case
    if name == "web/resiliparse":
        expected_dir = os.path.join(MY_DIR, "web", "expected", "resiliparse")

    os.makedirs(expected_dir, exist_ok=True)

    input_files = [f for f in os.listdir(input_dir) if f.endswith(input_ext)]

    print(f"Regenerating {name} ({len(input_files)} files)")
    for input_file in input_files:
        with open(os.path.join(input_dir, input_file), "r") as f:
            content = f.read()

        expected = transform_fn(content)
        output_name = input_file.replace(input_ext, ".md")

        with open(os.path.join(expected_dir, output_name), "w") as f:
            f.write(expected)
            f.write("\n")


def main():
    for name, (input_ext, transform_fn) in DATASETS.items():
        regenerate_dataset(name, input_ext, transform_fn)


if __name__ == "__main__":
    main()
