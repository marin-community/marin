import json
import os
import re
import subprocess

import pytest
from bs4 import BeautifulSoup

from experiments.exp575_wikipedia_markdownify import WIKI_BLACKLISTED_SELECTORS
from experiments.exp579_ar5iv_markdownify import ARXIV_BLACKLISTED_SELECTORS
from marin.schemas.web.convert import HtmlToMarkdownConfig, ResiliparseConfig, TrafilaturaConfig
from marin.web.convert import convert_page
from operations.transform.ar5iv.transform_ar5iv import clean_html
from operations.transform.wikipedia.transform_wikipedia import clean_wiki_html
from tests.snapshots.stackexchange.accept_changes import prepare_expected_output
from tests.test_utils import skip_if_module_missing

my_path = os.path.dirname(os.path.realpath(__file__))

web_input_path = os.path.join(my_path, "snapshots/web/inputs")  # Directory containing HTML input files
web_expected_path = os.path.join(my_path, "snapshots/web/expected")  # Directory containing expected output files
web_output_path = os.path.join(my_path, "snapshots/web/outputs")  # Directory containing actual output files
web_diff_path = os.path.join(my_path, "snapshots/web/diffs")  # Directory containing diff files

wiki_input_path = os.path.join(my_path, "snapshots/wiki/inputs")
wiki_expected_path = os.path.join(my_path, "snapshots/wiki/expected")
wiki_output_path = os.path.join(my_path, "snapshots/wiki/outputs")
wiki_diff_path = os.path.join(my_path, "snapshots/wiki/diffs")

ar5iv_input_path = os.path.join(my_path, "snapshots/ar5iv/inputs")
ar5iv_expected_path = os.path.join(my_path, "snapshots/ar5iv/expected")
ar5iv_output_path = os.path.join(my_path, "snapshots/ar5iv/outputs")
ar5iv_diff_path = os.path.join(my_path, "snapshots/ar5iv/diffs")

stackexchange_input_path = os.path.join(my_path, "snapshots/stackexchange/inputs")
stackexchange_expected_path = os.path.join(my_path, "snapshots/stackexchange/expected")
stackexchange_output_path = os.path.join(my_path, "snapshots/stackexchange/outputs")
stackexchange_diff_path = os.path.join(my_path, "snapshots/stackexchange/diffs")


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def compare_outputs(input_name, expected_file, output_file, diff_path):
    """Compare expected and actual output files using pandiff."""
    # first see if they're identical
    with open(expected_file, "r") as f, open(output_file, "r") as g:
        if f.read() == g.read():
            return

    os.makedirs(diff_path, exist_ok=True)
    base_name = os.path.basename(expected_file)
    diff_name = f"{base_name}.diff.md"
    # first see if we can even expect to run pandiff by using which
    try:
        subprocess.run(
            ["which", "pandiff"],
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        raise AssertionError(
            f"Output does not match expected for {input_name}. pandiff not found, skipping diff."
        ) from None

    try:
        subprocess.run(
            ["pandiff", expected_file, output_file, "-o", f"{diff_path}/{diff_name}"],
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise AssertionError(f"Output does not match expected for {input_name}. Error running pandiff: {e}") from e

    # show the diff
    raise AssertionError(f"Output does not match expected for {input_name}. See {diff_path}/{diff_name} for details.")


def parametrize_files(fn=None, *, split="web", ext=".html"):
    """Parametrize a test function with the files in the input directory."""

    def decorator(fn):
        input_path = None
        match split:
            case "web":
                input_path = web_input_path
            case "wiki":
                input_path = wiki_input_path
            case "ar5iv":
                input_path = ar5iv_input_path
            case "stackexchange":
                input_path = stackexchange_input_path
            case _:
                raise ValueError(f"Invalid split: {split}")

        files = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(input_path) if f.endswith(ext)]
        return pytest.mark.parametrize("input_name", files)(fn)

    if fn is None:
        return decorator
    return decorator(fn)


@parametrize_files
def test_generate_markdown_from_html_with_readability(input_name):
    """Test the Markdown generation from HTML and compare outputs using the Readability method."""
    input_file = os.path.join(web_input_path, f"{input_name}.html")
    input_content = read_file(input_file)

    output_dict = convert_page(input_content, extract_method="readability")

    output = output_dict["content"]

    expected_file = os.path.join(web_expected_path, "readability", f"{input_name}.md")
    output_file = os.path.join(web_output_path, "readability", f"{input_name}.md")

    os.makedirs(f"{web_output_path}/readability", exist_ok=True)

    with open(output_file, "w") as f:
        print(output, file=f)

    compare_outputs(input_name, expected_file, output_file, os.path.join(web_diff_path, "readability"))


@skip_if_module_missing("resiliparse_dom")
@parametrize_files
def test_generate_markdown_from_html_with_resiliparse(input_name):
    """Test the Markdown generation from HTML and compare outputs using the Resiliparse method[NON MARKDOWN]."""
    input_file = os.path.join(web_input_path, f"{input_name}.html")
    input_content = read_file(input_file)

    output_dict = convert_page(input_content, extract_method="resiliparse", config=ResiliparseConfig.default_config())

    output = output_dict["content"]

    expected_file = os.path.join(web_expected_path, "resiliparse", f"{input_name}.md")
    output_file = os.path.join(web_output_path, "resiliparse", f"{input_name}.md")

    os.makedirs(f"{web_output_path}/resiliparse", exist_ok=True)

    with open(output_file, "w") as f:
        print(output, file=f)

    compare_outputs(input_name, expected_file, output_file, os.path.join(web_diff_path, "resiliparse"))


@parametrize_files
def test_generate_markdown_from_html_with_trafilatura(input_name):
    """Test the Markdown generation from HTML and compare outputs using the Trafilatura method."""
    input_file = os.path.join(web_input_path, f"{input_name}.html")
    input_content = read_file(input_file)

    config = TrafilaturaConfig.default_config()
    output_dict = convert_page(input_content, extract_method="trafilatura", config=config)

    output = output_dict["content"]

    expected_file = os.path.join(web_expected_path, "trafilatura", f"{input_name}.md")
    output_file = os.path.join(web_output_path, "trafilatura", f"{input_name}.md")

    os.makedirs(f"{web_output_path}/trafilatura", exist_ok=True)

    with open(output_file, "w") as f:
        print(output, file=f)

    compare_outputs(input_name, expected_file, output_file, os.path.join(web_diff_path, "trafilatura"))


@skip_if_module_missing("resiliparse_dom")
@parametrize_files(split="wiki")
def test_markdownify_wikipedia(input_name):
    """Test the Markdownify method on Wikipedia."""
    input_file = os.path.join(wiki_input_path, f"{input_name}.html")
    input_content = read_file(input_file)

    filtered_html = clean_wiki_html(input_content, remove_reference_section=True)
    output_dict = convert_page(
        filtered_html,
        extract_method="resiliparse",
        config=ResiliparseConfig(
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
    )

    output = output_dict["content"]

    expected_file = os.path.join(wiki_expected_path, f"{input_name}.md")
    output_file = os.path.join(wiki_output_path, f"{input_name}.md")

    os.makedirs(f"{wiki_output_path}", exist_ok=True)

    with open(output_file, "w") as f:
        print(output, file=f)

    compare_outputs(input_name, expected_file, output_file, wiki_diff_path)


@skip_if_module_missing("resiliparse_dom")
@parametrize_files(split="ar5iv")
def test_markdownify_ar5iv(input_name):
    """Test the Markdownify method on ar5iv."""
    remove_reference_section = True
    input_file = os.path.join(ar5iv_input_path, f"{input_name}.html")
    input_content = read_file(input_file)

    bs4_html = BeautifulSoup(input_content, "html.parser")

    html = str(bs4_html)
    filtered_html = clean_html(html, remove_reference_section=True)

    output_dict = convert_page(
        filtered_html,
        extract_method="resiliparse",
        config=ResiliparseConfig(
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
        ),
    )

    output = output_dict["content"]

    if remove_reference_section:
        output = re.sub(r"\s?\\\[(?:\d+(?:,\s*\d+)*)\\\]", "", output)

    expected_file = os.path.join(ar5iv_expected_path, f"{input_name}.md")
    output_file = os.path.join(ar5iv_output_path, f"{input_name}.md")

    os.makedirs(f"{ar5iv_output_path}", exist_ok=True)

    with open(output_file, "w") as f:
        print(output, file=f)

    compare_outputs(input_name, expected_file, output_file, ar5iv_diff_path)


@skip_if_module_missing("resiliparse_dom")
@parametrize_files(split="stackexchange", ext=".json")
def test_markdownify_stackexchange(input_name):
    """Test the Markdownify method on stackexchange."""
    input_file = os.path.join(stackexchange_input_path, f"{input_name}.json")
    input_content = read_file(input_file)

    row = json.loads(input_content)
    output = prepare_expected_output(
        row,
        "resiliparse",
        ResiliparseConfig(
            preserve_formatting=True,
            main_content=True,
            links=False,
            prepend_title=True,
            skip_elements=[],
            use_custom_variant=True,
            markdownify_config=HtmlToMarkdownConfig(
                include_images=False,
                include_links=False,
            ),
        ),
        shuffle_answers_template=False,
    )
    expected_file = os.path.join(stackexchange_expected_path, f"{input_name}.md")
    output_file = os.path.join(stackexchange_output_path, f"{input_name}.md")

    os.makedirs(f"{stackexchange_output_path}", exist_ok=True)

    with open(output_file, "w") as f:
        print(output, file=f)

    compare_outputs(input_name, expected_file, output_file, stackexchange_diff_path)
