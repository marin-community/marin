import os
import subprocess

import pytest

from marin.schemas.web.convert import ResiliparseConfig, TrafilaturaConfig
from marin.web.convert import convert_page

my_path = os.path.dirname(os.path.realpath(__file__))
input_path = os.path.join(my_path, "snapshots/inputs")  # Directory containing HTML input files
expected_path = os.path.join(my_path, "snapshots/expected")  # Directory containing expected output files
output_path = os.path.join(my_path, "snapshots/outputs")  # Directory containing actual output files
diff_path = os.path.join(my_path, "snapshots/diffs")  # Directory containing diff files


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def compare_outputs(input_name, expected_file, output_file):
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
    raise AssertionError(
        f"Output does not match expected for {input_name}. See snapshots/diffs/{diff_name} for details."
    )


def parametrize_with_files(fn):
    """Parametrize a test function with the files in the input directory."""
    files = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(input_path) if f.endswith(".html")]
    return pytest.mark.parametrize("input_name", files)(fn)


@parametrize_with_files
def test_generate_markdown_from_html_with_readability(input_name):
    """Test the Markdown generation from HTML and compare outputs using the Readability method."""
    input_file = os.path.join(input_path, f"{input_name}.html")
    input_content = read_file(input_file)

    output_dict = convert_page(input_content, extract_method="readability")

    output = output_dict["content"]

    expected_file = os.path.join(expected_path, "readability", f"{input_name}.md")
    output_file = os.path.join(output_path, "readability", f"{input_name}.md")

    os.makedirs(f"{output_path}/readability", exist_ok=True)

    with open(output_file, "w") as f:
        f.write(output)

    compare_outputs(input_name, expected_file, output_file)


@parametrize_with_files
def test_generate_markdown_from_html_with_resiliparse(input_name):
    """Test the Markdown generation from HTML and compare outputs using the Resiliparse method[NON MARKDOWN]."""
    input_file = os.path.join(input_path, f"{input_name}.html")
    input_content = read_file(input_file)

    output_dict = convert_page(input_content, extract_method="resiliparse", config=ResiliparseConfig.default_config())

    output = output_dict["content"]

    expected_file = os.path.join(expected_path, "resiliparse", f"{input_name}.md")
    output_file = os.path.join(output_path, "resiliparse", f"{input_name}.md")

    os.makedirs(f"{output_path}/resiliparse", exist_ok=True)

    with open(output_file, "w") as f:
        f.write(output)

    compare_outputs(input_name, expected_file, output_file)


@parametrize_with_files
def test_generate_markdown_from_html_with_trafilatura(input_name):
    """Test the Markdown generation from HTML and compare outputs using the Trafilatura method."""
    input_file = os.path.join(input_path, f"{input_name}.html")
    input_content = read_file(input_file)

    config = TrafilaturaConfig.default_config()
    output_dict = convert_page(input_content, extract_method="trafilatura", config=config)

    output = output_dict["content"]

    expected_file = os.path.join(expected_path, "trafilatura", f"{input_name}.md")
    output_file = os.path.join(output_path, "trafilatura", f"{input_name}.md")

    os.makedirs(f"{output_path}/trafilatura", exist_ok=True)

    with open(output_file, "w") as f:
        f.write(output)

    compare_outputs(input_name, expected_file, output_file)


def accept_change(input_name):
    """Accept a change for a specific input."""
    expected_file = os.path.join(expected_path, f"{input_name}.md")
    output_file = os.path.join(output_path, f"{input_name}.md")
    os.replace(output_file, expected_file)
    print(f"Accepted changes for {input_name}.")
