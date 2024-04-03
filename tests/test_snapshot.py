import subprocess
import os

import pytest

from markweb.markweb import convert_page

my_dir = os.path.dirname(os.path.realpath(__file__))
input_dir = os.path.join(my_dir, "snapshots/inputs")  # Directory containing HTML input files
expected_dir = os.path.join(my_dir, "snapshots/expected")  # Directory containing expected output files
output_dir = os.path.join(my_dir, "snapshots/outputs")  # Directory containing actual output files
diff_dir = os.path.join(my_dir, "snapshots/diffs")  # Directory containing diff files

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def compare_outputs(input_name, expected_file, output_file):
    """Compare expected and actual output files using pandiff."""
    # first see if they're identical
    with open(expected_file, "r") as f, open(output_file, "r") as g:
        if f.read() == g.read():
            return

    os.makedirs(diff_dir, exist_ok=True)
    base_name = os.path.basename(expected_file)
    diff_name = f"{base_name}.diff.md"
    try:
        result = subprocess.run(["pandiff", expected_file, output_file, "-o", f"{diff_dir}/{diff_name}"], check=True, text=True, capture_output=True)
        # print stdout and stderr and the command
        print(result.stdout)
        print(result.stderr)
        print(result.args)
    except subprocess.CalledProcessError as e:
        print(f"Error running pandiff: {e}")
        raise

    # show the diff
    raise AssertionError(f"Output does not match expected for {input_name}. See snapshots/diffs/{diff_name} for details.")


def parametrize_with_files(fn):
    """Parametrize a test function with the files in the input directory."""
    files = [ os.path.splitext(os.path.basename(f))[0] for f in os.listdir(input_dir) if f.endswith(".html")]
    return pytest.mark.parametrize("input_name", files)(fn)


@parametrize_with_files
def test_generate_markdown_from_html(input_name):
    """Test the Markdown generation from HTML and compare outputs."""
    input_path = os.path.join(input_dir, f"{input_name}.html")
    input_content = read_file(input_path)

    output_dict = convert_page(input_content)

    output = output_dict["content"]

    expected_file = os.path.join(expected_dir, f"{input_name}.md")
    output_file = os.path.join(output_dir, f"{input_name}.md")

    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        f.write(output)

    compare_outputs(input_name, expected_file, output_file)

def accept_change(input_name):
    """Accept a change for a specific input."""
    expected_file = os.path.join(expected_dir, f"{input_name}.md")
    output_file = os.path.join(output_dir, f"{input_name}.md")
    os.replace(output_file, expected_file)
    print(f"Accepted changes for {input_name}.")
