# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import pytest

from tests.snapshots.generate_expected import (
    transform_ar5iv,
    transform_stackexchange,
    transform_web,
    transform_wiki,
)

TEST_PATH = os.path.dirname(os.path.realpath(__file__))
SPLITS = ["web", "wiki", "ar5iv", "stackexchange"]

SNAPSHOT_PATHS = {
    split: {
        "inputs": os.path.join(TEST_PATH, f"snapshots/{split}/inputs"),
        "expected": os.path.join(TEST_PATH, f"snapshots/{split}/expected"),
        "outputs": os.path.join(TEST_PATH, f"snapshots/{split}/outputs"),
        "diffs": os.path.join(TEST_PATH, f"snapshots/{split}/diffs"),
    }
    for split in SPLITS
}


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def compare_outputs(input_name, expected_file, output_file, diff_path):
    """Compare expected and actual output files using"""
    with open(expected_file, "r") as f, open(output_file, "r") as g:
        if f.read() == g.read():
            return

    os.makedirs(diff_path, exist_ok=True)
    import difflib

    diff = list(
        difflib.context_diff(
            read_file(expected_file).splitlines(keepends=True),
            read_file(output_file).splitlines(keepends=True),
            fromfile=expected_file,
            tofile=output_file,
        )
    )

    diff = "\n".join(diff)

    # add statistics, lines, total words etc
    expected_stats = {
        "lines": len(read_file(expected_file).splitlines()),
        "words": len(read_file(expected_file).split()),
    }
    output_stats = {
        "lines": len(read_file(output_file).splitlines()),
        "words": len(read_file(output_file).split()),
    }

    raise AssertionError(
        f"""Output does not match expected for {input_name}.

    Expected stats: {expected_stats}
    Output stats: {output_stats}

    Diff was:
    {diff}
    """
    )


def parametrize_files(fn=None, *, split="web", ext=".html"):
    """Parametrize a test function with the files in the input directory."""

    def decorator(fn):
        if split not in SNAPSHOT_PATHS:
            raise ValueError(f"Invalid split: {split}")

        input_path = SNAPSHOT_PATHS[split]["inputs"]

        files = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(input_path) if f.endswith(ext)]
        return pytest.mark.parametrize("input_name", files)(fn)

    if fn is None:
        return decorator
    return decorator(fn)


@parametrize_files
def test_generate_markdown_from_html_with_resiliparse(input_name):
    paths = SNAPSHOT_PATHS["web"]
    input_file = os.path.join(paths["inputs"], f"{input_name}.html")
    input_content = read_file(input_file)

    output = transform_web(input_content)

    expected_file = os.path.join(paths["expected"], "resiliparse", f"{input_name}.md")
    output_file = os.path.join(paths["outputs"], "resiliparse", f"{input_name}.md")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        f.write(output)
        f.write("\n")

    compare_outputs(input_name, expected_file, output_file, os.path.join(paths["diffs"], "resiliparse"))


@parametrize_files(split="wiki")
def test_markdownify_wikipedia(input_name):
    paths = SNAPSHOT_PATHS["wiki"]
    input_file = os.path.join(paths["inputs"], f"{input_name}.html")
    input_content = read_file(input_file)

    output = transform_wiki(input_content)

    expected_file = os.path.join(paths["expected"], f"{input_name}.md")
    output_file = os.path.join(paths["outputs"], f"{input_name}.md")

    os.makedirs(paths["outputs"], exist_ok=True)

    with open(output_file, "w") as f:
        f.write(output)
        f.write("\n")

    compare_outputs(input_name, expected_file, output_file, paths["diffs"])


@parametrize_files(split="ar5iv")
def test_markdownify_ar5iv(input_name):
    paths = SNAPSHOT_PATHS["ar5iv"]
    input_file = os.path.join(paths["inputs"], f"{input_name}.html")
    input_content = read_file(input_file)

    output = transform_ar5iv(input_content)

    expected_file = os.path.join(paths["expected"], f"{input_name}.md")
    output_file = os.path.join(paths["outputs"], f"{input_name}.md")

    os.makedirs(paths["outputs"], exist_ok=True)

    with open(output_file, "w") as f:
        f.write(output)
        f.write("\n")

    compare_outputs(input_name, expected_file, output_file, paths["diffs"])


@parametrize_files(split="stackexchange", ext=".json")
def test_markdownify_stackexchange(input_name):
    paths = SNAPSHOT_PATHS["stackexchange"]
    input_file = os.path.join(paths["inputs"], f"{input_name}.json")
    input_content = read_file(input_file)

    output = transform_stackexchange(input_content)

    expected_file = os.path.join(paths["expected"], f"{input_name}.md")
    output_file = os.path.join(paths["outputs"], f"{input_name}.md")

    os.makedirs(paths["outputs"], exist_ok=True)

    with open(output_file, "w") as f:
        f.write(output)
        f.write("\n")

    compare_outputs(input_name, expected_file, output_file, paths["diffs"])
