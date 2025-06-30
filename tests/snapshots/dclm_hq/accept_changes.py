import json
import os
import re

from tqdm import tqdm

from marin.schemas.web.convert import HtmlToMarkdownConfig, ResiliparseConfig
from marin.transform.dolmino.transform_dclm_hq import find_html_in_cc
from marin.web.convert import convert_page


def save_html(json_data, input_path):
    input_files = os.listdir(input_path)

    for input_file in tqdm(input_files):
        with open(os.path.join(input_path, input_file), "r") as f:
            json_data = json.load(f)

        match = re.search(r"isPartOf:\s*(CC-MAIN-\d{4}-\d{2})", json_data["metadata"]["warcinfo"])
        is_part_of = match.group(1)

        html = find_html_in_cc(is_part_of, json_data["metadata"]["WARC-Target-URI"])

        json_data["html"] = html

        if "text" in json_data:
            json_data.pop("text")

        with open(input_file.replace(".json", ".html"), "w") as f:
            print(html, file=f)


if __name__ == "__main__":
    input_path = "tests/snapshots/dclm_hq/inputs"
    expected_path = "tests/snapshots/dclm_hq/expected"

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
                skip_elements=[],
                use_custom_variant=True,
                markdownify_config=HtmlToMarkdownConfig(
                    include_images=False,
                    include_links=False,
                ),
            )
            expected = convert_page(html, extract_method=extract_method, config=extract_config)["content"]

            with open(os.path.join(expected_path, input_file.replace("html", "md")), "w") as f:
                print(expected, file=f)
