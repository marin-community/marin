import json
import os
import random

from tqdm import tqdm

from marin.schemas.web.convert import ExtractionConfig, HtmlToMarkdownConfig, ResiliparseConfig
from operations.transform.stackexchange.transform_stackexchange import prepare_md_template


def prepare_expected_output(row: dict, extract_method: str, extract_config: ExtractionConfig, shuffle_answers_template: bool = True) -> str:
    prepend_vote_count = random.random() < 0.5 if shuffle_answers_template else False

    title = row["metadata"]["title"] if "title" in row["metadata"] else row["title"]
    question = row["metadata"]["question"] if "question" in row["metadata"] else row["question"]
    answers = row["metadata"]["answers"]
    tags = row["metadata"]["tags"] if "tags" in row["metadata"] else row["tags"]

    content = prepare_md_template(title, question, answers, tags, extract_method, extract_config, prepend_vote_count)

    return content

if __name__ == "__main__":
    input_path = "tests/snapshots/stackexchange/inputs"
    expected_path = "tests/snapshots/stackexchange/expected"

    if not os.path.exists(expected_path):
        os.makedirs(expected_path)

    input_files = os.listdir(input_path)

    for input_file in tqdm(input_files):
        with open(os.path.join(input_path, input_file), "r") as f:
            json_data = json.load(f)

            extract_method = "resiliparse"
            extract_config = ResiliparseConfig(
                preserve_formatting=True,
                main_content=True,
                links=False,
                prepend_title=True,
                skip_elements=[],
                use_custom_variant=True,
                markdownify_config=HtmlToMarkdownConfig(
                    include_images=False,
                    include_links=False,
                )
            )
            expected = prepare_expected_output(json_data, extract_method, extract_config)

            with open(os.path.join(expected_path, input_file.replace("json", "md")), "w") as f:
                print(expected, file=f)
