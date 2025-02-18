import json
import os
import re
from dataclasses import dataclass

import draccus
import fsspec
import ray

from marin.core.runtime import cached_or_construct_output, map_files_in_directory

SCORE_OPTIONS_DICT = {
    "Great": 5,
    "Good": 4,
    "Okay": 3,
    "Poor": 2,
    "Useless": 1,
}


@dataclass
class ConvertClassifierDatasetConfig:
    input_path: str
    output_path: str

    # Ray data outputs it into json even if it's actually a jsonl file, we will rewrite it to jsonl.gz
    # to follow dolma format
    input_filetype: str = "json"
    output_filetype: str = "jsonl.gz"


def convert_filextension_to_output_filetype(output_path: str, desired_filetype: str):
    # Get the base path without extension
    base_path = os.path.splitext(output_path)[0]
    if os.path.splitext(output_path)[1] == ".gz":
        # Handle double extensions like .jsonl.gz
        base_path = os.path.splitext(base_path)[0]

    # Map common input extensions to desired output
    if desired_filetype == "jsonl.gz":
        return f"{base_path}.jsonl.gz"
    elif desired_filetype == "jsonl":
        return f"{base_path}.jsonl"
    elif desired_filetype == "json":
        return f"{base_path}.json"
    elif desired_filetype == "parquet":
        return f"{base_path}.parquet"
    else:
        return f"{base_path}.{desired_filetype}"


@ray.remote
@cached_or_construct_output(success_suffix="SUCCESS")
def process_file(input_file_path: str, output_file_path: str, output_filetype: str):
    output_file_path = convert_filextension_to_output_filetype(output_file_path, output_filetype)
    with fsspec.open(input_file_path, "r") as input_file:
        with fsspec.open(output_file_path, "w", compression="gzip") as output_file:
            for line in input_file:
                example = json.loads(line)
                prompt = example["prompt"]

                # Extract text between triple quotes using regex
                pattern = r"'''(.*?)'''"
                match = re.search(pattern, prompt, re.DOTALL)
                if match:
                    extracted_text = match.group(1).strip()
                else:
                    extracted_text = ""

                generated_text = example["generated_text"]
                if generated_text in SCORE_OPTIONS_DICT:
                    score = SCORE_OPTIONS_DICT[generated_text]
                else:
                    score = -1

                if score != -1 and extracted_text != "":
                    output_file.write(json.dumps({"text": extracted_text, "label": score}) + "\n")


@draccus.wrap()
def convert_classifier_dataset_func(config: ConvertClassifierDatasetConfig):
    responses = map_files_in_directory(
        process_file.remote,
        config.input_path,
        f"**/*.{config.input_filetype}",
        config.output_path,
        output_filetype=config.output_filetype,
    )

    ray.get(responses)


if __name__ == "__main__":
    convert_classifier_dataset_func()
