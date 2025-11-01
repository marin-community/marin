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

import json
import os
import tempfile
import zipfile
from dataclasses import dataclass

import draccus
import fsspec
import ray

from marin.utils import fsspec_mkdirs


@dataclass
class ConvertLingolyToDolmaConfig:
    input_path: str
    """The directory containing the zip file of the lingoly dataset."""
    output_path: str
    """The directory to save the lingoly dataset in dolma format."""


@ray.remote
def _convert_lingoly_to_dolma(config: ConvertLingolyToDolmaConfig) -> None:
    """Convert the lingoly dataset to dolma format by concatenating the preamble, context, and questions."""
    # Create a temporary directory to extract the zip file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Handle zip file from GCS or other cloud storage
        local_zip_path = os.path.join(temp_dir, "lingoly.zip")

        # Download the zip file if it's in cloud storage
        fs = fsspec.url_to_fs(config.input_path)[0]
        fs.get_file(config.input_path, local_zip_path)

        # Extract the zip file
        with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find and open the test.jsonl file
        test_crawl_file_path = os.path.join(temp_dir, "test.jsonl")

        if not os.path.exists(test_crawl_file_path):
            raise FileNotFoundError(f"test.jsonl not found in the extracted zip file at {test_crawl_file_path}")

        output_path = os.path.join(config.output_path, "lingoly_preamble_context_questions_joined")
        fsspec_mkdirs(output_path)

        shard_number = 0
        max_doc_length = 7000
        output_doc_file_path = os.path.join(output_path, f"{shard_number:05d}.jsonl")

        with open(test_crawl_file_path, "r") as f:
            # Process each line as JSON

            text_so_far = ""
            for line in f:
                data = json.loads(line)
                preamble = data["preamble"]
                context = data["context"]

                # Convert questions from string to list of dictionaries if it's a string
                questions = data["questions"]
                if isinstance(questions, str):
                    questions = json.loads(questions)

                for question in questions:
                    question_prompt = question["prompt"]
                    subprompt_question_joined = ""
                    for subprompt in question["subprompts"]:
                        subprompt_question_joined += f"{subprompt['question']}\n"

                final_text = f"{preamble}\n{context}\n{question_prompt}\n{subprompt_question_joined}"

                text_so_far += f"{final_text}\n"

                if len(text_so_far) > max_doc_length:
                    with fsspec.open(output_doc_file_path, "a") as f_out:
                        f_out.write(json.dumps({"text": text_so_far}) + "\n")
                    text_so_far = ""
                    shard_number += 1
                    output_doc_file_path = os.path.join(output_path, f"{shard_number:05d}.jsonl")

        if len(text_so_far) > 0:
            with fsspec.open(output_doc_file_path, "a") as f_out:
                f_out.write(json.dumps({"text": text_so_far}) + "\n")


@draccus.wrap()
def convert_lingoly_to_dolma(config: ConvertLingolyToDolmaConfig) -> None:
    ray.get(_convert_lingoly_to_dolma.remote(config))
