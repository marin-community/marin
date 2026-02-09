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

"""
Convert Lingoly dataset from zip to Dolma format.

Processes zip file containing test.jsonl, concatenates preamble/context/questions,
and chunks into ~7000 character text blocks.

Example Usage:
uv run zephyr --backend=sync \
    lib/marin/src/marin/transform/lingoly/to_dolma.py \
    --input_path gs://path/to/lingoly.zip \
    --output_path gs://path/to/output/lingoly_preamble_context_questions_joined
"""

import json
from dataclasses import dataclass

import draccus
from zephyr import Dataset, ZephyrContext, load_zip_members


@dataclass
class ConvertLingolyToDolmaConfig:
    input_path: str
    """Path to the zip file of the lingoly dataset."""
    output_path: str
    """Directory to save the lingoly dataset in dolma format."""
    max_doc_length: int = 7000
    """Maximum length of text chunks before starting a new document."""


def process_lingoly_member(member: dict, max_doc_length: int = 7000):
    """Process lingoly zip member, yielding chunked text blocks.

    Args:
        member: Dict with 'filename' and 'content' (bytes) from load_zip_members
        max_doc_length: Maximum character length before yielding a chunk

    Yields:
        Dicts with 'text' field containing concatenated preamble/context/questions
    """
    # Parse JSONL from bytes
    text_so_far = ""
    for line in member["content"].decode("utf-8").splitlines():
        if not line.strip():
            continue

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
                yield {"text": text_so_far}
                text_so_far = ""

    if text_so_far:
        yield {"text": text_so_far}


def convert_lingoly_to_dolma(config: ConvertLingolyToDolmaConfig) -> None:
    """Convert Lingoly dataset to Dolma format."""
    pipeline = (
        Dataset.from_list([config.input_path])
        .flat_map(lambda p: load_zip_members(p, pattern="test.jsonl"))
        .flat_map(lambda m: process_lingoly_member(m, max_doc_length=config.max_doc_length))
        .write_jsonl(f"{config.output_path}/{{shard:05d}}.jsonl")
    )
    with ZephyrContext(name="lingoly-to-dolma") as ctx:
        list(ctx.execute(pipeline))


@draccus.wrap()
def main(config: ConvertLingolyToDolmaConfig) -> None:
    """CLI entrypoint."""
    convert_lingoly_to_dolma(config)
