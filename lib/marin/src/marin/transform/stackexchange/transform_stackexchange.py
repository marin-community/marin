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
stackexchange/transform_stackexchange.py

Performs HTML->Text/MD conversion using the specified tools over a stackexchange dump save in DOLMA format.

Example Usage:
uv run zephyr --backend=ray --max-parallelism=50 --memory=2GB --cluster=us-central2 \
    lib/marin/src/marin/transform/stackexchange/transform_stackexchange.py \
    --input_path gs://path/to/input --output_path gs://path/to/output ...
"""

import logging
import os
import random
from dataclasses import dataclass

import draccus
from zephyr import Backend, Dataset, load_jsonl

from marin.schemas.web.convert import ExtractionConfig
from marin.utils import fsspec_glob
from marin.web.convert import convert_page

logger = logging.getLogger("ray")


@dataclass
class StackExchangeExtractionConfig:
    input_path: str
    output_path: str
    extract_method: str
    extract_config: ExtractionConfig
    max_files: int | None = None
    shuffle_answers_template: bool = True
    seed: int | None = None


def prepare_md_template(
    title: str,
    question: str,
    answers: list[dict],
    tags: list[str],
    extract_method: str,
    extract_config: ExtractionConfig,
    prepend_vote_count: bool = True,
) -> str:
    """
    Prepares a markdown template for a stackexchange question and answer.
    """

    md_question = convert_page(question, extract_method=extract_method, config=extract_config)["content"]
    template = f"# Question\nTitle: {title}\n{md_question}"

    for answer in answers:
        md_answer = convert_page(answer["body"], extract_method=extract_method, config=extract_config)["content"]
        if prepend_vote_count:
            template += f"\n\n# Answer\n{md_answer}\n> {answer['votes']} votes"
        else:
            template += f"\n\n# Answer\n> {answer['votes']} votes\n{md_answer}"

    if len(tags) > 0:
        template += f"\n\n---\nTags: {', '.join(tags)}\n---"

    return template


def process_record(
    row: dict,
    extract_method: str,
    extract_config: ExtractionConfig,
    shuffle_answers_template: bool = True,
    seed: int | None = None,
):
    """Process a single StackExchange record and return transformed record.

    Args:
        row: Record from JSONL file
        extract_method: Method to use for HTML extraction
        extract_config: Configuration for the extraction method
        shuffle_answers_template: Whether to shuffle answer template format
        seed: Random seed for reproducibility

    Returns:
        Transformed record in Dolma format, or None if record should be skipped
    """
    try:
        if seed is not None:
            random.seed(seed + hash(row["id"]))
        prepend_vote_count = random.random() < 0.5 if shuffle_answers_template else False

        title = row["metadata"]["title"] if "title" in row["metadata"] else row["title"]
        question = row["metadata"]["question"] if "question" in row["metadata"] else row["question"]
        answers = row["metadata"]["answers"]
        tags = row["metadata"]["tags"] if "tags" in row["metadata"] else row["tags"]
        url = row["metadata"]["url"] if "url" in row["metadata"] else row["url"]

        content = prepare_md_template(
            title,
            question,
            answers,
            tags,
            extract_method,
            extract_config,
            prepend_vote_count,
        )

        if content is None:
            return None

        out_dict = {
            "id": row["id"],
            "url": url,
            "title": title,
            "date_created": row["created"],
            "text": content,
        }

        return out_dict
    except Exception as e:
        logger.exception(f"Error processing line: {e}")
        raise e


@draccus.wrap()
def process_stackexchange_dump(cfg: StackExchangeExtractionConfig) -> None:
    logger.info(f"Starting processing of StackExchange dump in {cfg.input_path}")

    files = fsspec_glob(f"{cfg.input_path}/*.jsonl.gz")

    # only keep file of the form <id>.json.gz and not <language>.<id>.json.gz
    files = [file for file in files if len(os.path.basename(file).split(".")) == 3]
    logger.info(f"Found {len(files)} files to process")

    if cfg.max_files:
        files = files[: cfg.max_files]

    pipeline = (
        Dataset.from_list(files)
        .flat_map(load_jsonl)
        .map(
            lambda row: process_record(
                row,
                cfg.extract_method,
                cfg.extract_config,
                cfg.shuffle_answers_template,
                cfg.seed,
            )
        )
        .filter(lambda record: record is not None)
        .write_jsonl(f"{cfg.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz", skip_existing=True)
    )
    Backend.execute(pipeline)
