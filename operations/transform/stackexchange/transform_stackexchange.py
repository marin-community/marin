"""
stackexchange/transform_stackexchange.py

Performs HTML->Text/MD conversion using the specified tools over a stackexchange dump save in DOLMA format.
"""

import json
import logging
import os
import random
from dataclasses import dataclass

import draccus
import fsspec
import ray
from tqdm import tqdm

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

    template += f"\n\n---\nTags: {', '.join(tags)}\n---"

    return template


@ray.remote(memory=2 * 1024 * 1024 * 1024)
def process_file(
    input_file_path: str,
    output_path: str,
    extract_method: str,
    extract_config: ExtractionConfig,
    shuffle_answers_template: bool = True,
) -> None:
    output_file_path = os.path.join(output_path, input_file_path.split("/")[-1].replace(".ndjson", ".jsonl.gz"))

    logger.info(f"Starting processing of file {input_file_path}")
    logger.info(f"Source: {input_file_path}")
    logger.info(f"Destination: {output_file_path}")
    try:
        with (
            fsspec.open(input_file_path, compression="gzip") as source,
            fsspec.open(output_file_path, "wt", compression="gzip") as output,
        ):
            for line in tqdm(source, desc="Processing lines"):
                row = json.loads(line)

                try:
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

                    out_dict = {
                        "id": row["id"],
                        "url": url,
                        "title": title,
                        "date_created": row["created"],
                        "text": content,
                    }

                    if content is None:
                        continue

                    print(json.dumps(out_dict), file=output)  # Without this line, the JSON file will be corrupted
                except Exception as e:
                    logger.exception(f"Error processing line: {e}")
                    raise e

        logger.info("\nProcessing completed successfully!")
        logger.info(f"File available at: {output_path}")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


@draccus.wrap()
def process_stackexchange_dump(cfg: StackExchangeExtractionConfig) -> None:
    logger.info(f"Starting processing of StackExchange dump in {cfg.input_path}")

    files = fsspec_glob(f"{cfg.input_path}/*.jsonl.gz")

    # only keep file of the form <id>.json.gz and not <language>.<id>.json.gz
    files = [file for file in files if len(os.path.basename(file).split(".")) == 3]
    logger.info(f"Found {len(files)} files to process")

    result_refs = []
    MAX_CONCURRENT_WORKERS = 50

    if cfg.max_files:
        files = files[: cfg.max_files]

    for file in files:
        if len(result_refs) > MAX_CONCURRENT_WORKERS:
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            try:
                ray.get(ready_refs)
            except Exception as e:
                logger.exception(f"Error processing the group: {e}")
                continue

        result_refs.append(process_file.remote(file, cfg.output_path, cfg.extract_method, cfg.extract_config))
    try:
        ray.get(result_refs)
    except Exception as e:
        logger.exception(f"Error processing the group: {e}")
