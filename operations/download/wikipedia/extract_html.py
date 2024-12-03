"""
wikipedia/extract_html.py

Script for extracting HTML content from Wikipedia dumps in DOLMA format.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import draccus
import fsspec
import ray
from tqdm import tqdm

from marin.schemas.web.convert import ExtractionConfig
from marin.utils import fsspec_glob
from marin.web.convert import convert_page

logger = logging.getLogger("ray")


@dataclass
class WikiExtractionConfig:
    input_path: str
    output_path: str
    revision: str
    extract_method: str
    extract_config: ExtractionConfig


@ray.remote
def process_html(html_text: str, extract_method: str, extract_config: ExtractionConfig) -> dict[str, str]:
    return convert_page(html_text, extract_method=extract_method, extract_config=extract_config)


@ray.remote(memory=2 * 1024 * 1024 * 1024)
def process_file(input_file_path: str, output_path: str) -> None:
    output_file_path = os.path.join(output_path, input_file_path.replace(".ndjson.gz", ".jsonl.gz"))

    logger.info("Starting processing of file {input_file_path}")
    logger.info(f"Source: {input_file_path}")
    logger.info(f"Destination: {output_file_path}")

    try:
        with (
            fsspec.open(input_file_path, compression="gzip") as source,
            fsspec.open(output_file_path, "wt", compression="gzip") as output,
        ):
            MAX_PENDING_TASKS = 25

            pending_tasks = []
            for line in tqdm(source, desc="Processing lines"):
                pending_tasks.append(process_html.remote(line))

                if len(pending_tasks) > MAX_PENDING_TASKS:
                    ready_tasks, pending_tasks = ray.wait(pending_tasks, num_returns=1)
                    try:
                        result = ray.get(ready_tasks)
                        print(result, file=output)
                    except Exception as e:
                        print(f"Error processing line: {e}")
                        continue

            try:
                results = ray.get(pending_tasks)
                for result in results:
                    print(result, file=output)
            except Exception as e:
                print(f"Error processing remaining tasks: {e}")
                raise

        print("\nProcessing completed successfully!")
        print(f"File available at: {output_path}")

    except Exception as e:
        print(f"Error during processing: {e}")
        raise


@draccus.wrap()
def process_wiki_dump(cfg: WikiExtractionConfig) -> None:
    files = fsspec_glob(f"{cfg.input_path}/*.ndjson.gz")

    result_refs = []
    MAX_CONCURRENT_WORKERS = 15

    for file in files:
        if len(result_refs) > MAX_CONCURRENT_WORKERS:
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            try:
                ray.get(ready_refs)
            except Exception as e:
                logger.exception(f"Error processing the group: {e}")
                continue

        output_file = Path(cfg.output_path) / file.name.replace(".ndjson.gz", ".txt")
        result_refs.append(process_file.remote(file, output_file))
    try:
        ray.get(result_refs)
    except Exception as e:
        logger.exception(f"Error processing the group: {e}")
