"""
wikipedia/transform_wikipedia.py

Performs HTML->Text/MD conversion using the specified tools over a wiki dump save in DOLMA format.
"""

import json
import logging
import os
from dataclasses import dataclass

import draccus
import fsspec
import ray
from tqdm_loggable.auto import tqdm

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


@ray.remote(memory=2 * 1024 * 1024 * 1024)
def process_file(input_file_path: str, output_path: str, extract_method: str, extract_config: ExtractionConfig) -> None:
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
                    result = convert_page(
                        row["article_body"]["html"], extract_method=extract_method, config=extract_config
                    )
                    out_dict = {
                        "id": row["identifier"],
                        "url": row["url"],
                        "title": row["name"],
                        "abstract": row.get("abstract", ""),
                        "date_created": row["date_created"] if "date_created" in row else row.get("date_modified", ""),
                        "text": result["content"],
                    }

                    print(json.dumps(out_dict), file=output)  # Without this line, the JSON file will be corrupted
                except Exception as e:
                    logger.info(f"Keys in row: {row.keys()}")
                    logger.info(f"Article body keys: {row['article_body'].keys()}")

                    logger.exception(f"Error processing line: {e}")
                    continue

        logger.info("\nProcessing completed successfully!")
        logger.info(f"File available at: {output_path}")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


@draccus.wrap()
def process_wiki_dump(cfg: WikiExtractionConfig) -> None:
    logger.info(f"Starting processing of Wikipedia dump in {cfg.input_path}")

    files = fsspec_glob(f"{cfg.input_path}/*.ndjson")
    logger.info(f"Found {len(files)} files to process")

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

        output_path = os.path.join(cfg.output_path, cfg.revision)
        result_refs.append(process_file.remote(file, output_path, cfg.extract_method, cfg.extract_config))
    try:
        ray.get(result_refs)
    except Exception as e:
        logger.exception(f"Error processing the group: {e}")
