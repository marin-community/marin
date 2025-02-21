"""
wikipedia/transform_wikipedia.py

Performs HTML->Text/MD conversion using the specified tools over a wiki dump save in DOLMA format.
"""

import json
import logging
import os
import re
from dataclasses import dataclass

import draccus
import fsspec
import ray
from bs4 import BeautifulSoup
from tqdm import tqdm

from marin.schemas.web.convert import ExtractionConfig
from marin.utils import fsspec_glob
from marin.web.convert import convert_page
from operations.download.huggingface.stream_remove_columns import hf_fs


logger = logging.getLogger("ray")


@dataclass
class DCLMHQExtractionConfig:
    input_hf_path: str
    output_path: str
    extract_method: str
    extract_config: ExtractionConfig
    hf_repo_id: str
    hf_revision: str
    hf_paths: list[str]
    max_split: int | None = None


def find_html_in_cc(record_id: str, target_uri: str) -> str | None:
    """
    Find HTML code for the given record ID and target URI in the Common Crawl dataset.
    """
    pass


@ray.remote(memory=2 * 1024 * 1024 * 1024)
def process_file(
    input_file_path: str,
    output_path: str,
    extract_method: str,
    extract_config: ExtractionConfig,
) -> None:
    output_file_path = os.path.join(output_path, input_file_path.split("/")[-1].replace(".ndjson", ".jsonl.gz"))

    logger.info(f"Starting processing of file {input_file_path}")
    logger.info(f"Source: {input_file_path}")
    logger.info(f"Destination: {output_file_path}")
    try:
        with (
            fsspec.open(input_file_path, compression="zst") as source,
            fsspec.open(output_file_path, "wt", compression="gzip") as output,
        ):
            for line in tqdm(source, desc="Processing lines"):
                row = json.loads(line)

                try:
                    html_string = find_html_in_cc(row["metadata"]["WARC-Record-ID"], row["metadata"]["WARC-Target-URI"])
                    
                    if html_string is None:
                        logger.error(f"No HTML found for record ID: {row['metadata']['WARC-Record-ID']}")
                        continue

                    content = convert_page(
                        html_string, extract_method=extract_method, config=extract_config
                    )["content"]

                    if content is None:
                        continue

                    out_dict = {
                        "id": row["id"],
                        "source": row["source"],
                        "metadata": row["metadata"],
                        "text": content,
                    }

                    print(json.dumps(out_dict), file=output)  # Without this line, the JSON file will be corrupted
                except Exception as e:
                    logger.exception(f"Error processing line: {e}")
                    continue

        logger.info("\nProcessing completed successfully!")
        logger.info(f"File available at: {output_path}")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


@ray.remote(memory=2 * 1024 * 1024 * 1024)
def process_dclm_shard(
    input_path: str,
    output_path: str,
    extract_method: str,
    extract_config: ExtractionConfig,
) -> None:
    logger.info(f"Processing DCLM shard {input_path}")
    logger.info(f"Output path: {output_path}")

    result_refs = []
    MAX_CONCURRENT_WORKERS = 16

    shard_paths = [i.split("/")[-1] for i in hf_fs.glob(os.path.join(input_path, "*.json.zst"))]

    for shard_path in shard_paths:
        if len(result_refs) > MAX_CONCURRENT_WORKERS:
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            try:
                ray.get(ready_refs)
            except Exception as e:
                logger.exception(f"Error processing the group: {e}")
                continue
        
        input_file_path = os.path.join(input_path, shard_path)
        output_file_path = os.path.join(output_path, shard_path).replace(".json.zst", ".jsonl.gz")
        result_refs.append(
            process_file.remote(input_file_path, output_file_path, extract_method, extract_config)
        )

    try:
        ray.get(result_refs)
    except Exception as e:
        logger.exception(f"Error processing the group: {e}")


@draccus.wrap()
def process_dclm_hq_dump(cfg: DCLMHQExtractionConfig) -> None:
    logger.info(f"Starting processing of DCLM HQ dump in {cfg.input_hf_path}")

    result_refs = []
    MAX_CONCURRENT_WORKERS = 50

    paths = [i.split("/")[-1] for i in hf_fs.ls(cfg.input_hf_path, detail=False)]
    paths = paths[:cfg.max_split] if cfg.max_split else paths

    for path in paths:
        if len(result_refs) > MAX_CONCURRENT_WORKERS:
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            try:
                ray.get(ready_refs)
            except Exception as e:
                logger.exception(f"Error processing the group: {e}")
                continue
        
        input_path = os.path.join(cfg.input_hf_path, path)
        output_path = os.path.join(cfg.output_path, path)
        result_refs.append(
            process_dclm_shard.remote(
                input_path, output_path, cfg.extract_method, cfg.extract_config
            )
        )
    try:
        ray.get(result_refs)
    except Exception as e:
        logger.exception(f"Error processing the group: {e}")
