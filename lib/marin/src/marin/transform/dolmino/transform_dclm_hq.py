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
marin/transform/dolmino/transform_dclm_hq.py

Performs HTML->Text/MD conversion using the specified tools over a DCLM HQ dump save in DOLMA format.
"""

import json
import logging
import os
from dataclasses import dataclass

import draccus
import fsspec
from tqdm import tqdm
from zephyr import Dataset, flow_backend

from marin.core.runtime import cached_or_construct_output
from marin.download.dclm_hq.download_dclm_hq_html import find_html_in_cc
from marin.download.huggingface.stream_remove_columns import hf_fs
from marin.schemas.web.convert import ExtractionConfig
from marin.web.convert import convert_page

logger = logging.getLogger(__name__)


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


@cached_or_construct_output(success_suffix="SUCCESS")
def process_file(
    input_file_path: str,
    output_file_path: str,
    extract_method: str,
    extract_config: ExtractionConfig,
) -> None:
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

                    content = convert_page(html_string, extract_method=extract_method, config=extract_config)["content"]

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
        logger.info(f"File available at: {output_file_path}")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


@draccus.wrap()
def process_dclm_hq_dump(cfg: DCLMHQExtractionConfig) -> None:
    logger.info(f"Starting processing of DCLM HQ dump in {cfg.input_hf_path}")

    backend = flow_backend()

    # Glob all files across all shards upfront
    all_files = []
    paths = [i.split("/")[-1] for i in hf_fs.ls(cfg.input_hf_path, detail=False)]
    paths = paths[: cfg.max_split] if cfg.max_split else paths

    logger.info(f"Found {len(paths)} shards to process")

    for path in paths:
        input_path = os.path.join(cfg.input_hf_path, path)
        shard_paths = [i.split("/")[-1] for i in hf_fs.glob(os.path.join(input_path, "*.json.zst"))]

        for shard_path in shard_paths:
            input_file_path = os.path.join(input_path, shard_path)
            output_file_path = os.path.join(cfg.output_path, path, shard_path).replace(".json.zst", ".jsonl.gz")
            all_files.append(
                {
                    "input": input_file_path,
                    "output": output_file_path,
                    "extract_method": cfg.extract_method,
                    "extract_config": cfg.extract_config,
                }
            )

    logger.info(f"Total files to process: {len(all_files)}")

    # Single-level parallelism over all files
    pipeline = Dataset.from_list(all_files).map(
        lambda f: process_file(f["input"], f["output"], f["extract_method"], f["extract_config"])
    )

    list(backend.execute(pipeline))
