# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Download and process Ar5iv dataset from a zip file.

"""

import json
import logging
import zipfile
from collections import defaultdict
from dataclasses import dataclass

import draccus
from rigging.filesystem import open_url
from rigging.log_setup import configure_logging
from zephyr import Dataset, ZephyrContext
from zephyr.writers import atomic_rename

from marin.execution.step_spec import StepSpec

logger = logging.getLogger(__name__)


@dataclass
class Ar5ivDownloadConfig:
    input_path: str
    output_path: str
    max_files: int | None = None  # Maximum number of shards to process


def process_shard(shard_task: dict) -> dict:
    """
    Process a single shard by extracting its files from the zip in GCS and uploading the merged JSONL.

    Args:
        shard_task: Dict with keys 'input_path', 'output_path', 'shard_id', 'file_list'
    """
    input_path = shard_task["input_path"]
    output_path = shard_task["output_path"]
    shard_id = shard_task["shard_id"]
    file_list = shard_task["file_list"]
    gcs_path = f"{output_path}/{shard_id}.jsonl.gz"

    with open_url(str(input_path), "rb") as f:
        with zipfile.ZipFile(f) as zf:
            with atomic_rename(gcs_path) as temp_path, open_url(temp_path, "wt", compression="gzip") as out_f:
                for filename in file_list:
                    with zf.open(filename, "r") as file_handle:
                        content = file_handle.read()
                        record = {
                            "filename": filename,
                            "format": "html",
                            "content": content.decode("utf-8", errors="replace"),
                        }
                        print(json.dumps(record), file=out_f)

            logger.info(f"Shard {shard_id} with {len(file_list)} files uploaded to {gcs_path}")
            return {"shard_id": shard_id, "num_files": len(file_list), "output_path": gcs_path}


def download(cfg: Ar5ivDownloadConfig) -> None:
    """
    Download and process Ar5iv dataset from a zip file in GCS.

    This function can be called by the executor framework or used standalone.
    """
    logger.info("Starting transfer of Ar5iv dataset...")
    logger.info(f"Source: {cfg.input_path}")

    # Use fsspec+zipfile to list all files
    with open_url(str(cfg.input_path), "rb") as f:
        with zipfile.ZipFile(f) as zf:
            all_files = zf.infolist()

            # Group by shard directory
            # We assume structure: something like: shard_id/.../file
            # shard_id is derived from the second last component if files are nested.
            # Adjust as needed if directory structure differs.
            shard_dict = defaultdict(list)
            for info in all_files:
                if info.is_dir():
                    continue
                # E.g. path might look like: "003/something.html"
                # Extract shard_id from the directory:
                # Split by "/" and take the first part if we assume structure {shard_id}/file
                parts = info.filename.strip("/").split("/")
                if len(parts) < 2:
                    # File at root level - decide how to handle this case.
                    # If no directory structure is given, skip or treat differently.
                    continue
                shard_id = parts[-2]  # get the second-last directory as shard_id
                shard_dict[shard_id].append(info.filename)

            # Apply max_files limit if provided
            shard_ids = list(shard_dict.keys())
            if cfg.max_files is not None:
                shard_ids = shard_ids[: cfg.max_files]

            logger.info(f"Found {len(shard_ids)} shards to process.")

            # Build task list for each shard
            shard_tasks = []
            for shard_id in shard_ids:
                shard_tasks.append(
                    {
                        "input_path": cfg.input_path,
                        "output_path": cfg.output_path,
                        "shard_id": shard_id,
                        "file_list": shard_dict[shard_id],
                    }
                )

    # Execute pipeline with zephyr
    pipeline = (
        Dataset.from_list(shard_tasks)
        .map(process_shard)
        .write_jsonl(f"{cfg.output_path}/.metrics/part-{{shard:05d}}.jsonl", skip_existing=True)
    )
    ctx = ZephyrContext(name="download-ar5iv")
    ctx.execute(pipeline)

    logger.info("Transfer completed successfully!")


def ar5iv_step(
    name: str = "raw/ar5iv",
    *,
    input_path: str,
    max_files: int | None = None,
    deps: list[StepSpec] | None = None,
    output_path_prefix: str | None = None,
    override_output_path: str | None = None,
) -> StepSpec:
    """Create a StepSpec that downloads and processes the Ar5iv dataset from a zip file."""

    def _run(output_path: str) -> None:
        download(Ar5ivDownloadConfig(input_path=input_path, output_path=output_path, max_files=max_files))

    return StepSpec(
        name=name,
        fn=_run,
        deps=deps or [],
        hash_attrs={"input_path": input_path, "max_files": max_files},
        output_path_prefix=output_path_prefix,
        override_output_path=override_output_path,
    )


@draccus.wrap()
def main(cfg: Ar5ivDownloadConfig) -> None:
    """CLI entrypoint for downloading and processing Ar5iv dataset."""

    configure_logging(level=logging.INFO)
    download(cfg)
