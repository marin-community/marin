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
transform/fineweb/process_parquet_fw.py

Process FineWeb parquet files to extract HTML/markdown content from referenced WARC files.

Example Usage (production, large dataset):
uv run zephyr --backend=ray --max-parallelism=10 --entry-point=process_fw_dump \
    lib/marin/src/marin/transform/fineweb/process_parquet_fw.py \
    --input_path gs://marin-us-central2/raw/fineweb-edu \
    --md_output_path gs://marin-us-central2/documents/fineweb-edu-md \
    --text_output_path gs://marin-us-central2/documents/fineweb-edu-text \
    --cc_dumps '["CC-MAIN-2013-20"]' \
    --extract_method readability \
    --max_files 10

Example Usage (local testing, small dataset):
# Process 1 parquet file from smallest CC dump (CC-MAIN-2013-20)
uv run zephyr --backend=threadpool --max-parallelism=2 --entry-point=process_fw_dump \
    lib/marin/src/marin/transform/fineweb/process_parquet_fw.py \
    --input_path gs://marin-us-central2/raw/fineweb-edu \
    --md_output_path /tmp/fineweb_test/md \
    --cc_dumps '["CC-MAIN-2013-20"]' \
    --extract_method readability \
    --max_files 1

Note: Each parquet file is ~2GB and processes multiple WARC files. CC-MAIN-2013-20 is the
      smallest dump with 14 parquet files. Use --max_files 1 for quick local testing.
"""

import json
import logging
import os
import traceback
from dataclasses import dataclass, field
from datetime import datetime

import draccus
import fsspec
import pandas as pd
from marin.schemas.web.convert import ExtractionConfig, HtmlToMarkdownConfig, ResiliparseConfig
from marin.utils import fsspec_glob, fsspec_rm
from marin.web.convert import convert_page
from warcio import ArchiveIterator
from zephyr import Backend, Dataset
from zephyr.writers import atomic_rename

logger = logging.getLogger(__name__)


def extract_cc_dump(path: str) -> str:
    """Extract CC-MAIN-* identifier from path, or parent directory as fallback."""
    parts = path.rstrip("/").split("/")
    for part in parts:
        if part.startswith("CC-MAIN-"):
            return part
    return parts[-2] if len(parts) >= 2 else ""


def process_one_warc_file(
    df: pd.DataFrame,
    input_path: str,
    output_path: str,
    extract_method: str,
    config: ExtractionConfig,
    md_output_path: str,
    text_output_path: str | None = None,
    html_output_path: str | None = None,
):
    """
    Takes in the input file and processes it to get the html and md content.
    It scans the s3 bucket in input_file and returns the content of the urls in the input_file
    Args:
        df: Dataframe containing the fineweb records for a single WARC file
        input_path (str): The original parquet file path
        output_path (str): The output file to write the processed content
        extract_method (str): The method to use for extracting the content
        config (ExtractionConfig): The config to use for the extraction
        md_output_path (str): The output path to write the md content
        text_output_path (str): The output path to write the text content
        html_output_path (str): The output path to write the html content
    """
    # input_path = gs://marin-us-central2/processed/000_00000/0.parquet
    # output_path = gs://marin-us-central2/processed/000_00000/0_processed.jsonl.gz

    # We use different output files for md and fineweb, we do not store the html content
    base_folder = os.path.dirname(output_path).split("/")[-1]
    output_file_name = os.path.basename(output_path)

    md_output_file = os.path.join(md_output_path, base_folder, output_file_name)

    text_output_file = None
    if text_output_path:
        text_output_file = os.path.join(text_output_path, base_folder, output_file_name)

    html_output_file = None
    if html_output_path:
        html_output_file = os.path.join(html_output_path, base_folder, output_file_name)

    # Write the output to a file with md information.
    logger.info(f"Writing to {md_output_file = }, {text_output_file = }")

    df["md"] = None

    urls = df["url"].tolist()
    # All frames will have same file_path be design
    s3_url = df["file_path"].iloc[0]
    index = df.index.tolist()

    # url_dict is url to index in df so that we can update that record
    url_dict = {url: idx for idx, url in zip(index, urls, strict=False)}
    num_urls_found = 0  # Used to early terminate
    num_urls_processed = 0
    length_url_inp_list = len(urls)

    length_warc = 0

    # Detect filesystem protocol from URL
    if s3_url.startswith("s3://"):
        fs = fsspec.filesystem("s3", anon=False)  # make sure s3 keys are setup
    elif s3_url.startswith("file://"):
        fs = fsspec.filesystem("file")
    else:
        # Default to S3 for backward compatibility (URLs without protocol prefix)
        fs = fsspec.filesystem("s3", anon=False)

    with fs.open(s3_url, mode="rb") as file_stream:
        for record in ArchiveIterator(file_stream):
            if num_urls_found == length_url_inp_list:
                break

            # Check if it's a response record
            if record.rec_type == "response":
                # Process the record
                url = record.rec_headers.get_header("WARC-Target-URI")
                length_warc += 1

                if url in url_dict:
                    num_urls_found += 1
                    url_idx_in_df = url_dict[url]
                    if num_urls_found % 100 == 0:
                        logger.info(
                            f"Found Url {num_urls_found = }, Processed Url {num_urls_processed = }, "
                            f"length of warc {length_warc = }"
                        )

                    try:
                        content = record.content_stream().read()
                        html_decoded = content.decode(errors="ignore")

                        markdown = convert_page(html_decoded, url, extract_method, config)
                        df.loc[url_idx_in_df, "md"] = markdown["content"]
                        df.loc[url_idx_in_df, "html"] = html_decoded

                        num_urls_processed += 1
                    except Exception as e:
                        # We are just ignoring the error and moving forward as these errors are generally not a lot
                        logger.exception(f"Error processing {url} in {s3_url} for {input_path}: {e}")
                        traceback.print_exc()

    logger.info(
        f"Processed {input_path}, found {length_warc} records, {length_url_inp_list} urls, "
        f"{length_warc / length_url_inp_list} ratio"
    )

    with atomic_rename(md_output_file) as temp_path:
        with fsspec.open(temp_path, "wt", compression="gzip") as f:  # md output
            for _index, row in df.iterrows():
                out_fw = row.to_dict()
                out_dolma = {
                    "id": out_fw["id"],
                    "text": out_fw["md"] if out_fw["md"] else "",
                    "source": "fineweb",
                    "format": "md",
                    "metadata": {
                        f"fw_{key}": value for key, value in out_fw.items() if key not in ("md", "html", "text")
                    },
                }

                print(json.dumps(out_dolma), file=f)

    if text_output_file and "text" in df.columns:
        with atomic_rename(text_output_file) as temp_path:
            with fsspec.open(temp_path, "wt", compression="gzip") as f:  # text output
                for _index, row in df.iterrows():
                    out_fw = row.to_dict()
                    out_dolma = {
                        "id": out_fw["id"],
                        "text": out_fw["text"],
                        "source": "fineweb",
                        "format": "text",
                        "metadata": {
                            f"fw_{key}": value for key, value in out_fw.items() if key not in ("md", "html", "text")
                        },
                    }
                    print(json.dumps(out_dolma), file=f)

    if html_output_file:
        with atomic_rename(html_output_file) as temp_path:
            with fsspec.open(temp_path, "wt", compression="gzip") as f:  # html output
                for _index, row in df.iterrows():
                    out_fw = row.to_dict()
                    out_dolma = {
                        "id": out_fw["id"],
                        "source": "fineweb",
                        "format": "html",
                        "html": out_fw["html"],
                    }
                    print(json.dumps(out_dolma), file=f)

    # remove the input file
    fsspec_rm(input_path)

    # num_urls_found should be equal to length_url_inp_list
    logger.info(
        f"Found: {num_urls_found}, Processed: {num_urls_processed}, out of {length_url_inp_list} urls, "
        f"in {input_path}"
        f"AWS URL: {s3_url}"
        f"Found {length_warc} records in the WARC file"
    )

    return True


def process_fw_parquet(
    input_path: str,
    output_path: str,
    extract_method: str,
    config: ExtractionConfig,
    md_output_path: str,
    text_output_path: str | None = None,
    html_output_path: str | None = None,
) -> dict:
    """
    Converts fineweb files to html and markdown.

    This extracts each individual WARC file from the fineweb parquet file,
    processes it to extract html and markdown content, and writes the output
    to the specified output paths.

    Parameters:
    input_path (str): Path to the fineweb parquet file
    output_path (str): Path to the output folder where we will write the processed files
    """

    # Example of input_path = gs://marin-us-central2/raw/fineweb/fw-v1.0/CC-MAIN-2024-10/000_00000.parquet
    # Example of output_path = gs://marin-us-central2/processed/000_00000
    logger.info(f"Processing {input_path}")
    datetime_start = datetime.utcnow()

    try:
        df = pd.read_parquet(input_path)
    except FileNotFoundError as e:
        logger.exception(f"Error reading the parquet file: {e}")
        raise e

    # Process each group sequentially within this parquet file
    # file_path is s3 url
    grouped = df.groupby("file_path")

    was_successful = True

    for index, (_file_url, group_df) in enumerate(grouped):
        output_file_name = os.path.join(output_path, f"{index}_processed.jsonl.gz")
        # output_file_name = gs://marin-us-central2/processed/000_00000/0_processed.jsonl.gz

        logger.info(f"Processing the group: {index}, into {output_file_name}")

        # Note: custom resiliparse variant handling removed - ensure correct package is installed in environment
        if isinstance(config, ResiliparseConfig) and config.use_custom_variant:
            logger.warning(
                "Custom resiliparse variant requested - ensure resiliparse_dom is installed from "
                "git+https://github.com/nelson-liu/chatnoir-resiliparse@58247de82b4d881223435113f1a07a86ad66494c"
            )

        try:
            process_one_warc_file(
                group_df,
                input_path,
                output_file_name,
                extract_method,
                config,
                md_output_path,
                text_output_path,
                html_output_path,
            )
        except Exception as e:
            logger.exception(f"Error processing {input_path} - index={index}, Error: {e}")
            was_successful = False

    datetime_end = datetime.utcnow()

    return {
        "input_path": input_path,
        "output_file_path": output_path,
        "datetime_start": str(datetime_start),
        "datetime_end": str(datetime_end),
        "success": was_successful,
    }


@dataclass
class ParquetFWConfig:
    input_path: str
    md_output_path: str
    text_output_path: str | None = None
    html_output_path: str | None = None
    cc_dumps: list[str] | None = None
    extract_method: str = "readability"
    config: ExtractionConfig = field(default_factory=HtmlToMarkdownConfig.default_config)
    max_files: int | None = None


@draccus.wrap()
def process_fw_dump(cfg: ParquetFWConfig):
    # Glob all parquet files across all CC dumps upfront
    cc_dumps = cfg.cc_dumps or fsspec_glob(f"{cfg.input_path}/*")

    all_files = []
    for cc_dump in cc_dumps:
        files = fsspec_glob(os.path.join(cfg.input_path, cc_dump, "*.parquet"))

        if not files:
            logger.info(f"No files found in {cc_dump}, Skipping")
            continue

        for file in files:
            # Extract cc_dump identifier from file path
            cc_dump_id = extract_cc_dump(file)
            input_file_name = os.path.basename(file)

            # Derive output paths
            output_path = os.path.join(
                cfg.md_output_path,
                input_file_name.replace(".parquet", ""),
            )
            md_output_path = os.path.join(cfg.md_output_path, cc_dump_id)
            text_output_path = None
            if cfg.text_output_path:
                text_output_path = os.path.join(cfg.text_output_path, cc_dump_id)
            html_output_path = None
            if cfg.html_output_path:
                html_output_path = os.path.join(cfg.html_output_path, cc_dump_id)

            all_files.append(
                {
                    "input_path": file,
                    "output_path": output_path,
                    "extract_method": cfg.extract_method,
                    "config": cfg.config,
                    "md_output_path": md_output_path,
                    "text_output_path": text_output_path,
                    "html_output_path": html_output_path,
                }
            )

            if cfg.max_files and len(all_files) >= cfg.max_files:
                break

        if cfg.max_files and len(all_files) >= cfg.max_files:
            break

    logger.info(f"Total parquet files to process: {len(all_files)}")

    pipeline = (
        Dataset.from_list(all_files)
        .map(
            lambda f: process_fw_parquet(
                f["input_path"],
                f["output_path"],
                f["extract_method"],
                f["config"],
                f["md_output_path"],
                f["text_output_path"],
                f["html_output_path"],
            )
        )
        .write_jsonl(f"{cfg.md_output_path}/.metrics/process-{{shard:05d}}.jsonl", skip_existing=True)
    )

    Backend.execute(pipeline)


if __name__ == "__main__":
    process_fw_dump()
