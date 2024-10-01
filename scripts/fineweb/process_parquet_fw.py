"""Convert fineweb to markdown"""

import os
import ray
import json
import fsspec
import draccus
import logging
import traceback
import pandas as pd

from datetime import datetime
from dataclasses import dataclass
from warcio import ArchiveIterator

from marin.web.convert import convert_page
from marin.schemas.web.convert import TrafilaturaConfig
from marin.core.runtime import cached_or_construct_output
from marin.utils import fsspec_exists, fsspec_glob, fsspec_rm


logger = logging.getLogger(__name__)


@ray.remote(memory=1.5 * 1024 * 1024 * 1024)  # 1.5 GB
@cached_or_construct_output(success_suffix="SUCCESS")  # We use this decorator to make this function idempotent
def process_one_warc_file(
    input_path: str,
    output_path: str,
    extract_method: str,
    config: str | TrafilaturaConfig,
    output_path_md: str,
    output_path_text: str
):
    """
    Takes in the input file and processes it to get the html and md content.
    It scans the s3 bucket in input_file and returns the content of the urls in the input_file
    Args:
    input_path (str): The input file to process
    """
    # input_path = gs://marin-data/processed/000_00000/0.parquet
    # output_path = gs://marin-data/processed/000_00000/0_processed.jsonl.gz

    # We use different output files for md and fineweb, we do not store the html content
    base_folder = os.path.dirname(output_path).split("/")[-1]
    output_file_name = os.path.basename(output_path)

    md_output_file = os.path.join(output_path_md, base_folder, output_file_name)
    fw_output_file = os.path.join(output_path_text, base_folder, output_file_name)


    # Write the output to a file with md information.
    logger.info(f"Writing to {md_output_file = }, {fw_output_file = }")

    try:
        df = pd.read_parquet(input_path)
    except FileNotFoundError as e:
        logger.exception(f"Error reading the parquet file: {e}")
        raise e

    df["md"] = None

    urls = df["url"].tolist()
    # All frames will have same file_path be design
    s3_url = df["file_path"].iloc[0]
    index = df.index.tolist()

    # url_dict is url to index in df so that we can update that record
    url_dict = {url: idx for idx, url in zip(index, urls)}
    num_urls_found = 0  # Used to early terminate
    num_urls_processed = 0
    length_url_inp_list = len(urls)

    # Logging variables
    logger.info(f"Processing {s3_url} in {input_path}")
    length_warc = 0
    s3_fs = fsspec.filesystem("s3", anon=False)  # make sure s3 keys are setup

    with s3_fs.open(s3_url, mode="rb") as file_stream:
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

                        num_urls_processed += 1
                    except Exception as e:
                        # We are just ignoring the error and moving forward as these errors are generally not a lot
                        logger.exception(f"Error processing {url} in {s3_url} for {input_path}: {e}")
                        traceback.print_exc()

    logger.info(
        f"Processed {input_path}, found {length_warc} records, {length_url_inp_list} urls, "
        f"{length_warc / length_url_inp_list} ratio"
    )

    with fsspec.open(md_output_file, "wt", compression="gzip") as f:  # md output
        for index, row in df.iterrows():
            out_fw = row.to_dict()
            out_dolma = {
                "id": out_fw["id"],
                "text": out_fw["md"],
                "source": "fineweb",
                "format": "md",
                "metadata": {f"fw_{key}": value for key, value in out_fw.items() if key not in ("md", "text")},
            }

            f.write(json.dumps(out_dolma) + "\n")

    with fsspec.open(fw_output_file, "wt", compression="gzip") as f:  # html output
        for index, row in df.iterrows():
            out_fw = row.to_dict()
            out_dolma = {
                "id": out_fw["id"],
                "text": out_fw["text"],
                "source": "fineweb",
                "format": "text",
                "metadata": {f"fw_{key}": value for key, value in out_fw.items() if key not in ("md", "html", "text")},
            }
            f.write(json.dumps(out_dolma) + "\n")

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


@ray.remote(memory=10 * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs"]})  # 10 GB
def process_fw_parquet(
    input_path: str,
    output_path: str,
    extract_method: str,
    config: str | TrafilaturaConfig,
    output_path_md: str,
    output_path_text: str
):
    """
    Converts fineweb files to html and markdown. This will essentially take in fineweb and split different groups based
    on file_path and write all those file paths to a new folder and then run ray for each group

    Parameters:
    input_path (str): Path to the fineweb parquet file
    output_path (str): Path to the output folder where we will write the processed files
    """

    # Example of input_path = gs://marin-data/raw/fineweb/fw-v1.0/CC-MAIN-2024-10/000_00000.parquet
    # Example of output_path = gs://marin-data/processed/000_00000
    logger.info(f"Processing {input_path}")
    success_file = output_path + "/_SUCCESS"
    datetime_start = datetime.utcnow()

    if fsspec_exists(success_file):
        logger.info(f"Output file already processed. Skipping {input_path}")
        return True

    try:
        df = pd.read_parquet(input_path)
    except FileNotFoundError as e:
        logger.exception(f"Error reading the parquet file: {e}")
        raise e

    ray_waitable = []
    file_path = []
    # file_path is s3 url
    grouped = df.groupby("file_path")

    for index, (file_url, group_df) in enumerate(grouped):
        filename = os.path.join(output_path, f"{index}.parquet")
        # filename = gs://marin-data/processed/000_00000/0.parquet

        output_file_name = filename.replace(".parquet", "_processed.jsonl.gz")
        # output_file_name = gs://marin-data/processed/000_00000/0_processed.jsonl.gz

        # Save the group to a parquet file
        group_df.to_parquet(filename)
        logger.info(f"Processing the group: {filename}, into {output_file_name}")

        ray_waitable.append(process_one_warc_file.remote(filename, output_file_name, extract_method, config, output_path_md, output_path_text))
        file_path.append(filename)

    was_successful = True

    for waitable, filename in zip(ray_waitable, file_path):
        try:
            ray.get(waitable)
        except Exception as e:
            logger.exception(f"Error processing {filename = }, Error: {e}")
            was_successful = False
        finally:
            # We should still remove the filename
            fsspec_rm(filename)

    datetime_end = datetime.utcnow()

    if not was_successful:
        return False

    with fsspec.open(success_file, "w") as f:
        metadata = {
            "input_path": input_path,
            "output_file_path": output_path,
            "datetime_start": str(datetime_start),
            "datetime_end": str(datetime_end),
        }
        f.write(json.dumps(metadata))

    return True


@dataclass
class ParquetFWConfig:
    input_path: str
    output_path_md: str
    output_path_text: str
    extract_method: str = "readability"
    config: str | TrafilaturaConfig = "default"
    max_files: int | None = None


@draccus.wrap()
def process_fw_dump(cfg: ParquetFWConfig):
    files = fsspec_glob(os.path.join(cfg.input_path, "*.parquet"))
    MAX_NUM_PENDING_TASKS = 15  # Max number of parquet files we want to process in pending state
    NUM_TASKS = len(files)

    result_refs = []
    for file in files:
        if len(result_refs) > MAX_NUM_PENDING_TASKS:
            # update result_refs to only
            # track the remaining tasks.
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            try:
                ray.get(ready_refs)
            except Exception as e:
                logger.exception(f"Error processing the group: {e}")
                continue

        # Get the input file name
        # Example of file = gs://marin-data/raw/fineweb/fw-v1.0/CC-MAIN-2024-10/000_00000.parquet
        # input_file_name = 000_00000.parquet
        input_file_name = os.path.basename(file)
        
        output_path = os.path.join(
            cfg.output_path_text,
            input_file_name.replace(".parquet", ""),
        ) # gs://marin-data/processed/000_00000

        logger.info(f"Starting Processing for the fw parquet file: {file} in output_path: {output_path}")
        result_refs.append(process_fw_parquet.remote(file, output_path, cfg.extract_method, cfg.config, cfg.output_path_md, cfg.output_path_text))

        if cfg.max_files and len(result_refs) >= cfg.max_files:
            break
    # Wait for all the tasks to finish
    try:
        ray.get(result_refs)
    except Exception as e:
        logger.exception(f"Error processing the group: {e}")


if __name__=="__main__":
    process_fw_dump()
