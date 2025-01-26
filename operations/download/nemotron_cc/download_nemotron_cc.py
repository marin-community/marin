import json
import logging
import os
from dataclasses import dataclass

import draccus
import fsspec
import ray
import requests
from tqdm import tqdm

from marin.core.runtime import cached_or_construct_output
from marin.utils import fsspec_exists

logger = logging.getLogger("ray")

myagent = "marin-nemotron-ingress/1.0 (marin@marin.com)"
NCC_PATH_FILE_URL = "https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/data-jsonl.paths.gz"


@cached_or_construct_output(success_suffix="SUCCESS")
@ray.remote(memory=10 * 1024 * 1024 * 1024, runtime_env={"pip": ["gcsfs", "zstandard"]}, max_retries=5)  # 10 GB
def download_single_nemotron_path(input_file_path: str, output_file_path: str):
    """
    Fetches content from a Common Crawl path.
    Args:
        input_file_path (str): Path to the Common Crawl file
        output_file_path (str): Path to the output file
    Returns:
        list: List of content from the JSONL records
    """
    contents = []

    cc_url = f"https://data.commoncrawl.org/{input_file_path}"
    logger.info(f"Downloading Nemotron CC file {cc_url} to {output_file_path}")

    try:
        response = requests.get(cc_url, headers={"user-agent": myagent}, stream=True)

        if response.status_code == 200:
            import zstandard as zstd

            dctx = zstd.ZstdDecompressor()

            # Read chunk by chunk so we can split on newlines ourselves
            with dctx.stream_reader(response.raw) as reader:
                chunk_size = 65536
                buffer = b""

                # Get total size for progress bar
                total_size = int(response.headers.get("content-length", 0))
                progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True, desc="Downloading")
                bytes_read = 0

                while True:
                    chunk = reader.read(chunk_size)
                    if not chunk:
                        break
                    buffer += chunk
                    bytes_read += len(chunk)
                    progress_bar.update(len(chunk))

                    while True:
                        newline_pos = buffer.find(b"\n")
                        if newline_pos < 0:
                            break
                        line_bytes = buffer[:newline_pos]
                        buffer = buffer[newline_pos + 1 :]

                        if not line_bytes.strip():
                            continue

                        try:
                            content = json.loads(line_bytes.decode("utf-8"))
                            contents.append(content)
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse JSON line: {line_bytes[:80]}...")
                            continue

                progress_bar.close()
        else:
            logger.error(f"Failed to fetch data: {response.status_code}")
            return None

    except Exception as e:
        logger.exception(f"Error fetching content from {cc_url}: {e}")
        return None

    if not contents:
        logger.warning("No valid JSONL records found")
        return None

    with fsspec.open(output_file_path, "w", compression="gzip") as f:
        for content in contents:
            dolma_format = {
                "id": content["warc_record_id"],
                "text": content["text"],
                "source": "nemotron",
                "format": "text",
                "metadata": {
                    f"nemotron_{key}": value for key, value in content.items() if key not in ("warc_record_id", "text")
                },
            }
            print(json.dumps(dolma_format), file=f)


@dataclass
class NemotronIngressConfig:
    output_path: str


@draccus.wrap()
def download_nemotron_cc(cfg: NemotronIngressConfig):
    files = []

    paths_file_path = os.path.join(cfg.output_path, "data-jsonl.paths")
    logger.info(f"Downloading Nemotron CC path file {paths_file_path}")

    if fsspec_exists(paths_file_path):
        logger.warning(f"Paths file {paths_file_path} already exists. Skipping download.")
    else:
        with fsspec.open(NCC_PATH_FILE_URL, "rb") as f, fsspec.open(paths_file_path, "wb") as f_out:
            f_out.write(f.read())

    logger.info(f"Reading paths from {paths_file_path}")
    with fsspec.open(paths_file_path, "r", compression="gzip") as f:
        for line in f:
            files.append(line.strip())

    MAX_NUM_PENDING_TASKS = 5000

    result_refs = []
    for file in files:
        if len(result_refs) > MAX_NUM_PENDING_TASKS:
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            try:
                ray.get(ready_refs)
            except Exception as e:
                logger.exception(f"Error processing the group: {e}")
                continue

        file_name = os.path.basename(file)
        cc_split = file_name.split("-part-")[0]
        part_path = file_name.split("-part-")[1]

        output_file_path = os.path.join(cfg.output_path, cc_split, part_path).replace("jsonl.zstd", "jsonl.gz")
        logger.info(f"Starting Processing for the Nemotron CC file: {file} in output_path: {cfg.output_path}")

        result_refs.append(download_single_nemotron_path.remote(file, output_file_path))

    try:
        ray.get(result_refs)
    except Exception as e:
        raise Exception(f"Error processing the group: {e}")  # noqa


if __name__ == "__main__":
    download_nemotron_cc(NemotronIngressConfig(output_path="gs://marin-us-central2/raw/nemotron_cc"))
