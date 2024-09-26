# Example script on how to perform the Transform jobs on ray and process large amount of data.
# The scripts contains comments with critical information to follow along.
# Anything which is not inside ray.remote function will be executed on the head node, so keep it to minimum.
# path: scripts/hello_world_fw/process.py
# Inputs: jsonl.gz files in dolma format having html content, Output: jsonl.gz files in dolma format having markdown
# Example Usage:
# On Cluster (After starting ray dashboard):
# ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- \
# python scripts/hello_world_fw/process.py \
# --input_path gs://marin-us-central2/raw/hello_world_fw/v1.0/CC-MAIN-2024-10/000_00000
# --output_path gs://marin-us-central2/documents/hello_world_fw/v1.0/quickstart/CC-MAIN-2024-10/000_00000
# On Local:
# python scripts/hello_world_fw/process.py \
# --input_path gs://marin-us-central2/raw/hello_world_fw/v1.0/CC-MAIN-2024-10/000_00000
# --output_path gs://marin-us-central2/documents/hello_world_fw/v1.0/quickstart/CC-MAIN-2024-10/000_00000
import json
from dataclasses import dataclass
import logging

import draccus
import fsspec
import ray

from marin.core.runtime import cached_or_construct_output, map_files_in_directory
from marin.schemas.web.convert import TrafilaturaConfig
from marin.web.convert import convert_page

logger = logging.getLogger("ray")

# TODO: move into general utiltiies
# This function will be executed on the worker nodes. It is important to keep the function idempotent and resumable.
# default memory is unbound, default runtime_env is empty, default num_cpus is 1
# IMPORTANT: Ray resources are logical and not physical: https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
# Ray will not impose any physical limits on the resources used by the function, these numbers are used for scheduling.
@ray.remote(memory=1 * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs", "trafilatura"]}, num_cpus=1)  # 1 GB
@cached_or_construct_output(success_suffix="SUCCESS")  # We use this decorator to make this function idempotent
def html_to_md(input_file_path: str, output_file_path: str, extract_method: str, config):
    # The runtime for this function should be low (less than 5-10 min), as the machines are preemptible
    # Example of input_path = gs://marin-data/hello_world_fw/fineweb/fw-v1.0/CC-MAIN-2024-10/000_00000/0_processed_html.jsonl.gz

    # Read the input file
    with (
        fsspec.open(input_file_path, "rt", compression="gzip") as f,
        fsspec.open(output_file_path, "wt", compression="gzip") as output,
    ):
        num_lines = 0
        for line in f:
            data = json.loads(line)
            num_lines += 1

            id = data["id"]
            html = data["text"]
            source = data["source"]
            fw_metadata = data["metadata"]["fineweb_metadata"]
            url = fw_metadata["url"]

            # Convert page can throw exception based on the html content (e.g. invalid html, Empty page)
            try:
                logger.info(f"Converting line {num_lines}: {id} {url}")
                md = convert_page(html, url, extract_method, config)["content"]
                error = None
            except Exception as e:
                # Failed to convert
                logger.exception(f"{e} in processing {id = }, {url = }, {input_file_path = }")
                md = None
                error = e

            record = {
                "id": id,
                "source": source,
                "format": "md",
                "metadata": {key: value for key, value in fw_metadata.items()},
            }
            if md:
                record["text"] = md
            if error:
                record["error"] = str(error)
            print(json.dumps(record), file=output)

    return True


@dataclass(frozen=True)
class FineWebConfig:
    input_path: str
    output_path: str
    extract_method: str = "readability"
    config: str | TrafilaturaConfig = "default"


@ray.remote
def transform(cfg: FineWebConfig):
    refs = map_files_in_directory(html_to_md, cfg.input_path, "**/*.jsonl.gz", cfg.output_path, extract_method=cfg.extract_method, config=cfg.config)

    # Wait for all the tasks to finish.
    # The try and catch is important here as in case html_to_md throws any exception, that exception is passed here,
    # And if we don't catch it here, the script will exit, which will kill all the other tasks.
    try:
        ray.get(list(refs))
    except Exception as e:
        logger.exception(e)
        # Put your retry logic here, incase you want to get the file for which the processing failed, please see:
        # https://docs.ray.io/en/latest/ray-core/fault-tolerance.html
        # In practice, since we make html_to_md resumable and idempotent, you can just look at the logs in Ray dashboard
        # And retry the same command after fixing the bug.


@draccus.wrap()
def main(cfg: FineWebConfig):
    ray.get(transform.remote(cfg))


if __name__ == "__main__":
    main()
