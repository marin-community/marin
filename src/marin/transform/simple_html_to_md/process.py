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
This scripts performs a simple html to md conversion using marin. Gien an input directory with some jsonl.gz files
containing html content, it will convert them to markdown and save them in a new directory.
"""

import json
import logging
from dataclasses import dataclass, field

import draccus
import fsspec
import ray

from marin.core.runtime import cached_or_construct_output, map_files_in_directory
from marin.schemas.web.convert import ExtractionConfig, HtmlToMarkdownConfig

logger = logging.getLogger("ray")


@ray.remote(memory=1 * 1024 * 1024 * 1024, num_cpus=1)  # 1 GB
@cached_or_construct_output(success_suffix="SUCCESS")  # We use this decorator to make this function idempotent
def html_to_md(input_file_path: str, output_file_path: str, extract_method: str, config: ExtractionConfig) -> bool:
    from marin.web.convert import convert_page

    # Read the input file
    with (
        fsspec.open(input_file_path, "rt", compression="gzip") as f,
        fsspec.open(output_file_path, "wt", compression="gzip") as output,
    ):
        num_lines = 0
        for line in f:
            data = json.loads(line)
            num_lines += 1

            data_id = data["id"]
            html = data["text"]
            source = data["source"]

            # Since the input jsonl.gz files were extracted from fineweb, we have fineweb_metadata in the metadata.
            fw_metadata = data["metadata"]["fineweb_metadata"]
            url = fw_metadata["url"]

            # Convert page can throw exception based on the html content (e.g. invalid html, Empty page)
            try:
                logger.debug(f"Converting line {num_lines}: {data_id} {url}")
                md = convert_page(html, url, extract_method, config)["content"]
                error = None
            except Exception as e:
                # Failed to convert
                logger.exception(f"{e} in processing {data_id = }, {url = }, {input_file_path = }")
                md = None
                error = e

            record = {
                "id": data_id,
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
class SimpleHtmlToMdConfig:
    input_path: str  # Input directory containing jsonl.gz files
    output_path: str  # Output directory containing md files
    extract_method: str = "readability"
    # Extract method to use. Defaults to readability. Other options are traflatura and resiliparse.
    config: ExtractionConfig = field(default_factory=HtmlToMarkdownConfig.default_config())
    # Configuration for the extraction method.


@ray.remote
def transform(cfg: SimpleHtmlToMdConfig):
    """
    Transforms all the jsonl.gz files in the input directory having html content to markdown using the specified method.
    """

    # Map the files in the directory to the html_to_md function.
    refs = map_files_in_directory(
        html_to_md,
        cfg.input_path,
        "**/*.jsonl.gz",
        cfg.output_path,
        extract_method=cfg.extract_method,
        config=cfg.config,
    )

    # Wait for all the tasks to finish.
    try:
        ray.get(list(refs))
    except Exception as e:
        logger.exception(e)


@draccus.wrap()
def main(cfg: SimpleHtmlToMdConfig):
    ray.get(transform.remote(cfg))


if __name__ == "__main__":
    main()
