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
This scripts performs a simple html to md conversion using marin. Given an input directory with some jsonl.gz files
containing html content, it will convert them to markdown and save them in a new directory.

Example Usage:
uv run zephyr --backend=ray --max-parallelism=1000 --memory=1GB --num-cpus=1 --cluster=us-central2 \
    lib/marin/src/marin/transform/simple_html_to_md/process.py \
    --input_path gs://... --output_path gs://...
"""

import logging
from dataclasses import dataclass, field

import draccus
from marin.schemas.web.convert import ExtractionConfig, HtmlToMarkdownConfig
from zephyr import Dataset, flow_backend, load_jsonl

logger = logging.getLogger("ray")


def _html_to_md(data: dict, extract_method: str, config: ExtractionConfig):
    """Convert a single HTML record to markdown.

    Args:
        data: Record from JSONL file
        extract_method: Method to use for HTML extraction
        config: Configuration for the extraction method

    Returns:
        Transformed record with markdown content
    """
    from marin.web.convert import convert_page

    data_id = data["id"]
    html = data["text"]
    source = data["source"]

    # Since the input jsonl.gz files were extracted from fineweb, we have fineweb_metadata in the metadata.
    fw_metadata = data["metadata"]["fineweb_metadata"]
    url = fw_metadata["url"]

    # Convert page can throw exception based on the html content (e.g. invalid html, Empty page)
    try:
        logger.debug(f"Converting {data_id} {url}")
        md = convert_page(html, url, extract_method, config)["content"]
        error = None
    except Exception as e:
        # Failed to convert
        logger.exception(f"{e} in processing {data_id = }, {url = }")
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

    return record


@dataclass(frozen=True)
class SimpleHtmlToMdConfig:
    input_path: str  # Input directory containing jsonl.gz files
    output_path: str  # Output directory containing md files
    extract_method: str = "readability"
    # Extract method to use. Defaults to readability. Other options are traflatura and resiliparse.
    config: ExtractionConfig = field(default_factory=HtmlToMarkdownConfig.default_config())
    # Configuration for the extraction method.


@draccus.wrap()
def html_to_md(cfg: SimpleHtmlToMdConfig):
    """Transform HTML content to markdown using the specified extraction method."""
    backend = flow_backend()
    pipeline = (
        Dataset.from_files(cfg.input_path, "**/*.jsonl.gz")
        .flat_map(load_jsonl)
        .map(lambda data: _html_to_md(data, cfg.extract_method, cfg.config))
        .write_jsonl(f"{cfg.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )
    list(backend.execute(pipeline))
