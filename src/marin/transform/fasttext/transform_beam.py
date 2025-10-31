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
transform_beam.py

Apache Beam version of FastText to Dolma conversion using ml-flow.

This script converts FastText formatted files to Dolma JSONL format using
Apache Beam/Dataflow for scalable, cost-effective processing.

Usage:
    python transform_beam.py \
        --input_path gs://bucket/input.fasttext \
        --output_path gs://bucket/output/ \
        --source SOURCE_NAME \
        --gcp_project my-project \
        --runner DataflowRunner
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import draccus
from ml_flow import DataflowOptions, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fasttext-to-dolma-beam")


def generate_id(line: str, line_number: int) -> str:
    """
    Generate a unique ID for a line based on its content and line number.

    Args:
        line: The content of the line.
        line_number: The line number in the file.

    Returns:
        A SHA-256 hash hexdigest representing the unique ID.
    """
    unique_string = f"{line_number}:{line}"
    hash_object = hashlib.sha256(unique_string.encode("utf-8"))
    return hash_object.hexdigest()


def parse_fasttext_line(line_with_index: tuple[int, str], source: str) -> dict | None:
    """
    Parse a FastText line and convert to Dolma format.

    Args:
        line_with_index: Tuple of (line_number, line_text)
        source: Source identifier for Dolma format

    Returns:
        Dolma-formatted document dict, or None if line is malformed
    """
    line_number, line = line_with_index

    # Skip empty lines
    if not line.strip():
        return None

    # Extract text (everything after first space)
    try:
        text = line[line.index(" ") + 1 :].rstrip("\n")
    except ValueError:
        # No space found - malformed line
        logger.warning(f"Skipping malformed line {line_number}: {line[:50]}...")
        return None

    # Create Dolma document
    return {
        "id": generate_id(line, line_number),
        "text": text,
        "source": source,
        "added": datetime.now(timezone.utc).isoformat(),
    }


@dataclass
class TransformFasttextToDolmaBeamConfig:
    """
    Configuration for transforming FastText to Dolma using Beam.

    Attributes:
        input_path: Path to input FastText file (supports gs://, s3://, local)
        output_path: Output directory for Dolma JSONL files
        source: Source identifier per Dolma format (e.g., "dclm-fasttext-data")
        gcp_project: GCP project ID for Dataflow
        gcp_region: GCP region for Dataflow (default: us-central1)
        runner: Beam runner ("DirectRunner" for local, "DataflowRunner" for GCP)
        use_flex_rs: Use FlexRS for 40-60% cost savings (default: True)
        max_num_workers: Maximum number of Dataflow workers (default: 100)
    """

    input_path: str
    output_path: str
    source: str
    gcp_project: str
    gcp_region: str = "us-central1"
    runner: str = "DataflowRunner"
    use_flex_rs: bool = True
    max_num_workers: int = 100


def transform_fasttext_to_dolma_beam(cfg: TransformFasttextToDolmaBeamConfig):
    """
    Main transformation function using ml-flow Dataset API.

    Reads FastText file, parses lines, and writes Dolma-formatted JSONL.

    Args:
        cfg: Configuration object with input/output paths and Dataflow settings
    """
    logger.info("Starting FastText to Dolma transformation with Apache Beam")
    logger.info(f"Input: {cfg.input_path}")
    logger.info(f"Output: {cfg.output_path}")
    logger.info(f"Source: {cfg.source}")
    logger.info(f"Runner: {cfg.runner}")

    # Configure Dataflow options
    options = DataflowOptions(
        project=cfg.gcp_project,
        region=cfg.gcp_region,
        runner=cfg.runner,
        use_flex_rs=cfg.use_flex_rs,
        max_num_workers=cfg.max_num_workers,
    )

    # Read text file
    ds = Dataset.from_text_files(cfg.input_path, pipeline_options=options.to_pipeline_options())

    # Add line numbers (enumerate)
    ds = ds.map(lambda line: (hash(line) % 1000000, line), name="AddLineNumbers")

    # Parse FastText lines to Dolma format
    def parse_with_source(line_tuple):
        return parse_fasttext_line(line_tuple, cfg.source)

    ds = ds.map(parse_with_source, name="ParseFastTextLines").filter(lambda x: x is not None, name="FilterMalformed")

    # Handle output path - if it's a directory or ends with .jsonl.gz, adjust
    output_prefix = cfg.output_path
    if output_prefix.endswith(".jsonl.gz"):
        output_prefix = output_prefix[: -len(".jsonl.gz")]
    elif not output_prefix.endswith("/"):
        # Ensure directory has trailing slash for Beam
        output_prefix = output_prefix + "/output"
    else:
        output_prefix = output_prefix + "output"

    # Write Dolma JSONL
    ds.write_jsonl_gz(output_prefix)

    # Execute pipeline
    logger.info("Launching Beam pipeline...")
    result = ds.run_and_wait()

    # Check result
    state = result.state if isinstance(result.state, str) else result.state.name
    logger.info(f"Pipeline completed with state: {state}")

    if state == "DONE":
        logger.info("FastText to Dolma transformation completed successfully!")
    else:
        logger.error(f"Pipeline finished with unexpected state: {state}")


@draccus.wrap()
def main(cfg: TransformFasttextToDolmaBeamConfig):
    """CLI entry point using draccus."""
    transform_fasttext_to_dolma_beam(cfg)


if __name__ == "__main__":
    main()
