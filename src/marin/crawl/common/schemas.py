from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HtmlExtractionConfig:
    """
    Configuration for extracting HTML for the shards from parquet files.

    input_path: Path to the directory containing parquet files.
    output_path: Path to the directory to write the HTML files to.
    source_name: Name of the source of the HTML.
    columns: List of columns to extract from the parquet files.
    s3_url_modifier: Function to modify the S3 URL of the extracted HTML.
    warc_path_extractor: Function to set file_path if not provided in the parquet files columns.
    url_column: Column name for the URL in the parquet files.
    max_files: Maximum number of files to process.
    file_path_column: Column name for the file path in the parquet files.
    """

    input_path: str
    output_path: str
    source_name: str
    columns: list[str]

    s3_url_modifier: Callable[[str], str] | None = None
    warc_path_extractor: Callable[[dict[str, Any]], str] | None = None

    url_column: str = "url"
    max_files: int | None = None
    file_path_column: str = "file_path"


@dataclass(frozen=True)
class DolmaFormattedRecord:
    id: str
    source: str
    format: str
    html: str
    metadata: dict[str, Any]
