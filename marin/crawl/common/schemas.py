from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ParquetConfig:
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