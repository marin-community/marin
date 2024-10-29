import sys
import json
import ray
import draccus
import fsspec

from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Dict, Any


def validate_json_record(record: Any) -> bool:
    """
    Validate that the record is a dictionary.
    
    Args:
        record: The parsed JSON record to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    return isinstance(record, dict) and "text" in record


def process_jsonl_file(file_path: str) -> Iterator[Dict]:
    """
    Process a single jsonl.gz file, yielding valid dictionary records.
    
    Args:
        file_path: Path to the jsonl.gz file (GCS path)
        
    Yields:
        Dict: Each valid JSON record
    """
    fs = fsspec.filesystem('gcs')
    with fs.open(file_path, 'rt', compression='gzip', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                if not validate_json_record(record):
                    print(f"Warning: Line {line_num} in {file_path} is not a dictionary", file=sys.stderr)
                    continue
                yield record
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num} in {file_path}: {e}", file=sys.stderr)

@ray.remote
def process_jsonl_files(directory: str, pattern: str = "*.jsonl.gz"):
    """
    Iterate over and process all jsonl.gz files in a GCS directory.
    
    Args:
        directory: GCS directory path containing the jsonl.gz files
        pattern: Glob pattern to match files (default: *.jsonl.gz)
    """
    fs = fsspec.filesystem('gcs')
    if not fs.exists(directory):
        print(f"Error: Directory {directory} does not exist", file=sys.stderr)
        sys.exit(1)
        
    files = fs.glob(f"{directory}/{pattern}")
    if not files:
        print(f"Warning: No files matching pattern {pattern} found in {directory}", file=sys.stderr)
        return
    
    for file_path in files:
        process_jsonl_file(file_path)

@dataclass
class Args:
    directory: str

@draccus.wrap()
def main(args: Args):
    """
    Iterate over and process all jsonl.gz files in a GCS directory.
    
    Args:
        directory: GCS directory path containing the jsonl.gz files
        pattern: Glob pattern to match files (default: *.jsonl.gz)
    """
    fs = fsspec.filesystem('gcs')
    if not fs.exists(args.directory):
        print(f"Error: args.Directory {args.directory} does not exist", file=sys.stderr)
        sys.exit(1)
        
    dirs = fs.ls(args.directory)

    for dir in dirs:
        ray.get(process_jsonl_files.remote(dir))



if __name__ == "__main__":
    main()
