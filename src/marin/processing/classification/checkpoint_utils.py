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


def get_id_from_row(row: dict, id_column: str | dict[str, str]) -> str:
    """Get the ID from a row

    Args:
        row: The data row
        id_column: Either a string column name, or a dict with nested column access
                   e.g., {"metadata": "id"} means row["metadata"]["id"]

    Returns:
        The ID value from the row
    """
    if isinstance(id_column, dict):
        # Handle nested column access
        parent_key = next(iter(id_column.keys()))
        child_key = next(iter(id_column.values()))
        return row[parent_key][child_key]
    else:
        return row[id_column]


def has_id_column(row: dict, id_column: str | dict[str, str]) -> bool:
    """Check if a row has the required id column

    Args:
        row: The data row
        id_column: Either a string column name, or a dict with nested column access

    Returns:
        True if the id column exists in the row
    """
    if isinstance(id_column, dict):
        parent_key = next(iter(id_column.keys()))
        child_key = next(iter(id_column.values()))
        return parent_key in row and isinstance(row[parent_key], dict) and child_key in row[parent_key]
    else:
        return id_column in row


def get_finished_ids(output_filename: str, id_column: str | dict[str, str]) -> set:
    """Get the set of IDs that have already been processed in the output file

    Args:
        output_filename: Path to the output file
        id_column: Name of the column containing the ID

    Returns:
        Set of IDs that have already been processed
    """
    import json
    import fsspec

    finished_ids = set()

    try:
        # Check if file exists on local or remote filesystem
        fs, _ = fsspec.core.url_to_fs(output_filename)
        if not fs.exists(output_filename):
            return finished_ids

        if output_filename.endswith((".jsonl.gz", ".jsonl.zst", ".jsonl")):
            # Read JSON lines and extract IDs
            with fsspec.open(output_filename, "rt", compression="infer") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        if has_id_column(row, id_column):
                            finished_ids.add(get_id_from_row(row, id_column))
                    except json.JSONDecodeError:
                        continue
        elif output_filename.endswith(".parquet"):
            # Read parquet and extract IDs
            import pyarrow.parquet as pq

            with fsspec.open(output_filename, "rb") as f:
                if isinstance(id_column, dict):
                    # Handle nested column access
                    parent_key = next(iter(id_column.keys()))
                    child_key = next(iter(id_column.values()))
                    table = pq.read_table(f, columns=[parent_key])
                    # Extract child values from nested dicts
                    for row in table.to_pylist():
                        if parent_key in row and isinstance(row[parent_key], dict) and child_key in row[parent_key]:
                            finished_ids.add(row[parent_key][child_key])
                else:
                    # Simple column access
                    table = pq.read_table(f, columns=[id_column])
                    finished_ids = set(table[id_column].to_pylist())

        return finished_ids
    except (FileNotFoundError, Exception) as e:
        print(f"[!] Error reading finished IDs from {output_filename}: {e}")
        return finished_ids
