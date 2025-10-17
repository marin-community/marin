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


def get_id_from_row(row: dict, id_path: tuple[str, ...]) -> str | None:
    """Traverse a tuple path in a row to extract the ID, or return None if missing."""
    obj = row
    for key in id_path:
        obj = obj.get(key)
        if obj is None:
            raise ValueError(f"ID path {id_path} not found in row: {row}")
    return obj


def has_id_column(row: dict, id_path: tuple[str, ...]) -> bool:
    """Check whether the tuple path exists in the row."""
    try:
        get_id_from_row(row, id_path)
        return True
    except ValueError:
        return False


def get_finished_ids(output_filename: str, id_path: tuple[str, ...]) -> set:
    """Get the set of IDs that have already been processed in the output file

    Args:
        output_filename: Path to the output file
        id_path: Tuple path of keys leading to the ID (e.g., ("metadata", "id"))

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
                        id_value = get_id_from_row(row, id_path)
                        if id_value is not None:
                            finished_ids.add(id_value)
                    except json.JSONDecodeError:
                        continue
        elif output_filename.endswith(".parquet"):
            # Read parquet and extract IDs
            import pyarrow.parquet as pq

            with fsspec.open(output_filename, "rb") as f:
                table = pq.read_table(f, columns=[id_path[0]])
                for row in table.to_pylist():
                    id_value = get_id_from_row(row, id_path)
                    if id_value is not None:
                        finished_ids.add(id_value)

        return finished_ids
    except (FileNotFoundError, Exception) as e:
        print(f"[!] Error reading finished IDs from {output_filename}: {e}")
        return finished_ids
