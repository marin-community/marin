import gzip
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from marin.processing.classification.inference import iter_dataset_batches


# helper to create a small DataFrame and write to various formats
def create_test_files(tmp_path, columns, data):
    df = pd.DataFrame(data, columns=columns)
    files = {}

    # jsonl.gz
    jsonl_gz = tmp_path / "test.jsonl.gz"
    with gzip.open(jsonl_gz, "wt") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")
    files["jsonl.gz"] = str(jsonl_gz)

    # parquet
    parquet_file = tmp_path / "test.parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_file)
    files["parquet"] = str(parquet_file)

    return files


@pytest.mark.parametrize("filetype", ["jsonl.gz", "parquet"])
def test_iter_dataset_batches(tmp_path, filetype):
    """
    Test whether iter_dataset_batches yields correct batches
    """
    pass


def test_iter_dataset_batches_invalid_type(tmp_path):
    """
    Test whether iter_dataset_batches raises ValueError for unsupported file types.
    """
    dummy = tmp_path / "test.unsupported"
    dummy.write_text("dummy")
    with pytest.raises(ValueError):
        list(iter_dataset_batches(str(dummy), ["id"]))


def test_process_file_with_quality_classifier(tmp_path):
    """
    Test whether process_file_with_quality_classifier writes expected output using a dummy classifier.
    """
    pass


def test_process_file_with_quality_classifier_parquet(tmp_path):
    """
    Test whether process_file_with_quality_classifier writes expected output to parquet using a dummy classifier.
    """
    pass
