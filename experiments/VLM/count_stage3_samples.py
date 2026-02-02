import fsspec
import pyarrow.parquet as pq
from tqdm import tqdm


def count_parquet_samples(gcs_pattern: str) -> int:
    """Count total samples in parquet files matching the pattern."""
    # Get filesystem and list files
    fs, path = fsspec.core.url_to_fs(gcs_pattern)
    files = sorted(fs.glob(path))

    print(f"Found {len(files)} parquet files")

    total_samples = 0
    for file_path in tqdm(files, desc="Counting samples"):
        with fs.open(file_path, "rb") as f:
            pf = pq.ParquetFile(f)
            total_samples += pf.metadata.num_rows

    return total_samples


if __name__ == "__main__":
    pattern = "gs://marin-vlm/stage3_sharded_full/*.parquet"
    total = count_parquet_samples(pattern)
    print(f"Total samples: {total:,}")
