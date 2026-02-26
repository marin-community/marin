"""Count total data points across all parquet files in a GCS path."""

import gcsfs
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed

GCS_PATH = "marin-vlm-eu/stage2_sharded_full"


def count_rows(fs, path):
    meta = pq.read_metadata(fs.open(path))
    return meta.num_rows


def main():
    fs = gcsfs.GCSFileSystem()
    files = fs.glob(f"{GCS_PATH}/*.parquet")
    print(f"Found {len(files)} parquet files")

    total = 0
    done = 0
    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = {pool.submit(count_rows, fs, f): f for f in files}
        for future in as_completed(futures):
            total += future.result()
            done += 1
            if done % 1000 == 0:
                print(f"  Processed {done}/{len(files)} files, running total: {total:,}")

    print(f"\nTotal data points: {total:,}")


if __name__ == "__main__":
    main()
