import argparse
import json
import pyarrow.parquet as pq
import gcsfs
from tqdm import tqdm
from datasets import load_dataset


def load_ms_ids_from_gcs(gcs_base_path: str, num_shards: int) -> set[str]:
    """Load ms_ids from parquet files in GCS."""
    fs = gcsfs.GCSFileSystem()
    all_ms_ids = []

    pbar = tqdm(range(num_shards), desc="Loading ms_ids from GCS")
    for i in pbar:
        path = f"{gcs_base_path}/data-{i:05d}-of-{num_shards:05d}.parquet"
        with fs.open(path) as f:
            table = pq.read_table(f, columns=['ms_id'])
            all_ms_ids.extend(table['ms_id'].to_pylist())
        pbar.set_postfix(samples=f"{len(all_ms_ids):,}")

    return set(all_ms_ids)


def filter_and_dedupe_dataset(dataset, ms_ids_to_keep: set[str]):
    """Filter dataset by ms_ids and deduplicate."""
    filtered_dataset = dataset.filter(
        lambda x: x['ms_id'] in ms_ids_to_keep,
        desc="Filtering by ms_id"
    )
    print(f"Filtered dataset size (before dedup): {len(filtered_dataset):,}")

    seen_ms_ids = set()
    def dedupe(example):
        if example['ms_id'] in seen_ms_ids:
            return False
        seen_ms_ids.add(example['ms_id'])
        return True

    filtered_dataset = filtered_dataset.filter(dedupe, desc="Deduplicating by ms_id")
    print(f"Filtered dataset size (after dedup): {len(filtered_dataset):,}")

    return filtered_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Extract a subset of a dataset based on ms_ids from GCS parquet files"
    )
    parser.add_argument(
        "--gcs_base_path",
        type=str,
        default="marin-us-central1/documents/open-thoughts-4-30k-math-qwen3-235b-a22b-fp8-tput-annotated-ff5c97",
        help="GCS path containing the parquet shards (without trailing slash)"
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=3000,
        help="Number of parquet shards"
    )
    parser.add_argument(
        "--source_dataset",
        type=str,
        default="marin-community/open-thoughts-4-math-qwen3-32b-annotated",
        help="Source HuggingFace dataset to filter"
    )
    parser.add_argument(
        "--target_dataset",
        type=str,
        default="marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated",
        help="Target HuggingFace dataset to upload to"
    )
    parser.add_argument(
        "--save_ms_ids",
        type=str,
        default=None,
        help="Optional path to save ms_ids as JSON"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't upload, just print stats"
    )
    args = parser.parse_args()

    # Step 1: Load ms_ids from GCS parquet files
    print(f"Loading ms_ids from GCS: {args.gcs_base_path}")
    ms_ids_to_keep = load_ms_ids_from_gcs(args.gcs_base_path, args.num_shards)
    print(f"Loaded {len(ms_ids_to_keep):,} unique ms_ids")

    # Optionally save ms_ids to JSON
    if args.save_ms_ids:
        sorted_ids = sorted(ms_ids_to_keep)
        with open(args.save_ms_ids, 'w') as f:
            json.dump(sorted_ids, f, indent=2)
        print(f"Saved ms_ids to: {args.save_ms_ids}")

    # Step 2: Load source dataset from HuggingFace
    print(f"\nLoading source dataset: {args.source_dataset}")
    dataset = load_dataset(args.source_dataset, split="train")
    print(f"Source dataset size: {len(dataset):,}")

    # Step 3: Filter and deduplicate
    print("\nFiltering dataset...")
    filtered_dataset = filter_and_dedupe_dataset(dataset, ms_ids_to_keep)

    # Sanity check
    filtered_ms_ids = set(filtered_dataset['ms_id'])
    missing_ids = ms_ids_to_keep - filtered_ms_ids
    if missing_ids:
        print(f"\nWarning: {len(missing_ids):,} ms_ids not found in source dataset")

    if args.dry_run:
        print("\nDry run, skipping upload")
        return

    # Step 4: Upload to HuggingFace
    print(f"\nUploading to {args.target_dataset}...")
    filtered_dataset.push_to_hub(args.target_dataset, private=False)
    print("Upload complete!")


if __name__ == "__main__":
    main()
