#!/usr/bin/env python3
"""
Data Quality Checker for VLM Training Dataset

Checks for common data quality issues that can cause training crashes:
- Messages referencing images but images list is empty/null
- Corrupt or unloadable images
- Images with unusual dimensions
- Null/None entries in images list

Usage:
    # Check first 5 shards
    python experiments/VLM/check_data_quality.py \
        --data_path "gs://marin-vlm/stage3_sharded_full/*.parquet" \
        --start_shard 0 --end_shard 5

    # Check all shards (slow)
    python experiments/VLM/check_data_quality.py \
        --data_path "gs://marin-vlm/stage3_sharded_full/*.parquet"

    # Skip image loading (fast metadata-only check)
    python experiments/VLM/check_data_quality.py \
        --data_path "gs://marin-vlm/stage3_sharded_full/*.parquet" \
        --skip_image_load

    # Clean mode: preview what would be removed (dry run)
    python experiments/VLM/check_data_quality.py \
        --data_path "gs://marin-vlm/stage3_sharded_full/*.parquet" \
        --clean --dry_run --skip_image_load

    # Clean mode: remove bad rows, upload cleaned parquet, delete originals
    python experiments/VLM/check_data_quality.py \
        --data_path "gs://marin-vlm/stage3_sharded_full/*.parquet" \
        --clean --skip_image_load
"""

import argparse
import json
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import fsspec
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)
# Force unbuffered output so progress shows immediately
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
logger = logging.getLogger(__name__)


def list_parquet_files(data_path: str) -> List[str]:
    """List all parquet files matching the given path pattern.

    Uses gcsfs.ls() for GCS paths (much faster than glob on large dirs).
    """
    t0 = time.time()

    if data_path.startswith("gs://"):
        import gcsfs
        gfs = gcsfs.GCSFileSystem()
        # Strip glob pattern - just list the directory
        # e.g. "gs://bucket/path/*.parquet" -> "bucket/path"
        dir_path = data_path.replace("gs://", "")
        if "*" in dir_path:
            dir_path = dir_path[:dir_path.index("*")].rstrip("/")
        print(f"  Listing GCS directory: gs://{dir_path} ...", flush=True)
        all_files = gfs.ls(dir_path, detail=False)
        # Filter to .parquet files only
        paths = sorted(f for f in all_files if f.endswith(".parquet"))
        elapsed = time.time() - t0
        print(f"Found {len(paths)} parquet files in {elapsed:.1f}s", flush=True)
        return [f"gs://{p}" for p in paths]
    else:
        # Local filesystem - use glob
        import glob as glob_module
        paths = sorted(glob_module.glob(data_path))
        elapsed = time.time() - t0
        print(f"Found {len(paths)} parquet files in {elapsed:.1f}s", flush=True)
        return paths


def count_image_refs_in_messages(messages: List[Dict]) -> tuple[int, int, int]:
    """Count image references in a messages list, broken down by source.

    Returns:
        (image_dict_count, text_placeholder_count, string_content_count):
        - image_dict_count: number of {"type": "image"} content items
        - text_placeholder_count: number of "<image>" in {"type": "text"} content items
        - string_content_count: number of "<image>" in plain string content
    """
    image_dict_count = 0
    text_placeholder_count = 0
    string_content_count = 0
    if not messages:
        return 0, 0, 0
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "image":
                        image_dict_count += 1
                    elif item.get("type") == "text":
                        text_placeholder_count += item.get("text", "").count("<image>")
        elif isinstance(content, str):
            string_content_count += content.count("<image>")
    return image_dict_count, text_placeholder_count, string_content_count


def try_load_image(image_data: Any) -> tuple[Optional[Image.Image], Optional[str]]:
    """Try to load an image, returning (image, error_message)."""
    try:
        if image_data is None:
            return None, "image_data is None"
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB"), None
        elif isinstance(image_data, str):
            if image_data.startswith(("http://", "https://")):
                import requests
                response = requests.get(image_data, timeout=30)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert("RGB")
                return img, None
            elif image_data.startswith(("gs://", "s3://")):
                with fsspec.open(image_data, "rb") as f:
                    img = Image.open(f)
                    img.load()
                    return img.convert("RGB"), None
            else:
                img = Image.open(image_data).convert("RGB")
                return img, None
        elif isinstance(image_data, np.ndarray):
            img = Image.fromarray(image_data).convert("RGB")
            return img, None
        elif isinstance(image_data, dict):
            if "bytes" in image_data and image_data["bytes"] is not None:
                img = Image.open(BytesIO(image_data["bytes"])).convert("RGB")
                return img, None
            elif "path" in image_data and image_data["path"] is not None:
                return try_load_image(image_data["path"])
            else:
                return None, f"Unknown image dict format: {list(image_data.keys())}"
        elif isinstance(image_data, bytes):
            img = Image.open(BytesIO(image_data)).convert("RGB")
            return img, None
        else:
            return None, f"Unsupported image type: {type(image_data).__name__}"
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:200]}"


def check_image_dimensions(img: Image.Image) -> Optional[str]:
    """Check if image dimensions are unusual. Returns warning message or None."""
    w, h = img.size
    if w < 10 or h < 10:
        return f"Very small image: {w}x{h}"
    if w > 0 and h > 0:
        ratio = max(w, h) / min(w, h)
        if ratio > 20:
            return f"Extreme aspect ratio ({ratio:.1f}:1): {w}x{h}"
    return None


def check_row(
    row: Dict[str, Any],
    shard_name: str,
    row_idx: int,
    messages_key: str = "messages",
    images_key: str = "images",
    skip_image_load: bool = False,
) -> List[Dict]:
    """Check a single row for data quality issues. Returns list of issues."""
    issues = []
    base_info = {"shard": shard_name, "row": row_idx}

    messages = row.get(messages_key)
    images = row.get(images_key)

    # Count image references in messages, broken down by source
    if messages:
        image_dict_count, text_placeholder_count, string_content_count = count_image_refs_in_messages(messages)
    else:
        image_dict_count, text_placeholder_count, string_content_count = 0, 0, 0
    num_image_refs = image_dict_count + text_placeholder_count + string_content_count

    # Check images field
    if images is None:
        num_images = 0
    elif isinstance(images, (list, tuple)):
        num_images = len(images)
    else:
        num_images = 1  # Single image

    # Issue A: Messages reference images but images list is empty/null
    if num_image_refs > 0 and num_images == 0:
        issues.append({
            **base_info,
            "issue": "image_ref_but_no_images",
            "detail": f"Messages reference {num_image_refs} image(s) but images list is empty/null",
        })

    # Issue A2: Image placeholder count mismatch
    if num_image_refs > 0 and num_images > 0 and num_image_refs != num_images:
        issues.append({
            **base_info,
            "issue": "image_count_mismatch",
            "detail": (
                f"Messages produce {num_image_refs} <image> placeholder(s) "
                f"(image_dicts={image_dict_count}, text_placeholders={text_placeholder_count}, "
                f"string_content={string_content_count}) but images has {num_images} entries"
            ),
        })

    # Issue A3: <image> in text content alongside {"type": "image"} dicts
    # This is the root cause of the "Image token count mismatch" ValueError:
    # chat template converts each {"type": "image"} dict to <image>, AND passes through
    # any <image> already in text content, causing double-counting.
    if image_dict_count > 0 and text_placeholder_count > 0:
        issues.append({
            **base_info,
            "issue": "image_placeholder_in_text",
            "detail": (
                f"{image_dict_count} image dict(s) + {text_placeholder_count} <image> in text "
                f"= {image_dict_count + text_placeholder_count} placeholders for {num_images} image(s)"
            ),
        })

    # Issue D: Null entries in images list
    if isinstance(images, (list, tuple)):
        for img_idx, img_data in enumerate(images):
            if img_data is None:
                issues.append({
                    **base_info,
                    "issue": "null_image_entry",
                    "detail": f"images[{img_idx}] is None",
                })

    # Issue B & C: Try loading images
    if not skip_image_load and isinstance(images, (list, tuple)):
        for img_idx, img_data in enumerate(images):
            img, error = try_load_image(img_data)
            if error:
                issues.append({
                    **base_info,
                    "issue": "image_load_failed",
                    "detail": f"images[{img_idx}]: {error}",
                })
            elif img is not None:
                dim_warning = check_image_dimensions(img)
                if dim_warning:
                    issues.append({
                        **base_info,
                        "issue": "unusual_dimensions",
                        "detail": f"images[{img_idx}]: {dim_warning}",
                    })

    return issues


def is_row_clean(row: Dict[str, Any], messages_key: str = "messages", images_key: str = "images") -> bool:
    """Check if a row is clean (no image reference issues).

    Rejects rows with:
    - image_ref_but_no_images: messages reference images but images list is empty/null
    - image_count_mismatch: <image> placeholder count != actual images count
    - image_placeholder_in_text: {"type": "image"} dicts AND <image> in text content coexist
    """
    messages = row.get(messages_key)
    images = row.get(images_key)

    if messages:
        image_dict_count, text_placeholder_count, string_content_count = count_image_refs_in_messages(messages)
    else:
        image_dict_count, text_placeholder_count, string_content_count = 0, 0, 0
    num_image_refs = image_dict_count + text_placeholder_count + string_content_count

    if images is None:
        num_images = 0
    elif isinstance(images, (list, tuple)):
        num_images = len(images)
    else:
        num_images = 1

    # Reject: messages reference images but images list is empty/null
    if num_image_refs > 0 and num_images == 0:
        return False
    # Reject: placeholder count doesn't match actual images count
    if num_image_refs > 0 and num_images > 0 and num_image_refs != num_images:
        return False
    # Reject: both {"type": "image"} dicts and <image> in text (double-counting)
    if image_dict_count > 0 and text_placeholder_count > 0:
        return False
    return True


def clean_shard(
    shard_path: str,
    messages_key: str,
    images_key: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Clean a single parquet shard by removing rows with image_ref_but_no_images.

    Returns dict with stats: original_rows, cleaned_rows, removed_rows, cleaned_path.
    """
    import gcsfs
    import os
    import tempfile

    from datasets import Dataset

    shard_name = Path(shard_path).stem
    t0 = time.time()

    # Read parquet
    with fsspec.open(shard_path, "rb") as f:
        table = pq.read_table(f)

    original_rows = len(table)
    rows_dict = table.to_pydict()

    # Build boolean mask: True = keep, False = remove
    mask = []
    for row_idx in range(original_rows):
        row = {k: v[row_idx] for k, v in rows_dict.items()}
        mask.append(is_row_clean(row, messages_key, images_key))

    removed = sum(1 for m in mask if not m)
    cleaned_rows = original_rows - removed

    if dry_run:
        return {
            "shard_name": shard_name,
            "original_rows": original_rows,
            "cleaned_rows": cleaned_rows,
            "removed_rows": removed,
            "cleaned_path": None,
            "elapsed": time.time() - t0,
        }

    # Build cleaned rows as list of dicts for Dataset.from_list()
    clean_rows = [
        {k: rows_dict[k][i] for k in rows_dict}
        for i, m in enumerate(mask) if m
    ]

    # Use HuggingFace Dataset.to_parquet() instead of pq.write_table().
    # pq.write_table() produces nested type encoding that training workers'
    # PyArrow version can't read. Dataset.to_parquet() uses the same encoding
    # as the original data pipeline (convert_llava_onevision_to_levanter.py).
    dataset = Dataset.from_list(clean_rows)

    # Write to temp file, then upload to GCS (same pattern as convert script)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        dataset.to_parquet(tmp_path)

        if shard_path.startswith("gs://"):
            gfs = gcsfs.GCSFileSystem()
            # Write cleaned file with _cleaned suffix
            cleaned_path = shard_path.replace(".parquet", "_cleaned.parquet")
            gcs_cleaned_path = cleaned_path.replace("gs://", "")
            gfs.put(tmp_path, gcs_cleaned_path)
            # Delete original
            gcs_original_path = shard_path.replace("gs://", "")
            gfs.rm(gcs_original_path)
        else:
            cleaned_path = shard_path.replace(".parquet", "_cleaned.parquet")
            import shutil
            shutil.move(tmp_path, cleaned_path)
            os.remove(shard_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    del dataset, clean_rows
    import gc
    gc.collect()

    return {
        "shard_name": shard_name,
        "original_rows": original_rows,
        "cleaned_rows": cleaned_rows,
        "removed_rows": removed,
        "cleaned_path": cleaned_path,
        "elapsed": time.time() - t0,
    }


def main():
    parser = argparse.ArgumentParser(description="Check VLM training data quality")
    parser.add_argument("--data_path", required=True, help="GCS path pattern to parquet files (e.g., gs://bucket/path/*.parquet)")
    parser.add_argument("--start_shard", type=int, default=0, help="Start shard index (inclusive)")
    parser.add_argument("--end_shard", type=int, default=None, help="End shard index (exclusive)")
    parser.add_argument("--messages_key", default="messages", help="Key for messages field")
    parser.add_argument("--images_key", default="images", help="Key for images field")
    parser.add_argument("--skip_image_load", action="store_true", help="Skip actual image loading (fast metadata-only check)")
    parser.add_argument("--output", default="data_quality_report.json", help="Output JSON file for results")
    parser.add_argument("--max_issues_per_shard", type=int, default=100, help="Max issues to report per shard")
    parser.add_argument("--clean", action="store_true", help="Clean mode: remove bad rows and re-upload parquet files")
    parser.add_argument("--dry_run", action="store_true", help="With --clean: only report what would be removed, don't modify files")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers for processing shards")
    args = parser.parse_args()

    # List parquet files
    print(f"Listing parquet files from {args.data_path} ...", flush=True)
    parquet_files = list_parquet_files(args.data_path)
    if not parquet_files:
        logger.error("No parquet files found!")
        return

    # Apply shard range
    end_shard = args.end_shard if args.end_shard is not None else len(parquet_files)
    parquet_files = parquet_files[args.start_shard:end_shard]
    logger.info(f"Checking shards {args.start_shard} to {end_shard} ({len(parquet_files)} shards)")

    # ── Clean mode ──
    if args.clean:
        mode_str = "DRY RUN" if args.dry_run else "CLEANING"
        print(f"\n{'=' * 70}")
        print(f"  {mode_str}: Removing bad rows (missing images, count mismatch, placeholder in text) from {len(parquet_files)} shards")
        print(f"{'=' * 70}\n", flush=True)

        total_original = 0
        total_removed = 0
        total_cleaned = 0

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for shard_idx, shard_path in enumerate(parquet_files):
                future = executor.submit(
                    clean_shard, shard_path,
                    messages_key=args.messages_key,
                    images_key=args.images_key,
                    dry_run=args.dry_run,
                )
                futures[future] = (shard_idx, shard_path)

            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                shard_idx, shard_path = futures[future]
                global_shard_idx = args.start_shard + shard_idx
                shard_name = Path(shard_path).stem

                try:
                    stats = future.result()
                    total_original += stats["original_rows"]
                    total_removed += stats["removed_rows"]
                    total_cleaned += stats["cleaned_rows"]

                    if stats["removed_rows"] > 0:
                        print(
                            f"[{done_count}/{len(parquet_files)}] {mode_str} shard {global_shard_idx} ({shard_name}): "
                            f"{stats['original_rows']} -> {stats['cleaned_rows']} rows "
                            f"(removed {stats['removed_rows']}) in {stats['elapsed']:.1f}s",
                            flush=True,
                        )
                    else:
                        print(
                            f"[{done_count}/{len(parquet_files)}] {mode_str} shard {global_shard_idx} ({shard_name}): "
                            f"{stats['original_rows']} rows, no issues ({stats['elapsed']:.1f}s)",
                            flush=True,
                        )

                except Exception as e:
                    print(
                        f"[{done_count}/{len(parquet_files)}] {mode_str} shard {global_shard_idx} ({shard_name}): FAILED: {e}",
                        flush=True,
                    )
                    logger.error(f"Failed to clean shard {shard_path}: {e}")

        print(f"\n{'=' * 70}")
        print(f"{'DRY RUN ' if args.dry_run else ''}CLEAN SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total shards: {len(parquet_files)}")
        print(f"Total original rows: {total_original}")
        print(f"Total removed rows: {total_removed}")
        print(f"Total cleaned rows: {total_cleaned}")
        if args.dry_run:
            print("\n(Dry run - no files were modified)")
        return

    # ── Check mode (original behavior) ──

    # Collect all issues
    all_issues: List[Dict] = []
    issue_counts = {
        "image_ref_but_no_images": 0,
        "image_count_mismatch": 0,
        "image_placeholder_in_text": 0,
        "null_image_entry": 0,
        "image_load_failed": 0,
        "unusual_dimensions": 0,
    }
    total_rows = 0
    total_with_images = 0

    def check_shard(shard_path: str, messages_key: str, images_key: str,
                    skip_image_load: bool, max_issues: int) -> Dict[str, Any]:
        """Check a single shard for data quality issues. Returns results dict."""
        shard_name = Path(shard_path).stem
        t0 = time.time()
        with fsspec.open(shard_path, "rb") as f:
            table = pq.read_table(f)
        rows = table.to_pydict()
        num_rows = len(rows.get(messages_key, []))

        shard_issues = []
        with_images = 0
        for row_idx in range(num_rows):
            row = {k: v[row_idx] for k, v in rows.items()}
            row_issues = check_row(
                row, shard_name=shard_name, row_idx=row_idx,
                messages_key=messages_key, images_key=images_key,
                skip_image_load=skip_image_load,
            )
            shard_issues.extend(row_issues)
            images = row.get(images_key)
            if images and len(images) > 0:
                with_images += 1
            if len(shard_issues) >= max_issues:
                break

        return {
            "shard_name": shard_name,
            "num_rows": num_rows,
            "with_images": with_images,
            "issues": shard_issues,
            "elapsed": time.time() - t0,
        }

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for shard_idx, shard_path in enumerate(parquet_files):
            future = executor.submit(
                check_shard, shard_path,
                messages_key=args.messages_key,
                images_key=args.images_key,
                skip_image_load=args.skip_image_load,
                max_issues=args.max_issues_per_shard,
            )
            futures[future] = (shard_idx, shard_path)

        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            shard_idx, shard_path = futures[future]
            global_shard_idx = args.start_shard + shard_idx
            shard_name = Path(shard_path).stem

            try:
                result = future.result()
                total_rows += result["num_rows"]
                total_with_images += result["with_images"]
                shard_issues = result["issues"]

                for issue in shard_issues:
                    issue_type = issue["issue"]
                    if issue_type in issue_counts:
                        issue_counts[issue_type] += 1
                all_issues.extend(shard_issues)

                if shard_issues:
                    print(
                        f"[{done_count}/{len(parquet_files)}] shard {global_shard_idx} ({shard_name}): "
                        f"{result['num_rows']} rows, {len(shard_issues)} ISSUES in {result['elapsed']:.1f}s",
                        flush=True,
                    )
                else:
                    print(
                        f"[{done_count}/{len(parquet_files)}] shard {global_shard_idx} ({shard_name}): "
                        f"{result['num_rows']} rows, OK ({result['elapsed']:.1f}s)",
                        flush=True,
                    )

            except Exception as e:
                print(
                    f"[{done_count}/{len(parquet_files)}] shard {global_shard_idx} ({shard_name}): FAILED: {e}",
                    flush=True,
                )
                all_issues.append({
                    "shard": shard_name,
                    "row": -1,
                    "issue": "shard_read_failed",
                    "detail": str(e)[:500],
                })

    # Print summary
    print("\n" + "=" * 70)
    print("DATA QUALITY CHECK SUMMARY")
    print("=" * 70)
    print(f"Total shards checked: {len(parquet_files)}")
    print(f"Total rows: {total_rows}")
    print(f"Rows with images: {total_with_images}")
    print(f"Total issues found: {len(all_issues)}")
    print()
    print("Issues by type:")
    for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {issue_type}: {count}")
    print()

    if all_issues:
        print("First 20 issues:")
        for issue in all_issues[:20]:
            print(f"  [{issue['shard']}:row{issue['row']}] {issue['issue']}: {issue['detail']}")

    # Save full report
    report = {
        "summary": {
            "total_shards": len(parquet_files),
            "shard_range": [args.start_shard, end_shard],
            "total_rows": total_rows,
            "rows_with_images": total_with_images,
            "total_issues": len(all_issues),
            "issue_counts": issue_counts,
        },
        "issues": all_issues,
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()
