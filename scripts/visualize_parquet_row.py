#!/usr/bin/env python
"""Visualize a single row from a parquet file (local or GCS).

Usage:
    python scripts/visualize_parquet_row.py gs://bucket/path/to/data.parquet --row 42
    python scripts/visualize_parquet_row.py gs://bucket/path/to/data.parquet --row 0 --output-dir /tmp/viz
    python scripts/visualize_parquet_row.py /local/path/data.parquet
"""

import argparse
import base64
import json
import os
from io import BytesIO

import fsspec
import pyarrow.parquet as pq
from PIL import Image


def save_image(image_data, output_path):
    """Try to decode image_data and save as PNG. Returns True on success."""
    try:
        if isinstance(image_data, Image.Image):
            image_data.convert("RGB").save(output_path)
            return True
        if isinstance(image_data, bytes):
            Image.open(BytesIO(image_data)).convert("RGB").save(output_path)
            return True
        if isinstance(image_data, dict) and "bytes" in image_data:
            raw = image_data["bytes"]
            if isinstance(raw, bytes):
                Image.open(BytesIO(raw)).convert("RGB").save(output_path)
                return True
        if isinstance(image_data, str):
            if image_data.startswith("base64:"):
                raw = base64.b64decode(image_data[len("base64:"):])
                Image.open(BytesIO(raw)).convert("RGB").save(output_path)
                return True
            # Could be a file path or URL — just print it, don't try to fetch
            print(f"    (image reference: {image_data})")
            return False
    except Exception as e:
        print(f"    (failed to decode image: {e})")
    return False


def print_messages(messages):
    """Pretty-print a conversation messages list."""
    if not isinstance(messages, list):
        print(f"  {messages}")
        return
    for i, msg in enumerate(messages):
        if isinstance(msg, dict):
            role = msg.get("role", "?")
            content = msg.get("content", "")
            print(f"  [{i}] role={role}")
            if isinstance(content, str):
                print(f"      text: {content[:500]}")
            elif isinstance(content, list):
                for j, part in enumerate(content):
                    if isinstance(part, dict):
                        ptype = part.get("type", "?")
                        if ptype == "text":
                            print(f"      [{j}] text: {part.get('text', '')[:500]}")
                        elif ptype == "image":
                            print(f"      [{j}] <image>")
                        else:
                            print(f"      [{j}] {ptype}: {str(part)[:200]}")
                    else:
                        print(f"      [{j}] {str(part)[:200]}")
            else:
                print(f"      content: {str(content)[:500]}")
        else:
            print(f"  [{i}] {str(msg)[:300]}")


def truncate(value, max_len=200):
    """Truncate a value for display."""
    s = str(value)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def main():
    parser = argparse.ArgumentParser(description="Visualize a row from a parquet file")
    parser.add_argument("parquet_path", help="Path to parquet file (local or gs://...)")
    parser.add_argument("--row", type=int, default=0, help="Row index to visualize (default: 0)")
    parser.add_argument("--output-dir", default="./parquet_viz_output", help="Directory to save images")
    args = parser.parse_args()

    # Open parquet file via fsspec
    fs, path = fsspec.core.url_to_fs(args.parquet_path)
    pf = pq.ParquetFile(fs.open(path))

    total_rows = pf.metadata.num_rows
    print(f"Parquet file: {args.parquet_path}")
    print(f"Total rows: {total_rows}")
    print(f"Schema: {pf.schema_arrow}")
    print()

    if args.row < 0 or args.row >= total_rows:
        print(f"Error: row {args.row} out of range [0, {total_rows})")
        return

    # Read the target row by scanning row groups
    row_offset = 0
    target_table = None
    for i in range(pf.metadata.num_row_groups):
        rg_rows = pf.metadata.row_group(i).num_rows
        if row_offset + rg_rows > args.row:
            table = pf.read_row_group(i)
            local_idx = args.row - row_offset
            target_table = table.slice(local_idx, 1)
            break
        row_offset += rg_rows

    if target_table is None:
        print(f"Error: could not read row {args.row}")
        return

    row = target_table.to_pydict()
    # Each value is a list of length 1; unwrap
    row = {k: v[0] if v else None for k, v in row.items()}

    print(f"=== Row {args.row} ===")
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    saved_images = []

    for col_name, value in row.items():
        print(f"[{col_name}]")

        # Special handling for messages
        if col_name == "messages":
            print_messages(value)
            print()
            continue

        # Special handling for images column
        if col_name == "images" or col_name == "image":
            items = value if isinstance(value, list) else [value]
            print(f"  count: {len(items)}")
            for i, img_data in enumerate(items):
                out_path = os.path.join(args.output_dir, f"row_{args.row}_image_{i}.png")
                if save_image(img_data, out_path):
                    print(f"  saved: {out_path}")
                    saved_images.append(out_path)
            print()
            continue

        # Generic column display
        if isinstance(value, (list, dict)):
            print(f"  {truncate(json.dumps(value, default=str), 500)}")
        else:
            print(f"  {truncate(value, 500)}")
        print()

    if saved_images:
        print(f"Saved {len(saved_images)} image(s) to {args.output_dir}/")


if __name__ == "__main__":
    main()
