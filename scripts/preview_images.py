# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Save a grid of up to 16 images from a Hugging Face dataset.

Preview four images from the hosted LLaVA-558K ZIP:
    uv run python scripts/preview_images.py --count 4

Use another dataset's default files and columns:
    uv run python scripts/preview_images.py --dataset ORG/DATASET \\
        --data-files --image-column image --caption-column caption

For extracted LLaVA images, pass --image-root /path/to/images.
To fetch the original web images, pass --image-column url. Some links may have expired.
"""

import argparse
from itertools import islice
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import datasets
import fsspec
import matplotlib.pyplot as plt
from levanter.data.image import load_image
from PIL import Image

DEFAULT_DATASET = "liuhaotian/LLaVA-Pretrain"
DEFAULT_DATA_FILES = ("blip_laion_cc_sbu_558k_meta.json",)
DEFAULT_IMAGE_ARCHIVE = f"hf://datasets/{DEFAULT_DATASET}/images.zip"
ARCHIVE_BLOCK_SIZE = 1024 * 1024


def save_image_grid(
    records: list[dict[str, Any]],
    *,
    image_column: str,
    caption_column: str | None,
    image_root: Path | None,
    output: Path,
) -> None:
    """Save a grid from image references, arrays, or decoded PIL images."""
    if not records:
        raise ValueError("The selected dataset split contains no images")
    columns = min(4, len(records))
    rows = (len(records) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, squeeze=False, figsize=(4 * columns, 4 * rows))
    for axis in axes.flat:
        axis.axis("off")
    for axis, record in zip(axes.flat, records, strict=False):
        reference = record[image_column]
        if image_root is not None:
            reference = str(image_root / reference)
        axis.imshow(load_image(reference))
        if caption_column is not None:
            axis.set_title(str(record[caption_column]), wrap=True, fontsize=9)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", help="Hugging Face dataset configuration name")
    parser.add_argument(
        "--data-files",
        nargs="*",
        default=DEFAULT_DATA_FILES,
        help="Files or glob patterns. Pass --data-files alone to use default dataset files.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-column", default="image")
    parser.add_argument(
        "--caption-column", default="blip_caption", help="Caption column (pass an empty string to hide captions)"
    )
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--image-root", type=Path, help="Prefix for relative paths in the image column")
    parser.add_argument("--image-archive", help="ZIP containing the images (LLaVA uses its hosted ZIP by default)")
    parser.add_argument("--output", type=Path, default=Path("preview_images.png"))
    args = parser.parse_args()
    if not 1 <= args.count <= 16:
        raise ValueError("Choose between 1 and 16 images for this inspection example")
    if args.image_root is not None and args.image_archive is not None:
        raise ValueError("Choose either --image-root or --image-archive")
    return args


def _load_archive_images(records: list[dict[str, Any]], image_column: str, image_archive: str) -> None:
    with fsspec.open(image_archive, "rb", block_size=ARCHIVE_BLOCK_SIZE) as remote, ZipFile(remote) as archive:
        for record in records:
            with archive.open(record[image_column]) as image_file, Image.open(image_file) as image:
                record[image_column] = image.convert("RGB")


def main() -> None:
    args = _parse_args()
    source = datasets.load_dataset(
        args.dataset, name=args.config, data_files=args.data_files or None, split=args.split, streaming=True
    )
    records = list(islice(source, args.count))
    image_archive = args.image_archive
    if args.dataset == DEFAULT_DATASET and args.image_column == "image" and args.image_root is None:
        image_archive = image_archive or DEFAULT_IMAGE_ARCHIVE
    if image_archive is not None:
        _load_archive_images(records, args.image_column, image_archive)
    save_image_grid(
        records,
        image_column=args.image_column,
        caption_column=args.caption_column or None,
        image_root=args.image_root,
        output=args.output,
    )


if __name__ == "__main__":
    main()
