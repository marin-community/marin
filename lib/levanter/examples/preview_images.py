# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Preview a bounded sample of a Hugging Face image dataset.

Defaults to LLaVA-558K metadata and images hosted on Hugging Face:
    uv run python lib/levanter/examples/preview_images.py --count 4

Select another dataset's native image/caption columns and default data files:
    uv run python lib/levanter/examples/preview_images.py --dataset ORG/DATASET \\
        --data-files --image-column image --caption-column caption

For extracted LLaVA images, use --image-column image --image-root /path/to/images.
The default reads selected images from the hosted ZIP using range requests.
Use --image-column url for original website URLs, which may have expired.
"""

import argparse
from itertools import islice
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import datasets
import fsspec
import matplotlib.pyplot as plt
from PIL import Image

from levanter.data.image import load_image


DEFAULT_DATASET = "liuhaotian/LLaVA-Pretrain"
DEFAULT_DATA_FILES = ["blip_laion_cc_sbu_558k_meta.json"]
DEFAULT_IMAGE_ARCHIVE = f"hf://datasets/{DEFAULT_DATASET}/images.zip"
ARCHIVE_BLOCK_SIZE = 1024 * 1024


def show_images(
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
    for axis, record in zip(axes.flat, records):
        reference = record[image_column]
        if image_root is not None:
            reference = str(image_root / reference)
        axis.imshow(load_image(reference))
        if caption_column is not None:
            axis.set_title(str(record[caption_column]), wrap=True, fontsize=9)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", help="Hugging Face dataset configuration name")
    parser.add_argument(
        "--data-files",
        nargs="*",
        default=DEFAULT_DATA_FILES,
        help="Data files or glob patterns. Pass this option with no values to use the dataset's default files.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-column", default="image")
    parser.add_argument(
        "--caption-column", default="blip_caption", help="Caption column; use an empty string for no captions"
    )
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--image-root", type=Path, help="Prefix for relative paths in the image column")
    parser.add_argument("--image-archive", help="ZIP containing image-column paths; defaults to the hosted LLaVA ZIP")
    parser.add_argument("--output", type=Path, default=Path("preview_images.png"))
    args = parser.parse_args()
    if not 1 <= args.count <= 16:
        raise ValueError("Choose between 1 and 16 images for this inspection example")
    if args.image_root is not None and args.image_archive is not None:
        raise ValueError("Choose either --image-root or --image-archive")
    source = datasets.load_dataset(
        args.dataset, name=args.config, data_files=args.data_files or None, split=args.split, streaming=True
    )
    records = list(islice(source, args.count))
    image_archive = args.image_archive
    if args.dataset == DEFAULT_DATASET and args.image_column == "image" and args.image_root is None:
        image_archive = image_archive or DEFAULT_IMAGE_ARCHIVE
    if image_archive is not None:
        with fsspec.open(image_archive, "rb", block_size=ARCHIVE_BLOCK_SIZE) as remote:
            with ZipFile(remote) as archive:
                for record in records:
                    with archive.open(record[args.image_column]) as image_file:
                        with Image.open(image_file) as image:
                            record[args.image_column] = image.convert("RGB")
    show_images(
        records,
        image_column=args.image_column,
        caption_column=args.caption_column or None,
        image_root=args.image_root,
        output=args.output,
    )


if __name__ == "__main__":
    main()
