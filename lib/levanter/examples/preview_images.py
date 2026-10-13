# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Preview a bounded sample of a Hugging Face image dataset.

Defaults to LLaVA-558K metadata:
    uv run python lib/levanter/examples/preview_images.py --count 4

Select another dataset's native image/caption columns and default data files:
    uv run python lib/levanter/examples/preview_images.py --dataset ORG/DATASET \\
        --data-files --image-column image --caption-column caption

For extracted LLaVA images, use --image-column image --image-root /path/to/images.
Original remote image URLs may have expired.
"""

import argparse
from itertools import islice
from pathlib import Path
from typing import Any

import datasets
import matplotlib.pyplot as plt

from levanter.data.image import load_image


DEFAULT_DATASET = "liuhaotian/LLaVA-Pretrain"
DEFAULT_DATA_FILES = ["blip_laion_cc_sbu_558k_meta.json"]


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
    parser.add_argument("--image-column", default="url")
    parser.add_argument(
        "--caption-column", default="blip_caption", help="Caption column; use an empty string for no captions"
    )
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--image-root", type=Path, help="Prefix for relative paths in the image column")
    parser.add_argument("--output", type=Path, default=Path("preview_images.png"))
    args = parser.parse_args()
    if not 1 <= args.count <= 16:
        raise ValueError("Choose between 1 and 16 images for this inspection example")
    source = datasets.load_dataset(
        args.dataset, name=args.config, data_files=args.data_files or None, split=args.split, streaming=True
    )
    show_images(
        list(islice(source, args.count)),
        image_column=args.image_column,
        caption_column=args.caption_column or None,
        image_root=args.image_root,
        output=args.output,
    )


if __name__ == "__main__":
    main()
