# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect LLaVA-558K: uv run python lib/levanter/examples/llava558k_images.py --count 4.

Streams the explicitly selected metadata JSON and fetches only the selected image
URLs, which may have expired. --image-root uses a locally extracted images.zip.
"""

import argparse
from itertools import islice
from pathlib import Path
from typing import Any

import datasets
import matplotlib.pyplot as plt

from levanter.data.image import load_image


def show_images(records: list[dict[str, Any]], image_root: Path | None, output: Path) -> None:
    figure, axes = plt.subplots(1, len(records), squeeze=False, figsize=(4 * len(records), 4))
    for axis, record in zip(axes[0], records, strict=True):
        reference = str(image_root / record["image"]) if image_root is not None else record["url"]
        axis.imshow(load_image(reference))
        axis.set_title(record["blip_caption"], wrap=True, fontsize=9)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("llava558k.png"))
    args = parser.parse_args()
    if not 1 <= args.count <= 16:
        raise ValueError("Choose between 1 and 16 images for this inspection example")
    source = datasets.load_dataset("liuhaotian/LLaVA-Pretrain", data_files="blip_laion_cc_sbu_558k_meta.json", split="train", streaming=True)
    show_images(list(islice(source, args.count)), args.image_root, args.output)


if __name__ == "__main__":
    main()
