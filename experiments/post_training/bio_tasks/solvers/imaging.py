# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Image and spatial measurements from native P2 rasters or point tables."""

import json
import math
from collections import Counter
from functools import partial
from pathlib import Path
from statistics import median

from experiments.post_training.bio_tasks.solvers.formats import table


def pgm(path: Path) -> list[list[int]]:
    tokens = " ".join(line.split("#", 1)[0] for line in path.read_text().splitlines()).split()
    assert tokens[0] == "P2"
    width, height, maximum = map(int, tokens[1:4])
    values = list(map(int, tokens[4:]))
    assert len(values) == width * height and all(0 <= value <= maximum for value in values)
    return [values[start : start + width] for start in range(0, len(values), width)]


def solve_imaging(inputs: Path, operation: str) -> list[dict]:
    answer = []
    if operation == "image-threshold-components":
        rows = pgm(inputs / "image.pgm")
        metadata = json.loads((inputs / "metadata.json").read_text())
        scale = metadata["pixel_size_um"]
        unseen = {(y, x) for y, row in enumerate(rows) for x, value in enumerate(row) if value >= metadata["threshold"]}
        index = 0
        while unseen:
            start = min(unseen)
            unseen.remove(start)
            component = [start]
            stack = [start]
            while stack:
                y, x = stack.pop()
                for neighbor in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.append(neighbor)
                        stack.append(neighbor)
            index += 1
            n = len(component)
            answer.append(
                {
                    "id": str(index),
                    "pixels": n,
                    "area": n * scale * scale,
                    "centroid_x": (sum(x for _, x in component) / n + 0.5) * scale,
                    "centroid_y": (sum(y for y, _ in component) / n + 0.5) * scale,
                }
            )
    elif operation == "image-background-correction":
        image = pgm(inputs / "image.pgm")
        labels = pgm(inputs / "labels.pgm")
        background = median(image[y][x] for y, row in enumerate(labels) for x, label in enumerate(row) if label == 0)
        for label in sorted({label for row in labels for label in row} - {0}):
            values = [
                image[y][x] - background for y, row in enumerate(labels) for x, value in enumerate(row) if value == label
            ]
            answer.append(
                {
                    "id": str(label),
                    "background": float(background),
                    "corrected_sum": sum(values),
                    "corrected_mean": sum(values) / len(values),
                }
            )
    elif operation == "image-colocalization":
        a, b, regions = pgm(inputs / "channel_a.pgm"), pgm(inputs / "channel_b.pgm"), pgm(inputs / "regions.pgm")
        for label in sorted({value for row in regions for value in row} - {0}):
            pairs = [
                (a[y][x], b[y][x]) for y, row in enumerate(regions) for x, value in enumerate(row) if value == label
            ]
            n = len(pairs)
            sx, sy = sum(x for x, _ in pairs), sum(y for _, y in pairs)
            numerator = n * sum(x * y for x, y in pairs) - sx * sy
            denominator = math.sqrt(
                (n * sum(x * x for x, _ in pairs) - sx * sx) * (n * sum(y * y for _, y in pairs) - sy * sy)
            )
            answer.append({"id": str(label), "pearson": numerator / denominator, "pixels": n})
    elif operation == "spatial-neighbor-enrichment":
        cells = table(inputs / "cells.csv")
        radius = float((inputs / "radius_um.txt").read_text())
        counts = Counter()
        for a in cells:
            for b in cells:
                if (
                    a["cell"] != b["cell"]
                    and math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"])) <= radius
                ):
                    counts[a["type"], b["type"]] += 1
        types = sorted({row["type"] for row in cells})
        for a in types:
            total = sum(counts[a, b] for b in types)
            for b in types:
                answer.append(
                    {"id": a + ":" + b, "edges": counts[a, b], "fraction": counts[a, b] / total if total else 0.0}
                )
    elif operation == "spatial-region-counts":
        cells = table(inputs / "cells.csv")
        types = sorted({row["type"] for row in cells})
        for region in table(inputs / "regions.csv"):
            for kind in types:
                count = sum(
                    row["type"] == kind
                    and float(region["xmin"]) <= float(row["x"]) < float(region["xmax"])
                    and float(region["ymin"]) <= float(row["y"]) < float(region["ymax"])
                    for row in cells
                )
                answer.append({"id": region["region"] + ":" + kind, "count": count})
    else:
        a, b = pgm(inputs / "reference.pgm"), pgm(inputs / "prediction.pgm")
        for text in (inputs / "labels.txt").read_text().splitlines():
            label = int(text)
            left = {(y, x) for y, row in enumerate(a) for x, value in enumerate(row) if value == label}
            right = {(y, x) for y, row in enumerate(b) for x, value in enumerate(row) if value == label}
            overlap, union = len(left & right), len(left | right)
            answer.append(
                {
                    "id": text,
                    "intersection": overlap,
                    "dice": 2 * overlap / (len(left) + len(right)) if left or right else 1.0,
                    "iou": overlap / union if union else 1.0,
                }
            )
    return answer


NAMES = (
    "image-threshold-components",
    "image-background-correction",
    "image-colocalization",
    "spatial-neighbor-enrichment",
    "spatial-region-counts",
    "image-dice-iou",
)
SOLVERS = {name: partial(solve_imaging, operation=name) for name in NAMES}
