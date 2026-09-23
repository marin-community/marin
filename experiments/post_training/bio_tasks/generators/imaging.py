# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small native PGM images and physical-coordinate spatial measurements."""

import json
import math
import random
from functools import partial

from experiments.post_training.bio_tasks.contract import Column, Contract
from experiments.post_training.bio_tasks.recipe_types import Difficulty, Instance, Recipe, csv_text


def pgm_text(rows: list[list[int]]) -> str:
    return f"P2\n# quantitative fixture; values are linear intensities\n{len(rows[0])} {len(rows)}\n255\n" + "".join(
        " ".join(map(str, row)) + "\n" for row in rows
    )


def generate_imaging(seed: int, operation: str) -> Instance:
    rng = random.Random(seed)
    inputs, expected = {}, {}
    if operation == "image-threshold-components":
        intensity = rng.randint(120, 220)
        rows = [[0] * 6 for _ in range(6)]
        points = [[(0, 0), (0, 1), (1, 0)], [(2, 2), (2, 3), (3, 3)], [(5, 5)]]
        for group in points:
            for y, x in group:
                rows[y][x] = intensity
        # These diagonal pixels remain separate under four-connectivity.
        rows[1][1] = 100
        points[0].append((1, 1))
        scale = rng.choice([0.5, 1.0, 2.0])
        inputs = {
            "image.pgm": pgm_text(rows),
            "metadata.json": json.dumps({"pixel_size_um": scale, "threshold": 100}) + "\n",
        }
        for i, group in enumerate(points, 1):
            expected[str(i)] = {
                "pixels": len(group),
                "area": len(group) * scale**2,
                "centroid_x": sum(x + 0.5 for _, x in group) / len(group) * scale,
                "centroid_y": sum(y + 0.5 for y, _ in group) / len(group) * scale,
            }
        columns = {
            "pixels": Column(kind="integer", unit="pixels", description="component size"),
            "area": Column(
                kind="number",
                unit="square micrometers",
                description="pixel count times squared pixel size",
                atol=1e-8,
                rtol=1e-8,
            ),
            "centroid_x": Column(
                kind="number", unit="micrometers", description="mean column-center coordinate", atol=1e-8, rtol=1e-8
            ),
            "centroid_y": Column(
                kind="number", unit="micrometers", description="mean row-center coordinate", atol=1e-8, rtol=1e-8
            ),
        }
        prompt = (
            "Threshold the P2 image.pgm at intensity >= metadata.json threshold, then label foreground "
            "with four-neighbor connectivity (not diagonals). Number components by their first "
            "foreground pixel in row-major order. Report pixel count, physical area, and centroid of "
            "pixel centers: x=(column+0.5)*pixel_size_um, y=(row+0.5)*pixel_size_um. Use component "
            "number as string id."
        )
        wrong = [{"id": k, **v, "area": v["pixels"] * scale} for k, v in expected.items()]
        if scale == 1.0:
            wrong = [{"id": k, **v, "centroid_x": v["centroid_x"] - 0.5} for k, v in expected.items()]
        reason = "wrong_pixel_geometry"
    elif operation == "image-background-correction":
        background = rng.randint(8, 20)
        labels = [[0, 0, 0, 0], [0, 1, 1, 0], [0, 2, 2, 0], [0, 0, 0, 0]]
        image = [[background] * 4 for _ in range(4)]
        image[0][0] = background + 50
        image[1][1], image[1][2] = background + 10, background + 20
        image[2][1], image[2][2] = background - 3, background + 7
        inputs = {"image.pgm": pgm_text(image), "labels.pgm": pgm_text(labels)}
        expected = {
            "1": {"background": float(background), "corrected_sum": 30.0, "corrected_mean": 15.0},
            "2": {"background": float(background), "corrected_sum": 4.0, "corrected_mean": 2.0},
        }
        columns = {
            name: Column(kind="number", unit="intensity units", description=description, atol=1e-8, rtol=1e-8)
            for name, description in [
                ("background", "median intensity over label-zero pixels"),
                ("corrected_sum", "sum after background subtraction, no clipping"),
                ("corrected_mean", "mean after background subtraction, no clipping"),
            ]
        }
        prompt = (
            "Estimate background as the median image.pgm intensity over label=0 pixels in labels.pgm. "
            "Subtract that scalar from every nonzero-label pixel without clipping negative values. "
            "Report background, corrected sum and corrected mean per nonzero label id. Label-zero "
            "pixels are not objects."
        )
        wrong = [{"id": k, **v, "corrected_sum": 7.0 if k == "2" else v["corrected_sum"]} for k, v in expected.items()]
        reason = "clipped_negative_background_corrected_pixels"
    elif operation == "image-colocalization":
        offset = rng.randint(2, 9)
        a = [[offset + i for i in range(4)], [offset + i + 4 for i in range(4)]]
        b = [[3 * x + 5 for x in a[0]], [80 - 2 * x for x in a[1]]]
        b[0][-1] += rng.randint(1, 8)
        centered_x = [x - sum(a[0]) / 4 for x in a[0]]
        centered_y = [y - sum(b[0]) / 4 for y in b[0]]
        correlation = sum(x * y for x, y in zip(centered_x, centered_y, strict=True)) / math.sqrt(
            sum(x * x for x in centered_x) * sum(y * y for y in centered_y)
        )
        mask = [[1, 1, 1, 1], [2, 2, 2, 2]]
        inputs = {"channel_a.pgm": pgm_text(a), "channel_b.pgm": pgm_text(b), "regions.pgm": pgm_text(mask)}
        expected = {"1": {"pearson": correlation, "pixels": 4}, "2": {"pearson": -1.0, "pixels": 4}}
        columns = {
            "pearson": Column(
                kind="number",
                unit="correlation",
                description="centered Pearson correlation within region",
                atol=1e-8,
                rtol=1e-8,
            ),
            "pixels": Column(kind="integer", unit="pixels", description="region pixel count"),
        }
        prompt = (
            "Measure channel colocalization as the centered Pearson correlation of channel_a.pgm and "
            "channel_b.pgm pixels separately within every nonzero regions.pgm label. Include every "
            "pixel in a region, without thresholding or treating spatial coordinates as intensities. "
            "Use region label id."
        )
        wrong = [{"id": k, **v, "pearson": abs(v["pearson"])} for k, v in expected.items()]
        reason = "lost_negative_colocalization"
    elif operation == "spatial-neighbor-enrichment":
        scale = rng.choice([1, 2, 3])
        rows = [
            {"cell": f"c{i}", "x": x * scale, "y": y * scale, "type": kind}
            for i, (x, y, kind) in enumerate([(0, 0, "A"), (1, 0, "A"), (2, 0, "B"), (8, 0, "B"), (9, 0, "A")])
        ]
        rng.shuffle(rows)
        inputs = {"cells.csv": csv_text(rows), "radius_um.txt": str(scale) + "\n"}
        # Edges: c0-c1, c1-c2, c3-c4. Directed A->B=2 of A outgoing=4.
        expected = {
            "A:A": {"edges": 2, "fraction": 0.5},
            "A:B": {"edges": 2, "fraction": 0.5},
            "B:A": {"edges": 2, "fraction": 1.0},
            "B:B": {"edges": 0, "fraction": 0.0},
        }
        columns = {
            "edges": Column(
                kind="integer",
                unit="directed neighbor pairs",
                description="source-type to target-type edges excluding self",
            ),
            "fraction": Column(
                kind="number",
                unit="fraction",
                description="fraction among outgoing edges of source type",
                atol=1e-8,
                rtol=1e-8,
            ),
        }
        prompt = (
            "Build a radius-neighbor graph from physical x/y micrometer coordinates in cells.csv, "
            "connecting distinct cells at Euclidean distance <= radius_um.txt. Count both directions of"
            " each neighboring pair. For every ordered type pair, report edge count and its fraction of"
            " all outgoing edges from that source type. This is an observed mixing summary, not a "
            "permutation significance test. Use source_type:target_type."
        )
        wrong = [{"id": k, **v, "edges": v["edges"] + 1 if k == "A:A" else v["edges"]} for k, v in expected.items()]
        reason = "included_self_neighbors"
    elif operation == "spatial-region-counts":
        shift = rng.randint(0, 10)
        cells = [("c0", shift, 0, "A"), ("c1", shift + 2, 1, "B"), ("c2", shift + 4, 1, "A"), ("c3", shift + 1, 2, "B")]
        regions = [("left", shift, shift + 2, 0, 2), ("right", shift + 2, shift + 4, 0, 2)]
        inputs = {
            "cells.csv": csv_text([{"cell": name, "x": x, "y": y, "type": kind} for name, x, y, kind in cells]),
            "regions.csv": csv_text(
                [{"region": name, "xmin": a, "xmax": b, "ymin": c, "ymax": d} for name, a, b, c, d in regions]
            ),
        }
        expected = {"left:A": {"count": 1}, "left:B": {"count": 0}, "right:A": {"count": 0}, "right:B": {"count": 1}}
        columns = {
            "count": Column(kind="integer", unit="cells", description="cells of requested type in half-open rectangle")
        }
        prompt = (
            "Count cells.csv cell types in rectangular regions.csv regions using xmin <= x < xmax and "
            "ymin <= y < ymax in micrometers. Exclude upper/right boundaries; a point on a shared "
            "boundary belongs only to the region whose lower boundary it matches. Return every "
            "region:type combination, including zeros. Use region:type id."
        )
        wrong = [{"id": k, "count": 1} for k in expected]
        reason = "double_counted_shared_or_upper_boundaries"
    else:
        assert operation == "image-dice-iou"
        dx = rng.randint(0, 2)
        overlap = rng.randint(1, 3)
        reference = [[0] * 6 for _ in range(4)]
        predicted = [[0] * 6 for _ in range(4)]
        for y, x in [(0, dx), (0, dx + 1), (1, dx), (1, dx + 1)]:
            reference[y][x] = 1
        inside = [(0, dx), (0, dx + 1), (1, dx), (1, dx + 1)]
        outside = [(2, dx), (2, dx + 1), (3, dx)]
        for y, x in inside[:overlap] + outside[: 4 - overlap]:
            predicted[y][x] = 1
        inputs = {"reference.pgm": pgm_text(reference), "prediction.pgm": pgm_text(predicted), "labels.txt": "1\n2\n"}
        expected = {
            "1": {"intersection": overlap, "dice": overlap / 4, "iou": overlap / (8 - overlap)},
            "2": {"intersection": 0, "dice": 1.0, "iou": 1.0},
        }
        columns = {
            "intersection": Column(kind="integer", unit="pixels", description="shared pixels with this label"),
            "dice": Column(
                kind="number",
                unit="fraction",
                description="2 intersection / sum of mask sizes; both empty=1",
                atol=1e-8,
                rtol=1e-8,
            ),
            "iou": Column(
                kind="number", unit="fraction", description="intersection / union; both empty=1", atol=1e-8, rtol=1e-8
            ),
        }
        prompt = (
            "Compare prediction.pgm to reference.pgm independently for every label in labels.txt. "
            "Report intersection pixel count, Dice, and IoU. The background label 0 is excluded. Define"
            " Dice and IoU as 1 when both label masks are empty; do not average over background pixels."
            " Use label string id."
        )
        wrong = [{"id": k, **v, "dice": v["iou"]} for k, v in expected.items()]
        reason = "confused_dice_with_iou"
    return Instance(
        prompt + " Inputs are in /app/inputs.", inputs, Contract(columns=columns, expected=expected), {reason: wrong}
    )


SKILLS = {
    "image-threshold-components": ("connected-components", "pixel-centers", "physical-area"),
    "image-background-correction": ("background-estimation", "label-masks", "signed-intensities"),
    "image-colocalization": ("regionwise-correlation", "channel-pairing", "centering"),
    "spatial-neighbor-enrichment": ("physical-neighbors", "mixing-fractions", "self-exclusion"),
    "spatial-region-counts": ("spatial-boundaries", "type-counts", "zero-combinations"),
    "image-dice-iou": ("segmentation-overlap", "empty-masks", "foreground-denominator"),
}
RECIPES = tuple(
    Recipe(
        name,
        "1",
        Difficulty.MEDIUM,
        skills,
        (
            ("csv-header",)
            if name.startswith("spatial")
            else ("pgm-p2", "json-metadata") if name == "image-threshold-components" else ("pgm-p2",)
        ),
        ("https://netpbm.sourceforge.net/doc/pgm.html", "https://scikit-image.org/docs/stable/api/skimage.measure.html"),
        partial(generate_imaging, operation=name),
    )
    for name, skills in SKILLS.items()
)
