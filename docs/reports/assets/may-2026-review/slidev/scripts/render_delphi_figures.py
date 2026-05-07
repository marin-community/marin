"""Render Delphi blog Plotly JSON figures as static PNGs for the slidev deck.

Usage:
    uv run --with plotly --with kaleido scripts/render_delphi_figures.py

Reads from ~/open-athena.github.io/static/assets/images/blog/delphi/<name>.json
and writes ~/marin-slidev/.../slidev/public/charts/delphi/<name>.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.io as pio

BLOG_FIGURES_DIR = (
    Path.home()
    / "open-athena.github.io"
    / "static"
    / "assets"
    / "images"
    / "blog"
    / "delphi"
)
OUT_DIR = Path(__file__).resolve().parent.parent / "public" / "charts" / "delphi"

FIGURES = [
    ("delphi-ladder", 1500, 600),
    ("mmlu-emergence", 1600, 720),
    ("lucky-seeds", 1400, 520),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, width, height in FIGURES:
        src = BLOG_FIGURES_DIR / f"{name}.json"
        dst = OUT_DIR / f"{name}.png"
        fig = pio.from_json(src.read_text())
        fig.write_image(dst, width=width, height=height, scale=2)
        print(f"wrote {dst}")


if __name__ == "__main__":
    main()
