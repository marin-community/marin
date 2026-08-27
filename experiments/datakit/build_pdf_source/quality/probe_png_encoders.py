# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Which PNG encoder should the VLM feed use? The renderer is not the expensive half.

:mod:`~experiments.datakit.build_pdf_source.quality.probe_pdf_oxide` set out to price a different
rasteriser and found that rasterising is not where the feed's ~17.8 CPU core-hours per million pages
go. Splitting :mod:`~experiments.datakit.build_pdf_source.ocr_extract.render`'s three stages, the PNG
encode is roughly three quarters of the cost and ``get_pixmap`` is roughly a quarter.

That makes the encoder the lever, and a uniquely safe one. Every other way to make the feed cheaper
-- a different rasteriser, a smaller token budget, a lossy format -- changes the pixels the VLM sees
and so invalidates the extraction-quality numbers the pipeline already has. PNG is lossless, so a
different PNG encoder at the same bit depth is *provably* the same image: this module asserts that
per page by decoding both outputs and comparing bytes, and reports the count rather than assuming it.

Four candidates, all against MuPDF's own ``Pixmap.tobytes("png")``:

``mupdf_png``
    The incumbent.
``pillow_c1`` / ``pillow_c6``
    Pillow over the same pixmap buffer at zlib compression level 1 and 6. Level is the whole trade:
    it moves encode time against payload size, and the payload is not free either -- the feed base64s
    it and ships it to the serving pods, whose API-side CPU the budget sweep found to be what sets
    throughput.
``mupdf_jpeg``
    Included to close it off rather than because it is expected to win. It is lossy, so it would
    need a quality re-evaluation, and it is measured here only to establish whether it is even fast.

Timing is around the encode call alone, on a pixmap rendered beforehand at the production budget, so
neither the parquet read nor the render appears in it. Run on both architectures -- zlib's SIMD paths
differ between Emerald Rapids and Grace, and the feed runs on whichever pool has room:

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name png-encoders-x86 --extra pdf \\
        --cpu 8 --memory 32GB --disk 32GB --enable-extra-resources \\
        -- python -m experiments.datakit.build_pdf_source.quality.probe_png_encoders
"""

import io
import logging
import platform
import time
from dataclasses import dataclass
from importlib.metadata import version

import polars as pl

from experiments.datakit.build_pdf_source.ocr_extract.render import RenderOptions, target_dimensions
from experiments.datakit.build_pdf_source.quality.probe_pdf_inspector import probe_documents, storage

logger = logging.getLogger(__name__)

BUCKET = "marin-us-east-02a"
OUTPUT_PREFIX = f"s3://{BUCKET}/marin/data/pdf_quality/cc_focus_2026_22_png_encoder_probe"

# Pages encoded per document. The feed encodes every page; a per-page cost needs only enough pages
# to be representative, and the tail of the page distribution would otherwise dominate the run.
ENCODE_PAGES = 4
PILLOW_LEVELS = (1, 6)
JPEG_QUALITY = 85

RENDER_OPTIONS = RenderOptions()


@dataclass
class Totals:
    """Running cost and payload for one encoder."""

    seconds: float = 0.0
    encoded_bytes: int = 0
    identical: int = 0
    pages: int = 0

    def add(self, seconds: float, size: int, identical: bool) -> None:
        self.seconds += seconds
        self.encoded_bytes += size
        self.identical += identical
        self.pages += 1


def _core_hours_per_million(milliseconds_per_page: float) -> float:
    return 1e6 * milliseconds_per_page / 1000.0 / 3600.0


def _decoded(png: bytes) -> bytes:
    from PIL import Image  # noqa: PLC0415

    return Image.open(io.BytesIO(png)).convert("RGB").tobytes()


def encode_document(pdf: bytes, totals: dict[str, Totals]) -> int:
    """Every encoder over the same pixmaps, timed around the encode call alone."""
    import pymupdf  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    document = pymupdf.open(stream=pdf, filetype="pdf")
    encoded = 0
    try:
        for index in range(min(len(document), ENCODE_PAGES)):
            page = document[index]
            width, height = page.rect.width, page.rect.height
            if width < 1 or height < 1:
                continue
            target_height, target_width = target_dimensions(width, height, RENDER_OPTIONS)
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(target_width / width, target_height / height))

            started = time.perf_counter()
            reference = pixmap.tobytes("png")
            totals["mupdf_png"].add(time.perf_counter() - started, len(reference), True)
            reference_pixels = _decoded(reference)

            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            for level in PILLOW_LEVELS:
                buffer = io.BytesIO()
                started = time.perf_counter()
                image.save(buffer, format="PNG", compress_level=level)
                elapsed = time.perf_counter() - started
                payload = buffer.getvalue()
                totals[f"pillow_c{level}"].add(elapsed, len(payload), _decoded(payload) == reference_pixels)

            started = time.perf_counter()
            jpeg = pixmap.tobytes("jpg", jpg_quality=JPEG_QUALITY)
            # Lossy by construction; the identity column is there to show it is not a PNG substitute.
            totals["mupdf_jpeg"].add(time.perf_counter() - started, len(jpeg), _decoded(jpeg) == reference_pixels)
            encoded += 1
    finally:
        document.close()
    return encoded


def main() -> None:
    from rigging.log_setup import configure_logging  # noqa: PLC0415

    configure_logging(logging.INFO)
    logger.info("pymupdf %s, pillow %s, on %s", version("pymupdf"), version("pillow"), platform.machine())

    filesystem = storage()
    documents = probe_documents(filesystem)

    names = ["mupdf_png", *(f"pillow_c{level}" for level in PILLOW_LEVELS), "mupdf_jpeg"]
    totals = {name: Totals() for name in names}
    failures = 0
    for position, document in enumerate(documents.iter_rows(named=True)):
        try:
            encode_document(document["pdf"], totals)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            failures += 1
            logger.debug("Could not encode %s", document["url"], exc_info=True)
        if (position + 1) % 200 == 0:
            logger.info("encoders: %d/%d documents", position + 1, documents.height)

    baseline = totals["mupdf_png"]
    rows = []
    for name in names:
        entry = totals[name]
        if not entry.pages:
            continue
        milliseconds = 1000 * entry.seconds / entry.pages
        rows.append(
            {
                "encoder": name,
                "pages": entry.pages,
                "ms_per_page": milliseconds,
                "core_hours_per_million": _core_hours_per_million(milliseconds),
                "kib_per_page": entry.encoded_bytes / entry.pages / 1024,
                "pixel_identical": entry.identical,
                "speedup": baseline.seconds / entry.seconds if entry.seconds else float("nan"),
            }
        )
        logger.info(
            "%-11s %7.2f ms/page  %6.2f core-h/M  %6.0f KiB/page  pixel-identical %d/%d  %.2fx",
            name,
            milliseconds,
            _core_hours_per_million(milliseconds),
            entry.encoded_bytes / entry.pages / 1024,
            entry.identical,
            entry.pages,
            baseline.seconds / entry.seconds if entry.seconds else float("nan"),
        )
    logger.info("encoders: %d documents unencodable", failures)

    frame = pl.DataFrame(rows)
    output = f"{OUTPUT_PREFIX}/{platform.machine()}.parquet"
    with filesystem.open(output, "wb") as stream:
        frame.write_parquet(stream, compression="zstd", compression_level=1)
    logger.info("encoders: wrote %d rows -> %s", len(rows), output)


if __name__ == "__main__":
    main()
