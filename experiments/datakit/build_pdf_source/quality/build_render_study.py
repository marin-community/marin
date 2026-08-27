# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Does swapping MuPDF for PDFium change what the VLM reads off the page?

This is the gate on the renderer swap, and the reason the swap is worth evaluating at all is
licensing: PyMuPDF is AGPL, PDFium is BSD-3-Clause and ``pypdfium2`` is Apache-2.0/BSD, and after
the router pass and the Docling route are removed, rendering is the only thing PyMuPDF still does in
this pipeline. :mod:`~experiments.datakit.build_pdf_source.quality.probe_pdfium` prices both
rasterisers and measures how far apart their pixels are. Pixels being different is certain --
two rasterisers always draw glyphs differently -- and by itself it decides nothing. What decides it
is whether the *model's reading* changes.

So: the same page, rendered both ways at identical ``smart_resize`` dimensions and encoded by the
identical PNG encoder, through the same model with the same prompt at temperature 0, compared with
:mod:`~experiments.datakit.build_pdf_source.quality.route_agreement` -- the same format-normalised,
bigram-led machinery every other comparison in this evaluation uses.

**The control is the whole design.** A bigram agreement of 0.93 between two renderings means nothing
without knowing what the model scores against *itself*. vLLM's greedy decoding is not bit-reproducible
across batch compositions, so two requests carrying byte-identical images do not return identical
text, and on a dense or damaged page they can diverge substantially. Every page therefore goes
through the endpoint **three** times: the MuPDF rendering twice and the PDFium rendering once. The
MuPDF-against-MuPDF pair is the noise floor; the MuPDF-against-PDFium pair is the treatment; and the
result is the paired difference between them, not the treatment's absolute value.

**Divergence is characterised, not just counted.** Each page carries the cheap facts that would
explain a difference if there is one -- how much of it is bitmap, how many characters it has, how
small its glyphs are, how much vector line art it carries, whether it is CJK, and what effective DPI
the budget gave it -- so "where do they differ" is answered from the data rather than by inspecting
a handful of images.

**Rendering happens in a subprocess the driver is willing to lose.** This is not caution: an earlier
revision rendered in-process and the job died at exit 133 -- SIGTRAP, which is what PDFium's
``IMMEDIATE_CRASH`` compiles to on aarch64 -- a hundred documents in, taking the fleet's endpoint
with it. A hard abort is not catchable in the process it happens in, so a study that renders
in-process cannot finish a crawl sample. It reuses
:class:`~experiments.datakit.build_pdf_source.quality.probe_pdf_inspector.Worker`, and the document
that killed a worker is counted rather than silently lost.

**The serve is not production's fleet**, and :func:`study_inference_config` says why in detail. The
short version is that the study needs the same model reading the same prompt at temperature 0, not
the throughput operating point, and both arms go through whatever engine it gets -- so the paired
difference the study reports is unaffected by the difference.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \\
        --job-name pdfium-render-study --extra pdf \\
        --cpu 8 --memory 32GB --disk 32GB --enable-extra-resources \\
        -- python -m experiments.datakit.build_pdf_source.quality.build_render_study
"""

import base64
import io
import json
import logging
import os
import platform
import sys
import time
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from importlib.metadata import version

import numpy as np
import polars as pl
from fray.types import ANY_REGION, ResourceConfig, create_environment
from marin.inference.config import (
    IrisConfig,
    RemoteInferenceConfig,
    ServedModelConfig,
    VllmEngineConfig,
    VllmLauncherType,
)
from marin.inference.iris import remote_inference

from experiments.datakit.build_pdf_source.ocr_extract.client import OcrEndpoint, PageOcr, ocr_page
from experiments.datakit.build_pdf_source.ocr_extract.fleet import (
    GPU_TYPE,
    GPU_WORKER_CPU,
    GPU_WORKER_RAM_GB,
    MAX_MODEL_LEN,
    MODEL,
)
from experiments.datakit.build_pdf_source.ocr_extract.render import (
    RenderedPage,
    RenderOptions,
    effective_dpi,
    target_dimensions,
)
from experiments.datakit.build_pdf_source.quality.probe_pdf_inspector import (
    ERROR_SAMPLES,
    Outcome,
    Worker,
    failure_outcome,
    probe_documents,
    read_exactly,
    storage,
)
from experiments.datakit.build_pdf_source.quality.probe_pdfium import (
    PILLOW_COMPRESS_LEVEL,
    PIXEL_TOLERANCE,
    render_pdfium,
    sampled_page_indices,
)
from experiments.datakit.build_pdf_source.quality.route_agreement import VLM, page_agreement

logger = logging.getLogger(__name__)

BUCKET = "marin-us-east-02a"
OUTPUT_PREFIX = f"s3://{BUCKET}/marin/data/pdf_quality/cc_focus_2026_22_pdfium_render_study"

MODULE_NAME = "experiments.datakit.build_pdf_source.quality.build_render_study"
WORKER_FLAG = "--worker"
RENDER_OP = "render"
# Rendering four full-resolution buffers and encoding them is seconds, not minutes. Generous enough
# that a slow document is not called a hang, tight enough that a real hang does not stall the run.
RENDER_TIMEOUT = 120.0

# Pages sampled per document, spread evenly rather than taken from the front: a document's first
# page is a cover more often than it is representative.
STUDY_PAGES = 2
# In-flight requests. Well under the engine's slot budget, because the single-threaded render loop
# cannot offer more than this anyway and a deeper pool only buys queueing.
REQUEST_THREADS = 64
# Engine slots. Production runs 1024 against 512 in flight; this serve is offered a fraction of that
# and a smaller number leaves more HBM for the KV cache it does use.
MAX_NUM_SEQS = 256
MAX_NUM_BATCHED_TOKENS = 131_072
# An hour. The weights are ~30 GB and a cold node fetches them before it can answer.
STARTUP_TIMEOUT = 3600
# Rendered pages held at once. Each carries two ~2.5 MB data URIs and three requests in flight.
PAGES_IN_FLIGHT = 2 * REQUEST_THREADS

# Thresholds for the divergence buckets. Chosen to name recognisable page kinds rather than to split
# the sample evenly; each is reported with its own page count so a thin bucket is visible as one.
SCAN_IMAGE_AREA = 0.8
SCAN_MAX_CHARS = 100
LINE_ART_DRAWINGS = 100
SMALL_GLYPH_POINTS = 8.0
CJK_RATIO = 0.1
DENSE_TEXT_CHARS = 3000
DIVERGENT_PIXEL_FRACTION = 0.10

RENDER_OPTIONS = RenderOptions()

_CJK_RANGES = (
    (0x3040, 0x30FF),  # kana
    (0x3400, 0x4DBF),  # CJK extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xAC00, 0xD7AF),  # hangul syllables
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
)


def study_inference_config() -> RemoteInferenceConfig:
    """One vLLM engine on one GPU, served directly rather than through the broker.

    Deliberately *not*
    :func:`~experiments.datakit.build_pdf_source.ocr_extract.fleet.build_inference_config`. That
    config is the throughput operating point the budget sweep measured -- four engines packed to a
    GB200 node behind a broker at 2048 total in flight -- and it reaches it through serving-side
    options (``VllmEngineConfig.uv_with_packages`` for the prebuilt FlashInfer kernel artifacts)
    that exist only alongside the pipeline work. A study that needs 6,000 requests answered has no
    use for any of that, and tying it to those options would mean it could only run from one branch.

    What the study *does* need is the identical model reading the identical prompt at temperature 0,
    and it gets that. The engine differs from production's in how it computes the gated-delta-net
    prefill -- Triton here, FlashInfer there -- which is a kernel choice, not a model change; both
    arms of the comparison go through whichever one this serve has, so the paired difference the
    study reports is unaffected. Absolute agreement levels would move slightly under a different
    backend, which is exactly why the control arm exists.

    ``instances=1`` with no broker takes ``remote_inference``'s direct path: one engine, one
    endpoint, no proxy in between, and one less thing that can be the reason a run failed.
    """
    return RemoteInferenceConfig(
        model=ServedModelConfig(weights=MODEL, max_model_len=MAX_MODEL_LEN, tensor_parallel_size=1),
        engine=VllmEngineConfig(
            launcher=VllmLauncherType.CUDA,
            startup_timeout_seconds=STARTUP_TIMEOUT,
            max_num_seqs=MAX_NUM_SEQS,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            extra_args=(
                # The hybrid's GDN prefill kernel, from Triton rather than FlashInfer: FlashInfer
                # needs prebuilt cubins the CoreWeave runtime images cannot compile, and reaching
                # them needs the fleet options this config exists to avoid.
                "--gdn-prefill-backend",
                "triton",
                "--reasoning-parser",
                "qwen3",
            ),
        ),
        iris=IrisConfig(
            worker_resources=ResourceConfig.with_gpu(
                GPU_TYPE,
                count=1,
                cpu=GPU_WORKER_CPU,
                ram=f"{GPU_WORKER_RAM_GB}g",
                disk="300g",
                regions=[ANY_REGION],
            ),
            worker_environment=create_environment(),
            endpoint_ready_timeout_seconds=float(STARTUP_TIMEOUT),
            max_retries_failure=3,
        ),
        instances=1,
    )


# ---------------------------------------------------------------------------
# Worker: rendering, in a process the driver is willing to lose
# ---------------------------------------------------------------------------


def page_facts(page, dpi: float) -> dict:
    """One page's characterising facts, from a single text pass plus a drawings pass."""
    text = page.get_text("text")
    cjk = sum(1 for char in text if any(low <= ord(char) <= high for low, high in _CJK_RANGES))
    area = page.rect.width * page.rect.height
    blocks = page.get_text("dict")["blocks"]
    image_area = sum(
        max(0.0, block["bbox"][2] - block["bbox"][0]) * max(0.0, block["bbox"][3] - block["bbox"][1])
        for block in blocks
        if block["type"] == 1
    )
    sizes = [span["size"] for block in blocks if block["type"] == 0 for line in block["lines"] for span in line["spans"]]
    return {
        "dpi": dpi,
        "image_area_fraction": min(1.0, image_area / area) if area > 0 else 0.0,
        "char_count": len(text),
        "cjk_ratio": cjk / len(text) if text else 0.0,
        "drawing_count": len(page.get_cdrawings()),
        "median_font_size": float(np.median(sizes)) if sizes else 0.0,
    }


def _data_uri(samples: np.ndarray) -> str:
    from PIL import Image  # noqa: PLC0415

    buffer = io.BytesIO()
    Image.fromarray(samples).save(buffer, format="PNG", compress_level=PILLOW_COMPRESS_LEVEL)
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def _render_document(payload: bytes) -> dict:
    """Every sampled page of one document, rendered both ways onto identical dimensions.

    The two buffers are compared here rather than in the driver so full-resolution pixels never have
    to cross the pipe; what crosses is two PNG data URIs per page and the numbers describing them.
    """
    import pymupdf  # noqa: PLC0415
    import pypdfium2 as pdfium  # noqa: PLC0415

    other = pdfium.PdfDocument(payload)
    document = pymupdf.open(stream=payload, filetype="pdf")
    try:
        pages = []
        for index in sampled_page_indices(min(len(document), len(other)), STUDY_PAGES):
            page = document[index]
            page_width, page_height = page.rect.width, page.rect.height
            if page_width < 1 or page_height < 1:
                continue
            height, width = target_dimensions(page_width, page_height, RENDER_OPTIONS)
            dpi = effective_dpi(height * width, page_width, page_height)

            matrix = pymupdf.Matrix(width / page_width, height / page_height)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            left = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
            right = render_pdfium(other[index], height, width)

            difference = np.abs(left.astype(np.int16) - right.astype(np.int16)).max(axis=2)
            pages.append(
                {
                    "page_index": index,
                    "height": height,
                    "width": width,
                    "changed_fraction": float((difference > PIXEL_TOLERANCE).mean()),
                    "mean_absolute_difference": float(difference.mean()),
                    "mupdf_ink": float((left.min(axis=2) < 128).mean()),
                    "pdfium_ink": float((right.min(axis=2) < 128).mean()),
                    **page_facts(page, dpi),
                    "mupdf_uri": _data_uri(left),
                    "pdfium_uri": _data_uri(right),
                }
            )
        return {"pages": pages, "document_pages": len(document)}
    finally:
        document.close()
        other.close()


def _measure(payload: bytes) -> dict:
    """Render one document, reporting whatever happened instead of raising."""
    try:
        return {"outcome": str(Outcome.OK), **_render_document(payload)}
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as error:
        return {"outcome": str(failure_outcome(error)), "error": f"{type(error).__name__}: {error}"[:300]}


def worker_main() -> None:
    """Serve length-prefixed documents from stdin until the driver closes it.

    The reply channel is a duplicate of the inherited stdout and file descriptor 1 is then pointed
    at stderr, because MuPDF writes its content-stream warnings to fd 1 by default and would
    otherwise land them in the middle of the length-prefixed protocol.
    """
    import faulthandler  # noqa: PLC0415

    import pymupdf  # noqa: PLC0415

    faulthandler.enable()
    replies = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    # A crawl sample produces thousands of these and none of them is this study's subject.
    pymupdf.TOOLS.mupdf_display_errors(False)

    stdin = sys.stdin.buffer
    while True:
        header = stdin.readline()
        if not header:
            return
        request = json.loads(header)
        payload = read_exactly(stdin, request["size"])
        replies.write(json.dumps(_measure(payload)).encode() + b"\n")
        replies.flush()


# ---------------------------------------------------------------------------
# Driver: three readings of every page
# ---------------------------------------------------------------------------


def _f1(recall: float, precision: float) -> float:
    return 2 * recall * precision / (recall + precision) if recall + precision else 0.0


def _agreement_columns(agreement) -> dict:
    return {
        "unigram_recall": agreement.unigram_recall,
        "unigram_precision": agreement.unigram_precision,
        "unigram_f1": _f1(agreement.unigram_recall, agreement.unigram_precision),
        "bigram_recall": agreement.bigram_recall,
        "bigram_precision": agreement.bigram_precision,
        "bigram_f1": _f1(agreement.bigram_recall, agreement.bigram_precision),
        "reference_tokens": agreement.reference_tokens,
        "candidate_tokens": agreement.candidate_tokens,
    }


@dataclass
class StudyPage:
    """One page rendered both ways, with the three readings it is waiting on."""

    row: dict
    mupdf_first: "Future[PageOcr]"
    mupdf_second: "Future[PageOcr]"
    pdfium: "Future[PageOcr]"

    def resolve(self) -> dict:
        """Fold the three answers into the row, keeping a failed request as a marked row."""
        try:
            first, second, other = (
                self.mupdf_first.result(),
                self.mupdf_second.result(),
                self.pdfium.result(),
            )
        except Exception as error:
            return {**self.row, "request_error": f"{type(error).__name__}: {error}"[:300]}

        control = page_agreement(first.text, second.text, VLM, VLM)
        treatment = page_agreement(first.text, other.text, VLM, VLM)
        return {
            **self.row,
            "request_error": None,
            "mupdf_chars": len(first.text),
            "pdfium_chars": len(other.text),
            "mupdf_completion_tokens": first.completion_tokens,
            "pdfium_completion_tokens": other.completion_tokens,
            "mupdf_truncated": first.truncated,
            "pdfium_truncated": other.truncated,
            **{f"control_{key}": value for key, value in _agreement_columns(control).items()},
            **{f"treatment_{key}": value for key, value in _agreement_columns(treatment).items()},
        }


def run_study(documents: pl.DataFrame, endpoint: OcrEndpoint) -> list[dict]:
    """Render every sampled page in the worker and keep the model's three readings of it.

    Rendering and waiting overlap: a page's three requests are submitted the moment the worker hands
    it back, and the next document starts rendering while the previous one is still in flight, with
    the in-flight bound the only thing that blocks. Pages resolve in submission order, so the output
    is in document order without tracking indices.
    """
    worker = Worker(MODULE_NAME, timeout=RENDER_TIMEOUT)
    pool = ThreadPoolExecutor(max_workers=REQUEST_THREADS, thread_name_prefix="render-study")
    inflight: deque[StudyPage] = deque()
    rows: list[dict] = []
    outcomes: Counter = Counter()
    errors: Counter = Counter()

    def send(uri: str, index: int, dpi: float, pixels: int) -> "Future[PageOcr]":
        page = RenderedPage(data_uri=uri, page_index=index, pixels=pixels, dpi=dpi)
        return pool.submit(ocr_page, endpoint, REQUEST_THREADS, page)

    try:
        for position, document in enumerate(documents.iter_rows(named=True)):
            reply = worker.call(RENDER_OP, document["pdf"])
            outcomes[reply.result["outcome"]] += 1
            if reply.result["outcome"] != Outcome.OK:
                errors[reply.result.get("error", "worker lost")] += 1
                continue
            for page in reply.result["pages"]:
                while len(inflight) >= PAGES_IN_FLIGHT:
                    rows.append(inflight.popleft().resolve())
                pixels = page["height"] * page["width"]
                mupdf_uri, pdfium_uri = page.pop("mupdf_uri"), page.pop("pdfium_uri")
                inflight.append(
                    StudyPage(
                        row={"source_id": document["source_id"], "url": document["url"], **page},
                        mupdf_first=send(mupdf_uri, page["page_index"], page["dpi"], pixels),
                        mupdf_second=send(mupdf_uri, page["page_index"], page["dpi"], pixels),
                        pdfium=send(pdfium_uri, page["page_index"], page["dpi"], pixels),
                    )
                )
            if (position + 1) % 100 == 0:
                logger.info(
                    "study: %d/%d documents, %d pages resolved, %d worker spawns",
                    position + 1,
                    documents.height,
                    len(rows),
                    worker.spawns,
                )
        while inflight:
            rows.append(inflight.popleft().resolve())
    finally:
        worker.stop()
        pool.shutdown(wait=True)

    logger.info("study: %d pages over %d documents, %d worker spawns", len(rows), documents.height, worker.spawns)
    for outcome, count in outcomes.most_common():
        logger.info("study: render %-12s %5d  %6.2f%%", outcome, count, 100 * count / documents.height)
    for message, count in errors.most_common(ERROR_SAMPLES):
        logger.info("study: render failure x%d: %s", count, message)
    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

_BUCKETS = {
    "scanned": pl.col("image_area_fraction").ge(SCAN_IMAGE_AREA) & pl.col("char_count").lt(SCAN_MAX_CHARS),
    "born_digital": pl.col("image_area_fraction").lt(SCAN_IMAGE_AREA),
    "cjk": pl.col("cjk_ratio").gt(CJK_RATIO),
    "line_art": pl.col("drawing_count").ge(LINE_ART_DRAWINGS),
    "small_glyphs": pl.col("median_font_size").gt(0).and_(pl.col("median_font_size").lt(SMALL_GLYPH_POINTS)),
    "dense_text": pl.col("char_count").ge(DENSE_TEXT_CHARS),
    "below_legibility_floor": pl.col("dpi").lt(RENDER_OPTIONS.legibility_floor_dpi),
    "most_divergent_pixels": pl.col("changed_fraction").ge(DIVERGENT_PIXEL_FRACTION),
}


def _report(frame: pl.DataFrame, label: str) -> None:
    if not frame.height:
        logger.info("%-24s no pages", label)
        return
    row = frame.select(
        pl.col("control_bigram_f1").mean().alias("control"),
        pl.col("treatment_bigram_f1").mean().alias("treatment"),
        (pl.col("treatment_bigram_f1") - pl.col("control_bigram_f1")).mean().alias("delta"),
        (pl.col("treatment_bigram_f1") - pl.col("control_bigram_f1")).std().alias("delta_std"),
        pl.col("changed_fraction").median().alias("changed"),
    ).row(0, named=True)
    # The paired standard error is what says whether a delta is a finding: the two readings of a
    # page share every property of that page, so the pairing removes almost all of the variance
    # that makes the absolute agreement numbers look noisy.
    stderr = (row["delta_std"] or 0.0) / float(np.sqrt(frame.height))
    logger.info(
        "%-24s n=%5d  control %.4f  treatment %.4f  paired delta %+.4f +/- %.4f  changed px %.4f",
        label,
        frame.height,
        row["control"],
        row["treatment"],
        row["delta"],
        stderr,
        row["changed"],
    )


def summarize(frame: pl.DataFrame) -> None:
    failed = frame.filter(pl.col("request_error").is_not_null())
    logger.info("study: %d pages, %d with a failed request", frame.height, failed.height)
    for error, count in failed["request_error"].value_counts().rows():
        logger.info("study: request error x%d: %s", count, error)
    good = frame.filter(pl.col("request_error").is_null())
    if not good.height:
        logger.warning("study: no page has all three readings")
        return

    logger.info("--- bigram F1: MuPDF vs itself (control) against MuPDF vs PDFium (treatment) ---")
    _report(good, "all pages")
    for name, predicate in _BUCKETS.items():
        _report(good.filter(predicate), name)

    logger.info("--- the pages where the treatment falls furthest below the control ---")
    worst = good.with_columns((pl.col("treatment_bigram_f1") - pl.col("control_bigram_f1")).alias("delta")).sort("delta")
    for row in worst.head(15).iter_rows(named=True):
        logger.info(
            "delta %+.3f (control %.3f) changed px %.3f dpi %.0f chars %5d image %.2f drawings %4d font %.1f  %s p%d",
            row["delta"],
            row["control_bigram_f1"],
            row["changed_fraction"],
            row["dpi"],
            row["char_count"],
            row["image_area_fraction"],
            row["drawing_count"],
            row["median_font_size"],
            row["url"][:80],
            row["page_index"],
        )

    logger.info("--- output length, unigram agreement, and ink ---")
    logger.info(
        "chars/page MuPDF %.0f vs PDFium %.0f | completion tokens %.0f vs %.0f | unigram F1 control %.4f "
        "treatment %.4f",
        good["mupdf_chars"].mean(),
        good["pdfium_chars"].mean(),
        good["mupdf_completion_tokens"].mean(),
        good["pdfium_completion_tokens"].mean(),
        good["control_unigram_f1"].mean(),
        good["treatment_unigram_f1"].mean(),
    )
    logger.info(
        "truncated pages: MuPDF %d, PDFium %d | ink MuPDF %.4f vs PDFium %.4f | mean |delta| %.2f/255",
        good["mupdf_truncated"].sum(),
        good["pdfium_truncated"].sum(),
        good["mupdf_ink"].mean(),
        good["pdfium_ink"].mean(),
        good["mean_absolute_difference"].mean(),
    )


def main() -> None:
    import pypdfium2 as pdfium  # noqa: PLC0415
    from rigging.log_setup import configure_logging  # noqa: PLC0415

    configure_logging(logging.INFO)
    logger.info(
        "pypdfium2 %s (pdfium %s), pymupdf %s, on %s, python %s",
        version("pypdfium2"),
        pdfium.PDFIUM_INFO,
        version("pymupdf"),
        platform.machine(),
        sys.version,
    )

    filesystem = storage()
    documents = probe_documents(filesystem)

    started = time.perf_counter()
    with remote_inference(study_inference_config()) as session:
        endpoint = OcrEndpoint(
            base_url=session.model.endpoint.base_url,
            model=session.model.endpoint.model,
            max_visual_tokens=RENDER_OPTIONS.max_visual_tokens,
        )
        logger.info("study: endpoint ready at %s (%s)", endpoint.base_url, session.backend_name)
        rows = run_study(documents, endpoint)
    logger.info("study: %.1f minutes with the fleet up", (time.perf_counter() - started) / 60)

    frame = pl.DataFrame(rows, strict=False, infer_schema_length=None)
    output = f"{OUTPUT_PREFIX}/{platform.machine()}-{version('pypdfium2')}.parquet"
    with filesystem.open(output, "wb") as stream:
        frame.write_parquet(stream, compression="zstd", compression_level=1)
    logger.info("study: wrote %d rows -> %s", frame.height, output)

    summarize(frame)


if __name__ == "__main__":
    if WORKER_FLAG in sys.argv:
        worker_main()
    else:
        main()
