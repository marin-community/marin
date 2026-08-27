# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""When the VLM reads a PDFium rendering differently, is that reading *worse* or merely different?

:mod:`~experiments.datakit.build_pdf_source.quality.build_render_study` established that the two
renderings produce materially different model output -- a paired bigram-F1 delta of -0.0364 against
a 0.9954 self-agreement control, with 2.1% of pages read as something else entirely. It could not
establish *direction*. Bigram recall (0.9622) and precision (0.9629) moved symmetrically, which is
what two different-but-equivalent readings look like, and
[`pdfium-evaluation.md`](../pdfium-evaluation.md) says so explicitly: "it does **not** establish that
PDFium is worse". This module and
:mod:`~experiments.datakit.build_pdf_source.quality.judge_render_adjudication_set` supply the
missing half by putting the two readings in front of a judge that can see the page.

**The render study did not keep the text.** It kept ``mupdf_chars``, ``pdfium_chars`` and the
agreement columns, so there is nothing on storage to adjudicate. Every page is therefore rendered
and read again here, and the fresh readings carry their own control arm so the divergence can be
shown to have reproduced rather than assumed to have. Re-reading is cheap -- one GPU for about
fifteen minutes -- and it is the only honest option: adjudicating reconstructed text would be
adjudicating a different experiment.

**The reference renderer is the threat to validity, and it has no free answer.** A judge decides by
looking at a rendered page, that page is drawn by one of the two engines under test, and a MuPDF
reference plausibly favours the MuPDF-derived reading. :data:`NEUTRALITY_DPIS` measures whether the
engines converge at the judge's resolution -- if they did, either reference would be neutral and one
pass would settle it. They do not converge (the feed's p50 changed-pixel fraction is 0.0416 and the
160-DPI figure is 0.0382), so **every packet is judged twice, once against each engine's reference**,
with the pages, the text and the blinding held fixed and only the reference image moving. A verdict
that flips between the two arms is itself the finding.

**The draw oversamples the divergent tail on purpose, and the analysis undoes it.** A uniform draw
over 1,795 pages would spend three-quarters of the budget on pages whose two readings are already
identical. :data:`STRATA` allocates to the pages that carry the question -- the 38 ``catastrophic``
pages where control > 0.9 and treatment < 0.5, plus the large-loss, CJK, small-glyph and
below-legibility-floor sets -- and every stratum records its corpus page share so
``judge_render_adjudication_set`` can post-stratify back. The route adjudication is the cautionary
tale: its stratified headline read 0.414 and post-stratifying to corpus page share put it at ~0.51.
The oversampled number is not the number anyone should quote, and both are reported.

**The extractions are presented natively, unlike the route packets.** ``build_adjudication_set``
canonicalises because its three routes emit three dialects and a judge with a Markdown preference
would produce a style verdict wearing a quality verdict's clothes. Here both texts come from the
same model under the same prompt at temperature 0, so there is no dialect axis to neutralise, and
canonicalising would erase real structural differences -- a run that recovers a table and a run that
does not are supposed to look different.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \\
        --job-name pdfium-render-adjudication --extra pdf \\
        --cpu 8 --memory 32GB --disk 32GB --enable-extra-resources \\
        -- python -m experiments.datakit.build_pdf_source.quality.build_render_adjudication_set
"""

import base64
import io
import json
import logging
import platform
import random
import sys
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from importlib.metadata import version
from urllib.parse import urlparse

import numpy as np
import polars as pl
from marin.inference.iris import remote_inference

from experiments.datakit.build_pdf_source.ocr_extract.client import OcrEndpoint, PageOcr, ocr_page
from experiments.datakit.build_pdf_source.ocr_extract.render import (
    RenderedPage,
    RenderOptions,
    effective_dpi,
    target_dimensions,
)
from experiments.datakit.build_pdf_source.quality.build_render_study import (
    PAGES_IN_FLIGHT,
    REQUEST_THREADS,
    agreement_columns,
    page_facts,
    study_inference_config,
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
    render_mupdf,
    render_pdfium,
)
from experiments.datakit.build_pdf_source.quality.route_agreement import VLM, page_agreement

logger = logging.getLogger(__name__)

BUCKET = "marin-us-east-02a"
STUDY_ROOT = f"s3://{BUCKET}/marin/data/pdf_quality"
# The render study's only output. x86_64 produced none, so aarch64 is the whole record and this
# build runs on aarch64 too: rasterisation is deterministic per platform but not across platforms,
# and re-rendering on the other architecture would compare pages the study never measured.
STUDY_PATH = f"{STUDY_ROOT}/cc_focus_2026_22_pdfium_render_study/aarch64-5.13.0.parquet"

OUTPUT_PREFIX = f"{STUDY_ROOT}/cc_focus_2026_22_render_adjudication"
PACKETS_PREFIX = f"{OUTPUT_PREFIX}/packets"
KEY_PATH = f"{OUTPUT_PREFIX}/key.json"
READINGS_PATH = f"{OUTPUT_PREFIX}/readings.parquet"
NEUTRALITY_PATH = f"{OUTPUT_PREFIX}/neutrality.parquet"

MODULE_NAME = "experiments.datakit.build_pdf_source.quality.build_render_adjudication_set"
WORKER_FLAG = "--worker"
RENDER_OP = "render"
# Six full-page renders per page, two of them at 300 DPI, plus PNG encoding. Generous enough that a
# heavy page is not called a hang.
RENDER_TIMEOUT = 300.0

# The resolution the judge reads the page at. Deliberately equal to
# ``build_adjudication_set.RENDER_DPI`` so a verdict here is comparable with the route adjudication's,
# and deliberately *not* imported from it: that module is written against ``zephyr.execution``, which
# is ``zephyr.context`` on current main, so importing it kills this job at module scope.
RENDER_DPI = 160
# Resolutions the two engines are compared at, to answer whether either is a neutral reference.
# RENDER_DPI is what the judge sees; the rest establish whether divergence falls with resolution.
NEUTRALITY_DPIS = (RENDER_DPI, 220, 300)

SAMPLE_SEED = 20260827
# Characters of each reading shown. Deliberately larger than the route packets' 4,000: those clipped
# three routes over three pages to fit a judge's attention, while a packet here is one page and two
# readings. The model's 4,096-token cap bounds a page at roughly 16,000 characters, so at 12,000
# almost nothing clips -- and clipping is not neutral when the readings differ in length, because a
# judge cannot credit content it was not shown.
EXCERPT_CHARS = 12000
BLIND_LABELS = ("A", "B")
ARMS = ("mupdf", "pdfium")

RENDER_OPTIONS = RenderOptions()

# Stratum membership, assigned first-match so the strata are disjoint and per-stratum counts mean
# what they say. Ordered by how sharply each carries the question: the catastrophic pages are the
# ones the rejection was written around, and the residual signal strata (cjk, small_glyphs,
# below_legibility_floor) hold only their pages that did *not* already lose agreement, which is what
# makes them a check on whether the signal is the page kind or the loss.
#
# Quotas are a compromise between two jobs the same draw has to do. Per-stratum verdicts want the
# hard strata taken whole; the post-stratified estimate wants allocation proportional to corpus page
# share, because its standard error is `sum_s w_s^2 p_s(1-p_s)/(n_s-1)` and `unchanged` alone carries
# 67.9% of the weight. Under-drawing it would put +/-0.086 around the one number the report is meant
# to stand on. At these quotas the post-stratified standard error is ~0.021 and every stratum with a
# real question attached is drawn at 40 pages or its own size, whichever is smaller.
STRATA: tuple[tuple[str, pl.Expr, int], ...] = (
    ("catastrophic", pl.col("control_bigram_f1").gt(0.9) & pl.col("treatment_bigram_f1").lt(0.5), 38),
    ("reverse_catastrophic", pl.col("treatment_bigram_f1").gt(0.9) & pl.col("control_bigram_f1").lt(0.5), 1),
    ("large_loss", pl.col("delta").lt(-0.10), 104),
    ("below_legibility_floor", pl.col("dpi").lt(RENDER_OPTIONS.legibility_floor_dpi), 22),
    ("cjk", pl.col("cjk_ratio").gt(0.1), 50),
    ("small_glyphs", pl.col("median_font_size").gt(0) & pl.col("median_font_size").lt(8.0), 50),
    ("moderate_loss", pl.col("delta").lt(-0.01), 150),
    # Pages the two engines were read identically on. Judged anyway, and for two reasons: it is the
    # judge's null -- two readings that agree to within the model's own noise must come out at 0.5,
    # and a departure from parity here invalidates every other stratum's reading -- and it is two
    # thirds of the corpus, so the post-stratified estimate is mostly this stratum's answer.
    ("unchanged", pl.lit(True), 300),
)


# ---------------------------------------------------------------------------
# The draw
# ---------------------------------------------------------------------------


def registered_domain(url: str | None) -> str:
    """The host of a URL, which is the unit near-duplicate documents cluster in.

    Same derivation as :func:`train_route_model.registered_domain`, resolved here rather than
    imported: that module pulls in xgboost at scope, and neither this build nor the judging pass has
    any use for a booster.
    """
    return (urlparse(url).hostname or "").lower() if url else ""


def stratum_of() -> pl.Expr:
    """First-match stratum assignment, so every page lands in exactly one."""
    expression = pl.lit(None, dtype=pl.String)
    for name, predicate, _ in reversed(STRATA):
        expression = pl.when(predicate).then(pl.lit(name)).otherwise(expression)
    return expression


def select(study: pl.DataFrame, seed: int) -> pl.DataFrame:
    """The stratified draw, with each stratum's corpus page share attached to every row.

    The share is the weight ``judge_render_adjudication_set`` post-stratifies with, and it is
    computed over the whole study frame rather than over the draw -- that is the entire point of
    carrying it, since the draw is deliberately not corpus-shaped.
    """
    frame = study.with_columns(
        (pl.col("treatment_bigram_f1") - pl.col("control_bigram_f1")).alias("delta"),
        pl.col("url").map_elements(registered_domain, return_dtype=pl.String).alias("domain"),
    ).with_columns(stratum_of().alias("stratum"))

    shares = frame.group_by("stratum").agg((pl.len() / frame.height).alias("page_share"), pl.len().alias("corpus_pages"))
    sizes = dict(frame["stratum"].value_counts().iter_rows())
    drawn_total = sum(min(quota, sizes.get(name, 0)) for name, _, quota in STRATA)

    rng = random.Random(seed)
    drawn = []
    for name, _, quota in STRATA:
        stratum = frame.filter(pl.col("stratum") == name)
        if not stratum.height:
            logger.warning("draw: stratum %s is empty", name)
            continue
        take = min(quota, stratum.height)
        drawn.append(stratum.sample(n=take, shuffle=True, seed=rng.randrange(2**31)))
        # Both fractions, because they are the two halves of the post-stratification argument and
        # quoting the wrong one is how an oversampled tail gets read as a corpus statement.
        logger.info(
            "draw: %-24s %4d of %4d pages | draw share %.4f, corpus share %.4f",
            name,
            take,
            stratum.height,
            take / drawn_total,
            stratum.height / frame.height,
        )
    selection = pl.concat(drawn).join(shares, on="stratum", how="left")
    logger.info(
        "draw: %d pages over %d documents and %d domains",
        selection.height,
        selection["source_id"].n_unique(),
        selection["domain"].n_unique(),
    )
    return selection


# ---------------------------------------------------------------------------
# Worker: rendering, in a process the driver is willing to lose
# ---------------------------------------------------------------------------


def _reference_dimensions(width: float, height: float, dpi: int) -> tuple[int, int]:
    """Page size in points to ``(height, width)`` pixels at a fixed DPI.

    Both engines are handed this pair explicitly rather than each being asked for its own DPI-scaled
    pixmap: a paired pixel comparison needs identical buffer shapes, and PyMuPDF's ``dpi=`` argument
    and PDFium's ``size_x``/``size_y`` do not round the same way.
    """
    return max(1, round(height * dpi / 72.0)), max(1, round(width * dpi / 72.0))


def _divergence(left: np.ndarray, right: np.ndarray) -> dict:
    difference = np.abs(left.astype(np.int16) - right.astype(np.int16)).max(axis=2)
    return {
        "changed_fraction": float((difference > PIXEL_TOLERANCE).mean()),
        "mean_absolute_difference": float(difference.mean()),
    }


def _png(samples: np.ndarray) -> bytes:
    from PIL import Image  # noqa: PLC0415

    buffer = io.BytesIO()
    Image.fromarray(samples).save(buffer, format="PNG", compress_level=PILLOW_COMPRESS_LEVEL)
    return buffer.getvalue()


def _render_pages(payload: bytes, indices: list[int]) -> dict:
    """One document's selected pages, rendered both ways at the feed's dimensions and the judge's.

    The pixel comparison happens here so full-resolution buffers never cross the pipe; what crosses
    is two feed data URIs, two reference PNGs and the numbers describing them.
    """
    import pymupdf  # noqa: PLC0415
    import pypdfium2 as pdfium  # noqa: PLC0415

    other = pdfium.PdfDocument(payload)
    document = pymupdf.open(stream=payload, filetype="pdf")
    try:
        pages = []
        for index in indices:
            if index >= min(len(document), len(other)):
                continue
            page = document[index]
            page_width, page_height = page.rect.width, page.rect.height
            if page_width < 1 or page_height < 1:
                continue

            height, width = target_dimensions(page_width, page_height, RENDER_OPTIONS)
            feed_left = render_mupdf(page, height, width)
            feed_right = render_pdfium(other[index], height, width)

            neutrality = []
            references: dict[int, tuple[bytes, bytes]] = {}
            for dpi in NEUTRALITY_DPIS:
                reference_height, reference_width = _reference_dimensions(page_width, page_height, dpi)
                left = render_mupdf(page, reference_height, reference_width)
                right = render_pdfium(other[index], reference_height, reference_width)
                neutrality.append(
                    {
                        "dpi": dpi,
                        "height": reference_height,
                        "width": reference_width,
                        **_divergence(left, right),
                    }
                )
                if dpi == RENDER_DPI:
                    references[dpi] = (_png(left), _png(right))

            reference_pngs = references[RENDER_DPI]
            pages.append(
                {
                    "page_index": index,
                    "height": height,
                    "width": width,
                    **_divergence(feed_left, feed_right),
                    **page_facts(page, effective_dpi(height * width, page_width, page_height)),
                    "neutrality": neutrality,
                    "mupdf_uri": f"data:image/png;base64,{base64.b64encode(_png(feed_left)).decode()}",
                    "pdfium_uri": f"data:image/png;base64,{base64.b64encode(_png(feed_right)).decode()}",
                    "mupdf_reference": base64.b64encode(reference_pngs[0]).decode(),
                    "pdfium_reference": base64.b64encode(reference_pngs[1]).decode(),
                }
            )
        return {"pages": pages}
    finally:
        document.close()
        other.close()


def _measure(payload: bytes, indices: list[int]) -> dict:
    """Render one document, reporting whatever happened instead of raising."""
    try:
        return {"outcome": str(Outcome.OK), **_render_pages(payload, indices)}
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as error:
        return {"outcome": str(failure_outcome(error)), "error": f"{type(error).__name__}: {error}"[:300]}


def render_op(indices: list[int]) -> str:
    """The op string carrying which pages to render.

    :class:`~...probe_pdf_inspector.Worker` sends an op name and a byte payload and nothing else, so
    the page selection rides in the op rather than in a second protocol.
    """
    return f"{RENDER_OP}:{','.join(str(index) for index in indices)}"


def worker_main() -> None:
    """Serve length-prefixed documents from stdin until the driver closes it.

    The reply channel is a duplicate of the inherited stdout and file descriptor 1 is then pointed
    at stderr, because MuPDF writes its content-stream warnings to fd 1 by default and would
    otherwise land them in the middle of the length-prefixed protocol.
    """
    import faulthandler  # noqa: PLC0415
    import os  # noqa: PLC0415

    import pymupdf  # noqa: PLC0415

    faulthandler.enable()
    replies = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    pymupdf.TOOLS.mupdf_display_errors(False)

    stdin = sys.stdin.buffer
    while True:
        header = stdin.readline()
        if not header:
            return
        request = json.loads(header)
        payload = read_exactly(stdin, request["size"])
        indices = [int(part) for part in request["op"].split(":", 1)[1].split(",") if part]
        replies.write(json.dumps(_measure(payload, indices)).encode() + b"\n")
        replies.flush()


# ---------------------------------------------------------------------------
# Driver: three readings of every page, then the packet
# ---------------------------------------------------------------------------


@dataclass
class PageWork:
    """One selected page: its renders, and the three readings it is waiting on."""

    row: dict
    references: tuple[bytes, bytes]
    mupdf_first: "Future[PageOcr]"
    mupdf_second: "Future[PageOcr]"
    pdfium: "Future[PageOcr]"

    def resolve(self) -> tuple[dict, tuple[bytes, bytes]] | None:
        """Fold the three answers into the row, dropping a page whose requests did not all land.

        A page missing one of its readings cannot be adjudicated and cannot contribute a control
        either, so it is dropped rather than carried as a half-row: the draw is stratified and a
        silently truncated stratum would move a post-stratified estimate.
        """
        try:
            first, second, other = (
                self.mupdf_first.result(),
                self.mupdf_second.result(),
                self.pdfium.result(),
            )
        except Exception as error:
            logger.warning("%s p%d: %s", self.row["source_id"], self.row["page_index"], error)
            return None

        control = page_agreement(first.text, second.text, VLM, VLM)
        treatment = page_agreement(first.text, other.text, VLM, VLM)
        row = {
            **self.row,
            "mupdf_text": first.text,
            "mupdf_text_repeat": second.text,
            "pdfium_text": other.text,
            "mupdf_completion_tokens": first.completion_tokens,
            "pdfium_completion_tokens": other.completion_tokens,
            "mupdf_truncated": first.truncated,
            "pdfium_truncated": other.truncated,
            **{f"fresh_control_{key}": value for key, value in agreement_columns(control).items()},
            **{f"fresh_treatment_{key}": value for key, value in agreement_columns(treatment).items()},
        }
        return row, self.references


def read_pages(selection: pl.DataFrame, documents: pl.DataFrame, endpoint: OcrEndpoint) -> list[tuple[dict, tuple]]:
    """Render every selected page and keep the model's three readings of it.

    Rendering and waiting overlap exactly as in ``build_render_study``: a page's requests go out the
    moment the worker hands it back, and the next document renders while the previous one is still
    in flight.
    """
    # Grouped by hand rather than with ``group_by``, whose iteration yields the key as a one-tuple
    # on current polars and silently produces a dict nothing can look a plain source id up in.
    rows_by_key = {(row["source_id"], row["page_index"]): row for row in selection.iter_rows(named=True)}
    wanted: dict[str, list[int]] = {}
    for source_id, page_index in rows_by_key:
        wanted.setdefault(source_id, []).append(page_index)
    wanted = {source_id: sorted(indices) for source_id, indices in wanted.items()}
    payloads = dict(zip(documents["source_id"].to_list(), documents["pdf"].to_list(), strict=True))

    worker = Worker(MODULE_NAME, timeout=RENDER_TIMEOUT)
    pool = ThreadPoolExecutor(max_workers=REQUEST_THREADS, thread_name_prefix="render-adjudication")
    inflight: list[PageWork] = []
    resolved: list[tuple[dict, tuple]] = []
    outcomes: Counter = Counter()
    errors: Counter = Counter()

    def send(uri: str, index: int, dpi: float, pixels: int) -> "Future[PageOcr]":
        page = RenderedPage(data_uri=uri, page_index=index, pixels=pixels, dpi=dpi)
        return pool.submit(ocr_page, endpoint, REQUEST_THREADS, page)

    def drain(limit: int) -> None:
        while len(inflight) > limit:
            answer = inflight.pop(0).resolve()
            if answer is not None:
                resolved.append(answer)

    try:
        for position, (source_id, indices) in enumerate(wanted.items()):
            reply = worker.call(render_op(indices), payloads[source_id])
            outcomes[reply.result["outcome"]] += 1
            if reply.result["outcome"] != Outcome.OK:
                errors[reply.result.get("error", "worker lost")] += 1
                continue
            for page in reply.result["pages"]:
                key = (source_id, page["page_index"])
                if key not in rows_by_key:
                    continue
                drain(PAGES_IN_FLIGHT - 1)
                pixels = page["height"] * page["width"]
                mupdf_uri, pdfium_uri = page.pop("mupdf_uri"), page.pop("pdfium_uri")
                references = (
                    base64.b64decode(page.pop("mupdf_reference")),
                    base64.b64decode(page.pop("pdfium_reference")),
                )
                inflight.append(
                    PageWork(
                        row={**rows_by_key[key], **page},
                        references=references,
                        mupdf_first=send(mupdf_uri, page["page_index"], page["dpi"], pixels),
                        mupdf_second=send(mupdf_uri, page["page_index"], page["dpi"], pixels),
                        pdfium=send(pdfium_uri, page["page_index"], page["dpi"], pixels),
                    )
                )
            if (position + 1) % 50 == 0:
                logger.info("read: %d/%d documents, %d pages resolved", position + 1, len(wanted), len(resolved))
        drain(0)
    finally:
        worker.stop()
        pool.shutdown(wait=True)

    for outcome, count in outcomes.most_common():
        logger.info("read: render %-12s %5d", outcome, count)
    for message, count in errors.most_common(ERROR_SAMPLES):
        logger.info("read: render failure x%d: %s", count, message)
    logger.info("read: %d pages with all three readings, %d worker spawns", len(resolved), worker.spawns)
    return resolved


# ---------------------------------------------------------------------------
# Packets
# ---------------------------------------------------------------------------


def excerpt(text: str) -> str:
    collapsed = text.strip()
    if len(collapsed) <= EXCERPT_CHARS:
        return collapsed or "(this system produced no text for this page)"
    return f"{collapsed[:EXCERPT_CHARS]}\n... [clipped at {EXCERPT_CHARS} characters]"


def packet_markdown(packet_id: str, texts: dict[str, str], labels: dict[str, str]) -> str:
    """One packet's judging document: the two readings of one page, blinded.

    No image filename is named anywhere in it. The judge is sent one reference image per verdict and
    which engine drew it is the manipulation under test, so the document has to read identically in
    both arms.

    Sections are emitted in **label** order, not engine order. Randomising which letter an engine
    hides behind while always printing MuPDF's reading first would leave the presentation order
    perfectly correlated with the engine, so any positional preference the judge has would read as a
    quality verdict. The letter and the position are randomised together.
    """
    engine_by_label = {label: arm for arm, label in labels.items()}
    sections = [
        f"# Page {packet_id}",
        "",
        "The image above is the rendered page. Below are two transcriptions of that same page.",
        "",
    ]
    for label in BLIND_LABELS:
        sections.append(f"## Extraction {label}")
        sections.append(excerpt(texts[engine_by_label[label]]))
        sections.append("")
    return "\n".join(sections)


def write_packets(fs, resolved: list[tuple[dict, tuple]], seed: int) -> list[dict]:
    """Every page's packet directory and its blinding key entry.

    Both reference images live in one directory and the document is written once, so the two judging
    arms are paired down to the byte: same text, same labels, same order, only the image moves.

    Packet ids are flat and sequential, matching ``build_adjudication_set``'s ``doc_NNNN``. A id built
    out of the source id would carry the WARC path's slashes into the object key, nesting every packet
    six levels deep and leaving the verdict prefix un-listable -- which is how the judging pass finds
    out what it has already bought. Order is fixed by sorting first, so the ids are stable across
    reruns and ``page_0431`` means the same page tomorrow.
    """
    rng = random.Random(seed)
    key = []
    ordered = sorted(resolved, key=lambda item: (item[0]["source_id"], item[0]["page_index"]))
    for position, (row, references) in enumerate(ordered):
        packet_id = f"page_{position:04d}"
        labels = dict(zip(ARMS, rng.sample(list(BLIND_LABELS), len(BLIND_LABELS)), strict=True))
        destination = f"{PACKETS_PREFIX}/{packet_id}"
        for arm, payload in zip(ARMS, references, strict=True):
            with fs.open(f"{destination}/reference_{arm}.png", "wb") as stream:
                stream.write(payload)
        texts = {"mupdf": row["mupdf_text"], "pdfium": row["pdfium_text"]}
        with fs.open(f"{destination}/document.md", "w") as stream:
            stream.write(packet_markdown(packet_id, texts, labels))
        key.append(
            {
                "packet_id": packet_id,
                "source_id": row["source_id"],
                "url": row["url"],
                "domain": row["domain"],
                "stratum": row["stratum"],
                "page_share": row["page_share"],
                "corpus_pages": row["corpus_pages"],
                "page_index": row["page_index"],
                # The blinding: which engine's reading each shown label actually is.
                "labels": {label: arm for arm, label in labels.items()},
                "study_delta": row["delta"],
                "study_control_bigram_f1": row["control_bigram_f1"],
                "study_treatment_bigram_f1": row["treatment_bigram_f1"],
                "fresh_control_bigram_f1": row["fresh_control_bigram_f1"],
                "fresh_treatment_bigram_f1": row["fresh_treatment_bigram_f1"],
                "truncated": bool(row["mupdf_truncated"] or row["pdfium_truncated"]),
                "runaway_length": _runaway(row["mupdf_text"], row["pdfium_text"]),
                "dpi": row["dpi"],
                "cjk_ratio": row["cjk_ratio"],
                "median_font_size": row["median_font_size"],
                "char_count": row["char_count"],
                "changed_fraction": row["changed_fraction"],
            }
        )
    with fs.open(KEY_PATH, "w") as stream:
        json.dump(key, stream)
    logger.info("packets: %d written -> %s", len(key), PACKETS_PREFIX)
    return key


def _runaway(left: str, right: str) -> bool:
    """One reading more than three times the other's length, the render study's loop proxy."""
    shorter, longer = sorted((len(left), len(right)))
    return longer > 3 * max(shorter, 1)


# ---------------------------------------------------------------------------
# Reference-renderer neutrality
# ---------------------------------------------------------------------------


def neutrality_frame(resolved: list[tuple[dict, tuple]]) -> pl.DataFrame:
    """Per-page pixel divergence at the feed's dimensions and at each judging resolution."""
    rows = []
    for row, _ in resolved:
        base = {"source_id": row["source_id"], "page_index": row["page_index"], "stratum": row["stratum"]}
        rows.append(
            {
                **base,
                "dpi_label": "feed",
                "dpi": row["dpi"],
                "changed_fraction": row["changed_fraction"],
                "mean_absolute_difference": row["mean_absolute_difference"],
            }
        )
        for entry in row["neutrality"]:
            rows.append(
                {
                    **base,
                    "dpi_label": str(entry["dpi"]),
                    "dpi": float(entry["dpi"]),
                    "changed_fraction": entry["changed_fraction"],
                    "mean_absolute_difference": entry["mean_absolute_difference"],
                }
            )
    return pl.DataFrame(rows)


def report_neutrality(frame: pl.DataFrame) -> None:
    """Does the pixel divergence collapse at the resolution the judge reads at?

    If it did, either engine would be an acceptable neutral reference and one judging pass would
    settle the question. The number to read is the ratio of the 160-DPI figure to the feed figure:
    anything near 1 means the reference choice is a live confound and both arms have to be run.
    """
    logger.info("--- reference-renderer neutrality: pixel divergence by rendering resolution ---")
    summary = (
        frame.group_by("dpi_label")
        .agg(
            pl.len().alias("pages"),
            pl.col("dpi").mean().alias("mean_dpi"),
            pl.col("changed_fraction").median().alias("changed_p50"),
            pl.col("changed_fraction").quantile(0.9).alias("changed_p90"),
            pl.col("mean_absolute_difference").mean().alias("mean_abs"),
        )
        .sort("mean_dpi")
    )
    for row in summary.iter_rows(named=True):
        logger.info(
            "%-6s n=%4d  mean dpi %6.1f  changed px p50 %.4f  p90 %.4f  mean |delta| %.2f/255",
            row["dpi_label"],
            row["pages"],
            row["mean_dpi"],
            row["changed_p50"],
            row["changed_p90"],
            row["mean_abs"],
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def report_readings(frame: pl.DataFrame) -> None:
    """Did the divergence reproduce? The draw is only meaningful if it did."""
    logger.info("--- fresh readings against the study's, per stratum ---")
    summary = (
        frame.with_columns(
            (pl.col("fresh_treatment_bigram_f1") - pl.col("fresh_control_bigram_f1")).alias("fresh_delta")
        )
        .group_by("stratum")
        .agg(
            pl.len().alias("pages"),
            pl.col("delta").mean().alias("study_delta"),
            pl.col("fresh_delta").mean().alias("fresh_delta"),
            pl.col("fresh_control_bigram_f1").mean().alias("fresh_control"),
            pl.col("fresh_treatment_bigram_f1").mean().alias("fresh_treatment"),
        )
        .sort("pages", descending=True)
    )
    for row in summary.iter_rows(named=True):
        logger.info(
            "%-24s n=%4d  study delta %+.4f  fresh delta %+.4f  (fresh control %.4f treatment %.4f)",
            row["stratum"],
            row["pages"],
            row["study_delta"],
            row["fresh_delta"],
            row["fresh_control"],
            row["fresh_treatment"],
        )


def main() -> None:
    import pypdfium2 as pdfium  # noqa: PLC0415
    from rigging.log_setup import configure_logging  # noqa: PLC0415

    configure_logging(logging.INFO)
    logger.info(
        "pypdfium2 %s (pdfium %s), pymupdf %s, on %s",
        version("pypdfium2"),
        pdfium.PDFIUM_INFO,
        version("pymupdf"),
        platform.machine(),
    )

    fs = storage()
    with fs.open(STUDY_PATH, "rb") as stream:
        study = pl.read_parquet(stream)
    logger.info("study: %d pages over %d documents", study.height, study["source_id"].n_unique())

    selection = select(study, SAMPLE_SEED)
    # Plain Python lists rather than Series: ``is_in`` against a Series of the same dtype is
    # ambiguous on current polars and warns on every call.
    drawn_ids = selection["source_id"].unique().to_list()
    documents = probe_documents(fs).filter(pl.col("source_id").is_in(drawn_ids))
    logger.info("documents: %d of %d carried PDF bytes", documents.height, len(drawn_ids))
    selection = selection.filter(pl.col("source_id").is_in(documents["source_id"].to_list()))

    started = time.perf_counter()
    with remote_inference(study_inference_config()) as session:
        endpoint = OcrEndpoint(
            base_url=session.model.endpoint.base_url,
            model=session.model.endpoint.model,
            max_visual_tokens=RENDER_OPTIONS.max_visual_tokens,
        )
        logger.info("read: endpoint ready at %s (%s)", endpoint.base_url, session.backend_name)
        resolved = read_pages(selection, documents, endpoint)
    logger.info("read: %.1f minutes with the fleet up", (time.perf_counter() - started) / 60)

    neutrality = neutrality_frame(resolved)
    with fs.open(NEUTRALITY_PATH, "wb") as stream:
        neutrality.write_parquet(stream, compression="zstd", compression_level=1)
    report_neutrality(neutrality)

    key = write_packets(fs, resolved, SAMPLE_SEED)

    # ``neutrality`` is a list of structs and has its own frame; polars would have to infer a nested
    # schema for a column nothing reads from here.
    flat = [{name: value for name, value in row.items() if name != "neutrality"} for row, _ in resolved]
    readings = pl.DataFrame(flat, strict=False, infer_schema_length=None)
    with fs.open(READINGS_PATH, "wb") as stream:
        readings.write_parquet(stream, compression="zstd", compression_level=1)
    logger.info("readings: %d rows -> %s", readings.height, READINGS_PATH)
    report_readings(readings)
    logger.info("key: %d packets -> %s", len(key), KEY_PATH)


if __name__ == "__main__":
    if WORKER_FLAG in sys.argv:
        worker_main()
    else:
        main()
