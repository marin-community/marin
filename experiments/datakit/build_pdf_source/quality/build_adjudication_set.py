# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Package documents for blind adjudication of which extraction route read the page correctly.

The agreement numbers in :mod:`~experiments.datakit.build_pdf_source.quality.route_agreement` say how far
apart the routes are. They cannot say which one is right, and on this corpus that distinction is not
academic: sorting by disagreement surfaces chart-heavy pages where Docling transcribed axis labels
the VLM was told to ignore, and RTL documents where Docling's Unicode is visibly better than the
VLM's. Treating either as "the cheap route failed" would train the router to send good documents to
the expensive route, and the reverse mistake is worse.

So the label comes from looking at the page. This module writes, per document, a directory holding
the rendered pages and **all three** extractions of those same pages -- Docling, pdf-inspector and
the VLM -- with the routes blinded and their order randomized per document. A judge sees
"Extraction A", "Extraction B" and "Extraction C" and cannot tell which produced them. The key
mapping is written separately, next to the work rather than inside it.

Pages are chosen rather than taken from the front: the first page of a report is a title page every
route gets right and that says nothing about the document. :func:`informative_pages` picks the pages
where the routes are furthest apart, which is where the verdict is actually decided, and always
includes one page where they agree so a judge can calibrate on the document's own typography.

Rendering is at :data:`RENDER_DPI`, well above the ~146 DPI median the VLM itself saw, because the
judge has to be able to read what the model was working from and adjudicate ligatures and equation
layout at the same time.

**Blinding hides the route; it does not hide the dialect.** This is the threat this pass exists to
answer, and it is not hypothetical: pdf-inspector emits Markdown, which is the VLM's own dialect,
while Docling emits tagged plain text -- ``<docling_table>`` around a padded GitHub grid,
``<docling_formula>``, no heading markers. A judge with a systematic preference for Markdown would
produce a verdict that looks like a quality judgment and is not, and every number Stages 1-2
measured is agreement against a Markdown reference, so a residual dialect advantage cannot be
excluded from them.

Two presentations are therefore written for every document, over the *same* pages with the *same*
blinding:

``canonical``
    Every route re-rendered into one dialect-neutral form by :func:`canonical_page` -- tables become
    ``cell | cell`` rows whatever syntax carried them, formulas become ``[formula] ...``, figures
    become ``[figure] ...``, headings and emphasis lose their markers. Content and reading order
    survive untouched, so a route that scrambles a table's cells still shows scrambled cells; only
    the serialization convention is erased. This is the arm the verdict is read from.
``native``
    Each route exactly as it serializes, which is what a judge would have seen without this
    machinery.

Judging a subset under both and comparing the verdicts *on the same documents* measures the style
effect directly rather than assuming it away. The pairing is what gives that comparison power: the
two arms differ in presentation and in nothing else, so a verdict that moves between them moved
because of dialect.

**The draw is stratified, and RTL is deliberately oversampled.** The prior pass was 72 documents and
was underpowered where it mattered -- it reported ``n=4`` on RTL, which is a flag and not a finding.
:data:`STRATA` allocates :data:`SAMPLE_SIZE` documents across the axes where the routes are known or
suspected to diverge, assigning each document to the first stratum it matches so the strata stay
disjoint and per-stratum counts mean what they say. Documents are drawn at most one per registered
domain: the crawl holds ~9.8% exact-duplicate PDFs and many more near-duplicates from the same
publisher, so an unconstrained draw would spend the sample on one publisher's template.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name pdf-adjudication-set --extra pdf \\
        --cpu 8 --memory 24GB --disk 16GB --enable-extra-resources \\
        -- python -m experiments.datakit.build_pdf_source.quality.build_adjudication_set

``--extra pdf`` is required: pdf-inspector's text is not stored by
:mod:`~experiments.datakit.build_pdf_source.quality.build_inspector_study`, which keeps only
agreement columns and signals, so the extraction is re-run here against the same PDF bytes.
"""

import json
import logging
import random
import re
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import polars as pl
import pymupdf
from fray.types import ResourceConfig
from rigging.log_setup import configure_logging
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.datakit.build_pdf_source.quality import route_agreement
from experiments.datakit.build_pdf_source.quality.analyze_route_study import label, read_table, route_ok
from experiments.datakit.build_pdf_source.quality.build_route_study import shards, storage
from experiments.datakit.build_pdf_source.quality.probe_pdf_inspector import WORKER_FLAG, Worker, read_exactly

logger = logging.getLogger(__name__)

ROUTE_STUDY_PREFIX = "s3://marin-us-east-02a/marin/data/pdf_quality/cc_focus_2026_22_route_study"
INSPECTOR_STUDY_PREFIX = "s3://marin-us-east-02a/marin/data/pdf_quality/cc_focus_2026_22_inspector_study"
OUTPUT_PREFIX = "s3://marin-us-east-02a/marin/data/pdf_quality/cc_focus_2026_22_adjudication"

PACKETS_PREFIX = f"{OUTPUT_PREFIX}/packets"
KEY_SHARD_PREFIX = f"{OUTPUT_PREFIX}/key_parts"
KEY_PATH = f"{OUTPUT_PREFIX}/key.json"
MANIFEST_PATH = f"{OUTPUT_PREFIX}/manifest.json"
SELECTION_PATH = f"{OUTPUT_PREFIX}/selection.parquet"

MODULE_NAME = "experiments.datakit.build_pdf_source.quality.build_adjudication_set"
EXTRACT_OP = "extract"

RENDER_DPI = 160
# Pages shown per document. Three is what fits a judge's attention while still showing a
# disagreement, a control, and one more of whatever the document is mostly made of.
PAGES_PER_DOCUMENT = 3
# Characters of each extraction shown per page. Long enough to see structure and reading order,
# short enough that three pages of three routes stay readable.
EXCERPT_CHARS = 4000

ROUTES = ("docling", "inspector", "vlm")
BLIND_LABELS = ("A", "B", "C")

SAMPLE_SIZE = 360
SAMPLE_SEED = 20260813
# Documents one registered domain may contribute to the whole draw. See :func:`select`: the binding
# constraint is RTL, where 313 usable documents sit on 36 domains.
MAX_PER_DOMAIN = 2
# Documents also judged under the native-dialect presentation, to measure the style effect against
# the canonical arm on the same documents. Drawn proportionally across strata.
STYLE_CONTROL_SIZE = 90
# Documents packaged for human adjudication. Small enough to be judged by a person in one sitting,
# stratified so the axes that decide the question are all present.
HUMAN_SUBSET_SIZE = 45


class Presentation(StrEnum):
    """Which dialect a packet's extractions are rendered in."""

    CANONICAL = "canonical"
    NATIVE = "native"


# ---------------------------------------------------------------------------
# Page selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageChoice:
    """A page selected for adjudication: the PDF page, and where each route's text for it sits.

    ``source_index`` maps a route to its own index for this page, ``None`` when that route has no
    page matching this one -- a real and common outcome, since routes drop pages they read nothing
    from. ``recalls`` carries each cheap route's bigram recall against the VLM on this page, so the
    analysis can ask whether the verdict tracked the metric page by page rather than only per
    document.
    """

    page_index: int
    source_index: dict[str, int | None]
    reason: str
    recalls: dict[str, float]


def _aligned_to_vlm(cheap_pages: list[str], vlm_pages: list[str]) -> dict[int, int]:
    """Map a VLM page index to the cheap route's index for the same page.

    Pairing by index is not a cosmetic error here. Docling drops pages it reads nothing from -- ~8%
    of documents come back with fewer pages than the PDF has -- and after one drop every later page
    would be shown beside its neighbour's text. Judges reading packets built that way reported one
    route "fabricating" content and another "losing" it on documents where nothing had gone wrong
    except the alignment, and that bug invalidated an entire earlier pass.
    """
    return {
        vlm_index: cheap_index
        for cheap_index, vlm_index in route_agreement.align_pages(cheap_pages, vlm_pages)
        if vlm_index is not None and cheap_index is not None
    }


def informative_pages(pages_by_route: dict[str, list[str]], count: int) -> list[PageChoice]:
    """Pick the pages worth adjudicating: where the routes are furthest apart, plus a control.

    The VLM's page list is the PDF's own, so it is the reference every other route is aligned to and
    the page number shown to a judge is the PDF's. A page is scored by the *worst* any cheap route
    did on it, which selects both the pages where one route collapsed and the pages where both did
    -- and the second of those is a stratum in its own right, since two cheap routes that fail
    together cannot cover for each other.

    A page every route left empty is skipped: it adjudicates nothing, and on a scanned document it
    would be most of what gets shown.
    """
    vlm_pages = pages_by_route["vlm"]
    cheap = [route for route in ROUTES if route != "vlm"]
    aligned = {route: _aligned_to_vlm(pages_by_route[route], vlm_pages) for route in cheap}

    scored: list[PageChoice] = []
    for vlm_index, vlm_page in enumerate(vlm_pages):
        source_index = {route: aligned[route].get(vlm_index) for route in cheap}
        source_index["vlm"] = vlm_index
        recalls = {}
        any_content = bool(route_agreement.markdown_streams(vlm_page).tokens)
        for route in cheap:
            index = source_index[route]
            page_text = pages_by_route[route][index] if index is not None else ""
            measured = route_agreement.page_agreement(
                vlm_page, page_text, route_agreement.VLM, route_agreement.ROUTES_BY_NAME[route]
            )
            recalls[route] = measured.bigram_recall
            any_content = any_content or measured.candidate_tokens > 0
        if not any_content:
            continue
        scored.append(
            PageChoice(page_index=vlm_index, source_index=source_index, reason="disagreement", recalls=recalls)
        )
    if not scored:
        return []

    scored.sort(key=lambda choice: min(choice.recalls.values(), default=0.0))
    chosen = scored[: max(count - 1, 1)]
    # One page the routes agree on, as a control: it tells a judge what this document's typography
    # looks like when nothing has gone wrong, so a verdict rests on the difference rather than on
    # unfamiliarity with the layout.
    control = scored[-1]
    if control.page_index not in {choice.page_index for choice in chosen}:
        chosen.append(
            PageChoice(
                page_index=control.page_index,
                source_index=control.source_index,
                reason="control",
                recalls=control.recalls,
            )
        )
    return sorted(chosen, key=lambda choice: choice.page_index)


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stratum:
    """One slice of the draw: what it selects, and how many documents it is worth.

    Strata are matched in declaration order and a document joins the first it satisfies, so they
    partition the frame rather than overlap. That is what makes a per-stratum count a count of
    documents rather than of memberships, and it is why the two script strata come first: a
    multi-column RTL document is an RTL document for this study's purposes.
    """

    name: str
    target: int
    predicate: pl.Expr
    rationale: str


# Script, from the sampled pages' character mix. The thresholds are deliberately low: a document
# with 15% RTL characters is an RTL document for a router that reads `mean_latin_ratio` as its top
# feature.
_RTL = pl.col("mean_rtl_ratio") > 0.15
_CJK = pl.col("mean_cjk_ratio") > 0.15

# Both cheap routes fail the VLM's floor, yet agree closely with each other. Stage 2 found 28.3% of
# both-fail documents in the highest mutual-agreement decile and nobody has looked at what they are.
# Two readings are possible and they have opposite consequences: either both cheap routes lost the
# same content, or they agree because they are both right and the VLM is the one that diverged.
_MUTUAL_AGREEMENT = (
    pl.col("inspector_ok").not_()
    & pl.col("docling_ok").not_()
    & (pl.col("inspector_docling_bigram_recall_mean") >= 0.90)
)

STRATA: tuple[Stratum, ...] = (
    Stratum(
        "rtl",
        70,
        _RTL,
        "0.4% of the corpus and the axis the proxy label is backwards on; the prior pass had n=4.",
    ),
    Stratum("cjk", 50, _CJK, "4.0% of the corpus; the prior pass called it correct on n=5."),
    Stratum(
        "mutual_agreement_vlm_diverge",
        45,
        _MUTUAL_AGREEMENT,
        "Both cheap routes fail the label and agree with each other; never inspected.",
    ),
    Stratum(
        "encoding_damage",
        30,
        (pl.col("mean_fonts_unmappable") > 0.05)
        | (pl.col("mean_replacement_ratio") > 0.001)
        | (pl.col("garbled_text_ratio") > 0.1),
        "Broken ToUnicode CMaps and replacement characters: healthy 'not a scan' statistics, bad text.",
    ),
    Stratum(
        "math_dense",
        30,
        (pl.col("mean_math_unicode_ratio") > 0.01) | (pl.col("mean_math_font_ratio") > 0.1),
        "Equation layout is where a Markdown dialect and a tagged one diverge most.",
    ),
    Stratum(
        "multicolumn",
        30,
        pl.col("mean_column_count") > 1.5,
        "Reading-order damage is the failure the bigram metric exists to catch.",
    ),
    Stratum(
        "table_heavy",
        30,
        pl.col("inspector_extract_pages_with_tables") >= (0.5 * pl.col("inspector_page_count")),
        "The style confound's worst case: a pipe grid against a <docling_table>.",
    ),
    Stratum(
        "scanned_image_mixed",
        35,
        pl.col("inspector_pdf_type").is_in(["scanned", "image_based", "mixed"]),
        "pdf-inspector loses on these; 6.4% of the corpus and both sides must be represented.",
    ),
    Stratum(
        "latin_text_baseline",
        40,
        pl.lit(True),
        "The 93.6% majority case, where pdf-inspector's aggregate win was measured.",
    ),
)


def stratum_of() -> pl.Expr:
    """Assign each row to the first stratum it matches."""
    expression = pl.lit(None, dtype=pl.String)
    for stratum in reversed(STRATA):
        expression = pl.when(stratum.predicate).then(pl.lit(stratum.name)).otherwise(expression)
    return expression


def usable(frame: pl.DataFrame) -> pl.DataFrame:
    """The rows a verdict can be read from at all.

    Trustworthy on the VLM side for the reason the published report drops 10.7% of the sample: where
    the VLM extraction is itself truncated, loop-repaired or rendered below the legibility floor, a
    disagreement measures the VLM's failure rather than a cheap route's, and this study makes the
    VLM one of the three things being judged. Both cheap routes must have produced something, since
    a two-route packet cannot answer a three-route question.
    """
    return frame.filter(
        pl.col("trustworthy")
        & pl.col("feature_error").is_null()
        & (pl.col("domain") != "")
        & pl.col("docling_missing").not_()
        & pl.col("inspector_vlm_bigram_recall_mean").is_not_null()
        & pl.col("docling_vlm_bigram_recall_mean").is_not_null()
    )


def select(frame: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Draw the stratified sample, at most :data:`MAX_PER_DOMAIN` documents per registered domain.

    The cap is the whole reason this function is not a ``head`` per stratum. The crawl holds ~9.8%
    exact-duplicate PDFs and many more near-duplicates from one publisher's template, so an
    unconstrained draw buys repeated readings of the same document and reports them as independent.

    One document per domain would be the strict version, and on most strata it costs nothing. On RTL
    it costs the study: 313 usable RTL documents live on **36 domains**, 178 of them on four Israeli
    university sites, so a one-per-domain rule caps RTL at 36 and no oversampling can lift it. Two
    per domain reaches the target while keeping any single publisher to a bounded share -- and the
    number that should be read as this stratum's sample size is the *domain* count, which is carried
    through to the report for exactly that reason.

    A stratum short of its target takes what exists and says so rather than borrowing from another,
    because a stratum silently filled from elsewhere is how the prior pass ended up reporting an RTL
    conclusion drawn from four documents.
    """
    pool = frame.with_columns(stratum=stratum_of()).sample(fraction=1.0, shuffle=True, seed=seed)
    # The cap is per domain *within a stratum*, not per domain overall. Globally, a domain's two
    # slots go to whichever of its documents the shuffle put first, and the big RTL publishers hold
    # far more Latin documents than RTL ones -- so a global cap spends RTL's allowance on Latin rows
    # and starves the stratum it was meant to protect. Measured: a global cap fills RTL to 22 of 70.
    pool = pool.with_columns(_rank=pl.int_range(pl.len()).over("domain", "stratum")).filter(
        pl.col("_rank") < MAX_PER_DOMAIN
    )

    drawn = []
    for stratum in STRATA:
        rows = pool.filter(pl.col("stratum") == stratum.name).head(stratum.target)
        if rows.height < stratum.target:
            logger.warning("stratum %s: %d available, %d wanted", stratum.name, rows.height, stratum.target)
        drawn.append(rows)
    selection = pl.concat(drawn, how="vertical")

    rng = random.Random(seed)
    order = list(range(selection.height))
    rng.shuffle(order)
    control = {order[index] for index in range(min(STYLE_CONTROL_SIZE, len(order)))}
    human = {order[index] for index in range(min(HUMAN_SUBSET_SIZE, len(order)))}
    return selection.with_columns(
        packet_id=pl.format("doc_{}", pl.int_range(pl.len()).cast(pl.String).str.zfill(4)),
        style_control=pl.int_range(pl.len()).is_in(sorted(control)),
        human_subset=pl.int_range(pl.len()).is_in(sorted(human)),
    )


# ---------------------------------------------------------------------------
# One document's packet
# ---------------------------------------------------------------------------


_BLANK_RUN = re.compile(r"\n{3,}")


def excerpt(text: str) -> str:
    """A page's extraction, whitespace-tidied and clipped to a readable length."""
    collapsed = _BLANK_RUN.sub("\n\n", text).strip()
    if len(collapsed) <= EXCERPT_CHARS:
        return collapsed or "(this route produced no text for this page)"
    return f"{collapsed[:EXCERPT_CHARS]}\n... [clipped at {EXCERPT_CHARS} characters]"


def render_page(doc: pymupdf.Document, index: int, destination: Path) -> None:
    doc.load_page(index).get_pixmap(dpi=RENDER_DPI).save(destination)


def document_markdown(
    row: dict,
    choices: list[PageChoice],
    images: dict[int, str],
    pages_by_route: dict[str, list[str]],
    labels: dict[str, str],
    presentation: Presentation,
) -> str:
    """One packet's judging document: the pages, and every route's reading of them, blinded."""
    sections = [
        f"# Document {row['packet_id']}",
        f"pages in PDF: {row['num_pages']}",
        "",
        "Each section below is one page of this PDF. The rendered page image is the ground truth; "
        "the three extractions are what three different systems read off it.",
        "",
    ]
    for choice in choices:
        sections.append(f"## Page {choice.page_index + 1} (image: {images[choice.page_index]})")
        for route in ROUTES:
            index = choice.source_index[route]
            text = pages_by_route[route][index] if index is not None else ""
            if presentation is Presentation.CANONICAL:
                text = route_agreement.ROUTES_BY_NAME[route].canonical(text)
            sections.append(f"### Extraction {labels[route]}")
            sections.append(excerpt(text))
            sections.append("")
    return "\n".join(sections)


def write_document(row: dict, inspector_pages: list[str] | None, destination: Path, rng: random.Random) -> dict:
    """Write one document's packet in both presentations and return its blinding key entry."""
    pages_by_route = {
        "vlm": route_agreement.split_pages(row["text"], row["page_offsets"]),
        "docling": route_agreement.split_pages(row["docling_text"], row["docling_page_offsets"]),
        "inspector": inspector_pages if inspector_pages is not None else [],
    }
    choices = informative_pages(pages_by_route, PAGES_PER_DOCUMENT)
    if not choices:
        raise ValueError("no informative pages")

    labels = dict(zip(ROUTES, rng.sample(list(BLIND_LABELS), len(BLIND_LABELS)), strict=True))
    destination.mkdir(parents=True, exist_ok=True)

    images: dict[int, str] = {}
    with pymupdf.open(stream=row["pdf"], filetype="pdf") as doc:
        for choice in choices:
            if choice.page_index >= len(doc):
                continue
            name = f"page_{choice.page_index + 1:03d}.png"
            render_page(doc, choice.page_index, destination / name)
            images[choice.page_index] = name
    choices = [choice for choice in choices if choice.page_index in images]
    if not choices:
        raise ValueError("no page rendered")

    for presentation in Presentation:
        (destination / f"document_{presentation}.md").write_text(
            document_markdown(row, choices, images, pages_by_route, labels, presentation)
        )

    return {
        "packet_id": row["packet_id"],
        "source_id": row["source_id"],
        "url": row["url"],
        # Carried so the analysis can report a stratum's domain count beside its document count:
        # near-duplicates cluster by publisher, so domains are the independent unit and documents
        # are not.
        "domain": row["domain"],
        "stratum": row["stratum"],
        "style_control": bool(row["style_control"]),
        "human_subset": bool(row["human_subset"]),
        "num_pages": row["num_pages"],
        "inspector_missing": inspector_pages is None,
        # The blinding: which route each shown label actually is. Kept out of the packet directory.
        "labels": {label: route for route, label in labels.items()},
        "pages": [
            {
                "page_index": choice.page_index,
                "image": images[choice.page_index],
                "reason": choice.reason,
                "source_index": choice.source_index,
                "bigram_recall_vs_vlm": choice.recalls,
            }
            for choice in choices
        ],
        "document_metrics": {
            name: row[name]
            for name in (
                "inspector_vlm_bigram_recall_mean",
                "docling_vlm_bigram_recall_mean",
                "inspector_docling_bigram_recall_mean",
                "inspector_ok",
                "docling_ok",
                "inspector_pdf_type",
                "mean_rtl_ratio",
                "mean_cjk_ratio",
                "mean_latin_ratio",
                "mean_column_count",
            )
        },
    }


# ---------------------------------------------------------------------------
# Worker: pdf-inspector, in a process the task is willing to lose
# ---------------------------------------------------------------------------


def worker_main() -> None:
    """Serve length-prefixed documents from stdin until the driver closes it.

    Kept in-process-isolated for Stage 0's reason: three unbounded-depth recursions over nested Form
    XObjects remain in the crate, and a stack overflow is a ``SIGSEGV`` rather than a catchable
    panic. One lost document must not cost the shard.
    """
    import faulthandler  # noqa: PLC0415 - only the disposable process needs a fault handler

    import pdf_inspector  # noqa: PLC0415 - the whole point is to import it out of process

    faulthandler.enable()
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        header = stdin.readline()
        if not header:
            return
        payload = read_exactly(stdin, json.loads(header)["size"])
        try:
            pages = [page.markdown for page in pdf_inspector.extract_pages_markdown_bytes(payload).pages]
            reply = {"pages": pages}
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:  # PyO3 derives PanicException from BaseException.
            reply = {"error": f"{type(error).__name__}: {error}"[:500]}
        stdout.write(json.dumps(reply).encode() + b"\n")
        stdout.flush()


# ---------------------------------------------------------------------------
# One shard
# ---------------------------------------------------------------------------

READ_COLUMNS = (
    "source_id",
    "url",
    "num_pages",
    "text",
    "page_offsets",
    "docling_text",
    "docling_page_offsets",
    "pdf",
)

_TASK_RESOURCES = ResourceConfig(cpu=2, ram="12g", disk="12g")
_WORKER_RESOURCES = ResourceConfig(cpu=16, ram="96g", disk="64g")
# Explicit, and not the 1 GB default: the coordinator holds shard, retry and shuffle state for every
# task, and at this shard count the default is what dies at exit 137 one task short of the end.
_COORDINATOR_RESOURCES = ResourceConfig(cpu=2, ram="16g", preemptible=False)
_MAX_WORKERS = 23
_HEARTBEAT_TIMEOUT = 30 * 60


def build_shard(work: tuple[int, str, list[dict]]) -> int:
    """Package every selected document that lives in one sample shard.

    The selection travels with the task rather than being re-derived: the draw is a property of the
    study tables, and a task that re-read them would be drawing its own sample.
    """
    index, shard, wanted = work
    if not wanted:
        return 0
    fs = storage()
    key_output = f"{KEY_SHARD_PREFIX}/part-{index:05d}.json"
    if fs.exists(key_output):
        return 0

    by_source = {row["source_id"]: row for row in wanted}
    with fs.open(shard, "rb") as stream:
        table = pl.read_parquet(stream, columns=list(READ_COLUMNS))
    table = table.filter(pl.col("source_id").is_in(list(by_source)))
    if table.height == 0:
        with fs.open(key_output, "w") as stream:
            json.dump([], stream)
        return 0

    worker = Worker(MODULE_NAME)
    entries, failures = [], []
    staging = Path("/tmp") / f"adjudication-{index:05d}"
    try:
        for document in table.iter_rows(named=True):
            selected = by_source[document["source_id"]]
            row = {**document, **selected}
            rng = random.Random(f"{SAMPLE_SEED}:{row['packet_id']}")
            reply = worker.call(EXTRACT_OP, document["pdf"])
            pages = reply.result.get("pages")
            local = staging / row["packet_id"]
            try:
                entry = write_document(row, pages, local, rng)
            except Exception as error:
                # An excluded document is a reported row, not a silent gap: the prior pass dropped
                # 11 of 72 documents and only the count made it into the report.
                failures.append({"packet_id": row["packet_id"], "source_id": row["source_id"], "reason": str(error)})
                continue
            entry["inspector_error"] = reply.result.get("error")
            remote = f"{PACKETS_PREFIX}/{row['packet_id']}"
            for path in sorted(local.iterdir()):
                with path.open("rb") as source, fs.open(f"{remote}/{path.name}", "wb") as sink:
                    sink.write(source.read())
            entries.append(entry)
    finally:
        worker.stop()

    counters.pipeline.update_counter("adjudication/packets", len(entries))
    counters.pipeline.update_counter("adjudication/excluded", len(failures))
    with fs.open(key_output, "w") as stream:
        json.dump({"entries": entries, "excluded": failures}, stream)
    logger.info("shard %d: %d packets, %d excluded", index, len(entries), len(failures))
    return len(entries)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def study_frame(fs) -> pl.DataFrame:
    """The two study tables joined, labelled, and reduced to the columns the draw reads."""
    route = label(read_table(ROUTE_STUDY_PREFIX, fs))
    inspector = read_table(INSPECTOR_STUDY_PREFIX, fs)
    frame = route.drop("url", "num_pages", "pdf_bytes", "docling_missing").join(inspector, on="source_id", how="inner")
    return frame.with_columns(
        inspector_ok=route_ok("inspector_vlm", pl.col("inspector_vlm_bigram_recall_mean").is_null()),
        docling_ok=route_ok("docling_vlm", pl.col("docling_vlm_bigram_recall_mean").is_null()),
    )


def main() -> None:
    configure_logging(logging.INFO)
    fs = storage()

    frame = usable(study_frame(fs))
    logger.info("usable %d documents, %d domains", frame.height, frame["domain"].n_unique())
    selection = select(frame, SAMPLE_SEED)
    logger.info(
        "selected %d documents across %d strata: %s",
        selection.height,
        selection["stratum"].n_unique(),
        dict(selection["stratum"].value_counts().iter_rows()),
    )
    with fs.open(SELECTION_PATH, "wb") as stream:
        selection.write_parquet(stream)

    carried = [
        "source_id",
        "packet_id",
        "domain",
        "stratum",
        "style_control",
        "human_subset",
        "inspector_vlm_bigram_recall_mean",
        "docling_vlm_bigram_recall_mean",
        "inspector_docling_bigram_recall_mean",
        "inspector_ok",
        "docling_ok",
        "inspector_pdf_type",
        "mean_rtl_ratio",
        "mean_cjk_ratio",
        "mean_latin_ratio",
        "mean_column_count",
    ]
    # One task per sample shard. Which shard holds a given document is not knowable without reading
    # it, so every task carries the whole draw and keeps the rows its own shard turns out to have;
    # at 360 rows of identifiers that broadcast is far cheaper than an index pass over 130 GB.
    wanted = selection.select(carried).to_dicts()
    work = [(index, shard, wanted) for index, shard in shards()]
    logger.info("adjudication set: %d shards -> %s", len(work), PACKETS_PREFIX)

    outcome = ZephyrContext(
        name="pdf-adjudication-set",
        resources=_WORKER_RESOURCES,
        coordinator_resources=_COORDINATOR_RESOURCES,
        max_workers=_MAX_WORKERS,
        stage_runner_factory=SubprocessRunner,
        heartbeat_timeout=_HEARTBEAT_TIMEOUT,
    ).execute(Dataset.from_list(work).map(build_shard), map_task_resources=_TASK_RESOURCES)
    logger.info("packets built, counters %s", dict(outcome.counters))

    entries, excluded = [], []
    for path in sorted(fs.glob(f"{KEY_SHARD_PREFIX}/*.json")):
        with fs.open(path, "r") as stream:
            payload = json.load(stream)
        if isinstance(payload, dict):
            entries.extend(payload["entries"])
            excluded.extend(payload["excluded"])
    entries.sort(key=lambda entry: entry["packet_id"])

    with fs.open(KEY_PATH, "w") as stream:
        json.dump(entries, stream, indent=2)
    manifest = {
        "packets": len(entries),
        "requested": selection.height,
        "excluded": excluded,
        "render_dpi": RENDER_DPI,
        "pages_per_document": PAGES_PER_DOCUMENT,
        "routes": list(ROUTES),
        "presentations": [str(presentation) for presentation in Presentation],
        "style_control": sum(entry["style_control"] for entry in entries),
        "human_subset": sorted(entry["packet_id"] for entry in entries if entry["human_subset"]),
        "by_stratum": {stratum.name: sum(entry["stratum"] == stratum.name for entry in entries) for stratum in STRATA},
        "strata": [{"name": s.name, "target": s.target, "rationale": s.rationale} for s in STRATA],
    }
    with fs.open(MANIFEST_PATH, "w") as stream:
        json.dump(manifest, stream, indent=2)
    logger.info("wrote %s and %s: %s", KEY_PATH, MANIFEST_PATH, manifest["by_stratum"])


if __name__ == "__main__":
    if WORKER_FLAG in sys.argv:
        worker_main()
    else:
        main()
