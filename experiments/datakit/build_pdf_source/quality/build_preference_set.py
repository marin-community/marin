# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Package documents for blind adjudication of pdf-inspector against the VLM.

The router this feeds decides one thing: does a document stay on pdf-inspector, or does it go to
the VLM. :mod:`~experiments.datakit.build_pdf_source.quality.build_adjudication_set` packaged three
routes because the question then was which cheap extractor to keep; that is settled, Docling is
gone, and a third route in the packet now costs tokens and judge attention for a comparison nobody
will act on. This module writes the two-route version of the same packet, over the same corpus, for
the same kind of verdict.

**The target is a pairwise preference, not an agreement metric.** The router that shipped learned
``docling_ok`` -- bigram recall >= 0.80 against the VLM -- and blind adjudication proved that label
invalid as a *quality* ranking: on documents the label called fine, the other route still won 41-43%
of head-to-heads, and the label separated preference by 0.015 (0.404 True against 0.419 False),
which is inside the noise. Agreement cannot rank two extractions because it has no opinion about
which of them is right; it measures distance from one of the two candidates. The ground truth here
is the rendered page, which is what both routes were reading, and the label is which of them read it
better.

**VLM-damaged documents are labelled, not dropped.** Every previous pass filtered to the
``trustworthy`` subset -- no truncated, failed or unrendered pages, no loop repair, nothing below the
legibility floor -- because on those rows an agreement number measures the VLM's failure rather than
the cheap route's. Under a preference label that reasoning inverts. The packet shows the VLM's
*actual production output*, damage included, so a judge looking at a truncated transcription against
a complete one prefers the complete one, and the router learns not to escalate a document the VLM
will botch. That is 16.7% of the sample and it is exactly the behaviour
``pdf-extraction-routing.md`` had to leave as a separate gate. Folding it into the label costs
nothing and removes a gate.

**The draw is spread over domains rather than stratified over failure modes.** The adjudication
study drew 605 documents across ~245 domains, oversampling the axes where the routes were suspected
to diverge, because it was answering a question about specific strata. A training set wants the
opposite: the crawl holds ~9.8% exact duplicates and many more near-duplicates per publisher, the
evaluation split has to be domain-disjoint, and the sample's 2,589 domains are therefore the real
sample size. :data:`MAX_PER_DOMAIN` documents from every one of them reaches
:data:`TARGET_DOCUMENTS` while touching every domain the corpus has -- measured, the cap is what
binds, not the target: at 15 the pool holds 20,473 documents and all 2,589 domains.

A stratum is still recorded on every packet (:func:`stratum_of`, imported from the adjudication
study so the two label sets segment the corpus the same way), because a per-stratum read of the
verdicts is how the report says where the router's decision is hard. It just does not steer the
draw.

**One object per packet.** The adjudication set wrote each page image and each markdown document as
its own object, which is four objects per packet and four round trips per verdict. At 605 packets
that is invisible; at :data:`TARGET_DOCUMENTS` it is 80,000 objects and it makes the judging pass
I/O-bound on object listing. A packet is one JSON holding the markdown and the page images inline,
so the judge does one read.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name pdf-preference-set --extra pdf \\
        --cpu 8 --memory 24GB --disk 16GB --enable-extra-resources \\
        -- python -m experiments.datakit.build_pdf_source.quality.build_preference_set

``--extra pdf`` is required: pdf-inspector's text is not stored by
:mod:`~experiments.datakit.build_pdf_source.quality.build_inspector_study`, which keeps only
agreement columns and signals, so the extraction is re-run here against the same PDF bytes.
"""

import base64
import json
import logging
import random
import sys
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import version

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
from experiments.datakit.build_pdf_source.quality.build_adjudication_set import (
    EXCERPT_CHARS,
    PAGES_PER_DOCUMENT,
    RENDER_DPI,
    excerpt,
    stratum_of,
)
from experiments.datakit.build_pdf_source.quality.build_inspector_study import (
    OUTPUT_PREFIX as INSPECTOR_STUDY_PREFIX,
)
from experiments.datakit.build_pdf_source.quality.build_route_study import shards, storage
from experiments.datakit.build_pdf_source.quality.probe_pdf_inspector import WORKER_FLAG, Worker, read_exactly

logger = logging.getLogger(__name__)

STUDY_ROOT = "s3://marin-us-east-02a/marin/data/pdf_quality"
ROUTE_STUDY_PREFIX = f"{STUDY_ROOT}/cc_focus_2026_22_route_study"

LIBRARY_VERSION = "1.17.0"
OUTPUT_PREFIX = f"{STUDY_ROOT}/cc_focus_2026_22_preference_{LIBRARY_VERSION.replace('.', '_')}"

PACKETS_PREFIX = f"{OUTPUT_PREFIX}/packets"
KEY_SHARD_PREFIX = f"{OUTPUT_PREFIX}/key_parts"
KEY_PATH = f"{OUTPUT_PREFIX}/key.json"
MANIFEST_PATH = f"{OUTPUT_PREFIX}/manifest.json"
SELECTION_PATH = f"{OUTPUT_PREFIX}/selection.parquet"

MODULE_NAME = "experiments.datakit.build_pdf_source.quality.build_preference_set"
EXTRACT_OP = "extract"

# Documents to draw, and the per-domain cap that spreads them. The cap is the binding constraint,
# not the target: 2,589 domains at 15 apiece is 20,473 available documents, so a target near 20,000
# takes essentially every document the cap allows and touches every domain in the corpus. Raising
# the target past the cap's pool would mean deepening publishers rather than widening coverage,
# which buys near-duplicates and reports them as independent evidence.
TARGET_DOCUMENTS = 20_000
MAX_PER_DOMAIN = 15
SAMPLE_SEED = 20260826

# Documents judged a second time by a second model, for the consistency-derived graded target and
# the inter-judge number. Sized for a tight interval at a small share of the spend, and drawn as a
# prefix of the same shuffle so it is a uniform subsample of the draw rather than its own stratum.
SECOND_JUDGE_SIZE = 3_000

ROUTES = ("inspector", "vlm")
BLIND_LABELS = ("A", "B")


class Outcome(StrEnum):
    """Why a document is or is not in the judged set.

    ``inspector_failed`` and ``vlm_failed`` are decided rather than judged: a route that produced
    nothing has lost the document outright, so the packet would ask a judge to compare text against
    an empty string. Both are recorded as labels rather than dropped, because a router that never
    sees them cannot learn the decision they imply.

    ``inspector_failed`` covers two different events, and the difference matters downstream. The
    library raising is rare -- 269 documents in 100,000. An extraction that *succeeds* and returns
    nothing is not: it is ~9% of the draw, and it is what a scanned page looks like to a text
    extractor. Both mean the same thing to the router, so both are one outcome here;
    ``inspector_error`` distinguishes them on the row, and the analysis reads the distinction off
    ``inspector_markdown_chars`` in the study table.
    """

    JUDGED = "judged"
    INSPECTOR_FAILED = "inspector_failed"
    VLM_FAILED = "vlm_failed"
    NO_PAGES = "no_pages"


# ---------------------------------------------------------------------------
# Page selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageChoice:
    """A page selected for adjudication, and where each route's text for it sits.

    ``source_index`` is ``None`` for a route with no page matching this one, which is real and
    common: routes drop pages they read nothing from, and pairing by index after such a drop shows
    every later page beside its neighbour's text. That bug invalidated an entire earlier pass, so
    the mapping is content-based (:func:`route_agreement.align_pages`) and carried explicitly.
    """

    page_index: int
    source_index: dict[str, int | None]
    reason: str
    inspector_recall: float


def informative_pages(inspector_pages: list[str], vlm_pages: list[str], count: int) -> list[PageChoice]:
    """Pick the pages worth adjudicating: where the two routes are furthest apart, plus a control.

    The VLM's page list is the PDF's own, so it is the reference pdf-inspector is aligned to and the
    page number shown to a judge is the PDF's. Pages are ranked by pdf-inspector's bigram recall
    against the VLM, lowest first, because that is where the verdict is actually decided -- the
    first page of a report is a title page both routes get right and it says nothing about the
    document.

    One page the routes agree on is added as a control, so a judge sees what this document's
    typography looks like when nothing has gone wrong and the verdict rests on the difference rather
    than on unfamiliarity with the layout.
    """
    aligned = {
        vlm_index: inspector_index
        for inspector_index, vlm_index in route_agreement.align_pages(inspector_pages, vlm_pages)
        if vlm_index is not None and inspector_index is not None
    }
    scored: list[PageChoice] = []
    for vlm_index, vlm_page in enumerate(vlm_pages):
        inspector_index = aligned.get(vlm_index)
        inspector_page = inspector_pages[inspector_index] if inspector_index is not None else ""
        measured = route_agreement.page_agreement(
            vlm_page, inspector_page, route_agreement.VLM, route_agreement.INSPECTOR
        )
        # A page neither route read anything on adjudicates nothing, and on a scanned document it
        # would be most of what a judge is shown.
        if not route_agreement.markdown_streams(vlm_page).tokens and measured.candidate_tokens == 0:
            continue
        scored.append(
            PageChoice(
                page_index=vlm_index,
                source_index={"vlm": vlm_index, "inspector": inspector_index},
                reason="disagreement",
                inspector_recall=measured.bigram_recall,
            )
        )
    if not scored:
        return []

    scored.sort(key=lambda choice: choice.inspector_recall)
    chosen = scored[: max(count - 1, 1)]
    control = scored[-1]
    if control.page_index not in {choice.page_index for choice in chosen}:
        chosen.append(
            PageChoice(
                page_index=control.page_index,
                source_index=control.source_index,
                reason="control",
                inspector_recall=control.inspector_recall,
            )
        )
    return sorted(chosen, key=lambda choice: choice.page_index)


# ---------------------------------------------------------------------------
# The draw
# ---------------------------------------------------------------------------


def eligible(frame: pl.DataFrame) -> pl.DataFrame:
    """The rows a preference verdict can be read from.

    Deliberately *not* filtered on ``trustworthy``. A document whose VLM extraction is truncated or
    loop-repaired is one the router must learn to leave alone, and the only way it learns that is
    from a packet showing what the VLM actually produced. The rows removed here are the ones where
    the comparison has no second side at all -- see :class:`Outcome`, which records them as decided
    labels rather than as gaps.
    """
    return frame.filter(pl.col("feature_error").is_null() & (pl.col("domain") != ""))


def select(frame: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Draw at most :data:`MAX_PER_DOMAIN` documents per registered domain, up to the target.

    The shuffle is within a domain, so a publisher's contribution is a random sample of its
    documents rather than whatever the shard order put first.

    The draw is then taken **in depth order across domains**: every domain's first document before
    any domain's second, every domain's second before any domain's third. A plain subsample of the
    capped pool loses whole domains off the tail -- measured, 16 of them at this target -- and
    domains are the sample size a domain-disjoint split can spend, so losing one costs more than
    losing the fifteen documents a deep publisher would have contributed instead.
    """
    pool = frame.sample(fraction=1.0, shuffle=True, seed=seed)
    pool = pool.with_columns(_rank=pl.int_range(pl.len()).over("domain")).filter(pl.col("_rank") < MAX_PER_DOMAIN)
    logger.info("capped pool: %d documents, %d domains", pool.height, pool["domain"].n_unique())

    selection = pool.sort("_rank").head(TARGET_DOCUMENTS).sample(fraction=1.0, shuffle=True, seed=seed + 1)
    return selection.drop("_rank").with_columns(
        packet_id=pl.format("pref_{}", pl.int_range(pl.len()).cast(pl.String).str.zfill(6)),
        stratum=stratum_of(),
        # A prefix of an already-shuffled frame, so the second-judge subset is uniform over the draw.
        second_judge=pl.int_range(pl.len()) < SECOND_JUDGE_SIZE,
    )


# ---------------------------------------------------------------------------
# One document's packet
# ---------------------------------------------------------------------------


def render_page(document: pymupdf.Document, index: int) -> bytes:
    return document.load_page(index).get_pixmap(dpi=RENDER_DPI).tobytes("png")


def packet_markdown(
    packet_id: str,
    num_pages: int,
    choices: list[PageChoice],
    pages_by_route: dict[str, list[str]],
    labels: dict[str, str],
) -> str:
    """One packet's judging document: the pages, and both routes' reading of them, blinded.

    Both routes are re-rendered by :func:`route_agreement.Route.canonical` before they are shown.
    pdf-inspector and the VLM both emit Markdown, so unlike the three-route packets this is close to
    a no-op -- but it is the presentation the human calibration was performed under, and keeping it
    is what makes that calibration transfer to these verdicts.
    """
    sections = [
        f"# Document {packet_id}",
        f"pages in PDF: {num_pages}",
        "",
        f"Each section below is one page of this PDF, in the same order as the {len(choices)} images "
        "attached above. The rendered page image is the ground truth; the two extractions are what "
        "two different systems read off it.",
        "",
    ]
    for position, choice in enumerate(choices, start=1):
        sections.append(f"## Page {choice.page_index + 1} of the PDF (image {position} of {len(choices)})")
        for route in ROUTES:
            index = choice.source_index[route]
            text = pages_by_route[route][index] if index is not None else ""
            sections.append(f"### Extraction {labels[route]}")
            sections.append(excerpt(route_agreement.ROUTES_BY_NAME[route].canonical(text)))
            sections.append("")
    return "\n".join(sections)


def write_packet(row: dict, inspector_pages: list[str] | None, fs, rng: random.Random) -> dict:
    """Write one document's packet object and return its key entry.

    The key -- which blind label each route hides behind -- is returned rather than written into the
    packet, so nothing a judge reads carries the answer.
    """
    vlm_pages = route_agreement.split_pages(row["text"], row["page_offsets"])
    entry = {
        "packet_id": row["packet_id"],
        "source_id": row["source_id"],
        "url": row["url"],
        "domain": row["domain"],
        "stratum": row["stratum"],
        "second_judge": bool(row["second_judge"]),
        "num_pages": row["num_pages"],
        "trustworthy": bool(row["trustworthy"]),
    }
    if inspector_pages is None or not any(page.strip() for page in inspector_pages):
        return {**entry, "outcome": str(Outcome.INSPECTOR_FAILED)}
    if not vlm_pages or not any(page.strip() for page in vlm_pages):
        return {**entry, "outcome": str(Outcome.VLM_FAILED)}

    pages_by_route = {"vlm": vlm_pages, "inspector": inspector_pages}
    choices = informative_pages(inspector_pages, vlm_pages, PAGES_PER_DOCUMENT)
    if not choices:
        return {**entry, "outcome": str(Outcome.NO_PAGES)}

    labels = dict(zip(ROUTES, rng.sample(list(BLIND_LABELS), len(BLIND_LABELS)), strict=True))
    images = []
    with pymupdf.open(stream=row["pdf"], filetype="pdf") as document:
        for choice in choices:
            if choice.page_index >= len(document):
                continue
            images.append((choice, render_page(document, choice.page_index)))
    if not images:
        return {**entry, "outcome": str(Outcome.NO_PAGES)}

    chosen = [choice for choice, _ in images]
    packet = {
        "packet_id": row["packet_id"],
        "markdown": packet_markdown(row["packet_id"], row["num_pages"], chosen, pages_by_route, labels),
        "images": [base64.b64encode(payload).decode() for _, payload in images],
    }
    with fs.open(f"{PACKETS_PREFIX}/{row['packet_id']}.json", "w") as stream:
        json.dump(packet, stream)

    return {
        **entry,
        "outcome": str(Outcome.JUDGED),
        # The blinding: which route each shown label actually is. Kept out of the packet object.
        "labels": {blind: route for route, blind in labels.items()},
        "pages": [
            {
                "page_index": choice.page_index,
                "reason": choice.reason,
                "source_index": choice.source_index,
                "inspector_bigram_recall_vs_vlm": choice.inspector_recall,
            }
            for choice in chosen
        ],
    }


# ---------------------------------------------------------------------------
# Worker: pdf-inspector, in a process the task is willing to lose
# ---------------------------------------------------------------------------


def worker_main() -> None:
    """Serve length-prefixed documents from stdin until the driver closes it.

    Process-isolated for Stage 0's reason: three unbounded-depth recursions over nested Form
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


READ_COLUMNS = ("source_id", "url", "num_pages", "text", "page_offsets", "pdf")

_TASK_RESOURCES = ResourceConfig(cpu=2, ram="12g", disk="12g")
_WORKER_RESOURCES = ResourceConfig(cpu=16, ram="96g", disk="64g")
# Explicit, and not the 1 GB default: the coordinator holds shard, retry and shuffle state for every
# task, and at this shard count the default is what dies at exit 137 one task short of the end.
_COORDINATOR_RESOURCES = ResourceConfig(cpu=2, ram="16g", preemptible=False)
# 178 input shards is the cross-task parallelism ceiling; eight tasks to a 16-core worker covers it
# with a slot to spare rather than leaving a shard queued behind a finished worker.
_MAX_WORKERS = 23
_HEARTBEAT_TIMEOUT = 30 * 60


def build_shard(work: tuple[int, str, list[dict]]) -> int:
    """Package every selected document that lives in one sample shard.

    The selection travels with the task rather than being re-derived: the draw is a property of the
    study tables, and a task that re-read them would be drawing its own sample.
    """
    index, shard, wanted = work
    fs = storage()
    key_output = f"{KEY_SHARD_PREFIX}/part-{index:05d}.json"
    if fs.exists(key_output):
        return 0

    by_source = {row["source_id"]: row for row in wanted}
    with fs.open(shard, "rb") as stream:
        table = pl.read_parquet(stream, columns=list(READ_COLUMNS))
    table = table.filter(pl.col("source_id").is_in(list(by_source)))

    worker = Worker(MODULE_NAME)
    entries = []
    try:
        for document in table.iter_rows(named=True):
            row = {**document, **by_source[document["source_id"]]}
            rng = random.Random(f"{SAMPLE_SEED}:{row['packet_id']}")
            reply = worker.call(EXTRACT_OP, document["pdf"])
            try:
                entry = write_packet(row, reply.result.get("pages"), fs, rng)
            except Exception as error:
                # An excluded document is a reported row, not a silent gap.
                entry = {
                    "packet_id": row["packet_id"],
                    "source_id": row["source_id"],
                    "domain": row["domain"],
                    "stratum": row["stratum"],
                    "second_judge": bool(row["second_judge"]),
                    "num_pages": row["num_pages"],
                    "trustworthy": bool(row["trustworthy"]),
                    "outcome": str(Outcome.NO_PAGES),
                    "error": str(error)[:500],
                }
            entry["inspector_error"] = reply.result.get("error")
            entries.append(entry)
    finally:
        worker.stop()

    judged = sum(entry["outcome"] == str(Outcome.JUDGED) for entry in entries)
    counters.pipeline.update_counter("preference/packets", judged)
    counters.pipeline.update_counter("preference/decided", len(entries) - judged)
    with fs.open(key_output, "w") as stream:
        json.dump(entries, stream)
    logger.info("shard %d: %d packets, %d decided without a judge", index, judged, len(entries) - judged)
    return judged


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

CARRIED = (
    "source_id",
    "packet_id",
    "domain",
    "stratum",
    "second_judge",
    "trustworthy",
)


def study_frame(fs) -> pl.DataFrame:
    """The two study tables joined and reduced to the columns the draw reads."""
    route = label(read_table(ROUTE_STUDY_PREFIX, fs))
    inspector = read_table(INSPECTOR_STUDY_PREFIX, fs).select(
        "source_id",
        "domain",
        "inspector_pdf_type",
        "inspector_page_count",
        "inspector_extract_pages_with_tables",
        "inspector_vlm_bigram_recall_mean",
        "inspector_vlm_frac_pages_bigram_below_50",
        "inspector_docling_bigram_recall_mean",
    )
    frame = route.drop("url", "pdf_bytes", strict=False).join(inspector, on="source_id", how="inner")
    # `stratum_of` reads both cheap routes' labels, because one of its strata is the documents both
    # fail while agreeing closely with each other. The strata are recorded rather than drawn on, but
    # they are the segmentation the report reads the verdicts out by, so they stay the ones the
    # adjudication study defined.
    return frame.with_columns(
        inspector_ok=route_ok("inspector_vlm", pl.col("inspector_vlm_bigram_recall_mean").is_null())
    )


def main() -> None:
    configure_logging(logging.INFO)
    installed = version("pdf-inspector")
    if installed != LIBRARY_VERSION:
        raise RuntimeError(f"{OUTPUT_PREFIX} is the {LIBRARY_VERSION} packet set; pdf-inspector {installed} installed")
    fs = storage()

    frame = eligible(study_frame(fs))
    logger.info("eligible %d documents, %d domains", frame.height, frame["domain"].n_unique())
    selection = select(frame, SAMPLE_SEED)
    logger.info(
        "draw: %d documents, %d domains, %.1f%% VLM-trustworthy, strata %s",
        selection.height,
        selection["domain"].n_unique(),
        100 * selection["trustworthy"].mean(),
        dict(selection["stratum"].value_counts().iter_rows()),
    )
    with fs.open(SELECTION_PATH, "wb") as stream:
        selection.write_parquet(stream)

    # One task per sample shard. Which shard holds a given document is not knowable without reading
    # it, so every task carries the whole draw and keeps the rows its own shard turns out to have;
    # at 20,000 rows of identifiers that broadcast is far cheaper than an index pass over 126 GB.
    wanted = selection.select(CARRIED).to_dicts()
    work = [(index, shard, wanted) for index, shard in shards()]
    logger.info("preference set: %d shards -> %s", len(work), PACKETS_PREFIX)

    outcome = ZephyrContext(
        name="pdf-preference-set",
        resources=_WORKER_RESOURCES,
        coordinator_resources=_COORDINATOR_RESOURCES,
        max_workers=_MAX_WORKERS,
        stage_runner_factory=SubprocessRunner,
        heartbeat_timeout=_HEARTBEAT_TIMEOUT,
    ).execute(Dataset.from_list(work).map(build_shard), map_task_resources=_TASK_RESOURCES)
    logger.info("packets built, counters %s", dict(outcome.counters))

    entries = []
    for path in sorted(fs.glob(f"{KEY_SHARD_PREFIX}/*.json")):
        with fs.open(path, "r") as stream:
            entries.extend(json.load(stream))
    entries.sort(key=lambda entry: entry["packet_id"])

    judged = [entry for entry in entries if entry["outcome"] == str(Outcome.JUDGED)]
    manifest = {
        "library_version": LIBRARY_VERSION,
        "requested": selection.height,
        "built": len(entries),
        "judged": len(judged),
        "domains": len({entry["domain"] for entry in judged}),
        "second_judge": sum(entry["second_judge"] for entry in judged),
        "outcomes": dict(
            pl.DataFrame({"outcome": [entry["outcome"] for entry in entries]})["outcome"].value_counts().iter_rows()
        ),
        "excerpt_chars": EXCERPT_CHARS,
        "pages_per_document": PAGES_PER_DOCUMENT,
        "render_dpi": RENDER_DPI,
        "max_per_domain": MAX_PER_DOMAIN,
        "seed": SAMPLE_SEED,
    }
    with fs.open(KEY_PATH, "w") as stream:
        json.dump(entries, stream)
    with fs.open(MANIFEST_PATH, "w") as stream:
        json.dump(manifest, stream, indent=2)
    logger.info("manifest %s", json.dumps(manifest))


if __name__ == "__main__":
    if WORKER_FLAG in sys.argv:
        worker_main()
    else:
        main()
