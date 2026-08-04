# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local browser for the 10k-document PDF OCR sample (issue 7616 exploration).

Loads sample metadata and OCR text into memory at startup, joins the model
reasoning texts from the quality scoring checkpoints, and serves a two-pane
viewer (rendered PDF on the left, OCR text on the right).

Three educational-quality score sets are supported side by side: v1 (grade-school
capped rubric), v2 (the revised primary-school-through-graduate rubric), and ft
(the trained pooled fast-transformer). The oracle versions have their own score
columns and reasoning directories; ft has continuous score columns and no
reasoning. Filtering and the displayed chips follow the selected version; the
other versions' scores stay visible as secondary numbers for eyeballing
disagreements.

By default the browser is restricted to documents carrying an ft score. The
scorer is trained on a document-disjoint split, so only its holdout is scored --
showing a model score on a document the model trained on would display memorised
agreement as if it were prediction.

The sample parquets gain columns while scoring/fetch jobs run. Anything the
schema does not yet carry is reported as pending; restart the app to pick up
newly landed columns.

Launch:
    uv run experiments/build_pdf_source/_sample_browser/app.py
"""

import argparse
import glob
import logging
import os
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("sample_browser")

SCRATCH_DIR = "/private/tmp/claude-501/-Users-k3sc0re-openathena-marin/fad03565-29e1-47c5-a037-102cd7973876/scratchpad"

SAMPLE_DIR = "/tmp/cc_focus_2026_22_pdf_ocr_all_sample10k"
SCORES_DIR = os.path.join(SCRATCH_DIR, "quality")
SCORES_V2_DIR = os.path.join(SCRATCH_DIR, "quality_v2")
PDF_CACHE_DIR = Path.home() / ".cache" / "sample_browser_pdfs"
STATIC_DIR = Path(__file__).parent / "static"

PDF_COLUMN = "pdf"
PDF_STATUS_COLUMN = "pdf_fetch_status"
SEGMENTS = ("begin", "middle", "end")

V1 = "v1"
V2 = "v2"
FT = "ft"
VERSIONS = (V1, V2, FT)
# Only the oracle versions have per-segment reasoning checkpoints on disk.
REASONING_VERSIONS = (V1, V2)
VERSION_LABELS = {
    V1: "v1 · grade-school",
    V2: "v2 · through-graduate",
    FT: "ft · model (unseen docs)",
}

# Per-version score column names in the sample parquets. The oracle columns are
# int8 0..4; the ft columns are float32 on the same 0..4 scale.
SCORE_COLUMNS = {
    V1: {segment: f"edu_score_{segment}" for segment in SEGMENTS},
    V2: {segment: f"edu_score_v2_{segment}" for segment in SEGMENTS},
    FT: {segment: f"ft_score_{segment}" for segment in SEGMENTS},
}

# Score-slider bounds. The rubric tops out at 5 even though this sample reaches 4.
SCORE_MIN = 0.0
SCORE_MAX = 5.0

# Columns held in memory. `text` is kept separately; `pdf` is always read on demand.
METADATA_COLUMNS = [
    "id",
    "source_id",
    "source",
    "warc_filename",
    "warc_record_offset",
    "content_digest",
    "url",
    "num_pages",
    "page_offsets",
    "extraction_status",
    "extraction_error",
    "boilerplate_lines_removed",
    "pages_ocred",
    "pages_failed",
    "pages_truncated",
    "pages_unrendered",
    "mean_render_dpi",
    "pages_below_legibility_floor",
    "completion_tokens",
]

DEFAULT_PAGE_LIMIT = 50


@dataclass
class Document:
    doc_id: str
    text: str
    meta: dict
    title: str
    title_lower: str
    text_lower: str
    page_offsets: list[int]
    # version -> segment -> score (oracle versions are ints, ft is a float)
    scores: dict[str, dict[str, float | None]]
    # Location for on-demand PDF reads.
    file_path: str
    row_group: int
    row_in_group: int


@dataclass
class Sample:
    docs: list[Document]
    by_id: dict[str, Document]
    # version -> (id, segment) -> {"score", "response"}
    reasoning: dict[str, dict[tuple[str, str], dict]] = field(default_factory=dict)
    has_pdf_column: bool = False
    has_pdf_status_column: bool = False
    # version -> segment -> whether the score column exists in the parquets
    score_columns_present: dict[str, dict[str, bool]] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)

    def version_has_scores(self, version: str) -> bool:
        return any(self.score_columns_present.get(version, {}).values())

    def version_has_reasoning(self, version: str) -> bool:
        return bool(self.reasoning.get(version))

    def version_available(self, version: str) -> bool:
        """Whether anything is scoreable for this version, from columns or reasoning."""
        return self.version_has_scores(version) or self.version_has_reasoning(version)


def url_title(url: str) -> str:
    """Basename of the URL path, percent-decoded; falls back to the host."""
    path = urllib.parse.urlsplit(url).path
    base = path.rstrip("/").rsplit("/", 1)[-1]
    if not base:
        return urllib.parse.urlsplit(url).netloc or url
    return urllib.parse.unquote(base)


def load_reasoning(scores_dir: str) -> dict[tuple[str, str], dict]:
    """Join scoring checkpoints, later files superseding earlier ones per (id, segment).

    Rows with a negative score (scoring failure) never displace a successful row.
    A missing directory yields an empty mapping: the version is simply pending.
    """
    out: dict[tuple[str, str], dict] = {}
    paths = sorted(glob.glob(os.path.join(scores_dir, "scores-*.parquet")))
    for path in paths:
        table = pq.read_table(path, columns=["id", "segment", "score", "response"])
        for row in table.to_pylist():
            key = (row["id"], row["segment"])
            score = row["score"]
            existing = out.get(key)
            if existing is not None and score is not None and score < 0 and existing["score"] >= 0:
                continue
            out[key] = {"score": score, "response": row["response"]}
    logger.info("loaded %d reasoning rows from %d checkpoints in %s", len(out), len(paths), scores_dir)
    return out


def load_sample(sample_dir: str, scores_dirs: dict[str, str], restrict_to: str | None = None) -> Sample:
    paths = sorted(glob.glob(os.path.join(sample_dir, "sample-*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no sample-*.parquet under {sample_dir}")

    schema = pq.ParquetFile(paths[0]).schema_arrow
    present = set(schema.names)
    score_present = {version: {seg: SCORE_COLUMNS[version][seg] in present for seg in SEGMENTS} for version in VERSIONS}
    read_columns = ["text", *METADATA_COLUMNS]
    for version in VERSIONS:
        read_columns += [SCORE_COLUMNS[version][seg] for seg in SEGMENTS if score_present[version][seg]]
    if PDF_STATUS_COLUMN in present:
        read_columns.append(PDF_STATUS_COLUMN)

    docs: list[Document] = []
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        row_group_starts = []
        cursor = 0
        for group in range(parquet_file.metadata.num_row_groups):
            row_group_starts.append(cursor)
            cursor += parquet_file.metadata.row_group(group).num_rows
        table = parquet_file.read(columns=read_columns)
        for index, row in enumerate(table.to_pylist()):
            group = 0
            while group + 1 < len(row_group_starts) and row_group_starts[group + 1] <= index:
                group += 1
            text = row.pop("text")
            scores = {
                version: {
                    seg: row.pop(SCORE_COLUMNS[version][seg]) if score_present[version][seg] else None
                    for seg in SEGMENTS
                }
                for version in VERSIONS
            }
            title = url_title(row["url"])
            docs.append(
                Document(
                    doc_id=row["id"],
                    text=text,
                    meta=row,
                    title=title,
                    title_lower=title.lower(),
                    text_lower=text.lower(),
                    page_offsets=list(row["page_offsets"]),
                    scores=scores,
                    file_path=path,
                    row_group=group,
                    row_in_group=index - row_group_starts[group],
                )
            )
        logger.info("loaded %s (%d rows)", os.path.basename(path), table.num_rows)

    docs.sort(key=lambda d: d.doc_id)
    if restrict_to and any(score_present[restrict_to].values()):
        kept = [d for d in docs if any(d.scores[restrict_to][seg] is not None for seg in SEGMENTS)]
        logger.info("restricting to %d/%d docs carrying %s scores", len(kept), len(docs), restrict_to)
        docs = kept
    sample = Sample(
        docs=docs,
        by_id={d.doc_id: d for d in docs},
        reasoning={version: load_reasoning(scores_dirs[version]) for version in REASONING_VERSIONS},
        has_pdf_column=PDF_COLUMN in present,
        has_pdf_status_column=PDF_STATUS_COLUMN in present,
        score_columns_present=score_present,
        files=paths,
    )
    backfilled = backfill_scores_from_reasoning(sample)
    logger.info(
        "sample ready: %d docs, pdf column=%s, score columns=%s, backfilled from reasoning=%s",
        len(docs),
        sample.has_pdf_column,
        score_present,
        backfilled,
    )
    return sample


def backfill_scores_from_reasoning(sample: Sample) -> dict[str, int]:
    """Fill score gaps from the reasoning checkpoints.

    Reasoning lands before the merge job writes the matching score columns, so a
    version is browsable and filterable as soon as its checkpoints appear. The
    parquet column always wins where it exists; failed rows (negative score) are
    ignored.
    """
    counts = {version: 0 for version in REASONING_VERSIONS}
    for version in REASONING_VERSIONS:
        rows = sample.reasoning.get(version)
        if not rows:
            continue
        for doc in sample.docs:
            for segment in SEGMENTS:
                if doc.scores[version][segment] is not None:
                    continue
                row = rows.get((doc.doc_id, segment))
                if row is None or row["score"] is None or row["score"] < 0:
                    continue
                doc.scores[version][segment] = row["score"]
                counts[version] += 1
    return counts


def split_pages(doc: Document) -> list[str]:
    """Text sliced per page.

    ``page_offsets[k]`` is the cumulative END offset of page k+1 (the last entry equals
    ``len(text)``), so page k+1 spans ``text[offsets[k-1]:offsets[k]]``. Blank or failed
    pages appear as repeated offsets and yield empty strings, keeping the page index
    aligned with the PDF.
    """
    offsets = doc.page_offsets
    if not offsets:
        return [doc.text]
    bounds = [0, *offsets]
    return [doc.text[bounds[i] : bounds[i + 1]] for i in range(len(offsets))]


def overall_score(scores: dict[str, float | None]) -> float | None:
    values = [scores[seg] for seg in SEGMENTS]
    if any(v is None for v in values):
        return None
    return min(v for v in values if v is not None)


def in_range(value: float | None, low: float | None, high: float | None) -> bool:
    """Whether a score falls inside an inclusive [low, high] filter.

    An unscored segment fails any active bound: a document the model has not
    scored is not evidence of a score in the requested band.
    """
    if low is None and high is None:
        return True
    if value is None:
        return False
    return (low is None or value >= low) and (high is None or value <= high)


def read_pdf_bytes(sample: Sample, doc: Document) -> bytes | None:
    """Read one document's PDF bytes, caching to disk. Returns None when absent."""
    cached = PDF_CACHE_DIR / f"{doc.doc_id}.pdf"
    if cached.exists():
        return cached.read_bytes()
    table = pq.ParquetFile(doc.file_path).read_row_group(doc.row_group, columns=[PDF_COLUMN])
    value = table.column(PDF_COLUMN)[doc.row_in_group].as_py()
    if not value:
        return None
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(value)
    return value


def matches(doc: Document, query: str, q_field: str) -> bool:
    if q_field in ("title", "both") and query in doc.title_lower:
        return True
    if q_field in ("text", "both") and query in doc.text_lower:
        return True
    return False


def doc_summary(doc: Document) -> dict:
    """List-row payload. Both score sets ride along so the UI can show disagreements."""
    return {
        "id": doc.doc_id,
        "title": doc.title,
        "url": doc.meta["url"],
        "num_pages": doc.meta["num_pages"],
        "extraction_status": doc.meta["extraction_status"],
        "scores": {version: dict(doc.scores[version]) for version in VERSIONS},
        "overall": {version: overall_score(doc.scores[version]) for version in VERSIONS},
    }


def build_app(sample: Sample) -> FastAPI:
    app = FastAPI(title="PDF OCR sample browser")

    @app.get("/api/schema")
    def schema_info() -> dict:
        return {
            "num_docs": len(sample.docs),
            "has_pdf": sample.has_pdf_column,
            "has_pdf_status": sample.has_pdf_status_column,
            "score_columns": sample.score_columns_present,
            "versions": [
                {
                    "id": version,
                    "label": VERSION_LABELS[version],
                    "has_scores": sample.version_has_scores(version),
                    "available": sample.version_available(version),
                    "has_reasoning": sample.version_has_reasoning(version),
                    "num_reasoning_rows": len(sample.reasoning.get(version, {})),
                }
                for version in VERSIONS
            ],
            "files": [os.path.basename(p) for p in sample.files],
            "score_range": {"min": SCORE_MIN, "max": SCORE_MAX},
        }

    @app.get("/api/docs")
    def list_docs(
        offset: int = 0,
        limit: int = DEFAULT_PAGE_LIMIT,
        min_begin: float | None = None,
        max_begin: float | None = None,
        min_middle: float | None = None,
        max_middle: float | None = None,
        min_end: float | None = None,
        max_end: float | None = None,
        min_overall: float | None = None,
        max_overall: float | None = None,
        status: str | None = None,
        q: str | None = None,
        q_field: str = "both",
        score_version: str = V1,
    ):
        if q_field not in ("title", "text", "both"):
            return JSONResponse({"error": f"bad q_field: {q_field}"}, status_code=400)
        if score_version not in VERSIONS:
            return JSONResponse({"error": f"bad score_version: {score_version}"}, status_code=400)
        query = q.lower() if q else None
        bounds = {
            "begin": (min_begin, max_begin),
            "middle": (min_middle, max_middle),
            "end": (min_end, max_end),
        }

        hits = []
        for doc in sample.docs:
            if status and doc.meta["extraction_status"] != status:
                continue
            scores = doc.scores[score_version]
            if not all(in_range(scores[segment], low, high) for segment, (low, high) in bounds.items()):
                continue
            if not in_range(overall_score(scores), min_overall, max_overall):
                continue
            if query and not matches(doc, query, q_field):
                continue
            hits.append(doc)

        window = hits[offset : offset + limit]
        return {
            "total": len(hits),
            "offset": offset,
            "limit": limit,
            "score_version": score_version,
            "version_pending": not sample.version_available(score_version),
            "docs": [doc_summary(d) for d in window],
        }

    @app.get("/api/doc/{doc_id}")
    def get_doc(doc_id: str):
        doc = sample.by_id.get(doc_id)
        if doc is None:
            return JSONResponse({"error": "unknown id"}, status_code=404)
        scores: dict[str, dict] = {}
        for version in VERSIONS:
            per_segment = {}
            for segment in SEGMENTS:
                reasoning = sample.reasoning.get(version, {}).get((doc_id, segment))
                per_segment[segment] = {
                    "score": doc.scores[version][segment],
                    "reasoning": reasoning["response"] if reasoning else None,
                    "reasoning_score": reasoning["score"] if reasoning else None,
                }
            scores[version] = per_segment
        return {
            "id": doc_id,
            "title": doc.title,
            "metadata": doc.meta,
            "pages": split_pages(doc),
            "scores": scores,
            "overall": {version: overall_score(doc.scores[version]) for version in VERSIONS},
            "version_pending": {version: not sample.version_available(version) for version in VERSIONS},
            "has_pdf_column": sample.has_pdf_column,
        }

    @app.get("/api/pdf/{doc_id}")
    def get_pdf(doc_id: str):
        doc = sample.by_id.get(doc_id)
        if doc is None:
            return JSONResponse({"error": "unknown id"}, status_code=404)
        if not sample.has_pdf_column:
            return JSONResponse(
                {"error": "pdf column has not landed in the sample parquets yet"},
                status_code=404,
            )
        data = read_pdf_bytes(sample, doc)
        if data is None:
            return JSONResponse({"error": "no pdf bytes stored for this document"}, status_code=404)
        return Response(content=data, media_type="application/pdf")

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--sample-dir", default=SAMPLE_DIR)
    parser.add_argument("--scores-dir", default=SCORES_DIR, help="v1 reasoning checkpoints")
    parser.add_argument("--scores-v2-dir", default=SCORES_V2_DIR, help="v2 reasoning checkpoints")
    parser.add_argument(
        "--restrict-to-version",
        default=FT,
        help="only browse docs scored by this version (ignored when its columns are absent); '' for all docs",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    sample = load_sample(
        args.sample_dir,
        {V1: args.scores_dir, V2: args.scores_v2_dir},
        restrict_to=args.restrict_to_version or None,
    )
    uvicorn.run(build_app(sample), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
