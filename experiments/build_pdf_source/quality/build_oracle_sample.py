# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the 100k oracle-labeled PDF quality sample, entirely on-cluster and entirely in parquet.

This is the scale-up of the 10k sample that trained the first PDF scorer
(:mod:`~experiments.build_pdf_source.quality.build_labels`). That sample was a *cluster* sample --
80 whole shards drawn out of 1,773, then rows drawn inside them -- assembled on a laptop against
PDF bytes that lived in a local directory. Neither property survives at 100k, so this module draws
a uniform sample over the OCR corpus and emits one self-contained parquet set: OCR text, Docling
text, the source PDF, and the oracle scores are all columns of the same row.

**The sample is drawn over the OCR corpus, and Docling is joined onto it as a left join.** The two
routes were exact-deduplicated and decontaminated independently, so each dropped documents the
other kept. Conditioning the draw on Docling would have made this a sample of the PDFs both routes
happen to agree exist, and the documents it excluded are precisely the ones worth seeing: those the
Docling route lost. A sampled document with no Docling extraction keeps its OCR text, its PDF and
its oracle score, and carries null ``docling_*`` columns. The frame is intersected with the fetch
artifact only because the PDF is a required column.

**The three inputs are joined on a derived key, not on ``source_id``.** The clean corpora set
``source_id`` to ``warc_filename:warc_record_offset``, but the fetch artifact sets it to the WARC
record's UUID, so the two share no values and keying the PDF join on the column name matches
nothing at all. :func:`record_key` rebuilds the crawl-record identity from the two columns all
three artifacts do agree on, which is the key the earlier 10k pull used.

Four stages, each skipped when its output is already on storage:

``keys``
    Reads only the key columns of each input and draws ``SAMPLE_SIZE`` uniformly over the OCR
    corpus with a fixed seed, logging what fraction of the draw the Docling route also kept.
``docs``
    Streams both clean corpora (12 GB) through a semi-join against the draw and a left join with
    each other.
``labels``
    Scores the begin/middle/end 512-token windows of every OCR text against the FineWeb-Edu v2
    rubric, then streams the documents back out with the score, reason, scored-window and
    token-count columns attached. Windows are cut with the llama3 tokenizer the scorer is trained
    under, so a scored window is exactly a training example. A document under three windows long
    has no three distinct windows to score, so it is scored once on its leading window and that
    verdict covers all three segments -- which is also the point below which ``build_labels`` keeps
    one training row rather than three near-duplicates.
``merge``
    Zephyr, one task per ``RAW_SHARDS_PER_TASK`` fetch shards. Scans the 435 GB fetch artifact for
    the sampled PDFs and streams out the final rows. This is the only stage that touches the bulk
    data, and it reads it once: PDF bytes are never staged to an intermediate.

Every stage moves its data as a Polars ``LazyFrame`` collected or sunk under the streaming engine,
so no stage holds a dataset it is only passing through -- the merge task streams a 4.6 GB labeled
side against its own ~700 MB of PDFs rather than materializing either. The labeled dataset is
small enough that each merge task re-reads it instead of shuffling 130 GB of PDF bytes to meet it;
under LOTA that costs a cached read per node and saves a full round trip through storage.

The oracle is ``openai/gpt-5.6-luna`` at medium reasoning effort over OpenRouter, keyed by
``OR_KEY_SCALE_UP``. ``SAMPLE_SIZE`` documents cost up to three requests each; the run is
checkpointed and resumable so an interrupted driver does not re-buy scores it already has.

The llama3 tokenizer is a gated repository, so the driver needs ``HF_TOKEN`` as well as the
OpenRouter key:

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name pdf-oracle-sample-100k \\
        --cpu 32 --memory 192GB --disk 64GB --enable-extra-resources \\
        -e OR_KEY_SCALE_UP "$OR_KEY_SCALE_UP" -e HF_TOKEN "$HF_TOKEN" \\
        -- python -m experiments.build_pdf_source.quality.build_oracle_sample
"""

import asyncio
import logging
import os
import pathlib
import random
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import fsspec
import httpx
import polars as pl
from fray.types import ResourceConfig
from rigging.filesystem.cluster_config import StoreType
from rigging.filesystem.s3_compat import configure_coreweave_s3, s3_credentials, s3_endpoint
from rigging.log_setup import configure_logging
from transformers import AutoTokenizer
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

BUCKET = "marin-us-east-02a"
SIGNING_REGION = "US-EAST-02A"

# Inputs. The two clean corpora are the decontaminated, exact-deduplicated output of each
# extraction route; the fetch artifact is the only place the source PDFs exist.
OCR_CLEAN = f"s3://{BUCKET}/marin/data/datakit/clean/common_crawl_focus_2026_22_pdf_ocr_all_6357923a/outputs/main"
DOCLING_CLEAN = (
    f"s3://{BUCKET}/marin/data/datakit/clean/common_crawl_focus_2026_22_pdf_docling_all_2e74110d/outputs/main"
)
RAW_FETCH = f"s3://{BUCKET}/marin/data/datakit/raw/common_crawl_focus_2026_22_pdf_e70aa547/outputs/main"

WORK_PREFIX = f"s3://{BUCKET}/marin/tmp/pdf_oracle_sample100k"
OUTPUT_PREFIX = f"s3://{BUCKET}/marin/data/pdf_quality/cc_focus_2026_22_sample100k"

KEYS_PATH = f"{WORK_PREFIX}/keys.parquet"
DOCS_PATH = f"{WORK_PREFIX}/docs.parquet"
SCORES_PREFIX = f"{WORK_PREFIX}/scores"
LABELED_PATH = f"{WORK_PREFIX}/labeled.parquet"

SAMPLE_SIZE = 100_000
SAMPLE_SEED = 20260805

# Fetch shards per merge task. 1,773 shards over 10 gives ~178 tasks at 3 CPUs each, which fits
# the cw-us-east-02a CPU pool (4 nodes x 192 vCPU) with headroom.
RAW_SHARDS_PER_TASK = 10
_MERGE_TASK_RESOURCES = ResourceConfig(cpu=3, ram="16g", disk="24g")
_MERGE_COORDINATOR_RESOURCES = ResourceConfig(cpu=2, ram="16g", preemptible=False)

# Oracle. Windows are measured with the tokenizer the scorer will be trained under, so a scored
# window is exactly a training example rather than a gpt2-shaped approximation of one.
PROMPT_PATH = pathlib.Path(__file__).with_name("edu_score_v2_prompt.txt")
ORACLE_MODEL = "openai/gpt-5.6-luna"
ORACLE_TOKENIZER = llama3_tokenizer
ORACLE_KEY_VAR = "OR_KEY_SCALE_UP"
SEGMENT_TOKENS = 512
SEGMENTS = ("begin", "middle", "end")
# Below three windows the slices overlap, so the document is scored once instead of three times.
MIN_TOKENS_FOR_ALL_SEGMENTS = 3 * SEGMENT_TOKENS
SCORE_COLUMNS = {segment: f"edu_score_v2_{segment}" for segment in SEGMENTS}
REASON_COLUMNS = {segment: f"edu_reason_v2_{segment}" for segment in SEGMENTS}
SEGMENT_TEXT_COLUMNS = {segment: f"edu_segment_v2_{segment}" for segment in SEGMENTS}
DOC_TOKENS_COLUMN = "doc_tokens"
SCORE_PATTERN = re.compile(r"Educational score:\s*([0-5])")
FAILED_SCORE = -1

# A single driver process holds every request, so concurrency here is the whole job's in-flight
# count against OpenRouter rather than a per-worker share.
ORACLE_CONCURRENCY = 2048
ORACLE_MAX_ATTEMPTS = 8
ORACLE_TIMEOUT = 300.0
CHECKPOINT_EVERY = 5000
TOKENIZE_BATCH = 256

RECORD_KEY = "record_key"

# Columns carried out of each input. Docling's copies of the crawl identifiers are the same values
# as the OCR route's, so only one set is kept; its `extraction_error` column is all-null.
OCR_COLUMNS = (
    "source_id",
    "id",
    "text",
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
    "looped_pages",
    "loop_chars_dropped",
)
DOCLING_COLUMNS = {
    "id": "docling_id",
    "text": "docling_text",
    "num_pages": "docling_num_pages",
    "page_offsets": "docling_page_offsets",
    "extraction_status": "docling_extraction_status",
    "boilerplate_lines_removed": "docling_boilerplate_lines_removed",
    "layout_backend": "layout_backend",
    "needs_ocr": "needs_ocr",
}

PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 1


def storage_options() -> dict[str, str]:
    """Object-store settings for the CoreWeave bucket, for Polars' native reader.

    Polars reads through ``object_store`` rather than fsspec, so rigging's process-wide fsspec
    setup does not reach it and the settings have to be rebuilt here. Two things differ from a
    plain AWS endpoint: the bucket belongs in the *hostname*, because ``object_store``'s
    virtual-hosted mode does not splice it in and CoreWeave's domains reject path-style requests
    outright; and in-cluster the endpoint is the node-local LOTA cache over plain http, which
    ``object_store`` refuses unless told to allow it.
    """
    credentials = s3_credentials(StoreType.COREWEAVE)
    if credentials is None:
        raise ValueError("no CoreWeave credentials; set CW_KEY_ID and CW_KEY_SECRET")
    key, secret = credentials

    endpoint = urlparse(s3_endpoint(StoreType.COREWEAVE))
    host = endpoint.netloc if endpoint.netloc.startswith(f"{BUCKET}.") else f"{BUCKET}.{endpoint.netloc}"
    options = {
        "aws_access_key_id": key,
        "aws_secret_access_key": secret,
        "aws_endpoint_url": f"{endpoint.scheme}://{host}",
        "aws_region": SIGNING_REGION,
        "aws_virtual_hosted_style_request": "true",
    }
    if endpoint.scheme == "http":
        options["aws_allow_http"] = "true"
    return options


def storage() -> fsspec.AbstractFileSystem:
    """The object store as a filesystem, for the listing and existence checks Polars has no API for."""
    configure_coreweave_s3()
    return fsspec.filesystem("s3")


def object_paths(fs: fsspec.AbstractFileSystem, prefix: str) -> list[str]:
    """The parquet shards under *prefix*, as ``s3://`` URLs Polars can open."""
    paths = sorted(f"s3://{path}" for path in fs.glob(f"{prefix}/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet shards under {prefix}")
    return paths


def record_key() -> pl.Expr:
    """The crawl-record identity of a row: WARC file plus record offset.

    This is the only identity the three inputs share. The clean corpora already carry it as
    ``source_id``, but the fetch artifact's ``source_id`` is the WARC record's UUID
    (``<urn:uuid:...>``) and shares no values with it, so keying the PDF join on the column name
    silently matches nothing. Deriving it from the two columns all three artifacts agree on is
    what makes them joinable.
    """
    return pl.concat_str([pl.col("warc_filename"), pl.col("warc_record_offset").cast(pl.String)], separator=":").alias(
        RECORD_KEY
    )


def scan(source: str | list[str], options: dict[str, str]) -> pl.LazyFrame:
    return pl.scan_parquet(source, storage_options=options)


def sink(frame: pl.LazyFrame, path: str, options: dict[str, str]) -> None:
    """Stream *frame* to *path*.

    The object appears only once the upload completes, which is what makes the ``exists`` check
    every stage skips on trustworthy: a killed writer leaves no object at all.
    """
    frame.sink_parquet(
        path,
        storage_options=options,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
    )


def row_count(source: str | list[str], options: dict[str, str]) -> int:
    return scan(source, options).select(pl.len()).collect(engine="streaming").item()


# ---------------------------------------------------------------------------
# Stage 1: the sampled key set
# ---------------------------------------------------------------------------


def distinct_record_keys(fs: fsspec.AbstractFileSystem, prefix: str, options: dict[str, str]) -> pl.DataFrame:
    """Every crawl-record key in the corpus under *prefix*, read without its payload columns."""
    keys = scan(object_paths(fs, prefix), options).select(record_key()).unique().collect(engine="streaming")
    logger.info("%s: %d distinct records", prefix.rsplit("/", 3)[-3], keys.height)
    return keys


def build_keys(fs: fsspec.AbstractFileSystem, options: dict[str, str]) -> None:
    """Draw ``SAMPLE_SIZE`` uniformly over the OCR corpus.

    The frame is the OCR corpus, not its overlap with Docling: the Docling route
    exact-deduplicated and decontaminated independently and dropped documents the OCR route kept,
    and those documents are exactly the ones a Docling-conditioned sample would never show.
    Sampling the intersection would quietly make this a sample of *comparable* PDFs rather than of
    the corpus.

    The PDF is a required column, so the frame is intersected with the fetch artifact. That should
    be the whole corpus -- the OCR route was produced by reading it -- and the shortfall is logged
    rather than assumed away.
    """
    if fs.exists(KEYS_PATH):
        logger.info("keys: reusing %d sampled records", row_count(KEYS_PATH, options))
        return

    ocr = distinct_record_keys(fs, OCR_CLEAN, options)
    docling = distinct_record_keys(fs, DOCLING_CLEAN, options)
    fetched = distinct_record_keys(fs, RAW_FETCH, options)

    usable = ocr.join(fetched, on=RECORD_KEY, how="semi")
    logger.info("keys: %d OCR documents have no PDF in the fetch artifact", ocr.height - usable.height)
    if usable.height < SAMPLE_SIZE:
        raise ValueError(f"only {usable.height} OCR documents have their PDF; need {SAMPLE_SIZE}")

    # Sorted first so the draw depends on the seed alone, not on the order the scan happened to
    # return shards in.
    keys = usable.sort(RECORD_KEY).sample(n=SAMPLE_SIZE, seed=SAMPLE_SEED, shuffle=True).sort(RECORD_KEY)
    covered = keys.join(docling, on=RECORD_KEY, how="semi").height
    logger.info(
        "keys: sampled %d of %d OCR documents; %d (%.1f%%) also survive the Docling route -> %s",
        keys.height,
        usable.height,
        covered,
        100 * covered / keys.height,
        KEYS_PATH,
    )
    sink(keys.lazy(), KEYS_PATH, options)


# ---------------------------------------------------------------------------
# Stage 2: both extractions, joined
# ---------------------------------------------------------------------------


def build_docs(fs: fsspec.AbstractFileSystem, options: dict[str, str]) -> None:
    """Stream the two extractions of every sampled document into one row each."""
    if fs.exists(DOCS_PATH):
        logger.info("docs: already written, skipping")
        return

    keys = scan(KEYS_PATH, options)
    ocr = (
        scan(object_paths(fs, OCR_CLEAN), options)
        .with_columns(record_key())
        .join(keys, on=RECORD_KEY, how="semi")
        .select(RECORD_KEY, *OCR_COLUMNS)
    )
    # A left join: a sampled document the Docling route dropped keeps its OCR text, its PDF and
    # its label, and carries null Docling columns. Those documents are the reason the sample is
    # drawn over the OCR corpus, so they must not be filtered out here.
    docling = (
        scan(object_paths(fs, DOCLING_CLEAN), options)
        .with_columns(record_key())
        .join(keys, on=RECORD_KEY, how="semi")
        .select(RECORD_KEY, *[pl.col(column).alias(renamed) for column, renamed in DOCLING_COLUMNS.items()])
    )
    sink(ocr.join(docling, on=RECORD_KEY, how="left"), DOCS_PATH, options)

    rows = row_count(DOCS_PATH, options)
    if rows != SAMPLE_SIZE:
        raise ValueError(f"{DOCS_PATH} holds {rows} rows, expected {SAMPLE_SIZE}")
    with_docling = (
        scan(DOCS_PATH, options).select(pl.col("docling_text").is_not_null().sum()).collect(engine="streaming").item()
    )
    logger.info("docs: %d rows, %d (%.1f%%) with a Docling extraction", rows, with_docling, 100 * with_docling / rows)


# ---------------------------------------------------------------------------
# Stage 3: oracle labels
# ---------------------------------------------------------------------------


def segment_windows(token_ids: list[int]) -> list[tuple[list[str], list[int]]]:
    """The token windows to score, each paired with the segments its verdict labels.

    Three disjoint windows once a document is at least three windows long. Below that the windows
    would overlap -- at or under one window they are the same text outright -- so only the leading
    one is scored and its verdict covers all three segments. That is the same threshold
    ``build_labels`` uses to decide a short document contributes one training row rather than
    three near-duplicates, so nothing is bought that would not be trained on.
    """
    if len(token_ids) >= MIN_TOKENS_FOR_ALL_SEGMENTS:
        middle = (len(token_ids) - SEGMENT_TOKENS) // 2
        return [
            (["begin"], token_ids[:SEGMENT_TOKENS]),
            (["middle"], token_ids[middle : middle + SEGMENT_TOKENS]),
            (["end"], token_ids[-SEGMENT_TOKENS:]),
        ]
    return [(list(SEGMENTS), token_ids[:SEGMENT_TOKENS])]


@dataclass(frozen=True)
class OracleWork:
    """What to ask the oracle, and the document lengths that decided it.

    ``doc_tokens`` is carried out to the dataset so a consumer can tell a three-window document
    from a shared-verdict one without re-tokenizing behind a gated tokenizer.
    """

    requests: list[dict]
    doc_tokens: dict[str, int]


def segment_requests(keys: list[str], texts: list[str]) -> OracleWork:
    """One request per distinct (document, window), with the window's text to send."""
    tokenizer = AutoTokenizer.from_pretrained(ORACLE_TOKENIZER)
    requests: list[dict] = []
    doc_tokens: dict[str, int] = {}
    for start in range(0, len(keys), TOKENIZE_BATCH):
        batch = slice(start, start + TOKENIZE_BATCH)
        encoded = tokenizer(texts[batch], add_special_tokens=False)["input_ids"]
        for offset, token_ids in enumerate(encoded):
            key = keys[start + offset]
            doc_tokens[key] = len(token_ids)
            for segments, window in segment_windows(token_ids):
                # Pinned off: the cleanup pass is a WordPiece detokenizer that eats spaces before
                # punctuation, and this text is both what the oracle grades and what a scorer is
                # later trained on, so it has to stay the document's own text.
                text = tokenizer.decode(window, clean_up_tokenization_spaces=False)
                requests.append({RECORD_KEY: key, "segments": segments, "text": text})
        if start and start % (TOKENIZE_BATCH * 40) == 0:
            logger.info("labels: tokenized %d/%d documents", start, len(keys))
    shared = sum(1 for count in doc_tokens.values() if count < MIN_TOKENS_FOR_ALL_SEGMENTS)
    logger.info(
        "labels: %d requests for %d documents; %d are under %d %s tokens and are scored once",
        len(requests),
        len(keys),
        shared,
        MIN_TOKENS_FOR_ALL_SEGMENTS,
        ORACLE_TOKENIZER,
    )
    return OracleWork(requests=requests, doc_tokens=doc_tokens)


def parse_score(content: str) -> int:
    """The last score the rubric emits, or ``FAILED_SCORE`` when the response has none."""
    matches = SCORE_PATTERN.findall(content)
    return int(matches[-1]) if matches else FAILED_SCORE


RETRYABLE_STATUS = (408, 429, 500, 502, 503, 520, 524)
ORACLE_FAILURES = (httpx.HTTPError, KeyError, IndexError, ValueError)


async def ask_oracle(client: httpx.AsyncClient, prompt: str, extract: str) -> str:
    """The model's reply for one extract. Raises on anything worth another attempt."""
    response = await client.post(
        "/chat/completions",
        json={
            "model": ORACLE_MODEL,
            "messages": [{"role": "user", "content": prompt.replace("{example}", extract)}],
            "reasoning": {"effort": "medium"},
        },
    )
    if response.status_code in RETRYABLE_STATUS:
        raise httpx.HTTPStatusError(f"retryable {response.status_code}", request=response.request, response=response)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"] or ""


async def score_one(client: httpx.AsyncClient, prompt: str, item: dict) -> list[dict]:
    """The rubric's verdict on one segment text, fanned out to every segment sharing it.

    A request that never lands is recorded at ``FAILED_SCORE`` rather than dropped, so a
    checkpoint distinguishes "asked and got nothing" from "not asked yet" and a resumed run
    retries it instead of silently labelling the document.
    """

    def verdict(score: int, reason: str) -> list[dict]:
        return [
            {RECORD_KEY: item[RECORD_KEY], "segment": segment, "score": score, "reason": reason}
            for segment in item["segments"]
        ]

    for attempt in range(ORACLE_MAX_ATTEMPTS - 1):
        try:
            content = await ask_oracle(client, prompt, item["text"])
            return verdict(parse_score(content), content)
        except ORACLE_FAILURES:
            # Jittered so a throttled fleet of this size does not resynchronize on the next try.
            await asyncio.sleep(2**attempt + random.random())
    try:
        content = await ask_oracle(client, prompt, item["text"])
        return verdict(parse_score(content), content)
    except ORACLE_FAILURES as failure:
        logger.warning("labels: giving up on %s %s: %s", item[RECORD_KEY], item["segments"], failure)
        return verdict(FAILED_SCORE, f"ERROR: {failure}")


async def score_all(fs: fsspec.AbstractFileSystem, items: list[dict], prompt: str, options: dict[str, str]) -> None:
    """Send every request, ``ORACLE_CONCURRENCY`` in flight, checkpointing as answers land."""
    done: list[dict] = []
    written = 0
    checkpoint = len(fs.glob(f"{SCORES_PREFIX}/scores-*.parquet"))

    def flush() -> None:
        nonlocal checkpoint, written
        if len(done) == written:
            return
        batch = pl.DataFrame(
            done[written:],
            schema={RECORD_KEY: pl.String, "segment": pl.String, "score": pl.Int8, "reason": pl.String},
        )
        sink(batch.lazy(), f"{SCORES_PREFIX}/scores-{checkpoint:04d}.parquet", options)
        checkpoint += 1
        written = len(done)

    # A shared iterator rather than a semaphore over one task per request: at this scale the task
    # objects alone would outweigh the payloads they carry, and `next` never yields, so the
    # workers cannot race for an index.
    queue = iter(items)
    limits = httpx.Limits(max_connections=ORACLE_CONCURRENCY + 64, max_keepalive_connections=ORACLE_CONCURRENCY)
    async with httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1",
        headers={"Authorization": f"Bearer {os.environ[ORACLE_KEY_VAR]}"},
        timeout=httpx.Timeout(ORACLE_TIMEOUT),
        limits=limits,
    ) as client:

        async def worker() -> None:
            for item in queue:
                done.extend(await score_one(client, prompt, item))
                if len(done) - written >= CHECKPOINT_EVERY:
                    flush()
                    logger.info("labels: %d/%d requests answered", written, len(items))

        await asyncio.gather(*[worker() for _ in range(min(ORACLE_CONCURRENCY, len(items)))])
    flush()


def scored_segments(fs: fsspec.AbstractFileSystem, options: dict[str, str]) -> pl.DataFrame:
    """Every checkpointed score, one row per (document, segment). Later checkpoints win."""
    if not fs.glob(f"{SCORES_PREFIX}/scores-*.parquet"):
        return pl.DataFrame(schema={RECORD_KEY: pl.String, "segment": pl.String, "score": pl.Int8, "reason": pl.String})
    return (
        scan(object_paths(fs, SCORES_PREFIX), options)
        .collect(engine="streaming")
        .unique(subset=[RECORD_KEY, "segment"], keep="last")
    )


def label_columns(work: OracleWork, scores: pl.DataFrame) -> pl.DataFrame:
    """One row per document carrying its token count and, per segment, score / reason / window.

    Built by joining a frame per segment rather than by pivoting, so the column names are the ones
    written here rather than whatever a pivot chooses to derive.
    """
    windows = {(item[RECORD_KEY], segment): item["text"] for item in work.requests for segment in item["segments"]}
    keys = list(work.doc_tokens)
    labels = pl.DataFrame(
        {
            RECORD_KEY: keys,
            DOC_TOKENS_COLUMN: pl.Series([work.doc_tokens[key] for key in keys], dtype=pl.Int32),
        }
    )
    for segment in SEGMENTS:
        # The scored window travels with its score, so a consumer never has to reproduce the
        # slicing through a gated tokenizer to know what text a label describes.
        labels = labels.with_columns(
            pl.Series(SEGMENT_TEXT_COLUMNS[segment], [windows[(key, segment)] for key in keys], dtype=pl.String)
        )
        graded = scores.filter(pl.col("segment") == segment).select(
            RECORD_KEY,
            pl.col("score").alias(SCORE_COLUMNS[segment]),
            pl.col("reason").alias(REASON_COLUMNS[segment]),
        )
        labels = labels.join(graded, on=RECORD_KEY, how="left")
    return labels


def build_labels(fs: fsspec.AbstractFileSystem, options: dict[str, str]) -> None:
    """Score every sampled document and stream the corpus back out with the labels attached."""
    if fs.exists(LABELED_PATH):
        logger.info("labels: already written, skipping")
        return

    docs = scan(DOCS_PATH, options).select(RECORD_KEY, "text").collect(engine="streaming")
    work = segment_requests(docs[RECORD_KEY].to_list(), docs["text"].to_list())
    del docs

    already = set(
        scored_segments(fs, options)
        .filter(pl.col("score") != FAILED_SCORE)
        .select(pl.concat_str([RECORD_KEY, "segment"], separator="\t"))
        .to_series()
        .to_list()
    )
    todo = [item for item in work.requests if f"{item[RECORD_KEY]}\t{item['segments'][0]}" not in already]
    logger.info("labels: %d requests to send, %d already scored", len(todo), len(already))
    if todo:
        asyncio.run(score_all(fs, todo, PROMPT_PATH.read_text(), options))

    scores = scored_segments(fs, options)
    failed = scores.filter(pl.col("score") == FAILED_SCORE).height
    logger.info("labels: %d segment scores on hand (%d unscored)", scores.height, failed)

    sink(
        scan(DOCS_PATH, options).join(label_columns(work, scores).lazy(), on=RECORD_KEY, how="left"),
        LABELED_PATH,
        options,
    )
    logger.info("labels: wrote %d labeled rows -> %s", row_count(LABELED_PATH, options), LABELED_PATH)


# ---------------------------------------------------------------------------
# Stage 4: the PDF scan and the final rows
# ---------------------------------------------------------------------------


def merge_output_path(index: int, total: int) -> str:
    return f"{OUTPUT_PREFIX}/part-{index:05d}-of-{total:05d}.parquet"


def merge_chunk(chunk: tuple[int, int, list[str]]) -> int:
    """Emit the final rows for the sampled PDFs living in one slice of the fetch artifact.

    Runs as a Zephyr map task next to the storage. The labeled side streams through the join
    against this chunk's own PDFs, so the task holds its share of the bulk data and nothing else.
    """
    index, total, shards = chunk
    fs = storage()
    output = merge_output_path(index, total)
    if fs.exists(output):
        return 0

    options = storage_options()
    pdfs = (
        scan(shards, options)
        .with_columns(record_key())
        .join(scan(KEYS_PATH, options), on=RECORD_KEY, how="semi")
        .select(RECORD_KEY, "pdf")
    )
    merged = scan(LABELED_PATH, options).join(pdfs, on=RECORD_KEY, how="inner").drop(RECORD_KEY)
    sink(merged, output, options)
    rows = row_count(output, options)
    logger.info("merge: chunk %d wrote %d rows from %d shards", index, rows, len(shards))
    return rows


def build_merged(fs: fsspec.AbstractFileSystem) -> None:
    """Scan the fetch artifact once and write the final parquet set."""
    shards = object_paths(fs, RAW_FETCH)
    groups = [shards[start : start + RAW_SHARDS_PER_TASK] for start in range(0, len(shards), RAW_SHARDS_PER_TASK)]
    chunks = [(index, len(groups), group) for index, group in enumerate(groups)]
    logger.info("merge: %d chunks over %d fetch shards", len(chunks), len(shards))

    outcome = ZephyrContext(
        name="pdf-oracle-sample-merge",
        resources=_MERGE_TASK_RESOURCES,
        coordinator_resources=_MERGE_COORDINATOR_RESOURCES,
        max_workers=len(chunks),
        stage_runner_factory=SubprocessRunner,
    ).execute(
        Dataset.from_list(chunks).map(merge_chunk),
        # One chunk per worker: costing a map task at the full worker keeps Zephyr from packing
        # every chunk onto a single worker and serializing the scan.
        map_task_resources=_MERGE_TASK_RESOURCES,
    )
    logger.info("merge: done, counters %s", dict(outcome.counters))


def verify_output(fs: fsspec.AbstractFileSystem, options: dict[str, str]) -> None:
    """Confirm the dataset holds each sampled document exactly once.

    Read from the data rather than from the stage's own counters, which report a zero for every
    chunk a resumed run skipped. Two failures are worth catching: a document whose PDF no chunk
    claimed, and one two chunks both claimed because the fetch artifact holds its record twice.
    """
    paths = object_paths(fs, OUTPUT_PREFIX)
    ids = scan(paths, options).select("source_id").collect(engine="streaming")["source_id"]
    if ids.len() != SAMPLE_SIZE or ids.n_unique() != SAMPLE_SIZE:
        raise ValueError(
            f"{OUTPUT_PREFIX} holds {ids.len()} rows ({ids.n_unique()} distinct) over {len(paths)} files, "
            f"expected {SAMPLE_SIZE} of each"
        )
    logger.info("verified: %d distinct documents over %d files", ids.len(), len(paths))


def main() -> None:
    configure_logging(logging.INFO)
    fs = storage()
    options = storage_options()
    build_keys(fs, options)
    build_docs(fs, options)
    build_labels(fs, options)
    build_merged(fs)
    verify_output(fs, options)
    logger.info("oracle sample complete -> %s", OUTPUT_PREFIX)


if __name__ == "__main__":
    main()
