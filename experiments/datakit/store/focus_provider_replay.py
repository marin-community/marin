# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover selected Focus Crawl text from original WARC response byte ranges."""

import codecs
import gzip
import hashlib
import io
import json
import os
import re
import unicodedata
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from email.message import Message
from importlib.metadata import distribution

import requests
from marin.datakit.normalize import generate_id
from resiliparse.parse.encoding import bytes_to_str, detect_encoding, map_encoding_to_html5

HTML_MIME_TYPES = frozenset({"text/html", "application/xhtml+xml"})
CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
LONG_WHITESPACE = re.compile(r"\s{129,}")
INVALID_XML_CHARACTERS = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")
NUMERIC_CHARACTER_REFERENCE = re.compile(r"&#(x[0-9a-f]+|[0-9]+);?", re.IGNORECASE)
BOM_ENCODINGS = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)
EXCLUDED_RECORD_ID = "<urn:uuid:88562b9a-a0da-40be-a939-330f53017c9d>"


@dataclass(frozen=True)
class FocusTarget:
    record_id: str
    url: str
    normalized_id: str


@dataclass(frozen=True)
class WarcRecordRange:
    url: str
    offset: int
    length: int


def focus_targets_from_index(selected_rows: list[dict], locator_rows: Iterator[dict]) -> dict[str, list[FocusTarget]]:
    """Find WARC locators using cached index columns, without reading cached text.

    The selected ledger stores source IDs as JSON strings. The cache contributes
    only UUID, URL, and WARC filename; its text never establishes reconstruction.
    """
    expected = set()
    for row in selected_rows:
        source_id = json.loads(row["source_id"])
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("Selected Focus Crawl occurrence lacks a WARC UUID")
        expected.add((row["id"], source_id))
    locators = {}
    for row in locator_rows:
        key = (row["id"], row["source_id"])
        if key not in expected:
            continue
        locator = (row["warc_filename"], row["url"])
        if key in locators and locators[key] != locator:
            raise ValueError("Cached index gives conflicting original WARC locators")
        locators[key] = locator
    if expected - locators.keys():
        raise ValueError("Cached index lacks original WARC locators for selected occurrences")
    grouped: dict[str, list[FocusTarget]] = {}
    for (doc_id, record_id), (warc_filename, url) in sorted(locators.items()):
        if not warc_filename or not url:
            raise ValueError("Cached WARC locator requires a filename and URL")
        grouped.setdefault(warc_filename, []).append(FocusTarget(record_id, url, doc_id))
    return grouped


def focus_html_extractor(*, justext_revision: str) -> Callable[[bytes, str], str]:
    """Prepare the archived extractor with a verified jusText checkout revision.

    Decoding, sanitization, boilerplate extraction, and whitespace behavior follow
    Marin's completed Focus Crawl run at 20b7003fea. Matching every selected hash
    remains required: a pinned replay implementation alone is not historical proof.
    """
    if os.environ.get("JUSTEXT_MODEL") != "sklearn" or os.environ.get("JUSTEXT_NO_DOWNLOAD") != "1":
        raise ValueError("Archived Focus Crawl replay requires JUSTEXT_MODEL=sklearn and JUSTEXT_NO_DOWNLOAD=1")
    import justext  # noqa: PLC0415  # pyrefly: ignore[missing-import]  # Pinned by the WARC replay job.

    direct_url = distribution("jusText").read_text("direct_url.json")
    if direct_url is None or json.loads(direct_url).get("vcs_info", {}).get("commit_id") != justext_revision:
        raise ValueError("Installed jusText does not match the recorded replay revision")
    model = justext.get_model()
    if model is None:
        raise ValueError("Pinned jusText sklearn model is unavailable")
    model.model.set_params(n_jobs=1)
    stoplist = justext.get_stoplist("English")

    def extract(payload: bytes, content_type: str) -> str:
        encoding = next((name for bom, name in BOM_ENCODINGS if payload.startswith(bom)), None)
        if encoding is None:
            message = Message()
            message["content-type"] = content_type
            charset = message.get_param("charset")
            if isinstance(charset, str):
                encoding = map_encoding_to_html5(charset, fallback_utf8=False)
        if encoding is None:
            detected = detect_encoding(payload, html5_compatible=False, from_html_meta=True)
            encoding = map_encoding_to_html5(detected, fallback_utf8=False) if detected else None
        html = bytes_to_str(
            payload, encoding=encoding or "cp1252", errors="replace", fallback_encodings=("utf-8", "cp1252")
        )
        if "\ufffd" in html:
            raise ValueError("Archived Focus Crawl extractor rejects replacement characters")
        html = _sanitize_html(html)
        try:
            paragraphs = justext.justext(html, stoplist, model=model)
        except ValueError as error:
            if not str(error).startswith("invalid literal for int() with base 10"):
                raise
            normalized = "".join(
                str(unicodedata.digit(character)) if character.isdigit() and not character.isdecimal() else character
                for character in html
            )
            if normalized == html:
                raise
            paragraphs = justext.justext(normalized, stoplist, model=model)
        text = "\n\n".join(paragraph.text for paragraph in paragraphs if not paragraph.is_boilerplate)
        return LONG_WHITESPACE.sub(lambda match: match.group(0)[:128], text).strip()

    return extract


def _sanitize_html(html: str) -> str:
    html = INVALID_XML_CHARACTERS.sub("", html)

    def sanitize_reference(match: re.Match[str]) -> str:
        value = match.group(1)
        codepoint = int(value[1:], 16) if value.lower().startswith("x") else int(value)
        if codepoint > 0x10FFFF:
            return match.group(0)
        character = chr(codepoint)
        if INVALID_XML_CHARACTERS.fullmatch(character):
            return ""
        if character.isdigit() and not character.isdecimal():
            return str(unicodedata.digit(character))
        return match.group(0)

    return NUMERIC_CHARACTER_REFERENCE.sub(sanitize_reference, html)


def focus_record_ranges(cdx_bytes: bytes, target_urls: set[str]) -> Iterator[WarcRecordRange]:
    """Resolve target URLs through the original WARC's compressed CDX sidecar."""
    with gzip.open(io.BytesIO(cdx_bytes), "rt", encoding="utf-8") as stream:
        for line in stream:
            fields = line.split(" ", 2)
            if len(fields) != 3:
                raise ValueError("Malformed Common Crawl CDX record")
            record = json.loads(fields[2])
            if record["url"] not in target_urls:
                continue
            mime_types = {
                str(record.get(key, "")).partition(";")[0].strip().lower() for key in ("mime", "mime-detected")
            }
            if not str(record.get("status", "")).startswith("2") or not mime_types & HTML_MIME_TYPES:
                continue
            offset, length = int(record["offset"]), int(record["length"])
            if offset < 0 or length <= 0:
                raise ValueError("Invalid Common Crawl WARC byte range")
            yield WarcRecordRange(record["url"], offset, length)


def extract_focus_warc_record(
    warc_bytes: bytes,
    *,
    target: FocusTarget,
    warc_filename: str,
    extract_html: Callable[[bytes, str], str],
) -> dict:
    """Verify a WARC UUID and selected hash before returning reconstructed text."""
    from warcio.archiveiterator import ArchiveIterator  # noqa: PLC0415  # pyrefly: ignore[missing-import]

    found = None
    for record in ArchiveIterator(io.BytesIO(warc_bytes)):
        if record.rec_headers.get_header("WARC-Record-ID") != target.record_id:
            continue
        if found is not None:
            raise ValueError("Repeated WARC UUID in original response bytes")
        if target.record_id == EXCLUDED_RECORD_ID or record.rec_type != "response" or record.http_headers is None:
            raise ValueError("Selected Focus Crawl record is not an admitted response")
        if record.rec_headers.get_header("WARC-Target-URI") != target.url:
            raise ValueError("Original WARC URL differs from the selected locator")
        status = record.http_headers.get_statuscode()
        content_type = record.http_headers.get_header("Content-Type") or ""
        identified = record.rec_headers.get_header("WARC-Identified-Payload-Type") or ""
        mime_types = {value.partition(";")[0].strip().lower() for value in (content_type, identified) if value}
        if status is None or not status.startswith("2") or not mime_types & HTML_MIME_TYPES:
            raise ValueError("Selected Focus Crawl record is not a successful HTML response")
        text = extract_html(record.content_stream().read(), content_type)
        if not text or generate_id(text) != target.normalized_id:
            raise ValueError("Original WARC replay did not match the selected normalized hash")
        found = {
            "id": target.normalized_id,
            "text": text,
            "source_id": target.record_id,
            "url": target.url,
            "warc_filename": warc_filename,
            "warc_bytes_sha256": hashlib.sha256(warc_bytes).hexdigest(),
        }
    if found is None:
        raise ValueError("Original WARC range did not contain the selected UUID")
    return found


def fetch_focus_documents(
    warc_filename: str,
    targets: list[FocusTarget],
    *,
    session: requests.Session,
    base_url: str,
    extract_html: Callable[[bytes, str], str],
    max_record_bytes: int,
    max_cdx_bytes: int,
) -> list[dict]:
    """Fetch bounded original WARC ranges for selected locators from one file."""
    prefix, separator, filename = warc_filename.rpartition("/warc/")
    if not separator or not filename.endswith(".warc.gz"):
        raise ValueError("Unexpected Focus Crawl WARC filename")
    cdx_path = f"{prefix}/cdx/warc/{filename.removesuffix('.warc.gz')}.cdx.gz"
    with session.get(f"{base_url.rstrip('/')}/{cdx_path}", stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        cdx_bytes = response.raw.read(max_cdx_bytes + 1)
        if len(cdx_bytes) > max_cdx_bytes:
            raise ValueError("Original CDX sidecar exceeds the configured read bound")
        ranges = list(focus_record_ranges(cdx_bytes, {target.url for target in targets}))
    results = {}
    for selected in ranges:
        if selected.length > max_record_bytes:
            raise ValueError("Selected original WARC record exceeds the configured read bound")
        stop = selected.offset + selected.length - 1
        with session.get(
            f"{base_url.rstrip('/')}/{warc_filename}",
            headers={"Range": f"bytes={selected.offset}-{stop}"},
            stream=True,
            timeout=(30, 300),
        ) as response:
            response.raise_for_status()
            match = CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
            if (
                response.status_code != 206
                or match is None
                or tuple(map(int, match.groups()[:2])) != (selected.offset, stop)
            ):
                raise ValueError("Original WARC request did not return the exact requested byte range")
            payload = response.raw.read(selected.length + 1)
        if len(payload) != selected.length:
            raise ValueError("Truncated or oversized original WARC response")
        for target in targets:
            if target.url == selected.url and target.record_id not in results:
                # A URL can recur within a WARC; the UUID, not the URL, selects
                # the occurrence. Inspect headers before attempting its hash.
                from warcio.archiveiterator import ArchiveIterator  # noqa: PLC0415  # pyrefly: ignore[missing-import]

                record_ids = {
                    record.rec_headers.get_header("WARC-Record-ID") for record in ArchiveIterator(io.BytesIO(payload))
                }
                if target.record_id not in record_ids:
                    continue
                results[target.record_id] = extract_focus_warc_record(
                    payload, target=target, warc_filename=warc_filename, extract_html=extract_html
                )
    if len(results) != len({target.record_id for target in targets}):
        raise ValueError("Original WARC replay left selected UUIDs unresolved")
    return [results[target.record_id] for target in targets]
