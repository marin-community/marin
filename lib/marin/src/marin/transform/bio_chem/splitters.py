# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Format-preserving splitters and doc packers for bio/chem notation slices.

The splitters yield each record as a single string whose contents are exactly
the bytes that appeared in the input. Where a format requires a terminator
(SDF's ``$$$$``), the terminator is included in the yielded record so that
``"".join(records)`` round-trips the original stream up to leading whitespace
that was not associated with any record.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass


def iter_fasta_records(lines: Iterable[str]) -> Iterator[str]:
    """Yield FASTA records, each starting with a ``>`` header line.

    Sequence lines are preserved verbatim with their original line wrapping and
    case. The header line keeps its leading ``>`` and trailing newline (if any).
    """
    buf: list[str] = []
    for raw in lines:
        if raw.startswith(">"):
            if buf:
                yield "".join(buf)
                buf = []
        if buf or raw.startswith(">"):
            buf.append(raw)
        # Lines before the first header are dropped (they are not part of any
        # FASTA record). This matches the FASTA spec: text outside a record is
        # comment-only.
    if buf:
        yield "".join(buf)


def iter_gff_blocks(lines: Iterable[str]) -> Iterator[str]:
    """Yield GFF/GTF blocks, where a block is a contiguous run of feature lines.

    Lines beginning with ``##`` are treated as directives and attach to the
    following block. Comment lines (``#`` not ``##``) attach to the current
    block. Tab columns and the original line endings are preserved.

    A block boundary occurs whenever the seqid (column 1) changes for a
    non-comment line, or after a ``###`` directive.
    """
    block: list[str] = []
    pending_directives: list[str] = []
    current_seqid: str | None = None

    def _flush() -> str | None:
        nonlocal block
        if not block:
            return None
        out = "".join(block)
        block = []
        return out

    for raw in lines:
        line = raw.rstrip("\n").rstrip("\r")
        if not line:
            continue
        if line.startswith("###"):
            flushed = _flush()
            if flushed is not None:
                # Attach the ### terminator to the just-finished block.
                yield flushed + raw
            else:
                # No prior block; carry as a directive for the next block.
                pending_directives.append(raw)
            current_seqid = None
            continue
        if line.startswith("##"):
            flushed = _flush()
            if flushed is not None:
                yield flushed
            pending_directives.append(raw)
            current_seqid = None
            continue
        if line.startswith("#"):
            block.append(raw)
            continue
        seqid = line.split("\t", 1)[0]
        if current_seqid is None:
            if pending_directives:
                block.extend(pending_directives)
                pending_directives = []
            block.append(raw)
            current_seqid = seqid
        elif seqid == current_seqid:
            block.append(raw)
        else:
            flushed = _flush()
            if flushed is not None:
                yield flushed
            if pending_directives:
                block.extend(pending_directives)
                pending_directives = []
            block.append(raw)
            current_seqid = seqid

    flushed = _flush()
    if flushed is not None:
        yield flushed
    elif pending_directives:
        # Trailing directives with no following block.
        yield "".join(pending_directives)


def iter_smiles_records(lines: Iterable[str]) -> Iterator[str]:
    """Yield one record per non-blank line, preserving the full line including
    column delimiters and trailing newline.

    Header lines starting with ``#`` or ``//`` are skipped (these appear in
    PubChem/ChEMBL exports as commentary and are not records).
    """
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("//"):
            continue
        yield raw


def iter_sdf_records(text: Iterable[str]) -> Iterator[str]:
    """Yield SDF records terminated by ``$$$$``.

    Each yielded record includes the trailing ``$$$$`` line so that
    concatenating yielded records reproduces the original SDF stream (modulo
    whitespace that was not part of any record).

    The input is an iterable of arbitrary text chunks (commonly file-iter lines
    but a single concatenated string also works).
    """
    buf: list[str] = []
    for chunk in text:
        # Process line-by-line so the ``$$$$`` boundary check is exact.
        start = 0
        while True:
            nl = chunk.find("\n", start)
            if nl == -1:
                buf.append(chunk[start:])
                break
            line = chunk[start : nl + 1]
            buf.append(line)
            if line.rstrip("\r\n") == "$$$$":
                yield "".join(buf)
                buf = []
            start = nl + 1
    if buf and any(s.strip() for s in buf):
        # Trailing partial record without a $$$$ terminator.
        yield "".join(buf)


def iter_mmcif_blocks(text: Iterable[str]) -> Iterator[str]:
    """Yield mmCIF data blocks, each starting with a ``data_<id>`` line.

    All loop_ structures and column whitespace are preserved verbatim.
    """
    buf: list[str] = []
    in_block = False
    for chunk in text:
        start = 0
        while True:
            nl = chunk.find("\n", start)
            if nl == -1:
                tail = chunk[start:]
                if tail:
                    if in_block or tail.startswith("data_"):
                        if not in_block and tail.startswith("data_"):
                            in_block = True
                        buf.append(tail)
                break
            line = chunk[start : nl + 1]
            stripped = line.lstrip()
            if stripped.startswith("data_"):
                if buf:
                    yield "".join(buf)
                    buf = []
                buf.append(line)
                in_block = True
            elif in_block:
                buf.append(line)
            # Pre-block comments / whitespace are dropped.
            start = nl + 1
    if buf:
        yield "".join(buf)


@dataclass(frozen=True)
class SamplingCap:
    """Bound on how much input to consume from a single source.

    ``max_records`` and ``max_bytes`` are both upper bounds and the splitter
    stops at whichever is reached first.
    """

    max_records: int = 5000
    max_bytes: int = 64 * 1024 * 1024


def take_until_cap(records: Iterable[str], cap: SamplingCap) -> Iterator[str]:
    """Yield records until either the record-count or byte cap is reached."""
    seen_records = 0
    seen_bytes = 0
    for record in records:
        if seen_records >= cap.max_records or seen_bytes >= cap.max_bytes:
            return
        yield record
        seen_records += 1
        seen_bytes += len(record.encode("utf-8"))


def pack_records_into_docs(
    records: Iterable[str],
    *,
    target_doc_chars: int = 8192,
    max_records_per_doc: int = 64,
    record_separator: str = "",
) -> Iterator[str]:
    """Concatenate consecutive records into longer documents for ICL evaluation.

    Records are appended to a buffer until either the accumulated character
    count reaches ``target_doc_chars`` or ``max_records_per_doc`` records have
    been collected, at which point the buffer is flushed as one document.
    ``target_doc_chars`` is a soft floor: a doc may slightly exceed it because
    we always include the record that pushed the buffer over the threshold,
    and a single record longer than the target produces a doc on its own.

    ``record_separator`` is inserted between records — pass ``""`` for
    self-delimiting formats (FASTA, GFF, SDF, mmCIF) and ``"\\n"`` (or similar)
    for line-oriented formats like SMILES.
    """
    buf: list[str] = []
    buf_chars = 0
    buf_count = 0
    sep_len = len(record_separator)
    for record in records:
        if buf:
            buf_chars += sep_len
        buf.append(record)
        buf_chars += len(record)
        buf_count += 1
        if buf_chars >= target_doc_chars or buf_count >= max_records_per_doc:
            yield record_separator.join(buf) if record_separator else "".join(buf)
            buf = []
            buf_chars = 0
            buf_count = 0
    if buf:
        yield record_separator.join(buf) if record_separator else "".join(buf)
