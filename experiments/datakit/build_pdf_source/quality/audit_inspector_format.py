# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-check the agreement normalizer against the serialization pdf-inspector actually emits.

Every headline number in this evaluation is a token-overlap between two routes, computed after
:mod:`~experiments.datakit.build_pdf_source.quality.route_agreement` folds away each route's
serialization conventions. That normalizer is written against a *specific* dialect, and a dialect is
not a stable thing: 1.17.0 rewrote table recovery, added hyphenated-word rejoining at line breaks,
and changed what markdown structure the extractor emits around figures and references.

A rule that stops matching does not fail loudly. It leaks markup into the token stream, where it
counts as content one route "added" and the other did not -- inflating that route's token count,
deflating its precision, and doing so *only* for the route whose dialect drifted. The prior pass
found two genuine leaks this way (link targets, and Docling's ``formula-not-decoded`` placeholder),
and neither announced itself; both were found by looking at which tokens one route produced and the
other never did.

So this module does exactly that, and nothing else:

*Construct census.* How often each markup construct appears in the raw markdown, which says whether
a normalizer rule is still load-bearing and whether a construct the rules do not mention has
appeared. A rule matching nothing is as interesting as a construct nobody handles.

*Leak census.* The most common tokens that survive normalization on one side of a pair and never
appear on the other. Real content differences are long-tailed and document-specific; a leak is a
short list of the same few tokens on thousands of documents, which is what makes it visible at all.

*Furniture check.* Whether running headers and footers still reach the output. 1.17.0 gained
page-edge furniture stripping for short documents, and this repository already strips running
headers and footers downstream in ``boilerplate.py`` -- if both fire, real text is removed twice.
Reading 1.17.0 says the extraction entry point this evaluation uses does not enable it
(``extract_pages_markdown_mem`` passes ``strip_repeated_headers_footers=false``, and
``MarkdownOptions::default()`` leaves ``strip_headers_footers`` unset), but that is a claim about a
call graph and this is the measurement of it: the share of multi-page documents whose first line
repeats across pages should not move.

Deliberately a single task over :data:`SHARD_COUNT` shards rather than a fan-out. The output is a
frequency table whose shape is legible at a thousand documents and does not sharpen at a hundred
thousand, and the tokenizer runs on text already in memory.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name pdf-inspector-format-audit --extra pdf \\
        --cpu 8 --memory 32GB --disk 32GB --enable-extra-resources \\
        -- python -m experiments.datakit.build_pdf_source.quality.audit_inspector_format
"""

import json
import logging
import re
import sys
from collections import Counter
from importlib.metadata import version

import polars as pl
from rigging.log_setup import configure_logging

from experiments.datakit.build_pdf_source.quality import route_agreement
from experiments.datakit.build_pdf_source.quality.build_route_study import shards, storage
from experiments.datakit.build_pdf_source.quality.probe_pdf_inspector import (
    WORKER_FLAG,
    Worker,
    read_exactly,
)

logger = logging.getLogger(__name__)

OUTPUT_PATH = "s3://marin-us-east-02a/marin/data/pdf_quality/cc_focus_2026_22_inspector_format_audit.json"

MODULE_NAME = "experiments.datakit.build_pdf_source.quality.audit_inspector_format"
EXTRACT_OP = "extract"

SHARD_COUNT = 2
AUDIT_DOCUMENTS = 1000
READ_COLUMNS = ("source_id", "url", "num_pages", "text", "page_offsets", "docling_text", "docling_page_offsets", "pdf")

# Tokens to show per leak census. A leak is a short head; content differences are a long tail.
LEAK_SAMPLES = 40
# A token has to appear on at least this share of documents to be worth calling a leak rather than
# one publisher's vocabulary.
LEAK_DOCUMENT_FLOOR = 0.01

# Every construct the normalizer has a rule for, plus the ones 1.17.0 introduced. Counted on the raw
# markdown, before any rule runs.
CONSTRUCTS = {
    "pipe_table_row": re.compile(r"^\|.*\|[ \t]*$", re.MULTILINE),
    "pipe_delimiter_row": re.compile(r"^\|[\s:|-]+\|[ \t]*$", re.MULTILINE),
    "html_table": re.compile(r"<table\b", re.IGNORECASE),
    "html_row": re.compile(r"<tr\b", re.IGNORECASE),
    "markdown_heading": re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE),
    "markdown_list": re.compile(r"^\s*[-*+]\s+", re.MULTILINE),
    "markdown_image": re.compile(r"!\[.*?\]\(.*?\)", re.DOTALL),
    "markdown_link": re.compile(r"\[[^\]\n]*\]\([^)\s]*\)"),
    "underline_tag": re.compile(r"<u>", re.IGNORECASE),
    "html_comment": re.compile(r"<!--"),
    "latex_math": re.compile(r"\$\$.*?\$\$|\$[^$\n]+\$", re.DOTALL),
    "emphasis": re.compile(r"\*\*|__"),
    "code_fence": re.compile(r"^```", re.MULTILINE),
    "blockquote": re.compile(r"^>\s", re.MULTILINE),
    "footnote_marker": re.compile(r"\[\^\d+\]"),
    "page_comment": re.compile(r"<!--\s*Page\s+\d+", re.IGNORECASE),
    "any_html_tag": re.compile(r"</?[a-zA-Z][^>]*>"),
}


def worker_main() -> None:
    """Serve length-prefixed documents from stdin, replying with per-page markdown."""
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


def construct_counts(pages: list[str]) -> Counter[str]:
    """How many times each markup construct appears across one document's pages."""
    found: Counter[str] = Counter()
    for page in pages:
        for name, pattern in CONSTRUCTS.items():
            found[name] += len(pattern.findall(page))
    return found


def repeated_first_line(pages: list[str]) -> bool:
    """Whether the document's first non-empty line repeats on a later page.

    The cheapest observable signature of a running header surviving extraction. If 1.17.0's
    page-edge furniture stripping were reaching this entry point, this share would collapse.
    """
    heads = []
    for page in pages:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        if lines:
            heads.append(lines[0])
    return len(heads) > 1 and heads.count(heads[0]) > 1


class LeakCensus:
    """Tokens one route produced and the other never did, counted over documents rather than uses.

    Documents rather than occurrences because a leak is a rule that fires on every document, and a
    single pathological file that repeats one word ten thousand times would otherwise dominate the
    list and hide it.
    """

    def __init__(self) -> None:
        self.documents = 0
        self.candidate_only: Counter[str] = Counter()
        self.reference_only: Counter[str] = Counter()

    def add(self, reference_tokens: list[str], candidate_tokens: list[str]) -> None:
        self.documents += 1
        reference, candidate = set(reference_tokens), set(candidate_tokens)
        self.candidate_only.update(candidate - reference)
        self.reference_only.update(reference - candidate)

    def report(self) -> dict:
        floor = max(1, int(LEAK_DOCUMENT_FLOOR * self.documents))
        return {
            "documents": self.documents,
            "candidate_only": [
                {"token": token, "documents": count, "share": count / self.documents}
                for token, count in self.candidate_only.most_common(LEAK_SAMPLES)
                if count >= floor
            ],
            "reference_only": [
                {"token": token, "documents": count, "share": count / self.documents}
                for token, count in self.reference_only.most_common(LEAK_SAMPLES)
                if count >= floor
            ],
        }


def audit_documents(table: pl.DataFrame) -> dict:
    """Run the library over every document and accumulate all three censuses."""
    worker = Worker(MODULE_NAME)
    constructs: Counter[str] = Counter()
    documents_with: Counter[str] = Counter()
    against_vlm, against_docling = LeakCensus(), LeakCensus()
    audited = failed = furniture = multipage = 0
    try:
        for row in table.iter_rows(named=True):
            reply = worker.call(EXTRACT_OP, row["pdf"])
            pages = reply.result.get("pages")
            if pages is None:
                failed += 1
                continue
            audited += 1

            counts = construct_counts(pages)
            constructs.update(counts)
            documents_with.update(name for name, count in counts.items() if count)
            if len(pages) > 1:
                multipage += 1
                furniture += repeated_first_line(pages)

            inspector = route_agreement.markdown_streams("\n".join(pages)).tokens
            vlm_pages = route_agreement.split_pages(row["text"], row["page_offsets"])
            against_vlm.add(route_agreement.markdown_streams("\n".join(vlm_pages)).tokens, inspector)
            if row["docling_text"] is not None:
                docling_pages = route_agreement.split_pages(row["docling_text"], row["docling_page_offsets"])
                against_docling.add(route_agreement.docling_streams("\n".join(docling_pages)).tokens, inspector)
    finally:
        worker.stop()

    return {
        "library_version": version("pdf-inspector"),
        "documents_audited": audited,
        "documents_failed": failed,
        "constructs": {
            name: {
                "occurrences": constructs[name],
                "documents": documents_with[name],
                "document_share": documents_with[name] / max(audited, 1),
            }
            for name in CONSTRUCTS
        },
        "running_header_survives": {
            "multipage_documents": multipage,
            "with_repeated_first_line": furniture,
            "share": furniture / max(multipage, 1),
        },
        "leaks_vs_vlm": against_vlm.report(),
        "leaks_vs_docling": against_docling.report(),
    }


def audit_frame() -> pl.DataFrame:
    """The same fixed document set the probe uses, so the two are talking about one corpus."""
    fs = storage()
    frames = []
    for _, shard in shards()[:SHARD_COUNT]:
        with fs.open(shard, "rb") as stream:
            frames.append(pl.read_parquet(stream, columns=list(READ_COLUMNS)))
    return pl.concat(frames).sort("source_id").head(AUDIT_DOCUMENTS)


def report(audit: dict) -> str:
    lines = [
        f"pdf-inspector {audit['library_version']}: {audit['documents_audited']} documents audited, "
        f"{audit['documents_failed']} failed",
        "",
        f"{'construct':<24} {'occurrences':>12} {'documents':>10} {'share':>8}",
    ]
    for name, counts in sorted(audit["constructs"].items(), key=lambda item: -item[1]["documents"]):
        lines.append(
            f"{name:<24} {counts['occurrences']:>12,} {counts['documents']:>10,} {counts['document_share']:>7.1%}"
        )
    furniture = audit["running_header_survives"]
    lines.append(
        f"\nrunning header reaches output on {furniture['with_repeated_first_line']}/"
        f"{furniture['multipage_documents']} multi-page documents ({furniture['share']:.1%})"
    )
    for name in ("leaks_vs_vlm", "leaks_vs_docling"):
        lines.append(f"\n== {name} ({audit[name]['documents']} documents) ==")
        for side in ("candidate_only", "reference_only"):
            head = ", ".join(f"{item['token']} {item['share']:.1%}" for item in audit[name][side][:20])
            lines.append(f"  {side}: {head or '(nothing above the floor)'}")
    return "\n".join(lines)


def main() -> None:
    configure_logging(logging.INFO)
    fs = storage()
    table = audit_frame()
    logger.info("format audit: %d documents", table.height)
    audit = audit_documents(table)
    with fs.open(OUTPUT_PATH, "w") as stream:
        json.dump(audit, stream, indent=2)
    print(report(audit))
    logger.info("wrote %s", OUTPUT_PATH)


if __name__ == "__main__":
    if WORKER_FLAG in sys.argv:
        worker_main()
    else:
        main()
