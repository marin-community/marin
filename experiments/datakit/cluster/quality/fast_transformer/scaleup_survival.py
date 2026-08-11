# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Survival analysis of the 88k prefix labels under the bme window scheme.

The existing labels were graded on ``excerpt(text, 10_500)`` — the first ~10.5k
characters of each document. Under the window scheme a grade is a verdict on a
specific 512-gemma-token window, so an existing label survives as the
*begin-window* grade exactly when the graded prefix covered at least the first
512 gemma tokens (or the whole document). Documents over ``LONG_DOC_TOKENS``
still need their middle and end windows graded fresh, whether or not the begin
grade survives.

Writes three artifacts under ``--out``:

``survival_docs.parquet``
    Per-document stats: gemma token counts, whether the existing label survives
    as the begin grade, and how many fresh windows the document needs.
``topup_windows.parquet``
    Every window still to be graded for the existing 88k documents (middle/end
    top-ups for long docs, plus all windows of the rare non-surviving docs),
    with exact window text and token offsets — the labeling driver's input.
``survival_report.json``
    Per-content-type aggregates and the new-document deficits against the
    per-type target.
"""

import argparse
import json
import logging

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.bme_windows import (
    LONG_DOC_TOKENS,
    WINDOW_TOKENS,
    check_gigatoken_parity,
    doc_windows,
    encode_documents,
)
from experiments.datakit.cluster.quality.fast_transformer.joined_labels import DEFAULT_JOINED, load_joined
from experiments.datakit.cluster.quality.fast_transformer.rubric import CONTENT_TYPES
from experiments.datakit.cluster.quality.fast_transformer.sample_labels import EXCERPT_NOTICE, excerpt

logger = logging.getLogger(__name__)

# What the deployed labeler put in front of the grader (label_with_glm52.PROMPT_TEXT_CHARS).
GRADED_PREFIX_CHARS = 10_500
DOCS_PER_TYPE_TARGET = 20_000

SURVIVAL_COLUMNS = ["id", "text", "glm52_source", "glm52_content_type", "glm52_quality"]


def graded_prefix(text: str) -> str:
    """The text the original grader saw, minus the harness's excerpt notice."""
    return excerpt(text, GRADED_PREFIX_CHARS).removesuffix(EXCERPT_NOTICE)


def analyze(joined: dict[str, list]) -> tuple[list[dict], list[dict]]:
    """Per-doc survival rows and the fresh windows still needing a grade."""
    texts = joined["text"]
    check_gigatoken_parity(texts)
    doc_ids_tokens = encode_documents(texts)

    # A doc whose text fits the graded prefix was graded whole; otherwise the
    # label survives as a begin grade iff the prefix covered >= one full window.
    over_prefix = [i for i, t in enumerate(texts) if len(t) > GRADED_PREFIX_CHARS]
    prefix_tokens = {}
    if over_prefix:
        counts = encode_documents([graded_prefix(texts[i]) for i in over_prefix])
        prefix_tokens = {i: len(ids) for i, ids in zip(over_prefix, counts, strict=True)}

    docs: list[dict] = []
    fresh: list[dict] = []
    for i, token_ids in enumerate(doc_ids_tokens):
        n = len(token_ids)
        survives = i not in prefix_tokens or prefix_tokens[i] >= WINDOW_TOKENS
        windows = doc_windows(token_ids)
        needed = [w for w in windows if not (survives and w.position == "begin")]
        docs.append(
            {
                "id": joined["id"][i],
                "source": joined["glm52_source"][i],
                "content_type": joined["glm52_content_type"][i],
                "quality": joined["glm52_quality"][i],
                "doc_chars": len(texts[i]),
                "doc_tokens": n,
                "graded_prefix_tokens": prefix_tokens.get(i, n),
                "survives_begin": survives,
                "is_long": n > LONG_DOC_TOKENS,
                "windows_total": len(windows),
                "windows_needed": len(needed),
            }
        )
        for w in needed:
            fresh.append(
                {
                    "id": joined["id"][i],
                    "source": joined["glm52_source"][i],
                    "window": w.position,
                    "token_start": w.token_start,
                    "token_end": w.token_end,
                    "text": w.text,
                    "doc_tokens": n,
                    "kind": "topup" if survives else "regrade",
                }
            )
        if i and i % 10_000 == 0:
            logger.info("survival: analyzed %d/%d documents", i, len(texts))
    return docs, fresh


def report(docs: list[dict]) -> dict:
    """Per-type aggregates: surviving grades, top-up volume, new-doc deficits."""
    per_type: dict[str, dict] = {}
    for ct in CONTENT_TYPES:
        rows = [d for d in docs if d["content_type"] == ct]
        surviving = [d for d in rows if d["survives_begin"]]
        per_type[ct] = {
            "docs": len(rows),
            "docs_surviving_begin": len(surviving),
            "docs_not_surviving": len(rows) - len(surviving),
            "long_docs": sum(1 for d in rows if d["is_long"]),
            "surviving_begin_examples": len(surviving),
            "fresh_windows_needed": sum(d["windows_needed"] for d in rows),
            "total_examples_after_topup": sum(d["windows_total"] for d in rows),
            "new_doc_deficit": max(0, DOCS_PER_TYPE_TARGET - len(rows)),
        }
    totals = {
        "docs": len(docs),
        "surviving_begin": sum(1 for d in docs if d["survives_begin"]),
        "fresh_windows_needed": sum(d["windows_needed"] for d in docs),
        "new_doc_deficit": sum(v["new_doc_deficit"] for v in per_type.values()),
    }
    return {"per_type": per_type, "totals": totals, "docs_per_type_target": DOCS_PER_TYPE_TARGET}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joined-dir", default=DEFAULT_JOINED)
    parser.add_argument("--out", required=True, help="output prefix for the survival artifacts")
    parser.add_argument("--limit", type=int, default=None, help="analyze only the first N docs (smoke runs)")
    args = parser.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    joined = load_joined(args.joined_dir, columns=SURVIVAL_COLUMNS)
    if args.limit:
        joined = {c: v[: args.limit] for c, v in joined.items()}
    logger.info("survival: %d joined label rows", len(joined["id"]))

    docs, fresh = analyze(joined)
    out = args.out.rstrip("/")
    with StoragePath(f"{out}/survival_docs.parquet").open("wb") as fh:
        pq.write_table(pa.Table.from_pylist(docs), fh)
    with StoragePath(f"{out}/topup_windows.parquet").open("wb") as fh:
        pq.write_table(pa.Table.from_pylist(fresh), fh)
    summary = report(docs)
    with StoragePath(f"{out}/survival_report.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("survival report: %s", json.dumps(summary, indent=2))
    logger.info("survival: wrote artifacts under %s", out)


if __name__ == "__main__":
    main()
