# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scan trace files for the generation proxy's capability token, with the row builder's own test.

Every line of every file is checked with build.has_capability_token (decode each JWT-shaped string, line breaks and
JSON escapes tolerated; a hit is a payload whose audience is 'iris-proxy'), so this check and the builder cannot
disagree. The whole line is scanned, not only the fields the builder reads; each hit says whether it sits in
messages/tools (the builder's scope) or elsewhere. Tokens are never printed: a hit names the file, line, trial,
conversation, task and message index. Exits 1 if any file holds a token.

Arguments are .jsonl URIs or prefixes; a prefix is searched recursively for rebuilt*.jsonl and requests*.jsonl.

Usage:
    python -m experiments.post_training.sft_shootout_rows.check_tokens s3://.../traces/swesmith/ s3://.../x.jsonl
"""

import json
import sys

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sft_shootout_rows.build import has_capability_token

PATTERNS = ("rebuilt*.jsonl", "requests*.jsonl")


def expand(uri: str) -> list[str]:
    if uri.endswith(".jsonl"):
        return [uri]
    root = StoragePath(uri)
    return sorted({str(path) for pattern in PATTERNS for path in (root / f"**/{pattern}").glob()})


def scan(uri: str) -> dict:
    hits, lines = [], 0
    with StoragePath(uri).open("rt") as stream:
        for number, line in enumerate(stream):
            lines += 1
            if not has_capability_token(line):
                continue
            hit = {"line": number}
            try:
                record = json.loads(line)
            except ValueError:
                hits.append(hit | {"parse_error": True})
                continue
            messages = record.get("messages") or []
            hit |= {key: record.get(key) for key in ("trial_id", "conversation", "task_id") if key in record}
            hit["messages_with_token"] = [i for i, m in enumerate(messages) if has_capability_token(json.dumps(m))]
            hit["in_builder_scope"] = has_capability_token(json.dumps([messages, record.get("tools")]))
            hits.append(hit)
    return {"file": uri, "lines": lines, "hits": hits}


def main() -> None:
    files = [path for uri in sys.argv[1:] for path in expand(uri)]
    if not files:
        raise SystemExit("No trace files found")
    total = 0
    for path in files:
        report = scan(path)
        total += len(report["hits"])
        print("TOKEN_CHECK " + json.dumps(report), flush=True)
    print("TOKEN_CHECK_SUMMARY " + json.dumps({"files": len(files), "hits": total}), flush=True)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
