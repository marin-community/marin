# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Find and remove Iris capability tokens from every text file under a run prefix.

A capability token is a JWT whose payload has ``aud`` ``iris-proxy``. Agents can print the proxy URL
(e.g. mini-swe-agent's ``OPENAI_API_BASE`` in an ``env`` listing), so it can land in observations,
``requests.jsonl`` and the rebuilt traces. Other JWTs (task fixtures) are left alone. Tokens split
across lines (also JSON-escaped ``\\n``) are matched. With ``--check`` the files are only scanned, and
the exit code is 1 if any token is found; use it as the gate before publishing.

    uv run python experiments/post_training/sft_shootout/scrub_run.py <prefix> [--check]
"""

import argparse
import base64
import json
import re
import sys

from rigging.filesystem.storage_path import StoragePath

GAP = r"(?:\s|\\n)*"
SEGMENT = rf"[A-Za-z0-9_\-](?:[A-Za-z0-9_\-]|{GAP})*"
JWT = re.compile(rf"eyJ{SEGMENT}\.{GAP}{SEGMENT}\.{GAP}{SEGMENT}")
REPLACEMENT = "<capability-token>"


def is_capability(token: str) -> bool:
    parts = re.sub(GAP, "", token).split(".")
    if len(parts) != 3:
        return False
    try:
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
    except (ValueError, UnicodeDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("aud") == "iris-proxy"


def scrub_text(text: str) -> tuple[str, int]:
    found = 0

    def replace(match: re.Match) -> str:
        nonlocal found
        if is_capability(match.group(0)):
            found += 1
            return REPLACEMENT
        return match.group(0)

    return JWT.sub(replace, text), found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prefix")
    parser.add_argument("--check", action="store_true", help="Only report; exit 1 if any token is found.")
    args = parser.parse_args()
    total = 0
    for directory, _, names in StoragePath(args.prefix).walk():
        for name in names:
            path = StoragePath(f"{directory}/{name}")
            try:
                text = path.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            if "eyJ" not in text:
                continue
            scrubbed, found = scrub_text(text)
            if found:
                total += found
                print(f"{path}: {found}")
                if not args.check:
                    path.write_text(scrubbed)
    print(json.dumps({"prefix": args.prefix, "capability_tokens": total, "scrubbed": not args.check}))
    if args.check and total:
        sys.exit(1)


if __name__ == "__main__":
    main()
