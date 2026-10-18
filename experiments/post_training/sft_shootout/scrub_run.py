# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Find and remove Iris capability tokens from every text file under a run prefix.

A capability token is a JWT whose payload has ``aud`` ``iris-proxy``. Agents can print the proxy URL
(e.g. mini-swe-agent's ``OPENAI_API_BASE`` in an ``env`` listing), so it can land in observations,
``requests.jsonl`` and the rebuilt traces. Other JWTs (task fixtures) are left alone. Tokens split
across lines are matched, including escaped breaks behind any number of backslashes (a log line in a
tool output, inside JSON, inside JSON). The count strips every break and decodes every JWT, as the #211
guard does; a scrubbed file is counted again and must hold none. With ``--check`` the files are only
scanned. The exit code is 1 if ``--check`` finds a token or a scrub leaves one; use it as the gate
before publishing.

    uv run python experiments/post_training/sft_shootout/scrub_run.py <prefix> [--check]
"""

import argparse
import base64
import json
import re
import sys

from rigging.filesystem.storage_path import StoragePath

# A line break inside a token: whitespace, or an escaped n/r/t behind any number of backslashes (JSON inside
# JSON doubles them, e.g. a tool output holding a log line, inside rebuilt.jsonl). Same class as #211's guard.
BREAK = r"(?:\s|\\+[nrt])"
BREAKS = re.compile(rf"{BREAK}+")
SEGMENT = rf"[A-Za-z0-9_\-](?:[A-Za-z0-9_\-]|{BREAK})*"
JWT = re.compile(rf"e{BREAK}*y{BREAK}*J{SEGMENT}\.{BREAK}*{SEGMENT}\.{BREAK}*{SEGMENT}")
PLAIN_JWT = re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")
REPLACEMENT = "<capability-token>"


def is_capability(token: str) -> bool:
    parts = BREAKS.sub("", token).split(".")
    if len(parts) != 3:
        return False
    try:
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
    except (ValueError, UnicodeDecodeError):
        return False
    audience = payload.get("aud") if isinstance(payload, dict) else None
    return audience == "iris-proxy" or (isinstance(audience, list) and "iris-proxy" in audience)


def count_tokens(text: str) -> int:
    """Capability tokens in ``text`` with every line break removed first (the #211 guard's test)."""
    return sum(is_capability(token) for token in PLAIN_JWT.findall(BREAKS.sub("", text)))


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
    total = left = 0
    for directory, _, names in StoragePath(args.prefix).walk():
        for name in names:
            path = StoragePath(f"{directory}/{name}")
            try:
                text = path.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            found = count_tokens(text)
            if not found:
                continue
            total += found
            if args.check:
                print(f"{path}: {found}")
                continue
            scrubbed, replaced = scrub_text(text)
            remaining = count_tokens(scrubbed)
            left += remaining
            print(f"{path}: {found} found, {replaced} replaced, {remaining} left")
            path.write_text(scrubbed)
    print(json.dumps({"prefix": args.prefix, "capability_tokens": total, "left": left, "scrubbed": not args.check}))
    if (args.check and total) or left:
        sys.exit(1)


if __name__ == "__main__":
    main()
