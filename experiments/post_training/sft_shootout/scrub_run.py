# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Remove Iris capability tokens from every text file under a run prefix.

An agent that reaches ``trace_proxy`` through an Iris capability URL can print that URL, so the token
can land in observations, the request log and rebuilt traces. Run this over the outputs before
publishing them. A scrubbed file is counted again and must hold no token. With ``--check`` the files
are only scanned. Files that are not UTF-8 text are listed and counted but not scanned. The exit code
is 1 if ``--check`` finds a token or a scrub leaves one.

    uv run python experiments/post_training/sft_shootout/scrub_run.py <prefix> [--check]
"""

import argparse
import json
import sys

from iris.cluster.redaction import count_capability_tokens, scrub_capability_tokens
from rigging.filesystem.storage_path import StoragePath


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prefix")
    parser.add_argument("--check", action="store_true", help="Only report; exit 1 if any token is found.")
    args = parser.parse_args()
    total = left = binary = 0
    for directory, _, names in StoragePath(args.prefix).walk():
        for name in names:
            path = directory / name
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                binary += 1
                print(f"{path}: not text, not scanned")
                continue
            found = count_capability_tokens(text)
            if not found:
                continue
            total += found
            if args.check:
                print(f"{path}: {found}")
                continue
            scrubbed, replaced = scrub_capability_tokens(text)
            remaining = count_capability_tokens(scrubbed)
            left += remaining
            print(f"{path}: {found} found, {replaced} replaced, {remaining} left")
            path.write_text(scrubbed)
    summary = {"prefix": args.prefix, "capability_tokens": total, "left": left, "binary_files": binary}
    print(json.dumps({**summary, "scrubbed": not args.check}))
    if (args.check and total) or left:
        sys.exit(1)


if __name__ == "__main__":
    main()
