#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Switch the native (maturin) packages between dev mode (source build) and
user mode (pre-built wheel).

Covers marin-dupekit and marin-finelog. Operates on pyproject.toml by replacing
the block between RUST-DEV markers:
    # ### BEGIN RUST-DEV SOURCES ###
    ...
    # ### END RUST-DEV SOURCES ###

Usage:
    python scripts/rust_mode.py dev    # insert editable path sources
    python scripts/rust_mode.py user   # clear the block (use pre-built wheels)
    python scripts/rust_mode.py status # print current mode
"""

import pathlib
import re
import sys

BEGIN = "# ### BEGIN RUST-DEV SOURCES ###"
END = "# ### END RUST-DEV SOURCES ###"

# Editable path sources injected in dev mode. Each native package's wheel is
# built by maturin from its own lib/{pkg} dir (the [tool.maturin] manifest-path
# points at the package's rust/ crate), so an editable install builds the
# native extension from source.
DEV_SOURCES = "\n".join(
    [
        'marin-dupekit = { path = "lib/dupekit", editable = true }',
        'marin-finelog = { path = "lib/finelog", editable = true }',
    ]
)

PYPROJECT = pathlib.Path("pyproject.toml")


def _read() -> str:
    txt = PYPROJECT.read_text()
    if BEGIN not in txt or END not in txt:
        print("ERROR: RUST-DEV markers missing from pyproject.toml", file=sys.stderr)
        sys.exit(1)
    return txt


def _replace_block(txt: str, inner: str) -> str:
    block = BEGIN + "\n" + (inner + "\n" if inner else "") + END
    return re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), block, txt, flags=re.DOTALL)


def _current_mode(txt: str) -> str:
    m = re.search(re.escape(BEGIN) + r"(.*?)" + re.escape(END), txt, flags=re.DOTALL)
    if m and m.group(1).strip():
        return "dev"
    return "user"


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("dev", "user", "status"):
        print(__doc__.strip())
        sys.exit(1)

    mode = sys.argv[1]
    txt = _read()

    if mode == "status":
        current = _current_mode(txt)
        print(f"Rust build mode: {current}")
        if current == "dev":
            print("  dupekit/finelog are built from source (lib/dupekit, lib/finelog)")
        else:
            print("  dupekit/finelog are installed from pre-built wheels")
        return

    if mode == "dev":
        if _current_mode(txt) == "dev":
            print("Already in dev mode.")
            return
        new = _replace_block(txt, DEV_SOURCES)
        PYPROJECT.write_text(new)
        print("Switched to dev mode: dupekit/finelog will build from source.")
        print("Do NOT commit pyproject.toml in this state.")
    else:
        new = _replace_block(txt, "")
        PYPROJECT.write_text(new)
        print("Switched to user mode: dupekit/finelog from pre-built wheels.")


if __name__ == "__main__":
    main()
