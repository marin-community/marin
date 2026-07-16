#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Switch the native (maturin) packages between dev mode (source build) and
user mode (pre-built wheel).

Covers the native packages marin-dupekit-native and marin-finelog-server. Their
pure-Python fronts (marin-dupekit, marin-finelog) are permanent workspace
members (always built from source, see the root pyproject), so they are not
toggled here. Operates on each target pyproject.toml by replacing the block
between RUST-DEV markers:
    # ### BEGIN RUST-DEV SOURCES ###
    ...
    # ### END RUST-DEV SOURCES ###

Three files carry markers: the repo-root pyproject.toml (governs the root
workspace venv) and each pure package's pyproject.toml — lib/dupekit and
lib/finelog — which govern in-dir `uv run` in those members.

Usage:
    python scripts/rust_mode.py dev    # insert path sources (build from source)
    python scripts/rust_mode.py user   # clear the blocks (use pre-built wheels)
    python scripts/rust_mode.py status # print current mode
"""

import pathlib
import re
import sys

BEGIN = "# ### BEGIN RUST-DEV SOURCES ###"
END = "# ### END RUST-DEV SOURCES ###"

# Path sources injected in dev mode, per pyproject. Both native packages are
# plain path sources — their [tool.uv] cache-keys cover the Rust sources, so
# `uv sync` rebuilds the extension when they change. The pure fronts
# (marin-dupekit, marin-finelog) are omitted: they are permanent workspace
# members and always resolve from source.
TARGETS = [
    (
        pathlib.Path("pyproject.toml"),
        "\n".join(
            [
                'marin-dupekit-native = { path = "lib/dupekit/rust" }',
                'marin-finelog-server = { path = "lib/finelog/rust" }',
            ]
        ),
    ),
    (
        pathlib.Path("lib/dupekit/pyproject.toml"),
        'marin-dupekit-native = { path = "rust" }',
    ),
    (
        pathlib.Path("lib/finelog/pyproject.toml"),
        'marin-finelog-server = { path = "rust" }',
    ),
]


def _read(path: pathlib.Path) -> str:
    txt = path.read_text()
    if BEGIN not in txt or END not in txt:
        print(f"ERROR: RUST-DEV markers missing from {path}", file=sys.stderr)
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
    texts = {path: _read(path) for path, _ in TARGETS}

    if mode == "status":
        modes = {path: _current_mode(txt) for path, txt in texts.items()}
        overall = "dev" if "dev" in modes.values() else "user"
        print(f"Rust build mode: {overall}")
        if overall == "dev":
            print("  dupekit/finelog are built from source (lib/dupekit, lib/finelog)")
        else:
            print("  dupekit/finelog are installed from pre-built wheels")
        if len(set(modes.values())) > 1:
            for path, m in modes.items():
                print(f"  WARNING: mixed state — {path} is in {m} mode")
        return

    for path, dev_sources in TARGETS:
        inner = dev_sources if mode == "dev" else ""
        path.write_text(_replace_block(texts[path], inner))

    if mode == "dev":
        print("Switched to dev mode: dupekit/finelog will build from source.")
        marker_files = ", ".join(str(path) for path, _ in TARGETS)
        print(f"Do NOT commit these in this state: {marker_files}")
    else:
        print("Switched to user mode: dupekit/finelog from pre-built wheels.")


if __name__ == "__main__":
    main()
