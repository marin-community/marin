# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A stand-in for ``driver allocation`` in the seal tests: it ends with the terminal ``FAKE_TERMINAL`` names.

It appends a line to ``tree/progress.txt``, so a resumed allocation shows the tree it restored, and writes
``records.jsonl`` and ``terminal.json`` as the real child does.
"""

import json
import os
import sys
from pathlib import Path


def main() -> None:
    _, command, flag, spec_path = sys.argv
    assert (command, flag) == ("allocation", "--spec")
    spec_file = Path(spec_path)
    root = Path(json.loads(spec_file.read_text())["root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "progress.txt").open("a") as handle:
        handle.write("ran\n")
    terminal = os.environ["FAKE_TERMINAL"]
    (root / "records.jsonl").write_text(json.dumps({"row": "terminal", "decision": terminal}) + "\n")
    (spec_file.parent / "terminal.json").write_text(json.dumps({"terminal": terminal, "kind": None, "reason": None}))


if __name__ == "__main__":
    main()
