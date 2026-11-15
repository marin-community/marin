# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The image-local JSON protocol for the fresh OpenResearcher BM25 environment."""

import json
import sys
from importlib import import_module
from pathlib import Path

from browsecomp.tool import source_diagnostics

ROOT = Path("/opt/openresearcher")


def main() -> None:
    request = json.load(sys.stdin)
    state = json.loads((ROOT / "state.json").read_text())
    if request["state"] != state:
        raise ValueError("Requested retrieval state differs from the immutable OpenResearcher image")
    with source_diagnostics():
        backend = import_module("browsecomp.retrieval")
        result = backend.retrieve(request, state, ROOT)
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
