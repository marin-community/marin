# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Guest entry point for the original BM25 retrieval and snippet protocol."""

import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

ROOT = Path("/opt/browsecomp")


@contextmanager
def source_diagnostics() -> Iterator[None]:
    """Keep native and Python dependency logs off the JSON response channel."""
    # JVM logging writes to the native descriptor and bypasses redirect_stdout.
    sys.stdout.flush()
    with os.fdopen(os.dup(sys.stdout.fileno()), "w") as original:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        try:
            yield
        finally:
            sys.stdout.flush()
            os.dup2(original.fileno(), sys.stdout.fileno())


def main() -> None:
    request = json.load(sys.stdin)
    state = json.loads((ROOT / "state.json").read_text())
    if request["state"] != state:
        raise ValueError("BrowseComp provider state differs from its pinned image")
    with source_diagnostics():
        # The image-only backend initializes Java while importing its dependencies.
        backend = import_module(".retrieval", __package__)
        result = backend.retrieve(request, state, ROOT)
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
