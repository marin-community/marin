# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read a model's declared preset revision inside a Git archive."""

import argparse
import json
import sys
from contextlib import redirect_stdout
from importlib import import_module
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("describe",))
    parser.add_argument("entry_module")
    parser.add_argument("model")
    parser.add_argument("preset")
    args = parser.parse_args()

    with redirect_stdout(sys.stderr):
        from experiments.post_training.rl_presets import ModelPresets  # noqa: PLC0415 - reserve stdout for JSON

        entry = import_module(args.entry_module)
    if not entry.__file__ or not Path(entry.__file__).resolve().is_relative_to(Path.cwd()):
        parser.error(f"{args.entry_module} did not load from the archive")
    revisions = {
        table.by_name[args.preset].revision
        for table in vars(entry).values()
        if isinstance(table, ModelPresets) and table.model == args.model and args.preset in table.by_name
    }
    print(json.dumps(revisions.pop() if len(revisions) == 1 else None))


if __name__ == "__main__":
    main()
