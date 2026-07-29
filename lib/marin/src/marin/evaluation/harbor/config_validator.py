# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate and canonicalize one native Harbor config in an isolated pinned environment."""

import json
import sys
from collections.abc import Mapping
from pathlib import Path

import yaml
from harbor_config import JobConfig, canonical_json
from pydantic import ValidationError


def _document(path: Path) -> Mapping:
    if path.suffix in {".yaml", ".yml"}:
        document = yaml.safe_load(path.read_text())
    elif path.suffix == ".json":
        document = json.loads(path.read_text())
    else:
        raise ValueError(f"unsupported Harbor config file format: {path.suffix}")
    if not isinstance(document, Mapping):
        raise ValueError("Harbor config must contain a mapping")
    return document


def main() -> None:
    path = Path(sys.argv[1])
    document = dict(_document(path))
    document.setdefault("job_name", path.stem)
    try:
        config = JobConfig.model_validate(document, extra="forbid")
    except ValidationError as exc:
        print(exc.json(include_url=False, include_input=False), file=sys.stderr)
        raise SystemExit(2) from exc
    sys.stdout.buffer.write(canonical_json(config))


if __name__ == "__main__":
    main()
