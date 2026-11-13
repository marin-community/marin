# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replace pure-Python modules inside a released vLLM wheel, keeping its compiled kernels.

Each replacement is ``<path inside the wheel>=<local file>``. The wheel's own copy of every replaced
path must equal ``--expect-base-dir/<path>`` (the release's source commit), so an overlay can only
apply on top of the exact release it was merged against. RECORD hashes are rewritten.
"""

import argparse
import base64
import hashlib
import zipfile
from pathlib import Path


def _record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-base-dir", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    replacements = {path: (args.overlay_dir / path).read_bytes() for path in args.paths}
    with zipfile.ZipFile(args.base) as source:
        names = source.namelist()
        record_name = next(name for name in names if name.endswith(".dist-info/RECORD"))
        for path in args.paths:
            assert source.read(path) == (args.expect_base_dir / path).read_bytes(), f"{path} differs from base"
        with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as target:
            for info in source.infolist():
                if info.filename == record_name:
                    continue
                data = replacements.get(info.filename)
                if data is None:
                    target.writestr(info, source.read(info.filename))
                else:
                    target.writestr(info, data)
            record_lines = []
            for line in source.read(record_name).decode().splitlines():
                path = line.split(",", 1)[0]
                if path in replacements:
                    data = replacements[path]
                    line = f"{path},{_record_hash(data)},{len(data)}"
                record_lines.append(line)
            target.writestr(source.getinfo(record_name), "\n".join(record_lines) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"{args.output} sha256={digest}")


if __name__ == "__main__":
    main()
