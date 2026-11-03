"""Release the qualified FA2 binaries without the unqualified FA4 namespace."""

import argparse
import json
import re
import zipfile
from pathlib import Path

from scripts.wheels.wheel_payloads import digest, repack

SOURCE_VERSION = "2.8.3.post1+marin.cu132torch2141.1"
VERSION = "2.8.3.post1+marin.cu132torch2141.2"
SOURCE_HASHES = {
    "linux_x86_64": "8ce2a4135a6cb5db2d2d04d8d9d084ed598cc48039c817c2eaff22f51b3dd04f",
    "linux_aarch64": "2e880125005ebcce2a11220f8e44d3740d743f9fe12fa4c0d57d6bb7ef2d5725",
}


def package_fa2(wheel: Path, source_sha256: str, output: Path) -> dict:
    """Remove optional FA4 Python files and prove all FA2 native bytes unchanged."""
    assert digest(wheel.read_bytes()) == source_sha256
    old_info = f"flash_attn-{SOURCE_VERSION}.dist-info/"
    new_info = f"flash_attn-{VERSION}.dist-info/"
    destination = output / wheel.name.replace(f"-{SOURCE_VERSION}-", f"-{VERSION}-", 1)
    assert wheel.name.startswith(f"flash_attn-{SOURCE_VERSION}-cp312-cp312-linux_")
    machine = 183 if wheel.name.endswith("linux_aarch64.whl") else 62
    with zipfile.ZipFile(wheel) as source:
        metadata = source.read(old_info + "METADATA")
        removed = frozenset(name for name in source.namelist() if name.startswith("flash_attn/cute/"))
    assert removed, "Expected the obsolete bundled FA4 namespace"
    pattern = rb"(?m)^Version: " + re.escape(SOURCE_VERSION.encode()) + rb"\r?$"
    metadata, count = re.subn(pattern, b"Version: " + VERSION.encode(), metadata)
    assert count == 1
    proof = repack(
        wheel, source_sha256, destination, old_info, new_info,
        {old_info + "METADATA": metadata}, machine, removed,
    )
    proof.update(
        classification="packaging_only_no_binary_recompile",
        old_version=SOURCE_VERSION,
        new_version=VERSION,
        removed_namespace="flash_attn.cute",
        only_payload_changes=["Remove flash_attn/cute/**", "METADATA Version header", "RECORD and dist-info paths"],
    )
    return proof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    args = parser.parse_args()
    platform = next(
        platform for platform in SOURCE_HASHES
        if args.wheel.name == f"flash_attn-{SOURCE_VERSION}-cp312-cp312-{platform}.whl"
    )
    proof = package_fa2(args.wheel, SOURCE_HASHES[platform], args.output)
    args.proof.write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({key: proof[key] for key in ("input_sha256", "output_wheel", "output_sha256", "native_members")}))


if __name__ == "__main__":
    main()
