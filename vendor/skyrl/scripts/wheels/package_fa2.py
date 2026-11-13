"""Package the qualified FA2 kernels without the optional bundled FA4 code."""

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile


SOURCE_VERSION = "2.8.3.post1+marin.cu132torch2141.1"
VERSION = "2.8.3.post2+marin.fa2.cu132torch2141.1"
SOURCE_HASHES = {
    "aarch64": "2e880125005ebcce2a11220f8e44d3740d743f9fe12fa4c0d57d6bb7ef2d5725",
    "x86_64": "8ce2a4135a6cb5db2d2d04d8d9d084ed598cc48039c817c2eaff22f51b3dd04f",
}
SOURCE_RECIPE = "https://github.com/marin-community/MarinSkyRL/tree/604f2c1463032ffc1f47da114f43cf1d550dd473/scripts/wheels"


def package(wheel: Path, output: Path) -> dict:
    """Verify an exact native build and preserve every FA2 file in a new wheel."""
    architecture = wheel.name.removesuffix(".whl").rsplit("linux_", 1)[1]
    assert wheel.name == f"flash_attn-{SOURCE_VERSION}-cp312-cp312-linux_{architecture}.whl"
    with wheel.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    assert source_hash == SOURCE_HASHES[architecture], "Unexpected input wheel"
    old_info = f"flash_attn-{SOURCE_VERSION}.dist-info/"
    new_info = f"flash_attn-{VERSION}.dist-info/"
    output.mkdir(parents=True, exist_ok=True)
    destination = output / wheel.name.replace(SOURCE_VERSION, VERSION, 1)
    preserved = {}
    records = []
    with zipfile.ZipFile(wheel) as source, destination.open("xb") as stream:
        removed = sorted(name for name in source.namelist() if name.startswith("flash_attn/cute/"))
        assert len(removed) == 21 and all(name.endswith(".py") for name in removed)
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target:
            for name in sorted(set(source.namelist()) - set(removed) - {old_info + "RECORD"}):
                data = source.read(name)
                new_name = name.replace(old_info, new_info, 1) if name.startswith(old_info) else name
                if name == old_info + "METADATA":
                    old = f"Version: {SOURCE_VERSION}\n".encode()
                    assert data.count(old) == 1
                    data = data.replace(old, f"Version: {VERSION}\n".encode(), 1)
                else:
                    preserved[new_name] = hashlib.sha256(data).hexdigest()
                info = zipfile.ZipInfo(new_name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                target.writestr(info, data, compresslevel=9)
                checksum = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                records.append((new_name, f"sha256={checksum}", str(len(data))))
            record = io.StringIO(newline="")
            csv.writer(record, lineterminator="\n").writerows([*records, (new_info + "RECORD", "", "")])
            info = zipfile.ZipInfo(new_info + "RECORD", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            target.writestr(info, record.getvalue(), compresslevel=9)
    with destination.open("rb") as stream:
        output_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    native = {name: digest for name, digest in preserved.items() if name.endswith(".so")}
    assert len(native) == 1
    proof = {
        "source_recipe": SOURCE_RECIPE,
        "input_wheel": wheel.name,
        "input_sha256": source_hash,
        "output_wheel": destination.name,
        "output_sha256": output_hash,
        "removed_optional_fa4_files": removed,
        "preserved_members": preserved,
        "native_members": native,
        "changed_metadata": ["Version", "dist-info paths", "RECORD"],
    }
    (output / f"fa2-{architecture}-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    proof = package(args.wheel, args.output)
    print(json.dumps({key: proof[key] for key in ("output_wheel", "output_sha256", "native_members")}))


if __name__ == "__main__":
    main()
