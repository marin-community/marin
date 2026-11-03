"""FA2-only packaging preserves binaries and disables the optional FA4 import."""

import base64
import csv
import hashlib
import io
import subprocess
import sys
import zipfile

from scripts.wheels.package_flash_attn2 import SOURCE_VERSION, VERSION, package_fa2
from scripts.wheels.wheel_payloads import digest


def test_fa2_package_preserves_native_payload_and_optional_import_boundary(tmp_path):
    wheel = tmp_path / f"flash_attn-{SOURCE_VERSION}-cp312-cp312-linux_x86_64.whl"
    info = f"flash_attn-{SOURCE_VERSION}.dist-info/"
    native = b"\x7fELF\x02\x01" + bytes(12) + (62).to_bytes(2, "little") + b"original compiled FA2"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("flash_attn/__init__.py", "fa2_available = True\n")
        archive.writestr("flash_attn_2_cuda.so", native)
        archive.writestr("flash_attn/cute/__init__.py", "raise AttributeError('obsolete ThrMma')\n")
        archive.writestr("flash_attn/cute/utils.py", "# unqualified FA4\n")
        archive.writestr(info + "METADATA", f"Metadata-Version: 2.1\nName: flash-attn\nVersion: {SOURCE_VERSION}\n")
        archive.writestr(info + "WHEEL", "Wheel-Version: 1.0\nTag: cp312-cp312-linux_x86_64\n")
        archive.writestr(info + "RECORD", "")
    proof = package_fa2(wheel, digest(wheel.read_bytes()), tmp_path / "output")
    result = tmp_path / "output" / proof["output_wheel"]
    with zipfile.ZipFile(result) as archive:
        assert archive.read("flash_attn_2_cuda.so") == native
        assert not any(name.startswith("flash_attn/cute/") for name in archive.namelist())
        assert proof["native_members"] == {"flash_attn_2_cuda.so": hashlib.sha256(native).hexdigest()}
        record_name = f"flash_attn-{VERSION}.dist-info/RECORD"
        rows = list(csv.reader(io.StringIO(archive.read(record_name).decode())))
        assert {row[0] for row in rows} == set(archive.namelist())
        for name, expected_hash, size in rows:
            if name == record_name:
                assert (expected_hash, size) == ("", "")
                continue
            payload = archive.read(name)
            assert int(size) == len(payload)
            assert expected_hash == "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
    code = """
import sys
sys.path.insert(0, sys.argv[1])
from flash_attn import fa2_available
assert fa2_available
try:
    import flash_attn.cute
except ImportError:
    pass
else:
    raise AssertionError('FA4 unexpectedly available')
"""
    subprocess.run([sys.executable, "-I", "-c", code, str(result)], check=True)
