# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import zipfile
from pathlib import Path

import pytest

from infra.ci.check_skyrl import BUNDLE_MANIFEST, audit_wheel, wheel_source_files


@pytest.mark.parametrize("defect", ["missing", "changed", "untracked"])
def test_wheel_audit_accepts_source_edits_and_rejects_different_packaged_contents(tmp_path: Path, defect: str) -> None:
    source = tmp_path / "source"
    files = {
        BUNDLE_MANIFEST: b"marinskyrl/normal_edit.py\n",
        "skyrl-train/skyrl_train/assets/blob.bin": b"edited runtime data",
        "marinskyrl/normal_edit.py": b"value = 7\n",
    }
    for name, content in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    wheel = tmp_path / "marinskyrl.whl"
    asset = "skyrl_train/assets/blob.bin"

    def package(with_defect: bool) -> None:
        with zipfile.ZipFile(wheel, "w") as archive:
            for member, path in wheel_source_files(source).items():
                if with_defect and member == asset and defect != "untracked":
                    if defect == "changed":
                        archive.writestr(member, b"changed runtime data")
                    continue
                archive.write(path, member)
            if with_defect and defect == "untracked":
                archive.writestr("marinskyrl/untracked.py", "raise RuntimeError('untracked runtime')")

    package(with_defect=False)
    assert asset in audit_wheel(wheel, source)
    with zipfile.ZipFile(wheel) as archive:
        assert archive.read(asset) == b"edited runtime data"
        assert archive.read("marinskyrl/normal_edit.py") == b"value = 7\n"
    package(with_defect=True)
    with pytest.raises(ValueError, match=r"Wheel member missing or changed|outside the checked-out source"):
        audit_wheel(wheel, source)
