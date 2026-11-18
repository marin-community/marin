# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path


def test_portable_model_manifest_round_trip_without_installed_skyrl():
    # CPU coordinators import the vendor namespace without installing the GPU runtime.
    script = """
import json
import sys
sys.path.insert(0, sys.argv[1])
sys.modules['marinskyrl'] = None
from vendor.skyrl.marinskyrl.model_manifest import (
    ModelManifest, ModelManifestFile, build_model_manifest,
)
entry = ModelManifestFile(path='config.json', size=2, sha256='a' * 64)
manifest = build_model_manifest((entry,), 'test/model', 'b' * 40)
restored = ModelManifest.from_mapping(json.loads(manifest.model_dump_json()), '/tmp/model')
print(json.dumps({'identity': restored.identity, 'files': [file.path for file in restored.files]}))
assert restored.identity == manifest.identity
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(Path(__file__).resolve().parents[2])],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(result.stdout)["files"] == ["config.json"]
