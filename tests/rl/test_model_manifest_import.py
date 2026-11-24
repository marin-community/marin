# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path


def test_vendor_model_manifest_round_trip_without_top_level_namespace():
    # CPU coordinators import the vendor namespace without installing the GPU runtime.
    script = """
import json
import sys
sys.path.insert(0, sys.argv[1])
sys.modules['marinskyrl'] = None
from vendor.skyrl.marinskyrl.model_manifest import (
    ModelManifest, ModelManifestFile, build_model_manifest,
)
files = ('config.json', 'tokenizer.json', 'model.safetensors', 'model.safetensors.index.json')
entries = tuple(ModelManifestFile(path=path, size=2, sha256='a' * 64) for path in files)
manifest = build_model_manifest(entries, 'test/model', 'b' * 40)
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
    assert set(json.loads(result.stdout)["files"]) == {
        "config.json",
        "tokenizer.json",
        "model.safetensors",
        "model.safetensors.index.json",
    }
