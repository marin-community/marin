import copy
import hashlib
import json
from pathlib import Path

import pytest

from capability_pipeline.daytona_policy import verifier_snapshot_recipe
from capability_pipeline.image_runtime_metadata import (
    DEFAULT_CATALOG,
    ImageRuntimeMetadataError,
    derive_daytona_recipe,
)
from capability_pipeline.oci_artifact import ValidatedLayer, build_oci_metadata

ROOT = Path(__file__).parents[1]
PLAN = ROOT / "docs/audits/c32_image_publication_plan_002.json"
RESULTS = ROOT / "runs/c32-publication-002/results"
PUBLICATIONS = {
    "candidate": RESULTS / "candidate-publication.json",
    "private_verifier": RESULTS / "private-verifier-publication.json",
}


def _catalog():
    return json.loads(DEFAULT_CATALOG.read_text())


def test_catalog_is_exactly_the_published_plan_metadata():
    plan = json.loads(PLAN.read_text())
    catalog = _catalog()["images"]
    assert len(catalog) == 2
    for role, path in PUBLICATIONS.items():
        receipt = json.loads(path.read_text())
        published = receipt["publication"]
        planned = next(image for image in plan["images"] if image["role"] == role)
        layer = ValidatedLayer(
            published["layer_digest"],
            published["layer_bytes"],
            published["diff_id"],
            published["uncompressed_bytes"],
        )
        config, manifest = build_oci_metadata(
            layer,
            image_config=planned["image_config"],
            architecture=planned["architecture"],
            operating_system=planned["operating_system"],
        )
        assert hashlib.sha256(config).hexdigest() == published["config_digest"][7:]
        assert hashlib.sha256(manifest).hexdigest() == published["manifest_digest"][7:]
        assert len(config) == published["config_bytes"]
        assert len(manifest) == published["manifest_bytes"]
        assert catalog[published["image"]] == {
            "config": json.loads(config),
            "manifest": json.loads(manifest),
        }


@pytest.mark.parametrize(
    ("role", "entrypoint"),
    [("candidate", "/opt/task/entrypoint.sh"), ("private_verifier", "/opt/verifier/entrypoint.sh")],
)
def test_cataloged_recipe_restates_verified_entrypoint(role, entrypoint):
    receipt = json.loads(PUBLICATIONS[role].read_text())
    reference = receipt["publication"]["image"]
    assert derive_daytona_recipe(reference) == (
        f"FROM {reference}\nENTRYPOINT [\"{entrypoint}\"]\n"
    )


def test_unknown_digest_reference_retains_legacy_recipe():
    reference = "registry.example/repository@sha256:" + "0" * 64
    assert derive_daytona_recipe(reference) == f"FROM {reference}\n"


def test_cataloged_verifier_recipe_preserves_entrypoint_and_selected_supervisor():
    receipt = json.loads(PUBLICATIONS["private_verifier"].read_text())
    reference = receipt["publication"]["image"]
    recipe = verifier_snapshot_recipe(reference, "/opt/py312/bin/python3")
    assert recipe.startswith(
        f"FROM {reference}\nENTRYPOINT [\"/opt/verifier/entrypoint.sh\"]\nUSER root\n"
    )
    assert "RUN /opt/py312/bin/python3 -m pip install --no-cache-dir --no-deps " in recipe


def test_unknown_verifier_recipe_uses_default_supervisor_for_pip():
    reference = "registry.example/repository@sha256:" + "0" * 64
    recipe = verifier_snapshot_recipe(reference)
    assert recipe.startswith(
        f"FROM {reference}\nUSER root\nRUN python3 -m pip install "
    )
    assert "\nRUN pip install " not in recipe


@pytest.mark.parametrize("mutation", ["entrypoint", "config_digest", "manifest"])
def test_cataloged_metadata_tampering_fails_closed(tmp_path, mutation):
    document = copy.deepcopy(_catalog())
    reference, record = next(iter(document["images"].items()))
    if mutation == "entrypoint":
        record["config"]["config"]["Entrypoint"] = ["/tampered"]
    elif mutation == "config_digest":
        record["manifest"]["config"]["digest"] = "sha256:" + "0" * 64
    else:
        record["manifest"]["layers"][0]["size"] += 1
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ImageRuntimeMetadataError):
        derive_daytona_recipe(reference, catalog_path=path)


@pytest.mark.parametrize("reference", ["tagged:latest", "bad\nFROM attacker"])
def test_reference_must_be_canonical_digest(reference):
    with pytest.raises(ImageRuntimeMetadataError):
        derive_daytona_recipe(reference)
