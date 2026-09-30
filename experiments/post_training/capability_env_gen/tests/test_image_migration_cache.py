import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from capability_pipeline.daytona_policy import VERIFIER_BOOTSTRAP

MODULE = Path(__file__).parents[1] / "scripts/preflight_image_migration_cache.py"
SPEC = importlib.util.spec_from_file_location("image_migration_cache", MODULE)
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)


class Snapshots:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.lookups = []

    def get(self, name):
        self.lookups.append(name)
        return self.snapshot


def client(image: str, *, dockerfile: str | None = None, state: str = "ACTIVE"):
    expected = f"FROM {image}\nUSER root\n{VERIFIER_BOOTSTRAP}"
    digest = __import__("hashlib").sha256(expected.encode()).hexdigest()
    snapshot = SimpleNamespace(
        id="provider-snapshot-id",
        name=f"cap-verifier-{digest[:20]}",
        state=state,
        build_info=SimpleNamespace(dockerfile_content=dockerfile or expected),
    )
    return SimpleNamespace(snapshot=Snapshots(snapshot)), digest


def test_cache_preflight_binds_active_provider_recipe():
    image = "docker.io/library/python:3.12-slim@sha256:" + "a" * 64
    provider, digest = client(image)

    receipt = cache.validate_cache(
        provider, image=image, expected_dockerfile_sha256=digest
    )

    assert receipt["state"] == "passed"
    assert receipt["dockerfile_sha256"] == digest
    assert receipt["snapshot_id"] == "provider-snapshot-id"
    assert provider.snapshot.lookups == [f"cap-verifier-{digest[:20]}"]


def test_cache_preflight_rejects_local_recipe_hash_mismatch_before_lookup():
    image = "docker.io/library/python:3.12-slim@sha256:" + "a" * 64
    provider, _ = client(image)

    with pytest.raises(RuntimeError, match="reviewed migration receipt"):
        cache.validate_cache(provider, image=image, expected_dockerfile_sha256="b" * 64)

    assert provider.snapshot.lookups == []


def test_cache_preflight_rejects_wrong_provider_recipe_and_inactive_cache():
    image = "docker.io/library/python:3.12-slim@sha256:" + "a" * 64
    provider, digest = client(image, dockerfile="FROM wrong")
    with pytest.raises(RuntimeError, match="does not match"):
        cache.validate_cache(provider, image=image, expected_dockerfile_sha256=digest)

    provider, digest = client(image, state="ERROR")
    with pytest.raises(RuntimeError, match="not active"):
        cache.validate_cache(provider, image=image, expected_dockerfile_sha256=digest)
