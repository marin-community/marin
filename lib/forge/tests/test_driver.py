# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Never

import httpx
import pytest
from iris.cluster.types import JobName
from shellbox.backends.iris.machine import IrisMachineFactory

from forge import wiring
from forge.driver import inputs, proposal_key
from forge.policy import load_policy


def test_manifest_selects_ranks_and_path_list_preserves_output_identity(tmp_path: Path) -> None:
    paths = tuple(f"s3://bucket/run/d02.swe.repair/slot-{slot}.proposal.json" for slot in (15, 8, 24))
    sample = tmp_path / "sample.json"
    sample.write_text(
        json.dumps(
            {
                "format": "acceptance-measure-sample-v1",
                "sample": [
                    {"rank": rank, "proposal": path, "capability": "d02.swe.repair", "slot": slot}
                    for rank, path, slot in zip((0, 1, 2), paths, (15, 8, 24), strict=True)
                ],
            }
        )
    )
    assert inputs(str(sample), (0, 2)) == (paths[0], paths[2])
    assert str(proposal_key(paths[0])) == "d02.swe.repair/15"
    manifest = tmp_path / "proposals.txt"
    manifest.write_text("\n".join(paths) + "\n")
    assert inputs(str(manifest)) == paths
    with pytest.raises(ValueError, match="missing"):
        inputs(str(sample), (3,))
    manifest.write_text(f"{paths[0]}\n{paths[0]}\n")
    with pytest.raises(ValueError, match="duplicate output"):
        inputs(str(manifest))


def test_empty_prefix_fails_instead_of_reporting_a_successful_empty_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No proposal"):
        inputs(str(tmp_path))


async def test_iris_runtime_on_linux_never_constructs_local_backend_or_probes_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("Runtime initialization must not probe a sandbox or transport")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("IRIS_CONTROLLER_URL", raising=False)
    monkeypatch.setattr(wiring, "get_job_info", lambda: SimpleNamespace(controller_address="http://controller"))
    monkeypatch.setattr(wiring, "local_factory", unexpected)
    monkeypatch.setattr(subprocess, "Popen", unexpected)
    monkeypatch.setattr(httpx.AsyncClient, "send", unexpected)
    policy = load_policy(Path(__file__).parents[1] / "src/forge/policy/defaults").overlaid({"runtime.host": "iris"})
    source = wiring.endpoint_source("http://fake/v1", None, "fake", "glm-5.3")
    async with wiring.runtime(policy, "S", source, "fake") as (_, resources):
        assert set(resources.factories) == {"gvisor", "shellsim"}
        factory = resources.factories["gvisor"]
        assert isinstance(factory, IrisMachineFactory)
        assert factory.controller_url == "http://controller"
        assert factory.cluster is None
        assert factory.scheduling_timeout == 600


def test_relay_reresolves_by_job_and_model_without_exposing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    class Resolver:
        def __init__(self) -> None:
            self.url = "http://first:8000"
            self.models: list[str] = []

        def resolve(self, name: str) -> SimpleNamespace:
            self.models.append(name)
            return SimpleNamespace(endpoints=(SimpleNamespace(url=self.url),))

    resolver = Resolver()
    jobs: list[str] = []

    def for_job(job: JobName) -> Resolver:
        jobs.append(str(job))
        return resolver

    monkeypatch.setattr(wiring, "iris_ctx", lambda: SimpleNamespace(client=SimpleNamespace(resolver_for_job=for_job)))
    source = wiring.endpoint_source(None, "/muchanem/relay", "secret-token", "glm-5.3")
    assert source.current().base_url == "http://first:8000/v1"
    resolver.url = "http://replacement:8000/v1/"
    assert source.current().base_url == "http://replacement:8000/v1"
    assert jobs == ["/muchanem/relay", "/muchanem/relay"]
    assert resolver.models == ["glm-5.3", "glm-5.3"]
    assert "secret-token" not in repr(source)
    assert (
        wiring.endpoint_source("http://fixed/v1", None, "secret-token", "glm-5.3").current().base_url
        == "http://fixed/v1"
    )
    with pytest.raises(ValueError, match="exactly one"):
        wiring.endpoint_source("http://fixed/v1", "/muchanem/relay", "secret-token", "glm-5.3")
    with pytest.raises(ValueError, match="exactly one"):
        wiring.endpoint_source(None, None, "secret-token", "glm-5.3")
