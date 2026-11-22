# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml
from marin.rl.skyrl import SkyRLSpec, skyrl_cpu_command
from marin.rl.skyrl import _run_launcher as run_launcher_for_test

from experiments.post_training import rl_compose
from tests.rl.launch_document_census import generation_geometry_step, render_launch_census, runtime_launch


def _render_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    monkeypatch.setattr(rl_compose, "SELECTION_ROOT", str(tmp_path / "selections"))
    monkeypatch.setattr(rl_compose, "artifact_prefix", lambda _cluster: str(tmp_path / "artifacts"))
    producers = set()
    validate = SkyRLSpec.__post_init__

    def record_producer(spec):
        module = inspect.stack()[2].frame.f_globals["__name__"]
        producers.add(module.replace(".", "/") + ".py")
        validate(spec)

    monkeypatch.setattr(SkyRLSpec, "__post_init__", record_producer)
    # PyYAML writes these hex digits unquoted, and OmegaConf reads them as a float.
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(hex="104e63883176".ljust(32, "0")))
    documents, failures = render_launch_census(tmp_path / "artifacts")
    assert not failures, failures
    with contextmanager(runtime_launch)(tmp_path) as render:
        documents["native/generation"] = render(generation_geometry_step()).launch_config_yaml
    root = Path(__file__).resolve().parents[2]
    authored = {
        str(path.relative_to(root)) for path in (root / "experiments").rglob("*.py") if "SkyRLSpec(" in path.read_text()
    }
    # The Snowball preset goldens and override test own the shared runner's documents.
    separate_owners = {"experiments/post_training/rl_compose.py"}
    assert authored <= producers | separate_owners, authored - producers - separate_owners
    return documents


def test_rl_launch_documents_render_from_authored_specs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    generation = yaml.safe_load(_render_documents(tmp_path, monkeypatch)["native/generation"])
    assert generation["iris"]["allocation"]["num_nodes"] == 1
    assert generation["skyrl"]["generator"]["num_inference_engines"] == 4
    assert generation["skyrl"]["entrypoint"] == "generate"


# A cold SkyRL environment sync and the full batch of compositions share this deadline.
@pytest.mark.slow
@pytest.mark.timeout(540)
def test_rl_launch_documents_load_with_the_vendored_launcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    documents = _render_documents(tmp_path, monkeypatch)
    source = tmp_path / "source.json"
    source.write_text(json.dumps(documents))
    output = tmp_path / "loaded.json"
    result = run_launcher_for_test(
        skyrl_cpu_command("python", str(Path(__file__).with_name("launch_document_loader.py")), str(source), str(output))
    )
    assert result.returncode == 0, result.stderr
    failures = json.loads(output.read_text())
    assert not failures, failures
