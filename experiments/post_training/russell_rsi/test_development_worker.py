# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from iris.cluster.runtime.env import WORKDIR_PATH
from marin.execution.lazy import ArtifactStep
from marin.external_dependencies import MARIN_SKYRL
from rigging.runtime_bundle import RuntimeBundle

from experiments.post_training.russell_rsi.launch import development_step


def test_wrong_worker_package_fails_before_runtime_or_model_start(tmp_path):
    installed = tmp_path / "site-packages" / "rolloutengine"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("")
    (installed / "contracts.py").write_text("# Historical installed package without rejected-response contracts.\n")
    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(installed.parent),
            *(str(path) for path in sorted((root / "lib").glob("*/src")) if path.parent.name != "rolloutengine"),
            str(root),
        ]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from experiments.post_training.russell_rsi.rollout_eval import run_development_evaluation; "
            "run_development_evaluation(None)",
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "ValueError: Development worker imported rolloutengine outside the packaged branch" in result.stderr


def test_packaged_worker_records_import_hashes_and_pinned_renderer(tmp_path):
    root = Path(__file__).resolve().parents[3]
    renderer = tmp_path / "skyrl_train" / "inference_engines" / "chat_continuation.py"
    renderer.parent.mkdir(parents=True)
    (renderer.parent / "__init__.py").write_text("")
    (renderer.parent.parent / "__init__.py").write_text("")
    renderer.write_text(
        "def render_exact_chat_continuation():\n    raise RuntimeError('No inference in this fixture')\n"
    )
    metadata = tmp_path / "marinskyrl-0.1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: marinskyrl\nVersion: 0.1.0\n")
    (metadata / "direct_url.json").write_text(
        json.dumps({"url": MARIN_SKYRL.repository, "vcs_info": {"commit_id": MARIN_SKYRL.commit}})
    )
    seed = ArtifactStep.adopt("documents/seed", "2026.10.06.1", "/tmp/seed")
    model = ArtifactStep.adopt("checkpoints/model", "2026.10.06.1", "/tmp/model")
    runtime = RuntimeBundle("/tmp/runtime.json", "0" * 64, "/tmp/runtime.tar.gz", "0" * 64)
    step = development_step(seed, model, "2026.10.06.1", runtime, "parent")
    env = dict(os.environ)
    worker_paths = step.run.env_vars["PYTHONPATH"].replace(WORKDIR_PATH, str(root))
    env["PYTHONPATH"] = os.pathsep.join(
        [
            worker_paths,
            *(
                str(path)
                for path in sorted((root / "lib").glob("*/src"))
                if path.parent.name not in ("rolloutengine", "taskcompendium", "shellbox")
            ),
            str(tmp_path),
            str(tmp_path / "site-packages"),
        ]
    )
    # The worker must select the branch over this historical installed package.
    installed = tmp_path / "site-packages" / "rolloutengine"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("")
    (installed / "contracts.py").write_text("")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys\n"
            "from pathlib import Path\n"
            "from types import SimpleNamespace\n"
            "from rigging.runtime_bundle import RuntimeBundle\n"
            "from experiments.post_training.russell_rsi.rollout_eval import run_development_evaluation\n"
            "root = Path(sys.argv[1])\n"
            "config = SimpleNamespace(output_path=str(root), "
            "runtime_bundle=RuntimeBundle(str(root/'missing.json'),'0'*64,str(root/'missing.tar.gz'),'0'*64))\n"
            "try:\n    run_development_evaluation(config)\n"
            "except FileNotFoundError:\n    pass\n"
            "print((root/'worker-import-provenance.json').read_text())\n",
            str(tmp_path / "worker-output"),
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    proof = json.loads(result.stdout)
    for record in proof["modules"].values():
        assert record["sha256"] == hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
    assert proof["modules"]["skyrl_train.inference_engines.chat_continuation"]["path"] == str(renderer)
    assert proof["branch_root"] == str(root)
