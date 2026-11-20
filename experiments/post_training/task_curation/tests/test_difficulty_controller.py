# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller credentials reach Iris through versioned Secret Manager references."""

import json
import sys
from types import SimpleNamespace

import google.cloud
from rigging.secrets import resolve_secret_spec

from experiments.post_training.task_curation.difficulty import controller


def test_controller_resolves_versioned_credentials_before_external_submission(tmp_path, monkeypatch):
    requested = []

    class SecretManager:
        def access_secret_version(self, *, name):
            requested.append(name)
            return SimpleNamespace(payload=SimpleNamespace(data=b"fixture-value"))

    sdk = SimpleNamespace(SecretManagerServiceClient=SecretManager)
    monkeypatch.setitem(sys.modules, "google.cloud.secretmanager", sdk)
    monkeypatch.setattr(google.cloud, "secretmanager", sdk, raising=False)
    monkeypatch.setattr(controller, "list_iris_jobs", lambda: [])
    monkeypatch.setattr(controller, "load_credentials", lambda profile: SimpleNamespace(edge_refresh_token="fixture"))
    accepted = tmp_path / "external-submission.json"

    def scheduler_submit(command, *, check, env):
        for index, value in enumerate(command):
            if value == "--secret" and command[index + 2].startswith("gcp-secret://"):
                resolve_secret_spec(command[index + 2])
        accepted.write_text(json.dumps({"job_name": command[command.index("--job-name") + 1]}))

    monkeypatch.setattr(controller.subprocess, "run", scheduler_submit)
    controller.launch_controller("ledger", "applet", False, frozenset(), "generation", 800)
    assert requested == [
        "projects/hai-gcp-models/secrets/DAYTONA_EVAL_API_KEY/versions/1",
        "projects/hai-gcp-models/secrets/DAYTONA_RL_API_KEY/versions/1",
    ]
    assert json.loads(accepted.read_text())["job_name"].startswith("atlas-difficulty-reconciler-controller-")
