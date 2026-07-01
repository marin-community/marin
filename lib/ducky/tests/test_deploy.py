# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from click.testing import CliRunner
from ducky.deploy import cli


def test_deploy_rejects_cluster_and_controller_url_together():
    result = CliRunner().invoke(cli, ["--cluster", "marin", "--controller-url", "http://x"])
    assert result.exit_code != 0
    assert "not both" in result.output


def test_deploy_requires_a_target(monkeypatch):
    monkeypatch.delenv("IRIS_CONTROLLER_URL", raising=False)
    result = CliRunner().invoke(cli, [])
    assert result.exit_code != 0
    assert "--cluster" in result.output
