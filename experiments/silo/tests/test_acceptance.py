# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""silo.acceptance must run to completion against any live stack.

Against the in-process stack some verdicts fail by nature -- this machine has a
network and no container cgroups -- so the test asserts the run completes and the
structural checks pass, which is what catches a crash before a cluster round trip.
"""

import json

from silo import acceptance
from test_end_to_end import stack  # noqa: F401 - fixture

DIGEST = "docker.io/library/alpine@sha256:" + "a" * 64


def test_acceptance_completes_and_writes_a_receipt(stack, monkeypatch, tmp_path, capsys):  # noqa: F811
    monkeypatch.setenv("SILO_API_TOKEN", "api-token")
    monkeypatch.setenv("SILO_BROKER_URL", stack.broker_server.url)
    monkeypatch.setenv("IRIS_OUTPUT_DIR", str(tmp_path))
    code = acceptance.main(["--image", DIGEST, "--runtimes", "runc", "--fanout", "3"])
    assert code in (0, 1)

    receipt = json.loads((tmp_path / "silo-acceptance.json").read_text())
    checks = receipt["checks"]
    for structural in (
        "broker_reachable_and_hosts_live",
        "snapshot_active_and_identity_verified[cap-harbor]",
        "snapshot_active_and_identity_verified[cap-verifier]",
        "create[runc/4cpu]",
        "network_block_all_attested[runc/4cpu]",
        "exit_code_exact_one_shot[runc/4cpu]",
        "exit_code_and_stderr_exact_session[runc/4cpu]",
        "fs_roundtrip_4mib[runc/4cpu]",
        "deletion_observed_async[runc/4cpu]",
        "concurrent_ids_unique",
    ):
        assert checks.get(structural) is True, (structural, checks)
    # Every trial ran to its end rather than dying partway.
    assert not any(name.startswith("trial_completed") for name in checks), checks
    assert "=== SILO ACCEPTANCE RECEIPT ===" in capsys.readouterr().out
