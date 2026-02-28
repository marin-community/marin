# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cluster.platform.remote_exec import GceRemoteExec


def test_gce_remote_exec_build_cmd_without_user():
    conn = GceRemoteExec(project_id="test-project", zone="us-central2-b", vm_name="vm-a")
    cmd = conn._build_cmd("echo ok")
    assert cmd[:4] == ["gcloud", "compute", "ssh", "vm-a"]


def test_gce_remote_exec_build_cmd_with_user():
    conn = GceRemoteExec(project_id="test-project", zone="us-central2-b", vm_name="vm-a", ssh_user="iris")
    cmd = conn._build_cmd("echo ok")
    assert cmd[:4] == ["gcloud", "compute", "ssh", "iris@vm-a"]
