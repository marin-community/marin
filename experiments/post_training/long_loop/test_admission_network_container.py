# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise simultaneous Docker controls without host network or firewall changes."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from experiments.post_training.long_loop import admission_network

PROGRAM = """
import contextlib, json, os, pathlib, subprocess, time
from admission_network import private_network
daemon = subprocess.Popen([
    'dockerd', '--bridge=none', '--iptables=false', '--ip-forward=false', '--ip-masq=false',
], stdout=open('/tmp/dockerd.log', 'w'), stderr=subprocess.STDOUT)
containers = []
try:
    deadline = time.monotonic() + 60
    while subprocess.run(['docker', 'info'], capture_output=True).returncode:
        assert daemon.poll() is None and time.monotonic() < deadline
        time.sleep(0.1)
    before = subprocess.check_output(['iptables-save'])
    subprocess.run(['docker', 'pull', 'python:3.12-slim-bookworm'], check=True)
    with contextlib.ExitStack() as stack:
        for index in range(2):
            container = subprocess.check_output([
                'docker', 'run', '-d', '--network', 'none', 'python:3.12-slim-bookworm', 'sleep', 'infinity'
            ], text=True).strip()
            containers.append(container)
            stack.enter_context(private_network(container, pathlib.Path(f'/tmp/network-{index}.log')))
            subprocess.run(['docker', 'exec', '-d', container, 'python', '-m', 'http.server', '18765'], check=True)
        for container in containers:
            subprocess.run(['docker', 'exec', container, 'python', '-c',
                "import socket,urllib.request; socket.getaddrinfo('github.com',443); "
                "assert urllib.request.urlopen('http://127.0.0.1:18765').status == 200"], check=True)
        assert subprocess.check_output(['iptables-save']) == before
        print(json.dumps({'independent_servers': len(containers), 'firewall_unchanged': True}))
finally:
    for container in containers:
        subprocess.run(['docker', 'rm', '-f', container], check=False)
    daemon.terminate()
    daemon.wait(timeout=20)
"""


@pytest.mark.docker
@pytest.mark.timeout(600)
def test_simultaneous_controls_keep_network_and_firewall_isolated(tmp_path):
    shutil.copyfile(Path(admission_network.__file__), tmp_path / "admission_network.py")
    (tmp_path / "check.py").write_text(PROGRAM)
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.12-slim-bookworm\n"
        "RUN apt-get update && apt-get install -y docker.io slirp4netns iptables "
        "&& rm -rf /var/lib/apt/lists/*\n"
        'COPY admission_network.py check.py /check/\nCMD ["python","/check/check.py"]\n'
    )
    image = subprocess.check_output(["docker", "build", "--quiet", str(tmp_path)], text=True, timeout=450).strip()
    result = subprocess.run(
        ["docker", "run", "--rm", "--privileged", image],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    receipt = json.loads(next(line for line in result.stdout.splitlines() if line.startswith("{")))
    assert receipt == {"independent_servers": 2, "firewall_unchanged": True}
