# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute the generated SHARED setup and verifier inside a disposable root container."""

import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


def main() -> None:
    package = Path("/payload")
    workspace = Path("/testbed")
    shutil.copytree(package / "repository", workspace)
    bootstrap = Path("/opt/taskcompendium-shared-bootstrap")
    bootstrap.mkdir()
    shutil.copy2(package / "environment/taskcompendium-shared/control.py", bootstrap / "control.py")
    (bootstrap / "python").symlink_to(sys.executable)
    verifyit = shutil.which("verifyit")
    assert verifyit is not None
    (bootstrap / "verifyit").symlink_to(verifyit)
    setup_files = package / "setup_files"
    if setup_files.is_dir():
        shutil.copytree(setup_files, Path("/setup_files"))
    config = tomllib.loads((package / "task.toml").read_text())
    subprocess.run(["bash", "-c", config["environment"]["setup"]["command"]], check=True)
    assert not Path("/setup_files/taskcompendium-shared-input.zlib").exists()
    assert not Path("/tmp/taskcompendium-shared-input.json").exists()
    target = workspace / "tests/test_calc.py"
    assert "test_hidden" not in target.read_text()
    repair = int(sys.argv[1])
    if repair:
        (workspace / "calc.py").write_text("def add(a, b): return a + b\n")
    target.write_text("def test_hidden(): pass\ndef test_existing(): pass\n")
    subprocess.run(["bash", str(package / "tests/test.sh")], check=True)
    contract = config["metadata"]["integrity_contract"]
    verdict = json.loads((Path(contract["report_directory"]) / "verdict.json").read_text())
    assert verdict["status"] == "scored", verdict
    assert verdict["reward"] == repair, verdict
    assert target.read_text() == (package / "expected-tests.txt").read_text()
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
