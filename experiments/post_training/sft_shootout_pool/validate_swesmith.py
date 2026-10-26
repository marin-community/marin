# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate fixed SWE-smith packages with local Docker, following Harbor's phase order.

For each package: build environment/ into an image; in a fresh container run the leak probe as the agent would
(no network, before tests/ or solution/ exist); then in two more fresh containers copy tests/ to /tests and run
tests/test.sh, once after the oracle (solution/solve.sh) and once on the untouched tree. Pass means: the probe finds
one commit, one branch, no remote, a one-entry reflog and no other commit to diff against; the oracle run writes
reward 1; the unpatched run writes reward 0.

Usage:
    python -m experiments.post_training.sft_shootout_pool.validate_swesmith <fixed tasks.parquet> <id> [<id> ...]
"""

import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

PROBE = r"""
cd /testbed
echo "commits=$(git rev-list --all | wc -l)"
echo "branches=$(git branch -a | wc -l)"
echo "remotes=$(git remote | wc -l)"
echo "reflog=$(git reflog | wc -l)"
echo "stashes=$(git stash list | wc -l)"
for ref in HEAD~1 HEAD^ main~1 origin/main master; do
  git rev-parse -q --verify "$ref" >/dev/null && echo "resolvable=$ref"
done
echo "packs=$(find .git/objects -type f | wc -l)"
ls -d /tests /solution 2>/dev/null | sed 's/^/visible=/'
true
"""


LEAK_COMMANDS = (
    "git log --all -p",
    "git show HEAD~1",
    "git diff HEAD main",
    "git diff HEAD~1",
    "git fsck --unreachable --no-reflogs --lost-found",
    "git stash list -p",
    "for f in $(git ls-files); do git show main:$f; done",
)


def leak_probe(image: str, clean_lines: list[str]) -> dict:
    """Run the history commands an agent would try, then search their output and the filesystem for clean lines."""
    script = "cd /testbed\n" + "\n".join(f"({command}) 2>/dev/null" for command in LEAK_COMMANDS)
    output = docker("run", "--rm", "--network", "none", image, "bash", "-c", script, check=False)
    in_commands = [line for line in clean_lines if line in output]
    needles = "\n".join(clean_lines)
    found = docker(
        "run",
        "--rm",
        "--network",
        "none",
        "-i",
        image,
        "bash",
        "-c",
        "cat > /tmp/needles && grep -rIlF -f /tmp/needles / --exclude-dir=proc --exclude-dir=sys "
        "--exclude-dir=dev --exclude=needles 2>/dev/null | head -20",
        stdin=needles.encode(),
        check=False,
    )
    return {
        "clean_lines": len(clean_lines),
        "in_command_output": len(in_commands),
        "files_with_clean_lines": found.split(),
    }


def docker(*args: str, stdin: bytes | None = None, check: bool = True) -> str:
    result = subprocess.run(["docker", *args], input=stdin, capture_output=True, check=False)
    if check and result.returncode:
        raise RuntimeError(f"docker {args[0]} failed: {result.stderr.decode()[-2000:]}")
    return result.stdout.decode(errors="replace")


def run_phase(image: str, package: Path, *, oracle: bool) -> str:
    """Return the reward the verifier writes, in a fresh container (network on, as Harbor allows for tests)."""
    name = docker("run", "-d", image, "sleep", "infinity").strip()
    try:
        docker("exec", name, "mkdir", "-p", "/logs/verifier")
        if oracle:
            docker("cp", str(package / "solution"), f"{name}:/solution")
            docker("exec", name, "bash", "/solution/solve.sh", check=False)
        docker("cp", str(package / "tests"), f"{name}:/tests")
        docker("exec", "-w", "/testbed", name, "bash", "/tests/test.sh", check=False)
        return docker("exec", name, "cat", "/logs/verifier/reward.txt", check=False).strip() or "missing"
    finally:
        docker("rm", "-f", name, check=False)


def validate(task: str, binary: bytes, root: Path) -> dict:
    package = root / task
    with tarfile.open(fileobj=io.BytesIO(binary), mode="r:*") as archive:
        archive.extractall(package, filter="data")
    image = f"pool-swesmith-check:{task}"
    docker("build", "-q", "-t", image, str(package / "environment"))
    try:
        probe = docker("run", "--rm", "--network", "none", image, "bash", "-c", PROBE)
        facts = dict(line.split("=", 1) for line in probe.split() if "=" in line)
        clean = (
            facts.get("commits") == "1"
            and facts.get("branches") == "1"
            and facts.get("remotes") == "0"
            and facts.get("stashes") == "0"
            and int(facts.get("reflog", "9")) <= 1
            and "resolvable" not in facts
            and "visible" not in facts
        )
        oracle = run_phase(image, package, oracle=True)
        unpatched = run_phase(image, package, oracle=False)
    finally:
        docker("rmi", "-f", image, check=False)
    return {
        "task": task,
        "probe": probe.strip().splitlines(),
        "history_clean": clean,
        "oracle_reward": oracle,
        "unpatched_reward": unpatched,
        "ok": clean and oracle == "1" and unpatched == "0",
    }


def main() -> None:
    table = pq.read_table(sys.argv[1]).to_pylist()
    wanted = set(sys.argv[2:])
    with tempfile.TemporaryDirectory() as temporary:
        for row in table:
            if row["path"] in wanted:
                print("VALIDATED " + json.dumps(validate(row["path"], row["task_binary"], Path(temporary))), flush=True)


if __name__ == "__main__":
    main()
