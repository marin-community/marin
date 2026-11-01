"""Build four graded training tasks for the Pi verifier-GRPO signal smoke.

These sum tasks use new operands and identifiers, not the disjoint 32-task
teacher-trace evaluation pack. A prior matched evaluation of that task family
had mixed successes; this pack still needs its own independent GPU calibration.
"""

import argparse
import io
import tarfile
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from harbor.models.task.task import Task
from skyrl_train.dataset.harbor_import import harbor_task
from taskcompendium.models import Source

IMAGE = "docker.io/library/python@sha256:e529028263dbe6910a2d96f7d2b8f5266385e917fd45d286ef166977c094a51e"
INPUTS = (
    (29, 31, 21, 12, 51),
    (31, 34, 20, 13, 56),
    (33, 37, 19, 14, 61),
    (35, 40, 18, 15, 66),
)
TASK_CONFIG = f'''version = "1.0"
artifacts = ["/tmp/pi-heldout-answer.txt"]
[environment]
docker_image = "{IMAGE}"
workdir = "/tmp"
allow_internet = true
cpus = 1
memory_mb = 1024
storage_mb = 1024
[verifier]
environment_mode = "separate"
'''


def task_archive(name: str, operands: tuple[int, ...]) -> bytes:
    """Create a real Harbor task with a private verifier, never an assigned reward."""
    instruction = (
        "Use the bash tool to add these five integers, then write only the decimal sum "
        f"to `/tmp/pi-heldout-answer.txt`: {', '.join(map(str, operands))}. Read the answer file back."
    )
    expected = sum(operands)
    test = (
        "#!/bin/sh\nmkdir -p /logs/verifier\n"
        f'if [ "$(cat /tmp/pi-heldout-answer.txt 2>/dev/null)" = "{expected}" ]; then\n'
        "  printf '1\\n' > /logs/verifier/reward.txt\n"
        "else\n  printf '0\\n' > /logs/verifier/reward.txt\nfi\n"
    )
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:") as archive:
        for path, contents, mode in (
            ("task.toml", TASK_CONFIG, 0o644),
            ("instruction.md", instruction, 0o644),
            ("tests/test.sh", test, 0o755),
        ):
            payload = contents.encode()
            item = tarfile.TarInfo(path)
            item.mode = mode
            item.size = len(payload)
            archive.addfile(item, io.BytesIO(payload))
    with tempfile.TemporaryDirectory() as root:
        directory = Path(root) / name
        directory.mkdir()
        with tarfile.open(fileobj=io.BytesIO(stream.getvalue()), mode="r:") as archive:
            archive.extractall(directory, filter="data")
        Task(directory)
        spec = harbor_task(
            directory,
            source=Source(dataset="pi-verifier-grpo-sum", revision="1", row=name, importer_revision="1"),
        )
        assert len(spec.grader.artifacts) == 2
        assert spec.grader.artifacts[0].source == "/tmp/pi-heldout-answer.txt"
    return stream.getvalue()


def build_taskpack() -> bytes:
    """Return a deterministic four-row Harbor task parquet for GPU staging."""
    rows = [
        {"path": f"pi-grpo-sum-{index:02d}", "task_binary": task_archive(f"pi-grpo-sum-{index:02d}", values)}
        for index, values in enumerate(INPUTS, start=8)
    ]
    stream = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), stream, compression="zstd")
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New local task parquet to stage in the compute region")
    args = parser.parse_args()
    with args.output.open("xb") as output:
        output.write(build_taskpack())


if __name__ == "__main__":
    main()
