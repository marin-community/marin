import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "create_registry_subset.py"


def test_create_registry_subset_writes_seeded_sample(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    tasks = [
        {
            "name": f"task-{index}",
            "git_url": "https://example.com/tasks.git",
            "git_commit_id": "0123456789abcdef",
            "path": f"tasks/{index}",
        }
        for index in range(10)
    ]
    registry_path.write_text(
        json.dumps(
            [
                {
                    "name": "example",
                    "version": "1.0",
                    "description": "Full dataset",
                    "tasks": tasks,
                }
            ]
        )
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--registry",
            str(registry_path),
            "--source-name",
            "example",
            "--source-version",
            "1.0",
            "--target-version",
            "mini-4",
            "--count",
            "4",
            "--seed",
            "42",
            "--description",
            "Seeded subset",
        ],
        check=True,
    )

    datasets = json.loads(registry_path.read_text())
    assert datasets[0]["tasks"] == tasks
    assert datasets[1] == {
        "name": "example",
        "version": "mini-4",
        "description": "Seeded subset",
        "tasks": [tasks[index] for index in (0, 1, 4, 9)],
    }
