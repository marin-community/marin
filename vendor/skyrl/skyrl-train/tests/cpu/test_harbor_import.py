"""Restore native Harbor packages without exposing private solutions to the actor."""

from taskcompendium.models import Source

from skyrl_train.dataset.harbor_import import harbor_trial_task, restore_harbor_trial_task


def test_original_trial_restores_private_oracle_files_with_native_package(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    contents = {
        "instruction.md": b"Repair the workspace.",
        "task.toml": b'[environment]\ndocker_image = "fixture/image:latest"\n',
        "tests/test.sh": b"#!/bin/sh\ncheck-workspace\n",
        "solution/solve.sh": b"#!/bin/sh\nrepair-workspace\n",
    }
    for name, data in contents.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o750)
    task = harbor_trial_task(package, source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"))
    task = task.model_copy(
        update={
            "resources": task.resources.model_copy(
                update={
                    "oracle": tuple(
                        resource for resource in task.resources.verifier if resource.path.startswith("solution/")
                    ),
                    "verifier": tuple(
                        resource for resource in task.resources.verifier if not resource.path.startswith("solution/")
                    ),
                }
            )
        }
    )
    restored = tmp_path / "restored"
    restore_harbor_trial_task(task, restored)
    assert {
        path.relative_to(restored).as_posix(): path.read_bytes()
        for path in restored.rglob("*") if path.is_file()
    } == contents
    assert (restored / "solution/solve.sh").stat().st_mode & 0o777 == 0o750
    assert (restored / "solution/solve.sh").stat().st_mtime_ns == (package / "solution/solve.sh").stat().st_mtime_ns
