# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a balanced setup/closure cohort while preserving the previous validation set."""

import base64
import hashlib
import json
import random
import shutil
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.parquet as pq
from rolloutengine.spec import LoweredTaskSpec, TaskSessionSpec
from skyrl_train.dataset.harbor import lower_harbor_task
from skyrl_train.dataset.harbor_import import harbor_trial_task
from taskcompendium.models import Source

from experiments.post_training.smartie.mixture import structural_sample
from experiments.post_training.task_curation.swesmith_staged import stage1_program


def stage1_package(record: dict, directory: Path, image: str) -> None:
    """Package the original readiness inspector for Harbor's post-agent verification."""
    original = LoweredTaskSpec.model_validate(record["lowered_task_spec"])
    contract_resource = next(r for r in original.task.resources.verifier if r.path == "staged-contract.json")
    contract = json.loads(base64.b64decode(contract_resource.source.content_base64))
    if contract["stage"] != "stage1":
        raise ValueError("Only setup tasks belong in this experiment")
    instruction = original.task.context.events[0].content.replace(
        "Finish with a final response when ready.",
        "When ready, submit with a separate bash call: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT.",
    )
    agent_config = Path(__file__).with_name("stage1-agent.yaml").read_bytes()
    agent_sha256 = hashlib.sha256(agent_config).hexdigest()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "tests").mkdir()
    (directory / "environment").mkdir()
    (directory / "agent").mkdir()
    (directory / "agent/setup.yaml").write_bytes(agent_config)
    (directory / "instruction.md").write_text(instruction)
    (directory / "environment/Dockerfile").write_text(f"FROM {image}\nWORKDIR /testbed\n")
    (directory / "task.toml").write_text(
        'version = "1.0"\n[agent]\ntimeout_sec = 1800.0\n'
        "[verifier]\ntimeout_sec = 120.0\n[environment]\n"
        f'docker_image = "{image}"\ncpus = 2\nmemory_mb = 3072\nallow_internet = true\n'
        '\n[metadata.agent_profile]\nschema_version = 1\nconfig_file = "agent/setup.yaml"\n'
        f'config_sha256 = "{agent_sha256}"\n'
        '[metadata.agent_profile.agent]\nname = "mini-swe-agent"\n'
        '[metadata.agent_profile.agent.kwargs]\nversion = "2.1.0"\n'
    )
    (directory / "tests/readiness_check.py").write_text(stage1_program(contract))
    (directory / "tests/test.sh").write_text(
        "#!/bin/bash\nset -eu\nmkdir -p /logs/verifier\ncd /tmp\n"
        "/usr/local/bin/python /tests/readiness_check.py > /logs/verifier/details.json\n"
        "/usr/local/bin/python - <<'PY'\nimport json\nfrom pathlib import Path\n"
        "result = json.loads(Path('/logs/verifier/details.json').read_text())\n"
        "Path('/logs/verifier/reward.txt').write_text('1.0' if result['passed'] else '0.0')\nPY\n"
    )
    (directory / "tests/test.sh").chmod(0o755)


def build(staged: Path, previous: Path, output: Path, image: str, seed: int) -> dict:
    """Write immutable Harbor packages and an identity-audited balanced parquet dataset."""
    if output.exists():
        raise FileExistsError(output)
    if "@sha256:" not in image:
        raise ValueError("Use the published AMD64 image manifest digest")
    records = [json.loads(line) for line in (staged / "train.jsonl").read_text().splitlines()]
    records = [record for record in records if record["stage"] == "stage1"]
    heldout = {json.loads(line)["parent_id"] for line in (staged / "validation.jsonl").read_text().splitlines()}
    parents = [record["parent_id"] for record in records]
    if len(set(parents)) != len(parents) or heldout.intersection(parents):
        raise ValueError("Stage 1 parents overlap or repeat")
    previous_rows = pq.read_table(previous / "train.parquet").to_pylist()
    structural, counts = structural_sample(
        [row for row in previous_rows if row["teacher_route"] == "structural"], len(records), seed
    )
    validation_rows = pq.read_table(previous / "validation.parquet").to_pylist()
    heldout_ids = {json.loads(row["lowered_task_spec"])["task"]["id"] for row in validation_rows}
    if heldout_ids.intersection(parents):
        raise ValueError("Stage 1 training parents overlap previous validation")
    output.mkdir(parents=True)
    rows = list(structural)
    staged_revision = hashlib.sha256((staged / "train.jsonl").read_bytes()).hexdigest()
    session = TaskSessionSpec(
        task_session="harbor_trial",
        max_turns=32,
        model_turn_timeout=600,
        command_timeout=60,
        tool_turn_timeout=None,
        total_turn_timeout=None,
        attempt_timeout=3000,
        verifier_timeout=120,
        cleanup_timeout=120,
        max_total_tokens=None,
    )
    for record in records:
        original = LoweredTaskSpec.model_validate(record["lowered_task_spec"])
        name = original.task.id.rsplit("/", 1)[-1]
        directory = output / "harbor" / name
        stage1_package(record, directory, image)
        source = Source(
            dataset=original.task.source.dataset,
            revision=staged_revision,
            row=original.task.id.split(":", 1)[1],
            importer_revision="smartie-stage1-harbor-v2-agent-profile",
        )
        task = harbor_trial_task(directory, source=source)
        assert task.id == original.task.id
        lowered = lower_harbor_task(directory, task, session=session)
        rows.append({"lowered_task_spec": lowered.model_dump_json(), "teacher_route": "swesmith_stage1"})
    random.Random(seed).shuffle(rows)
    ids = [json.loads(row["lowered_task_spec"])["task"]["id"] for row in rows]
    if len(set(ids)) != len(ids) or heldout_ids.intersection(ids):
        raise ValueError("Mixed training identities repeat or overlap validation")
    pq.write_table(pa.Table.from_pylist(rows), output / "train.parquet")
    shutil.copyfile(previous / "validation.parquet", output / "validation.parquet")
    (output / "README.md").write_text(
        "# SWE-Smith repository setup and structural retention\n\n"
        f"This revision contains {len(records):,} repository-setup tasks and an equal number of structural tasks. "
        "Setup tasks require the specified fork, named branch and commit in `/testbed`, unchanged "
        "tracked source, and an editable installation importable outside the checkout. No bug repair "
        "is requested. The verifier accepts the optional `.git` suffix for the same remote URL; "
        "other forks, detached HEAD and changed tracked files still fail.\n\n"
        "## Source-owned agent profile\n\n"
        "Every Stage 1 Harbor package contains `agent/setup.yaml` and a "
        "`metadata.agent_profile` declaration in `task.toml`. The profile selects Mini SWE Agent "
        "2.1.0 and binds the prompt file by SHA-256. The same files are embedded in each Parquet "
        "task's host-private resources, so the profile survives export and relocation.\n\n"
        "MarinSkyRL's source-agent-profile loader reads these defaults separately for each task. "
        "Use the full Harbor trial session and installed-agent runtime; do not apply the Stage 1 "
        "prompt globally to other sources. Explicit run overrides win and are recorded in trial "
        "results. Missing or corrupted prompt files fail preflight. Other training systems must "
        "read this declaration or pass the bundled YAML through Harbor's agent `kwargs.config_file`. "
        "The YAML overrides Mini SWE Agent's system and instance templates while retaining its bash "
        "tool and submission protocol. Submit using a separate bash call containing only "
        "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.\n\n"
        "Validation rows are copied unchanged from the previous cohort. This is a private release; "
        "publishing task packages or trace content requires a separate redistribution review.\n"
    )
    report = {
        "seed": seed,
        "image": image,
        "train_rows": len(rows),
        "stage1_rows": len(records),
        "structural_rows": len(structural),
        "structural_families": counts,
        "stage1_parent_ids": sorted(parents),
        "ordered_train_ids": ids,
        "validation_rows": len(validation_rows),
        "validation_payload_changed": False,
        "agent_profile": {
            "schema_version": 1,
            "name": "mini-swe-agent",
            "version": "2.1.0",
            "config_sha256": hashlib.sha256(Path(__file__).with_name("stage1-agent.yaml").read_bytes()).hexdigest(),
        },
        "runtime_validation": "pending; local qualification does not certify AMD64 Daytona execution",
        "inputs_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                staged / "train.jsonl",
                staged / "validation.jsonl",
                previous / "train.parquet",
                previous / "validation.parquet",
            )
        },
        "outputs_sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("train.parquet", "validation.parquet")
        },
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


@click.command()
@click.option("--staged", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--previous", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--image", required=True)
@click.option("--seed", type=int, default=42)
def main(staged: Path, previous: Path, output: Path, image: str, seed: int) -> None:
    report = build(staged, previous, output, image, seed)
    click.echo(json.dumps({key: report[key] for key in ("train_rows", "structural_families", "validation_rows")}))


if __name__ == "__main__":
    main()
