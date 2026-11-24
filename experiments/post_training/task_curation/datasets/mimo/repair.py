# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repair the pinned MiMo population: pin agent images and declare the source's own grader.

The published population carries an unresolved Docker build context on every task and a
``NoGrader`` placeholder that the rollout engine can never grade. This module rewrites the
pinned retained release at the population level: each task's agent environment is replaced
by its digest-pinned prepared image (built by ``images.py`` from the task's own Dockerfile),
and the grader is declared from the source's own grading code (the native verifier package:
``tests/setup.sh`` resets the touched paths to the captured base ref, ``tests/test.sh`` runs
the published test command and writes the reward file). Everything else about a task stays
unchanged, and the per-task ledger records exactly which fields changed.

Faithfulness notes
------------------

- The healthcheck command in each task's ``task.toml`` captures the workspace's base commit
  into ``/var/lib/mimo/base_ref`` before the agent starts (upstream: "capture the base ref
  before the agent starts"). The native Harbor trial ran it at startup; here it is the task's
  one agent setup command so the grading reset has the same baseline.
- The verifier pipeline runs on the agent machine exactly as upstream: ``setup.sh`` then
  ``test.sh``, with the upstream reward normalized into the numeric reward file the runtime
  parses. A task whose verifier produces no reward file fails grading (fail closed), never
  passes.
- The private verifier files stay private: they are delivered at grading time from the
  grader's own collect commands, and the original bytes remain archived under
  ``source_archive/`` for provenance.
"""

import argparse
import datetime
import hashlib
import json
import re
import shutil
import subprocess
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.harbor.root_delivery import TEST_EXIT, TEST_OUTPUT, native_file_reward_grader
from taskcompendium.identity import canonical_sha256
from taskcompendium.integrity import DeliveredVerifierFile, RootDeliveryContract, SharedDependencyState
from taskcompendium.models import TaskSpec, VerifierCommand, require_resolved_environment
from taskcompendium.runtime.resources import resource_bytes

REPAIR_VERSION = "2026.11.23.pinned-images.1"

BASE_IMAGE_PATTERN = re.compile(r"^FROM docker\.io/xiaomimimo/mimo-v2\.6-rl-oss@(sha256:[0-9a-f]{64})$", re.MULTILINE)
WORKDIR_PATTERN = re.compile(r"^WORKDIR (/[A-Za-z0-9._/-]+)$", re.MULTILINE)
RUN_PATTERN = re.compile(r"^RUN (.+)$", re.MULTILINE)

SETUP_OUTPUT = "/logs/verifier/setup-output.txt"
REWARD_JSON = "/logs/verifier/reward.json"
REWARD_TXT = "/logs/verifier/reward.txt"

TESTS_PREFIX = "tests/"
ARCHIVE_PREFIX = "source_archive/"

EXPECTED_VERIFIER_FILES = (
    "environment/Dockerfile",
    "instruction.md",
    "task.toml",
    "tests/reset_test_files.sh",
    "tests/setup.sh",
    "tests/test.patch",
    "tests/test.sh",
    "tests/touched_paths.txt",
)

UNCHANGED_FIELDS = (
    "id",
    "source",
    "context",
    "answer_type",
    "answer_format",
    "execution_requirements",
    "verifier_execution_requirements",
    "tool_bindings",
    "output_paths",
    "reasoning_markup",
    "schema_version",
    "final_tools",
    "interaction_tools",
)

# The upstream test command writes exactly {"reward": <number>}; one sed extraction covers it
# and any other shape fails closed through the empty-reward check below.
REWARD_EXTRACTION = (
    r"reward=$(sed -n 's/.*\"reward\"[[:space:]]*:[[:space:]]*\(-\{0,1\}[0-9][0-9.eE+-]*\).*/\1/p' "
    + REWARD_JSON
    + r" | head -n 1)"
)


class RepairRejected(ValueError):
    """A task the repair refuses to rewrite instead of guessing."""


@dataclass(frozen=True)
class ImageBinding:
    """One task's prepared agent image, as recorded by the image-preparation lane."""

    task_id: str
    image: str
    digest: str
    base_digest: str
    recipe_sha256: str


@dataclass(frozen=True)
class Recipe:
    """The measured shape of one task's Dockerfile."""

    base_digest: str
    workdir: str
    run_commands: tuple[str, ...]


def parse_recipe(dockerfile: str) -> Recipe:
    """Measure the task's Dockerfile; anything outside the known shape fails the repair."""
    base = BASE_IMAGE_PATTERN.search(dockerfile)
    if base is None:
        raise RepairRejected("the Dockerfile does not pin the expected upstream base image")
    workdirs = WORKDIR_PATTERN.findall(dockerfile)
    if len(workdirs) != 1:
        raise RepairRejected("the Dockerfile must declare exactly one WORKDIR")
    commands = RUN_PATTERN.findall(dockerfile)
    if len(commands) != 2:
        raise RepairRejected("the Dockerfile must declare exactly two RUN steps")
    rest = RUN_PATTERN.sub("", WORKDIR_PATTERN.sub("", BASE_IMAGE_PATTERN.sub("", dockerfile)))
    leftovers = [
        line
        for line in rest.splitlines()
        if line.strip() and not line.strip().startswith("#") and not line.strip().startswith("RUN ")
    ]
    # The only measured noise is the generated ot-validation marker comment.
    noise = all(line.strip().startswith("# ot-validation:") or not line.strip() for line in rest.splitlines())
    if leftovers or not noise:
        raise RepairRejected("the Dockerfile carries unmeasured directives")
    return Recipe(base.group(1), workdirs[0], tuple(commands))


def verifier_reward_execute() -> VerifierCommand:
    """Run the source's verifier pipeline and normalize its reward, failing closed.

    ``setup.sh`` resets the touched paths to the captured base ref and applies the test
    patch; upstream scores an unresettable or unpatchable testbed as reward 0 through its
    own ``reward.txt`` write, which is retained. ``test.sh`` runs the published test
    command, writes ``reward.json``, and resets the touched paths again. The numeric reward
    file the runtime parses is derived from that upstream output; a verifier that produces
    neither fails the grading command.
    """
    script = "\n".join(
        (
            "set -eu",
            "mkdir -p /logs/verifier",
            f": > {TEST_OUTPUT}",
            f"rm -f {REWARD_TXT} {REWARD_JSON}",
            "status=0",
            "if [ -f /tests/setup.sh ]; then",
            f"  bash /tests/setup.sh > {SETUP_OUTPUT} 2>&1 || status=$?",
            "fi",
            'if [ "$status" -ne 0 ]; then',
            f'  if test ! -e {REWARD_JSON} && test -f {REWARD_TXT} && test "$(cat {REWARD_TXT})" = 0; then',
            f"    printf '%s\\n' \"$status\" > {TEST_EXIT}",
            "    exit 0",
            "  fi",
            f"  printf '%s\\n' \"$status\" > {TEST_EXIT}",
            '  exit "$status"',
            "fi",
            "status=0",
            f"bash /tests/test.sh > {TEST_OUTPUT} 2>&1 || status=$?",
            f"printf '%s\\n' \"$status\" > {TEST_EXIT}",
            f"if [ -f {REWARD_JSON} ]; then",
            REWARD_EXTRACTION,
            '  if [ -n "$reward" ]; then',
            f"    printf '%s\\n' \"$reward\" > {REWARD_TXT}",
            "  fi",
            "fi",
            f"if test ! -s {REWARD_TXT}; then",
            '  echo "grading failed closed: the source verifier produced no reward file" >&2',
            "  exit 7",
            "fi",
            "exit 0",
        )
    )
    return VerifierCommand(argv=("bash", "-c", script), cwd="/")


def healthcheck_setup_command(config: dict) -> str:
    """The startup base-ref capture the native Harbor trial ran as its healthcheck."""
    command = (config.get("environment", {}).get("healthcheck", {}) or {}).get("command")
    if not isinstance(command, str) or not command.strip():
        raise RepairRejected("the native package declares no healthcheck to capture the base ref")
    return command


def verifier_timeout(config: dict) -> float:
    timeout = config.get("verifier", {}).get("timeout_sec")
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise RepairRejected("the native package declares no verifier timeout")
    return float(timeout)


@dataclass(frozen=True)
class TaskExclusion:
    """One task the repaired variant omits, with its typed reason and evidence."""

    task_id: str
    reason: str
    evidence: str


CONVERSION_RECEIPT_PROTOCOL = "mimo-taskspec-conversion-v1"
CONVERTER_NAME = "experiments.post_training.task_curation.datasets.mimo.repair"


def converter_revision() -> str:
    """The landing revision the converter runs from, read from its own checkout."""
    root = Path(__file__).resolve().parents[4]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RepairRejected("the converter cannot read its own checkout revision")
    return result.stdout.strip()


def exclusions_from_document(document: dict) -> dict[str, TaskExclusion]:
    """Validate the repair lane's typed per-task exclusion document."""
    exclusions = {}
    for entry in document["tasks"]:
        exclusion = TaskExclusion(
            task_id=entry["task_id"],
            reason=entry["reason"],
            evidence=entry["evidence"],
        )
        if exclusion.task_id in exclusions:
            raise RepairRejected(f"the exclusion document repeats task {exclusion.task_id}")
        if not exclusion.reason or not exclusion.evidence:
            raise RepairRejected(f"the exclusion for {exclusion.task_id} lacks its typed reason or evidence")
        exclusions[exclusion.task_id] = exclusion
    return exclusions


def private_verifier_files(resources: tuple) -> tuple:
    """The private tests/* resources, renamed relative to their /tests restore root."""
    private = []
    for resource in resources:
        if not resource.path.startswith(TESTS_PREFIX):
            continue
        name = resource.path.removeprefix(TESTS_PREFIX)
        if not name or "/" in name:
            raise RepairRejected(f"the verifier file {resource.path} does not sit directly under tests/")
        private.append(resource.model_copy(update={"path": name}))
    if not private:
        raise RepairRejected("the native package carries no private verifier files")
    return tuple(private)


def repair_task(row: dict, binding: ImageBinding) -> tuple[TaskSpec, dict]:
    """Rewrite one retained task and return the new spec with its change ledger."""
    task = TaskSpec.model_validate_json(row["task_spec"])
    if task.id != binding.task_id or task.id != row["id"]:
        raise RepairRejected(f"the image binding does not belong to task {row['id']}")
    environment = task.environment_requirements
    if environment.docker_image is not None or environment.docker_build is None:
        raise RepairRejected("the task does not carry the expected unresolved Docker build context")
    dockerfile = resource_bytes(environment.docker_build.files[0])
    recipe = parse_recipe(dockerfile.decode())
    recipe_sha = hashlib.sha256(dockerfile).hexdigest()
    if recipe.base_digest != binding.base_digest:
        raise RepairRejected("the pinned image was not built from the task's own base digest")
    if recipe_sha != binding.recipe_sha256:
        raise RepairRejected("the pinned image was not built from the task's own Dockerfile bytes")

    resources = task.resources.verifier
    by_path = {resource.path: resource for resource in resources}
    missing = [name for name in EXPECTED_VERIFIER_FILES if name not in by_path]
    if missing:
        raise RepairRejected(f"the native package is missing verifier files: {missing}")
    package = tomllib.loads(resource_bytes(by_path["task.toml"]).decode())
    setup = healthcheck_setup_command(package)
    timeout = verifier_timeout(package)
    private = private_verifier_files(resources)

    pinned_environment = environment.model_copy(
        update={
            "docker_image": binding.image,
            "docker_build": None,
            "working_directory": recipe.workdir,
            "setup_commands": (setup,),
        }
    )
    grader = native_file_reward_grader(pinned_environment, private, verifier_reward_execute(), timeout)
    contract = RootDeliveryContract(
        dependency_state=SharedDependencyState.AGENT_MUTABLE,
        agent_user=None,
        undeclared_deadline=False,
        verifier_files=tuple(
            DeliveredVerifierFile(
                name=resource.path,
                mode=int(resource.mode, 8),
                sha256=hashlib.sha256(resource_bytes(resource)).hexdigest(),
            )
            for resource in sorted(private, key=lambda item: item.path)
        ),
    )
    repaired = task.model_copy(
        update={
            "environment_requirements": pinned_environment,
            "grader": grader,
            "integrity_contract": contract,
            "resources": task.resources.model_copy(
                update={
                    "worker": (),
                    "verifier": tuple(
                        resource.model_copy(update={"path": ARCHIVE_PREFIX + resource.path}) for resource in resources
                    ),
                }
            ),
            "tags": tuple(dict.fromkeys((*task.tags, "root-delivery", "prepared-images"))),
            "labels": {
                **task.labels,
                "mimo_repair_version": REPAIR_VERSION,
                "mimo_base_digest": recipe.base_digest,
                "mimo_pinned_image": binding.image,
                "mimo_pinned_digest": binding.digest,
                "mimo_recipe_sha256": recipe_sha,
            },
        }
    )
    require_resolved_environment(repaired.environment_requirements)
    require_resolved_environment(repaired.grader.environment)
    return repaired, ledger_entry(task, repaired, recipe, binding, recipe_sha)


def ledger_entry(original: TaskSpec, repaired: TaskSpec, recipe: Recipe, binding: ImageBinding, recipe_sha: str) -> dict:
    """The exact per-task change record the release evidence binds."""
    before = original.model_dump(mode="json")
    after = repaired.model_dump(mode="json")
    changed = sorted(key for key in after if before.get(key) != after.get(key))
    unexpected = sorted(set(UNCHANGED_FIELDS) & set(changed))
    if unexpected:
        raise RepairRejected(f"the repair changed unmeasured task fields: {unexpected}")
    return {
        "task_id": original.id,
        "base_digest": recipe.base_digest,
        "workdir": recipe.workdir,
        "recipe_sha256": recipe_sha,
        "pinned_image": binding.image,
        "pinned_digest": binding.digest,
        "changed_fields": changed,
    }


def load_bindings(path: Path) -> dict[str, ImageBinding]:
    """Read the image-preparation lane's binding document, keyed by task id."""
    return bindings_from_document(json.loads(path.read_text()))


def bindings_from_document(document: dict) -> dict[str, ImageBinding]:
    """Validate the image-preparation lane's binding document, keyed by task id."""
    bindings = {}
    for entry in document["images"]:
        binding = ImageBinding(
            task_id=entry["task_id"],
            image=entry["image"],
            digest=entry["digest"],
            base_digest=entry["base_digest"],
            recipe_sha256=entry["recipe_sha256"],
        )
        if binding.task_id in bindings:
            raise RepairRejected(f"the image binding repeats task {binding.task_id}")
        repository, _, suffix = binding.image.partition("@")
        if (
            not repository.startswith("ghcr.io/marin-community/")
            or not binding.digest.startswith("sha256:")
            or suffix != binding.digest
        ):
            raise RepairRejected(f"the pinned image reference is not digest-pinned for task {binding.task_id}")
        bindings[binding.task_id] = binding
    return bindings


def repair_population(
    input_parquet: Path,
    bindings: dict[str, ImageBinding],
    output_dir: Path,
    exclusions: dict[str, TaskExclusion] | None = None,
    revision: str | None = None,
) -> dict:
    """Rewrite the whole retained population and write the repaired release artifacts."""
    schema = pq.ParquetFile(input_parquet).schema_arrow
    revision = revision or converter_revision()
    exclusions = exclusions or {}
    ledger = []
    conversions = []
    excluded = []
    seen = set()
    input_hashes = {}
    rows = []
    for batch in pq.ParquetFile(input_parquet).iter_batches(batch_size=16):
        for row in batch.to_pylist():
            task_id = row["id"]
            if task_id in seen:
                raise RepairRejected(f"the retained release repeats task {task_id}")
            seen.add(task_id)
            if task_id in exclusions:
                excluded.append({"task_id": task_id, **exclusions[task_id].__dict__})
                input_hashes[task_id] = row["task_sha256"]
                continue
            binding = bindings.get(task_id)
            if binding is None:
                raise RepairRejected(f"the image bindings omit task {task_id}")
            repaired, entry = repair_task(row, binding)
            ledger.append(entry)
            serialized = repaired.model_dump_json()
            output_sha = hashlib.sha256(serialized.encode()).hexdigest()
            input_hashes[task_id] = row["task_sha256"]
            conversions.append(
                {
                    "task_id": task_id,
                    "input_task_sha256": row["task_sha256"],
                    "output_task_sha256": output_sha,
                    "base_digest": binding.base_digest,
                    "pinned_image": binding.image,
                    "pinned_digest": binding.digest,
                    "changed_fields": entry["changed_fields"],
                }
            )
            rows.append(
                {
                    "id": row["id"],
                    "path": row["path"],
                    "task_spec": serialized,
                    "task_sha256": output_sha,
                    "runtime": row["runtime"],
                }
            )
    unknown = sorted(set(bindings) - seen - set(exclusions))
    if unknown:
        raise RepairRejected(f"the image bindings name tasks outside the population: {unknown[:5]}")
    missing_exclusions = sorted(set(exclusions) - seen)
    if missing_exclusions:
        raise RepairRejected(f"the exclusions name tasks outside the population: {missing_exclusions[:5]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "tasks.parquet"
    ledger_path = output_dir / "repair-ledger.json"
    ledger_path.write_text(
        json.dumps({"version": REPAIR_VERSION, "tasks": ledger, "excluded_tasks": excluded}, indent=1) + "\n"
    )
    with output.open("wb") as stream:
        with pq.ParquetWriter(stream, schema) as writer:
            for start in range(0, len(rows), 64):
                writer.write_batch(pa.RecordBatch.from_pylist(rows[start : start + 64], schema=schema))
    receipt = {
        "protocol": CONVERSION_RECEIPT_PROTOCOL,
        "converter": {
            "name": CONVERTER_NAME,
            "version": REPAIR_VERSION,
            "revision": revision,
        },
        "input": {
            "tasks_parquet_sha256": hashlib.file_digest(input_parquet.open("rb"), "sha256").hexdigest(),
            "task_specs_sha256": canonical_sha256([input_hashes[task_id] for task_id in sorted(input_hashes)]),
            "rows": len(input_hashes),
        },
        "output": {
            "tasks_parquet_sha256": hashlib.file_digest(output.open("rb"), "sha256").hexdigest(),
            "rows": len(rows),
        },
        "image_bindings": {
            "tasks": len(bindings),
        },
        "excluded_tasks": excluded,
        "conversions": conversions,
        "completed_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    receipt_path = output_dir / "conversion-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=1) + "\n")
    summary = {
        "version": REPAIR_VERSION,
        "input_parquet_sha256": hashlib.file_digest(input_parquet.open("rb"), "sha256").hexdigest(),
        "tasks": len(rows),
        "excluded_tasks": len(excluded),
        "exclusion_reasons": dict(Counter(entry["reason"] for entry in excluded)),
        "ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "conversion_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "output_sha256": hashlib.file_digest(output.open("rb"), "sha256").hexdigest(),
    }
    (output_dir / "repair-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="the pinned retained tasks.parquet")
    parser.add_argument("--repo", default=None, help="the pinned release repo, downloaded when --input is absent")
    parser.add_argument("--revision", default=None, help="the pinned release revision")
    parser.add_argument("--bindings", required=True, help="the image lane's binding document URI")
    parser.add_argument("--exclude", default=None, help="the typed per-task exclusion document URI")
    parser.add_argument("--output", required=True, help="directory for the repaired release, local or object-store")
    arguments = parser.parse_args()
    if (arguments.input is None) == (arguments.repo is None):
        parser.error("provide exactly one of --input or --repo/--revision")
    input_parquet = arguments.input
    if input_parquet is None:
        input_parquet = Path(
            hf_hub_download(arguments.repo, "tasks.parquet", repo_type="dataset", revision=arguments.revision)
        )
    bindings = bindings_from_document(json.loads(StoragePath(arguments.bindings).read_text()))
    exclusions = (
        exclusions_from_document(json.loads(StoragePath(arguments.exclude).read_text())) if arguments.exclude else {}
    )
    output = StoragePath(arguments.output)
    if output.is_local:
        summary = repair_population(input_parquet, bindings, Path(arguments.output), exclusions)
        print(json.dumps(summary, indent=1))
        return
    with TemporaryDirectory(prefix="mimo-repair-") as temporary:
        staging = Path(temporary)
        summary = repair_population(input_parquet, bindings, staging, exclusions)
        for name in ("tasks.parquet", "repair-ledger.json", "repair-summary.json", "conversion-receipt.json"):
            destination = output / name
            if destination.exists():
                raise FileExistsError(str(destination))
            copy_file_verified(staging / name, destination)
        print(json.dumps(summary, indent=1))


def copy_file_verified(source: Path, destination: StoragePath) -> None:
    """Upload one artifact and verify the stored bytes against the source file."""
    digest = hashlib.file_digest(source.open("rb"), "sha256").hexdigest()
    with source.open("rb") as reader, destination.open("wb", auto_mkdir=True) as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
    with destination.open("rb") as reader:
        if hashlib.file_digest(reader, "sha256").hexdigest() != digest:
            raise ValueError(f"the repaired artifact readback differs: {destination}")


if __name__ == "__main__":
    main()
