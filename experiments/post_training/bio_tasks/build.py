# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build and validate an inspectable Harbor corpus without model calls.

uv run python -m experiments.post_training.bio_tasks.build --help
"""

import argparse
import hashlib
import html
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, replace
from functools import cache
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import tomlkit

from experiments.post_training.bio_tasks.contract import grade_answer, grade_files
from experiments.post_training.bio_tasks.environments import PROFILES, environment_files
from experiments.post_training.bio_tasks.real_data import source_catalog
from experiments.post_training.bio_tasks.recipe_types import DataOrigin, Instance, Recipe
from experiments.post_training.bio_tasks.recipes import RECIPES
from experiments.post_training.tasktrove.task_format import dockerfile_id, render_task_toml
from experiments.post_training.tasktrove.taskbinary import TaskFiles, write_task_binary

SOURCE_DIR = Path(__file__).parent
HARBOR_REVISION = "d072bef08e54050880b484eb81d892944d1d82fb"
MAX_DISTINCT_INSTANCE_ATTEMPTS = 64
PARQUET_SCHEMA = pa.schema(
    [
        ("path", pa.string()),
        ("source", pa.string()),
        ("family", pa.string()),
        ("template_id", pa.string()),
        ("converter", pa.string()),
        ("mode", pa.string()),
        ("dockerfile_id", pa.string()),
        ("language", pa.string()),
        ("tags", pa.list_(pa.string())),
        ("has_solution", pa.bool_()),
        ("task_binary", pa.binary()),
        ("solution_binary", pa.binary()),
    ]
)


@dataclass(frozen=True)
class Identity:
    task_id: str
    lineage: str
    split: str
    seed: int


def identity(recipe: Recipe, seed: int, index: int) -> Identity:
    # Keep lineage and generated inputs stable when the recipe version changes.
    lineage = f"synthetic:{recipe.id}:{seed}:{index}"
    digest = hashlib.sha256(lineage.encode()).digest()
    return Identity(
        f"{recipe.id}-v{recipe.version}-{seed}-{index:05d}", lineage, "train", int.from_bytes(digest[8:16], "big")
    )


@cache
def oracle_archive() -> bytes:
    """Package independent solvers for execution with Python isolated mode."""
    prefix = "experiments/post_training/bio_tasks/"
    files = {
        "__main__.py": b"from experiments.post_training.bio_tasks.oracle import main\nmain()\n",
        "experiments/__init__.py": b"",
        "experiments/post_training/__init__.py": b"",
        prefix + "__init__.py": b"",
        prefix + "oracle.py": (SOURCE_DIR / "oracle.py").read_bytes(),
    }
    files.update(
        {
            prefix + path.relative_to(SOURCE_DIR).as_posix(): path.read_bytes()
            for path in (SOURCE_DIR / "solvers").rglob("*.py")
        }
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    return buffer.getvalue()


def oracle_files(recipe: Recipe) -> TaskFiles:
    return TaskFiles(
        {
            "solution/oracle.pyz": oracle_archive(),
            "solution/solve.sh": f"#!/bin/sh\nset -eu\npython3 -I /solution/oracle.pyz {recipe.id}\n".encode(),
        }
    )


def validate_instance(
    recipe: Recipe,
    instance: Instance,
    reference_output: Path | None = None,
    previous_outputs: tuple[Path, ...] = (),
) -> dict:
    """Require an independently executed solver and discriminating negative controls."""
    with tempfile.TemporaryDirectory(prefix="bio-reference-") as directory:
        root = Path(directory)
        for name, text in instance.inputs.items():
            (root / name).write_text(text)
        answer = root / "answer.json"
        program = root / "oracle.pyz"
        program.write_bytes(oracle_archive())
        subprocess.run(
            [
                sys.executable,
                "-I",
                str(program),
                recipe.id,
                "--inputs",
                str(root),
                "--answer",
                str(answer),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        solved = json.loads(answer.read_text())
        reference = root / "reference.json"
        reference.write_text(instance.contract.model_dump_json())
        if grade_files(reference, answer).reward != 1:
            raise ValueError(f"{recipe.id}: validation oracle outputs failed verification")
        artifact_checks = {}
        for name in instance.contract.fastq:
            artifact = root / name
            original = artifact.read_bytes()
            artifact.unlink()
            artifact_checks[f"missing_artifact:{name}"] = grade_files(reference, answer).reward
            artifact.write_bytes(original + b"@truncated\n")
            artifact_checks[f"malformed_artifact:{name}"] = grade_files(reference, answer).reward
            if original:
                lines = original.splitlines(keepends=True)
                lines[1] = (b"A" if lines[1][:1] != b"A" else b"C") + lines[1][1:]
                artifact.write_bytes(b"".join(lines))
                artifact_checks[f"changed_base:{name}"] = grade_files(reference, answer).reward
            artifact.write_bytes(original)
        for name in instance.contract.alignments:
            artifact = root / name
            original = artifact.read_text()
            artifact.unlink()
            artifact_checks[f"missing_artifact:{name}"] = grade_files(reference, answer).reward
            lines = original.splitlines()
            index = next(i for i, line in enumerate(lines) if line and not line.startswith(">"))
            lines[index] = ("A" if lines[index][0] != "A" else "C") + lines[index][1:]
            artifact.write_text("\n".join(lines) + "\n")
            artifact_checks[f"changed_residue:{name}"] = grade_files(reference, answer).reward
            artifact.write_text(original + ">duplicate\nX\n")
            artifact_checks[f"malformed_artifact:{name}"] = grade_files(reference, answer).reward
            artifact.write_text(original)
        for name in instance.contract.tables:
            artifact = root / name
            original = artifact.read_bytes()
            artifact.unlink()
            artifact_checks[f"missing_artifact:{name}"] = grade_files(reference, answer).reward
            lines = original.splitlines(keepends=True)
            artifact.write_bytes(b"".join(lines[:-1]))
            artifact_checks[f"missing_table_row:{name}"] = grade_files(reference, answer).reward
            artifact.write_bytes(original + lines[-1])
            artifact_checks[f"duplicate_table_row:{name}"] = grade_files(reference, answer).reward
            artifact.write_bytes(original)
        if any(value != 0 for value in artifact_checks.values()):
            raise ValueError(f"{recipe.id}: invalid native artifact passed: {artifact_checks}")
        if previous_outputs:
            for previous in previous_outputs:
                verdict = grade_files(reference, previous / "answer.json")
                if verdict.reward != 0 or verdict.status.value != "scored":
                    raise ValueError(f"{recipe.id}: copied outputs from another instance were not rejected")
            artifact_checks["copied_other_instance"] = 0
        if reference_output is not None:
            reference_output.mkdir(parents=True, exist_ok=False)
            for name in ("answer.json", *instance.contract.artifacts()):
                shutil.copyfile(root / name, reference_output / name)
    candidates = {"oracle": solved, "row_permutation": list(reversed(solved))}
    rejected = {"empty": [], "missing_id": solved[:-1], "duplicate_id": [*solved, solved[0]], **instance.mutations}
    checks = dict(artifact_checks)
    for name, candidate in {**candidates, **rejected}.items():
        verdict = grade_answer(instance.contract, json.dumps(candidate, allow_nan=False))
        expected_reward = float(name in candidates)
        if verdict.reward != expected_reward:
            raise ValueError(
                f"{recipe.id}: validation {name} expected {expected_reward}, "
                f"got reward={verdict.reward}, status={verdict.status}"
            )
        checks[name] = verdict.reward
    malformed = grade_answer(instance.contract, "not json")
    if malformed.reward != 0:
        raise ValueError(f"{recipe.id}: malformed answer passed")
    checks["malformed"] = malformed.reward
    return checks


def task_files(recipe: Recipe, instance: Instance, task: Identity, base_image: str, tool_ref: str) -> TaskFiles:
    profile = PROFILES.get(recipe.id)
    solver_environment = environment_files(recipe.id, base_image)
    metadata = {
        "task_id": task.task_id,
        "tool_environment": profile.evidence if profile else "python-only",
        "recipe": recipe.id,
        "recipe_version": recipe.version,
        "lineage": task.lineage,
        "split": task.split,
        "difficulty": recipe.difficulty.value,
        **instance_provenance(instance),
        "domain": recipe.domain,
        "repositories": list(recipe.repositories),
        "tool_execution": "pending",
        "network": "offline",
        "skills": list(recipe.skills),
        "input_formats": list(recipe.formats),
        "sources": list(recipe.sources),
        "generation_seed": str(task.seed),
        "scientific_review": "pending",
    }
    config = tomlkit.parse(render_task_toml(1800, 60, metadata))
    config["artifacts"] = ["/app/answer.json", *(f"/app/{name}" for name in instance.contract.artifacts())]
    # A fresh verifier environment receives only the declared submitted artifacts.
    config["verifier"] = {
        "timeout_sec": 60,
        "environment_mode": "separate",
        "environment": {"allow_internet": False, "cpus": 1, "memory_mb": 1024},
    }
    config["environment"] = {
        "allow_internet": False,
        "cpus": 1,
        "memory_mb": profile.memory_mb if profile else 1024,
        "storage_mb": 8192,
    }
    verifier = (
        f"FROM {base_image}\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends git "
        "&& rm -rf /var/lib/apt/lists/*\n"
        "RUN python3 -m venv /opt/verifier && /opt/verifier/bin/pip install --no-cache-dir "
        '"pydantic==2.12.5" "tomlkit==0.13.3" '
        f'"tasktrove-verify @ git+https://github.com/marin-community/marin@{tool_ref}'
        '#subdirectory=lib/tasktrove-verify"\n'
        "COPY test.sh contract.py reference.json /tests/\n"
        "RUN chmod 755 /tests/test.sh\n"
        "WORKDIR /app\n"
    )
    files = {
        "instruction.md": (instance.instruction + "\n\n" + instance.contract.instructions() + "\n").encode(),
        "task.toml": tomlkit.dumps(config).encode(),
        **solver_environment.files,
        "tests/Dockerfile": verifier.encode(),
        "tests/test.sh": b"#!/bin/sh\nset -eu\nexec /opt/verifier/bin/python -I /tests/contract.py\n",
        "tests/contract.py": (SOURCE_DIR / "contract.py").read_bytes(),
        "tests/reference.json": instance.contract.model_dump_json(indent=2).encode(),
    }
    files.update({f"setup_files/inputs/{name}": text.encode() for name, text in instance.inputs.items()})
    return TaskFiles(files)


def instance_provenance(instance: Instance) -> dict:
    return {
        "data_origin": instance.data_origin.value,
        "biological_sources": list(instance.source_ids),
        "biological_lineages": [source_catalog()[source]["lineage"] for source in instance.source_ids],
        "derivation": instance.derivation,
        "scale_profile": "small-fixture" if instance.data_origin == DataOrigin.SIMULATED else "observed-study",
        "workflow_scope": "component",
        "training_ready": False,
    }


def inspection_page(root: Path, task: Identity, instance: Instance, files: TaskFiles, validation: dict) -> None:
    inputs = "".join(
        f"<details><summary>{html.escape(name)}</summary><pre>{html.escape(text[:8000])}</pre>" "</details>"
        for name, text in instance.inputs.items()
    )
    sections = {
        "Biological data provenance": json.dumps(
            {
                "instance": instance_provenance(instance),
                "sources": {key: source_catalog()[key] for key in instance.source_ids},
            },
            indent=2,
        ),
        "Exact instruction": files.text("instruction.md"),
        "Expected output (private)": json.dumps(instance.contract.answer(), indent=2),
        "Native artifact contracts (private)": json.dumps(
            {
                "fastq": {name: target.model_dump() for name, target in instance.contract.fastq.items()},
                "alignments": {name: target.model_dump() for name, target in instance.contract.alignments.items()},
                "tables": {
                    name: {
                        "columns": {column: spec.model_dump() for column, spec in target.columns.items()},
                        "rows": len(target.expected),
                        "max_bytes": target.max_bytes,
                        "expected_preview": dict(list(target.expected.items())[:20]),
                    }
                    for name, target in instance.contract.tables.items()
                },
            },
            indent=2,
        ),
        "Validation controls": json.dumps(validation, indent=2),
        "Scientific negative controls": json.dumps(instance.mutations, indent=2),
        "Task metadata": files.text("task.toml"),
        "Verifier": files.text("tests/contract.py"),
    }
    content = "".join(
        f"<details><summary>{title}</summary><pre>{html.escape(text)}</pre></details>"
        for title, text in sections.items()
    )
    output_links = " ".join(
        f'<a href="../reference-outputs/{task.task_id}/{name}">{html.escape(name)}</a>'
        for name in ("answer.json", *instance.contract.artifacts())
    )
    page = (
        '<!doctype html><meta charset="utf-8"><title>' + html.escape(task.task_id) + "</title>"
        "<style>body{max-width:1000px;margin:2rem auto;font:16px system-ui}"
        "pre{white-space:pre-wrap;background:#f4f4f4;padding:1rem}details{margin:1rem 0}</style>"
        '<a href="../index.html">Corpus</a><h1>' + html.escape(task.task_id) + "</h1>"
        f"<p>Data origin: <strong>{instance.data_origin.value}</strong>. "
        f"{html.escape(instance.derivation)}</p><p>Component coverage; end-to-end workflow validation is pending. "
        "Scientific review pending. Private inspection view; never mount this directory in solver environments. "
        "Input previews are limited to 8,000 characters; complete files are in the task bundle.</p>"
        + content
        + "<p>Verified reference outputs (private): "
        + output_links
        + "</p>"
        + "<h2>Inputs</h2>"
        + inputs
    )
    (root / "inspect" / f"{task.task_id}.html").write_text(page)


def benchmark_page(output: Path, registry: dict) -> None:
    rows = []
    for task in registry["tasks"]:
        source = registry["benchmarks"][task["benchmark"]]
        examples = " ".join(
            f'<a href="inspect/{identifier}.html">{html.escape(identifier)}</a>' for identifier in task["examples"]
        )
        details = html.escape(json.dumps({key: value for key, value in task.items() if key != "examples"}, indent=2))
        rows.append(
            f'<tr data-benchmark="{html.escape(task["benchmark"])}" data-status="{task["status"]}" '
            f'data-distribution="{source["distribution"]}">'
            f'<td>{source["distribution"]}</td><td>{html.escape(task["benchmark"])}</td>'
            f'<td><a href="{html.escape(source["source_url"])}">'
            f'{html.escape(task["task_id"])}</a></td><td>{task["status"]}</td>'
            "<td><details><summary>Stages, gaps and evidence</summary>"
            f"<pre>{details}</pre>{examples}</details></td></tr>"
        )
    summaries = {
        distribution: dict(
            Counter(
                task["status"]
                for task in registry["tasks"]
                if registry["benchmarks"][task["benchmark"]]["distribution"] == distribution
            )
        )
        for distribution in ("ID", "OOD")
    }
    options = "".join(f"<option>{html.escape(name)}</option>" for name in registry["benchmarks"])
    statuses = "".join(f"<option>{name}</option>" for name in registry["status_definitions"])
    page = (
        '<!doctype html><meta charset="utf-8"><title>Benchmark workflow coverage</title>'
        "<style>body{margin:2rem;font:16px system-ui}td,th{padding:.5rem;text-align:left;vertical-align:top}"
        "pre{white-space:pre-wrap;max-width:70rem}select,input{padding:.5rem}[hidden]{display:none}</style>"
        '<a href="index.html">Corpus</a><h1>Benchmark workflow coverage</h1>'
        f'<p>{len(registry["tasks"])} task identifiers. {html.escape(str(dict(summaries)))}</p>'
        "<p>Component mappings identify shared operations. They do not establish benchmark workflow coverage. "
        "Reference runtimes below measure local solver checks, not teacher attempts. "
        "BioMysteryBench is OOD and excluded from training authoring. "
        "ID workflow coverage is the task-dataset target.</p>"
        '<p><a href="benchmark_coverage.json">Download pinned registry</a></p>'
        '<input id="search" type="search" placeholder="Search IDs, stages or gaps" aria-label="Search tasks">'
        '<select id="distribution" aria-label="Distribution"><option selected>ID</option><option>OOD</option>'
        '<option value="">All distributions</option></select>'
        f'<select id="benchmark" aria-label="Benchmark"><option value="">All benchmarks</option>{options}</select>'
        f'<select id="status" aria-label="Coverage status"><option value="">All statuses</option>{statuses}</select>'
        "<table><thead><tr><th>Distribution</th><th>Benchmark</th><th>Task</th><th>Coverage</th><th>Details</th></tr></thead><tbody>"
        + "".join(rows)
        + '</tbody></table><script>const q=document.querySelector("#search"),b=document.querySelector("#benchmark"),'
        's=document.querySelector("#status"),d=document.querySelector("#distribution");'
        'function filter(){for(const r of document.querySelectorAll("tbody tr"))'
        "{r.hidden=!(r.textContent.toLowerCase().includes(q.value.toLowerCase())&&"
        "(!b.value||r.dataset.benchmark===b.value)&&(!s.value||r.dataset.status===s.value)&&"
        "(!d.value||r.dataset.distribution===d.value));}}"
        'q.addEventListener("input",filter);b.addEventListener("change",filter);'
        's.addEventListener("change",filter);d.addEventListener("change",filter);filter();</script>'
    )
    (output / "benchmark-coverage.html").write_text(page)


def build(output: Path, instances_per_recipe: int, seed: int, base_image: str, tool_ref: str) -> dict:
    """Stream validated task binaries, manifests, oracles, and private inspection pages."""
    if instances_per_recipe < 1 or seed < 0:
        raise ValueError("instances_per_recipe must be positive and seed must be nonnegative")
    if not re.fullmatch(r"[a-zA-Z0-9./:_-]+@sha256:[0-9a-f]{64}", base_image):
        raise ValueError("base_image must be a digest-pinned Python image")
    if not re.fullmatch(r"[0-9a-f]{40}", tool_ref):
        raise ValueError("tool_ref must be a full Marin commit SHA")
    output.mkdir(parents=True, exist_ok=False)
    (output / "tasks").mkdir()
    (output / "inspect").mkdir()
    inventory = (SOURCE_DIR / "source_inventory.json").read_bytes()
    (output / "source_inventory.json").write_bytes(inventory)
    coverage_bytes = (SOURCE_DIR / "repository_coverage.json").read_bytes()
    (output / "repository_coverage.json").write_bytes(coverage_bytes)
    coverage = json.loads(coverage_bytes)["repositories"]
    native_evidence = (SOURCE_DIR / "native_validation.json").read_bytes()
    (output / "native_validation.json").write_bytes(native_evidence)
    native_run_hashes = {}
    for run in json.loads(native_evidence)["runs"]:
        name = run["checks_file"]
        content = (SOURCE_DIR / name).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != run["checks_sha256"]:
            raise ValueError(f"Changed native validation evidence: {name}")
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        native_run_hashes[name] = digest
    data_sources = (SOURCE_DIR / "data_sources.json").read_bytes()
    (output / "data_sources.json").write_bytes(data_sources)
    benchmark_bytes = (SOURCE_DIR / "benchmark_coverage.json").read_bytes()
    benchmark_registry = json.loads(benchmark_bytes)
    known_recipes = {recipe.id for recipe in RECIPES}
    for task in benchmark_registry["tasks"]:
        benchmark = benchmark_registry["benchmarks"][task["benchmark"]]
        if benchmark["distribution"] not in {"ID", "OOD"}:
            raise ValueError(f"Unknown benchmark distribution: {task['benchmark']}")
        if task["recipes"] and (benchmark["distribution"] == "OOD" or not benchmark["training_mapping_allowed"]):
            raise ValueError(f"Evaluation-only task mapped to training recipes: {task['task_id']}")
        if unknown := set(task["recipes"]) - known_recipes:
            raise ValueError(f"Unknown benchmark coverage recipes: {unknown}")
    counts: Counter = Counter()
    dockerfiles = {}
    recipe_sections = []
    examples: dict[str, list[dict]] = {}
    origins: Counter = Counter()
    seen_inputs = set()
    seen_targets = set()
    previous_outputs: dict[str, list[Path]] = {}
    manifest = {
        "schema_version": 1,
        "corpus_stage": "authoring-candidates-and-controls",
        "training_ready": False,
        "seed": seed,
        "instances_per_recipe": instances_per_recipe,
        "base_image": base_image,
        "tasktrove_revision": tool_ref,
        "harbor_revision": HARBOR_REVISION,
        "scientific_review": "pending",
        "container_validation": "pending",
        "split_policy": "single train split; evaluation benchmarks remain separate",
        "source_hashes": {
            path.relative_to(SOURCE_DIR).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(SOURCE_DIR.rglob("*.py"))
        },
        "source_inventory_sha256": hashlib.sha256(inventory).hexdigest(),
        "repository_coverage_sha256": hashlib.sha256(coverage_bytes).hexdigest(),
        "native_validation_sha256": hashlib.sha256(native_evidence).hexdigest(),
        "native_validation_run_sha256": native_run_hashes,
        "data_sources_sha256": hashlib.sha256(data_sources).hexdigest(),
        "benchmark_mapping_source_sha256": hashlib.sha256(benchmark_bytes).hexdigest(),
        "repository_tool_execution": {row["name"]: row["tool_execution"] for row in coverage},
    }
    # A failed generation leaves an explicit incomplete manifest, never an apparently finished release.
    (output / "manifest.json").write_text(json.dumps({**manifest, "status": "incomplete"}, indent=2) + "\n")
    with (
        (output / "ledger.jsonl").open("w") as ledger,
        pq.ParquetWriter(output / "tasks" / "part-00000.parquet", PARQUET_SCHEMA) as writer,
    ):
        for recipe in RECIPES:
            index_rows = []
            for index in range(instances_per_recipe):
                task = identity(recipe, seed, index)
                for draw in range(MAX_DISTINCT_INSTANCE_ATTEMPTS):
                    if draw:
                        digest = hashlib.sha256(f"{task.lineage}:distinct-input:{draw}".encode()).digest()
                        task = replace(task, seed=int.from_bytes(digest[:8], "big"))
                    instance = recipe.generate(task.seed)
                    input_hash = hashlib.sha256(json.dumps(instance.inputs, sort_keys=True).encode()).hexdigest()
                    target_hash = hashlib.sha256(
                        json.dumps(
                            {
                                "answer": instance.contract.expected,
                                "fastq": {name: target.sha256 for name, target in instance.contract.fastq.items()},
                                "alignments": {
                                    name: target.model_dump() for name, target in instance.contract.alignments.items()
                                },
                                "tables": {name: target.expected for name, target in instance.contract.tables.items()},
                            },
                            sort_keys=True,
                        ).encode()
                    ).hexdigest()
                    if input_hash not in seen_inputs and (recipe.id, target_hash) not in seen_targets:
                        break
                else:
                    raise ValueError(
                        f"No distinct inputs and target after {MAX_DISTINCT_INSTANCE_ATTEMPTS} draws: {task.task_id}"
                    )
                seen_inputs.add(input_hash)
                seen_targets.add((recipe.id, target_hash))
                validation_start = time.monotonic()
                reference_output = output / "reference-outputs" / task.task_id
                validation = validate_instance(
                    recipe, instance, reference_output, tuple(previous_outputs.get(recipe.id, ()))
                )
                validation_runtime = time.monotonic() - validation_start
                previous_outputs.setdefault(recipe.id, []).append(reference_output)
                files = task_files(recipe, instance, task, base_image, tool_ref)
                task_dir = output / "harbor" / task.split / task.task_id
                files.write_to(task_dir)
                solution = oracle_files(recipe)
                solution.write_to(output / "oracles" / task.task_id)
                blob = write_task_binary(files)
                image_id = dockerfile_id(files.text("environment/Dockerfile"))
                dockerfiles[image_id] = {"base_image": base_image}
                row = {
                    "path": task.task_id,
                    "source": "bio-tasks",
                    "family": recipe.id,
                    "template_id": f"{recipe.id}-v{recipe.version}",
                    "converter": "bio-tasks-v1",
                    "mode": "script",
                    "dockerfile_id": image_id,
                    "language": "python",
                    "tags": [
                        recipe.difficulty.value,
                        task.split,
                        "offline",
                        f"data-origin:{instance.data_origin.value}",
                        f"domain:{recipe.domain}",
                        *[f"repository:{name}" for name in recipe.repositories],
                        *recipe.skills,
                        *[f"format:{profile}" for profile in recipe.formats],
                    ],
                    "has_solution": True,
                    "task_binary": blob,
                    "solution_binary": write_task_binary(solution),
                }
                writer.write_table(pa.Table.from_pylist([row], schema=PARQUET_SCHEMA))
                entry = {
                    **asdict(task),
                    "recipe": recipe.id,
                    "recipe_version": recipe.version,
                    "difficulty": recipe.difficulty.value,
                    "domain": recipe.domain,
                    "repositories": recipe.repositories,
                    "tool_execution": "pending",
                    "skills": recipe.skills,
                    "input_formats": recipe.formats,
                    "sources": recipe.sources,
                    "network": "offline",
                    "input_sha256": input_hash,
                    "input_bytes": sum(len(text.encode("utf-8")) for text in instance.inputs.values()),
                    **instance_provenance(instance),
                    "distinct_instance_draw": draw,
                    "target_sha256": target_hash,
                    "task_sha256": hashlib.sha256(blob).hexdigest(),
                    "validation": validation,
                    "scientific_review": "pending",
                    "teacher_attempts": 0,
                }
                ledger.write(json.dumps(entry) + "\n")
                examples.setdefault(recipe.id, []).append({**entry, "reference_validation_seconds": validation_runtime})
                origins[instance.data_origin.value] += 1
                inspection_page(output, task, instance, files, validation)
                index_rows.append(
                    f'<tr><td><a href="inspect/{task.task_id}.html">{task.task_id}</a></td>'
                    f"<td>{recipe.difficulty.value}</td>"
                    f"<td>{entry['input_bytes']:,} bytes</td>"
                    f"<td>{len(validation)}</td></tr>"
                )
                counts[task.split] += 1
                counts[recipe.id] += 1
            recipe_sections.append(
                f'<section class="recipe" data-domain="{html.escape(recipe.domain)}" '
                f'data-origin="{instance.data_origin.value}" '
                f'id="{html.escape(recipe.id)}"><h2>{html.escape(recipe.id)}</h2>'
                f"<p>{instances_per_recipe} examples · {recipe.difficulty.value}<br>"
                f"Domain: {html.escape(recipe.domain)}<br>"
                f"Data origin: <strong>{instance.data_origin.value}</strong><br>"
                "Repository-derived operations: "
                f"{html.escape(', '.join(recipe.repositories)) or 'Additional domain coverage'}"
                "<br>Tool availability in generated task environment: pending<br>"
                f"Input formats: {html.escape(', '.join(recipe.formats))}<br>"
                f"Skills: {html.escape(', '.join(recipe.skills))}</p>"
                "<table><thead><tr><th>Task</th><th>Difficulty</th><th>Input size</th>"
                "<th>Validation controls</th></tr></thead>"
                "<tbody>" + "\n".join(index_rows) + "</tbody></table></section>"
            )
    manifest.update(
        status="validated_locally",
        counts=dict(counts),
        tasks=counts["train"],
        dockerfiles=dockerfiles,
        by_source={"bio-tasks": {"converted": counts["train"]}},
        recipe_formats={recipe.id: recipe.formats for recipe in RECIPES},
        recipe_domains={recipe.id: recipe.domain for recipe in RECIPES},
        domain_counts=dict(Counter(recipe.domain for recipe in RECIPES)),
        data_origin_counts=dict(origins),
    )
    for task in benchmark_registry["tasks"]:
        entries = [entry for recipe in task["recipes"] for entry in examples[recipe]]
        task["examples"] = [entry["task_id"] for entry in entries]
        task["validation_evidence"] = [
            {
                "task_id": entry["task_id"],
                "task_sha256": entry["task_sha256"],
                "data_origin": entry["data_origin"],
                "sources": entry["biological_sources"],
                "independent_solver": "passed",
                "reference_validation_seconds": entry["reference_validation_seconds"],
            }
            for entry in entries
        ]
    registry_bytes = (json.dumps(benchmark_registry, indent=2) + "\n").encode()
    (output / "benchmark_coverage.json").write_bytes(registry_bytes)
    manifest["benchmark_coverage_sha256"] = hashlib.sha256(registry_bytes).hexdigest()
    manifest["benchmark_coverage_counts"] = dict(Counter(task["status"] for task in benchmark_registry["tasks"]))
    distribution_counts = Counter(
        benchmark_registry["benchmarks"][task["benchmark"]]["distribution"] for task in benchmark_registry["tasks"]
    )
    manifest["benchmark_distribution_counts"] = dict(distribution_counts)
    benchmark_page(output, benchmark_registry)
    domain_options = "".join(
        f'<option value="{html.escape(domain)}">{html.escape(domain)} ({count} recipes)</option>'
        for domain, count in sorted(manifest["domain_counts"].items())
    )
    repository_rows = []
    for row in coverage:
        status = html.escape(row["tool_execution"]["status"])
        if evidence := row["tool_execution"]["evidence"]:
            status = f'<a href="{html.escape(evidence)}">{status}</a>'
        repository_rows.append(
            f'<tr><td>{row["index"]}</td><td>{html.escape(row["name"])}</td><td>'
            + ", ".join(f'<a href="#{name}">{name}</a>' for name in row["recipes"])
            + f"</td><td>{status}</td></tr>"
        )
    passed_packages = sum(row["tool_execution"]["status"] == "passed_reference_check" for row in coverage)
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Biology task inspection</title>'
        "<style>body{margin:2rem;font:16px system-ui}td,th{padding:.4rem;text-align:left}"
        "section{border-top:1px solid #ccc;padding-top:1rem}input,select{font:inherit;padding:.5rem}"
        ".filters{position:sticky;top:0;background:white;padding:1rem 0;display:flex;gap:1rem}"
        "[hidden]{display:none!important}</style>"
        "<h1>Biology task inspection</h1><p>Private references included. Scientific review and container validation "
        "are pending. No teacher attempts have run.</p>"
        f"<p>{len(RECIPES)} recipes &times; {instances_per_recipe} examples = {counts['train']} tasks. "
        "All tasks belong to the train split.</p>"
        f"<p>Data origins: {html.escape(str(dict(origins)))}. Real observations are authoring candidates; "
        "simulated examples are small correctness controls. Benchmark data-lineage exclusion, scientific review "
        "and end-to-end workflow validation remain pending.</p>"
        "<p>All 50 source repositories have an explicit recipe mapping below. The table tracks separately "
        "recorded package checks. Locked MUSCLE, fastp and Picard image contexts are prepared "
        "for their real-data tasks; "
        "other environments contain Python only. Container execution and teacher tool use require separate checks.</p>"
        f"<p>{passed_packages}/50 packages have passed three reference cases each.</p>"
        "<p>Format labels describe supplied inputs. Generic CSV/JSON summaries do not establish native format coverage. "
        "Newick, Matrix Market, PDB, mmCIF, SBML, MGF and PGM are supplied where labeled; H5AD, BAM, "
        "native SRA, and OME-TIFF are not covered by their text intermediates.</p>"
        '<p><a href="native_validation.json">Recorded package reference checks</a></p>'
        '<p><a href="benchmark-coverage.html">Inspect workflow coverage: '
        f'{distribution_counts["ID"]} ID tasks; {distribution_counts["OOD"]} OOD tasks held out</a> · '
        '<a href="data_sources.json">Biological sources and provenance</a></p>'
        "<details><summary>All 50 repositories: scientific operation and execution status</summary>"
        "<table><thead><tr><th>#</th><th>Repository</th><th>Recipes</th><th>CLI/API execution</th></tr></thead><tbody>"
        + "".join(repository_rows)
        + "</tbody></table></details>"
        '<div class="filters"><input id="search" type="search" placeholder="Search recipe, repo, skill or format" '
        'aria-label="Search recipes"><select id="domain" aria-label="Filter domain">'
        '<option value="">All domains</option>'
        + domain_options
        + '</select><select id="origin" aria-label="Data origin"><option value="real">Real data</option>'
        '<option value="">All origins</option><option value="simulated">Simulated controls</option>'
        '<option value="modified-real">Modified real data</option></select><span id="visible"></span></div>'
        + "\n".join(recipe_sections)
        + "<script>const search=document.querySelector('#search'),domain=document.querySelector('#domain'),"
        "origin=document.querySelector('#origin');"
        "function filter(){let n=0;for(const s of document.querySelectorAll('.recipe')){"
        "s.hidden=!(s.textContent.toLowerCase().includes(search.value.toLowerCase())&&"
        "(!domain.value||s.dataset.domain===domain.value)&&"
        "(!origin.value||s.dataset.origin===origin.value));if(!s.hidden)n++;}"
        "document.querySelector('#visible').textContent=n+' recipes';}"
        "search.addEventListener('input',filter);domain.addEventListener('change',filter);"
        "origin.addEventListener('change',filter);filter();"
        "document.querySelectorAll('a[href^=\"#\"]').forEach(a=>a.addEventListener('click',()=>{"
        "search.value='';domain.value='';origin.value='';filter();}));</script>"
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances-per-recipe", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base-image", required=True, help="digest-pinned Python image with pip and venv")
    parser.add_argument("--tool-ref", required=True, help="full Marin commit for tasktrove-verify")
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.instances_per_recipe, args.seed, args.base_image, args.tool_ref), indent=2))


if __name__ == "__main__":
    main()
