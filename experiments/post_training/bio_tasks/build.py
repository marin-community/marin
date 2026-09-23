# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build and validate an inspectable Harbor corpus without model calls.

uv run python -m experiments.post_training.bio_tasks.build --help
"""

import argparse
import hashlib
import html
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import tomlkit

from experiments.post_training.bio_tasks.contract import grade_answer
from experiments.post_training.bio_tasks.recipes import RECIPES, Instance, Recipe
from experiments.post_training.tasktrove.task_format import dockerfile_id, render_task_toml
from experiments.post_training.tasktrove.taskbinary import TaskFiles, write_task_binary

SOURCE_DIR = Path(__file__).parent
HARBOR_REVISION = "d072bef08e54050880b484eb81d892944d1d82fb"
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


def oracle_files(recipe: Recipe) -> TaskFiles:
    return TaskFiles(
        {
            "solution/oracle.py": (SOURCE_DIR / "oracle.py").read_bytes(),
            "solution/solve.sh": f"#!/bin/sh\nset -eu\npython3 /solution/oracle.py {recipe.id}\n".encode(),
        }
    )


def validate_instance(recipe: Recipe, instance: Instance) -> dict:
    """Require an independently executed solver and discriminating negative controls."""
    with tempfile.TemporaryDirectory(prefix="bio-reference-") as directory:
        root = Path(directory)
        for name, text in instance.inputs.items():
            (root / name).write_text(text)
        answer = root / "answer.json"
        subprocess.run(
            [
                sys.executable,
                "-I",
                str(SOURCE_DIR / "oracle.py"),
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
    candidates = {"oracle": solved, "row_permutation": list(reversed(solved))}
    rejected = {"empty": [], "missing_id": solved[:-1], "duplicate_id": [*solved, solved[0]], **instance.mutations}
    checks = {}
    for name, candidate in {**candidates, **rejected}.items():
        verdict = grade_answer(instance.contract, json.dumps(candidate, allow_nan=False))
        expected_reward = float(name in candidates)
        if verdict.reward != expected_reward:
            raise ValueError(f"{recipe.id}: validation {name} expected {expected_reward}, got {verdict}")
        checks[name] = verdict.reward
    malformed = grade_answer(instance.contract, "not json")
    if malformed.reward != 0:
        raise ValueError(f"{recipe.id}: malformed answer passed")
    checks["malformed"] = malformed.reward
    return checks


def task_files(recipe: Recipe, instance: Instance, task: Identity, base_image: str, tool_ref: str) -> TaskFiles:
    metadata = {
        "task_id": task.task_id,
        "recipe": recipe.id,
        "recipe_version": recipe.version,
        "lineage": task.lineage,
        "split": task.split,
        "difficulty": recipe.difficulty.value,
        "network": "offline",
        "skills": list(recipe.skills),
        "input_formats": list(recipe.formats),
        "sources": list(recipe.sources),
        "generation_seed": str(task.seed),
        "scientific_review": "pending",
    }
    config = tomlkit.parse(render_task_toml(1800, 60, metadata))
    config["artifacts"] = ["/app/answer.json"]
    # A fresh verifier environment receives only the submitted answer artifact.
    config["verifier"] = {
        "timeout_sec": 60,
        "environment_mode": "separate",
        "environment": {"allow_internet": False, "cpus": 1, "memory_mb": 1024},
    }
    config["environment"] = {"allow_internet": False, "cpus": 1, "memory_mb": 1024}
    environment = f"FROM {base_image}\nWORKDIR /app\nCOPY inputs/ /app/inputs/\n"
    verifier = (
        f"FROM {base_image}\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends git "
        "&& rm -rf /var/lib/apt/lists/*\n"
        "RUN python3 -m venv /opt/verifier && /opt/verifier/bin/pip install --no-cache-dir "
        '"pydantic==2.12.5" "tomlkit==0.13.3" '
        f'"tasktrove-verify @ git+https://github.com/marin-community/marin@{tool_ref}'
        '#subdirectory=lib/tasktrove-verify"\n'
        "WORKDIR /app\n"
    )
    files = {
        "instruction.md": (instance.instruction + "\n\n" + instance.contract.instructions() + "\n").encode(),
        "task.toml": tomlkit.dumps(config).encode(),
        "environment/Dockerfile": environment.encode(),
        "tests/Dockerfile": verifier.encode(),
        "tests/test.sh": b"#!/bin/sh\nset -eu\nexec /opt/verifier/bin/python -I /tests/contract.py\n",
        "tests/contract.py": (SOURCE_DIR / "contract.py").read_bytes(),
        "tests/reference.json": instance.contract.model_dump_json(indent=2).encode(),
    }
    files.update({f"environment/inputs/{name}": text.encode() for name, text in instance.inputs.items()})
    return TaskFiles(files)


def inspection_page(root: Path, task: Identity, instance: Instance, files: TaskFiles, validation: dict) -> None:
    inputs = "".join(
        f"<details><summary>{html.escape(name)}</summary><pre>{html.escape(text[:8000])}</pre>" "</details>"
        for name, text in instance.inputs.items()
    )
    sections = {
        "Exact instruction": files.text("instruction.md"),
        "Expected output (private)": json.dumps(instance.contract.answer(), indent=2),
        "Validation controls": json.dumps(validation, indent=2),
        "Scientific negative controls": json.dumps(instance.mutations, indent=2),
        "Task metadata": files.text("task.toml"),
        "Verifier": files.text("tests/contract.py"),
    }
    content = "".join(
        f"<details><summary>{title}</summary><pre>{html.escape(text)}</pre></details>"
        for title, text in sections.items()
    )
    page = (
        '<!doctype html><meta charset="utf-8"><title>' + html.escape(task.task_id) + "</title>"
        "<style>body{max-width:1000px;margin:2rem auto;font:16px system-ui}"
        "pre{white-space:pre-wrap;background:#f4f4f4;padding:1rem}details{margin:1rem 0}</style>"
        '<a href="../index.html">Corpus</a><h1>' + html.escape(task.task_id) + "</h1>"
        "<p>Scientific review pending. Private inspection view; never mount this directory in solver environments. "
        "Input previews are limited to 8,000 characters; complete files are in the task bundle.</p>"
        + content
        + "<h2>Inputs</h2>"
        + inputs
    )
    (root / "inspect" / f"{task.task_id}.html").write_text(page)


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
    counts: Counter = Counter()
    dockerfiles = {}
    recipe_sections = []
    seen_inputs = set()
    previous_answers: dict[str, str] = {}
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "instances_per_recipe": instances_per_recipe,
        "base_image": base_image,
        "tasktrove_revision": tool_ref,
        "harbor_revision": HARBOR_REVISION,
        "scientific_review": "pending",
        "container_validation": "pending",
        "split_policy": "single train split; evaluation benchmarks remain separate",
        "source_hashes": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(SOURCE_DIR.glob("*.py"))
        },
        "source_inventory_sha256": hashlib.sha256(inventory).hexdigest(),
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
                instance = recipe.generate(task.seed)
                input_hash = hashlib.sha256(json.dumps(instance.inputs, sort_keys=True).encode()).hexdigest()
                if input_hash in seen_inputs:
                    raise ValueError(f"duplicate inputs: {task.task_id}")
                seen_inputs.add(input_hash)
                validation = validate_instance(recipe, instance)
                if recipe.id in previous_answers:
                    different_target = json.loads(previous_answers[recipe.id]) != instance.contract.answer()
                    if different_target and grade_answer(instance.contract, previous_answers[recipe.id]).reward != 0:
                        raise ValueError(f"copied answer from another instance passes: {task.task_id}")
                    if different_target:
                        validation["copied_other_instance"] = 0
                previous_answers[recipe.id] = json.dumps(instance.contract.answer())
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
                    "skills": recipe.skills,
                    "input_formats": recipe.formats,
                    "sources": recipe.sources,
                    "network": "offline",
                    "input_sha256": input_hash,
                    "task_sha256": hashlib.sha256(blob).hexdigest(),
                    "validation": validation,
                    "scientific_review": "pending",
                    "teacher_attempts": 0,
                }
                ledger.write(json.dumps(entry) + "\n")
                inspection_page(output, task, instance, files, validation)
                index_rows.append(
                    f'<tr><td><a href="inspect/{task.task_id}.html">{task.task_id}</a></td>'
                    f"<td>{recipe.difficulty.value}</td>"
                    f"<td>{len(validation)}</td></tr>"
                )
                counts[task.split] += 1
                counts[recipe.id] += 1
            recipe_sections.append(
                f"<section><h2>{html.escape(recipe.id)}</h2>"
                f"<p>{instances_per_recipe} examples · {recipe.difficulty.value}<br>"
                f"Input formats: {html.escape(', '.join(recipe.formats))}<br>"
                f"Skills: {html.escape(', '.join(recipe.skills))}</p>"
                "<table><thead><tr><th>Task</th><th>Difficulty</th><th>Validation controls</th></tr></thead>"
                "<tbody>" + "\n".join(index_rows) + "</tbody></table></section>"
            )
    manifest.update(
        status="validated_locally",
        counts=dict(counts),
        tasks=counts["train"],
        dockerfiles=dockerfiles,
        by_source={"bio-tasks": {"converted": counts["train"]}},
        recipe_formats={recipe.id: recipe.formats for recipe in RECIPES},
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Biology task inspection</title>'
        "<style>body{margin:2rem;font:16px system-ui}td,th{padding:.4rem;text-align:left}</style>"
        "<h1>Biology task inspection</h1><p>Private references included. Scientific review and container validation "
        "are pending. No teacher attempts have run.</p>"
        f"<p>{len(RECIPES)} recipes &times; {instances_per_recipe} examples = {counts['train']} tasks. "
        "All tasks belong to the train split.</p>"
        "<p>Format labels describe supplied inputs. Generic CSV/JSON summaries do not establish coverage "
        "of native formats such as Newick, H5AD, or OME-TIFF.</p>" + "\n".join(recipe_sections)
    )
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
