# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Screen and sample the TMax-15K Harbor reserve (``TMaxxx/TMax-15K-Harbor``) for a generation pilot.

``select`` screens out, without running anything:
- hidden verifier assets, by two rules (a task is excluded when either fires):
  - ``image_path``: test_final_state.py reads an absolute path the image provides (a shipped fixture, or a path in
    the Dockerfile/install scripts) that the instruction names neither by path nor by file name;
  - ``answer_name``: pool P's tmax_leaks rule (truth/expected/oracle/solution/verifier/answer/reference paths created
    by the build and read by the test), applied to the Harbor layout;
- every task shipping an ``_fixtures/.../oracle`` file;
- exact overlap (normalized instruction or test_final_state.py hash) with the TMax SFT environments.

It then runs pool P's DataKit decon (13-gram, overlap 0.5, >= 2 features) of the screened tasks against SWE-bench
Verified, Terminal-Bench 2.1, TBLite 2.0 and, as a near-duplicate check, the TMax SFT environments.

``select`` then takes a seeded sample stratified by ``metadata.domain`` from the screened, decon-clean tasks.

``package`` writes the sampled tasks as local Harbor task folders, unchanged except for ``TMUX_BAKE``.
"""

import argparse
import collections
import hashlib
import json
import random
import re
import tempfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.tmax_reserve.decon import decon_marks, eval_documents

RESERVE = ("TMaxxx/TMax-15K-Harbor", "48a77eb0")
SFT_ENVS = ("allenai/TMax-SFT-16.5K", "848e218270672080a92a94791c028ee43ae33387")
ABSOLUTE_PATH = re.compile(r"(/(?:app|opt|home|root|srv|var|usr/local|etc|data|tmp)/[A-Za-z0-9_./-]+)")
ANSWER_PATH = re.compile(
    r"/[A-Za-z0-9_./-]*(?:truth|expected|oracle|solution|verifier|answer|reference)[A-Za-z0-9_./-]*", re.IGNORECASE
)
BUILD_FILES = ("environment/Dockerfile", "environment/post_install.sh", "environment/base_install.sh")
ORACLE_FIXTURE = "fixtures/oracle"
# The only change to a task: Terminus-2 installs tmux at agent setup when the image lacks it (no reserve image has
# it), and the pilot's sandboxes have no general egress. Eval sandboxes install tmux at setup instead.
TMUX_BAKE = "RUN apt-get update && apt-get install -y --no-install-recommends tmux && rm -rf /var/lib/apt/lists/*"


def hub_zip(repo: str, revision: str, dest: Path) -> zipfile.ZipFile:
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/tasks.zip"
    urllib.request.urlretrieve(url, dest)
    return zipfile.ZipFile(dest)


def _normalized_hash(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().lower().encode()).hexdigest()


def _names_path(path: str, instruction: str) -> bool:
    return path in instruction or path.rstrip("/").rsplit("/", 1)[-1] in instruction


def task_files(archive: zipfile.ZipFile) -> dict[str, set[str]]:
    files: dict[str, set[str]] = collections.defaultdict(set)
    for name in archive.namelist():
        if not name.endswith("/"):
            task, rel = name.split("/", 1)
            files[task].add(rel)
    return files


def hidden_assets(task: str, files: set[str], archive: zipfile.ZipFile) -> dict[str, list[str]]:
    """The hidden paths each rule finds for one Harbor-layout task."""

    def read(rel: str) -> str:
        return archive.read(f"{task}/{rel}").decode(errors="ignore") if rel in files else ""

    tests, instruction = read("tests/test_final_state.py"), read("instruction.md")
    build = "".join(read(rel) for rel in BUILD_FILES)
    fixtures = {"/" + rel.split("_fixtures/", 1)[1] for rel in files if "_fixtures/" in rel}
    provided = fixtures | set(ABSOLUTE_PATH.findall(build))
    image_path = sorted(
        path
        for path in set(ABSOLUTE_PATH.findall(tests))
        if any(path == p or p.startswith(path.rstrip("/") + "/") for p in provided)
        and not _names_path(path, instruction)
    )
    created = {path.rstrip("/.") for path in ANSWER_PATH.findall(build + "\n" + "\n".join(fixtures))}
    answer_name = sorted(
        path for path in created if len(path) > 6 and path in tests and not _names_path(path, instruction)
    )
    return {"image_path": image_path, "answer_name": answer_name}


def sft_hashes(archive: zipfile.ZipFile) -> set[str]:
    hashes = set()
    for name in archive.namelist():
        if name.endswith("/task.json"):
            hashes.add(_normalized_hash(json.loads(archive.read(name))["description"]))
        elif name.endswith("/test_final_state.py") and name.count("/") == 1:
            hashes.add(hashlib.sha256(archive.read(name)).hexdigest())
    return hashes


def screen(reserve: zipfile.ZipFile, sft: zipfile.ZipFile) -> list[dict]:
    known = sft_hashes(sft)
    rows = []
    for task, files in sorted(task_files(reserve).items()):
        instruction = reserve.read(f"{task}/instruction.md").decode(errors="ignore")
        tests = reserve.read(f"{task}/tests/test_final_state.py")
        metadata = tomllib.loads(reserve.read(f"{task}/task.toml").decode())["metadata"]
        hidden = hidden_assets(task, files, reserve)
        reasons = [rule for rule, paths in hidden.items() if paths]
        if any(rel.endswith(ORACLE_FIXTURE) for rel in files):
            reasons.append("oracle_fixture")
        if _normalized_hash(instruction) in known or hashlib.sha256(tests).hexdigest() in known:
            reasons.append("sft_overlap")
        rows.append(
            {
                "task": task,
                "domain": metadata["domain"],
                "excluded": reasons,
                **{f"hidden_{k}": v for k, v in hidden.items()},
            }
        )
    return rows


def tmax_docs(archive: zipfile.ZipFile, tasks: list[str], layout: str) -> list[dict]:
    """Pool P's TMax decon document: the instruction plus test_final_state.py."""
    if layout == "harbor":
        return [
            {
                "id": task,
                "text": (
                    archive.read(f"{task}/instruction.md").decode(errors="ignore")
                    + "\n\n"
                    + archive.read(f"{task}/tests/test_final_state.py").decode(errors="ignore")
                ),
            }
            for task in tasks
        ]
    return [
        {
            "id": f"tmax-sft/{task}",
            "text": (
                json.loads(archive.read(f"{task}/task.json"))["description"]
                + "\n\n"
                + archive.read(f"{task}/test_final_state.py").decode(errors="ignore")
            ),
        }
        for task in tasks
    ]


def draw(rows: list[dict], decon: dict[str, dict], size: int, seed: int) -> list[str]:
    """Seeded sample stratified by domain (largest remainder) from the screened, decon-clean tasks."""
    eligible: dict[str, list[str]] = collections.defaultdict(list)
    for row in rows:
        if not row["excluded"] and not decon[row["task"]]["contaminated"]:
            eligible[row["domain"]].append(row["task"])
    total = sum(len(v) for v in eligible.values())
    quotas = {d: size * len(v) / total for d, v in eligible.items()}
    counts = {d: int(q) for d, q in quotas.items()}
    for d in sorted(quotas, key=lambda d: (counts[d] - quotas[d], d))[: size - sum(counts.values())]:
        counts[d] += 1
    rng = random.Random(seed)
    picked = []
    for domain in sorted(eligible):
        picked += rng.sample(sorted(eligible[domain]), counts[domain])
    return sorted(picked)


def select(output: str, size: int, seed: int) -> None:
    out = StoragePath(output)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        reserve = hub_zip(*RESERVE, work / "reserve.zip")
        sft = hub_zip(*SFT_ENVS, work / "sft.zip")
        rows = screen(reserve, sft)
        clean = [row["task"] for row in rows if not row["excluded"]]
        evals = eval_documents(work)
        sft_tasks = sorted({name.split("/")[0] for name in sft.namelist() if name.endswith("/task.json")})
        evals["tmax-sft-2200"] = tmax_docs(sft, sft_tasks, "sft")
        decon = decon_marks(tmax_docs(reserve, clean, "harbor"), evals, work / "decon")
    picked = draw(rows, decon, size, seed)
    reasons = collections.Counter(reason for row in rows for reason in row["excluded"])
    summary = {
        "reserve": f"{RESERVE[0]}@{RESERVE[1]}",
        "sft_envs": f"{SFT_ENVS[0]}@{SFT_ENVS[1]}",
        "reserve_tasks": len(rows),
        "excluded_any": sum(bool(row["excluded"]) for row in rows),
        "excluded_by_reason": dict(reasons),
        "screened": len(clean),
        "decon_contaminated": sorted(t for t, d in decon.items() if d["contaminated"]),
        "decon_near": {t: d["max_overlap"] for t, d in decon.items() if d["max_overlap"] >= 0.2},
        "eval_documents": {name: len(docs) for name, docs in evals.items()},
        "seed": seed,
        "drawn": len(picked),
    }
    (out / "screen.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (out / "decon.json").write_text(json.dumps(decon, indent=1))
    (out / "draw.json").write_text(json.dumps({"tasks": picked, **summary}, indent=1))
    print(json.dumps(summary, indent=1))


def package(reserve: zipfile.ZipFile, tasks: list[str], dest: Path) -> dict:
    """Write the drawn tasks as Harbor task folders with tmux baked into each image (see TMUX_BAKE)."""
    dest.mkdir(parents=True)
    files = task_files(reserve)
    digests = {}
    for task in tasks:
        digest = hashlib.sha256()
        for rel in sorted(files[task]):
            data = reserve.read(f"{task}/{rel}")
            if rel == "environment/Dockerfile":
                data = data.rstrip(b"\n") + b"\n" + TMUX_BAKE.encode() + b"\n"
            path = dest / task / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            digest.update(rel.encode() + b"\0" + hashlib.sha256(data).digest())
        digests[task] = digest.hexdigest()
    manifest = {
        "reserve": f"{RESERVE[0]}@{RESERVE[1]}",
        "environment_change": TMUX_BAKE,
        "tasks": digests,
        "sha256": hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest(),
    }
    (dest.parent / f"{dest.name}.manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    select_parser = commands.add_parser("select", help="Screen, decon and draw; write outputs to --output.")
    select_parser.add_argument("--output", required=True, help="Directory (local or remote) for the outputs.")
    select_parser.add_argument("--size", type=int, default=200)
    select_parser.add_argument("--seed", type=int, default=20261113)
    package_parser = commands.add_parser("package", help="Write the drawn tasks as local Harbor task folders.")
    package_parser.add_argument("--draw", required=True, help="draw.json written by select (local or remote).")
    package_parser.add_argument("--dest", required=True, type=Path, help="New directory for the task folders.")
    args = parser.parse_args()
    if args.command == "select":
        select(args.output, args.size, args.seed)
        return
    tasks = json.loads(StoragePath(args.draw).read_text())["tasks"]
    with tempfile.TemporaryDirectory() as tmp:
        manifest = package(hub_zip(*RESERVE, Path(tmp) / "reserve.zip"), tasks, args.dest)
    print(json.dumps({"tasks": len(manifest["tasks"]), "sha256": manifest["sha256"]}))


if __name__ == "__main__":
    main()
