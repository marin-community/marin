# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Commit complete candidate outputs to object storage before evaluating the next."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for

from experiments.weight_merging.playground.cloud_io import require_coreweave

REFERENCE_FIELDS = (
    "panel_sha256",
    "models",
    "torch",
    "transformers",
    "gpu",
    "vllm",
    "max_num_seqs",
    "dtype",
    "defer_code_scoring",
    "max_new_tokens",
    "max_model_len",
    "decode_seeds",
    "prompt_policy",
    "source_sha256",
)
RUN_FILES = ("panel.json", "model-paths.json", "model-sources.json", "run-metadata.json", "reload-checks.json")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_candidate(value: dict) -> dict:
    recipe = value.get("recipe")
    if recipe is not None:
        recipe = {
            "method": recipe["method"],
            "coefficients": recipe["coefficients"],
            **{
                key: recipe.get(key, default)
                for key, default in (("density", 1.0), ("scale", 1.0), ("seed", 42), ("ties_density", 1.0))
            },
        }
    return {
        "name": value["name"],
        "source": value["source"],
        "donors": list(value.get("donors", [])),
        "recipe": recipe,
        "rope_theta": value.get("rope_theta", 1_000_000),
        "donor_precision": value.get("donor_precision", "float32"),
        "output_precision": value.get("output_precision", "float32"),
    }


class CandidateStore:
    def __init__(self, remote: str, reference: dict, candidates: list[dict]):
        if remote.startswith("s3://"):
            require_coreweave(remote)
        self.fs, self.prefix = filesystem_for(remote)
        self.prefix = self.prefix.rstrip("/")
        self.reference = reference
        self.candidates = {row["name"]: normalized_candidate(row) for row in candidates}
        if len(self.candidates) != len(candidates):
            raise ValueError("Duplicate candidate names")
        if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in self.candidates):
            raise ValueError("Candidate name is not a safe object key")

    def contract(self, name: str) -> dict:
        return {**{key: self.reference[key] for key in REFERENCE_FIELDS}, "candidate": self.candidates[name]}

    def initialize(self) -> None:
        path = self.prefix + "/contract.json"
        value = {"reference": {key: self.reference[key] for key in REFERENCE_FIELDS}, "candidates": self.candidates}
        if self.fs.exists(path):
            if json.loads(self.fs.cat(path)) != value:
                raise ValueError("Durable phase contract changed")
        else:
            self.fs.makedirs(self.prefix, exist_ok=True)
            self.fs.pipe_file(path, (json.dumps(value, sort_keys=True) + "\n").encode())

    def done(self, name: str) -> bool:
        prefix = self.prefix + "/candidates/" + name
        path = prefix + "/done.json"
        if not self.fs.exists(path):
            return False
        marker = json.loads(self.fs.cat(path))
        if marker["contract"] != self.contract(name):
            raise ValueError(f"Candidate contract changed: {name}")
        for filename, expected in marker["files"].items():
            data = self.fs.cat(prefix + "/run/" + filename)
            if len(data) != expected["bytes"] or digest(data) != expected["sha256"]:
                raise ValueError(f"Durable candidate object changed: {name}/{filename}")
        return True

    def publish(self, run: Path, name: str) -> None:
        if self.done(name):
            return
        metadata = json.loads((run / "run-metadata.json").read_text())
        candidate = next(value for value in metadata["candidates"] if value["name"] == name)
        actual = {**{key: metadata[key] for key in REFERENCE_FIELDS}, "candidate": normalized_candidate(candidate)}
        if actual != self.contract(name):
            raise ValueError(f"Generation contract differs from frozen reference: {name}")
        panel = json.loads((run / "panel.json").read_text())
        if digest((run / "panel.json").read_bytes()) != metadata["panel_sha256"]:
            raise ValueError("Panel hash mismatch")
        modes = [("greedy", 0), *[("sample", seed) for seed in metadata["decode_seeds"]]]
        expected_keys = {(item["id"], mode, seed) for item in panel["items"] for mode, seed in modes}
        rows = [json.loads(line) for line in (run / "outputs.jsonl").read_text().splitlines()]
        selected = [row for row in rows if row["candidate"] == name]
        actual_keys = {(row["item_id"], row["mode"], row["seed"]) for row in selected}
        if actual_keys != expected_keys or len(selected) != len(expected_keys):
            raise ValueError(f"Candidate raw output is incomplete or duplicated: {name}")
        metadata["candidates"] = [candidate]
        payloads = {filename: (run / filename).read_bytes() for filename in RUN_FILES}
        payloads["run-metadata.json"] = (json.dumps(metadata, indent=2) + "\n").encode()
        payloads["outputs.jsonl"] = ("\n".join(json.dumps(row) for row in selected) + "\n").encode()
        prefix = self.prefix + "/candidates/" + name
        self.fs.makedirs(prefix + "/run", exist_ok=True)
        files = {}
        for filename, data in payloads.items():
            path = prefix + "/run/" + filename
            self.fs.pipe_file(path, data)
            self.fs.invalidate_cache(path)
            if digest(self.fs.cat(path)) != digest(data):
                raise ValueError(f"Publication readback differs: {name}/{filename}")
            files[filename] = {"bytes": len(data), "sha256": digest(data)}
        marker = {
            "contract": self.contract(name),
            "rows": len(selected),
            "files": files,
            "publisher_sha256": digest(Path(__file__).read_bytes()),
        }
        self.fs.pipe_file(prefix + "/done.json", (json.dumps(marker, indent=2) + "\n").encode())
        self.fs.invalidate_cache(prefix + "/done.json")
        if not self.done(name):
            raise ValueError(f"Completion marker was not published: {name}")
        print(f"Durable candidate committed: {name}, {len(selected)} rows", flush=True)

    def asset_done(self, name: str) -> bool:
        prefix = self.prefix + "/assets/" + name
        if not self.fs.exists(prefix + "/done.json"):
            return False
        marker = json.loads(self.fs.cat(prefix + "/done.json"))
        if marker["reference"] != {key: self.reference[key] for key in REFERENCE_FIELDS}:
            raise ValueError("Asset contract changed")
        if digest(self.fs.cat(prefix + "/result.json")) != marker["sha256"]:
            raise ValueError("Asset object changed")
        return True

    def publish_asset(self, path: Path, name: str) -> None:
        if self.asset_done(name):
            return
        prefix = self.prefix + "/assets/" + name
        data = path.read_bytes()
        self.fs.makedirs(prefix, exist_ok=True)
        self.fs.pipe_file(prefix + "/result.json", data)
        if digest(self.fs.cat(prefix + "/result.json")) != digest(data):
            raise ValueError("Asset readback differs")
        marker = {"reference": {key: self.reference[key] for key in REFERENCE_FIELDS}, "sha256": digest(data)}
        self.fs.pipe_file(prefix + "/done.json", (json.dumps(marker, indent=2) + "\n").encode())

    def collect(self, output: Path) -> None:
        output.mkdir(parents=True)
        completed = [name for name in self.candidates if self.done(name)]
        if not completed:
            raise ValueError("No complete durable candidates")
        source = self.prefix + "/candidates/" + completed[0] + "/run/"
        for filename in RUN_FILES:
            (output / filename).write_bytes(self.fs.cat(source + filename))
        metadata = json.loads((output / "run-metadata.json").read_text())
        metadata["candidates"] = [self.candidates[name] for name in completed]
        metadata["durable_source"] = self.prefix
        (output / "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        with (output / "outputs.jsonl").open("wb") as stream:
            for name in completed:
                stream.write(self.fs.cat(self.prefix + "/candidates/" + name + "/run/outputs.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("init", "check", "publish", "recover", "collect", "check-asset", "publish-asset")
    )
    parser.add_argument("--remote", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--recipes", type=Path, required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    store = CandidateStore(args.remote, reference, json.loads(args.recipes.read_text()))
    if args.action == "init":
        source_root = Path(__file__).resolve().parents[1]
        actual = {path.name: digest(path.read_bytes()) for path in source_root.glob("*.py")}
        if actual != reference["source_sha256"]:
            raise ValueError("Generation Python sources differ from the frozen bundle")
        store.initialize()
    elif args.action == "check":
        raise SystemExit(0 if store.done(args.candidate) else 3)
    elif args.action == "publish":
        store.publish(args.run, args.candidate)
    elif args.action == "check-asset":
        raise SystemExit(0 if store.asset_done(args.candidate) else 3)
    elif args.action == "publish-asset":
        store.publish_asset(args.run, args.candidate)
    elif args.action == "recover":
        panel = json.loads((args.run / "panel.json").read_text())
        expected = len(panel["items"]) * (1 + len(reference["decode_seeds"]))
        rows = [json.loads(line) for line in (args.run / "outputs.jsonl").read_text().splitlines()]
        for name in store.candidates:
            count = sum(row["candidate"] == name for row in rows)
            if count >= expected:
                store.publish(args.run, name)
            elif count:
                print(f"Incomplete recovered candidate retained only in snapshot: {name}, {count} rows", flush=True)
    else:
        store.collect(args.run)


if __name__ == "__main__":
    main()
