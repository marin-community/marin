# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate archived Open-SWE outcome evidence before dispatch and publication."""

import hashlib
import json
from itertools import pairwise
from pathlib import Path

import xxhash


def eligibility(source: dict) -> str:
    binding = source["eligibility"]
    proof_path = Path(binding["proof_path"])
    if hashlib.sha256(proof_path.read_bytes()).hexdigest() != binding["proof_sha256"]:
        raise ValueError("Eligibility proof changed")
    proof = json.loads(proof_path.read_text())
    original_path = Path(proof["source_file"])
    if hashlib.sha256(original_path.read_bytes()).hexdigest() != proof["source_file_sha256"]:
        raise ValueError("Eligibility source changed")
    original = json.loads(original_path.read_text())
    if source != {**original, "eligibility": binding} or proof["id"] != source["id"]:
        raise ValueError("Eligibility proof belongs to a different source")
    evidence = {}
    for name, expected in proof["evidence_files"].items():
        path = Path(name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Eligibility evidence changed")
        if path.name in evidence:
            raise ValueError("Duplicate eligibility evidence")
        evidence[path.name] = json.loads(path.read_text())
    chain = [evidence[name + ".json"] for name in ("store", "tokenized", "normalized", "rendered", "chat")]
    for parent, child in pairwise(chain):
        if parent["dep_paths"] != [child["output_path"]]:
            raise ValueError("Broken source artifact lineage")
    identity = proof["identity"]
    if identity != evidence[f"{source['id']}-identity.json"]:
        raise ValueError("Source identity receipt differs")
    text = source["text"]
    bos, eos = "<|begin_of_text|>", " <|end_of_text|>"
    if not text.startswith(bos) or not text.endswith(eos):
        raise ValueError("Unexpected source framing")
    if xxhash.xxh3_128_hexdigest(text[len(bos) : -len(eos)].encode()) != identity["identity"]["id"]:
        raise ValueError("Source content identity differs")
    if not identity["url"].startswith(chain[2]["output_path"] + "/outputs/main/"):
        raise ValueError("Identity outside normalized artifact")
    outcome = proof["outcome"]
    if outcome != evidence["matched-outcomes.json"][identity["identity"]["source_id"]]:
        raise ValueError("Outcome identity differs")
    if outcome["id"] != identity["identity"]["source_id"]:
        raise ValueError("Outcome is for another conversation")
    if not outcome["url"].startswith(chain[4]["output_path"] + "/outputs/main/"):
        raise ValueError("Outcome outside chat artifact")
    resolved = outcome["resolved"]
    if type(resolved) is not int or resolved not in (-1, 0, 1):
        raise ValueError("Unknown outcome encoding")
    disposition = {1: "eligible", 0: "excluded_unsuccessful", -1: "excluded_unknown"}[resolved]
    if disposition != proof["disposition"]:
        raise ValueError("Outcome disposition differs")
    return disposition


def bind_manifest(manifest: Path, report: Path, root: Path) -> Path:
    """Bind existing outcome proofs to source copies without changing archived inputs."""
    root.mkdir(parents=True, exist_ok=False)
    proofs = {row["id"]: Path(row["proof"]) for row in json.loads(report.read_text())["rows"]}
    rows = []
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        source = json.loads(Path(row["source_file"]).read_text())
        if source["id"] in proofs:
            proof = proofs[source["id"]]
            source["eligibility"] = {
                "proof_path": str(proof.resolve()),
                "proof_sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
            }
            eligibility(source)
            directory = root / source["id"]
            directory.mkdir(parents=True)
            path = directory / "source.json"
            path.write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")
            row = {**row, "source_file": str(path.resolve())}
        rows.append(row)
    output = root / "manifest.jsonl"
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return output
