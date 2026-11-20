# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate archived Open-SWE outcome evidence before dispatch and publication."""

import hashlib
import json
from itertools import pairwise
from pathlib import Path

import xxhash

from experiments.tool_normalization.eligibility import BOS, EOS_SUFFIX


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
    bos, eos = BOS, EOS_SUFFIX
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
