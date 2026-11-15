# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-derive acceptance from source, model and execution receipts."""

import hashlib
import json
import tempfile
from pathlib import Path

import jsonschema

from experiments.tool_normalization.candidate import (
    assembled_messages,
    calls_for_candidate,
    has_source_calls,
    source_messages,
)
from experiments.tool_normalization.eligibility_check import eligibility
from experiments.tool_normalization.historical_projection import historical_projection
from experiments.tool_normalization.native_seed import native_seed
from experiments.tool_normalization.review_encoding import ENCODING_INSTRUCTIONS, FORMAT, decode_payload
from experiments.tool_normalization.review_real import (
    AUDIT,
    MAPPING_SCHEMA,
    PROPOSE,
    REVIEW,
    REVIEW_SCHEMA,
    proposal_payload,
    proposal_seed,
)
from experiments.tool_normalization.rules import approved, rule_record
from experiments.tool_normalization.serialize import export


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def model_value(directory: Path, schema: dict) -> dict:
    receipt = read(directory / "receipt.json")
    if receipt["finish_reason"] != "stop" or not receipt["done"] or receipt["lease_state"] != "terminal":
        raise ValueError("Nonterminal model receipt")
    for name, key in (("request.json", "request_sha256"), ("response.sse", "raw_sha256")):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != receipt[key]:
            raise ValueError("Model receipt hash mismatch")
    value = read(directory / "value.json")
    streamed = ""
    for line in (directory / "response.sse").read_text().splitlines():
        if not line.startswith("data:") or line.removeprefix("data:").strip() == "[DONE]":
            continue
        event = json.loads(line.removeprefix("data:").strip())
        streamed += "".join(choice.get("delta", {}).get("content") or "" for choice in event.get("choices", []))
    if streamed != receipt["content"]:
        raise ValueError("Model receipt content differs from raw response")
    if json.loads(receipt["content"]) != value:
        raise ValueError("Stored model value differs from receipt")
    jsonschema.validate(value, schema)
    return value


def verify_adapter(directory: Path, cfg: dict) -> dict:
    """Reconstruct a mechanical conversion and bind both reviews to its exact source and export."""
    ref = read(directory / "source-reference.json")
    source_path = Path(ref["path"])
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != ref["sha256"]:
        raise ValueError("Source reference changed")
    source = read(source_path)
    if source["id"] != directory.name or hashlib.sha256(source["text"].encode()).hexdigest() != source["text_sha256"]:
        raise ValueError("Source identity changed")
    if source["source"].startswith("open_swe_traces/") and eligibility(source) != "eligible":
        raise ValueError("Open-SWE source is not eligible")
    environment = read(Path(__file__).with_name("environments.json"))[cfg["execution"]["image"]]
    original = source_messages(source["text"])
    seed = native_seed(original, environment["packages"])
    proposed = seed is None
    if proposed:
        proposal = model_value(directory / "proposal", MAPPING_SCHEMA)
        seed = proposal_seed(original, proposal)
        request = read(directory / "proposal/request.json")
        encoded = json.loads(request["messages"][1]["content"])
        expected_prompt = PROPOSE + ENCODING_INSTRUCTIONS if encoded.get("encoding") == FORMAT else PROPOSE
        schemas = read(Path(__file__).with_name("opencode-tools.json"))
        if decode_payload(encoded) != proposal_payload(original, schemas, environment):
            raise ValueError("Proposal used different source or target schemas")
        if request["messages"][0]["content"] != expected_prompt:
            raise ValueError("Proposal prompt changed")
    candidate, observed, lineage = historical_projection(original, seed)
    schemas = read(Path(__file__).with_name("opencode-tools.json"))
    calls = calls_for_candidate(original, candidate, schemas)
    messages = assembled_messages(original, candidate, observed)
    for filename, expected in (
        ("candidate.json", candidate),
        ("messages.json", messages),
        ("observation-lineage.json", lineage),
    ):
        if read(directory / filename) != expected:
            raise ValueError(f"Reconstructed {filename} differs")
    review = seed if proposed else None
    needs_review = bool(calls) or has_source_calls(original)
    rules_path = directory / "rules.json"
    if rules_path.exists():
        expected_rules = rule_record(original, environment["packages"])
        if read(rules_path) != expected_rules or not approved(expected_rules, cfg):
            raise ValueError("Rule conversion lacks matching audited approval")
        needs_review = False
    if not needs_review and any((directory / stage).exists() for stage in ("review", "audit")):
        raise ValueError("Tool-free structural normalization unexpectedly has model receipts")
    stages = (("audit", AUDIT),) if proposed else (("review", REVIEW), ("audit", AUDIT))
    if not needs_review:
        stages = ()
    for stage, prompt in stages:
        verdict = model_value(directory / stage, REVIEW_SCHEMA)
        if verdict["decision"] != "pass":
            raise ValueError("Review did not pass")
        request = read(directory / stage / "request.json")
        encoded = json.loads(request["messages"][1]["content"])
        payload = decode_payload(encoded)
        expected_prompt = prompt + ENCODING_INSTRUCTIONS if encoded.get("encoding") == FORMAT else prompt
        expected = {
            "original": original,
            "target_tool_schemas": schemas,
            "candidate": messages,
            "observation_provenance": lineage,
            "adapter_test_environment": {"image": cfg["execution"]["image"], **environment},
            "validation_scope": (
                "isolated adapter tests; historical source outputs retained, original task not reexecuted"
            ),
        }
        if stage == "audit":
            expected["first_review"] = review
        if payload != expected or request["messages"][0]["content"] != expected_prompt:
            raise ValueError("Model reviewed different inputs")
        review = verdict
    with tempfile.TemporaryDirectory() as temp:
        expected_export = export(messages, schemas, Path(cfg["sft_tokenizer"]), Path(temp) / "export")
    actual = read(directory / "export/manifest.json")
    if actual != expected_export:
        raise ValueError("Export differs from reconstructed candidate")
    for name, expected_hash in actual["files"].items():
        if hashlib.sha256((directory / "export" / name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Export file changed: {name}")
    return {"tokens": actual["tokens"], "historical_results": len(lineage), "original_task_reexecuted": False}
