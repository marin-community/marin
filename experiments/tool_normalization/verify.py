# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-derive acceptance from source, model and execution receipts."""

import hashlib
import json
import tempfile
from pathlib import Path

import jsonschema

from experiments.tool_normalization.candidate import (
    CANDIDATE_SCHEMA,
    assembled_messages,
    calls_for_candidate,
    has_source_calls,
    source_messages,
)
from experiments.tool_normalization.eligibility_check import eligibility
from experiments.tool_normalization.fixture_checks import fixture_verification_calls, model_environment
from experiments.tool_normalization.historical_projection import historical_projection
from experiments.tool_normalization.native_seed import EDIT_SCHEMA, apply_edits, native_seed
from experiments.tool_normalization.native_tools import observations
from experiments.tool_normalization.observation_review import observation_alignment, validate_observation_review
from experiments.tool_normalization.pipeline import AUDIT_SCHEMA, verification_calls
from experiments.tool_normalization.replay_review import REPLAY_SCHEMA, replay_payload, validate_replay_review
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


def execution_messages(directory: Path, calls: list[dict]) -> list[dict]:
    receipt = read(directory / "receipt.json")
    if receipt["exit_code"] != 0 or receipt["stdout_truncated"] or receipt["stderr_truncated"]:
        raise ValueError("Incomplete execution receipt")
    for name, key in (("stdout.json", "stdout_sha256"), ("stderr.txt", "stderr_sha256")):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != receipt[key]:
            raise ValueError("Execution receipt hash mismatch")
    if read(directory / "calls.json") != calls:
        raise ValueError("Executed calls differ from candidate")
    report = read(directory / "stdout.json")
    if report["exit_code"] != 0 or report["dispatch_errors"]:
        raise ValueError("Native dispatch failed")
    return observations(calls, report["tool_events"])


def verify(directory: Path, sft_tokenizer: Path) -> dict:
    seal = read(directory / "seal.json")
    for name, expected in seal.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Sealed artifact changed: {name}")
    if read(directory / "result.json")["status"] != "accepted":
        raise ValueError("Record was not accepted")
    source_ref = read(directory / "source-reference.json")
    source_file = Path(source_ref["path"])
    if hashlib.sha256(source_file.read_bytes()).hexdigest() != source_ref["sha256"]:
        raise ValueError("Source reference changed")
    source = read(source_file)
    if source["source"].startswith("open_swe_traces/"):
        if eligibility(source) != "eligible" or read(directory / "eligibility.json") != source["eligibility"]:
            raise ValueError("Open-SWE source is not eligible")
    original = source_messages(source["text"])
    stages = sorted(path for path in directory.glob("attempt-*") if (path / "replay").exists())
    if len(stages) != 1:
        raise ValueError("No unique final replay")
    stage = stages[0]
    schemas = read(Path(__file__).with_name("opencode-tools.json"))
    requested = json.loads(read(stage / "convert/request.json")["messages"][1]["content"])
    if requested["source"] != [dict(index=index, **message) for index, message in enumerate(original)]:
        raise ValueError("Conversion request used a different source")
    environment = read(directory / "environment.json")
    if requested["environment_manifest"] != model_environment(environment):
        raise ValueError("Conversion request used a different environment")
    seed = native_seed(original, environment["packages"])
    if seed is not None:
        if requested["seed"] != seed:
            raise ValueError("Prepared candidate differs from exact source mapping")
        candidate = apply_edits(seed, model_value(stage / "convert", EDIT_SCHEMA))
    else:
        candidate = model_value(stage / "convert", CANDIDATE_SCHEMA)
    if candidate != read(stage / "materialized-candidate.json"):
        raise ValueError("Materialized candidate differs from model proposal")
    for name in ("execution", "replay"):
        execution_config = read(stage / name / "manifest.json")["config"]
        if any(execution_config[key] != environment[key] for key in ("image", "workdir", "recorder_python")):
            raise ValueError("Execution used a different environment")
    calls = calls_for_candidate(original, candidate, schemas)
    observed = execution_messages(stage / "execution", calls)
    messages = assembled_messages(original, candidate, observed)
    audit = model_value(stage / "audit", AUDIT_SCHEMA)
    audit_input = json.loads(read(stage / "audit/request.json")["messages"][1]["content"])
    if audit_input["original"] != original or audit_input["candidate"] != messages:
        raise ValueError("Audit reviewed a different source or candidate")
    alignment = observation_alignment(original, candidate)
    if audit_input["observation_alignment"] != alignment:
        raise ValueError("Audit observation alignment differs")
    validate_observation_review(alignment, audit["observation_reviews"])
    if (
        audit["decision"] != "pass"
        or audit["issues"]
        or not audit["requirements"]
        or not all(check["fulfilled"] for check in audit["requirements"])
    ):
        raise ValueError("Audit does not pass")
    checks = verification_calls(audit["verification_script"]) + fixture_verification_calls(source, environment)
    if (calls and not checks) or any(call["name"] != "bash" for call in checks):
        raise ValueError("Missing task verification")
    replay = execution_messages(stage / "replay", calls + checks)
    if replay[: len(calls)] != observed:
        comparison = replay_payload(calls, observed, replay[: len(calls)], messages)
        replay_audit = model_value(stage / "replay-audit", REPLAY_SCHEMA)
        supplied = json.loads(read(stage / "replay-audit/request.json")["messages"][1]["content"])
        if supplied != comparison:
            raise ValueError("Replay audit reviewed different evidence")
        validate_replay_review(comparison, replay_audit)
    events = read(stage / "replay/stdout.json")["tool_events"][len(calls) :]
    if any(
        event["part"]["state"]["status"] != "completed" or event["part"]["state"].get("metadata", {}).get("exit") != 0
        for event in events
    ):
        raise ValueError("Task verification failed")
    if messages != read(directory / "export/messages.json"):
        raise ValueError("Export differs from executed candidate")
    with tempfile.TemporaryDirectory() as temporary:
        rebuilt = Path(temporary) / "export"
        export(messages, schemas, sft_tokenizer, rebuilt)
        for name in ("text.txt", "input_ids.i32", "loss_mask.u8", "tools.json"):
            if (rebuilt / name).read_bytes() != (directory / "export" / name).read_bytes():
                raise ValueError(f"Export failed reconstruction: {name}")
    return {"verified": True, "calls": len(calls), "checks": len(checks), "files": len(seal)}


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
