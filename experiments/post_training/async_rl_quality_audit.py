# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "wandb==0.26.0", "s3fs==2026.1.0", "hydra-core==1.4.0.dev1",
#   "omegaconf==2.4.0.dev15", "PyYAML==6.0.3", "tokenizers==0.22.2",
# ]
# ///

"""Qualify supplemental answer extraction on retained development responses.

Run in a CPU job in the artifacts' region, from the repository root:
PYTHONPATH=. uv run --locked --script experiments/post_training/async_rl_quality_audit.py --spec spec.json
The JSON specification
contains historical run specs accepted by async_rl_audit, matching prior dump
proofs, and an output_prefix for bounded adjudication artifacts. Only 128-row
development evaluations are accepted. No model generation or reward changes.
An optional exclude_response_text_sha256 list excludes previously adjudicated
full-turn UTF-8 texts before endpoint-balanced sampling, including duplicates.
"""

import argparse
import collections
import hashlib
import json
import posixpath
import random
from dataclasses import asdict
from urllib.request import urlopen

from tokenizers import Tokenizer

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.async_rl_quality import QUALITY_VERSION, extract_numeric_answer, normalize_numeric_answer

MAX_TOKENIZER_BYTES = 25 * 1024 * 1024
MAX_ADJUDICATION_BYTES = 1024 * 1024
MAX_ADJUDICATION_ROWS = 48
MAX_ADJUDICATION_EXCLUSIONS = 6 * 2 * 128
REGIONAL_PREFIX = "s3://marin-us-east-02a/"
EOS_MARKERS = {"<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"}
THINKING_MARKERS = {"<think>", "</think>", "<|start_think|>", "<|end_think|>"}


def assistant_turn_text(decoder: Tokenizer, prompt_tokens: list[int], tokens: list[int], *, thinking: bool) -> str:
    """Show the full generated turn, including its actual inherited thinking prefix.

    The reader sees all generated reasoning and delimiters, independently of the
    extractor's boundary decision. No question or other prompt content is copied.
    """
    prefix = []
    start = decoder.token_to_id("<|start_think|>") if thinking else None
    if start is not None and start in prompt_tokens:
        last_start = len(prompt_tokens) - 1 - prompt_tokens[::-1].index(start)
        if not decoder.decode(prompt_tokens[last_start + 1 :], skip_special_tokens=False).strip():
            prefix = prompt_tokens[last_start:]
    return decoder.decode(prefix + tokens, skip_special_tokens=False)


def final_assistant_segment(
    decoder: Tokenizer, tokens: list[int], *, thinking: bool, prompt_tokens: list[int] | None = None
) -> tuple[str | None, str]:
    """Use actual token boundaries, leaving incomplete or repeated thinking unresolved."""
    if thinking:
        end = decoder.token_to_id("<|end_think|>")
        start = decoder.token_to_id("<|start_think|>")
        audit.check(end is not None and start is not None, "Thinking tokenizer has no declared boundary tokens")
        if not prompt_tokens or start not in prompt_tokens:
            return None, "missing_thinking_prompt"
        last_start = len(prompt_tokens) - 1 - prompt_tokens[::-1].index(start)
        if decoder.decode(prompt_tokens[last_start + 1 :], skip_special_tokens=False).strip():
            return None, "invalid_thinking_prompt"
        if start in tokens:
            return None, "unexpected_thinking_start"
        count = tokens.count(end)
        if count != 1:
            return None, "missing_thinking_end" if count == 0 else "multiple_thinking_ends"
        tokens = tokens[tokens.index(end) + 1 :]
    else:
        markers = {decoder.token_to_id(text) for text in THINKING_MARKERS} - {None}
        if any(token in markers for token in tokens):
            return None, "unexpected_thinking_tokens"
    # Retain role delimiters for the extractor's continuation check. EOS tokens
    # can occur after a legitimate bare numeric final line and are removed alone.
    while tokens and (
        decoder.id_to_token(tokens[-1]) in EOS_MARKERS
        or decoder.decode(tokens[-1:], skip_special_tokens=False).isspace()
    ):
        tokens = tokens[:-1]
    forbidden = EOS_MARKERS | THINKING_MARKERS | {"<|start_header_id|>", "<|end_header_id|>", "<|im_start|>"}
    if any(decoder.id_to_token(token) in forbidden for token in tokens):
        return None, "role_or_thinking_continuation"
    return decoder.decode(tokens, skip_special_tokens=False), "resolved"


def load_decoder(model: dict) -> tuple[Tokenizer, str]:
    revision = model["tokenizer_revision"]
    audit.check(len(revision) == 40 and all(c in "0123456789abcdef" for c in revision), "Tokenizer must be immutable")
    uri = f"https://huggingface.co/{model['tokenizer_uri']}/resolve/{revision}/tokenizer.json"
    with urlopen(uri, timeout=60) as source:
        data = source.read(MAX_TOKENIZER_BYTES + 1)
    audit.check(len(data) <= MAX_TOKENIZER_BYTES, "Tokenizer exceeds bound")
    return Tokenizer.from_str(data.decode()), hashlib.sha256(data).hexdigest()


def write_json(uri: str, value: dict) -> None:
    data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
    audit.check(len(data) <= MAX_ADJUDICATION_BYTES, "Quality artifact exceeds bound")
    fs, path = audit.fs_path(uri)
    with fs.open(path, "xb") as output:
        output.write(data)


def development_manifest(request: dict) -> dict:
    """Prove the source window before any response rows can be read."""
    data = request["validation_data"]
    audit.check(len(data) == 1 and data[0]["relative_path"] == "validation.parquet", "Unexpected validation dataset")
    audit.check(data[0]["uri"].startswith(REGIONAL_PREFIX), "Unqualified validation region")
    manifest = audit.read_json(posixpath.join(data[0]["uri"], "selection.json"))
    audit.check(manifest["dataset"] == "openai/gsm8k", "Expected GSM8K development data")
    audit.check(not manifest.get("validation_window"), "Do not read a locked validation window")
    audit.check(manifest["rows"]["test"] == [f"test/{index}" for index in range(128)], "Expected exactly test[0:128]")
    return manifest


def select_adjudication(candidates: dict, seed: int) -> tuple[list[dict], int]:
    """Balance endpoints, then categories, retaining whole cases within the byte bound."""
    rng = random.Random(seed)
    queues = {}
    for endpoint, categories in sorted(candidates.items()):
        strata = sorted(categories)
        rng.shuffle(strata)
        for rows in categories.values():
            rng.shuffle(rows)
        queues[endpoint] = collections.deque(
            categories[stratum][index] for index in range(128) for stratum in strata if index < len(categories[stratum])
        )
    endpoints = sorted(queues)
    rng.shuffle(endpoints)
    selected, excluded_for_bytes = [], 0
    while any(queues.values()) and len(selected) < MAX_ADJUDICATION_ROWS:
        for endpoint in endpoints:
            while queues[endpoint] and len(selected) < MAX_ADJUDICATION_ROWS:
                row = {**queues[endpoint].popleft(), "blind_id": "case-0000"}
                payload = {"quality_version": QUALITY_VERSION, "rows": [*selected, row]}
                if len(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()) > MAX_ADJUDICATION_BYTES:
                    excluded_for_bytes += 1
                    continue
                selected.append(row)
                break
    rng.shuffle(selected)
    for index, row in enumerate(selected, start=1):
        row["blind_id"] = f"case-{index:04d}"
    return selected, excluded_for_bytes


def qualify(specification: dict) -> dict:
    """Recheck dump identity, reduce candidate metrics and retain a blinded sample."""
    runs = specification["runs"]
    audit.check(0 < len(runs) <= 6, "Qualify one to six development runs")
    prefix = specification["output_prefix"].rstrip("/")
    audit.check(prefix.startswith(REGIONAL_PREFIX), "Unqualified output region")
    audit.check(type(specification["sample_seed"]) is int, "Declare an integer sample seed")
    excluded_hashes = specification.get("exclude_response_text_sha256", [])
    audit.check(
        isinstance(excluded_hashes, list)
        and len(excluded_hashes) <= MAX_ADJUDICATION_EXCLUSIONS
        and all(
            isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
            for value in excluded_hashes
        ),
        "Expected bounded SHA256 response-text exclusions within the development population",
    )
    excluded_hashes = set(excluded_hashes)
    excluded_previous = collections.Counter()
    for name in ("adjudication-blind.json", "adjudication-key.json", "summary.json"):
        fs, path = audit.fs_path(prefix + "/" + name)
        audit.check(not fs.exists(path), "Qualification output already exists; choose a new candidate prefix")
    labels = [run["label"] for run in runs]
    audit.check(len(set(labels)) == len(labels), "Duplicate run labels")
    results, candidates, decoder_cache = {}, collections.defaultdict(lambda: collections.defaultdict(list)), {}
    for run in runs:
        audit.check(run["expected_eval_rows"] == 128, "Quality qualification uses development128 only")
        audit.check(not run.get("locked_validation"), "Do not tune extraction on held-out responses")
        audit.check(
            run["quality_steps"] == [0, run["expected_steps"]] and run["expected_steps"] > 0,
            "Qualify initial/final endpoints only",
        )
        if "envelope_uri" in run:
            audit.check(run["envelope_uri"].startswith(REGIONAL_PREFIX), "Unqualified envelope region")
            envelope = audit.read_json(run["envelope_uri"])
        else:
            envelope = run["envelope"]
        locator = envelope["request"]
        audit.check(
            all(uri.startswith(REGIONAL_PREFIX) for uri in locator["output"].values()), "Unqualified input region"
        )
        manifest = development_manifest(locator)
        request, resolved, storage = audit.audit_storage(run)
        audit.check(storage["clean_end_to_end"], "Qualification requires a clean historical run")
        root = request["output"]["export_root"]
        audit.check(root.startswith(REGIONAL_PREFIX), "Unqualified input region")
        model = request["model"]
        model_key = (model["tokenizer_uri"], model["tokenizer_revision"])
        if model_key not in decoder_cache:
            decoder_cache[model_key] = load_decoder(model)
        decoder, tokenizer_sha = decoder_cache[model_key]
        thinking = run["thinking"]
        audit.check(type(thinking) is bool, "Declare thinking mode explicitly")
        resolved_args = audit.parse_hydra_args(resolved["hydra_args"])
        if not thinking:
            audit.check(
                audit.lookup(resolved_args, "generator.chat_template_kwargs.enable_thinking") is False,
                "Nonthinking qualification requires the actual nonthinking prompt contract",
            )
        outputs = []
        for step in run["quality_steps"]:
            audit.check(step in run["expected_eval_steps"], "Undeclared evaluation step")
            expected = run["prior_dump_proofs"][str(step)]
            proof, ordered = audit.audit_eval_dump(root, step, 128, 1, expected["metrics"], True)
            for key in ("ordered_prompt_sha256", "ordered_response_sha256", "ordered_result_sha256"):
                audit.check(proof[key] == expected[key], f"Historical proof differs: {key}")
            counts = collections.Counter()
            seen = set()
            byte_count = 0
            uri = posixpath.join(root, "dumped_evals", f"global_step_{step}_evals")
            for entry in sorted(audit.list_entries(uri), key=lambda item: item["name"]):
                name = posixpath.basename(entry["name"])
                if name == "aggregated_results.jsonl":
                    continue
                audit.check(entry["type"] == "file" and name.endswith(".jsonl"), "Changed dump inventory")
                fs, path = audit.fs_path(posixpath.join(uri, name))
                with fs.open(path, "rb") as source:
                    for line in audit.bounded_lines(source):
                        byte_count += len(line)
                        audit.check(byte_count <= audit.MAX_EVAL_BYTES, "Quality pass exceeds byte bound")
                        row = json.loads(line)
                        ordinal = row["row_ordinal"]
                        audit.check(
                            type(ordinal) is int and 0 <= ordinal < 128 and ordinal not in seen,
                            "Duplicate/invalid response in quality pass",
                        )
                        seen.add(ordinal)
                        tokens = row["response_ids"]
                        audit.check(audit.canonical_sha(tokens) == row["response_ids_sha256"], "Changed response tokens")
                        audit.check(
                            decoder.decode(tokens, skip_special_tokens=False) == row["output_response"], "Decode differs"
                        )
                        raw = sum(row["score"]) if isinstance(row["score"], list) else row["score"]
                        audit.check(
                            [
                                row["uid"],
                                row["prompt_token_ids_sha256"],
                                row["response_ids_sha256"],
                                raw,
                                row["stop_reason"],
                            ]
                            == ordered[ordinal][:5],
                            "Quality pass differs from verified dump",
                        )
                        audit.check(
                            audit.canonical_sha(row["prompt_token_ids"]) == row["prompt_token_ids_sha256"],
                            "Changed prompt tokens",
                        )
                        segment, boundary = final_assistant_segment(
                            decoder, tokens, thinking=thinking, prompt_tokens=row["prompt_token_ids"]
                        )
                        answer = extract_numeric_answer(segment) if segment is not None else None
                        # Extraction is complete before the reference answer is consulted.
                        reference = normalize_numeric_answer(row["env_extras"]["reward_spec"]["ground_truth"])
                        audit.check(reference is not None, "Unsupported reference numeric format")
                        status = answer.status.value if answer else boundary
                        correct = bool(answer is not None and answer.value is not None and answer.value == reference)
                        stopped = row["stop_reason"] in {"stop", "complete", "end_turn", "eos"}
                        audit.check(raw in (0, 1), "Expected binary GSM8K reward")
                        counts.update(
                            {
                                "rows": 1,
                                "raw_correct": int(raw),
                                "extracted_correct": int(correct),
                                "extracted_correct_and_stop": int(correct and stopped),
                                "boundary/" + boundary: 1,
                                "status/" + status: 1,
                            }
                        )
                        # Stratify by endpoint, extraction status and stop class, but hide
                        # those labels, arm identity, prediction and gold from the reader.
                        text = assistant_turn_text(decoder, row["prompt_token_ids"], tokens, thinking=thinking)
                        text_sha = hashlib.sha256(text.encode()).hexdigest()
                        if text_sha in excluded_hashes:
                            excluded_previous[f"{run['label']}/{step}"] += 1
                            continue
                        candidates[(run["label"], step)][(status, stopped)].append(
                            {
                                "text": text,
                                "response_text_sha256": text_sha,
                                "response_ids_sha256": row["response_ids_sha256"],
                                "boundary_resolved": segment is not None,
                                "label": run["label"],
                                "step": step,
                                "ordinal": ordinal,
                                "prediction": asdict(answer) if answer else None,
                                "reference": reference,
                                "raw_reward": raw,
                                "stop_reason": row["stop_reason"],
                                "row_sha256": audit.canonical_sha(row),
                            }
                        )
            audit.check(seen == set(range(128)), "Quality row coverage differs")
            audit.check(counts["raw_correct"] == proof["metrics"]["eval/all/avg_score"] * 128, "Raw reduction changed")
            outputs.append({"step": step, "counts": dict(counts), "proof": expected})
        results[run["label"]] = {
            "attempt_id": request["attempt_id"],
            "tokenizer_sha256": tokenizer_sha,
            "selection_sha256": audit.canonical_sha(manifest),
            "evaluations": outputs,
        }
    selected, excluded_for_bytes = select_adjudication(candidates, specification["sample_seed"])
    blind = [{key: row[key] for key in ("blind_id", "text")} for row in selected]
    write_json(prefix + "/adjudication-blind.json", {"quality_version": QUALITY_VERSION, "rows": blind})
    write_json(prefix + "/adjudication-key.json", {"quality_version": QUALITY_VERSION, "rows": selected})
    result = {
        "quality_version": QUALITY_VERSION,
        "qualification_complete": False,
        "runs": results,
        "sample_seed": specification["sample_seed"],
        "sample_rows": len(blind),
        "output_prefix": prefix,
        "sample_candidates_excluded_for_bytes": excluded_for_bytes,
        "exclude_response_text_sha256": sorted(excluded_hashes),
        "sample_candidates_excluded_previous_by_endpoint": dict(excluded_previous),
        "sample_unique_texts": len({row["text"] for row in blind}),
        "scope": "Retrospective development diagnostics; independent blinded adjudication pending",
    }
    write_json(prefix + "/summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    result = qualify(audit.read_json(args.spec))
    print("ASYNC_RL_QUALITY_AUDIT_JSON " + json.dumps(result, allow_nan=False), flush=True)
    print("ASYNC_RL_QUALITY_CAPTURE_PASS", flush=True)


if __name__ == "__main__":
    main()
