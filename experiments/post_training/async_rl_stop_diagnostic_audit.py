# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import collections
import hashlib
import json
import math
import posixpath
import re
import statistics
import urllib.request
from pathlib import Path

import fsspec
import wandb
from tokenizers import Tokenizer

from experiments.post_training.async_rl_stop_diagnostics import parser_inside_thinking, summarize

SPEC = {}
OUTPUT = None
EOS_PROOF = None
MAX_BYTES = 200 * 1024 * 1024
byte_count = 0
PAT = re.compile(r"#### (\-?[0-9\.\,]+)")


def fs_path(uri):
    assert uri.startswith("s3://marin-us-east-02a/"), uri
    fs, _, paths = fsspec.get_fs_token_paths(
        uri, storage_options={"config_kwargs": {"s3": {"addressing_style": "virtual"}}}
    )
    return fs, paths[0]


def read_json(uri, limit=1024 * 1024, optional=False):
    fs, path = fs_path(uri)
    if optional and not fs.exists(path):
        return None
    with fs.open(path, "rb") as f:
        raw = f.read(limit + 1)
    assert len(raw) <= limit, "Metadata bound exceeded"
    return json.loads(raw)


def sha(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def quantiles(v):
    v = sorted(v)
    return (
        {
            "n": len(v),
            "mean": statistics.mean(v),
            "min": v[0],
            "p50": v[(len(v) - 1) // 2],
            "p90": v[math.ceil(0.9 * len(v)) - 1],
            "max": v[-1],
        }
        if v
        else None
    )


def clean(s):
    return re.sub(r"\s+", " ", s).strip()


def features(row, eos_ids=(), decoder=None):
    text = row["output_response"]
    tokens = row["response_ids"]
    matches = list(PAT.finditer(text))
    answers = [m.group(1).replace(",", "") for m in matches]
    grams = [tuple(tokens[i : i + 16]) for i in range(max(0, len(tokens) - 15))]
    lines = [clean(x) for x in text.splitlines() if len(clean(x)) >= 30]
    counts = collections.Counter(lines)
    end_think = [i for i, t in enumerate(tokens) if t == 128003]
    structure = {
        "end_think_count": len(end_think),
        "first_end_think_token_index": end_think[0] if end_think else None,
        "decoder_verified": False,
        "first_marker_region": "unverified",
        "post_end_think_marker_count": None,
        "first_marker_preceded_by_quote": bool(
            matches and matches[0].start() > 0 and text[matches[0].start() - 1] in chr(34) + chr(39)
        ),
    }
    if decoder is not None:
        rendered = decoder.decode(tokens, skip_special_tokens=False)
        assert rendered == text, "Pinned tokenizer decode differs from finalized output_response"
        structure["decoder_verified"] = True
        prefix = decoder.decode(tokens[: end_think[0] + 1], skip_special_tokens=False) if end_think else None
        assert prefix is None or text.startswith(prefix)
        structure["first_marker_region"] = (
            "no_marker"
            if not matches
            else (
                "no_end_think"
                if prefix is None
                else "before_end_think" if matches[0].start() < len(prefix) else "after_end_think"
            )
        )
        structure["post_end_think_marker_count"] = len(PAT.findall(text[len(prefix) :])) if prefix is not None else None
    score = sum(row["score"]) if isinstance(row["score"], list) else row["score"]
    assert score in (0, 1), ("Nonbinary reward", score)
    return {
        **structure,
        "eos_token_counts": {str(i): tokens.count(i) for i in eos_ids},
        "eos_interior_counts": {str(i): tokens[:-1].count(i) for i in eos_ids},
        "score": int(score),
        "stop": row["stop_reason"],
        "length": len(tokens),
        "chars": len(text),
        "marker_count": len(matches),
        "different_marker_values": len(set(answers)),
        "chars_after_first_marker": len(text) - matches[0].end() if matches else None,
        "post_first_marker_fraction": (len(text) - matches[0].end()) / len(text) if matches else None,
        "repeated_16gram_fraction": 1 - len(set(grams)) / len(grams) if grams else 0,
        "max_same_long_line": max(counts.values(), default=0),
        "last_token": tokens[-1] if tokens else None,
        "first_marker": matches[0].start() if matches else None,
        "text": text,
        "uid_sha256": hashlib.sha256(row["uid"].encode()).hexdigest()[:16],
        "env_extra_keys": (
            sorted(row.get("env_extras", {}))
            if isinstance(row.get("env_extras"), dict)
            else [type(row.get("env_extras")).__name__]
        ),
    }


def group_summary(items):
    return {
        "rows": len(items),
        "end_think_counts": dict(collections.Counter(str(x["end_think_count"]) for x in items)),
        "first_marker_regions": dict(collections.Counter(x["first_marker_region"] for x in items)),
        "decoder_verified_rows": sum(x["decoder_verified"] for x in items),
        "rows_with_post_end_think_marker": sum((x["post_end_think_marker_count"] or 0) > 0 for x in items),
        "rows_first_marker_immediately_quoted": sum(x["first_marker_preceded_by_quote"] for x in items),
        "finalized_token_lengths": quantiles([x["length"] for x in items]),
        "marker_counts": dict(collections.Counter(str(x["marker_count"]) for x in items)),
        "with_marker": sum(x["marker_count"] > 0 for x in items),
        "multiple_marker_values": sum(x["different_marker_values"] > 1 for x in items),
        "chars_after_first_marker": quantiles([x["chars_after_first_marker"] for x in items if x["marker_count"]]),
        "post_first_marker_fraction": quantiles([x["post_first_marker_fraction"] for x in items if x["marker_count"]]),
        "repeated_16gram_fraction": quantiles([x["repeated_16gram_fraction"] for x in items]),
        "max_same_long_line": quantiles([x["max_same_long_line"] for x in items]),
        "last_token_ids": dict(collections.Counter(str(x["last_token"]) for x in items)),
        "eos_token_counts": {
            i: sum(x["eos_token_counts"][i] for x in items) for i in (items[0]["eos_token_counts"] if items else {})
        },
        "rows_with_interior_eos": {
            i: sum(x["eos_interior_counts"][i] > 0 for x in items)
            for i in (items[0]["eos_interior_counts"] if items else {})
        },
    }


def validate_eos_proof(raw, protocol):
    assert hashlib.sha256(raw).hexdigest() == protocol["eos_method_proof_sha256"]
    proof = json.loads(raw)
    assert proof["status"] == "E61_PINNED_EOS_METHOD_PASS"
    assert proof["vllm_revision"] == "fa50698a9a30"
    assert proof["actual_default_ignore_eos"] is False
    assert {row["ignore_eos"]: row["effective_stop_ids"] for row in proof["cases"]} == {
        False: [128001, 128009],
        True: [],
    }
    return proof


def main():
    global byte_count

    result = {
        "scope": (
            "Nonexclusive operational diagnostics of retained development responses. "
            "Finalized trajectory tokens and finish reasons are preserved; raw generation is unavailable. "
            "Repetition measures duplicate window occurrences and is length-dependent."
        ),
        "runs": {},
        "metadata": {},
        "effective_eos_configuration_proof": validate_eos_proof(EOS_PROOF.read_bytes(), SPEC["analysis_protocol"]),
        "effective_eos_scope": "Configuration diagnostic; not a causal attribution of any truncation.",
    }
    model = SPEC["runs"][0]["envelope"]["request"]["model"]
    for filename in ["config.json", "generation_config.json", "tokenizer_config.json", "special_tokens_map.json"]:
        d = read_json(posixpath.join(model["uri"], filename), optional=True)
        result["metadata"][filename] = (
            {
                k: v
                for k, v in (d or {}).items()
                if k
                in [
                    "eos_token_id",
                    "bos_token_id",
                    "pad_token_id",
                    "eos_token",
                    "bos_token",
                    "pad_token",
                    "added_tokens_decoder",
                    "additional_special_tokens",
                ]
            }
            if d is not None
            else {"absent": True}
        )
    for filename in ["tokenizer_config.json", "tokenizer.json"]:
        uri = (
            "https://huggingface.co/"
            + model["tokenizer_uri"]
            + "/resolve/"
            + model["tokenizer_revision"]
            + "/"
            + filename
        )
        with urllib.request.urlopen(uri, timeout=45) as f:
            raw = f.read(25 * 1024 * 1024 + 1)
        assert len(raw) <= 25 * 1024 * 1024
        assert hashlib.sha256(raw).hexdigest() == SPEC["tokenizer_file_hashes"][filename]
        d = json.loads(raw)
        if filename == "tokenizer.json":
            decoder = Tokenizer.from_str(raw.decode())
            assert decoder.token_to_id("<|start_think|>") == 128002
            assert decoder.token_to_id("<|end_think|>") == 128003
        result["metadata"]["requested_hf_" + filename] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "special_tokens": {
                k: v for k, v in d.items() if k in ["eos_token", "bos_token", "pad_token", "additional_special_tokens"]
            },
            "added_tokens": d.get("added_tokens", d.get("added_tokens_decoder", {})),
        }
    # Keep only special token mappings, not vocabulary or templates.
    for d in result["metadata"].values():
        if isinstance(d.get("added_tokens_decoder"), dict):
            d["added_tokens_decoder"] = {
                k: v for k, v in d["added_tokens_decoder"].items() if v.get("special") or "end" in v.get("content", "")
            }
        if isinstance(d.get("added_tokens"), list):
            d["added_tokens"] = [t for t in d["added_tokens"] if t.get("special") or "end" in t.get("content", "")]
    eos_ids = set()
    for d in result["metadata"].values():
        v = d.get("eos_token_id", [])
        eos_ids.update(v if isinstance(v, list) else [v])
    td = result["metadata"]["requested_hf_tokenizer_config.json"]
    eos_text = td["special_tokens"].get("eos_token")
    if isinstance(eos_text, dict):
        eos_text = eos_text.get("content")
    for t in result["metadata"]["requested_hf_tokenizer.json"].get("added_tokens", []):
        if t.get("content") == eos_text:
            eos_ids.add(t["id"])
    eos_ids = {i for i in eos_ids if type(i) is int}
    result["metadata"]["observed_eos_ids"] = sorted(eos_ids)
    print("STOP_DIAGNOSTIC_METADATA " + json.dumps(result["metadata"]), flush=True)
    for spec in SPEC["runs"]:
        locator = spec["envelope"]["request"]
        attempt = read_json(
            posixpath.join(locator["output"]["attempts_root"], spec["attempt_id"] + ".json"),
            optional=not spec.get("require_actual_attempt_manifest", True),
        )
        req = attempt["request"] if attempt else dict(locator, attempt_id=spec["attempt_id"])
        assert req["attempt_id"] == spec["attempt_id"] and req["run_id"] == spec["run_id"]
        assert {k: v for k, v in req.items() if k != "attempt_id"} == {
            k: v for k, v in locator.items() if k != "attempt_id"
        }
        assert req["runtime"]["commit"] == "fb55d3bb66fe96b7cb85ccfc313afe06a5386c02" and req["model"] == model
        arm = {
            "actual_attempt_manifest_verified": attempt is not None,
            "identity_scope": (
                "verified terminal attempt request"
                if attempt
                else "supplied actual running attempt; stable output locator only, terminal manifest absent"
            ),
            "attempt_id": req["attempt_id"],
            "run_id": req["run_id"],
            "steps": {},
            "examples": [],
        }
        run = wandb.Api(timeout=45).run(spec["wandb_url"].split("wandb.ai/", 1)[1].replace("/runs/", "/"))
        assert run.name.endswith(spec["attempt_id"]), "W&B attempt identity mismatch"
        gc = run.config["generator"]
        arm["actual_wandb_generator_controls"] = {
            k: gc.get(k)
            for k in [
                "use_conversation_multi_turn",
                "append_eos_token_after_stop_str_in_multi_turn",
                "chat_template",
                "chat_template_kwargs",
                "sampling_params",
                "eval_sampling_params",
                "max_turns",
                "engine_init_kwargs",
            ]
        }
        for step in spec["steps"]:
            uri = posixpath.join(
                req["output"]["export_root"], "dumped_evals", f"global_step_{step}_evals", "g03-gsm8k.jsonl"
            )
            fs, path = fs_path(uri)
            if not fs.exists(path):
                arm["steps"][str(step)] = {"available": False, "dump_uri": uri}
                print("STOP_DIAGNOSTIC_UNAVAILABLE", spec["label"], step, flush=True)
                continue
            items = []
            uids = set()
            dump_hash = hashlib.sha256()
            dump_bytes = 0
            with fs.open(path, "rb") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    dump_hash.update(line)
                    dump_bytes += len(line)
                    byte_count += len(line)
                    assert len(line) <= 1024 * 1024 and byte_count <= MAX_BYTES
                    row = json.loads(line)
                    assert len(items) < 128 and row["uid"] not in uids
                    uids.add(row["uid"])
                    assert row["token_provenance"] == "finalized_trajectory" and row["response_length"] == len(
                        row["response_ids"]
                    )
                    assert (
                        sha(row["response_ids"]) == row["response_ids_sha256"]
                        and sha(row["prompt_token_ids"]) == row["prompt_token_ids_sha256"]
                    )
                    feature = features(row, eos_ids, decoder)
                    feature["parser_inside_thinking"] = parser_inside_thinking(
                        row["prompt_token_ids"], row["response_ids"], row["output_response"], decoder, 128002, 128003
                    )
                    feature["stop_reason"] = feature["stop"]
                    feature["reward"] = feature["score"]
                    feature["no_effective_eos"] = (
                        False
                        if gc["eval_sampling_params"].get("ignore_eos", False) is False
                        and not gc.get("engine_init_kwargs", {}).get("generation_config")
                        and sorted(eos_ids) == [128001, 128009]
                        else None
                    )
                    items.append(feature)
            assert len(items) == 128
            by = {}
            for stop, score in sorted(set((x["stop"], x["score"]) for x in items)):
                by[f"{stop}|reward{score}"] = group_summary(
                    [x for x in items if (x["stop"], x["score"]) == (stop, score)]
                )
            arm["steps"][str(step)] = {
                "correct": sum(x["score"] for x in items),
                "stop_counts": dict(collections.Counter(x["stop"] for x in items)),
                "crosstab": by,
                "all": group_summary(items),
                "env_extra_keys": sorted(set(k for x in items for k in x["env_extra_keys"])),
                "nonexclusive_diagnostics": summarize(items),
                "dump_uri": uri,
                "dump_sha256": dump_hash.hexdigest(),
                "dump_bytes": dump_bytes,
            }
            print("STOP_DIAGNOSTIC_PROGRESS", spec["label"], step, "rows", len(items), flush=True)
        result["runs"][spec["label"]] = arm
        print(
            "STOP_DIAGNOSTIC_ARM " + json.dumps({"label": spec["label"], "summary": arm}, separators=(",", ":")),
            flush=True,
        )
    result["bytes_streamed"] = byte_count
    print("SNOWBALL_STOP_DIAGNOSTIC_RESULT " + json.dumps(result, separators=(",", ":")), flush=True)
    result["analysis_protocol"] = SPEC["analysis_protocol"]
    result["spec_sha256"] = hashlib.sha256(json.dumps(SPEC, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    OUTPUT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")))
    print("SNOWBALL_STOP_DIAGNOSTIC_PASS", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--eos-proof", type=Path, required=True)
    args = p.parse_args()
    SPEC = json.loads(args.spec.read_text())
    OUTPUT = args.output
    EOS_PROOF = args.eos_proof
    main()
