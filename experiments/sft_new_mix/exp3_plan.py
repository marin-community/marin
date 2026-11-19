# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the control inventory and bind independently reversible candidate stages."""

import argparse
import hashlib
import json
from pathlib import Path

from levanter.store.cache import CacheLedger
from rigging.filesystem.storage_path import prefix_join

from experiments.sft_new_mix.exp3_filters import Stage
from experiments.sft_new_mix.exp3_runtime import BUILD_IMPLEMENTATION_FILES, configure_storage, implementation_identity
from experiments.sft_new_mix.score_replay import MODEL_DIR, model_digests

OPENCODE_TOOLS = frozenset(
    (
        "bash",
        "read",
        "write",
        "edit",
        "glob",
        "grep",
        "todowrite",
        "todoread",
        "task",
        "skill",
        "webfetch",
        "websearch",
        "question",
        "invalid",
        "apply_patch",
    )
)
OPENHANDS_TOOLS = frozenset(("execute_bash", "str_replace_editor", "think", "finish"))
SWEAGENT_TOOLS = frozenset(("bash", "str_replace_editor", "submit"))
MINISWE_TOOLS = frozenset(("bash", "finish", "submit"))


def vetted_tools(source: str) -> list[str]:
    tools = OPENCODE_TOOLS
    if "opencode" in source or source.startswith("nemotron_sft_v3/cuda_v1/"):
        tools = OPENCODE_TOOLS
    if "/openhands/" in source or source.startswith(("nemotron_sft_v3/swe_v1/", "nemotron_sft_v3/swe_v2/")):
        tools = OPENHANDS_TOOLS
    if "/sweagent/" in source:
        tools = SWEAGENT_TOOLS
    if "/minisweagent/" in source:
        tools = MINISWE_TOOLS
    if source == "science-tool-use-conversations":
        tools = frozenset(("execute_bash", "bash"))
    return sorted(tools)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--science-input", type=Path, required=True)
    parser.add_argument("--science-uri", required=True)
    parser.add_argument("--science-selection", choices=("accepted", "control"), required=True)
    parser.add_argument("--junkgate", choices=("enabled", "disabled"), required=True)
    parser.add_argument("--replay-line-length", choices=("enabled", "disabled"), required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disable", action="append", choices=[stage.value for stage in Stage], default=[])
    args = parser.parse_args()
    configure_storage()
    args.output.mkdir(parents=True, exist_ok=False)
    inventory_path = args.artifacts / "tool-analysis/inventory.json"
    replay_path = args.artifacts / "packing-ablation/data/lengths-replay-manifest.json"
    science_path = args.science_input / "manifest.json"
    inventory = json.loads(inventory_path.read_text())
    replay = json.loads(replay_path.read_text())
    science = json.loads(science_path.read_text())
    use_pass_inputs = Stage.OUTCOME.value not in args.disable
    source_names = {source["source"] for source in inventory["sources"]}
    target_reasoning = {
        name
        for name in source_names
        if name.startswith(
            (
                "nemotron_sft_v3/math_v3/",
                "nemotron_sft_v3/competitive_programming_v2/competitive_coding_",
                "nemotron_sft_v3/math_proofs_v2/",
                "nemotron_sft_v3/arc_agi_v1/",
                "openthoughts4-code-",
            )
        )
    }
    config = {
        "version": "2026.11.15-scripted-v3",
        "readiness": "intermediate; final decontamination, replacement overlay and full resolve required",
        "enabled": [
            stage.value
            for stage in Stage
            if stage.value not in args.disable
            and (stage != Stage.REPLAY_LINE_LENGTH or args.replay_line_length == "enabled")
        ],
        "science_selection": args.science_selection,
        "recorded_pass_input_selection": "recorded_pass" if use_pass_inputs else "control",
        "think_limits": {name: 50_000 for name in sorted(target_reasoning)},
        "think_limit_basis": (
            "Explicit 50k whole-thought rejection limit; correctness-conditioned p90 is not established"
        ),
        "give_up_sources": sorted(target_reasoning | {"nemotron_sft_v3/science_v2/vendor"}),
        "tools_by_source": {source: vetted_tools(source) for source in sorted(source_names)},
        "tool_policy_evidence_sha256": (
            hashlib.sha256((args.artifacts / "tool-analysis/harbor-tools.json").read_bytes()).hexdigest()
        ),
        "decontamination_enabled": False,
        "bloom_path": "",
        "junkgate_enabled": args.junkgate == "enabled",
        "junkgate_cutoff": 0.32026717512511493,
        "junkgate_calibration": "junkgate-calibration-v1.json; heldout recall1/12, natural false-drop2/67",
    }
    if args.junkgate == "enabled":
        # The cutoff is calibrated for one specific set of model weights; freeze
        # those bytes into the config so the build can verify them before scoring.
        config["junkgate_model_dir"] = MODEL_DIR
        config["junkgate_model_sha256"] = model_digests()
    config_bytes = json.dumps(config, sort_keys=True, indent=2).encode()
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    config_uri = prefix_join(args.destination, "config.json")
    tokenizer_sha = hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
    tokenizer_uri = prefix_join(args.destination, "tokenizer.json")
    base = {
        "output_prefix": args.destination,
        "config_uri": config_uri,
        "config_sha256": config_sha,
        "implementation_sha256": implementation_identity(BUILD_IMPLEMENTATION_FILES),
        "tokenizer_uri": tokenizer_uri,
        "tokenizer_sha256": tokenizer_sha,
    }
    replacements = []
    success_inputs = {}
    graphs = json.loads((args.artifacts / "reports/open-swe-cache-graph.json").read_text())["graphs"]
    source_by_partition = {graph["partition"]: graph["source"] for graph in graphs}
    cleaned_paths = [
        args.artifacts / "cleaned/open-swe-partition-first",
        *sorted((args.artifacts / "cleaned/open-swe-remaining").iterdir()),
    ]
    for directory in cleaned_paths:
        publication = directory / "s3-verified-copy-manifest.json"
        selection = directory / "manifest.json"
        if not publication.exists():
            continue
        published = json.loads(publication.read_text())
        accepted = json.loads(selection.read_text())
        source = source_by_partition[accepted["partition"]]
        ledger = CacheLedger.load(published["destination"])
        if not ledger.is_finished:
            raise ValueError("Accepted Open-SWE cache is unfinished")
        success_inputs[source] = [
            {
                **base,
                "source": source,
                "input_path": published["destination"],
                "expected_rows": ledger.total_num_rows,
                "expected_tokens": ledger.field_counts["input_ids"],
                "row_start": 0,
                "row_stop": ledger.total_num_rows,
                "loss_mode": "chat_assistant_only",
                "source_revision": accepted["source_revision"],
                "passed": True,
                "pass_receipt_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
            }
        ]
    if use_pass_inputs and len(success_inputs) != 8:
        raise ValueError("Expected all eight frozen successful Open-SWE partitions")
    jobs = []
    for source in inventory["sources"]:
        name = source["source"]
        if args.science_selection == "accepted" and name.startswith("nemotron_sft_v3/science_v2/"):
            replacements.append(
                {
                    "source": name,
                    "transform": "accepted_science_swap",
                    "before_rows": source["ledger"]["total_num_rows"],
                    "before_tokens": source["ledger"]["field_counts"]["input_ids"],
                    "revision": science["science_revision"],
                }
            )
            continue
        if use_pass_inputs and name in success_inputs:
            jobs.extend(success_inputs[name])
            replacements.append(
                {
                    "source": name,
                    "transform": "open_swe_recorded_pass_inputs",
                    "before_rows": source["ledger"]["total_num_rows"],
                    "before_tokens": source["ledger"]["field_counts"]["input_ids"],
                    "after_rows": success_inputs[name][0]["expected_rows"],
                    "after_tokens": success_inputs[name][0]["expected_tokens"],
                }
            )
            continue
        jobs.extend(
            {
                **base,
                "source": name,
                "input_path": shard["path"],
                "expected_rows": shard["rows"],
                "expected_tokens": shard["tokens"],
                "row_start": 0,
                "row_stop": shard["rows"],
                "loss_mode": "chat_assistant_only",
                "source_revision": source["ledger_sha256"],
                "passed": False,
                "pass_receipt_sha256": "",
            }
            for shard in source["shards"]
        )
    for shard in science["shards"] if args.science_selection == "accepted" else []:
        relative = str(Path(shard["path"]).relative_to(args.science_input))
        jobs.append(
            {
                **base,
                "source": shard["source"],
                "input_path": prefix_join(args.science_uri, relative),
                "expected_rows": shard["rows"],
                "expected_tokens": shard["tokens"],
                "row_start": 0,
                "row_stop": shard["rows"],
                "loss_mode": "chat_assistant_only",
                "source_revision": science["science_revision"],
                "passed": False,
                "pass_receipt_sha256": "",
            }
        )
    for source in replay:
        suffix = source["cache_dir"].split("/replay/", 1)[1]
        name = "replay/" + suffix.split("/")[0] + "/" + suffix.rsplit("/", 1)[-1]
        jobs.append(
            {
                **base,
                "source": name,
                "input_path": source["cache_dir"],
                "expected_rows": source["rows"],
                "expected_tokens": source["tokens"],
                "row_start": 0,
                "row_stop": source["rows"],
                "loss_mode": "full_token",
                "source_revision": source["lengths_sha256"],
                "passed": False,
                "pass_receipt_sha256": "",
            }
        )
    manifest = {
        "output_prefix": args.destination,
        "jobs": jobs,
        "source_replacements": replacements,
        "input_manifest_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (inventory_path, replay_path, science_path)
        },
        "source_loss": {
            job["source"]: {
                "loss_mode": job["loss_mode"],
                "loss_weights_key": "loss_weights",
                "loss_weights_indexing": "prediction",
                "teacher_only": False,
                "source_revision": job["source_revision"],
            }
            for job in jobs
        },
    }
    (args.output / "plan.json").write_text(json.dumps(manifest, sort_keys=True, indent=2))
    (args.output / "config.json").write_bytes(config_bytes)
    (args.output / "tokenizer.json").write_bytes(args.tokenizer.read_bytes())
    print(
        json.dumps(
            {
                "jobs": len(jobs),
                "sources": len(manifest["source_loss"]),
                "science_rows": sum(
                    job["expected_rows"] for job in jobs if job["source"].startswith("nemotron_sft_v3/science_v2/")
                ),
                "replay_rows": sum(row["rows"] for row in replay),
            }
        )
    )


if __name__ == "__main__":
    main()
