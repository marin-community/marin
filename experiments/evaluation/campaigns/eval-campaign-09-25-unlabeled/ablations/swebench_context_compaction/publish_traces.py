# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the sealed SWE-bench ablation's accepted trial traces for public inspection."""

import argparse
import hashlib
import json
import re
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import zstandard
from huggingface_hub import HfApi
from rigging.filesystem.buckets import filesystem_for

ARM_NAMES = (
    "65k-16k-no-summary",
    "131k-16k-no-summary",
    "32k-8k-summary",
    "65k-16k-summary",
)
MODEL_NAMES = ("qwen36", "gemma4", "nemotron-lightning", "snowball-v0.1")
ARCHIVE_ARM_MARKERS = (
    ("131", "131k-16k-no-summary"),
    ("32_SUMMARY", "32k-8k-summary"),
    ("65_SUMMARY", "65k-16k-summary"),
    ("65", "65k-16k-no-summary"),
)
SENSITIVE = re.compile(
    r"iris\.oa\.dev/proxy/t/|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}|"
    r"hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9_-]{20,}|Bearer\s+[A-Za-z0-9_.-]{20,}",
    re.IGNORECASE,
)
REDACTIONS = (
    re.compile(r"https?://iris\.oa\.dev/proxy/t/[^\s\"'<>]+", re.IGNORECASE),
    re.compile(r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}(?:\.[A-Za-z0-9_-]+)?"),
    re.compile(r"hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9_-]{20,}|Bearer\s+[A-Za-z0-9_.-]{20,}", re.IGNORECASE),
)


def redact(value):
    if isinstance(value, str):
        for pattern in REDACTIONS:
            value = pattern.sub("[REDACTED_CREDENTIAL]", value)
        return value
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    return value


def read_audit(archive: Path, member: str) -> dict:
    with archive.open("rb") as source:
        decompressed = zstandard.ZstdDecompressor().stream_reader(source)
        with tarfile.open(fileobj=decompressed, mode="r|", errorlevel=2) as bundle:
            for item in bundle:
                if item.name == member:
                    payload = bundle.extractfile(item)
                    assert payload is not None
                    return json.load(payload)
    raise ValueError(f"Missing {member} in {archive}")


def accepted_manifest(audit_root: Path) -> dict:
    completion = json.loads((audit_root / "audits/COMPLETION_AUDIT_20261022_0311.json").read_text())
    assert len(completion["rows"]) == len(MODEL_NAMES) * len(ARM_NAMES)
    arms = []
    for index, item in enumerate(completion["rows"]):
        archive_name = Path(item["archive"]).name.upper()
        arm_name = next(name for marker, name in ARCHIVE_ARM_MARKERS if marker in archive_name)
        audit = read_audit(audit_root / item["archive"], item["member"])
        retained = set(audit.get("retained_tasks", []))
        trials = audit.get("trials", audit.get("observed_trials"))
        assert trials is not None
        selected = [
            {key: trial[key] for key in ("task_name", "source_uri", "sha256", "reward")}
            for trial in trials
            if trial.get("scored") is True and (not retained or trial["task_name"] in retained)
        ]
        assert len(selected) == item["valid_tasks"], item["archive"]
        assert len({trial["task_name"] for trial in selected}) == len(selected)
        assert abs(sum(trial["reward"] for trial in selected) / len(selected) - item["mean_reward"]) < 1e-10
        arms.append(
            {
                "model": MODEL_NAMES[index // len(ARM_NAMES)],
                "arm": arm_name,
                "audit_archive": item["archive"],
                "audit_member": item["member"],
                "mean_reward": item["mean_reward"],
                "trials": sorted(selected, key=lambda trial: trial["task_name"]),
            }
        )
    assert {(arm["model"], arm["arm"]) for arm in arms} == {(model, name) for model in MODEL_NAMES for name in ARM_NAMES}
    return {"benchmark": "SWE-bench Verified", "expected_tasks_per_arm": 500, "arms": arms}


def public_trial(trial: dict) -> dict:
    uri = trial["source_uri"]
    fs, path = filesystem_for(uri)
    payload = fs.cat(path)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != trial["sha256"]:
        raise ValueError(f"Result changed: {uri}")
    result = json.loads(payload)
    if result["task_name"] != trial["task_name"]:
        raise ValueError(f"Task identity changed: {uri}")
    if ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward") != trial["reward"]:
        raise ValueError(f"Reward changed: {uri}")
    base = path.rsplit("/", 1)[0]
    attempt_paths = fs.ls(base + "/attempts", detail=False)
    attempts = []
    for attempt_path in sorted(attempt_paths):
        trajectory_path = attempt_path.rstrip("/") + "/agent/mini-swe-agent.trajectory.json"
        if not fs.exists(trajectory_path):
            continue
        trajectory_payload = fs.cat(trajectory_path)
        trajectory_sha256 = hashlib.sha256(trajectory_payload).hexdigest()
        try:
            trajectory = json.loads(trajectory_payload)
        except json.JSONDecodeError:
            # Timed-out attempts can leave a zero-byte or truncated trajectory.
            attempts.append(
                {
                    "attempt": attempt_path.rsplit("/", 1)[-1],
                    "trajectory_uri": "s3://" + trajectory_path,
                    "trajectory_sha256": trajectory_sha256,
                    "trajectory_bytes": len(trajectory_payload),
                    "parse_status": "invalid_json",
                }
            )
            continue
        info = trajectory.get("info") or {}
        # The full mini-swe-agent config contains the signed Iris proxy URL.
        safe_trajectory = {
            "trajectory_format": trajectory.get("trajectory_format"),
            "messages": trajectory["messages"],
            "info": {key: info.get(key) for key in ("exit_status", "mini_version", "model_stats", "submission")},
        }
        attempts.append(
            {
                "attempt": attempt_path.rsplit("/", 1)[-1],
                "trajectory_uri": "s3://" + trajectory_path,
                "trajectory_sha256": trajectory_sha256,
                "trajectory": redact(safe_trajectory),
            }
        )
    public = {
        **trial,
        "trial_name": result.get("trial_name"),
        "exception_type": (result.get("exception_info") or {}).get("exception_type"),
        "agent_execution": result.get("agent_execution"),
        "attempts": attempts,
    }
    public = redact(public)
    serialized = json.dumps(public, ensure_ascii=False, separators=(",", ":"))
    if SENSITIVE.search(serialized):
        raise ValueError(f"Sensitive endpoint or credential in public trial {uri}")
    return public


def export(manifest: dict, output: Path, workers: int, repo_id: str | None, model: str | None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for arm in manifest["arms"]:
        if model and arm["model"] != model:
            continue
        relative = f"traces/{arm['model']}/{arm['arm']}.jsonl.zst"
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        compressor = zstandard.ZstdCompressor(level=6)
        missing = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            with destination.open("wb") as raw, compressor.stream_writer(raw) as compressed:
                for index, row in enumerate(pool.map(public_trial, arm["trials"]), start=1):
                    missing += not any("trajectory" in attempt for attempt in row["attempts"])
                    compressed.write((json.dumps(row, ensure_ascii=False) + "\n").encode())
                    if index % 50 == 0:
                        print(f"{arm['model']} {arm['arm']}: {index}/{len(arm['trials'])}", flush=True)
        summary.append(
            {
                "model": arm["model"],
                "arm": arm["arm"],
                "accepted_trials": len(arm["trials"]),
                "mean_reward": arm["mean_reward"],
                "missing_trajectory_count": missing,
                "file": relative,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )
        if repo_id:
            HfApi().upload_file(
                path_or_fileobj=str(destination), path_in_repo=relative, repo_id=repo_id, repo_type="dataset"
            )
        print(json.dumps(summary[-1]), flush=True)
    (output / f"MANIFEST-{model or 'all'}.json").write_text(json.dumps({"arms": summary}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--repo-id")
    parser.add_argument("--model", choices=MODEL_NAMES)
    args = parser.parse_args()
    if args.audit_root:
        manifest = accepted_manifest(args.audit_root)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n")
        return
    if not args.manifest:
        parser.error("--manifest is required for export")
    export(json.loads(args.manifest.read_text()), args.output, args.workers, args.repo_id, args.model)


if __name__ == "__main__":
    main()
