#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor full-atlas M0/M1 Iris jobs; when ≥1 succeeds per model, download
inference output, convert to generations.jsonl, submit gpt-5.1 score batch,
collect, compute BCG.

Emits one stdout line per event for the Monitor tool.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger("bcg_full_pipeline_monitor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# Each entry: (model_key, iris_job_prefix, target_id, job_root_dir, model_label,
#              regional_eval_dirs_to_check)
MODELS = [
    {
        "key": "M0_full",
        "target": "sft",
        "iris_prefix": "/ahmedah/bcg-full-sft-",
        "step_name_prefix": "bcg_probe_sft",  # from bcg_probe_infer.py _step_name
        "job_root": Path("experiments/posttrain/stage4_output/bcg_M0_full"),
        "model_label": "marin-8b-instruct",
    },
    {
        "key": "M1_full",
        "target": "tune_lora_lr1e5_seed0_step1699",
        "iris_prefix": "/ahmedah/bcg-full-lora1e5-",
        "step_name_prefix": "bcg_probe_tune_lora_lr1e5_seed0_step1699",
        "job_root": Path("experiments/posttrain/stage4_output/bcg_M1_full"),
        "model_label": "marin-8b-dpo-tune_lora_lr1e5_seed0_step1699",
    },
]
RUBRICS = Path("experiments/posttrain/stage3_output/paired_rubrics_full.jsonl")
REGIONS = ["us-east1", "us-east5", "eu-west4"]
REGION_BUCKET = {"us-east1": "marin-us-east1", "us-east5": "marin-us-east5",
                 "eu-west4": "marin-eu-west4", "europe-west4": "marin-eu-west4"}
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", 120))  # 2 min default
MAX_MINUTES = int(os.environ.get("MAX_MINUTES", 120))


def emit(kind: str, msg: str = "") -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now} {kind} {msg}".rstrip(), flush=True)


def run(cmd: list[str], timeout: int = 1800) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
        tail = (p.stdout + p.stderr).splitlines()[-10:]
        return p.returncode, "\n".join(tail)
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def iris_job_states(prefix: str) -> dict[str, str]:
    """Parse `iris job list` output for a prefix, return job_id -> state.

    Includes both outer wrapper jobs and inner eval-* subjobs so we can see
    when the actual inference task succeeds.
    """
    rc, out = run(
        ["uv", "run", "iris", "--config", "lib/iris/examples/marin.yaml",
         "job", "list", "--prefix", prefix],
        timeout=90,
    )
    states = {}
    for line in out.splitlines():
        if line.startswith("/ahmedah/"):
            parts = line.split()
            if len(parts) >= 2:
                states[parts[0]] = parts[1]
    return states


def find_inference_output(step_name_prefix: str, region: str) -> str | None:
    """Look for a succeeded inference shard for (step_name_prefix, region)."""
    region_bucket = REGION_BUCKET.get(region, f"marin-{region}")
    tpu_suffixes = ["v6e4_full2573", "v6e8_full2573"]
    region_compact = region.replace("-", "")
    for tpu_suffix in tpu_suffixes:
        step_name = f"{step_name_prefix}_{region_compact}_{tpu_suffix}"
        prefix = f"gs://{region_bucket}/eval/{step_name}/inference-"
        rc, out = run(["gsutil", "ls", prefix], timeout=60)
        if rc != 0:
            continue
        # Find line ending with /shard_00000.jsonl.gz
        for line in out.splitlines():
            line = line.strip()
            if line.endswith("/shard_00000.jsonl.gz"):
                return line
    return None


def download_and_convert(inference_url: str, job_root: Path, model_label: str) -> int:
    """Download shard, convert to generations.jsonl. Return row count."""
    job_root.mkdir(parents=True, exist_ok=True)
    raw_path = job_root / "inference_raw.jsonl.gz"
    rc, out = run(["gsutil", "cp", inference_url, str(raw_path)], timeout=300)
    if rc != 0:
        raise RuntimeError(f"download failed: {out}")

    rows = []
    with gzip.open(raw_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            behavior_id = r["behavior_id"]
            if not behavior_id.startswith("bcg::"):
                continue
            pair_id = behavior_id[len("bcg::"):]
            config_id = r["config_id"]
            if not config_id.startswith("tp"):
                continue
            tp_idx = int(config_id[2:])
            sample_idx = int(r["sample_idx"])
            cid = f"gen::{pair_id}::{tp_idx:03d}::s{sample_idx:02d}"
            rows.append({
                "pair_id": pair_id,
                "tension_point_idx": tp_idx,
                "sample_idx": sample_idx,
                "custom_id": cid,
                "prompt": r["user_message"],
                "model": model_label,
                "response": r["response_text"],
                "usage": {},
            })

    out_path = job_root / "generations.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def process_model(model_cfg: dict) -> bool:
    """Download output for one model, submit score batch. Returns True on success."""
    key = model_cfg["key"]
    # Find a succeeded job — check all regions / both TPU types.
    found = None
    for region in REGIONS:
        url = find_inference_output(model_cfg["step_name_prefix"], region)
        if url is not None:
            found = url
            emit(f"{key}_FOUND", url)
            break
    if found is None:
        return False

    try:
        n = download_and_convert(found, model_cfg["job_root"], model_cfg["model_label"])
        emit(f"{key}_DOWNLOADED", f"{n} rows")
    except Exception as e:
        emit(f"{key}_ERR", str(e))
        return False

    # Submit score batch via stage4_bcg_eval.py
    rc, tail = run([
        "uv", "run", "python", "experiments/posttrain/stage4_bcg_eval.py",
        "score-submit",
        "--rubrics", str(RUBRICS),
        "--job-root", str(model_cfg["job_root"]),
        "--judge-model", "gpt-5.1",
    ])
    if rc != 0:
        emit(f"{key}_ERR", f"score-submit rc={rc}: {tail}")
        return False
    # Read batch_id from job state
    score_state = model_cfg["job_root"] / "score" / "batch_state.json"
    if score_state.exists():
        state = json.loads(score_state.read_text())
        emit(f"{key}_SCORE_SUBMITTED", f"batch={state['batch_id']} requests={state['num_requests']}")
    return True


def main() -> int:
    if "OPENAI_API_KEY" not in os.environ:
        emit("ERR", "OPENAI_API_KEY not set")
        return 2

    emit("STARTED", f"poll_every={POLL_INTERVAL}s max={MAX_MINUTES}min")
    deadline = time.time() + MAX_MINUTES * 60
    done_models: set[str] = set()

    while time.time() < deadline:
        any_change = False
        for model_cfg in MODELS:
            if model_cfg["key"] in done_models:
                continue
            # Check Iris states for this model.
            states = iris_job_states(model_cfg["iris_prefix"])
            # Count how many nested eval-* jobs have succeeded.
            succeeded = [j for j, s in states.items() if s == "succeeded" and "eval-bcg_probe" in j]
            running = [j for j, s in states.items() if s == "running"]
            pending = [j for j, s in states.items() if s == "pending"]
            # Emit state summary only on change or every 5 minutes
            # (we re-emit on every poll for visibility)
            emit(
                f"{model_cfg['key']}_STATES",
                f"succeeded={len(succeeded)} running={len(running)} pending={len(pending)}",
            )
            if succeeded:
                ok = process_model(model_cfg)
                if ok:
                    done_models.add(model_cfg["key"])
                    any_change = True

        if len(done_models) == len(MODELS):
            emit("ALL_MODELS_SUBMITTED", "both M0_full and M1_full score batches submitted")
            break
        time.sleep(POLL_INTERVAL)
    else:
        emit("BUDGET_EXHAUSTED", f"done={list(done_models)}")
        return 1

    emit("EXIT_OK", "score batches running; use stage4_bcg_eval.py score-collect + compute when each terminates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
