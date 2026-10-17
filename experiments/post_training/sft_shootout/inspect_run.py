# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize a harbor_glm.py output folder in the job log, for runs whose storage is only readable
in-cluster: per trial the reward, error, phase timings, verifier output tail and one line per
trajectory step; then the request log (settings as sent, token usage, latency).

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a --cpu 1 --memory 4GB \
      --job-name inspect-<name> -- uv run python experiments/post_training/sft_shootout/inspect_run.py <prefix>
"""

import json
import statistics
import sys
from datetime import datetime

from rigging.filesystem.storage_path import StoragePath

STEP_TEXT = 160


def seconds(phase: dict) -> float | None:
    if not phase or not phase.get("started_at") or not phase.get("finished_at"):
        return None
    start, end = (datetime.fromisoformat(phase[k]) for k in ("started_at", "finished_at"))
    return round((end - start).total_seconds(), 1)


def show_trial(result_path: StoragePath) -> None:
    result = json.loads(result_path.read_text())
    print(f"\n== trial {result['trial_name']}")
    print(f"  reward {(result.get('verifier_result') or {}).get('rewards')}")
    print(f"  exception {result.get('exception_info')}")
    phases = {k: seconds(v) for k, v in result.items() if isinstance(v, dict) and "started_at" in v}
    print(f"  seconds total {seconds(result)} phases {phases}")
    agent = result.get("agent_result") or {}
    print(f"  agent tokens in {agent.get('n_input_tokens')} cached {agent.get('n_cache_tokens')} out {agent.get('n_output_tokens')}")
    print(f"  agent metadata {json.dumps(agent.get('metadata'))[:600]}")
    stdout = (result.get("verifier_result") or {}).get("stdout") or ""
    print(f"  verifier stdout tail:\n{stdout[-1500:]}")
    agent_dir = StoragePath(str(result_path.parent) + "/attempts/000/agent")
    for path in agent_dir.ls() if agent_dir.exists() else []:
        if str(path).endswith(".txt"):
            print(f"  agent log {path.name}:\n{path.read_text()[-4000:]}")
        if path.name in ("agent_network.json", "network_probe.json"):
            print(f"  {path.name}:\n{path.read_text()}")
    trajectory = StoragePath(f"{agent_dir}/trajectory.json")
    if not trajectory.exists():
        return
    document = json.loads(trajectory.read_text())
    steps = document.get("steps", [])
    print(f"  trajectory {document.get('schema_version')} agent {document.get('agent')} steps {len(steps)}")
    print(f"  step keys {sorted({k for s in steps for k in s})}")
    for step in steps:
        message = step.get("message") or ""
        message = message if isinstance(message, str) else json.dumps(message)
        calls = [json.dumps(c.get("arguments"))[:STEP_TEXT] for c in step.get("tool_calls") or []]
        observation = json.dumps(step.get("observation"))[:STEP_TEXT] if step.get("observation") else ""
        print(
            f"  [{step.get('step_id')}] {step.get('source')} msg {len(message)}ch "
            f"reasoning {len(step.get('reasoning_content') or '')}ch metrics {step.get('metrics')}\n"
            f"      text {message[:STEP_TEXT]!r}\n      calls {calls}\n      obs {observation}"
        )


def main() -> None:
    root = StoragePath(sys.argv[1])
    for directory, _, names in root.walk():
        if "result.json" in names and "attempts" not in str(directory) and str(directory) != str(root):
            show_trial(StoragePath(f"{directory}/result.json"))
    log = StoragePath(f"{sys.argv[1].rstrip('/')}/requests.jsonl")
    if not log.exists():
        return
    records = [json.loads(line) for line in log.read_text().splitlines()]
    chats = records
    print(f"\n== requests.jsonl: {len(records)} records, trials {len({r.get('trial') for r in records})}")
    print(f"  authors/triggers: {sorted({(r.get('author'), r.get('trigger')) for r in records})}")
    for key in sorted({k for r in chats for k in r["settings"]}):
        print(f"  setting {key}: {sorted({json.dumps(r['settings'].get(key)) for r in chats})}")
    usage = [r["usage"] for r in chats if r.get("usage")]
    if usage:
        for field in ("prompt_tokens", "completion_tokens"):
            values = [u[field] for u in usage]
            print(f"  {field}: sum {sum(values)} median {statistics.median(values)} max {max(values)}")
        reasoning = [(u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0 for u in usage]
        print(f"  reasoning_tokens: sum {sum(reasoning)}")
    mismatch = [
        (r["usage"]["prompt_tokens"], r["prompt_tokens_local"][r["author"]])
        for r in chats
        if r.get("usage") and r.get("prompt_tokens_local", {}).get(r.get("author"))
    ]
    print(f"  prompt-count mismatches (service, local): {[m for m in mismatch if m[0] != m[1]][:20]}")
    print(f"  latency_s sum {sum(r['latency_s'] for r in chats):.1f}")
    print(f"  statuses: {sorted({r['status'] for r in records})}; errors: {[r.get('error') for r in records if r.get('error')][:5]}")


if __name__ == "__main__":
    main()
