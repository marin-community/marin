"""Fresh GLM attacks against the public learner interface, with retained evidence."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
from pathlib import Path

STRATEGIES = ("injection", "shortcut", "boundary")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trial_failure(trial):
    """Classify retained Harbor failures without copying provider exception bodies."""
    exception = trial.get("exception_info")
    if not exception:
        return None
    if exception.get("exception_type") == "ExtractionError":
        # The caller must still verify the matching extraction/null artifact.
        return None
    if exception.get("exception_message") == "Incomplete GLM solver output: length":
        return "model_output_truncated"
    return "harbor_trial_failed"


def assess_attack_results(records, max_reward=0.2):
    """An apparent rewarded attack needs adjudication; it is not automatically a bug."""
    issues = []
    graded = 0
    if {record.get("strategy") for record in records} != set(STRATEGIES) or len(
        records
    ) != len(STRATEGIES):
        issues.append("independent attack suite is incomplete or duplicated")
    for record in records:
        strategy = record.get("strategy", "unknown")
        if record.get("error"):
            issues.append(f"{strategy}: attack execution failed")
            continue
        steps = record.get("steps", [record])
        if not steps:
            issues.append(f"{strategy}: no step outcomes")
        for step in steps:
            label = strategy + ":" + str(step.get("step_index", 0))
            result = step.get("result", {})
            status, reward = result.get("status"), result.get("reward")
            if status == "graded":
                graded += 1
                if (
                    type(reward) not in (int, float)
                    or not math.isfinite(reward)
                    or not 0 <= reward <= 1
                ):
                    issues.append(f"{label}: invalid graded reward")
                elif reward > max_reward:
                    issues.append(
                        f"{label}: rewarded attack requires independent adjudication"
                    )
            elif status != "extraction_error" or reward is not None:
                issues.append(f"{label}: no valid semantic outcome")
    if not graded:
        issues.append("no well-formed attack reached semantic grading")
    return issues


async def run_independent_attacks(
    package,
    execution,
    trials,
    output_root,
    kind,
    *,
    seen_sandbox_ids=None,
    max_reward=0.2,
    token_limits: tuple[int | None, ...] | None = None,
    request_timeout: float | None = None,
):
    from taskcompendium.harbor.runner import run_trial

    from capability_pipeline.runtime import provider_isolation_record

    package, trials, output_root = map(Path, (package, trials, output_root))
    step_names = json.loads((package / "manifest.json").read_text())["step_names"]
    seen_sandbox_ids = seen_sandbox_ids if seen_sandbox_ids is not None else set()
    # Reuse only the trusted launch config. No solver transcript, private source,
    # builder notes or previous candidate is supplied to another agent.
    if "api_key" in execution["agent"].get("kwargs", {}):
        raise ValueError(
            "adversary execution config contains a literal credential field"
        )
    if token_limits is not None and (
        not token_limits
        or token_limits[-1] is not None
        or any(type(limit) is not int or limit < 1 for limit in token_limits[:-1])
    ):
        raise ValueError("adversary token limits must end with remaining context")
    if request_timeout is not None and request_timeout <= 0:
        raise ValueError("adversary request timeout must be positive")

    async def attempt(strategy):
        config = copy.deepcopy(execution)
        config["agent"]["import_path"] = (
            "capability_pipeline.runtime_agents:AdversarialChatAgent"
            if kind == "none"
            else "capability_pipeline.runtime_agents:AdversarialShellToolAgent"
        )
        config["agent"]["kwargs"]["attack_strategy"] = strategy
        if token_limits is not None:
            config["agent"]["kwargs"]["token_limits"] = list(token_limits)
            config["agent"]["kwargs"]["max_tokens"] = token_limits[0]
        if request_timeout is not None:
            config["agent"]["kwargs"]["request_timeout"] = request_timeout
        name = "independent-attack-" + strategy
        root = trials / name
        record = {"strategy": strategy, "trial": name}
        try:
            await run_trial(package, config, trials, name)
            record["trial_artifact"] = str(
                (root / "result.json").relative_to(output_root)
            )
            record["trial_sha256"] = sha256(root / "result.json")
            trial = json.loads((root / "result.json").read_text())
            failure = trial_failure(trial)
            if failure:
                record["error"] = failure
                # result.json binds the underlying exception even when Harbor
                # could not finish writing a transcript or grading artifact.
                return record
            record["steps"] = []
            for index, step_name in enumerate(step_names):
                step_root = root if len(step_names) == 1 else root / "steps" / step_name
                step = {"step_index": index, "step_name": step_name}
                for label, path in (
                    ("transcript", step_root / "agent" / "transcript.json"),
                    ("grading", step_root / "verifier" / "taskcompendium-result.json"),
                ):
                    step[label + "_artifact"] = str(path.relative_to(output_root))
                    step[label + "_sha256"] = sha256(path)
                step["result"] = json.loads(
                    (step_root / "verifier" / "taskcompendium-result.json").read_text()
                )
                record["steps"].append(step)
            if trial.get("exception_info") and (
                trial.get("verifier_result") is not None
                or not any(
                    step["result"].get("status") == "extraction_error"
                    and step["result"].get("reward") is None
                    for step in record["steps"]
                )
            ):
                record["error"] = "inconsistent_extraction_transport"
            if kind == "docker":
                record["isolation"] = provider_isolation_record(
                    root / "daytona-environment.json", output_root, seen_sandbox_ids
                )
        except Exception as error:  # noqa: BLE001 -- isolate arbitrary provider failures per trial
            # Preserve sibling trials and record the failure without pretending it
            # was a zero-reward attack. Exception bodies may contain provider data.
            record["error"] = type(error).__name__
        return record

    records = await asyncio.gather(*(attempt(strategy) for strategy in STRATEGIES))
    issues = assess_attack_results(records, max_reward)
    report = {
        "state": "passed" if not issues else "needs_adjudication_or_retry",
        "independent": True,
        "candidate_interface": "public learner instructions and fresh learner environment only",
        "max_false_reward": max_reward,
        "issues": issues,
        "cases": records,
        "implementation_sha256": sha256(Path(__file__)),
        "agent_implementation_sha256": sha256(
            Path(__file__).with_name("runtime_agents.py")
        ),
        "transport_policy": {
            "token_limits": list(token_limits) if token_limits is not None else None,
            "request_timeout": request_timeout,
        },
        "limitations": [
            "Three bounded attacks are a pilot gate, not an exhaustive security or validity guarantee.",
            "A rewarded attack may be a legitimate partial/correct solution; independently adjudicate before calling it a reward bug.",
            "Same model family as construction and solving can share blind spots.",
        ],
    }
    path = output_root / "independent-adversary.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return {
        "report": report,
        "artifact": str(path.relative_to(output_root)),
        "artifact_sha256": sha256(path),
    }
