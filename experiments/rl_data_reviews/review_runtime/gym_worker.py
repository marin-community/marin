# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a prepared task through MarinSkyRL's registered Gym and verifier adapters."""

import argparse
import contextlib
import copy
import json
import traceback
from pathlib import Path

import skyrl_gym
from omegaconf import OmegaConf
from review_io import append_event, capture_native_sources, model_completion, native_calls, write_json
from skyrl_gym.verification import RolloutEvidence, VerificationResult
from skyrl_train.trajectory_runners.skyrl_gym_contracts import fold_verification_results, verification_from_env_step


def run_task(data: dict, root: Path) -> dict:
    task, config = data["task"], data["config"]
    extras = copy.deepcopy(task["extras"])
    extras["max_turns"] = config["max_turns"]
    env = skyrl_gym.make(
        task["env_id"], env_config=OmegaConf.create(config.get("gym_config", {}).get(task["env_id"], {})), extras=extras
    )
    verdicts = []
    trace = root / "verifier-trace.jsonl"
    done = False
    try:
        messages, metadata = env.init(copy.deepcopy(task["prompt"]))
        options = dict(metadata.get("chat_completion_params") or {})
        options.pop("model", None)
        options.pop("input", None)
        options.pop("store", None)
        if options.get("tools"):
            options["tools"] = [
                (
                    tool
                    if "function" in tool
                    else {"type": "function", "function": {k: v for k, v in tool.items() if k != "type"}}
                )
                for tool in options["tools"]
            ]
        if "max_output_tokens" in options:
            options["max_completion_tokens"] = options.pop("max_output_tokens")
        if not options.get("tools"):
            options.pop("tools", None)
        if "max_tokens" in data["model"]["parameters"]:
            options.pop("max_completion_tokens", None)
        for turn in range(config["max_turns"]):
            reply = model_completion(
                data["model"], messages, root / "solver" / f"turn-{turn:03d}", {**options, **data["model"]["parameters"]}
            )
            choice = reply["choices"][0]
            assistant = choice["message"]
            action = assistant.get("content") or ""
            messages.append(assistant)
            usage = reply.get("usage") or {}
            evidence = RolloutEvidence(
                messages=tuple(copy.deepcopy(messages)),
                response=action,
                stop_reason=choice["finish_reason"],
                generated_token_count=usage.get("completion_tokens"),
                metadata={"assistant_message": assistant},
            )
            env.set_rollout_evidence(evidence)
            append_event(
                trace, {"event": "verify_start", "turn": turn, "action": action, "assistant_message": assistant}
            )
            with (root / "verifier.stdout").open("a") as stdout, (root / "verifier.stderr").open("a") as stderr:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    step = env.step(action)
            verdict = verification_from_env_step(step)
            verdicts.append(verdict)
            append_event(trace, {"event": "verify_end", "turn": turn, "output": step, "verification": verdict})
            done = step["done"]
            if done:
                break
            messages = (
                copy.deepcopy(step["reset_conversation"])
                if "reset_conversation" in step
                else messages + step["observations"]
            )
        verification, _ = fold_verification_results(verdicts)
        if not done:
            verification = VerificationResult.unavailable("Review turn cap exhausted before a terminal verdict")
        if task["env_id"] in {"preference", "prompt_only"}:
            verification = VerificationResult.unavailable(
                "Environment has a placeholder reward, not a native task verifier"
            )
        if any(v.diagnostics.get("cohort_reward_pending") for v in verdicts):
            verification = VerificationResult.unavailable(
                "GenRM evaluation requires a comparison cohort; placeholder reward discarded"
            )
        write_json(root / "solver-trace.json", {"messages": messages, "turns": turn + 1, "done": done})
        return {"verification": verification, "verifier_executed": bool(verdicts), "done": done, "turns": turn + 1}
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with native_calls() as called:
        try:
            result = run_task(json.loads(args.input.read_text()), args.output)
        except Exception as error:
            append_event(
                args.output / "verifier-trace.jsonl",
                {
                    "event": "execution_error",
                    "type": type(error).__name__,
                    "detail": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            result = {
                "verification": {
                    "status": "error",
                    "score": None,
                    "passed": None,
                    "reason": str(error),
                    "diagnostics": {"exception_type": type(error).__name__},
                },
                "verifier_executed": any(
                    json.loads(line)["event"] == "verify_start"
                    for line in (args.output / "verifier-trace.jsonl").read_text().splitlines()
                ),
                "done": False,
            }
        finally:
            capture_native_sources(args.output, called)
    write_json(args.output / "attempt.json", result)


if __name__ == "__main__":
    main()
