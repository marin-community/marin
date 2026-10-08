# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay recorded streaming token trajectories to measure expert selections."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import requests

from experiments.weight_merging.routing_capture import summarize_routing


@dataclass(frozen=True)
class TokenTrajectory:
    request_id: str
    model: str
    prompt: list[int]
    generated: list[int]
    finish_reason: str

    @property
    def forwarded(self) -> list[int]:
        return self.prompt + self.generated[:-1]


def streaming_trajectory(response: bytes) -> TokenTrajectory:
    """Extract literal tokens from one completed native vLLM chat SSE response."""
    prompt = None
    generated = []
    request_id = None
    model = None
    finish_reason = None
    done = False
    usage = None
    for event in response.decode().replace("\r\n", "\n").split("\n\n"):
        data = "\n".join(line[5:].lstrip() for line in event.splitlines() if line.startswith("data:"))
        if not data:
            continue
        if done:
            raise ValueError("Response contains events after DONE")
        if data == "[DONE]":
            done = True
            continue
        chunk = json.loads(data)
        if request_id is None:
            request_id, model = chunk["id"], chunk["model"]
        if (chunk["id"], chunk["model"]) != (request_id, model):
            raise ValueError("Response mixes request identities")
        if chunk.get("prompt_token_ids") is not None:
            tokens = chunk["prompt_token_ids"]
            if prompt is not None and prompt != tokens:
                raise ValueError("Response changes prompt token IDs")
            prompt = tokens
        if chunk.get("usage") is not None:
            usage = chunk["usage"]
        for choice in chunk["choices"]:
            if choice["index"] != 0 or len(chunk["choices"]) != 1:
                raise ValueError("Replay requires one completion per request")
            generated.extend(choice.get("token_ids") or [])
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    if not done or not prompt or not generated or not finish_reason or request_id is None or model is None:
        raise ValueError("Incomplete streaming token trajectory")
    if usage is not None and (usage["prompt_tokens"], usage["completion_tokens"]) != (len(prompt), len(generated)):
        raise ValueError("Literal token counts disagree with response usage")
    return TokenTrajectory(request_id, model, prompt, generated, finish_reason)


def replay_request(trajectory: TokenTrajectory) -> dict:
    """Teacher-force the original forwarded tokens without rendering another prompt."""
    return {
        "model": trajectory.model,
        "prompt": trajectory.forwarded,
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
        "return_token_ids": True,
        "add_special_tokens": False,
        "routed_experts_prompt_start": 0,
    }


def summarize_replay(
    payload: dict, trajectory: TokenTrajectory, *, num_layers: int, num_experts: int, top_k: int
) -> dict:
    """Restore the original prompt/generation boundary after teacher-forced replay."""
    (choice,) = payload["choices"]
    if choice["prompt_token_ids"] != trajectory.forwarded or len(choice["token_ids"]) != 1:
        raise ValueError("Replay did not forward the exact recorded trajectory")
    normalized = {
        "id": trajectory.request_id,
        "model": trajectory.model,
        "prompt_token_ids": trajectory.prompt,
        "choices": [{**choice, "token_ids": trajectory.generated, "finish_reason": trajectory.finish_reason}],
    }
    result = summarize_routing(normalized, num_layers=num_layers, num_experts=num_experts, top_k=top_k)
    return {**result, "measurement": "teacher-forced replay", "replay_request_id": payload["id"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-layers", required=True, type=int)
    parser.add_argument("--num-experts", required=True, type=int)
    parser.add_argument("--top-k", required=True, type=int)
    parser.add_argument("--timeout", required=True, type=float)
    args = parser.parse_args()
    trajectory = streaming_trajectory(args.response.read_bytes())
    body = replay_request(trajectory)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "request.json").write_text(json.dumps(body) + "\n")
    response = requests.post(f"{args.base_url.rstrip('/')}/completions", json=body, timeout=args.timeout)
    response.raise_for_status()
    payload = response.json()
    (args.output / "response.json").write_text(json.dumps(payload) + "\n")
    summary = summarize_replay(
        payload, trajectory, num_layers=args.num_layers, num_experts=args.num_experts, top_k=args.top_k
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
