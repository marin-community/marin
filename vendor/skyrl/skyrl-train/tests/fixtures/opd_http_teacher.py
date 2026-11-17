"""Deterministic OpenAI-compatible teacher test server."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from aiohttp import web


ChoiceFactory = Callable[[int, list[int]], dict[str, Any]]


def completion_choice(index: int, sequence: list[int], logprob: float) -> dict[str, Any]:
    return {
        "index": index,
        "text": "",
        "finish_reason": "length",
        "logprobs": {
            "tokens": [f"token_id:{token_id}" for token_id in sequence],
            "token_logprobs": [None, *([logprob] * (len(sequence) - 1))],
            "top_logprobs": [None, *({} for _ in sequence[1:])],
            "text_offset": [0] * len(sequence),
        },
    }


def application(
    logprob: float,
    *,
    requests: list[dict[str, Any]] | None = None,
    bearer_token: str | None = None,
    choice_factory: ChoiceFactory | None = None,
) -> web.Application:
    """Return a server that assigns one fixed logprob to every supplied token."""

    async def completions(request: web.Request) -> web.Response:
        if bearer_token is not None and request.headers.get("Authorization") != f"Bearer {bearer_token}":
            return web.json_response({"error": "unauthorized"}, status=401)
        body = await request.json()
        prompts = body.get("prompt")
        if not isinstance(prompts, list) or not all(
            isinstance(sequence, list) and sequence and all(isinstance(token, int) for token in sequence)
            for sequence in prompts
        ):
            return web.json_response({"error": "prompt must be a non-empty batch of token-ID sequences"}, status=400)
        if requests is not None:
            requests.append(body)
        make_choice = choice_factory or (lambda index, sequence: completion_choice(index, sequence, logprob))
        choices = [make_choice(index, sequence) for index, sequence in enumerate(prompts)]
        return web.json_response({"choices": choices})

    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--logprob", type=float, default=-0.25)
    args = parser.parse_args()
    web.run_app(application(args.logprob), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
