# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HTTP access to a caller-owned vLLM server, with observations kept in memory."""

import gzip
import hashlib
import json
import time
from dataclasses import dataclass, field

import aiohttp
from pydantic import ConfigDict, JsonValue

from marin.testing.inference.backend_parity import TokenDistribution
from marin.testing.inference.goldens import Record, history_digest

TOP_K = 50
HTTP_TIMEOUT = aiohttp.ClientTimeout(total=900)
RANK_HEADER = "X-data-parallel-rank"


class HttpObservation(Record):
    """One completion response; ``body_gzip`` retains its exact uncompressed bytes."""

    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")

    label: str
    history_sha256: str
    prompt_tokens: int
    rank: int
    requested_tokens: int
    prompt_scores: bool
    http_status: int
    response_sha256: str
    seconds: float
    body_gzip: bytes


def response_distribution(entries: dict) -> TokenDistribution:
    return TokenDistribution.from_logprob_map(
        {
            int(str(token).removeprefix("token_id:")): float(value["logprob"] if isinstance(value, dict) else value)
            for token, value in entries.items()
        }
    )


def response_top_distribution(entries: dict) -> TokenDistribution:
    """Exclude an appended prompt token below the requested top-k cutoff."""
    scores = response_distribution(entries)
    return TokenDistribution(scores.token_ids[:TOP_K], scores.logprobs[:TOP_K])


@dataclass
class VllmEndpoint:
    """Use the server's default model unless an HTTP model name is supplied.

    The context manages HTTP connections only. It neither deploys the server nor
    writes files. Completion bodies, including HTTP failures, remain available
    for the caller's report until the endpoint is discarded.
    """

    base_url: str
    model: str | None = None
    session: aiohttp.ClientSession = field(init=False)
    requests: list[HttpObservation] = field(default_factory=list, init=False)
    metadata: dict[str, JsonValue] = field(default_factory=dict, init=False)
    _information: dict | None = field(default=None, init=False, repr=False)

    async def __aenter__(self) -> "VllmEndpoint":
        # Parsing and compressing prompt scores can outlast the server keepalive.
        self.session = aiohttp.ClientSession(
            timeout=HTTP_TIMEOUT, connector=aiohttp.TCPConnector(limit=0, force_close=True)
        )
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.session.close()

    @property
    def root(self) -> str:
        return self.base_url.rstrip("/").removesuffix("/v1")

    async def get(self, path: str) -> dict:
        async with self.session.get(f"{self.root}/{path}") as response:
            response.raise_for_status()
            return await response.json()

    async def server_info(self) -> dict:
        if self._information is None:
            self._information = await self.get("server_info?config_format=json")
            self.metadata["version"] = await self.get("version")
        return self._information

    async def reset_cache(self) -> None:
        async with self.session.post(f"{self.root}/reset_prefix_cache") as response:
            response.raise_for_status()

    async def metrics(self) -> str:
        async with self.session.get(f"{self.root}/metrics") as response:
            response.raise_for_status()
            return await response.text()

    async def complete(
        self,
        tokens: tuple[int, ...],
        *,
        label: str,
        count: int = 1,
        prompt_scores: bool = False,
        rank: int = 0,
    ) -> dict:
        if self.model is None:
            models = await self.get("v1/models")
            self.model = models["data"][0]["id"]
        self.metadata["request_model"] = self.model
        body = {
            "model": self.model,
            "prompt": tokens,
            "add_special_tokens": False,
            "max_tokens": count,
            "temperature": 0,
            "ignore_eos": True,
            "logprobs": TOP_K,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
        }
        if prompt_scores:
            body["prompt_logprobs"] = TOP_K
        started = time.time()
        async with self.session.post(
            f"{self.root}/v1/completions", json=body, headers={RANK_HEADER: str(rank), "X-Request-Id": label}
        ) as response:
            raw = await response.read()
            compressed = gzip.compress(raw, mtime=0)
            self.requests.append(
                HttpObservation(
                    label=label,
                    history_sha256=history_digest(tokens),
                    prompt_tokens=len(tokens),
                    rank=rank,
                    requested_tokens=count,
                    prompt_scores=prompt_scores,
                    http_status=response.status,
                    response_sha256=hashlib.sha256(raw).hexdigest(),
                    seconds=time.time() - started,
                    body_gzip=compressed,
                )
            )
            response.raise_for_status()
        result = json.loads(raw)
        (choice,) = result["choices"]
        if choice.get("prompt_token_ids") != list(tokens):
            raise ValueError(f"The endpoint changed the exact input tokens: {label}")
        if len(choice["token_ids"]) != count:
            raise ValueError(f"The endpoint returned an incomplete continuation: {label}")
        for token, top in zip(choice["token_ids"], choice["logprobs"]["top_logprobs"], strict=True):
            distribution = response_distribution(top)
            if int(token) not in distribution.token_ids or int(token) != distribution.token_ids[0]:
                scores = dict(zip(distribution.token_ids, distribution.logprobs, strict=True))
                if scores.get(int(token)) != max(scores.values()):
                    raise ValueError(f"The greedy token lacks its finite maximum score: {label}")
        return result
