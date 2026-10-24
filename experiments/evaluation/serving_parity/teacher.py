# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect exact teacher-forced token identities from OpenAI completions endpoints."""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import requests
from marin.testing.inference.sequence_parity import ScoredSequence, ScoredToken


def normalize_teacher_response(response: dict, expected_ids: list[int]) -> tuple[ScoredToken, ...]:
    """Reject missing/extra echo tokens rather than score a generated suffix as prompt."""
    choice = response["choices"][0]
    echo = choice["logprobs"]
    rows = []
    for token, logprob, top in zip(echo["tokens"], echo["token_logprobs"], echo["top_logprobs"], strict=True):
        if not token.startswith("token_id:"):
            raise ValueError("Teacher responses require return_tokens_as_token_ids")
        top_token = max(top, key=top.get) if top else None
        if top_token is not None and not top_token.startswith("token_id:"):
            raise ValueError("Ambiguous top token identity")
        rows.append(
            ScoredToken(
                int(token.removeprefix("token_id:")),
                logprob,
                int(top_token.removeprefix("token_id:")) if top_token else None,
            )
        )
    if [row.token_id for row in rows] != expected_ids:
        raise ValueError("Echo token IDs differ from frozen teacher input")
    if response["usage"]["completion_tokens"] != 0:
        raise ValueError("Teacher probe unexpectedly generated tokens")
    return tuple(rows)


def collect_teacher(endpoint: str, model: str, item: dict, output: Path, *, concurrency: int) -> list[ScoredSequence]:
    """Run identical frozen teacher input serially or concurrently; retain raw responses."""
    output.mkdir(parents=True, exist_ok=False)
    text = item["prompt"] + item["completion"]
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != item["input_sha256"]:
        raise ValueError("Teacher text checksum differs from frozen manifest")
    payload = dict(
        model=model,
        prompt=text,
        max_tokens=0,
        echo=True,
        logprobs=1,
        temperature=0,
        add_special_tokens=False,
        return_tokens_as_token_ids=True,
        return_token_ids=True,
    )

    def collect(index: int) -> ScoredSequence:
        started = time.monotonic()
        response = requests.post(endpoint.rstrip("/") + "/completions", json=payload, timeout=180)
        # Do not embed a capability-bearing URL in an HTTPError/log artifact.
        if response.status_code != 200:
            raise RuntimeError(f"Teacher request returned HTTP {response.status_code}")
        raw = response.json()
        tokens = normalize_teacher_response(raw, item["token_ids"])
        scored = ScoredSequence(str(item["id"]), digest, tokens)
        (output / f"{index}.raw.json").write_text(json.dumps(raw))
        (output / f"{index}.json").write_text(json.dumps(asdict(scored)))
        (output / f"{index}.timing.json").write_text(json.dumps({"seconds": time.monotonic() - started}))
        return scored

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(collect, range(concurrency)))
