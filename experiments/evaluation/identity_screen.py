# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture explicit identity-screen requests with independently verified native prefixes.

Example (the endpoint must already be running)::

    python -m experiments.evaluation.identity_screen --base-url http://host:port \\
        --model served-model --tokenizer /regional/tokenizer --requests cases.jsonl \\
        --rules rules.json --start-id 128002 --end-id 128003 --output fresh-results.jsonl

Each input row contains ``id`` and an unchanged OpenAI chat ``request`` with its
explicit ``chat_template_kwargs.enable_thinking``, ``temperature``, and ``max_tokens``.
Rules contain ``name_pattern``, ``creator_pattern`` and optional
``foreign_self_claim_pattern``. Pre-screen matches require manual adjudication.
No model is allocated, no identity system prompt or prefill is injected, and no
failed request is retried. Paths use fsspec; output must be new.
"""

import copy
import hashlib
import json
import re

import click
import fsspec
import requests
from marin.evaluation.identity import (
    IdentityRules,
    NativeCaptureError,
    capture_native_response,
    decode_views,
    prescreen,
)
from transformers import AutoTokenizer


@click.command()
@click.option("--base-url", required=True, help="Endpoint root, without /v1.")
@click.option("--model", required=True)
@click.option("--tokenizer", required=True)
@click.option(
    "--tokenizer-revision", default=None, help="Immutable 40-character commit, required for a remote HF tokenizer."
)
@click.option("--requests", "requests_path", required=True)
@click.option("--rules", "rules_path", required=True)
@click.option("--start-id", required=True, type=int)
@click.option("--end-id", required=True, type=int)
@click.option("--output", required=True)
@click.option("--timeout", default=900.0, type=float)
def main(base_url, model, tokenizer, tokenizer_revision, requests_path, rules_path, start_id, end_id, output, timeout):
    with fsspec.open(requests_path, "rb") as source:
        request_bytes = source.read()
    rows = [json.loads(line) for line in request_bytes.splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise click.ClickException("Screen requires nonempty unique row IDs")
    for row in rows:
        body = row["request"]
        if type(body["chat_template_kwargs"]["enable_thinking"]) is not bool:
            raise click.ClickException("Each request must explicitly bind thinking mode")
        if "temperature" not in body or "max_tokens" not in body:
            raise click.ClickException("Each request must explicitly bind decoding settings")
    with fsspec.open(rules_path, "rb") as source:
        rules_bytes = source.read()
    rules = IdentityRules(**json.loads(rules_bytes))
    for value in (rules.name_pattern, rules.creator_pattern, rules.foreign_self_claim_pattern):
        if value is not None:
            re.compile(value)
    # A local tokenizer directory is already content-bound by the caller. For HF,
    # require an explicit revision instead of silently resolving a moving default.
    if not fsspec.filesystem("file").isdir(tokenizer) and (
        tokenizer_revision is None or re.fullmatch(r"[0-9a-f]{40}", tokenizer_revision) is None
    ):
        raise click.ClickException("Remote tokenizer requires an immutable 40-character --tokenizer-revision")
    tok = AutoTokenizer.from_pretrained(tokenizer, revision=tokenizer_revision)
    failed = 0
    with fsspec.open(output, "xt") as target, requests.Session() as session:
        target.write(
            json.dumps(
                {
                    "record_type": "provenance",
                    "requests_sha256": hashlib.sha256(request_bytes).hexdigest(),
                    "rules_sha256": hashlib.sha256(rules_bytes).hexdigest(),
                    "tokenizer": tokenizer,
                    "tokenizer_revision": tokenizer_revision,
                    "start_id": start_id,
                    "end_id": end_id,
                }
            )
            + "\n"
        )
        for row in rows:
            body = copy.deepcopy(row["request"])
            body["model"] = model
            thinking = body["chat_template_kwargs"]["enable_thinking"]
            template = {
                **body["chat_template_kwargs"],
                "tokenize": True,
                "return_dict": False,
                "add_generation_prompt": body.get("add_generation_prompt", True),
                "continue_final_message": body.get("continue_final_message", False),
            }
            if "tools" in body:
                template["tools"] = body["tools"]
            prefix = tok.apply_chat_template(body["messages"], **template)
            record = {"record_type": "response", "id": row["id"], "source_request": row["request"]}
            try:
                record["capture"] = capture_native_response(
                    session, base_url, body, expected_prompt_ids=prefix, timeout=timeout
                )
                views = decode_views(
                    tok, record["capture"]["generated_ids"], thinking=thinking, start_id=start_id, end_id=end_id
                )
                record["views"] = views
                record["prescreen"] = prescreen(views["native_final_text"], rules)
                record["status"] = "response_received"
            except NativeCaptureError as error:
                failed += 1
                record.update(status="capture_failed", error=str(error), capture=error.evidence, prescreen=None)
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            target.flush()
    if failed:
        raise click.ClickException(f"{failed} native captures failed; evidence preserved, no retries")


if __name__ == "__main__":
    main()
