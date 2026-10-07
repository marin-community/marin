# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

from pydantic import BaseModel
from rigging.timing import ExponentialBackoff

from taskforge.llm.client import GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.store import CallStore
from taskforge.llm.structured import StructuredTool


class Capital(BaseModel):
    city: str


TOOL = StructuredTool(name="answer", description="Record the answer", output_type=Capital)


def run_store(fake_glm, root, calls):
    async def go():
        endpoint = GlmEndpoint(base_url=fake_glm.base_url, token="test-token", pool=Pool.HIGH)
        async with GlmClient(endpoint, backoff=ExponentialBackoff(initial=0.001, maximum=0.001)) as client:
            store = CallStore(root, client)
            return [await call(store) for call in calls]

    return asyncio.run(go())


def ask(text, policy=LLMPolicy(), sample=0):
    return lambda store: store.complete("draft", [{"role": "user", "content": text}], policy, sample=sample)


def test_identical_request_is_served_from_disk(fake_glm, tmp_path):
    fake_glm.stream(content="first answer", reasoning="r")
    first, again = run_store(fake_glm, tmp_path, [ask("q"), ask("q")])
    assert len(fake_glm.requests) == 1
    assert again == first
    [item] = (tmp_path / "items" / "draft").iterdir()
    assert sorted(p.name for p in item.iterdir()) == ["request.json", "response.json", "result.json"]
    assert "test-token" not in (item / "request.json").read_text()


def test_any_request_difference_misses_the_cache(fake_glm, tmp_path):
    for content in ("a", "b", "c"):
        fake_glm.stream(content=content)
    results = run_store(fake_glm, tmp_path, [ask("q"), ask("other q"), ask("q", LLMPolicy(temperature=0.1))])
    assert [r.content for r in results] == ["a", "b", "c"]


def test_samples_of_one_request_are_cached_separately_across_restarts(fake_glm, tmp_path):
    for content in ("a", "b"):
        fake_glm.stream(content=content)
    first = run_store(fake_glm, tmp_path, [ask("q", sample=0), ask("q", sample=1)])
    again = run_store(fake_glm, tmp_path, [ask("q", sample=0), ask("q", sample=1)])
    assert [c.content for c in first] == [c.content for c in again] == ["a", "b"]
    assert len(fake_glm.requests) == 2


def test_stall_timeout_does_not_change_the_key(fake_glm, tmp_path):
    fake_glm.stream(content="a")
    run_store(fake_glm, tmp_path, [ask("q"), ask("q", LLMPolicy(stall_timeout=5.0))])
    assert len(fake_glm.requests) == 1


def test_structured_value_is_cached_typed(fake_glm, tmp_path):
    fake_glm.stream(tool_calls=(("answer", '{"city": "Paris"}'),))
    messages = [{"role": "user", "content": "Capital of France?"}]
    first, again = run_store(fake_glm, tmp_path, [lambda s: s.structured("capital", messages, LLMPolicy(), TOOL)] * 2)
    assert again.value == first.value == Capital(city="Paris")
    assert len(fake_glm.requests) == 1
    [item] = (tmp_path / "items" / "capital").iterdir()
    assert json.loads((item / "result.json").read_text())["value"] == {"city": "Paris"}
