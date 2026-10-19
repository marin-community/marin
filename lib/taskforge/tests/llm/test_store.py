# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import pytest
from pydantic import BaseModel

from taskforge.llm.glm_client import GlmClient
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.store import CallStore
from tests.conftest import fake_source


class Capital(BaseModel):
    city: str


@pytest.fixture
def run_store(fake_glm, fast_provider):
    """Make ``calls`` in order through one ``CallStore`` under ``root``."""

    def go(root, calls):
        async def call_all():
            async with GlmClient(fake_source(fake_glm), fast_provider) as client:
                store = CallStore(root, client)
                return [await call(store) for call in calls]

        return asyncio.run(call_all())

    return go


def ask(text, policy=LLMPolicy(), sample=0, request_fields=None):
    return lambda store: store.complete(
        "draft", [{"role": "user", "content": text}], policy, request_fields=request_fields or {}, sample=sample
    )


def test_identical_request_is_served_from_disk(fake_glm, tmp_path, run_store):
    fake_glm.stream(content="first answer", reasoning="r")
    first, again = run_store(tmp_path, [ask("q"), ask("q")])
    assert len(fake_glm.requests) == 1
    assert again == first
    [item] = (tmp_path / "items" / "draft").iterdir()
    assert sorted(p.name for p in item.iterdir()) == ["request.json", "response.json", "result.json"]
    assert "test-token" not in (item / "request.json").read_text()


def test_any_request_difference_misses_the_cache(fake_glm, tmp_path, run_store):
    for content in ("a", "b", "c"):
        fake_glm.stream(content=content)
    results = run_store(tmp_path, [ask("q"), ask("other q"), ask("q", LLMPolicy(temperature=0.1))])
    assert [r.content for r in results] == ["a", "b", "c"]


def test_samples_of_one_request_are_cached_separately_across_restarts(fake_glm, tmp_path, run_store):
    for content in ("a", "b"):
        fake_glm.stream(content=content)
    first = run_store(tmp_path, [ask("q", sample=0), ask("q", sample=1)])
    again = run_store(tmp_path, [ask("q", sample=0), ask("q", sample=1)])
    assert [c.content for c in first] == [c.content for c in again] == ["a", "b"]
    assert len(fake_glm.requests) == 2


def test_structured_value_is_cached_typed(fake_glm, tmp_path, run_store):
    fake_glm.stream(content='{"city": "Paris"}')
    messages = [{"role": "user", "content": "Capital of France?"}]
    first, again = run_store(tmp_path, [lambda s: s.structured("capital", messages, Capital, LLMPolicy())] * 2)
    assert again.value == first.value == Capital(city="Paris")
    assert len(fake_glm.requests) == 1
    [item] = (tmp_path / "items" / "capital").iterdir()
    assert json.loads((item / "result.json").read_text())["value"] == {"city": "Paris"}


def test_request_fields_are_part_of_the_key(fake_glm, tmp_path, run_store):
    regex = {"structured_outputs": {"regex": "a+"}}
    for content in ("free", "aaa"):
        fake_glm.stream(content=content)
    results = run_store(tmp_path, [ask("q"), ask("q", request_fields=regex), ask("q", request_fields=regex)])
    assert [r.content for r in results] == ["free", "aaa", "aaa"]
    assert len(fake_glm.requests) == 2
    assert fake_glm.requests[1]["structured_outputs"] == {"regex": "a+"}
