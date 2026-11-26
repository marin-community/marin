# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import gzip
import hashlib
import json
import math
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from aiohttp import web
from iris.cluster.types import ResourceSpec, gpu_device
from marin.evaluation.model_config import ServeBackend
from marin.execution.artifact import read_artifact
from marin.inference.config import VllmCompilationCacheMode
from marin.inference.vllm_server import VllmEnvironment
from marin.testing.inference.backend_parity import TokenDistribution, distribution_difference
from marin.testing.inference.endpoint import VllmEndpoint, response_distribution, response_top_distribution
from marin.testing.inference.endpoint_checks import (
    TestSpec as InferenceTestSpec,
)
from marin.testing.inference.endpoint_checks import (
    check_batch,
    check_cache,
    check_decode,
    check_reference,
)
from marin.testing.inference.fake_vllm_server import FakeVllmLauncher
from marin.testing.inference.goldens import (
    ArtifactRef,
    ErrorLimits,
    LookupStatus,
    ModelIdentity,
    ReferenceCase,
    ReferenceManifest,
    ScoreRow,
    canonical_bytes,
    history_digest,
    lookup_reference,
)
from marin.testing.inference.results import CheckReport, CheckStatus

RUNTIME = "0123456789abcdef0123456789abcdef01234567"
LOGPROBS = {token: math.log(0.5 if token == 1 else 0.3 if token == 2 else 0.2 / 14) for token in range(16)}


def test_prompt_token_outside_top_k_does_not_understate_probability_error():
    reference = [0.019] * 50 + [1 / 600] * 30
    tail = [0.01] + [1 / 600] * 28 + [0.000001]
    actual = [(1 - sum(tail)) / 50] * 50 + tail
    wire = {str(token): {"logprob": math.log(actual[token]), "rank": token + 1} for token in range(50)}
    wire["79"] = {"logprob": math.log(actual[79]), "rank": 80}
    expected = TokenDistribution(tuple(range(64)), tuple(math.log(value) for value in reference[:64]))

    difference = distribution_difference(expected, response_top_distribution(wire))
    # The full normalized distributions provide an independent error oracle.
    true_error = max(abs(a - b) for a, b in zip(reference, actual, strict=True))
    assert difference.max_probability_error_lower_bound <= true_error <= difference.max_probability_error
    # Decode rescoring still needs the observed prompt token outside the top k.
    returned = response_distribution(wire)
    assert dict(zip(returned.token_ids, returned.logprobs, strict=True))[79] == pytest.approx(math.log(actual[79]))


def artifact(path: Path, value: object) -> ArtifactRef:
    raw = canonical_bytes(value)
    path.write_bytes(raw)
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(raw).hexdigest())


@pytest.fixture
def reference_fixture(tmp_path) -> InferenceTestSpec:
    model_path = tmp_path / "model"
    model_path.mkdir()
    config = b'{"dtype":"bfloat16"}'
    (model_path / "config.json").write_bytes(config)
    identity = ModelIdentity(
        weights=str(model_path),
        revision="test-revision",
        config_sha256=hashlib.sha256(config).hexdigest(),
        tokenizer="test-tokenizer",
        tokenizer_revision="tokenizer-revision",
        eos_token_id=0,
    )
    cases = [
        ReferenceCase(id=f"case-{i}", tokens=tuple((n % 13) + 3 for n in range(length)), positions=(length - 1,))
        for i, length in enumerate([5, 7, 10, 80])
    ]
    prompts = artifact(tmp_path / "cases.json", [case.model_dump(mode="json") for case in cases])
    scores = artifact(
        tmp_path / "scores.json",
        [
            ScoreRow(
                case_id=case.id,
                position=case.positions[0],
                token_ids=tuple(LOGPROBS),
                logprobs=tuple(LOGPROBS.values()),
                history_sha256=history_digest(case.tokens),
            ).model_dump(mode="json")
            for case in cases
        ],
    )
    manifest = ReferenceManifest(
        model=identity,
        suite_id=prompts.sha256,
        cases=prompts,
        scores=scores,
        reference_source="independent-fixture",
        reference_commit="fixture-revision",
        reference_config={},
        lineage={},
    )
    pointer = artifact(tmp_path / "manifest.json", manifest.model_dump(mode="json"))
    limits = ErrorLimits(maximum=0.01, p99=0.01, p90=0.01, fraction_above_0075=0)
    return InferenceTestSpec(
        id="protocol-fixture",
        reference=pointer,
        resources=ResourceSpec(device=gpu_device("HTTP test GPU")),
        serving_engine=ServeBackend.VLLM,
        engine_kwargs={"dtype": "bfloat16", "quantization": None},
        hardware=("HTTP test GPU",),
        tolerances={
            "reference.representative": limits,
            "cache.same_execution": limits,
            "cache.changed_prefill": limits,
            "batch": limits,
            "decode": limits,
        },
        tolerance_version="fixture-policy",
        provisional=True,
        rationale="Protocol fixture",
        evidence=(),
        boundary_lengths=(7, 8, 9),
    )


def model_identity(spec: InferenceTestSpec) -> ModelIdentity:
    references = lookup_reference(spec.reference)
    assert references.manifest is not None
    return references.manifest.model


def requests(report: CheckReport):
    return [row for name in ("reference", "batch", "cache", "decode") for row in getattr(report, name).requests]


def test_spec_model_identity_reaches_native_launcher(tmp_path, unused_tcp_port, reference_fixture):
    identity = model_identity(reference_fixture)
    received = tmp_path / "received-argv.json"
    with VllmEnvironment(
        reference_fixture.make_model_config(),
        port=unused_tcp_port,
        launcher=FakeVllmLauncher("record-args", str(received)),
        compilation_cache_mode=VllmCompilationCacheMode.CALLER_MANAGED,
    ):
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("model")
        parser.add_argument("--revision")
        parser.add_argument("--tokenizer")
        parser.add_argument("--tokenizer-revision")
        parser.add_argument("--override-generation-config", type=json.loads)
        arguments, _ = parser.parse_known_args(json.loads(received.read_bytes()))
        assert arguments.model == identity.weights
        assert arguments.revision == identity.revision
        assert arguments.tokenizer == identity.tokenizer
        assert arguments.tokenizer_revision == identity.tokenizer_revision
        assert arguments.override_generation_config["eos_token_id"] == identity.eos_token_id


@dataclass
class HttpEngine:
    mode: str
    identity: ModelIdentity
    retention_interval: int | None = None
    prefixes: set[tuple[int, ...]] = field(default_factory=set)
    histories: set[tuple[int, ...]] = field(default_factory=set)
    corrupted: set[tuple[int, ...]] = field(default_factory=set)
    active_batch: int = 0
    observed_batch: asyncio.Event = field(default_factory=asyncio.Event)

    async def complete(self, request: web.Request) -> web.Response:
        body = await request.json()
        assert body["model"] == "test-model"
        if self.mode == "reference-error" and request.headers["X-Request-Id"] == "reference-case-1":
            return web.json_response({"error": "Reference request interrupted"}, status=503)
        if self.mode == "cache-error" and request.headers["X-Request-Id"] == "cache-7-fresh":
            return web.json_response({"error": "Cache request interrupted"}, status=503)
        tokens = tuple(body["prompt"])
        count = body["max_tokens"]
        assert request.headers["X-data-parallel-rank"] == "0"
        cached = max(
            (len(prefix) for prefix in self.prefixes if len(prefix) < len(tokens) and tokens[: len(prefix)] == prefix),
            default=0,
        )
        # Inject a sibling write after the prepared branch has been observed.
        if self.mode == "branch-corrupt" and request.headers["X-Request-Id"].endswith("-settled-alternate"):
            for previous in self.histories:
                if len(previous) != len(tokens):
                    continue
                shared = next((i for i, (a, b) in enumerate(zip(previous, tokens, strict=True)) if a != b), len(tokens))
                protected = max(
                    (len(p) for p in self.prefixes if len(p) < len(previous) and previous[: len(p)] == p), default=0
                )
                if 0 < shared < protected:
                    self.corrupted.add(previous)
        self.histories.add(tokens)
        is_batch = request.headers["X-Request-Id"].startswith("batch-")
        if is_batch:
            self.active_batch += 1
            # The real metrics request releases these responses after observing the
            # engine. This avoids a timer race in the HTTP fixture.
            await self.observed_batch.wait()
        probabilities = dict(LOGPROBS)
        if self.mode == "bad-scores" or (cached and (self.mode == "corrupt-cache" or tokens in self.corrupted)):
            probabilities[1] = math.log(0.1)
            probabilities[2] = math.log(0.7)
        if self.mode == "non-finite":
            probabilities[1] = float("nan")
        top = {f"token_id:{token}": value for token, value in probabilities.items()}
        winner = max(probabilities, key=probabilities.get)
        generated = [winner] * count
        computed = len(tokens) - 1 if self.mode == "prefill-only" else len(tokens) + count - 1
        if self.retention_interval is None:
            retained = set(range(4, computed + 1, 4))
        elif self.retention_interval:
            retained = set(range(self.retention_interval, computed + 1, self.retention_interval))
        else:
            retained = set()
        # Align snapshots occur at scheduler ends. A cold aligned prompt can
        # finish beyond the replay point retained by interval zero.
        if self.retention_interval != 0 or len(tokens) % 4:
            retained.add((len(tokens) - 1) // 4 * 4)
        self.prefixes.update(tuple((*tokens, *generated)[:length]) for length in retained if length > 0)
        if is_batch:
            self.active_batch -= 1
        choice = {"token_ids": generated, "prompt_token_ids": list(tokens), "logprobs": {"top_logprobs": [top] * count}}
        if body.get("prompt_logprobs"):
            prompt_top = {
                token: {"logprob": value, "rank": index + 1} for index, (token, value) in enumerate(top.items())
            }
            choice["prompt_logprobs"] = [None, *[prompt_top for _ in tokens[1:]]]
        return web.json_response(
            {
                "choices": [choice],
                "usage": {
                    "prompt_tokens_details": {
                        "cached_tokens": (
                            0
                            if self.mode == "no-cache"
                            or (
                                self.mode == "no-branch-reuse"
                                and request.headers["X-Request-Id"].endswith("-settled-alternate")
                            )
                            else (
                                min(cached, tokens.index(1) // 4 * 4)
                                if self.mode == "no-generated-branch-reuse"
                                and 1 in tokens
                                and request.headers["X-Request-Id"].endswith("-settled-alternate")
                                else cached
                            )
                        )
                    }
                },
            }
        )

    async def metrics(self, _request):
        running = self.active_batch if self.mode != "no-batch" else min(1, self.active_batch)
        if self.active_batch >= 4:
            self.observed_batch.set()
        return web.Response(text=f'vllm:num_requests_running{{engine="0"}} {running}\n')

    async def reset(self, _request):
        self.prefixes.clear()
        self.histories.clear()
        self.corrupted.clear()
        return web.json_response(True)


@asynccontextmanager
async def http_endpoint(engine: HttpEngine):
    app = web.Application()
    app.router.add_post("/v1/completions", engine.complete)
    app.router.add_get("/metrics", engine.metrics)
    app.router.add_post("/reset_prefix_cache", engine.reset)

    async def info(_request):
        streamed = engine.mode in ("streamed-model", "wrong-weights")
        return web.json_response(
            {
                "system_env": {"nvidia_gpu_models": "HTTP test GPU"},
                "vllm_config": {
                    "model_config": {
                        "model": "/local/model-streamer/metadata" if streamed else engine.identity.weights,
                        "model_weights": (
                            ("/wrong-checkpoint" if engine.mode == "wrong-weights" else engine.identity.weights)
                            if streamed
                            else ""
                        ),
                        "revision": "test-revision",
                        "tokenizer": "test-tokenizer",
                        "tokenizer_revision": "tokenizer-revision",
                        "dtype": "bfloat16",
                        "max_model_len": 8192,
                        "override_generation_config": {"eos_token_id": 0},
                    },
                    "scheduler_config": {},
                    "cache_config": {
                        "enable_prefix_caching": True,
                        "cache_dtype": "auto",
                        "block_size": 4,
                        "prefix_cache_retention_interval": engine.retention_interval,
                    },
                    "parallel_config": {},
                    "compilation_config": {"cudagraph_mode": "PIECEWISE"},
                },
            }
        )

    async def version(_request):
        return web.json_response({"version": f"0.0.0+marin.{RUNTIME[:12]}"})

    async def models(_request):
        return web.json_response({"data": [{"id": "test-model"}]})

    app.router.add_get("/server_info", info)
    app.router.add_get("/version", version)
    app.router.add_get("/v1/models", models)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0]
    try:
        yield f"http://{host}:{port}/v1"
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,expected_status,affected_check",
    [
        ("correct", "passed", None),
        ("streamed-model", "passed", None),
        ("wrong-weights", "failed", "reference"),
        ("bad-scores", "failed", "reference"),
        ("non-finite", "failed", "reference"),
        ("reference-error", "failed", "reference"),
        ("cache-error", "failed", "cache"),
        ("prefill-only", "incomplete", "cache"),
        ("no-cache", "incomplete", "cache"),
        ("no-branch-reuse", "incomplete", "cache"),
        ("no-generated-branch-reuse", "incomplete", "cache"),
        ("no-batch", "incomplete", "batch"),
        ("corrupt-cache", "failed", "cache"),
        ("branch-corrupt", "failed", "cache"),
    ],
)
async def test_http_suite_requires_numerics_and_exercised_paths(
    tmp_path, reference_fixture, mode, expected_status, affected_check
):
    spec = reference_fixture
    if mode == "branch-corrupt":
        spec = spec.model_copy(
            update={
                "tolerances": (
                    spec.tolerances
                    | {
                        "cache.changed_prefill": ErrorLimits(maximum=1, p99=1, p90=1, fraction_above_0075=1),
                        "cache.same_execution": ErrorLimits(maximum=0, p99=0, p90=0, fraction_above_0075=0),
                    }
                )
            }
        )
    async with http_endpoint(HttpEngine(mode, model_identity(spec))) as url:
        async with VllmEndpoint(url) as endpoint:
            report = await spec.run(endpoint)
    assert report.status == expected_status
    output = str(tmp_path / "saved-report")
    report.save(output)
    assert read_artifact(output, CheckReport) == report
    if affected_check:
        assert getattr(report, affected_check).status == expected_status
        if mode == "wrong-weights":
            assert requests(report) == []
            assert report.reference.details["identity"]["mismatches"]["weights"]
            return
        if mode == "reference-error":
            reference = report.reference
            assert reference.details["completed_cases"] == ["case-0", "case-2", "case-3"]
            assert reference.details["total_cases"] == 4
            assert [failure["case"] for failure in reference.details["failures"]] == ["case-1"]
            assert reference.differences["reference.representative"].positions == 3
            failed = [row for row in reference.requests if row.http_status == 503]
            assert json.loads(gzip.decompress(failed[0].body_gzip))["error"] == "Reference request interrupted"
        if mode == "cache-error":
            cache = report.cache
            assert [failure["boundary"] for failure in cache.details["failures"]] == [7]
            assert [row["boundary"] for row in cache.details["cache_evidence"]] == [7, 8, 9]
            assert [row["boundary"] for row in cache.coverage["decode_reuse"].observations] == [8, 9]
            assert cache.coverage["matched_repeat"].status == "passed"
        if mode == "no-cache":
            cache = report.cache
            assert cache.coverage["prefill_reuse"].status == "incomplete"
            assert len(cache.coverage["decode_reuse"].observations) == len(spec.boundary_lengths)
            assert cache.differences["cache.changed_prefill"].positions == len(spec.boundary_lengths)
        if mode == "prefill-only":
            cache = report.cache
            assert cache.coverage["decode_reuse"].status == "incomplete"
            assert len(cache.details["cache_evidence"]) == len(spec.boundary_lengths)
            assert cache.differences["cache.same_execution"].positions > 0
            assert cache.differences["cache.changed_prefill"].positions >= len(spec.boundary_lengths)
        if mode == "no-generated-branch-reuse":
            cache = report.cache
            assert cache.coverage["settled_branches"].status == "passed"
            assert cache.coverage["replayed_decode_reuse"].status == "incomplete"
            for row in cache.coverage["replayed_decode_reuse"].observations:
                assert 0 < row["alternate_cached_tokens"] < row["required_alternate_checkpoint"]
        if mode == "branch-corrupt":
            cache = report.cache
            assert cache.differences["cache.changed_prefill"].status == "passed"
            assert cache.coverage["settled_branches"].status == "passed"
            assert cache.differences["cache.same_execution"].status == "failed"
    else:
        assert report.batch.details["running_peak"] >= 2
        assert report.cache.details["cache_evidence"][0]["warm_cached_tokens"] > 0
        assert report.cache.coverage["settled_branches"].status == "passed"
        assert report.decode.differences["decode"].positions == 64
    assert requests(report)
    for row in requests(report):
        assert hashlib.sha256(gzip.decompress(row.body_gzip)).hexdigest() == row.response_sha256


@pytest.mark.asyncio
async def test_python_composition_keeps_checks_in_memory_until_save(tmp_path, monkeypatch, reference_fixture):
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    async with http_endpoint(HttpEngine("correct", model_identity(reference_fixture))) as url:
        async with VllmEndpoint(url) as endpoint:
            # Caller provenance differs deliberately from the observed server version.
            endpoint.metadata["runtime_source"] = "caller-declared-runtime"
            report = CheckReport(
                reference=await check_reference(endpoint, reference_fixture),
                batch=await check_batch(endpoint, reference_fixture),
                cache=await check_cache(endpoint, reference_fixture),
                decode=await check_decode(endpoint, reference_fixture),
            )
    assert report.status == CheckStatus.PASSED
    assert list(working.iterdir()) == []
    assert report.reference.run_metadata["runtime_source"] == "caller-declared-runtime"
    assert report.reference.run_metadata["version"]["version"] == f"0.0.0+marin.{RUNTIME[:12]}"
    report.save(str(working / "report"))
    restored = read_artifact(str(working / "report"), CheckReport)
    assert restored == report
    assert json.loads(gzip.decompress(restored.reference.requests[0].body_gzip))["choices"]


@pytest.mark.asyncio
@pytest.mark.parametrize("retention", [None, 0, 128, 512])
async def test_cache_contract_covers_retention_and_replayed_generation(reference_fixture, retention):
    async with http_endpoint(
        HttpEngine("correct", model_identity(reference_fixture), retention_interval=retention)
    ) as url:
        async with VllmEndpoint(url) as endpoint:
            report = await reference_fixture.run(endpoint)
    assert report.status == CheckStatus.PASSED
    cache = report.cache
    direct = cache.coverage["decode_reuse"]
    assert direct.status == ("not_required" if retention == 0 else "passed")
    for row in direct.observations:
        if retention == 0:
            assert row["cached_tokens"] <= row["prompt_tokens"]
            assert row["required_checkpoint"] is None
        else:
            assert row["cached_tokens"] >= row["required_checkpoint"] > row["prompt_tokens"]
    replayed = cache.coverage["replayed_decode_reuse"]
    assert replayed.status == "passed"
    for row, initial in zip(replayed.observations, direct.observations, strict=True):
        assert row["repeat_cached_tokens"] == row["expected_cached_tokens"] > initial["prompt_tokens"]
        assert initial["prompt_tokens"] < row["required_alternate_checkpoint"] < row["expected_cached_tokens"]
        assert row["alternate_cached_tokens"] == row["required_alternate_checkpoint"]
    for row in cache.coverage["prefill_reuse"].observations:
        assert min(row["cached_tokens"]) >= row["required_checkpoint"] > 0
    assert cache.differences["cache.same_execution"].statistics.maximum == 0
    assert cache.differences["cache.same_execution"].positions == 2 * len(reference_fixture.boundary_lengths) + 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,expected_status,affected_check,retention",
    [
        ("correct", "passed", None, None),
        ("correct", "passed", None, 0),
        ("no-cache", "incomplete", "cache", None),
        ("branch-corrupt", "failed", "cache", 0),
        ("wrong-weights", "failed", "reference", None),
        ("bad-scores", "failed", "reference", None),
        ("reference-error", "failed", "reference", None),
        ("missing-reference", "incomplete", None, None),
        ("mismatched-reference", "failed", "reference", None),
    ],
)
async def test_endpoint_adapters_agree_with_retained_outcome(
    tmp_path, reference_fixture, mode, expected_status, affected_check, retention
):
    identity = model_identity(reference_fixture)
    if mode == "mismatched-reference":
        reference_fixture = reference_fixture.model_copy(
            update={"reference": reference_fixture.reference.model_copy(update={"sha256": "0" * 64})}
        )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(reference_fixture.model_dump_json())
    if mode == "missing-reference":
        (tmp_path / "scores.json").unlink()
    for adapter in ("cli", "pytest"):
        output = tmp_path / f"{adapter}-report"
        async with http_endpoint(HttpEngine(mode, identity, retention_interval=retention)) as url:
            arguments = (
                [
                    "marin.testing.inference.check_endpoint",
                    "--spec",
                    str(spec_path),
                    "--url",
                    url,
                    "--output",
                    str(output),
                ]
                if adapter == "cli"
                else [
                    "pytest",
                    "--confcutdir=tests/cluster/vllm",
                    "tests/cluster/vllm/test_inference_correctness.py",
                    "-o",
                    "addopts=",
                    "-o",
                    f"cache_dir={tmp_path / 'pytest-cache'}",
                    "--endpoint-url",
                    url,
                    "--correctness-spec",
                    str(spec_path),
                    "--correctness-output",
                    str(output),
                ]
            )
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            assert process.returncode == (0 if expected_status == "passed" else 1), (stdout.decode(), stderr.decode())
            report = read_artifact(str(output), CheckReport)
            assert report.status == expected_status
            if adapter == "cli":
                assert json.loads(stdout)["status"] == expected_status
            if affected_check:
                assert getattr(report, affected_check).status == expected_status


def test_lookup_distinguishes_missing_and_modified_immutable_reference(tmp_path, reference_fixture):
    assert lookup_reference(reference_fixture.reference).status == LookupStatus.FOUND
    wrong = reference_fixture.reference.model_copy(update={"sha256": "0" * 64})
    assert lookup_reference(wrong).status == LookupStatus.MISMATCHED
    (tmp_path / "scores.json").write_text("[]")
    assert lookup_reference(reference_fixture.reference).status == LookupStatus.MISMATCHED
    (tmp_path / "scores.json").unlink()
    assert lookup_reference(reference_fixture.reference).status == LookupStatus.MISSING
