# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Builds in a child process: the loader's test program, its model call served by ``fake_glm``.

Invariants: none (pins that a child build returns the in-process draft and each failure kind across the process).
"""

import asyncio
import json
import time
from dataclasses import asdict, replace

import pytest
from shellbox.machine import Backend

from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.ledger.records import EntryKind
from taskforge.llm.agent import ReasoningReplay
from taskforge.llm.glm_client import Pool
from taskforge.sandbox.factories import MachineHost
from taskforge.sandbox.image_builder import IrisImageBuilder, RegistryCredential
from taskforge.stage40_build_with_program.build_process import (
    LOG_FILE,
    BuildModelConfig,
    BuildRequest,
    BuildServiceConfig,
    image_builder,
    run_build_in_process,
)
from taskforge.stage40_build_with_program.context import (
    BuildExpired,
    BuildFailure,
    BuildInfrastructureFailure,
    ExpiredClock,
)
from taskforge.stage40_build_with_program.failure_text import BuildCrash
from taskforge.stage40_build_with_program.infrastructure import InfrastructureCause
from taskforge.stage40_build_with_program.program_sdk import load_program
from taskforge.stage40_build_with_program.run import ResearchRequirement
from taskforge.stage40_build_with_program.step_store import CacheStatus
from tests.stage30_author.programs import PROGRAM as PLAIN_PROGRAM
from tests.stage40_build_with_program.conftest import LIMITS, SAMPLING, WORKPIECE, build_draft

TOKEN_ENV = "TASKFORGE_TEST_GLM_TOKEN"
CONTROLLER_URL = "http://controller.example:10000"
TOKEN = "child-token"
REGISTRY_ENV = "TASKFORGE_TEST_REGISTRY_CREDENTIAL"
CREDENTIAL = RegistryCredential(registry="registry.example", user="publisher", password="registry-password")
WALL_TIMEOUT = 50.0
"""Long enough for the child to start and build; the tests expecting a kill set their own."""
SPIN_TIMEOUT = 8.0
QUESTION = '    return Question(text="What is 6 * 7?")'
ASKED = '    return await structured(context, [{"role": "user", "content": "Ask for 6 * 7."}], Question)'
PROGRAM = (
    PLAIN_PROGRAM.replace("from dataclasses import dataclass\n", "from pydantic import BaseModel\n")
    .replace("@dataclass(frozen=True)\nclass Question:", "class Question(BaseModel):")
    .replace("import ANSWER_FORMAT", "import ANSWER_FORMAT, structured")
    .replace(
        "import RLEnvBuildContext",
        "import BuildExpired, BuildFailure, BuildInfrastructureFailure, ExpiredClock, RLEnvBuildContext",
    )
    .replace(QUESTION, ASKED)
)


def with_question(body: str) -> str:
    """``PROGRAM`` with the body of its ``question`` step replaced by ``body``."""
    assert ASKED in PROGRAM
    return PROGRAM.replace(ASKED, body)


@pytest.fixture(autouse=True)
def token(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, TOKEN)


@pytest.fixture
def make_request(tmp_path, proposal, fake_glm, fast_provider):
    def make(source: str) -> BuildRequest:
        return BuildRequest(
            workpiece_id=WORKPIECE,
            round=0,
            program_source=source,
            proposal=proposal,
            round_dir=tmp_path / "child" / "round",
            store_root=str(tmp_path / "child" / "store"),
            model=BuildModelConfig(fake_glm.base_url, Pool.HIGH, TOKEN_ENV, fast_provider),
            sampling=SAMPLING,
            research=ResearchRequirement.OFF,
            invalidate=frozenset(),
            model_digest=None,
            services=BuildServiceConfig(
                host=MachineHost.LAPTOP,
                backends=(Backend.SHELLSIM,),
                controller_url=None,
                parallel_key_env=None,
                ledger_root=tmp_path / "child" / "ledger",
                reasoning_replay=ReasoningReplay.KEEP,
                agent_limits=LIMITS,
                image_cluster=None,
                registry_credential_env=None,
            ),
        )

    return make


def on_iris(request: BuildRequest, image_cluster: str | None) -> BuildRequest:
    """``request`` for a child in an Iris task, publishing images on ``image_cluster`` when it is set."""
    services = replace(
        request.services,
        host=MachineHost.IRIS,
        backends=(Backend.SHELLSIM, Backend.GVISOR),
        controller_url=CONTROLLER_URL,
        image_cluster=image_cluster,
        registry_credential_env=None if image_cluster is None else REGISTRY_ENV,
    )
    return replace(request, services=services)


async def test_a_build_in_a_child_process_writes_the_draft_and_ledger_of_the_build_in_process(
    tmp_path, proposal, services, glm_client, fake_glm, make_request
):
    for _ in range(2):
        fake_glm.stream(content=json.dumps({"text": "What is 6 * 7?"}))
    in_process = await build_draft(tmp_path / "here", load_program(PROGRAM), proposal, services, glm_client)
    request = make_request(PROGRAM)

    child = await run_build_in_process(request, wall_timeout=WALL_TIMEOUT)

    assert child == in_process
    assert {outcome.status for outcome in child.provenance.steps} == {CacheStatus.MISS}
    assert fake_glm.request_headers[1]["authorization"] == f"Bearer {TOKEN}"
    ledger = JsonlLedger(tmp_path / "child" / "ledger")
    entries = list(read_entries(ledger.path_for(WORKPIECE)))
    steps = [entry.step for entry in entries if entry.kind == EntryKind.STEP]
    # The ledger records steps as they finish and the provenance keeps program order; independent steps overlap.
    assert sorted(steps) == sorted(outcome.memo.name for outcome in child.provenance.steps)
    assert [entry.step for entry in entries if entry.kind == EntryKind.LLM_CALL] == ["question"]


async def test_a_step_that_never_yields_expires_at_the_wall_clock_while_the_driver_loop_runs(make_request):
    ticks = 0

    async def tick() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    ticker = asyncio.create_task(tick())
    request = make_request(with_question('    print("spinning", flush=True)\n    while True:\n        pass'))
    started = time.monotonic()

    with pytest.raises(BuildExpired) as expired:
        await run_build_in_process(request, wall_timeout=SPIN_TIMEOUT)

    elapsed = time.monotonic() - started
    ticker.cancel()
    assert (expired.value.clock, expired.value.step) == (ExpiredClock.WALL, None)
    assert str(expired.value) == f"the build ran for more than {SPIN_TIMEOUT:.0f} seconds"
    assert "spinning" in (request.round_dir / LOG_FILE).read_text()
    assert elapsed < SPIN_TIMEOUT + 5
    assert ticks > SPIN_TIMEOUT / 0.05 / 2


EXPIRY = "ran for more than 1800 seconds without returning"


@pytest.mark.parametrize(
    ("body", "error", "fields"),
    [
        (
            'raise BuildFailure("the question is ambiguous")',
            BuildFailure,
            {"message": "the question is ambiguous", "step": "question"},
        ),
        (
            f'raise BuildInfrastructureFailure("{InfrastructureCause.HOST_UNREACHABLE}", "the controller refused")',
            BuildInfrastructureFailure,
            {"cause": InfrastructureCause.HOST_UNREACHABLE, "message": "the controller refused"},
        ),
        (
            f'raise BuildExpired("{EXPIRY}", ExpiredClock.STEP)',
            BuildExpired,
            {"clock": ExpiredClock.STEP, "step": "question", "message": EXPIRY},
        ),
    ],
    ids=["build-failure", "host-failure", "step-clock"],
)
async def test_a_failure_in_the_child_comes_back_with_its_fields(make_request, body, error, fields):
    with pytest.raises(error) as failure:
        await run_build_in_process(make_request(with_question(f"    {body}")), wall_timeout=WALL_TIMEOUT)

    assert {name: getattr(failure.value, name) for name in fields} == fields


async def test_any_other_exception_in_the_child_comes_back_led_by_the_program_line_that_raised_it(make_request):
    source = with_question("    return 1 / 0")
    with pytest.raises(BuildCrash) as crash:
        await run_build_in_process(make_request(source), wall_timeout=WALL_TIMEOUT)

    line = source.splitlines().index("    return 1 / 0") + 1
    assert crash.value.head == f"program.py:{line}, in step question: return 1 / 0\nZeroDivisionError: division by zero"
    assert crash.value.traceback.startswith("Traceback (most recent call last)")
    assert "return 1 / 0" in crash.value.traceback


async def test_a_child_that_cannot_start_its_services_is_a_host_failure_carrying_its_log(make_request, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV)

    with pytest.raises(BuildInfrastructureFailure) as failure:
        await run_build_in_process(make_request(PROGRAM), wall_timeout=WALL_TIMEOUT)

    assert failure.value.cause == InfrastructureCause.BUILD_PROCESS
    assert f"the build reads a secret from {TOKEN_ENV}, which is not set" in failure.value.message


def test_the_child_builds_images_with_the_credential_its_environment_holds_and_no_gcloud(make_request, monkeypatch):
    config = on_iris(make_request(PROGRAM), "cw-us-east-02a").services
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv(REGISTRY_ENV, raising=False)
    with pytest.raises(ValueError, match=f"the build reads a secret from {REGISTRY_ENV}, which is not set"):
        image_builder(config)
    monkeypatch.setenv(REGISTRY_ENV, json.dumps({"registry": "registry.example", "user": "publisher"}))
    with pytest.raises(ValueError, match="not a JSON object of strings"):
        image_builder(config)
    monkeypatch.setenv(REGISTRY_ENV, json.dumps(asdict(CREDENTIAL)))

    builder = image_builder(config)

    assert isinstance(builder, IrisImageBuilder)
    assert (builder.controller_url, builder.target_cluster, builder.credential) == (
        CONTROLLER_URL,
        "cw-us-east-02a",
        CREDENTIAL,
    )
    assert image_builder(make_request(PROGRAM).services) is None
