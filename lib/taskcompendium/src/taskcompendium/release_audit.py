# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate complete demonstration artifacts through their reconstructed Harbor tasks."""

import hashlib
import importlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

from taskcompendium.harbor.runner import ChatLaunch, run_trial
from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.importers.nemo_workplace import PROVIDER, workplace_environment_config
from taskcompendium.lowering import HarborEnvironmentConfig, lower_to_harbor, selected_tool_definitions
from taskcompendium.mixed_release import PublishedRow, published_task
from taskcompendium.models import VerifierKind
from taskcompendium.provider_sources import PROVIDER_SOURCES_DIR, ToolProviderCache
from taskcompendium.submission import PlainText, ProviderState, SubmissionConvention, chat_request


async def _harbor_reward(
    row: PublishedRow,
    convention: SubmissionConvention,
    environment: HarborEnvironmentConfig,
    actions: list[dict[str, Any]],
    answer: str,
    directory: Path,
    trusted_checkout: Path,
) -> float:
    task = published_task(row)
    task_dir = lower_to_harbor(
        task,
        convention,
        environment,
        directory / "task",
        trusted_provider_sources={"workplace": trusted_checkout} if task.tool_providers else {},
    )
    pending = iter(enumerate(actions))
    request_matches = []
    expected_request = {**chat_request(task, convention), "model": "release-audit"}
    provider_tools = []
    with ToolProviderCache() as cache:
        for name, binding in environment.tool_providers.items():
            staged = task_dir / "environment" / PROVIDER_SOURCES_DIR / name
            provider = cache.stage(binding.provider, staged).factory
            module = importlib.import_module(provider.__module__)
            provider_tools.extend(selected_tool_definitions(module.TOOL_DEFINITIONS, binding.tools))
    tools = [*provider_tools, *expected_request.get("tools", [])]
    if tools:
        expected_request["tools"] = tools
    prefix = expected_request["messages"]

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib override signature
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            request_matches.append(
                request["messages"][: len(prefix)] == prefix
                and {key: value for key, value in request.items() if key != "messages"}
                == {key: value for key, value in expected_request.items() if key != "messages"}
            )
            selected = next(pending, None)
            message = {"role": "assistant", "content": answer}
            if selected is not None:
                index, action = selected
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": f"audit-call-{index}", "type": "function", "function": action}],
                }
            body = json.dumps({"choices": [{"message": message}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = await run_trial(
            task_dir,
            environment,
            ChatLaunch(
                model="release-audit", api_base=f"http://127.0.0.1:{server.server_port}/v1", max_turns=len(actions) + 2
            ),
            directory / "trials",
            "reconstructed",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    if not request_matches or not all(request_matches):
        raise ValueError("Harbor request differs from the reviewed conversation, tools, or request surface")
    if result.exception_info is not None or result.verifier_result is None or result.verifier_result.rewards is None:
        raise ValueError("Reconstructed demonstration Harbor trial failed")
    return result.verifier_result.rewards["reward"]


async def audit_demonstration(
    ready: Path, source_dir: Path, provider_source: Path, trusted_checkout: Path
) -> dict[str, Any]:
    """Check every Workplace expected state and sampled correct/wrong reconstructed trials."""
    with ToolProviderCache() as cache:
        provider_class = cache.stage(PROVIDER, provider_source).factory
        provider = importlib.import_module(provider_class.__module__)
        environment = workplace_environment_config(provider_source)
        seed = json.loads(provider.expected_state_json([]))
        state_matches = {}
        outcomes = {}
        for split in ("train", "validation"):
            sample = None
            matches = 0
            with (
                (ready / f"data/workplace/{split}.jsonl").open(encoding="utf-8") as published,
                (source_dir / f"{split}.jsonl").open(encoding="utf-8") as source,
            ):
                for exported, original in zip(published, source, strict=True):
                    row = PublishedRow.model_validate_json(exported)
                    source_row = json.loads(original)
                    identity = f"{split}:{source_row["id"]}:{hashlib.sha256(original.encode()).hexdigest()}"
                    if row.source.row != identity:
                        raise ValueError("Published Workplace row differs from its source identity")
                    actions = [
                        {"name": action["name"], "arguments": action["arguments"]}
                        for action in source_row["ground_truth"]
                    ]
                    expected = json.loads(provider.expected_state_json(actions))
                    if row.verifier.kind is not VerifierKind.STRUCTURED_EXACT:
                        raise ValueError("Workplace demonstration lost its structured state verifier")
                    if canonical_sha256(json.loads(row.verifier.parameters_json)) != canonical_sha256(
                        {"expected": expected}
                    ):
                        raise ValueError("Published Workplace verifier differs from source canonical state")
                    matches += 1
                    if sample is None and canonical_sha256({"state": expected}) != canonical_sha256({"state": seed}):
                        sample = (row, actions)
            if sample is None:
                raise ValueError("Workplace split has no state-changing reconstruction sample")
            row, actions = sample
            convention = ProviderState(id="state", provider="workplace")
            rewards = []
            for name, calls in (("correct", actions), ("wrong", [])):
                reward = await _harbor_reward(
                    row, convention, environment, calls, "Done.", source_dir / "audit" / split / name, trusted_checkout
                )
                rewards.append(reward)
            if rewards != [1.0, 0.0]:
                raise ValueError("Reconstructed Workplace correct/wrong grading gate failed")
            state_matches[split] = matches
            outcomes[f"workplace/{split}"] = rewards
        for cohort, kind in (("mcqa", VerifierKind.MCQ_ANSWER), ("prism_math", VerifierKind.MATHEMATICAL_ANSWER)):
            with (ready / f"data/tasktrove_clean/{cohort}.jsonl").open(encoding="utf-8") as stream:
                row = PublishedRow.model_validate_json(next(stream))
            if row.verifier.kind is not kind:
                raise ValueError("TaskTrove demonstration has an unexpected verifier")
            parameters = json.loads(row.verifier.parameters_json)
            correct = parameters["expected"]
            wrong = "A" if kind is VerifierKind.MCQ_ANSWER and correct != "A" else "B"
            if kind is VerifierKind.MATHEMATICAL_ANSWER:
                wrong = "not a mathematical answer"
            rewards = []
            for name, answer in (("correct", correct), ("wrong", wrong)):
                rewards.append(
                    await _harbor_reward(
                        row,
                        PlainText(id="plain"),
                        HarborEnvironmentConfig(),
                        [],
                        answer,
                        source_dir / "audit" / cohort / name,
                        trusted_checkout,
                    )
                )
            if rewards != [1.0, 0.0]:
                raise ValueError("Reconstructed TaskTrove correct/wrong grading gate failed")
            outcomes[f"tasktrove_clean/{cohort}"] = rewards
        return {"workplace_expected_state_matches": state_matches, "harbor_rewards": outcomes, "harbor_trials": 8}
