# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded GLM conversion and native execution of controlled tool fixtures."""

import argparse
import asyncio
import json
import re
import shlex
from pathlib import Path

from transformers import AutoTokenizer

from experiments.tool_normalization.candidate import CALL_SCHEMA
from experiments.tool_normalization.execute import ExecutionConfig, execute
from experiments.tool_normalization.model import ModelConfig, complete, initialize_ledger
from experiments.tool_normalization.native_seed import native_seed

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {"calls": {"type": "array", "items": CALL_SCHEMA}, "rationale": {"type": "string"}},
    "required": ["calls", "rationale"],
    "additionalProperties": False,
}
AUDIT_SCHEMA = {
    "type": "object",
    "properties": {"decision": {"enum": ["pass", "revise"]}, "rationale": {"type": "string"}},
    "required": ["decision", "rationale"],
    "additionalProperties": False,
}
CONVERT = """Convert these controlled legacy tool operations to the supplied real native tools.
This is an isolated adapter test, not an original repository task. Preserve each operation's semantics.
execute_bash becomes bash. Python computations must run through bash, retaining state between cells.
Use the supplied tested native seed for stateful cells, including its explicit setup. Do not substitute answers.
Convert ONLY the listed legacy operations. Fixture setup and verification are run by the harness, never by your
proposal. Do not create or rewrite files to make a view operation succeed. The requested file already exists.
str_replace_editor str_replace becomes native edit, not a Bash file rewrite. The fixture supplies initial files
and reads them before your operations. Return only calls and rationale; never invent observations."""
AUDIT = """Review this isolated tool adapter test: legacy operations, proposed native calls and actual receipts.
Check semantic mapping, Python state, native edit behavior and the independent fixture checks. Do not solve
or require a larger repository task. Fixture outputs apply only to this fixture, not an original dataset trace.
Pass only if all tool operations are faithfully mapped and the reported checks agree with real observations."""
REVIEW_SEED = """Review this exact mechanical conversion of controlled legacy tool operations to native calls.
Check that every operation retains its semantics and that Python state persists through the explicit Bash kernel
setup. The runner will execute the supplied calls unchanged if you approve; do not regenerate command text.
This is Stage 1. A separate Stage 2 will audit real execution receipts. Return pass with no issues only if the
mapping is correct; otherwise return revise with the concrete problem. Explain the decision in rationale.
No larger task needs to be solved."""


def bash(command: str) -> dict:
    return {"name": "bash", "arguments": {"command": command}}


def fixtures() -> list[dict]:
    return [
        {
            "name": "shell_alias",
            "legacy": [{"name": "execute_bash", "arguments": {"command": "python3 -c 'print(6 * 7)'"}}],
            "setup": [],
            "checks": [],
            "expected_last_output": "42",
        },
        {
            "name": "python_state",
            "legacy": [
                {"name": "stateful_python_code_exec", "arguments": {"code": "counter = 6"}},
                {"name": "stateful_python_code_exec", "arguments": {"code": "counter += 7\nprint(counter)"}},
            ],
            "setup": [],
            "checks": [],
            "expected_last_output": "13",
        },
        {
            "name": "native_edit",
            "legacy": [
                {
                    "name": "str_replace_editor",
                    "arguments": {
                        "command": "str_replace",
                        "path": "/workspace/edit.txt",
                        "old_str": "before",
                        "new_str": "after",
                    },
                }
            ],
            "setup": [
                bash("printf 'before\\n' > /workspace/edit.txt"),
                {"name": "read", "arguments": {"filePath": "/workspace/edit.txt"}},
            ],
            "checks": [
                bash(
                    'python3 -c "from pathlib import Path; '
                    "assert Path('/workspace/edit.txt').read_text() == 'after\\n'; print('fixture_pass')\""
                )
            ],
            "expected_last_output": "fixture_pass",
        },
    ]


def run(config_path: Path, root: Path, fixture_path: Path | None = None) -> None:
    root.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(config_path.read_text())
    model_config = cfg["model"]
    for key in ("api_key_file", "capacity_file", "ledger"):
        model_config[key] = Path(model_config[key])
    model = ModelConfig(**model_config)
    initialize_ledger(model)
    tokenizer = AutoTokenizer.from_pretrained(cfg["serving_tokenizer"])
    execution = ExecutionConfig(**cfg["execution"])
    schemas = json.loads(Path(__file__).with_name("opencode-tools.json").read_text())
    results = []
    selected = json.loads(fixture_path.read_text()) if fixture_path is not None else fixtures()
    for fixture in selected:
        directory = root / fixture["name"]
        directory.mkdir()
        (directory / "fixture.json").write_text(json.dumps(fixture, indent=2))
        source = [
            {"role": "assistant", "content": "<tool_call>" + json.dumps(call) + "</tool_call>"}
            for call in fixture["legacy"]
        ]
        seed = native_seed(source, {"jupyter_client"})
        if seed is not None:
            review = complete(
                model,
                tokenizer,
                directory / "convert",
                REVIEW_SEED,
                {"fixture": fixture, "tools": schemas, "native_seed": seed},
                AUDIT_SCHEMA,
                reasoning_effort="low",
            )
            if review["decision"] != "pass":
                raise ValueError("Known mapping requires review; no automatic rewrite")
            proposal = {
                "calls": [call for turn in seed["turns"] for call in turn["calls"]],
                "rationale": seed["rationale"],
            }
        else:
            proposal = complete(
                model,
                tokenizer,
                directory / "convert",
                CONVERT,
                {
                    "legacy": fixture["legacy"],
                    "semantics": fixture.get("semantics", "Preserve the legacy operation semantics."),
                    "tools": schemas,
                },
                PROPOSAL_SCHEMA,
                reasoning_effort="low",
            )
        (directory / "proposal.json").write_text(json.dumps(proposal, indent=2))
        calls = proposal["calls"]
        if not calls or any(call["name"] not in {"bash", "edit", "read"} for call in calls):
            raise ValueError("Adapter emitted unsupported operations")
        if fixture["name"] == "native_edit" and not any(call["name"] == "edit" for call in calls):
            raise ValueError("Editor was not mapped to native edit")
        observed = asyncio.run(execute(fixture["setup"] + calls + fixture["checks"], execution, directory / "execution"))
        events = json.loads((directory / "execution/stdout.json").read_text())["tool_events"]
        assert all(e["part"]["state"]["status"] == "completed" for e in events), "Tool operation failed"
        assert all(e["part"]["state"].get("metadata", {}).get("exit", 0) == 0 for e in events), "Command failed"
        check_fixture_output(
            fixture, observed[len(fixture["setup"]) + len(calls) - 1]["content"], observed[-1]["content"]
        )
        boundary_checks = {}
        if fixture["name"] == "editor_create":
            boundary_checks = asyncio.run(
                existing_file_contract(
                    calls, execution, directory / "existing-file", fixture["legacy"][0]["arguments"]["path"]
                )
            )
        if fixture["name"] == "editor_view_range":
            boundary_checks = asyncio.run(view_contract(calls, execution, directory / "changed-view"))
        audit = complete(
            model,
            tokenizer,
            directory / "audit",
            AUDIT,
            {
                "fixture": fixture,
                "proposal": proposal,
                "observations": observed,
                "native_events": events,
                "boundary_checks": boundary_checks,
            },
            AUDIT_SCHEMA,
            reasoning_effort="high",
        )
        results.append({"fixture": fixture["name"], "passed": audit["decision"] == "pass", "audit": audit})
        (root / "summary.json").write_text(json.dumps({"fixtures": results, "complete": False}, indent=2))
        if not results[-1]["passed"]:
            raise ValueError("Fixture review needs revision; no automatic retry")
    asyncio.run(error_contracts(execution, root / "error-contracts"))
    (root / "summary.json").write_text(
        json.dumps(
            {"fixtures": results, "error_contracts_passed": True, "complete": True, "scope": "controlled fixtures only"},
            indent=2,
        )
    )


def check_fixture_output(fixture: dict, operation_output: str, final_output: str) -> None:
    if "expected_numbered_lines" in fixture:
        lines = re.findall(r"^\s*(\d+)(?:\t|: )(.*)$", operation_output, re.MULTILINE)
        assert lines == list(fixture["expected_numbered_lines"].items()), "Numbered view differs from requested range"
        return
    assert final_output.strip() == fixture["expected_last_output"], "Fixture output mismatch"


async def view_contract(calls: list[dict], execution: ExecutionConfig, directory: Path) -> dict:
    """Reject view proposals that recreate known fixture contents or ignore the file."""
    content = "first changed\nsecond changed\nthird changed\nfourth changed\n"
    setup = f"from pathlib import Path; Path('/workspace/view.txt').write_text({content!r})"
    check = (
        f"from pathlib import Path; assert Path('/workspace/view.txt').read_text() == {content!r}; print('preserved')"
    )
    observed = await execute(
        [bash("python3 -c " + shlex.quote(setup)), *calls, bash("python3 -c " + shlex.quote(check))],
        execution,
        directory,
    )
    events = json.loads((directory / "stdout.json").read_text())["tool_events"]
    assert all(e["part"]["state"]["status"] == "completed" for e in events), "View operation failed"
    assert all(e["part"]["state"].get("metadata", {}).get("exit", 0) == 0 for e in events), "View modified file contents"
    check_fixture_output(
        {"expected_numbered_lines": {"2": "second changed", "3": "third changed"}},
        observed[-2]["content"],
        observed[-1]["content"],
    )
    result = {"passed": True, "observations": observed, "native_events": events}
    (directory / "checks.json").write_text(json.dumps(result, indent=2))
    return result


async def existing_file_contract(calls: list[dict], execution: ExecutionConfig, directory: Path, path: str) -> dict:
    """Require a proposed create operation to fail without overwriting existing bytes."""
    sentinel = "existing file: preserve this different content\n"
    setup = f"from pathlib import Path; Path({path!r}).write_text({sentinel!r})"
    check = f"from pathlib import Path; assert Path({path!r}).read_text() == {sentinel!r}; print('preserved')"
    observed = await execute(
        [bash("python3 -c " + shlex.quote(setup)), *calls, bash("python3 -c " + shlex.quote(check))],
        execution,
        directory,
    )
    events = json.loads((directory / "stdout.json").read_text())["tool_events"]
    failed = [
        event["part"]["state"]["status"] == "error" or event["part"]["state"].get("metadata", {}).get("exit", 0) != 0
        for event in events
    ]
    assert not failed[0] and not failed[-1], "Existing-file fixture setup or preservation failed"
    assert any(failed[1:-1]), "Create did not report failure for an existing file"
    assert observed[-1]["content"].strip() == "preserved", "Create overwrote existing content"
    result = {"passed": True, "observations": observed, "native_events": events}
    (directory / "checks.json").write_text(json.dumps(result, indent=2))
    return result


async def error_contracts(execution: ExecutionConfig, directory: Path) -> None:
    calls = [
        bash("exit 7"),
        bash("printf 'x\\nx\\n' > /workspace/repeated.txt"),
        {"name": "read", "arguments": {"filePath": "/workspace/repeated.txt"}},
        {"name": "edit", "arguments": {"filePath": "/workspace/repeated.txt", "oldString": "x", "newString": "y"}},
        {
            "name": "edit",
            "arguments": {"filePath": "/workspace/repeated.txt", "oldString": "x", "newString": "y", "replaceAll": True},
        },
        {"name": "edit", "arguments": {"filePath": "/workspace/repeated.txt", "oldString": "absent", "newString": "z"}},
        bash("cat /workspace/repeated.txt"),
    ]
    observed = await execute(calls, execution, directory)
    events = json.loads((directory / "stdout.json").read_text())["tool_events"]
    assert events[0]["part"]["state"]["metadata"]["exit"] == 7, "Nonzero command exit was hidden"
    assert events[3]["part"]["state"]["status"] == "error", "Ambiguous edit incorrectly succeeded"
    assert events[4]["part"]["state"]["status"] == "completed", "Explicit replaceAll failed"
    assert events[5]["part"]["state"]["status"] == "error", "Missing edit target incorrectly succeeded"
    assert observed[-1]["content"] == "y\ny\n", "Failed edit changed file contents"
    (directory / "checks.json").write_text(
        json.dumps(
            {
                "passed": True,
                "checks": ["nonzero_exit", "ambiguous_edit", "replace_all", "missing_target", "final_contents"],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, help="Authored isolated fixtures; defaults to the original three")
    args = parser.parse_args()
    run(args.config, args.root, args.fixtures)


if __name__ == "__main__":
    main()
