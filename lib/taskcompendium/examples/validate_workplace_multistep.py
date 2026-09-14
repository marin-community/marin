# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write an inspectable all-good native Harbor run for the derived Workplace workflow."""

import argparse
import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from taskcompendium.execution import HarborExecutionConfig, HarborLaunchConfig
from taskcompendium.harbor.runner import run_trial
from taskcompendium.importers.nemo_workplace_multistep import build_multistep_sample
from taskcompendium.lowering import lower_to_harbor


def _assistant_call(call, call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
        ],
    }


async def validate(output_root: Path, fixture_root: Path) -> None:
    """Run the local scripted agent and persist its package, trial results, and HTTP trace."""
    if output_root.exists():
        raise FileExistsError(f"Validation output already exists: {output_root}")
    sample = build_multistep_sample(fixture_root)
    requests: list[dict] = []
    responses: list[dict] = []
    for step_index, calls in enumerate(sample.all_good, start=1):
        responses.extend(
            _assistant_call(call, f"step-{step_index}-call-{call_index}") for call_index, call in enumerate(calls, 1)
        )
        responses.append({"role": "assistant", "content": "Completed."})

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            message = responses[len(requests) - 1]
            body = json.dumps(
                {
                    "id": f"chatcmpl-{len(requests)}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "scripted-fixture",
                    "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        output_root.mkdir(parents=True)
        task = lower_to_harbor(
            sample.specification,
            sample.renderings,
            sample.binding,
            output_root / "task",
            reference_execution=HarborExecutionConfig(sample.binding, HarborLaunchConfig("provider_chat")),
            model_name="scripted-fixture",
            agent_kwargs={"api_base": f"http://127.0.0.1:{server.server_port}/v1", "max_turns": 4},
        )
        result = await run_trial(
            task, json.loads((task / "reference-execution.json").read_text()), output_root / "trials", "all-good"
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    (output_root / "http-requests.json").write_text(json.dumps(requests, indent=2) + "\n")
    (output_root / "result.json").write_text(result.model_dump_json(indent=2) + "\n")
    if result.exception_info is not None or result.verifier_result is None:
        raise RuntimeError("Native Harbor validation did not produce an aggregate result")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/taskcompendium-workplace-multistep-validation"))
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).parents[1] / "tests/fixtures/nemo")
    args = parser.parse_args()
    asyncio.run(validate(args.output_root, args.fixtures))


if __name__ == "__main__":
    main()
