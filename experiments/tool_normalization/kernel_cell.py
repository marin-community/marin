# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Example client written into the workspace by a visible Bash setup call.

This executes stdin in a real local IPython kernel. It is not a model-facing
tool and is not silently installed: a converted demonstration must include its
file creation and kernel startup before invoking it through Bash.
"""

import sys
import time

from jupyter_client import BlockingKernelClient


def main() -> None:
    client = BlockingKernelClient(connection_file=sys.argv[1])
    client.load_connection_file()
    client.start_channels()
    failed = False
    try:
        client.wait_for_ready(timeout=30)
        request = client.execute(sys.stdin.read(), store_history=True, stop_on_error=True)
        deadline = time.monotonic() + 120
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Kernel cell did not finish")
            message = client.get_iopub_msg(timeout=remaining)
            if message.get("parent_header", {}).get("msg_id") != request:
                continue
            content = message["content"]
            kind = message["msg_type"]
            if kind == "stream":
                stream = sys.stderr if content["name"] == "stderr" else sys.stdout
                print(content["text"], end="", file=stream, flush=True)
            elif kind in ("execute_result", "display_data"):
                if "text/plain" not in content["data"] or any(name.startswith("image/") for name in content["data"]):
                    raise ValueError("Non-text kernel output requires a different grounded conversion")
                print(content["data"]["text/plain"], flush=True)
            elif kind == "error":
                print("\n".join(content["traceback"]), file=sys.stderr, flush=True)
                failed = True
            elif kind == "status" and content["execution_state"] == "idle":
                break
    finally:
        client.stop_channels()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
