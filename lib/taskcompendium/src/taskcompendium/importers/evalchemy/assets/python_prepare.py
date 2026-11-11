# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the private source test program without executing the candidate."""

import json
from pathlib import Path

import numpy as np  # noqa: F401 - fail preparation if the declared test dependency is missing
from code_protocol import test_program


def main() -> None:
    config = json.loads((Path(__file__).parent / "config.json").read_text())
    response = Path("/app/answer.txt").read_text()
    program = test_program(config["problem"], response, config["suite"], config.get("direction", ""))
    # The receipt and exception handler reject early exits, including os._exit(0).
    wrapper = (
        "import json\n"
        "import multiprocessing\n"
        "import signal\n"
        "from pathlib import Path\n"
        "import numpy\n"
        "\n"
        "def timeout_handler(signum, frame):\n"
        "    raise TimeoutError('Candidate timed out')\n"
        "\n"
        "def execute(connection):\n"
        "    signal.signal(signal.SIGALRM, timeout_handler)\n"
        f"    signal.setitimer(signal.ITIMER_REAL, {10 if config['suite'] == 'cruxeval' else 3})\n"
        "    try:\n"
        f"        exec(compile({program!r}, '<candidate>', 'exec'), {{}})\n"
        "    except BaseException:\n"
        "        passed = False\n"
        "    else:\n"
        "        passed = True\n"
        "    finally:\n"
        "        signal.setitimer(signal.ITIMER_REAL, 0)\n"
        "    connection.send(passed)\n"
        "\n"
        "context = multiprocessing.get_context('fork')\n"
        "receiver, sender = context.Pipe(duplex=False)\n"
        "process = context.Process(target=execute, args=(sender,))\n"
        "process.start()\n"
        "sender.close()\n"
        f"process.join({11 if config['suite'] == 'cruxeval' else 4})\n"
        "if process.is_alive():\n"
        "    process.kill()\n"
        "    process.join()\n"
        "passed = False\n"
        "if receiver.poll():\n"
        "    try:\n"
        "        passed = receiver.recv() is True and process.exitcode == 0\n"
        "    except EOFError:\n"
        "        pass\n"
        "Path('/app/reward.json').write_text(json.dumps({'reward': float(passed)}))\n"
    )
    Path("/app/program.py").write_text(wrapper)


if __name__ == "__main__":
    main()
