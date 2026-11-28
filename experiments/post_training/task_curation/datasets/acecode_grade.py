# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a code reply against the shipped AceCode assert cases: 1 when every case passes.

The candidate's last fenced block, or the full reply if there is no fence, runs in a
child process with the row's assert cases.
Candidate failures and timeouts score 0. Invalid grader inputs exit nonzero.
"""

import json
import multiprocessing
import os
import re
from contextlib import closing
from multiprocessing.connection import Connection
from pathlib import Path
from types import CodeType

CANDIDATE_TIMEOUT = 30.0  # Leave time for the grader to return a reward before its 330-second timeout.
CODE_BLOCK = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)
PASSED = b"1"
SETUP_FAILED = b"output redirection failed"


def run_candidate(code: str, cases: tuple[CodeType, ...], verdict: Connection, output_fd: int) -> None:
    try:
        os.dup2(output_fd, 1)  # Only the parent writes to the reward stream.
        os.dup2(output_fd, 2)
    except OSError:
        verdict.send_bytes(SETUP_FAILED)
        return
    namespace: dict = {}
    try:
        exec(code, namespace)
        for case in cases:
            exec(case, namespace)
    except (Exception, SystemExit, KeyboardInterrupt):
        verdict.send_bytes(b"0")
    else:
        verdict.send_bytes(PASSED)


def candidate_passes(code: str, cases: tuple[CodeType, ...], timeout: float) -> bool:
    """Return whether the candidate completed every case within the budget."""
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    with receiver, sender, open(os.devnull, "wb") as output:
        process = context.Process(target=run_candidate, args=(code, cases, sender, output.fileno()))
        with closing(process) as child:
            child.start()
            sender.close()
            try:
                if not receiver.poll(timeout):
                    return False
                result = receiver.recv_bytes()
            except EOFError:
                # The candidate process exited before reporting a verdict.
                return False
            finally:
                if child.is_alive():
                    child.kill()
                child.join()
    if result == SETUP_FAILED:
        raise OSError("Could not redirect AceCode candidate output")
    return result == PASSED


def main(timeout: float = CANDIDATE_TIMEOUT) -> None:
    config = json.loads(Path("/tests/config.json").read_text())
    cases = config["test_cases"]
    assert isinstance(cases, list) and cases, "the shipped case list must be nonempty"
    compiled_cases = tuple(compile(case, f"<AceCode case {index}>", "exec") for index, case in enumerate(cases, 1))

    answer = Path("/app/answer.txt").read_text()
    blocks = CODE_BLOCK.findall(answer)
    code = blocks[-1] if blocks else answer
    print(int(candidate_passes(code, compiled_cases, timeout)))


if __name__ == "__main__":
    main()
