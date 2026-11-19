# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write the host-captured final text, then execute the unchanged source verifier."""

import json
import subprocess
from pathlib import Path


def write_answer(conversation_path: Path, answer_path: Path) -> None:
    conversation = json.loads(conversation_path.read_text())
    final = conversation[-1]
    if final["role"] != "assistant" or not isinstance(final["content"], str):
        raise ValueError("PDBThink requires one final assistant text")
    answer_path.write_text(final["content"])


def main():
    write_answer(Path("/tests/conversation.json"), Path("/app/answer.txt"))
    subprocess.run(("/bin/sh", "/tests/test.sh"), check=True)
    print(Path("/logs/verifier/details.json").read_text())


if __name__ == "__main__":
    main()
