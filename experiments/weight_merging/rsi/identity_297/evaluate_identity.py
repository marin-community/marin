# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the shared dev scorer on Iris, publishing each suite before proceeding."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for
from rigging.secrets import require_secret

from experiments.interp.snowball_identity.identity_eval import cli

SCORER_REVISION = "2a1de41d86"
MODEL_REVISION = "59852d6f07261722bbec1edb6ab148317156b58e"
PROMPTS = Path(__file__).parent / "prompts"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model-revision", default=MODEL_REVISION, help="HF pin or immutable merge recipe revision.")
    args = parser.parse_args()
    fs, destination = filesystem_for(args.output.rstrip("/"))
    provenance = {
        "label": args.label,
        "dtype": args.dtype,
        "model_revision": args.model_revision,
        "scorer_revision": SCORER_REVISION,
        "split": "dev",
        "workers": args.workers,
        "prompt_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PROMPTS.iterdir())},
    }
    for suite in ("contrastive", "generate"):
        prefix = f"{destination}/{suite}"
        marker = f"{prefix}/done.json"
        if fs.exists(marker):
            previous = json.loads(fs.cat_file(marker))
            assert previous["provenance"] == provenance, "Refusing to reuse a different evaluation"
            for name, digest in previous["sha256"].items():
                assert hashlib.sha256(fs.cat_file(f"{prefix}/{name}")).hexdigest() == digest
            print(f"Verified completed suite: {suite}", flush=True)
            continue
        with tempfile.TemporaryDirectory() as directory:
            command = [
                suite,
                "--base-url",
                require_secret("IDENTITY_ENDPOINT"),
                "--label",
                args.label,
                "--out",
                directory,
                "--prompt-dir",
                str(PROMPTS),
                "--split",
                "dev",
                "--workers",
                str(args.workers),
            ]
            if suite == "contrastive":
                command.extend(["--tokenizer", "open-athena/Marin-Snowball-67B-A2B-v0.1"])
            cli.main(args=command, standalone_mode=False)
            hashes = {}
            for artifact in sorted(Path(directory).iterdir()):
                payload = artifact.read_bytes()
                fs.pipe_file(f"{prefix}/{artifact.name}", payload)
                assert fs.cat_file(f"{prefix}/{artifact.name}") == payload
                hashes[artifact.name] = hashlib.sha256(payload).hexdigest()
            fs.pipe_file(marker, json.dumps({"provenance": provenance, "sha256": hashes}, indent=2).encode())
            print(f"Published completed suite: {suite}", flush=True)


if __name__ == "__main__":
    main()
