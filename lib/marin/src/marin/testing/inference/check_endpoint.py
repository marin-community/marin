# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check an existing vLLM endpoint against saved reference evidence."""

import argparse
import asyncio
import json
from pathlib import Path

from marin.testing.inference.endpoint_checks import check_endpoint
from marin.testing.inference.goldens import LookupStatus, lookup_reference, read_fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, help="Path to a JSON fixture")
    parser.add_argument("--lookup", action="store_true", help="Verify saved artifacts without contacting a server")
    parser.add_argument("--url", help="Dedicated vLLM endpoint URL")
    parser.add_argument("--model", help="Served model name")
    parser.add_argument("--runtime-source", help="Expected vLLM source commit")
    parser.add_argument("--output", type=Path, default=Path("correctness.json"))
    args = parser.parse_args()
    fixture = read_fixture(args.fixture)
    if args.lookup:
        found = lookup_reference(fixture)
        result = {
            "status": found.status,
            "reason": found.reason,
            "manifest": fixture.manifest.model_dump(),
            "histories": len(found.cases),
            "positions": len(found.scores),
        }
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if found.status == LookupStatus.FOUND else 1)
    if not args.url or not args.model or not args.runtime_source:
        parser.error("Supply --url, --model and --runtime-source")
    report = asyncio.run(check_endpoint(args.url, args.model, fixture, args.runtime_source, args.output))
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(args.output),
                "policy": fixture.policy.id,
                "provisional": fixture.policy.provisional,
                "checks": {name: row["status"] for name, row in report["checks"].items()},
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
