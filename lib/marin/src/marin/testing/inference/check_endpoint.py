# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check an existing vLLM endpoint against saved reference evidence."""

import argparse
import asyncio
import json

from marin.testing.inference.endpoint import VllmEndpoint
from marin.testing.inference.endpoint_checks import TestSpec, read_test_spec
from marin.testing.inference.goldens import LookupStatus, lookup_reference
from marin.testing.inference.results import CheckReport, CheckStatus


async def run(spec: TestSpec, url: str, model: str | None, runtime_source: str | None) -> CheckReport:
    async with VllmEndpoint(url, model=model) as endpoint:
        if runtime_source:
            endpoint.metadata["runtime_source"] = runtime_source
        return await spec.run(endpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Path to a JSON test specification")
    parser.add_argument("--lookup", action="store_true", help="Verify saved artifacts without contacting a server")
    parser.add_argument("--url", help="Dedicated vLLM endpoint URL")
    parser.add_argument("--model", help="Override the HTTP adapter's default model")
    parser.add_argument("--runtime-source", help="Optional runtime provenance; not a numerical acceptance limit")
    parser.add_argument("--output", help="Save a typed Marin artifact at this directory or storage URI")
    args = parser.parse_args()
    spec = read_test_spec(args.spec)
    if args.lookup:
        found = lookup_reference(spec.reference)
        result = {
            "status": found.status,
            "reason": found.reason,
            "reference": spec.reference.model_dump(),
            "histories": len(found.cases),
            "positions": len(found.scores),
        }
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if found.status == LookupStatus.FOUND else 1)
    if not args.url:
        parser.error("Supply --url")
    report = asyncio.run(run(spec, args.url, args.model, args.runtime_source))
    if args.output:
        report.save(args.output)
    print(
        json.dumps(
            {
                "status": report.status,
                "report": args.output,
                "spec": spec.id,
                "provisional": spec.provisional,
                "checks": {name: getattr(report, name).status for name in ("reference", "batch", "cache", "decode")},
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report.status == CheckStatus.PASSED else 1)


if __name__ == "__main__":
    main()
