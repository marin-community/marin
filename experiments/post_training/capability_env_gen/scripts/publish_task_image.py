#!/usr/bin/env python3
"""Trusted-worker entrypoint for reviewed S3 rootfs -> authenticated OCI image."""

import argparse
import json
from pathlib import Path

from capability_pipeline.image_publication import publish_captured_image
from capability_pipeline.oci_registry import RegistryClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--capture-receipt", type=Path, required=True)
    parser.add_argument("--layer", type=Path, required=True)
    parser.add_argument("--approved-plan-sha256", required=True)
    parser.add_argument("--max-uncompressed-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("publication receipt already exists")
    client = None
    if args.execute:
        if args.credentials_file is None:
            raise ValueError("trusted publication credentials are required")
        credentials = json.loads(args.credentials_file.read_text())
        role = json.loads(args.capture_receipt.read_text())["role"]
        image = next(
            image
            for image in json.loads(args.plan.read_text())["images"]
            if image["role"] == role
        )
        client = RegistryClient(
            credentials["registry"],
            image["repository"],
            credentials["user"],
            credentials["password"],
        )
    receipt = publish_captured_image(
        plan_path=args.plan,
        workspace=args.workspace,
        capture_path=args.capture_receipt,
        layer_path=args.layer,
        approved_plan_sha256=args.approved_plan_sha256,
        max_uncompressed_bytes=args.max_uncompressed_bytes,
        registry_client=client,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(receipt, output, indent=2, sort_keys=True)
        output.write("\n")
    print(
        json.dumps(
            {
                "state": receipt["state"],
                "role": receipt["role"],
                "receipt": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 -- never log credential-bearing transport details
        print(json.dumps({"state": "failed", "error_type": type(error).__name__}))
        raise SystemExit(1) from None
