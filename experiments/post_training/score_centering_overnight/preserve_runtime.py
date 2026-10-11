# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Promote completed-run sampler groups and logs within their original S3 bucket."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import boto3
import yaml
from botocore.config import Config
from botocore.exceptions import ClientError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["CW_S3_ENDPOINT"],
        aws_access_key_id=os.environ["CW_KEY_ID"],
        aws_secret_access_key=os.environ["CW_KEY_SECRET"],
        config=Config(s3={"addressing_style": "virtual"}),
    )
    destination = urlparse(config["artifacts"]["attempts_root"].removesuffix("/attempts") + "/retained-runtime")
    destination_prefix = destination.path.lstrip("/").rstrip("/")
    inventory = []
    sources = [
        ("groups", config["skyrl"]["trainer"]["rollout_buffer"]["object_store_root"]),
        ("ray-logs", config["ray"]["log_dir"]),
        ("debug", config["ray"]["rendezvous_dir"].rstrip("/") + "/debug_artifacts"),
    ]
    checkpoint_uri = config["artifacts"]["checkpoint_root"]
    if urlparse(checkpoint_uri).path.startswith("/tmp/"):
        sources.append(("checkpoints", checkpoint_uri))
    for label, uri in sources:
        source = urlparse(uri)
        assert source.netloc == destination.netloc, "Runtime retention must stay in the source bucket"
        prefix = source.path.lstrip("/").rstrip("/") + "/"
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=source.netloc, Prefix=prefix):
            for item in page.get("Contents", []):
                assert item["Size"] <= 5 * 1024**3
                inventory.append(
                    {
                        "source_key": item["Key"],
                        "destination_key": destination_prefix + "/" + label + "/" + item["Key"].removeprefix(prefix),
                        "bytes": item["Size"],
                        "source_etag": item["ETag"],
                        "source_last_modified_ms": int(item["LastModified"].timestamp() * 1000),
                    }
                )

    def promote(item: dict) -> dict:
        metadata = {
            "source-etag": item["source_etag"].strip('"'),
            "source-key-sha256": hashlib.sha256(item["source_key"].encode()).hexdigest(),
        }
        try:
            existing = client.head_object(Bucket=destination.netloc, Key=item["destination_key"])
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
                raise
            existing = None
        if existing is None or existing["Metadata"] != metadata:
            client.copy_object(
                Bucket=destination.netloc,
                Key=item["destination_key"],
                CopySource={"Bucket": destination.netloc, "Key": item["source_key"]},
                CopySourceIfMatch=item["source_etag"],
                MetadataDirective="REPLACE",
                Metadata=metadata,
            )
            existing = client.head_object(Bucket=destination.netloc, Key=item["destination_key"])
        assert existing["ContentLength"] == item["bytes"] and existing["Metadata"] == metadata
        return {**item, "destination_etag": existing["ETag"]}

    with ThreadPoolExecutor(max_workers=4) as executor:
        copied = list(executor.map(promote, inventory))
    report = {
        "run_id": config["run"]["id"],
        "source": config["runtime"]["launcher_commit"],
        "bucket": destination.netloc,
        "objects": copied,
        "bytes": sum(x["bytes"] for x in copied),
        "basis": (
            "Same-bucket conditional server copy with captured source ETag and verified destination length/provenance; "
            "ETags are not full weight SHA256s."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    client.put_object(
        Bucket=destination.netloc,
        Key=destination_prefix + "/promotion-manifest-01a126ce.json",
        Body=args.output.read_bytes(),
    )
    print(json.dumps({"run_id": report["run_id"], "objects": len(copied), "bytes": report["bytes"]}))


if __name__ == "__main__":
    main()
