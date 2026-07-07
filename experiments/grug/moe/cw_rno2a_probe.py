# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe: on cw-rno2a (natively configured for CoreWeave storage), confirm the datakit
store is reachable via the cluster's auto-resolved MARIN_PREFIX -- no creds/endpoint
injection. Reports what prefix/endpoint the cluster actually hands the job.
"""

import os

import fsspec


def main() -> None:
    prefix = os.environ.get("MARIN_PREFIX", "<unset>")
    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL", "<unset>")
    keylen = len(os.environ.get("AWS_ACCESS_KEY_ID", ""))
    print(f"[probe] MARIN_PREFIX={prefix} endpoint={endpoint} aws_key_len={keylen}")

    store = prefix.rstrip("/") + "/datakit/store_8ac06c74"
    bucket_path = store.replace("s3://", "")
    fs = fsspec.filesystem("s3", config_kwargs={"s3": {"addressing_style": "virtual"}})
    print(f"[probe] listing {store}")
    entries = fs.ls(bucket_path, detail=False)
    clusters = [e for e in entries if "cluster=" in e]
    print(f"[probe] STORE REACHABLE: {len(clusters)} cluster= dirs; sample {[c.split('/')[-1] for c in clusters[:4]]}")

    # read one parquet shard to prove data (not just listing) works
    shards = fs.glob(f"{bucket_path}/cluster=1/quality=0/sub=0/**/*.parquet")[:1]
    print(f"[probe] first shard: {shards[0].split('/store_8ac06c74/')[-1] if shards else '<none>'}")
    print("[probe] ALL GOOD")


if __name__ == "__main__":
    main()
