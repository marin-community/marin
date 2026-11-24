# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe the pool endpoint's auth contract from inside the cluster.

Runs as an Iris job; writes the result matrix to in-region S3 (job stdout is not
reliably retrievable for short jobs). Matrix: proxied vs direct URL x no auth vs
arbitrary Bearer vs the endpoint's capability JWT as Bearer.
"""

import json
import urllib.error
import urllib.request

from iris.client.client import iris_ctx
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

_RESULT_PATH = "s3://marin-us-east-08a/marin/users/benfeuer/glm-synthesis-serving/auth-probe.json"


def _status(url: str, headers: dict[str, str] | None = None) -> int | str:
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except Exception as error:
        return str(error)[:120]


def main() -> None:
    client = iris_ctx().client
    resolved = client.resolve_endpoint("/serve/glm-5-3-flash").rstrip("/")
    mint = client.mint_endpoint_token("/serve/glm-5-3-flash")
    jwt = mint.token
    proxy = "http://iris-controller-svc:10000/proxy/serve.glm-5-3-flash/v1/models"
    results = {
        "resolved": resolved,
        "proxy_no_auth": _status(proxy),
        "proxy_arbitrary_bearer": _status(proxy, {"Authorization": "Bearer not-a-real-token"}),
        "proxy_jwt_bearer": _status(proxy, {"Authorization": f"Bearer {jwt}"}),
        "direct_arbitrary_bearer": _status(f"{resolved}/v1/models", {"Authorization": "Bearer not-a-real-token"}),
        "direct_no_auth": _status(f"{resolved}/v1/models"),
    }
    configure_coreweave_s3()

    fs, _ = filesystem_for(_RESULT_PATH)
    with fs.open(_RESULT_PATH, "w") as stream:
        stream.write(json.dumps(results))
    print(json.dumps(results))


if __name__ == "__main__":
    main()
