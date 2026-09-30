"""Exercise the maintained publisher with benign synthetic bytes on CoreWeave."""

import gzip
import io
import json
import os
import random
import tarfile
import tempfile
from pathlib import Path

from capability_pipeline.oci_artifact import digest
from capability_pipeline.oci_registry import RegistryClient


def main():
    credentials = json.loads(Path(os.environ["REGISTRY_CREDENTIALS_FILE"]).read_text())
    client = RegistryClient(
        credentials["registry"],
        "capability-infra/streaming-oci-smoke",
        credentials["user"],
        credentials["password"],
    )
    # Deliberately cross the transport chunk boundary with incompressible bytes.
    payload = random.Random(701).randbytes(3 << 20)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        member = tarfile.TarInfo("synthetic-probe.bin")
        member.size, member.mode, member.mtime = len(payload), 0o644, 0
        archive.addfile(member, io.BytesIO(payload))
    layer = gzip.compress(buffer.getvalue(), mtime=0)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "layer.tar.gz"
        path.write_bytes(layer)
        receipt = client.publish_layer(
            path,
            expected_digest=digest(layer),
            expected_bytes=len(layer),
            max_uncompressed_bytes=4 << 20,
            image_config={
                "Env": [],
                "WorkingDir": "",
                "User": "",
                "Entrypoint": None,
                "Cmd": None,
            },
            architecture="amd64",
            operating_system="linux",
        )
    receipt["fixture_only_not_executable"] = True
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 -- omit credential-bearing transport details
        print(json.dumps({"state": "failed", "error_type": type(error).__name__}))
        raise SystemExit(1) from None
