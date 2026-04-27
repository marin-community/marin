# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import threading
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from marin.datakit.download.uwf_zeek import (
    DEFAULT_OUTPUT_FILENAME,
    DownloadUwfZeekSampleConfig,
    UwfZeekSampleSource,
    download_uwf_zeek_sample,
)


@pytest.fixture()
def local_uwf_server(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    server_root = tmp_path / "server"
    server_root.mkdir()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(server_root), **kwargs)

        def log_message(self, format, *args):  # noqa: A002  # stdlib signature
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = httpd.server_address
        yield f"http://{host}:{port}", server_root
    finally:
        httpd.shutdown()
        thread.join()


def _publish(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_download_uwf_zeek_sample_lists_categories_and_caps_rows(tmp_path: Path, local_uwf_server) -> None:
    base_url, server_root = local_uwf_server
    _publish(
        server_root / "csv" / "Benign" / "index.html",
        '<html><body><a href="part-00001.csv">part-00001.csv</a></body></html>',
    )
    _publish(
        server_root / "csv" / "Discovery" / "index.html",
        '<html><body><a href="part-00002.csv">part-00002.csv</a></body></html>',
    )
    _publish(
        server_root / "csv" / "Reconnaissance" / "index.html",
        '<html><body><a href="part-00003.csv">part-00003.csv</a></body></html>',
    )

    header = (
        "community_id,conn_state,duration,history,src_ip_zeek,src_port_zeek,dest_ip_zeek,dest_port_zeek,"
        "local_orig,local_resp,missed_bytes,orig_bytes,orig_ip_bytes,orig_pkts,proto,resp_bytes,"
        "resp_ip_bytes,resp_pkts,service,ts,uid,datetime,vlan,label_tactic,label_technique,label_binary,label_cve\n"
    )
    benign_rows = [
        "1:a,S0,0.1,D,10.0.0.1,12345,8.8.8.8,53,F,F,0,76,132,2,udp,0,0,0,dns,1.0,UidBenign,2025-05-24T00:11:40.768Z,102,none,none,False,none\n",
        "1:b,S0,0.2,D,10.0.0.2,12346,8.8.4.4,53,F,F,0,88,144,2,udp,0,0,0,dns,2.0,UidBenign2,2025-05-24T00:12:40.768Z,102,none,none,False,none\n",
    ]
    discovery_row = (
        "1:c,REJ,0.3,Sr,10.0.1.1,44444,10.0.1.2,80,F,F,0,0,44,1,tcp,0,40,1,,3.0,UidDiscovery,"
        "2025-06-03T16:48:15.421Z,103,Discovery,T1046,True,none\n"
    )
    recon_row = (
        "1:d,REJ,0.4,Sr,10.0.2.1,55555,10.0.2.2,22,F,F,0,0,44,1,tcp,0,40,1,,4.0,UidRecon,"
        "2025-05-30T16:28:57.666Z,103,Reconnaissance,T1595,True,none\n"
    )

    _publish(server_root / "csv" / "Benign" / "part-00001.csv", header + "".join(benign_rows))
    _publish(server_root / "csv" / "Discovery" / "part-00002.csv", header + discovery_row)
    _publish(server_root / "csv" / "Reconnaissance" / "part-00003.csv", header + recon_row)

    output_dir = tmp_path / "output"
    manifest = download_uwf_zeek_sample(
        DownloadUwfZeekSampleConfig(
            source=UwfZeekSampleSource(
                slice_key="binary_network_security/uwf_zeek",
                base_url=f"{base_url}/csv/",
                max_rows_per_category=1,
            ),
            output_path=str(output_dir),
        )
    )

    records = _read_jsonl_gz(output_dir / DEFAULT_OUTPUT_FILENAME)
    assert len(records) == 3
    assert [record["category"] for record in records] == ["Benign", "Discovery", "Reconnaissance"]
    assert records[0]["id"] == "binary_network_security/uwf_zeek#benign:0"
    assert records[0]["source"] == "binary_network_security/uwf_zeek"
    assert records[0]["text"].splitlines()[0].startswith("community_id,conn_state,duration")
    assert "UidBenign" in records[0]["text"]
    assert "UidBenign2" not in records[0]["text"]
    assert "UidDiscovery" in records[1]["text"]
    assert "UidRecon" in records[2]["text"]

    assert manifest["total_rows"] == 3
    assert [entry["rows_written"] for entry in manifest["files"]] == [1, 1, 1]
    assert manifest["files"][0]["csv_url"].endswith("/csv/Benign/part-00001.csv")
