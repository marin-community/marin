# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from pathlib import Path

import marin.datakit.download.uwf_zeek as uwf_zeek
import pytest
from marin.datakit.download.uwf_zeek import (
    DEFAULT_OUTPUT_FILENAME,
    UwfZeekSampleSource,
    download_uwf_zeek_sample,
)


class _FakeResponse:
    def __init__(self, *, text: str = "", lines: list[str] | None = None):
        self.text = text
        self._lines = [] if lines is None else lines

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, *, decode_unicode: bool = False):
        for line in self._lines:
            yield line if decode_unicode else line.encode("utf-8")

    def close(self) -> None:
        return None


class _FakeSession:
    def __init__(self, responses: dict[str, _FakeResponse]):
        self._responses = responses

    def get(self, url: str, *, timeout: int, stream: bool = False):
        return self._responses[url]

    def close(self) -> None:
        return None


def _read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_download_uwf_zeek_sample_lists_categories_and_caps_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_url = "https://example.test/csv/"
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

    responses = {
        f"{base_url}Benign/": _FakeResponse(
            text='<html><body><a href="part-00001.csv">part-00001.csv</a></body></html>'
        ),
        f"{base_url}Discovery/": _FakeResponse(
            text='<html><body><a href="part-00002.csv">part-00002.csv</a></body></html>'
        ),
        f"{base_url}Reconnaissance/": _FakeResponse(
            text='<html><body><a href="part-00003.csv">part-00003.csv</a></body></html>'
        ),
        f"{base_url}Benign/part-00001.csv": _FakeResponse(lines=(header + "".join(benign_rows)).splitlines()),
        f"{base_url}Discovery/part-00002.csv": _FakeResponse(lines=(header + discovery_row).splitlines()),
        f"{base_url}Reconnaissance/part-00003.csv": _FakeResponse(lines=(header + recon_row).splitlines()),
    }
    monkeypatch.setattr(uwf_zeek, "_build_session", lambda: _FakeSession(responses))

    output_dir = tmp_path / "output"
    manifest = download_uwf_zeek_sample(
        source=UwfZeekSampleSource(
            slice_key="binary_network_security/uwf_zeek",
            base_url=base_url,
            max_rows_per_category=1,
        ),
        output_path=str(output_dir),
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
    assert manifest["files"][0]["csv_url"] == f"{base_url}Benign/part-00001.csv"
