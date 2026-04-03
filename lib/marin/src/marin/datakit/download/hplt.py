# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download and filter HPLT v3.0 dataset, keeping only non-Common Crawl sources (WIDE, survey)."""

import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import closing

import requests
import urllib3
import zstandard
from zephyr import counters
from marin.execution.step_spec import StepSpec
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from zephyr import Dataset, ZephyrContext

logger = logging.getLogger(__name__)

HPLT_BASE_URL = "https://data.hplt-project.org/three/sorted/eng_Latn"
HPLT_MAP_URL = f"{HPLT_BASE_URL}/eng_Latn.map"

# Shard counts per WDS quality tier
HPLT_SHARD_COUNTS = {5: 74, 6: 119, 7: 275, 8: 479, 9: 344, 10: 3}

# Turku web-register codes (https://turkunlp.org/register-annotation-docs/)
# Top-level: MT=Machine Translated, LY=Lyrical, SP=Spoken, ID=Interactive Discussion,
# NA=Narrative, HI=How-To/Instructions, IN=Informational, OP=Opinion, IP=Informational Persuasion
TOP_LEVEL_REGISTERS = frozenset({"MT", "LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP"})

# All known sub-register codes from the Turku schema
ALL_SUB_REGISTERS = frozenset(
    {
        "ob",
        "nb",
        "ne",
        "ra",
        "re",
        "rv",
        "sr",
        "it",
        "en",
        "av",
        "rs",  # good prose
        "lt",
        "fi",
        "dtp",
        "oi",
        "on",
        "oh",
        "oo",
        "os",
        "ed",
        "oe",
        "ds",  # other sub-registers
    }
)

# Sub-registers indicating good prose content:
# ob=Opinion Blog, nb=Narrative Blog, ne=News Report, ra=Research Article, re=Recipe,
# rv=Review, sr=Sports Report, it=Interview, en=Encyclopedia Article, av=Advice,
# rs=Religious Blog/Sermon
GOOD_SUB_REGISTERS = frozenset({"ob", "nb", "ne", "ra", "re", "rv", "sr", "it", "en", "av", "rs"})

# Top-level registers that indicate good content on their own
GOOD_TOP_REGISTERS = frozenset({"HI", "LY", "SP"})

# Crawl ID prefixes for non-CC sources we want to keep
NON_CC_PREFIXES = ("wide", "survey")


def _iter_jsonl_from_zstd_stream(raw_stream) -> Iterator[dict]:
    """Yield parsed JSON objects from a zstd-compressed JSONL stream."""
    dctx = zstandard.ZstdDecompressor()
    with dctx.stream_reader(raw_stream) as reader:
        buf = bytearray()
        while True:
            chunk = reader.read(1048576)
            if not chunk:
                break
            buf.extend(chunk)
            while True:
                newline_pos = buf.find(b"\n")
                if newline_pos < 0:
                    break
                line_bytes = bytes(buf[:newline_pos])
                del buf[: newline_pos + 1]
                if not line_bytes.strip():
                    continue
                yield json.loads(line_bytes)
        # Flush trailing bytes (last record may lack a trailing newline)
        if buf.strip():
            yield json.loads(bytes(buf))


def passes_quality_filter(doc: dict, wds_tier: int) -> bool:
    """Apply register-based quality filter to an HPLT document.

    Filter rules derived from empirical analysis with Haiku as ground truth classifier,
    achieving ~99% agreement on coherent/incoherent classification.
    """
    wr = doc.get("web-register", {})
    mt = wr.get("MT", 0)  # Machine Translated score
    ds = wr.get("ds", 0)  # Description with Intent to Sell score

    # Hard rejections
    if doc.get("pii"):
        return False
    if mt >= 0.2:
        return False
    if ds > 0.1:
        return False

    top_regs = {k: v for k, v in wr.items() if k in TOP_LEVEL_REGISTERS}
    sub_regs = {k: v for k, v in wr.items() if k in ALL_SUB_REGISTERS}
    dominant_top = max(top_regs, key=top_regs.get) if top_regs else None
    dominant_sub = max(sub_regs, key=sub_regs.get) if sub_regs else None

    if dominant_sub == "lt":  # Legal Terms and Conditions
        return False
    if dominant_top == "IP" and not any(
        wr.get(s, 0) > 0.15 for s in GOOD_SUB_REGISTERS
    ):  # pure Informational Persuasion
        return False

    # Keep rules
    if wds_tier >= 8:
        return True
    if dominant_sub in GOOD_SUB_REGISTERS:
        return True
    if dominant_top in GOOD_TOP_REGISTERS and mt < 0.15:
        return True

    return False


def _is_non_cc_source(crawl_id: str) -> bool:
    return any(crawl_id.startswith(prefix) for prefix in NON_CC_PREFIXES)


def _get_all_shard_urls() -> list[tuple[int, int, str]]:
    """Return (wds_tier, shard_num, url) for every HPLT English shard."""
    shards = []
    for tier, count in HPLT_SHARD_COUNTS.items():
        for shard in range(1, count + 1):
            url = f"{HPLT_BASE_URL}/{tier}_{shard}.jsonl.zst"
            shards.append((tier, shard, url))
    return shards


def _make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1.0, status_forcelist=[500, 502, 503, 504], allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


MAX_SHARD_RETRIES = 3


def _download_and_filter_shard(tier: int, shard: int, url: str) -> Iterator[dict]:
    """Download an HPLT shard, filter, and yield dolma-format records with retries."""
    session = _make_session()
    for attempt in range(MAX_SHARD_RETRIES):
        try:
            logger.info(f"Processing HPLT shard {tier}_{shard} (attempt {attempt + 1}/{MAX_SHARD_RETRIES})")
            response = session.get(url, headers={"user-agent": "marin-hplt-ingress/1.0"}, stream=True)
            response.raise_for_status()

            num_input = 0
            num_non_cc = 0
            num_kept = 0
            filtered_records = []
            with closing(response):
                for record in _iter_jsonl_from_zstd_stream(response.raw):
                    num_input += 1
                    crawl_id = record.get("crawl_id", "")

                    if not _is_non_cc_source(crawl_id):
                        continue
                    num_non_cc += 1

                    if not passes_quality_filter(record, tier):
                        continue

                    num_kept += 1
                    filtered_records.append(
                        {
                            "id": record.get("id", ""),
                            "text": record.get("text", ""),
                            "source": f"hplt_v3_{crawl_id}",
                            "format": "text",
                            "metadata": {
                                "hplt_wds_tier": tier,
                                "hplt_crawl_id": crawl_id,
                                "hplt_url": record.get("u", ""),
                                "hplt_timestamp": record.get("ts", ""),
                                "hplt_web_register": record.get("web-register", {}),
                                "hplt_doc_scores": record.get("doc_scores", []),
                            },
                        }
                    )

            counters.increment("hplt/input", num_input)
            counters.increment("hplt/non_cc", num_non_cc)
            counters.increment("hplt/kept", num_kept)
            yield from filtered_records
            return
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            urllib3.exceptions.ProtocolError,
        ) as e:
            if attempt + 1 == MAX_SHARD_RETRIES:
                raise
            wait = 2 ** (attempt + 1)
            logger.warning(f"Shard {tier}_{shard} failed (attempt {attempt + 1}): {e}. Retrying in {wait}s.")
            time.sleep(wait)
    raise RuntimeError(f"unreachable: retry loop for shard {tier}_{shard} exited without return or raise")


def download_hplt_v3(output_path: str) -> None:
    """Download and filter HPLT v3.0 English dataset, keeping only non-CC sources."""
    all_shards = _get_all_shard_urls()
    logger.info(f"Processing {len(all_shards)} HPLT shards")

    pipeline = (
        Dataset.from_list(all_shards)
        .flat_map(lambda info: _download_and_filter_shard(info[0], info[1], info[2]))
        .write_parquet(os.path.join(output_path, "data-{shard:05d}-of-{total:05d}.parquet"), skip_existing=True)
    )

    ctx = ZephyrContext(name="download-hplt-v3", max_workers=32)
    ctx.execute(pipeline)

    logger.info(f"Downloaded HPLT v3 files to {output_path}")


def download_hplt_v3_step() -> StepSpec:
    """Create a StepSpec that downloads and filters the HPLT v3 English dataset."""
    return StepSpec(
        name="raw/hplt_v3",
        fn=lambda output_path: download_hplt_v3(output_path=output_path),
        override_output_path="raw/hplt_v3",
    )
