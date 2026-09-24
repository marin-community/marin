# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select unchanged, evenly spaced paired records from checksum-verified ENA archives."""

import argparse
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.post_training.bio_tasks.real_data import source_text

SOURCE = Path(__file__).resolve().parents[1]
RUN = "ENA:ERR266411"
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024


def sample_pairs(output: Path, pairs: int) -> None:
    """Check the full run and retain matched record bytes at fixed ordinal positions."""
    metadata = json.loads((SOURCE / "data_sources.json").read_text())["sources"][RUN]["run_metadata"]
    total_reads = int(metadata["read_count"])
    if total_reads % 2 or not 0 < pairs <= total_reads // 2:
        raise ValueError("Requested pair count is inconsistent with deposited read count")
    total_pairs = total_reads // 2
    selected = {index * total_pairs // pairs for index in range(pairs)}
    output.mkdir(parents=True, exist_ok=False)
    fetches = []
    with TemporaryDirectory(prefix="ena-pairs-", dir=output.parent) as temporary:
        archives = []
        for mate, (host_path, expected_md5, expected_bytes) in enumerate(
            zip(
                metadata["fastq_ftp"].split(";"),
                metadata["fastq_md5"].split(";"),
                metadata["fastq_bytes"].split(";"),
                strict=True,
            ),
            1,
        ):
            if not host_path.startswith("ftp.sra.ebi.ac.uk/") or int(expected_bytes) > MAX_ARCHIVE_BYTES:
                raise ValueError("Unexpected ENA host or archive size")
            url = "https://" + host_path
            archive = Path(temporary) / f"mate{mate}.fastq.gz"
            md5, sha256, size = hashlib.md5(), hashlib.sha256(), 0
            with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > int(expected_bytes):
                        raise ValueError("Archive exceeds deposited byte count")
                    md5.update(chunk)
                    sha256.update(chunk)
                    handle.write(chunk)
            if size != int(expected_bytes) or md5.hexdigest() != expected_md5:
                raise ValueError("Complete ENA archive checksum or size differs from deposited metadata")
            fetches.append({"url": url, "bytes": size, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()})
            archives.append(archive)
        if len(archives) != 2:
            raise ValueError("Expected exactly two mate archives")
        seen = set()
        count = 0
        with (
            gzip.open(archives[0], "rb") as left,
            gzip.open(archives[1], "rb") as right,
            (output / "reads_R1.fastq").open("wb") as first,
            (output / "reads_R2.fastq").open("wb") as second,
        ):
            for index in range(total_pairs):
                records = [[stream.readline() for _ in range(4)] for stream in (left, right)]
                for header, sequence, separator, qualities in records:
                    if (
                        not header.startswith(b"@")
                        or not separator.startswith(b"+")
                        or not qualities.endswith(b"\n")
                        or len(sequence.rstrip(b"\r\n")) != len(qualities.rstrip(b"\r\n"))
                    ):
                        raise ValueError(f"Malformed complete FASTQ record at zero-based index {index}")
                identifiers = [record[0].split()[0][1:] for record in records]
                if identifiers[0] != identifiers[1]:
                    raise ValueError(f"Mate identifiers disagree at zero-based index {index}")
                if index in selected:
                    if identifiers[0] in seen:
                        raise ValueError("Selected pair identifiers repeat")
                    seen.add(identifiers[0])
                    count += 1
                    for handle, record in zip((first, second), records, strict=True):
                        handle.write(b"".join(record))
            if left.read(1) or right.read(1) or count != pairs:
                raise ValueError("Archive length or selected pair count differs from the declared run")
    (output / "reference.fa").write_text(source_text("RefSeq:NC_001422.1", "nc_001422-1.fa.gz"))
    (output / "query.json").write_text(json.dumps({"kmers": [21, 33, 55]}, indent=2) + "\n")
    assets = {}
    for mate in (1, 2):
        path = output / f"reads_R{mate}.fastq"
        raw = path.read_bytes()
        compressed = gzip.compress(raw, mtime=0)
        path.with_suffix(".fastq.gz").write_bytes(compressed)
        assets[path.name] = {
            "content_sha256": hashlib.sha256(raw).hexdigest(),
            "content_bytes": len(raw),
            "vendored_sha256": hashlib.sha256(compressed).hexdigest(),
            "vendored_bytes": len(compressed),
        }
    (output / "sampling.json").write_text(
        json.dumps(
            {
                "source_id": RUN,
                "total_pairs": total_pairs,
                "selected_pairs": pairs,
                "zero_based_record_indices": sorted(selected),
                "selection_rule": "floor(i * total_pairs / selected_pairs), i=0..selected_pairs-1",
                "interpretation": (
                    "Systematic sample in archive order; unchanged paired records, not biological replication."
                ),
                "archives": fetches,
                "assets": assets,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, required=True)
    args = parser.parse_args()
    sample_pairs(args.output, args.pairs)


if __name__ == "__main__":
    main()
