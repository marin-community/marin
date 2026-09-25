# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe HTSlib indexed-fetch behavior for placed and unplaced unmapped reads."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pysam

from experiments.post_training.bio_tasks.bam_artifacts import summarize_bam


def main() -> None:
    with TemporaryDirectory(prefix="bam-index-probe-") as directory:
        root = Path(directory)
        unsorted = root / "reads.unsorted.bam"
        bam_path = root / "reads.bam"
        header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr", "LN": 100}]}
        with pysam.AlignmentFile(unsorted, "wb", header=header) as bam:
            for name, flag, reference_id, start, cigar in (
                ("mapped", 0, 0, 10, "10M"),
                ("placed_unmapped", 0x4, 0, 20, None),
                ("unplaced_unmapped", 0x4, -1, -1, None),
            ):
                record = pysam.AlignedSegment(bam.header)
                record.query_name = name
                record.flag = flag
                record.reference_id = reference_id
                record.reference_start = start
                record.mapping_quality = 60 if not flag else 0
                record.cigarstring = cigar
                record.query_sequence = "ACGTACGTAC"
                record.query_qualities = pysam.qualitystring_to_array("IIIIIIIIII")
                bam.write(record)
        pysam.sort("-o", str(bam_path), str(unsorted))
        pysam.index(str(bam_path))
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            result = {
                "pysam_version": pysam.__version__,
                "all": [r.query_name for r in bam.fetch(until_eof=True)],
                "whole_reference": [r.query_name for r in bam.fetch("chr", 0, 100)],
                "placed_position": [r.query_name for r in bam.fetch("chr", 20, 21)],
                "mapped_position": [r.query_name for r in bam.fetch("chr", 10, 11)],
            }
        references, _, records = summarize_bam(bam_path, 1024 * 1024, 1024 * 1024, 3)
        if references != {"chr": 100} or records != 3:
            raise ValueError("BAM checker disagrees with placed/unplaced fixture")
        result["checker_records"] = records
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
