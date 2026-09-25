# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze age-associated brown-anole CpGs from the observed GEO methylation matrix."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import tempfile
import time
import urllib.request
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

CHUNK_SITES = 4096
AGE_Q = 0.05
MIN_SAMPLES = 30
SAMPLE_TITLE = re.compile(r"S(\d+)_(\d+)_(F|M)")
TREATMENT = re.compile(r"treatment: (\d+)mo (Female|Male)")


def fetch_small(url: str, expected_sha256: str) -> bytes:
    """Fetch and identify a small source manifest."""
    with urllib.request.urlopen(url, timeout=90) as response:
        content = response.read(1_000_001)
    if len(content) > 1_000_000 or hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"Unexpected source metadata: {url}")
    return content


def sample_metadata(source: bytes) -> dict[str, tuple[int, str, str]]:
    """Join the GEO title, treatment, and accession for each observed animal."""
    lines = gzip.decompress(source).decode().splitlines()
    fields: dict[str, list[str]] = {}
    for line in lines:
        if line.startswith(("!Sample_title\t", "!Sample_geo_accession\t", "!Sample_characteristics_ch1\t")):
            values = next(csv.reader([line], delimiter="\t"))
            if values[0] == "!Sample_characteristics_ch1" and not any("treatment:" in value for value in values[1:]):
                continue
            fields[values[0]] = values[1:]
    titles = fields["!Sample_title"]
    accessions = fields["!Sample_geo_accession"]
    treatments = fields["!Sample_characteristics_ch1"]
    if not len(titles) == len(accessions) == len(treatments) == 40:
        raise ValueError("Unexpected GEO sample metadata length")
    samples = {}
    for title, accession, treatment in zip(titles, accessions, treatments, strict=True):
        title_match = SAMPLE_TITLE.fullmatch(title)
        treatment_match = TREATMENT.fullmatch(treatment)
        if title_match is None or treatment_match is None:
            raise ValueError(f"Unparseable GEO sample metadata: {title} / {treatment}")
        sample, age, sex = title_match.groups()
        if int(age) != int(treatment_match.group(1)) or sex != treatment_match.group(2)[0]:
            raise ValueError(f"GEO sample age/sex mismatch: {title} / {treatment}")
        if sample in samples:
            raise ValueError(f"Duplicate GEO sample: {sample}")
        samples[sample] = int(age), sex, accession
    return samples


def chromosome_lengths(report: bytes, sizes: bytes) -> dict[str, int]:
    """Map the 14 longest RefSeq scaffolds to source-table names and lengths."""
    size_by_accession = dict(line.split("\t") for line in sizes.decode().splitlines())
    top_accessions = sorted(size_by_accession, key=lambda key: -int(size_by_accession[key]))[:14]
    chromosomes = {}
    for line in report.decode().splitlines():
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 9 or fields[6] not in top_accessions:
            continue
        if int(fields[8]) != int(size_by_accession[fields[6]]):
            raise ValueError(f"Assembly length mismatch: {fields[0]}")
        chromosomes[fields[0]] = int(fields[8])
    if len(chromosomes) != 14 or set(chromosomes) != {f"scaffold_{i}" for i in range(1, 15)}:
        raise ValueError("Unexpected AnoSag2.1 chromosome mapping")
    return dict(sorted(chromosomes.items(), key=lambda item: int(item[0].split("_")[1])))


def download_once(url: str, expected_bytes: int, destination: Path) -> str:
    """Download one compressed GEO source to reserved-worker scratch while hashing."""
    digest = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as output:
        if int(response.headers.get("Content-Length", "0")) != expected_bytes:
            raise ValueError("GEO source Content-Length changed")
        while chunk := response.read(4 * 1024 * 1024):
            total += len(chunk)
            if total > expected_bytes:
                raise ValueError("GEO source exceeded pinned byte count")
            digest.update(chunk)
            output.write(chunk)
    if total != expected_bytes:
        raise ValueError(f"Incomplete GEO source: {total} != {expected_bytes}")
    return digest.hexdigest()


def site_pvalues(values: np.ndarray, ages: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided Spearman tests with pairwise omissions and tied-age ranks."""
    valid = np.isfinite(values)
    x = stats.rankdata(np.where(valid, values, np.nan), axis=1, nan_policy="omit")
    y = stats.rankdata(np.where(valid, ages[None, :], np.nan), axis=1, nan_policy="omit")
    x -= np.nanmean(x, axis=1)[:, None]
    y -= np.nanmean(y, axis=1)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.nansum(x * y, axis=1) / np.sqrt(np.nansum(x * x, axis=1) * np.nansum(y * y, axis=1))
        degrees = valid.sum(axis=1) - 2
        statistic = np.abs(rho) * np.sqrt(degrees / np.maximum(1 - rho * rho, np.finfo(float).tiny))
        pvalues = 2 * stats.t.sf(statistic, degrees)
    pvalues[~np.isfinite(pvalues)] = 1.0
    rho[~np.isfinite(rho)] = 0.0
    return rho, pvalues


def freeze(specification: Path, output: Path) -> dict:
    """Build a complete age-associated set without retaining the bulk source."""
    started = time.monotonic()
    spec = json.loads(specification.read_text())
    metadata = sample_metadata(fetch_small(spec["source_metadata_url"], spec["source_metadata_sha256"]))
    chromosomes = chromosome_lengths(
        fetch_small(spec["assembly_report_url"], spec["assembly_report_sha256"]),
        fetch_small(spec["assembly_sizes_url"], spec["assembly_sizes_sha256"]),
    )
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    candidate_pvalues = array("d")
    candidate_rows = output / "candidates.tmp.tsv"

    with tempfile.TemporaryDirectory(prefix="anole-cpg-source-") as scratch:
        source_path = Path(scratch) / "methylation.tab.gz"
        source_sha = download_once(spec["source_data_url"], spec["source_data_bytes"], source_path)
        counts["downloaded_bytes"] = source_path.stat().st_size
        with gzip.open(source_path, "rt", newline="") as source, candidate_rows.open("w", newline="") as candidates:
            reader = csv.reader(source, delimiter="\t")
            header = next(reader)
            if header[:4] != ["chr", "start", "end", "strand"]:
                raise ValueError("Unexpected GEO matrix columns")
            sample_ids = header[4:]
            if len(sample_ids) != 37 or len(set(sample_ids)) != 37 or not set(sample_ids) <= set(metadata):
                raise ValueError("GEO matrix samples cannot be joined uniquely to ages")
            ages = np.array([metadata[sample][0] for sample in sample_ids], dtype=float)
            pending: dict[int, tuple[list[float], list[int]]] = {}
            batch: list[tuple[str, int, list[float]]] = []
            current_chrom = ""
            last_position = 0
            completed_chromosomes: set[str] = set()

            def write_batch() -> None:
                if not batch:
                    return
                matrix = np.asarray([item[2] for item in batch], dtype=float)
                rho, pvalues = site_pvalues(matrix, ages)
                counts["tested_sites"] += len(batch)
                for (chrom, position, values), coefficient, pvalue in zip(batch, rho, pvalues, strict=True):
                    if pvalue >= AGE_Q:
                        continue
                    candidate_pvalues.append(float(pvalue))
                    fields = [
                        chrom,
                        str(position),
                        str(sum(math.isfinite(value) for value in values)),
                        format(float(coefficient), ".17g"),
                        format(float(pvalue), ".17g"),
                    ]
                    fields.extend("" if not math.isfinite(value) else format(value, ".17g") for value in values)
                    candidates.write("\t".join(fields) + "\n")
                batch.clear()

            def finish_site(position: int) -> None:
                values, observations = pending.pop(position)
                merged = [
                    value / count if count else math.nan for value, count in zip(values, observations, strict=True)
                ]
                counts["collapsed_dyads"] += 1
                if sum(math.isfinite(value) for value in merged) < MIN_SAMPLES:
                    counts["insufficient_samples"] += 1
                    return
                batch.append((current_chrom, position, merged))
                if len(batch) >= CHUNK_SITES:
                    write_batch()

            for fields in reader:
                counts["source_rows"] += 1
                if len(fields) != len(header):
                    raise ValueError("Malformed GEO matrix row")
                chrom = fields[0]
                if chrom not in chromosomes:
                    counts["other_scaffold_rows"] += 1
                    continue
                if chrom != current_chrom:
                    for position in sorted(pending):
                        finish_site(position)
                    write_batch()
                    if current_chrom:
                        completed_chromosomes.add(current_chrom)
                    if chrom in completed_chromosomes:
                        raise ValueError("GEO matrix scaffold order is not grouped")
                    current_chrom = chrom
                    last_position = 0
                position = int(fields[1])
                if position != int(fields[2]) or position < last_position or position > chromosomes[chrom]:
                    raise ValueError(f"Unexpected GEO matrix coordinate: {chrom}:{position}")
                last_position = position
                canonical = position if fields[3] == "+" else position - 1 if fields[3] == "-" else -1
                if canonical < 1:
                    raise ValueError(f"Invalid CpG strand coordinate: {chrom}:{position}")
                counts["chromosome_strand_rows"] += 1
                observations = [float(value) if value else math.nan for value in fields[4:]]
                if any(math.isfinite(value) and not 0 <= value <= 100 for value in observations):
                    raise ValueError("Methylation percentage outside 0-100")
                sums, seen = pending.setdefault(canonical, ([0.0] * len(sample_ids), [0] * len(sample_ids)))
                for index, value in enumerate(observations):
                    if math.isfinite(value):
                        sums[index] += value
                        seen[index] += 1
                counts["observed_measurements"] += sum(math.isfinite(value) for value in observations)
                for earlier in sorted(position_key for position_key in pending if position_key < position - 1):
                    finish_site(earlier)
            for position in sorted(pending):
                finish_site(position)
            write_batch()
            completed_chromosomes.add(current_chrom)
            if completed_chromosomes != set(chromosomes):
                raise ValueError("Not all 14 expected scaffolds were observed")

    pvalues = np.asarray(candidate_pvalues, dtype=float)
    order = np.argsort(pvalues, kind="stable")
    sorted_pvalues = pvalues[order]
    tested = counts["tested_sites"]
    ranks = np.arange(1, len(sorted_pvalues) + 1)
    adjusted = np.minimum.accumulate((sorted_pvalues * tested / ranks)[::-1])[::-1]
    qvalues = np.empty_like(pvalues)
    qvalues[order] = adjusted
    selected = qvalues < AGE_Q
    counts["nominal_candidates"] = len(pvalues)
    counts["age_associated_sites"] = int(selected.sum())
    if counts["age_associated_sites"] < 100 or counts["age_associated_sites"] > 100_000:
        raise ValueError("Age-associated site count outside task-quality or packet-size gate")

    output_sites = output / "age_associated_sites.tsv.gz"
    with candidate_rows.open() as source, output_sites.open("wb") as raw_output:
        with gzip.GzipFile(fileobj=raw_output, mode="wb", mtime=0, compresslevel=9) as compressed:
            header = ["chrom", "position_1", "n_samples", "spearman_rho", "p_value", "bh_q_value"]
            header.extend(f"S{sample}" for sample in sample_ids)
            compressed.write(("\t".join(header) + "\n").encode())
            for index, line in enumerate(source):
                if selected[index]:
                    fields = line.rstrip("\n").split("\t")
                    fields.insert(5, format(float(qvalues[index]), ".17g"))
                    compressed.write(("\t".join(fields) + "\n").encode())
    candidate_rows.unlink()

    (output / "chromosomes.tsv").write_text(
        "chrom\treference_bp\n" + "".join(f"{chrom}\t{length}\n" for chrom, length in chromosomes.items())
    )
    (output / "samples.tsv").write_text(
        "sample_id\tage_months\tsex\tgeo_accession\n"
        + "".join(
            f"S{sample}\t{metadata[sample][0]}\t{metadata[sample][1]}\t{metadata[sample][2]}\n" for sample in sample_ids
        )
    )
    result = {
        "source_id": spec["source_id"],
        "source_data_sha256": source_sha,
        "source_data_bytes": spec["source_data_bytes"],
        "source_metadata_sha256": spec["source_metadata_sha256"],
        "assembly_report_sha256": spec["assembly_report_sha256"],
        "assembly_sizes_sha256": spec["assembly_sizes_sha256"],
        "counts": dict(counts),
        "runtime_seconds": round(time.monotonic() - started, 3),
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "sample_ids": [f"S{sample}" for sample in sample_ids],
        "chromosomes": chromosomes,
        "bh_test_universe": tested,
        "bh_q_threshold": AGE_Q,
        "derived_files": {
            name: {
                "bytes": (output / name).stat().st_size,
                "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
            }
            for name in ("age_associated_sites.tsv.gz", "chromosomes.tsv", "samples.tsv")
        },
    }
    (output / "source-preparation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.specification, args.output)


if __name__ == "__main__":
    main()
