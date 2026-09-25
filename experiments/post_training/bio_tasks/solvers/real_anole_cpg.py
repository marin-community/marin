# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SciPy oracle for the observed brown-anole CpG-density workflow."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import chisquare


def write_table(path: Path, rows: dict[str, dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", *next(iter(rows.values()))], delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows({"id": identifier, **row} for identifier, row in sorted(rows.items()))


def solve_anole_cpg(inputs: Path, output: Path) -> list[dict]:
    """Classify complete site measurements and test scaffold-length-normalized counts."""
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((inputs / "protocol.json").read_text())
    if (
        protocol["assembly"] != "AnoSag2.1 (GCF_025583915.1)"
        or protocol["low_strict_percent"] != 10.0
        or protocol["high_strict_percent"] != 90.0
        or protocol["reference_scaffolds"] != 14
    ):
        raise ValueError("Unexpected observed CpG task protocol")

    with (inputs / "chromosomes.tsv").open(newline="") as handle:
        lengths = {row["chrom"]: int(row["reference_bp"]) for row in csv.DictReader(handle, delimiter="\t")}
    if len(lengths) != 14 or min(lengths.values()) <= 0:
        raise ValueError("Incomplete AnoSag2.1 scaffold lengths")
    with (inputs / "samples.tsv").open(newline="") as handle:
        sample_ids = [row["sample_id"] for row in csv.DictReader(handle, delimiter="\t")]
    if len(sample_ids) != 37 or len(set(sample_ids)) != 37:
        raise ValueError("Incomplete observed sample identities")

    site_rows = {}
    age_sites = defaultdict(set)
    extreme_sites = defaultdict(set)
    removed_measurements = 0
    observed_measurements = 0
    with (inputs / "age_associated_sites.tsv").open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not set(sample_ids) <= set(reader.fieldnames):
            raise ValueError("Methylation and sample metadata disagree")
        for row in reader:
            chrom = row["chrom"]
            position = int(row["position_1"])
            if chrom not in lengths or not 1 <= position <= lengths[chrom] or float(row["bh_q_value"]) >= 0.05:
                raise ValueError("Invalid age-associated CpG")
            identifier = f"{chrom}:{position}"
            if identifier in site_rows:
                raise ValueError("Duplicate canonical CpG identity")
            measurements = np.array([float(row[sample]) for sample in sample_ids if row[sample] != ""], dtype=float)
            if measurements.size != int(row["n_samples"]) or not 30 <= measurements.size <= 37:
                raise ValueError("Invalid observed sample count")
            if not np.isfinite(measurements).all() or ((measurements < 0) | (measurements > 100)).any():
                raise ValueError("Invalid methylation percentage")
            low = int(np.count_nonzero(measurements < 10))
            high = int(np.count_nonzero(measurements > 90))
            middle = int(measurements.size - low - high)
            if low or high:
                extreme_sites[chrom].add(position)
            age_sites[chrom].add(position)
            site_rows[identifier] = {
                "chrom": chrom,
                "position_1": position,
                "observed_rows": int(measurements.size),
                "low_rows": low,
                "high_rows": high,
                "middle_rows": middle,
                "extreme_site": int(bool(low or high)),
            }
            removed_measurements += middle
            observed_measurements += int(measurements.size)
    if not site_rows:
        raise ValueError("Empty age-associated CpG input")

    ordered = sorted(lengths, key=lambda name: int(name.split("_")[1]))
    age_observed = np.array([len(age_sites[chrom]) for chrom in ordered], dtype=float)
    observed = np.array([len(extreme_sites[chrom]) for chrom in ordered], dtype=float)
    total_sites = int(observed.sum())
    if not total_sites:
        raise ValueError("No extreme CpG sites")
    bases = np.array([lengths[chrom] for chrom in ordered], dtype=float)
    expected = total_sites * bases / bases.sum()
    statistic, pvalue = chisquare(f_obs=observed, f_exp=expected)
    contribution = (observed - expected) ** 2 / expected
    age_density = age_observed / bases
    filtered_density = observed / bases
    highest_age_index = max(range(len(ordered)), key=lambda index: (age_density[index], -index))
    highest_filtered_index = max(range(len(ordered)), key=lambda index: (filtered_density[index], -index))
    scaffold_rows = {
        chrom: {
            "reference_bp": int(lengths[chrom]),
            "age_associated_sites": int(age_observed[index]),
            "age_associated_density_per_bp": float(age_density[index]),
            "unique_extreme_sites": int(observed[index]),
            "filtered_density_per_bp": float(filtered_density[index]),
            "expected_sites_length_null": float(expected[index]),
            "chi2_contribution": float(contribution[index]),
        }
        for index, chrom in enumerate(ordered)
    }
    write_table(output / "sites.tsv", site_rows)
    write_table(output / "scaffolds.tsv", scaffold_rows)
    return [
        {
            "id": "anole",
            "age_associated_sites": len(site_rows),
            "observed_measurement_rows": observed_measurements,
            "removed_middle_rows": removed_measurements,
            "unique_extreme_sites": total_sites,
            "scaffolds_with_filtered_sites": int(np.count_nonzero(observed)),
            "mean_filtered_density_per_bp": float(filtered_density[observed > 0].mean()),
            "highest_age_associated_density_scaffold": ordered[highest_age_index],
            "highest_age_associated_density_per_bp": float(age_density[highest_age_index]),
            "highest_filtered_density_scaffold": ordered[highest_filtered_index],
            "chi2_length_null": float(statistic),
            "chi2_df": len(ordered) - 1,
            "chi2_pvalue": float(pvalue),
        }
    ]
