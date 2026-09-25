# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Chromosome-normalized age-associated CpG audit on observed brown anoles."""

import csv
import io
import json
from collections import Counter

from scipy.stats import chi2

from experiments.post_training.bio_tasks.contract import Column, Contract, TableContract
from experiments.post_training.bio_tasks.real_data import source_text
from experiments.post_training.bio_tasks.recipe_types import DataOrigin, Instance, OracleRuntime, Recipe, WorkflowScope

SOURCE_ID = "GEO:GSE285624-anole-age-methylation-AnoSag2.1"
SITES_ASSET = "anole-age-associated-sites.tsv.gz"
CHROMOSOMES_ASSET = "anole-chromosomes.tsv.gz"
SAMPLES_ASSET = "anole-samples.tsv.gz"
LOW_PERCENT = 10.0
HIGH_PERCENT = 90.0


def integer(description: str, unit: str) -> Column:
    return Column(kind="integer", description=description, unit=unit)


def number(description: str, unit: str) -> Column:
    return Column(kind="number", description=description, unit=unit, atol=1e-12, rtol=1e-8)


def text(description: str, unit: str) -> Column:
    return Column(kind="text", description=description, unit=unit)


SITE_COLUMNS = {
    "chrom": text("AnoSag2.1 scaffold", "scaffold"),
    "position_1": integer("canonical 1-based CpG coordinate", "base"),
    "observed_rows": integer("nonmissing sample methylation measurements", "rows"),
    "low_rows": integer("measurements strictly below 10 percent", "rows"),
    "high_rows": integer("measurements strictly above 90 percent", "rows"),
    "middle_rows": integer("measurements from 10 through 90 percent inclusive", "rows"),
    "extreme_site": integer("one when any sample measurement is extreme", "indicator"),
}
CHROMOSOME_COLUMNS = {
    "reference_bp": integer("full AnoSag2.1 scaffold length", "bases"),
    "age_associated_sites": integer("distinct supplied age-associated CpG dyads", "sites"),
    "age_associated_density_per_bp": number("all age-associated CpG dyads per reference base", "sites/base"),
    "unique_extreme_sites": integer("distinct age-associated CpG dyads with any extreme measurement", "sites"),
    "filtered_density_per_bp": number("unique extreme age-associated CpGs per reference base", "sites/base"),
    "expected_sites_length_null": number("site count expected under uniform per-base distribution", "sites"),
    "chi2_contribution": number("(observed minus expected)^2 divided by expected", "chi-square"),
}
SUMMARY_COLUMNS = {
    "age_associated_sites": integer("all supplied BH-significant CpG dyads", "sites"),
    "observed_measurement_rows": integer("all nonmissing site-by-sample measurements", "rows"),
    "removed_middle_rows": integer("measurement rows removed by strict extreme filter", "rows"),
    "unique_extreme_sites": integer("deduplicated extreme CpG dyads", "sites"),
    "scaffolds_with_filtered_sites": integer("scaffolds containing at least one extreme CpG dyad", "scaffolds"),
    "mean_filtered_density_per_bp": number("mean over scaffolds with filtered CpGs only", "sites/base"),
    "highest_age_associated_density_scaffold": text(
        "scaffold with the largest unfiltered age-associated density", "scaffold"
    ),
    "highest_age_associated_density_per_bp": number("largest unfiltered age-associated density", "sites/base"),
    "highest_filtered_density_scaffold": text("scaffold with the largest extreme-filtered density", "scaffold"),
    "chi2_length_null": number("Pearson statistic for reference-length-proportional expected counts", "chi-square"),
    "chi2_df": integer("14 eligible scaffolds minus one", "degrees of freedom"),
    "chi2_pvalue": number("upper-tail chi-square probability", "probability"),
}


def reference(sites_text: str, chromosomes_text: str, samples_text: str) -> dict:
    """Compute complete per-site, per-scaffold, and cohort audit records."""
    chromosomes = {
        row["chrom"]: int(row["reference_bp"]) for row in csv.DictReader(io.StringIO(chromosomes_text), delimiter="\t")
    }
    if len(chromosomes) != 14 or any(length <= 0 for length in chromosomes.values()):
        raise ValueError("Expected all 14 observed AnoSag2.1 scaffold lengths")
    reader = csv.DictReader(io.StringIO(sites_text), delimiter="\t")
    if reader.fieldnames is None or reader.fieldnames[:6] != [
        "chrom",
        "position_1",
        "n_samples",
        "spearman_rho",
        "p_value",
        "bh_q_value",
    ]:
        raise ValueError("Unexpected age-associated site header")
    samples = [row["sample_id"] for row in csv.DictReader(io.StringIO(samples_text), delimiter="\t")]
    if len(samples) != 37 or len(set(samples)) != 37 or set(samples) != set(reader.fieldnames[6:]):
        raise ValueError("Expected all 37 observed animals")
    site_rows = {}
    age_by_chromosome = Counter()
    extreme_by_chromosome = Counter()
    observed_total = 0
    removed_total = 0
    for row in reader:
        chrom = row["chrom"]
        position = int(row["position_1"])
        if chrom not in chromosomes or not 1 <= position <= chromosomes[chrom]:
            raise ValueError("Age-associated CpG outside a selected scaffold")
        if float(row["bh_q_value"]) >= 0.05:
            raise ValueError("Input site is not BH-significant")
        site_id = f"{chrom}:{position}"
        if site_id in site_rows:
            raise ValueError("Duplicate collapsed CpG dyad")
        percentages = [float(row[sample]) for sample in samples if row[sample]]
        if len(percentages) != int(row["n_samples"]) or not 30 <= len(percentages) <= 37:
            raise ValueError("Observed sample count mismatch")
        if any(not 0 <= value <= 100 for value in percentages):
            raise ValueError("Methylation percentage outside 0-100")
        low = sum(value < LOW_PERCENT for value in percentages)
        high = sum(value > HIGH_PERCENT for value in percentages)
        middle = len(percentages) - low - high
        extreme = int(low + high > 0)
        site_rows[site_id] = {
            "chrom": chrom,
            "position_1": position,
            "observed_rows": len(percentages),
            "low_rows": low,
            "high_rows": high,
            "middle_rows": middle,
            "extreme_site": extreme,
        }
        age_by_chromosome[chrom] += 1
        extreme_by_chromosome[chrom] += extreme
        observed_total += len(percentages)
        removed_total += middle
    if not site_rows or not sum(extreme_by_chromosome.values()):
        raise ValueError("No extreme age-associated CpGs")
    total_extreme = sum(extreme_by_chromosome.values())
    total_bases = sum(chromosomes.values())
    chromosome_rows = {}
    for chrom, length in chromosomes.items():
        count = extreme_by_chromosome[chrom]
        expected = total_extreme * length / total_bases
        chromosome_rows[chrom] = {
            "reference_bp": length,
            "age_associated_sites": age_by_chromosome[chrom],
            "age_associated_density_per_bp": age_by_chromosome[chrom] / length,
            "unique_extreme_sites": count,
            "filtered_density_per_bp": count / length,
            "expected_sites_length_null": expected,
            "chi2_contribution": (count - expected) ** 2 / expected,
        }
    statistic = sum(row["chi2_contribution"] for row in chromosome_rows.values())
    eligible = [row["filtered_density_per_bp"] for row in chromosome_rows.values() if row["unique_extreme_sites"]]
    highest_age = max(
        chromosomes,
        key=lambda chrom: (chromosome_rows[chrom]["age_associated_density_per_bp"], -int(chrom.split("_")[1])),
    )
    highest_filtered = max(
        chromosomes, key=lambda chrom: (chromosome_rows[chrom]["filtered_density_per_bp"], -int(chrom.split("_")[1]))
    )
    summary = {
        "age_associated_sites": len(site_rows),
        "observed_measurement_rows": observed_total,
        "removed_middle_rows": removed_total,
        "unique_extreme_sites": total_extreme,
        "scaffolds_with_filtered_sites": len(eligible),
        "mean_filtered_density_per_bp": sum(eligible) / len(eligible),
        "highest_age_associated_density_scaffold": highest_age,
        "highest_age_associated_density_per_bp": chromosome_rows[highest_age]["age_associated_density_per_bp"],
        "highest_filtered_density_scaffold": highest_filtered,
        "chi2_length_null": statistic,
        "chi2_df": 13,
        "chi2_pvalue": float(chi2.sf(statistic, 13)),
    }
    return {"summary": {"anole": summary}, "sites": site_rows, "chromosomes": chromosome_rows}


def generate_anole_cpg(_seed: int) -> Instance:
    sites = source_text(SOURCE_ID, SITES_ASSET)
    chromosomes = source_text(SOURCE_ID, CHROMOSOMES_ASSET)
    samples = source_text(SOURCE_ID, SAMPLES_ASSET)
    result = reference(sites, chromosomes, samples)
    contract = Contract(
        columns=SUMMARY_COLUMNS,
        expected=result["summary"],
        tables={
            "sites.tsv": TableContract(columns=SITE_COLUMNS, expected=result["sites"], max_bytes=32 * 1024 * 1024),
            "scaffolds.tsv": TableContract(
                columns=CHROMOSOME_COLUMNS, expected=result["chromosomes"], max_bytes=128 * 1024
            ),
        },
    )
    protocol = {
        "assembly": "AnoSag2.1 (GCF_025583915.1)",
        "source": SOURCE_ID,
        "age_association": "two-sided Spearman with BH q<0.05 over all tested CpG dyads on 14 scaffolds",
        "low_strict_percent": LOW_PERCENT,
        "high_strict_percent": HIGH_PERCENT,
        "reference_scaffolds": 14,
        "distribution_null": "extreme-site counts proportional to full reference scaffold bases",
    }
    original = contract.answer()[0]
    return Instance(
        "Audit the observed brown-anole age-associated CpGs in /app/inputs. The site table contains every "
        "BH-significant CpG dyad selected from 37 age-linked GEO GSE285624 methylomes on the 14 longest "
        "AnoSag2.1 scaffolds. It uses 1-based canonical coordinates and methylation percentages from 0 to 100; "
        "blank cells are unmeasured and never count as rows. The age-association selection is already fixed and "
        "must not be rerun or thinned. For every site, count observed sample measurements strictly below 10%, "
        "strictly above 90%, and in the inclusive middle range [10%,90%]. A site is extreme when any observed "
        "sample is below 10% or above 90%. Write one complete sites.tsv row per input site, keyed by "
        "scaffold:position; use 0/1 for extreme_site. For every one of the 14 scaffolds, count unique extreme "
        "sites once regardless of how many samples are extreme. Also count every supplied age-associated site "
        "per scaffold before the extreme filter. Divide each count by the full matched reference scaffold length "
        "in base pairs. Write all 14 scaffold results to scaffolds.tsv, including zero counts if any. For the "
        "filtered-density mean, include only scaffolds with at least one extreme CpG and take the unweighted "
        "arithmetic mean of their per-base densities, not a pooled density. Identify the scaffold with the "
        "highest density of all age-associated CpGs before the extreme filter, and separately the scaffold "
        "with the highest filtered density. Resolve an exact density tie by the lower numeric scaffold suffix. "
        "For the distribution test, expect counts proportional to full reference base lengths and report each "
        "Pearson chi-square contribution, the summed statistic with 13 degrees of freedom, and its upper-tail "
        "p-value. In answer.json count removed measurement rows in the inclusive middle range, not unique "
        "removed sites. Use all supplied age-associated CpGs and all 37 sample columns. The source's strand "
        "values were collapsed by unweighted within-sample means before this task; it is a GEO-derived "
        "independent adaptation, not the avian benchmark study.",
        {
            "age_associated_sites.tsv": sites,
            "chromosomes.tsv": chromosomes,
            "samples.tsv": samples,
            "protocol.json": json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        },
        contract,
        {
            "pooled_instead_of_nonzero_mean_density": [
                {
                    **original,
                    "mean_filtered_density_per_bp": (
                        original["unique_extreme_sites"]
                        / sum(row["reference_bp"] for row in result["chromosomes"].values())
                    ),
                }
            ],
            "count_removed_sites_instead_of_rows": [
                {**original, "removed_middle_rows": original["age_associated_sites"]}
            ],
            "wrong_distribution_statistic": [{**original, "chi2_length_null": original["chi2_length_null"] + 1}],
            "wrong_highest_age_associated_scaffold": [
                {
                    **original,
                    "highest_age_associated_density_scaffold": next(
                        chrom
                        for chrom in result["chromosomes"]
                        if chrom != original["highest_age_associated_density_scaffold"]
                    ),
                }
            ],
        },
        data_origin=DataOrigin.REAL,
        source_ids=(SOURCE_ID,),
        derivation=(
            "All 37 observed GEO GSE285624 QC sample columns joined to GEO age metadata; CpG complementary "
            "strands collapsed at canonical coordinates, at least 30 measured samples, two-sided Spearman/BH "
            "q<0.05 over every eligible site on the 14 longest matched AnoSag2.1 scaffolds. The complete "
            "age-associated set is supplied without effect-size or top-k thinning. The task independently "
            "classifies every observed measurement and every unique CpG, reports the nonzero-filtered mean and "
            "all-site density maximum, and tests full-reference-length densities."
        ),
        workflow_scope=WorkflowScope.CONNECTED,
    )


RECIPES = (
    Recipe(
        id="real-anole-age-cpg-density",
        version="1",
        skills=(
            "CpG coordinate identity",
            "strict methylation thresholds",
            "cross-sample deduplication",
            "chromosome-normalized density",
            "argmax and nonzero-only mean",
            "chi-square distribution test",
        ),
        formats=("TSV", "JSON"),
        sources=("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE285624",),
        generate=generate_anole_cpg,
        oracle_timeout=1800,
        oracle_runtime=OracleRuntime.NATIVE,
    ),
)
