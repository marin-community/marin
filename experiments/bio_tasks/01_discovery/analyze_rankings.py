# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recompute the BioTasks discovery comparison from the retained source tables."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path

RANKINGS = ("bioconda", "bioconductor", "pypi", "github")
SOFTWARE_TYPES = {"Software", "Infrastructure", "Workflow", "Research implementation"}
BROAD_DOMAINS = {"Computing infrastructure", "General biology & multi-omics"}
CUTOFFS = (10, 25, 50, 100)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def effective_categories(counts: Counter) -> float:
    """Return exp(Shannon entropy), the number of equally frequent categories."""
    total = counts.total()
    if not total:
        return 0.0
    return math.exp(-sum((count / total) * math.log(count / total) for count in counts.values() if count))


def diversity(rows: list[dict[str, str]]) -> dict:
    domains = Counter(row["primary_domain"] for row in rows)
    topics = Counter(row["manual_topic"] for row in rows)
    types = Counter(row["source_type"] for row in rows)
    return {
        "n": len(rows),
        "domains": dict(sorted(domains.items())),
        "source_types": dict(sorted(types.items())),
        "domain_bins": len(domains),
        "manual_topic_bins": len(topics),
        "effective_domain_bins": effective_categories(domains),
        "largest_domain_share": max(domains.values(), default=0) / len(rows) if rows else 0.0,
    }


def analyze(data_dir: Path, date: str, size: int = 100) -> dict:
    """Validate retained measurements and compute overlap and diversity."""
    provenance = json.loads((data_dir / f"ranking-provenance-{date}.json").read_text())
    for filename, expected in provenance["retained_artifact_sha256"].items():
        assert hashlib.sha256((data_dir / filename).read_bytes()).hexdigest() == expected, filename

    rows = read_rows(data_dir / f"rankings-{date}.csv")
    annotation_rows = read_rows(data_dir / f"source-annotations-{date}.csv")
    annotations = {row["source_id"]: row for row in annotation_rows}
    observations = json.loads((data_dir / f"source-observations-{date}.json").read_text())
    sources = {row["source_id"]: row for row in observations}
    assert len(annotations) == len(annotation_rows)
    assert len(sources) == len(observations)
    assert set(annotations) == set(sources) == {row["source_id"] for row in rows}
    assert {row["ranking"] for row in rows} == set(RANKINGS)
    assert all(row["source_type"] and row["primary_domain"] and row["manual_topic"] for row in annotation_rows)
    tag_mapping = {row["tag"]: row["domain"] for row in read_rows(data_dir / f"tag-domains-{date}.csv")}

    cohorts = {ranking: [row for row in rows if row["ranking"] == ranking] for ranking in RANKINGS}
    cutoffs = sorted({k for k in CUTOFFS if k <= size} | {size})
    members = {}
    screens = {}
    for ranking, cohort in cohorts.items():
        assert [int(row["rank"]) for row in cohort] == list(range(1, size + 1)), ranking
        members[ranking] = {row["source_id"] for row in cohort}
        assert len(members[ranking]) == size
        assert cohort == sorted(cohort, key=lambda row: (-int(row["score"]), row["source_id"]))
        screen = read_rows(data_dir / f"ranking-candidates-{ranking}-{date}.csv")
        screens[ranking] = screen
        assert all(row["status"] != "unresolved" for row in screen)
        by_package = {row["package"]: row for row in screen}
        for row in cohort:
            packages = [by_package[name] for name in row["packages"].split(";")]
            assert all(package["source_id"] == row["source_id"] for package in packages)
            assert int(row["score"]) == max(int(package["score"]) for package in packages)
        selected = {row["source_id"] for row in screen if row["status"] == "selected source"}
        assert selected == members[ranking]
        assert all(
            int(row["score"]) < int(cohort[-1]["score"]) for row in screen if row["status"].startswith("below cutoff")
        )

    result = {"date": date, "union_sources": len(sources), "rankings": {}, "pairs": []}
    tag_counts = {}
    for ranking, cohort in cohorts.items():
        ids = members[ranking]
        labels = [annotations[row["source_id"]] for row in cohort]
        github = [sources[key]["github"] for key in sorted(ids) if sources[key]["github"] is not None]
        tagged = [record for record in github if record["topics"]]
        counts = Counter(tag for record in tagged for tag in sorted(set(record["topics"])))
        tag_counts[ranking] = counts
        tag_domains = Counter()
        fractional_domains = Counter()
        mapped_sources = 0
        for record in tagged:
            domains = {tag_mapping[tag] for tag in record["topics"] if tag in tag_mapping}
            if domains:
                mapped_sources += 1
                tag_domains.update(sorted(domains))
                for domain in sorted(domains):
                    fractional_domains[domain] += 1 / len(domains)
        assert math.isclose(fractional_domains.total(), mapped_sources)
        others = set().union(*(members[other] for other in RANKINGS if other != ranking))
        result["rankings"][ranking] = {
            **diversity(labels),
            "cutoff_score": int(cohort[-1]["score"]),
            "last_raw_rank": int(cohort[-1]["first_raw_rank"]),
            "unique_to_ranking": len(ids - others),
            "software_subset": diversity([row for row in labels if row["source_type"] in SOFTWARE_TYPES]),
            "specific_domain_subset": diversity([row for row in labels if row["primary_domain"] not in BROAD_DOMAINS]),
            "tags": {
                "github_mapped": len(github),
                "tagged_sources": len(tagged),
                "github_without_tags": len(github) - len(tagged),
                "no_github_mapping": size - len(github),
                "distinct_raw_tags": len(counts),
                "tag_assignments": counts.total(),
                "effective_raw_tags": effective_categories(counts),
                "top_tags": sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:12],
                "mapped_domain_sources": mapped_sources,
                "domain_presence": dict(sorted(tag_domains.items())),
                "fractional_domains": dict(sorted(fractional_domains.items())),
                "effective_domain_bins": effective_categories(fractional_domains),
            },
            "diversity_by_cutoff": {str(k): diversity(labels[:k]) for k in cutoffs},
        }

    for first, second in combinations(RANKINGS, 2):
        shared = members[first] & members[second]
        ranks_first = {row["source_id"]: int(row["rank"]) for row in cohorts[first]}
        ranks_second = {row["source_id"]: int(row["rank"]) for row in cohorts[second]}
        ids = sorted(shared)
        # Re-rank within the intersection to calculate Spearman, rather than
        # correlating the gaps in ranks after censoring at the cohort boundary.
        order_first = {key: rank for rank, key in enumerate(sorted(ids, key=ranks_first.__getitem__))}
        order_second = {key: rank for rank, key in enumerate(sorted(ids, key=ranks_second.__getitem__))}
        rho = (
            statistics.correlation([order_first[key] for key in ids], [order_second[key] for key in ids])
            if len(ids) > 1
            else None
        )
        result["pairs"].append(
            {
                "first": first,
                "second": second,
                "intersection": len(shared),
                "jaccard": len(shared) / len(members[first] | members[second]),
                "shared_sources": ids,
                "spearman_within_intersection": rho,
                "intersection_by_cutoff": {
                    str(k): len(
                        {row["source_id"] for row in cohorts[first][:k]}
                        & {row["source_id"] for row in cohorts[second][:k]}
                    )
                    for k in cutoffs
                },
                "raw_tag_jaccard": (
                    len(set(tag_counts[first]) & set(tag_counts[second]))
                    / len(set(tag_counts[first]) | set(tag_counts[second]))
                ),
            }
        )

    search_only = [
        row for row in screens["github"] if row["status"] != "excluded" and "github-search:" in row["discovery_routes"]
    ]
    search_only.sort(key=lambda row: (-int(row["score"]), row["source_id"]))
    search_ids = {row["source_id"] for row in search_only[:size]}
    assert len(search_ids) == size
    result["github_search_only_sensitivity"] = {
        f"shared_with_supplemented_top{size}": len(search_ids & members["github"]),
        "supplement_only_sources": sorted(members["github"] - search_ids),
        "cutoff_stars": int(search_only[size - 1]["score"]),
        "intersection_with_other_rankings": {
            ranking: len(search_ids & members[ranking]) for ranking in RANKINGS if ranking != "github"
        },
    }
    # Every truncated query returned below the selected cutoff, so page 2
    # cannot displace a selected source within these queries.
    for query in provenance["github"]["search_queries"]:
        assert not query["incomplete_results"]
        if query["total_matches"] > query["returned"]:
            assert query["last_returned_stars"] < int(cohorts["github"][-1]["score"])

    with (data_dir / f"ranking-topic-frequencies-{date}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["tag", "mapped_domain", *RANKINGS])
        for tag in sorted(set().union(*(set(counts) for counts in tag_counts.values()))):
            writer.writerow([tag, tag_mapping.get(tag, ""), *(tag_counts[ranking][tag] for ranking in RANKINGS)])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--size", type=int, default=100)
    args = parser.parse_args()
    result = analyze(args.data_dir, args.date, args.size)
    output = args.data_dir / f"ranking-results-{args.date}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Validated 4 x {args.size} ranking positions, {result['union_sources']} unique sources; wrote {output}")


if __name__ == "__main__":
    main()
