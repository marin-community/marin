# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the retained top-100 baseline with four expanded top-200 lists."""

import argparse
import json
from itertools import combinations, pairwise
from pathlib import Path

from analyze_rankings import (
    RANKINGS,
    SOFTWARE_TYPES,
    analyze,
    diversity,
    effective_categories,
    read_rows,
    tag_domain_counts,
)

BASELINE_SIZE = 100
EXPANDED_SIZE = 200
DEPTHS = (BASELINE_SIZE, 125, 150, 175, EXPANDED_SIZE)
LABEL_FIELDS = ("source_type", "primary_domain", "manual_topic")


def tag_summary(ids: set[str], sources: dict, mapping: dict[str, str]) -> dict:
    """Count observed tags and fractional domain assignments without filling missing tags."""
    github = [sources[key]["github"] for key in sorted(ids) if sources[key]["github"]]
    tagged = [record for record in github if record["topics"]]
    tags = {tag for record in tagged for tag in record["topics"]}
    domains = tag_domain_counts(tagged, mapping)
    return {
        "github_mapped": len(github),
        "tagged_sources": len(tagged),
        "raw_tags": sorted(tags),
        "raw_tag_count": len(tags),
        "mapped_sources": domains.mapped_sources,
        "mapped_domains": sorted(domains.fractional),
        "fractional_domains": dict(sorted(domains.fractional.items())),
        "effective_domain_bins": effective_categories(domains.fractional),
    }


def compare(baseline_dir: Path, expanded_dir: Path, date: str) -> dict:
    """Validate the unchanged baseline and measure incremental coverage at each depth."""
    baseline_result = analyze(baseline_dir, date, BASELINE_SIZE).summary
    expanded_result = analyze(expanded_dir, date, EXPANDED_SIZE).summary
    baseline_rows = read_rows(baseline_dir / f"rankings-{date}.csv")
    rows = read_rows(expanded_dir / f"rankings-{date}.csv")
    original_labels = {row["source_id"]: row for row in read_rows(baseline_dir / f"source-annotations-{date}.csv")}
    labels = {row["source_id"]: row for row in read_rows(expanded_dir / f"source-annotations-{date}.csv")}
    sources = {
        row["source_id"]: row for row in json.loads((expanded_dir / f"source-observations-{date}.json").read_text())
    }
    original_sources = json.loads((baseline_dir / f"source-observations-{date}.json").read_text())
    for key, label in original_labels.items():
        assert all(labels[key][field] == label[field] for field in LABEL_FIELDS), key
    for source in original_sources:
        current = sources[source["source_id"]]["github"]
        assert bool(current) == bool(source["github"]), source["source_id"]
        if current:
            assert current["topics"] == source["github"]["topics"], source["source_id"]
            assert current["stars"] == source["github"]["stars"], source["source_id"]
    mapping = {row["tag"]: row["domain"] for row in read_rows(expanded_dir / f"tag-domains-{date}.csv")}
    cohorts = {name: [row for row in rows if row["ranking"] == name] for name in RANKINGS}
    baseline_union = set(original_labels)
    old_domains = {row["primary_domain"] for row in original_labels.values()}
    old_topics = {row["manual_topic"] for row in original_labels.values()}
    result = {
        "date": date,
        "baseline_union": len(baseline_union),
        "expanded_union": len(labels),
        "new_sources": len(set(labels) - baseline_union),
        "new_domains": sorted({row["primary_domain"] for row in labels.values()} - old_domains),
        "new_topics": sorted({row["manual_topic"] for row in labels.values()} - old_topics),
        "rankings": {},
        "pairs": [],
        "union_by_depth": {},
        "tag_comparison": "Both depths use the expanded exact-string mapping and frozen baseline metadata.",
    }
    for name, cohort in cohorts.items():
        original = [row for row in baseline_rows if row["ranking"] == name]
        assert [(row["source_id"], row["score"]) for row in cohort[:BASELINE_SIZE]] == [
            (row["source_id"], row["score"]) for row in original
        ], name
        first = {row["source_id"] for row in cohort[:BASELINE_SIZE]}
        second = {row["source_id"] for row in cohort[BASELINE_SIZE:]}
        first_domains = {labels[key]["primary_domain"] for key in first}
        first_topics = {labels[key]["manual_topic"] for key in first}
        second_topics = {labels[key]["manual_topic"] for key in second}
        tags_first = tag_summary(first, sources, mapping)
        tags_second = tag_summary(second, sources, mapping)
        tags_full = tag_summary(first | second, sources, mapping)
        result["rankings"][name] = {
            "first100": baseline_result["rankings"][name],
            "second100": diversity([labels[row["source_id"]] for row in cohort[BASELINE_SIZE:]]),
            "full200": expanded_result["rankings"][name],
            "new_to_baseline_union": sorted(second - baseline_union),
            "new_domains_to_own100": sorted({labels[key]["primary_domain"] for key in second} - first_domains),
            "new_topics_to_own100": sorted(second_topics - first_topics),
            "new_topics_to_baseline_union": sorted(second_topics - old_topics),
            "new_topic_sources": {
                topic: sorted(key for key in second if labels[key]["manual_topic"] == topic)
                for topic in sorted(second_topics - old_topics)
            },
            "second100_software_subset": diversity(
                [
                    labels[row["source_id"]]
                    for row in cohort[BASELINE_SIZE:]
                    if labels[row["source_id"]]["source_type"] in SOFTWARE_TYPES
                ]
            ),
            "tags_first100": tags_first,
            "tags_second100": tags_second,
            "tags_full200": tags_full,
            "new_raw_tags": sorted(set(tags_second["raw_tags"]) - set(tags_first["raw_tags"])),
            "new_tag_domains": sorted(set(tags_full["mapped_domains"]) - set(tags_first["mapped_domains"])),
            "blocks": [],
        }
        for start, stop in pairwise(DEPTHS):
            earlier = [labels[row["source_id"]] for row in cohort[:start]]
            block = [labels[row["source_id"]] for row in cohort[start:stop]]
            result["rankings"][name]["blocks"].append(
                {
                    "start_rank": start + 1,
                    "end_rank": stop,
                    **diversity(block),
                    "new_domains": sorted({r["primary_domain"] for r in block} - {r["primary_domain"] for r in earlier}),
                    "new_topics": sorted({r["manual_topic"] for r in block} - {r["manual_topic"] for r in earlier}),
                    "sources_new_to_baseline_union": sum(r["source_id"] not in baseline_union for r in block),
                    "cumulative_diversity": diversity(earlier + block),
                }
            )

    for first, second in combinations(RANKINGS, 2):
        pair = {"first": first, "second": second, "by_depth": {}}
        for depth in DEPTHS:
            a = {row["source_id"] for row in cohorts[first][:depth]}
            b = {row["source_id"] for row in cohorts[second][:depth]}
            pair["by_depth"][str(depth)] = {
                "shared": len(a & b),
                "overlap_fraction": len(a & b) / depth,
                "jaccard": len(a & b) / len(a | b),
            }
        result["pairs"].append(pair)
    for depth in DEPTHS:
        ids = {row["source_id"] for cohort in cohorts.values() for row in cohort[:depth]}
        result["union_by_depth"][str(depth)] = {
            **diversity([labels[key] for key in sorted(ids)]),
            "new_sources": len(ids - baseline_union),
            "new_topics": sorted({labels[key]["manual_topic"] for key in ids} - old_topics),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--expanded-dir", type=Path, required=True)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    result = compare(args.baseline_dir, args.expanded_dir, args.date)
    output = args.expanded_dir / f"expansion-results-{args.date}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Baseline preserved; {result['new_sources']} new sources; wrote {output}")


if __name__ == "__main__":
    main()
