# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage report for verified cross-source fuzzy deduplication."""

from pathlib import PurePosixPath

from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData

from experiments.datakit.reports.common import StageReport, render_template, write_report

VERIFICATION_COUNTER = "dedup/fuzzy/verification"


def _source_label(source_main_dir: str) -> str:
    return "/".join(source_main_dir.rstrip("/").split("/")[-3:])


def _counter_suffix(dedup: FuzzyDupsAttrData, prefix: str) -> dict[str, int]:
    start = f"{prefix}/"
    return {key[len(start) :]: int(value) for key, value in dedup.counters.items() if key.startswith(start)}


def _histogram(dedup: FuzzyDupsAttrData, metric: str) -> list[dict[str, int | str]]:
    counts = _counter_suffix(dedup, f"{VERIFICATION_COUNTER}/histogram/{metric}")
    return [{"bin": key, "pairs": value} for key, value in sorted(counts.items(), key=lambda item: int(item[0]))]


def _lsh_collision_curve(dedup: FuzzyDupsAttrData) -> dict:
    bands = dedup.params.num_bands
    rows_per_band = dedup.params.num_perms // bands

    def collision_probability(similarity: float) -> float:
        return 1 - (1 - similarity**rows_per_band) ** bands

    return {
        "bands": bands,
        "rows_per_band": rows_per_band,
        "midpoint": (1 - 0.5 ** (1 / bands)) ** (1 / rows_per_band),
        "points": [
            {
                "similarity": similarity,
                "collision_probability": collision_probability(similarity),
            }
            for similarity in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
        ],
    }


def dedup_report(output_path: str, dedup: FuzzyDupsAttrData) -> StageReport:
    """Render exact candidate, rejection, and score evidence."""
    candidates = int(dedup.counters.get(f"{VERIFICATION_COUNTER}/candidates", 0))
    verified = int(dedup.counters.get(f"{VERIFICATION_COUNTER}/decision/accepted", 0))
    rejection_counts = _counter_suffix(dedup, f"{VERIFICATION_COUNTER}/decision")
    rejection_counts.pop("accepted", None)
    rejected = candidates - verified
    transitive_kept = int(dedup.counters.get("dedup/fuzzy/document/transitive_members_kept", 0))

    per_source = []
    for source_main_dir, entry in dedup.sources.items():
        source_tag = PurePosixPath(entry.attr_dir).name
        decisions = _counter_suffix(dedup, f"{VERIFICATION_COUNTER}/source/{source_tag}/decision")
        source_candidates = sum(decisions.values())
        source_verified = decisions.get("accepted", 0)
        per_source.append(
            {
                "label": _source_label(source_main_dir),
                "source_main_dir": source_main_dir,
                "candidates": source_candidates,
                "verified": source_verified,
                "rejected": source_candidates - source_verified,
                "acceptance_rate": source_verified / source_candidates if source_candidates else 0.0,
            }
        )

    stats = {
        "candidates": candidates,
        "verified_duplicates": verified,
        "rejected_candidates": rejected,
        "acceptance_rate": verified / candidates if candidates else 0.0,
        "transitive_members_kept": transitive_kept,
        "n_sources": len(dedup.sources),
    }
    data = {
        "params": dedup.params.model_dump(),
        "verification": dedup.verification.model_dump(),
        "decisions_dir": dedup.decisions_dir,
        "lsh_collision_curve": _lsh_collision_curve(dedup),
        "stats": stats,
        "rejections": [
            {"reason": reason, "pairs": pairs}
            for reason, pairs in sorted(rejection_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "sources": sorted(per_source, key=lambda row: (-row["candidates"], row["label"])),
        "histograms": {
            metric: _histogram(dedup, metric)
            for metric in ("member_containment", "jaccard", "member_unique", "shared_buckets")
        },
    }
    page = render_template("dedup.html", title="Datakit dedup", data=data)
    return StageReport(html_path=write_report(output_path, page), stats=stats)
