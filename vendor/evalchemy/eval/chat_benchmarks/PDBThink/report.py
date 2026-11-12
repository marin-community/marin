"""Reduce compact PDBThink grades with the upstream report protocol."""


def report_metrics(rows: list[dict]) -> dict:
    # Reporting belongs to the same pinned optional package as parsing/scoring.
    from pdbthink.reporting.report import analyse_run

    primary = [row for row in rows if row["representation"] != "context_only" and not row["is_rotation_variant"]]
    if not primary:
        return {"macro_score": None, "micro_score": None, "n_results": len(rows)}
    report = analyse_run(rows, bootstrap_samples=2000)
    return {
        "macro_score": report["macro"]["point"],
        "macro_score_ci_low": report["macro"]["ci_low"],
        "macro_score_ci_high": report["macro"]["ci_high"],
        "micro_score": report["micro_score"],
        "n_results": report["n_results"],
        "n_instances": report["n_instances"],
        "n_clusters": report["n_clusters"],
        "tool_calls": sum(row["tool_call"] for row in rows),
        **{f"{family}/score": values["mean_score"] for family, values in report["per_family"].items()},
        "pdbthink_report": report,
    }
