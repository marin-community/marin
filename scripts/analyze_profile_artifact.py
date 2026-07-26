"""Download and summarize one XProf artifact from same-region object storage."""

import argparse
from collections import Counter
from pathlib import Path

from levanter.utils.profile_dirs import mirror_profile_dir
from marin.profiling.report import build_markdown_report
from marin.profiling.xplane import _load_tables, summarize_xplane


_INTERESTING_OP_TOKENS = (
    "clone",
    "dispatch",
    "combine",
    "ragged",
    "all_to_all",
    "moe_up_down",
    "mnnvl",
    "sendrowskernel",
    "copyrowskernel",
    "signalpeerskernel",
    "waitsignalskernel",
)


def _print_interesting_tables(table_dir: Path, table_name: str) -> None:
    print(f"XPROF_TABLE {table_name}")
    for table_index, (columns, rows, properties) in enumerate(_load_tables(table_dir / f"{table_name}.json")):
        print(
            f"XPROF_TABLE_INFO index={table_index} columns={columns!r} "
            f"rows={len(rows)} properties={properties!r}"
        )
        interesting = [
            row
            for row in rows
            if any(token in str(value).lower() for token in _INTERESTING_OP_TOKENS for value in row.values())
        ]
        ranked = sorted(
            interesting,
            key=lambda row: max(
                (float(value) for value in row.values() if isinstance(value, (int, float))),
                default=0.0,
            ),
            reverse=True,
        )
        for row in ranked[:100]:
            print(f"XPROF_ROW {row!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    local_profile = mirror_profile_dir(
        args.uri,
        Path("/tmp/profile-analysis"),
        run_id=args.run_id,
    )
    files = [path for path in local_profile.rglob("*") if path.is_file()]
    roots = Counter(path.relative_to(local_profile).parts[0] for path in files)
    print(f"PROFILE_FILES count={len(files)} bytes={sum(path.stat().st_size for path in files)} roots={roots!r}")

    xplanes = sorted(local_profile.rglob("steps-*/*.xplane.pb"))
    if len(xplanes) != 1:
        raise ValueError(f"expected one renamed XPlane session, found {xplanes}")
    table_dir = Path("/tmp/xprof-tables")
    summary = summarize_xplane(
        xplanes[0],
        output_dir=table_dir,
        warmup_steps=0,
        hot_op_limit=100,
        breakdown_mode="exclusive_global",
    )
    print(build_markdown_report(summary, top_k=50))
    _print_interesting_tables(table_dir, "framework_op_stats")
    _print_interesting_tables(table_dir, "hlo_stats")


if __name__ == "__main__":
    main()
