# Vendored upstream harness provenance

Source: `github.com/SWE-rebench/SWE-rebench-V2` (MIT), pinned revision
`3beb41a49dcfa23dcfe9112e92a35ae7bbc3a03d` (main, "Merge pull request #6 from
SWE-rebench/swe-rebench-v2.1", committed 2026-11-13T20:42:57Z).

Vendored from upstream:

- `log_parsers.py` ← `lib/harbor_tasks/log_parsers.py` (adds type aliases for
  upstream annotations that otherwise reference undefined names)
- `task_files.py` ← `lib/harbor_tasks/task_files.py` (adapted: `HERE` points at this directory so the
  vendored templates and parsers load; module docstring describes the grader change)
- `templates/{test.sh, solve.sh, apply_patch.sh, grade.py.tmpl}` ← `lib/harbor_tasks/templates/`
- `LICENSE` ← the repository MIT license

The dataset rows come from `ibragim-bad/SWE-rebench-V2.1` (CC BY 4.0), pinned at
`26c0145ded93168b0d5d095420155901ad2d5a2f` — the same author's re-verified release these templates render.
The renderer keeps the upstream task format and tests, subject to the grader change below.

The vendored `grade.py.tmpl` marks an empty parsed log with no expected test IDs as
`UNSCORABLE`, leaves `resolved` false, and exits nonzero. This prevents a vacuous
subset pass and leaves `test.sh`'s initialized reward at zero. The source template
at the pinned revision lacks this guard.
