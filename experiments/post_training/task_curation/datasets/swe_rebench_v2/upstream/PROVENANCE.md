# Vendored upstream harness provenance

Source: `github.com/SWE-rebench/SWE-rebench-V2` (MIT), pinned revision
`3beb41a49dcfa23dcfe9112e92a35ae7bbc3a03d` (main, "Merge pull request #6 from
SWE-rebench/swe-rebench-v2.1", committed 2026-11-13T20:42:57Z).

Vendored verbatim:

- `log_parsers.py` ← `lib/harbor_tasks/log_parsers.py`
- `task_files.py` ← `lib/harbor_tasks/task_files.py` (adapted: `HERE` points at this directory so the
  vendored templates and parsers load; module docstring unchanged)
- `templates/{test.sh, solve.sh, apply_patch.sh, grade.py.tmpl}` ← `lib/harbor_tasks/templates/`
- `LICENSE` ← the repository MIT license

The dataset rows come from `ibragim-bad/SWE-rebench-V2.1` (CC BY 4.0), pinned at
`26c0145ded93168b0d5d095420155901ad2d5a2f` — the same author's re-verified release these templates render.
Rendering a row through the vendored `task_files.py` produces files byte-identical to the tasks the upstream
publishes on the Harbor Hub (their own claim, preserved in the module docstring).

Repair deviations from upstream are made in the converter (`../convert.py`), never by editing the vendored
files; each deviation is disclosed in the task package and receipt evidence.
