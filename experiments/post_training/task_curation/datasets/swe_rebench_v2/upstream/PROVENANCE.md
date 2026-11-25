# Vendored upstream harness provenance

Source: `github.com/SWE-rebench/SWE-rebench-V2` (MIT), pinned revision
`3beb41a49dcfa23dcfe9112e92a35ae7bbc3a03d` (main, "Merge pull request #6 from
SWE-rebench/swe-rebench-v2.1", committed 2026-11-13T20:42:57Z).

Vendored from upstream:

- `log_parsers.py` ← `lib/harbor_tasks/log_parsers.py` (adds type aliases for
  upstream annotations that otherwise reference undefined names)
- `task_files.py` ← `lib/harbor_tasks/task_files.py` (adapted: `HERE` points at this directory so the
  vendored templates and parsers load)
- `templates/{test.sh, solve.sh, apply_patch.sh}` ← `lib/harbor_tasks/templates/`
- `templates/grade.py.tmpl` ← `lib/harbor_tasks/templates/grade.py` (renamed and changed as described below)
- `LICENSE` ← the repository MIT license

The dataset rows come from `ibragim-bad/SWE-rebench-V2.1` (CC BY 4.0), pinned at
`26c0145ded93168b0d5d095420155901ad2d5a2f` — the same author's re-verified release these templates render.
The renderer keeps the upstream task format and tests, subject to the grader change below.

The vendored `grade.py.tmpl` adds `unscorable: true` and an `UNSCORABLE` log line
when the parser reports no named test results, even when the config expects test
IDs. Package build-error markers and parser placeholders do not count as named
results. The grader exits nonzero for an unscorable log, and `test.sh` leaves
reward at zero with a diagnostic. A package build failure also prevents a
resolved verdict when the configured tests passed; other parsed errors do not.
`normalized_passed_subset` still requires a nonempty expected set.

The vendored Go parser strips the package prefix from `go test -json` test
results so they match the published config IDs. Package-level `ok` and JSON
`pass` lines remain summaries, not named results. When multiple packages emit
the same test name, a failure or skip takes precedence over a pass.

The vendored `log_parsers.py` also changes `parse_java_mvn`: upstream marks every
class seen in a `Running` line as passed when no result was parsed. This copy
requires a completed-test result, such as a Maven summary or `BUILD SUCCESS`,
before marking that class passed. A forked JVM that terminates before reporting
a result therefore leaves the class unresolved and reward at zero.
