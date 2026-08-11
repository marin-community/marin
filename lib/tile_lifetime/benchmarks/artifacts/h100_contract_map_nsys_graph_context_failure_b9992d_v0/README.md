# H100 evidence image Nsight graph-context failure

GitHub Actions run
[31488863448](https://github.com/marin-community/marin/actions/runs/31488863448)
used exact source `b9992d83251e35e06d362d1d2717842173d50652` for the
manual `image_set=h100-evidence` build. The sole non-matrix H100 image job ran;
all five legacy image jobs were skipped. The run was dispatched once and was
not rerun.

The bounded v3 diagnostic recovered the exact 51-byte
`cuda-graph-trace` declaration. It found nine locally scoped candidate lines
with six standalone `graph` and seven standalone `node` occurrences. The exact
sequence was ambiguous, so it emitted no candidate-line text. This artifact
does not justify a graph grammar change.

`run.json` is the terminal workflow snapshot. `source-identity.json` binds the
workflow and image inputs. `log-identity.json` records the complete remote job
log identity; `failure-excerpt.log` is the bounded exact excerpt.

No image was exported, no OCI digest was produced, and no GPU query or H100
evidence job occurred.
