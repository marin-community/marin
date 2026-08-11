# H100 evidence image Nsight graph-clause failure

GitHub Actions run
[31487416543](https://github.com/marin-community/marin/actions/runs/31487416543)
used exact source `b2083f35f3d6990eb021a52ece8e6bf27a5deb9d` for the
manual `image_set=h100-evidence` build. The sole non-matrix H100 image job ran;
all five legacy image jobs were skipped. The run was dispatched once and was
not rerun.

The bounded v2 diagnostic shows that the repaired `capture-range-end` grammar
accepted the exact 103-byte clause. It could not recover one unique
`cuda-graph-trace` possible-values clause. This artifact therefore records no
graph-option syntax and does not justify a graph grammar change.

`run.json` is the terminal workflow snapshot. `source-identity.json` binds the
workflow and image inputs. `log-identity.json` records the complete remote job
log identity; `failure-excerpt.log` is the bounded exact excerpt.

No image was exported, no OCI digest was produced, and no GPU query or H100
evidence job occurred.
