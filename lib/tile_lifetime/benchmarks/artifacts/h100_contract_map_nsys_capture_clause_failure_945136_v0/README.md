# H100 evidence image Nsight capture-clause failure

GitHub Actions run
[31486333761](https://github.com/marin-community/marin/actions/runs/31486333761)
used exact source `945136d32807420c882f0d10bb1c833ee22dfbe5` for the
manual `image_set=h100-evidence` build. The sole non-matrix H100 image job ran;
all five legacy image jobs were skipped. The run was dispatched once and was
not rerun.

The bounded v2 diagnostic recovered the exact 103-byte
`capture-range-end` possible-values clause. It did not recover a unique
`cuda-graph-trace` clause, so this artifact supports only the capture grammar
repair and makes no claim about the graph option's real syntax.

`run.json` is the terminal workflow snapshot. `source-identity.json` binds the
workflow and image inputs. `log-identity.json` records the complete remote job
log identity; `failure-excerpt.log` is the bounded exact excerpt.

No image was exported, no OCI digest was produced, and no GPU query or H100
evidence job occurred.
