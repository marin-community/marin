# H100 evidence image bounded Nsight help failure

GitHub Actions run
[31485145628](https://github.com/marin-community/marin/actions/runs/31485145628)
used exact source `eb8cf4c5cee6b288822fe6600d1702ecd7c5aa83` for the
manual `image_set=h100-evidence` build. The sole non-matrix H100 image job ran;
all five legacy image jobs were skipped. The run was dispatched once and was
not rerun.

The pinned Nsight Systems help validator retained the original closed-parser
rejection and emitted its bounded diagnostic. Both parser-authoritative option
blocks exceeded the reviewed 1,024-byte text limit, so the diagnostic records
only their exact normalized UTF-8 byte lengths and SHA-256 identities. It does
not contain enough text to repair the grammar without a narrower reviewed
diagnostic.

`run.json` is the terminal workflow snapshot. `source-identity.json` binds the
workflow and image inputs. `log-identity.json` records the complete remote job
log identity; `failure-excerpt.log` is the bounded exact excerpt.

No image was exported, no OCI digest was produced, and no GPU query or H100
evidence job occurred.
