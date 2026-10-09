# Campaign analysis

`snowball_tournament/pairwise.py` selects one Snowball-lineage checkpoint from
the five candidate rows in the campaign tracker. It adapts the September 17
campaign's `generate_snowball_paired_win_rates.py` and writes the complete
pairwise table, ranking, and tracker-snapshot hash.

`critical_difference/plot.py` adapts the September 17 campaign's
`release-figures/generate_release_figures.py`. It plots the selected checkpoint
against the ten non-Snowball baselines in `critical_difference/baseline_flops.csv`.
It reads each scored archive and any audited recovered metrics, then writes the
cell-level uncertainty table, FLOP table, rank tables, and uncontrolled and
FLOP-controlled critical-difference figures. The output README records the
common benchmark panel and tracker-snapshot hash.

Run the tournament before plotting. Both commands and their required inputs are
specified in the campaign release document's “Reproduce the Snowball tournament
and critical-difference figures” section. These scripts report the results in
the supplied tracker; they do not attest policy conformance.
