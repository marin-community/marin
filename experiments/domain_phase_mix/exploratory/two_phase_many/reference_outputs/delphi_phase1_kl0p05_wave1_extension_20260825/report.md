# Delphi KL0.05 phase-1 Wave 1 extension

This freezes 50 additional fit-budget continuations for the KL0.05 prefix before inspecting any branch
outcomes. It repeats the original five exposure-stratum and TV quotas, but treats all original 50 fit
directions plus proportional and UniMax-8 as fixed maximin repellers. The combined first wave therefore
contains 100 distinct runtime mixtures rather than a second independent draw of the same design.

All rows retain the original per-bucket phase-1 and total-exposure support checks across all five frozen
prefix candidates. Wave 1B launches the 50 fit rows plus one non-fit cross-wave anchor. That anchor repeats
Wave 1A's `fit_maximin_00` mixture with a distinct
continuation identity, isolating wave-to-wave execution drift without spending fit budget.

The candidate pool increased from 20,000 to 100,000 draws per concentration because excluding Wave 1A left
cells 1/1 and 4/3 empty and cell 2/2 forced. This was detected before reading endpoint outcomes; all exposure
and TV quotas remained frozen. Forced candidate cells in the enlarged pool:
`[]`.

Minimum extension-to-existing direction distance: 0.2412

Minimum combined Wave 1 direction distance: 0.2412

Continuation weights SHA-256: `2860d0e1f177f1728580ec1cdda05e049734e7977b868a8c0abd05d9d8bd0ec3`
