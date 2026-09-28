# Shared Findings

## Current Status

No investigated surrogate has established globally reliable two-phase policy selection. The strongest result is local: phase asymmetry can be modeled compactly around exact aggregate fibers, but the resulting global optimum is nearly phase-tied and aggregate extrapolation remains optimistic.

## Cross-Session Synthesis

The most stable compact chronological law is finite-potential transport:

\[
\widehat L_t=F_t(a)+\theta_t O_t+\gamma_tJ_t,
\]

\[
O_t=\alpha_0\alpha_1\left[F_t(w^{(0)})-F_t(w^{(1)})\right],\qquad J_t=\alpha_0F_t(w^{(0)})+\alpha_1F_t(w^{(1)})-F_t(a),
\]

where \(a=\alpha_0w^{(0)}+\alpha_1w^{(1)}\). Its implied early-state recency share,

\[
r=\alpha_0+\theta\alpha_0\alpha_1,
\]

is about \(0.53\)–\(0.55\) across both targets and two aggregate spines. Sharing this coefficient across targets preserves paired-delta CV fit.

This is not a global winner:

- Zero phase transition remains best on the exposed exact aggregate fibers.
- Target-matched heldout changes are much smaller than the remaining modeling errors and do not improve Regret@1.
- Raw two-phase optima have phase TV near zero.
- Aggregate raw optima remain outside ordinary support and optimistic.

## Structural Result

Writing an aggregate-preserving phase contrast as

\[
w^{(0)}=a+\alpha_1d,\qquad w^{(1)}=a-\alpha_0d,\qquad \mathbf 1^\top d=0,
\]

the first-order finite-potential transport incentive is proportional to \(\nabla F(a)^\top d\). At an interior constrained optimum of \(F\), the simplex KKT condition makes this term zero. With the fitted convex separation term, the tied policy becomes a local minimum along phase directions. A phase law that only transports the aggregate potential therefore cannot create a robust asymmetric optimum near the aggregate optimum.

The required escape hatch is

\[
\widehat L(a,d)=F(a)+G(a)^\top d+C(a,d),\qquad C(a,0)=0,
\]

with a phase-specific marginal-value state \(G(a)\) that has a mechanistic definition and is independently identified. A free residual field, an ensemble, or post-hoc calibration is not admissible.

## What the Five Sessions Added

- Session 1 developed reversal-aware and kernelized phase decompositions, then exposed severe null spaces and non-identification.
- Session 2 found local residual-spectrum transport variants with some RMSE gains, but unstable optima and incomplete shape transfer.
- Session 3 showed that apparent phase gains can be inherited from aggregate-support error; its initial winner failed broader falsification.
- Session 4 produced finite-potential transport and useful identified-set diagnostics.
- Session 5 developed commutator-style chronology and formal amplitude/direction non-identification arguments.

The complete evidence is under `evidence/prior_sessions/`; the merged mechanism registry is `evidence/cross_session_phase_transport_20260723/consolidated_approach_registry.csv`.
