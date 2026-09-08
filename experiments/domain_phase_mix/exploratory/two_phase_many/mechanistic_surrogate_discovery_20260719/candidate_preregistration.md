# Round 1 candidate preregistration

The exposed adversarial panel is not used to fit or select any configuration in
this round. Candidate forms and expected signatures are frozen here before the
batch is evaluated against adversarial outcomes.

## Paired marginal-value transport

Let

$$
a=\alpha_0w^{(0)}+\alpha_1w^{(1)},\qquad
d=\alpha_0\alpha_1\left(w^{(1)}-w^{(0)}\right).
$$

The tied policy \((a,a)\) has exactly the same total physical bucket exposure
as \((w^{(0)},w^{(1)})\). A tied-data model (F_{1p}(a)) therefore identifies
the aggregate response independently of temporal order. For family (f), let

$$
m_i(a)=\frac{1}{\tau_f+E_i(a)}
$$

be remaining marginal learnability under a saturating acquisition law. The
phase correction is

$$
\Delta_{\mathrm{PMVT}}(a,d)
=\sum_f\theta_f\sum_{i\in f}m_i(a)d_i
+\sum_f\chi_f\sum_{i\in f}\left[m_i(a)d_i\right]^2,
\qquad \chi_f\ge 0.
$$

The linear term changes sign under phase reversal; the curvature term does not.
Both vanish exactly for a tied policy.

## Family commutator flow

If family update vector fields (V_f) do not commute, the
Baker--Campbell--Hausdorff expansion of two ordered phases contains

$$
\Delta_{\mathrm{FCF}}
=\alpha_0\alpha_1\sum_{f<g}K_{fg}
\left(W_f^{(0)}W_g^{(1)}-W_g^{(0)}W_f^{(1)}\right)r_{fg}(a),
\qquad K_{fg}=-K_{gf}.
$$

With the predeclared broad-text, tech-code, and reasoning partition this adds
three signed coefficients. The term is zero on the phase-tied diagonal and
antisymmetric under phase reversal.

## Identified fast-slow consolidation

For each family,

$$
\dot f_f=qW_f(1-f_f)-h(1-W_f)f_f,
\qquad
\dot s_f=k(f_f-s_f),
$$

and terminal capability is

$$
c_f=(1-\omega)f_f+\omega s_f.
$$

This repeats the prior two-pool state law only under a new identification
strategy: tied outcomes fit aggregate acquisition and response, while paired
two-minus-one-phase outcomes fit retention and consolidation. If this does not
remove the prior boundary/equifinality failure, the route remains rejected.

## Round 1 freeze

- Hyperparameters are selected only with algebraic checks, StarCoder,
  grouped-CV, paired Delphi/300M outcomes, production, and historical heldouts.
- The three forms are evaluated on the adversarial panel together, once, after
  their equations, ablations, and hyperparameters are frozen.
- The running frontier phase-fiber panel is untouched future confirmation
  evidence and is not read.
