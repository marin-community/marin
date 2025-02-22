1 Introduction
==============

1 Introduction
--------------

The $`2S`$ state of muonic hydrogen offers interesting possibilities to
do precision tests of QED and to determine the proton RMS charge radius
(see and references therein). An isolated $`(\mu p)_{2S}`$ is metastable
with a lifetime mainly determined by muon decay (about $`2.2\mu `$s). In
liquid or gaseous hydrogen the lifetime of the $`2S`$ state is shortened
considerably because of Stark mixing followed by $`2P1S`$ radiative
transitions. If a sizeable fraction of muonic hydrogen atoms ends up in
the $`2S`$ state with a sufficiently long lifetime, then precision laser
experiments with this metastable $`2S`$ state become feasible. If the
$`(\mu p)_{2S}`$ has kinetic energy below the $`2P`$ threshold
(laboratory kinetic energy $`T_0=0.3`$ eV), then Stark transitions
$`2S2P`$ are energetically forbidden^11^1Some quenching will, however,
occur because $`2S2P`$ mixing during collisions allows radiative
transitions to the $`1S`$ state (See Refs. ).. The metastable fraction
of $`(\mu p)_{2S}`$ in hydrogen depends on the kinetic energy at the
time of formation.

The first estimate of the $`(\mu p)_{2S}`$ lifetime was done by Kodosky
and Leon . They calculated the inelastic $`2S2P`$ cross section in a
semiclassical framework and concluded that the $`2S`$ state for
$`T>T_0`$ will be rapidly depopulated except for very small target
densities. However, this model did not consider deceleration due to
elastic $`2S2S`$ scattering. A more elaborate approach was developed by
Carboni and Fiorentini . They calculated both elastic $`2S2S`$ and
inelastic $`2S2P`$ cross sections quantum mechanically and estimated the
probability for a $`(\mu p)_{2S}`$ atom to slow down below threshold
from a given initial energy. The results of their calculations show that
a sizeable fraction of $`(\mu p)_{2S}`$ formed at kinetic energies less
than 1.3 eV can slow down below the $`2P`$ threshold.

The metastable fraction of $`(\mu p)_{2S}`$ per stopped muon can in
principle be calculated in a cascade model which takes the different
processes (Stark mixing, radiative decays, etc.) into account . However,
if one knows the fraction of stopped muons which reaches the $`2S`$
state (regardless of energy) and the kinetic energy distribution on
arrival in this state, then it is sufficient to treat the final part of
the cascade ($`n=1,2`$). This information can be obtained from
experiments. The fraction of stopped muons which arrives in the $`2S`$
state can be determined from the radiative yields : it was found in Ref.
that between 2% and 7% of the $`\mu p`$ reach the $`2S`$ state in the
pressure range $`0.33800`$ hPa. The kinetic energy distribution for
$`\mu p`$ in the $`1S`$ state, which for low pressures is expected to be
very similar to that of the $`2S`$ state just after arrival, can be
obtained from diffusion experiments . The median energy is found to be
about 1.5 eV for a target pressure of 0.25 hPa .

The purpose of this paper is to calculate the fraction of $`\mu p`$ in
the $`2S`$ state which reaches kinetic energies below the $`2P`$
threshold as a function of the initial kinetic energy $`T`$. We will
also present a fully quantum mechanical calculation of $`\mu p+`$H
differential cross sections which are used in our Monte Carlo simulation
of the kinetics.

The paper is organized as follows. The theoretical framework of the
quantum mechanical calculation of the cross sections is outlined in
Section 2. The calculated cross sections are discussed in Section 3.
Section 4 presents the calculations of the metastable $`2S`$ fraction.
The summary of the results is given in Section 5.

Unless otherwise stated, atomic units ($`\mathrm{}=a_0=m_e=1`$) are used
throughout this paper. The unit of cross section is
$`a_0^2=2.810^{17}\mathrm{cm}^2`$.

2 Quantum Mechanical Approach to the Calculation of the Cross Sections
$`(\mu p)_{nl}+\mathrm{H}(\mu p)_{nl^{}}+\mathrm{H}`$
-----------------------------------------------------

For the benefit of the reader, we briefly describe the quantum
mechanical calculation of $`\mu p+`$H scattering in the coupled-channel
approximation. The three-body wave function $`\psi (𝒓,𝑹)`$ where the
coordinates $`𝒓`$ and $`𝑹`$ are defined in Fig. 1 satisfies the
Schrödinger equation

$$H\psi (𝒓,𝑹)=E\psi (𝒓,𝑹)$$ (1)\
where the Hamiltonian is given by

$$H=\frac{^2}{2\mu }+H_{\mu p}+V(𝒓,𝑹).$$ (2)\
Here $`\mu =m_pm_{\mu p}/(m_p+m_{\mu p})`$ is the reduced mass of the
$`p\mu p`$ system, with $`m_p`$ being the proton mass and $`m_{\mu p}`$
the total $`\mu p`$ mass. The two-body Hamiltonian of the $`\mu p`$
atom, $`H_{\mu p}`$, includes the Coulomb interaction and a term that
describes the shift of the $`nS`$ state (mainly because of the vacuum
polarization) with respect to the states with $`l>0`$. For the case
$`n=2`$ considered below, the $`2S`$ state is lower than the $`2P`$ by
$`\mathrm{\Delta }E=0.21`$ eV. The much smaller fine and hyperfine
structure splitting is neglected. The potential $`V(𝒓,𝑹)`$ describes the
interaction of the $`\mu p`$ system with the target proton^22^2For the
sake of simplicity, we ignore the fact that the protons are identical
particles.:

$$V(𝒓,𝑹)=\frac{1}{|𝑹ϵ𝒓|}\frac{1}{|𝑹+(1ϵ)𝒓|}$$ (3)\
where $`ϵ=m_\mu /m_{\mu p}=0.101`$.

Equation (1) is solved in the coupled-channel approximation by using a
finite number of basis functions to describe the state of the $`\mu p`$.
For the problem of $`nlmnl^{}m^{}`$ scattering considered in this paper,
the set of $`n^2`$ eigenstates with principal quantum number $`n`$ has
been selected but the basis can be extended in a straightforward manner.
With $`n`$ fixed, let $`\chi _{lm}(𝒓)`$ denote the normalized
eigenfunctions of the atomic Hamiltonian $`H_{\mu p}`$ with the energy
$`E_{nl}`$, the square of the $`\mu p`$ internal angular momentum
$`𝐥^2`$ (eigenvalue $`l(l+1)`$) and its projection along the $`z`$-axis
$`l_z`$ (eigenvalue $`m`$). The total wave function $`\psi (𝒓,𝑹)`$ is
expanded as follows

$$\psi (𝒓,𝑹)=R^1\underset{JMLl}{}\xi _{JLl}(R)𝒴_{Ll}^{JM}(𝛀,𝒓)$$ (4)\
where

$$𝒴_{Ll}^{JM}(𝛀,𝒓)=\underset{M_Lm}{}LlM_Lm|JMY_{LM_L}(𝛀)\chi
_{lm}(𝒓),𝛀=𝑹/R.$$ (5)\
are simultaneous eigenfunctions of $`𝐉^2`$, $`𝐋^2`$, $`𝐥^2`$ and $`J_z`$
with eigenvalues $`J(J+1)`$, $`L(L+1)`$, $`l(l+1)`$ and $`M`$,
respectively. Here L is the $`p\mu p`$ relative angular momentum,
$`𝐉=𝐋+𝐥`$ is the total orbital angular momentum of the system. For a
given value of $`J`$ the system of radial Schrödinger equations has the
form

$$\left(\frac{1}{2\mu }\frac{d^2}{dR^2}+\frac{L(L+1)}{2\mu
R^2}+E_{nl}E\right)\xi
_{JLl}(R)+\underset{L^{}l^{}}{}L^{}l^{}JM|V|LlJM\xi _{JL^{}l^{}}(R)=0$$
(6)\
where the potential matrix elements are calculated in the basis (5):

$$𝛀,𝒓|LlJM=𝒴_{Ll}^{JM}(𝛀,𝒓).$$ (7)\
The matrix elements of the potential (3) have been calculated
analytically; the corresponding formulas are rather lengthy and will be
given elsewhere. From the asymptotic form of the solution of the $`n^2`$
coupled^33^3 Because of parity conservation the equations decouple into
two sets of respectively $`n(n+1)/2`$ and $`n(n1)/2`$ coupled equations.
equations (6), the scattering matrix $`S`$ is extracted and cross
sections can be calculated using standard formulas. The scattering
amplitude for $`nlmnl^{}m^{}`$ is given by

$$f_{nlmnl^{}m^{}}(𝛀)=\frac{4\pi
}{2i\sqrt{k^{}k}}\underset{L^{}LM_L^{}}{}i^{LL^{}}Y_{L^{}M_L^{}}(𝛀)L^{}l^{}M_L^{}m^{}|S1|Ll0mY_{L0}^{}(0)$$
(8)\
where

$$L^{}l^{}M_L^{}m^{}|S|Ll0m=\underset{JM}{}L^{}l^{}M_L^{}m^{}|JMJM|Ll0mL^{}l^{}J|S|LlJ.$$
(9)\
As a consequence of rotational symmetry, the matrix elements
$`L^{}l^{}J|S|LlJ`$ do not depend on the quantum number $`M`$. The
differential cross sections for the transitions $`nlnl^{}`$ are given by

$$\frac{d\sigma _{nlnl^{}}}{d\mathrm{\Omega
}}=\frac{1}{(2l+1)}\underset{m^{}m}{}\frac{k^{}}{k}|f_{nlmnl^{}m^{}}|^2$$
(10)\
where $`k`$ and $`k^{}`$ are the magnitudes of the relative momenta in
the initial and final state, correspondingly.

The total cross sections of the transitions $`nlnl^{}`$ have the form

$$\sigma _{nlnl^{}}=\frac{1}{(2l+1)}\frac{\pi
}{k^2}\underset{J}{}(2J+1)\underset{LL^{}}{}|L^{}l^{}J|S1|LlJ|^2$$ (11)\
and the corresponding transport cross sections are given by

$$\sigma _{nlnl^{}}^{tr}=𝑑\mathrm{\Omega }(1\mathrm{cos}\theta
)\frac{d\sigma _{nlnl^{}}}{d\mathrm{\Omega }}.$$ (12)\
In order to treat the long distance behaviour of the $`\mu p+`$H
interaction properly, the effect of electron screening must be taken
into account. This is done by multiplying the matrix elements
$`L^{}l^{}JM|V|LlJM`$ in Eq. (6) by the screening factor

$$F(R)=(1+2R+2R^2)e^{2R}$$ (13)\
which corresponds to the assumption that the electron of the hydrogen
atom remains unaffected in the $`1S`$ state during the collision.

For $`p\mu p`$ separations $`R`$ smaller than a few units of the
$`\mu p`$ Bohr radius, $`a_\mu =0.0054`$, our model cannot be expected
to be valid because the truncated set of basis functions in Eq. (4) is
not sufficient to describe the total three-body wave function
$`\psi (𝒓,𝑹)`$. Furthermore, exchange symmetry between the two protons
must be taken into account. We can estimate the sensitivity of our
results to the short range part of the interaction by using the dipole
approximation for the potential (3). The interaction in the dipole
approximation is given by the first nonzero term in the expansion of Eq.
(3) in inverse powers of $`R`$:

$$V_{DA}(𝒓,𝑹)=\frac{𝒓𝑹}{R^3}=\frac{z}{R^2}.$$ (14)\
A certain problem arises in the dipole approximation for a few low
partial waves ($`J5`$): the Schrödinger equation becomes ill defined
because of the attractive $`1/R^2`$ singularity^44^4This is a problem
only in the dipole approximation. The exact matrix elements are all
finite for $`R=0`$.. Following Ref. we cure this difficulty by placing
an infinitely repulsive sphere of radius $`r_{\mathrm{min}}`$ around the
target proton. The sensitivity of the results to this cutoff parameter
$`r_{\mathrm{min}}`$ will be used below as an estimate of the importance
of detailed description of the interaction at short distances.

In this paper we are interested in the $`2l2l^{}`$ transitions, and only
four states $`n=2`$ are used to describe the $`\mu p`$ part of the total
wave function. The four coupled second order equations (6) are solved
numerically for $`J=0,1,\mathrm{},J_{\mathrm{max}}`$ where the highest
partial wave $`J_{\mathrm{max}}`$ is chosen large enough to ensure the
convergence of the partial wave expansion at given collision energy.

Until now we have considered the $`\mu p`$ collisions with the atomic
target. Treating the collisions with hydrogen molecules is a formidable
task (even for $`\mu p`$ in the ground state ) which we do not attempt
here. The inelastic threshold $`T_0`$ for $`2S2P`$ transitions is 0.44
eV for atomic target and 0.33 eV for a molecular target. To get the
correct threshold value for the inelastic cross sections one can
substitute the atomic hydrogen mass with the molecular one. By varying
$`r_{\mathrm{min}}`$ and the target mass one can obtain an estimate of
the theoretical uncertainty of our approach.

The present model for calculating cross sections for $`(\mu p)_{n=2}`$H
scattering is a straightforward extension of the one by Carboni and
Fiorentini . There are three major differences: we solve the four
coupled differential equations exactly while Ref. treated non adiabatic
terms as a perturbation. The second difference is that we include the
full angular coupling while Ref. omitted some minor terms. Finally, we
use exact matrix elements for the $`\mu pp`$ interaction while Ref.
considers only the dipole approximation. These approximations were
justified by Ref. as follows: for small kinetic energies the velocity of
the muonic hydrogen atom is so low that the motion can be regarded as
nearly adiabatic. The angular coupling terms that were omitted in Ref.
are of the order 1 which is much smaller than the remaining angular
coupling terms of the order of $`J(J+1)`$ (the angular momenta as high
as $`J15`$ contribute to the $`2S2P`$ cross section at $`T=1`$ eV). The
electric field from a hydrogen atom is strong enough to induce Stark
transitions in the $`\mu p`$ for the distances $`Ra_0`$. Therefore, the
regions where the dipole approximation is valid ($`Ra_\mu `$) are
supposed to be most important. The diffusion experiments have shown that
a sizeable fraction of $`(\mu p)_{n=2}`$ atoms has kinetic energies of
several eV. In this high energy region the non adiabatic couplings
become strong and the model of Ref. can not be expected to give accurate
results.

The problem of $`\mu p+`$H scattering has been treated fully quantum
mechanically in Refs. . However, these calculations did not include the
$`2S2P`$ energy splitting and the question of the metastability of
$`(\mu p)_{2S}`$ was not addressed. Stark mixing has been studied in the
semiclassical straight-line-trajectory approximation in Refs. . We have
calculated the $`2S2P`$ Stark mixing cross sections in the semiclassical
approach in order to compare with the quantum mechanical results. A more
detailed comparison between semiclassical and quantum mechanical
calculations of $`\mu p+\mathrm{H}`$ scattering will be given elsewhere.

3 The Cross Sections of $`(\mu p)_{2l}`$ Scattering from Hydrogen
-----------------------------------------------------------------

Using the method described in Section 2 the S-matrix has been calculated
for the laboratory kinetic energy range $`T_0<T<6`$ eV. Unless otherwise
explicitly stated, the results shown are obtained with the exact
potential (3). Electron screening is always taken into account. Both
atomic and molecular mass of the target
($`M_{\mathrm{target}}=M_\mathrm{H}`$, $`M_{\mathrm{H}_2}`$) have been
used.

Figure 2 shows the $`2S2S`$ transport cross section and the $`2S2P`$
Stark mixing cross section in comparison with the results from Ref. (the
molecular mass is used in both cases). There is a good agreement for the
Stark mixing $`2S2P`$ cross section below 1.7 eV. For the $`2S2S`$
transport cross section, the agreement is fair, with the discrepancy
being typically less than 30%.

In order to estimate the theoretical uncertainty of our approach, we
calculated the cross sections in the dipole approximation with short
distance cutoff $`0.01r_{\mathrm{min}}0.05`$ for
$`M_{\mathrm{target}}=M_\mathrm{H}`$ and
$`M_{\mathrm{target}}=M_{\mathrm{H}_2}`$. For fixed value of
$`M_{\mathrm{target}}`$, the cross sections for the three reactions
$`2S2P`$, $`2P2S`$ and $`2P2P`$ are weakly dependent on
$`r_{\mathrm{min}}`$. This shows that these reactions are dominated by
the long range part of the interaction $`V(𝒓,𝑹)`$. The only process
rather sensitive to the value of $`r_{\mathrm{min}}`$ is the elastic
scattering $`2S2S`$. This can be understood by considering the adiabatic
energy curves for low angular momentum. The energy curve which
corresponds asymptotically to the $`2S`$ state is attractive while those
corresponding to the three $`2P`$ states are repulsive. Therefore, in
the adiabatic approximation the $`2S2S`$ cross sections are expected to
depend on the short range part of the potential while this is not the
case for $`2P2P`$ scattering. At energies above 2 eV the semiclassical
approximation is in a good agreement with our quantum mechanical
results. However, this semiclassical approximation does not treat the
threshold behaviour correctly.

A more detailed comparison of the approximations used can be done by
plotting the $`J`$ dependence of the partial wave cross sections
$`\sigma _J`$ at fixed energy as shown in Fig. 3. The quantum mechanical
results obtained for the exact potential and the dipole approximation
with the short range cutoff agree well for angular momentum $`J>5`$
while the lowest partial waves are sensitive to the short range
behaviour of the approximating potentials. The reason is that for
$`J>5`$ the centrifugal barrier is strong enough to prevent the
$`\mu p`$ from approaching close to the target proton. For $`J5`$ the
$`\mu p`$ can get very close to the proton and the use of a small number
of atomic orbitals is not sufficient --- a better description in this
region is needed in a true three--body framework. It is seen that a
substantial part of the $`2S2S`$ cross section comes from partial waves
with low $`J`$, so this result also explains why the uncertainty of the
elastic $`2S`$ cross section is larger than for the other reactions. The
semiclassical calculation can be compared with the partial wave cross
sections by using the relation between the impact parameter $`\rho `$,
the relative momentum $`k`$ and the angular momentum $`J`$

$$k\rho =J+1/2.$$ (15)\
For large $`J`$ (large impact parameter) there is a very good agreement
between the semiclassical contribution to the $`2S2P`$ cross section and
the quantum mechanical partial wave result. An example of the
differential cross sections for the reaction $`2S2S`$ given in Fig. 4
shows a characteristic pattern with a strong forward peak and a set of
maxima and minima, which is in qualitative agreement with Ref. where the
adiabatic approach was used.

4 The Surviving Fraction of the Metastable $`(\mu p)_{2S}`$ State
-----------------------------------------------------------------

The surviving metastable fraction $`f(T)`$ is defined as the probability
that the $`\mu p`$ atom in the $`2S`$ state with initial kinetic energy
$`T`$ reaches the energy below the $`2P`$ threshold by slowing down in
elastic collisions. Assuming that the rate of the radiative transition
$`2P1S`$, $`\lambda _{2P1S}=1.210^{11}\mathrm{s}^1`$, is much larger
than the Stark mixing rate^55^5With our result for the Stark mixing rate
$`2P2S`$ at 1 eV as a function of the target density $`N`$,
$`\lambda _{2P2S}=Nv\sigma _{2P2S}410^{12}(N/N_0)\mathrm{s}^1`$ where
$`N_0`$ is the liquid hydrogen density $`4.2510^{22}`$ atoms/cm^3^, the
range of validity is $`N0.03N_0`$., the surviving fraction $`f(T)`$ was
estimated in by the formula

$$f(T)=\mathrm{exp}\left(\frac{(m_{\mu
p}+M_{\mathrm{target}})^2}{2m_{\mu
p}M_{\mathrm{target}}}_{T_0}^T\frac{\sigma _{2S2P}(T^{})}{T^{}\sigma
_{2S2S}^{tr}(T^{})}𝑑T^{}\right)$$ (16)\
with $`M_{\mathrm{target}}=M_{\mathrm{H}_2}`$. It was found that a
sizeable fraction of $`(\mu p)_{2S}`$ atoms formed at kinetic energies
below 1.3 eV slows down below threshold.

Equation (16) is based on the approximation of continuous energy loss.
To provide a more realistic treatment of the evolution in kinetic energy
we use a Monte Carlo program based on the differential cross sections
for the four processes $`2S2S`$, $`2S2P`$, $`2P2S`$ and $`2P2P`$. In
addition to the collisional processes, the $`2P1S`$ radiative transition
is also included in the code. The fate of a $`\mu p`$ formed in the
$`2S`$ state with kinetic energy $`T`$ is thus either to undergo
$`2P1S`$ radiative transition after the Stark mixing $`2S2P`$ or to end
up in the $`2S`$ state with kinetic energy below the threshold with
probability $`f(T)`$. Figure 5 shows the rates,
$`\lambda _{2l2l^{}}=N_0v\sigma _{2l2l^{}}`$, for the collisional
transitions in liquid hydrogen in comparison with the radiative
deexcitation rate $`\lambda _{2P1S}`$. In liquid hydrogen the Stark
mixing rates are so large that the $`\mu p`$ states are expected to be
statistically populated for kinetic energies $`T2`$eV (where threshold
effects can be neglected).

Figure 6 shows the surviving fraction $`f(T)`$ calculated with the Monte
Carlo program for target density $`10^6<N/N_0<10^2`$. The approximation
(16) gives somewhat higher values for the survival probability than the
exact kinetics calculation at $`T<1.4`$eV. The Monte Carlo results at
high energies ($`T>1.5`$eV) are significantly larger than those obtained
from Eq. (16) where continuous energy loss is assumed. The reason is
that the backward scattering (see Fig. 4) with maximum possible energy
loss plays an important role in bringing the $`(\mu p)_{2S}`$ atoms
below the $`2P`$ threshold for higher energies.

In order to estimate the theoretical uncertainty of $`f(T)`$ we
performed the Monte Carlo calculation with the cross sections obtained
in the dipole approximation. In all cases the Monte Carlo calculations
are consistent with the results corresponding to the exact potential: at
2 eV the surviving fraction is in the range $`1520\%`$ for atomic target
mass and $`1016\%`$ for molecular target mass. The use of the target
mass $`M_\mathrm{H}`$ instead of $`M_{\mathrm{H}_2}`$ leads to somewhat
higher survival fractions because of a simple kinematical reason: the
loss of kinetic energy in a collision with the same angle in the CMS is
larger for the target of smaller mass and, furthermore, the inelastic
threshold is higher for the scattering from the atomic target. Merely
substituting the atomic mass with the molecular mass for the hydrogen
target does not account for the additional energy loss due to rotational
and vibrational excitations of H~2~. One would therefore expect that the
slowing down process is more efficient than this model suggests and the
survival probability calculated with molecular target mass is
underestimated. The opposite is true for calculations with atomic
target: here the transfer of kinetic energy from the $`\mu p`$ to the
individual hydrogen atoms is not restricted by molecular bindings. Thus
results with atomic target probably give somewhat optimistic results for
the surviving fraction.

5 Conclusion
------------

The main results of this paper can be summarized as follows. The
detailed Monte Carlo kinetics calculations predict the surviving
metastable fraction of the $`2S`$ state of $`\mu p`$ to be larger than
$`50\%`$ for the initial kinetic energy 1 eV in agreement with earlier
estimates . For higher initial kinetic energies, our result is
significantly larger than the earlier estimates: the surviving
metastable fraction for $`T=5`$ eV is about $`4\%`$. This effect is due
to a sizeable contribution of backward scattering in elastic collisions.

Our Monte Carlo calculations are based on the cross sections calculated
in the coupled-channel approximation. The main limitation of this method
for the problem concerned comes from the use of a small number of atomic
states to describe the $`\mu p`$ system and the neglect of the molecular
structure of the target. A more accurate treatment of the $`\mu pp`$
three body problem is needed in order to do reliable calculations for a
few lowest partial wave amplitudes. Our approach, however, is well
suited for the description of the collisions with the characteristic
scale of impact parameters of the order of $`a_0`$ which is exactly the
case for the problem involved. Therefore our results provide a
significantly improved basis for a better estimate of the metastable
$`(\mu p)_{2S}`$ fraction which is very important for the planned
Lamb--shift experiment at PSI .

Further details and more results concerning the scattering of the
$`\mu p`$ atoms in the excited states $`n2`$ will be published
elsewhere.

Acknowledgement
---------------

We thank P. Hauser, F. Kottmann, L. Simons, D. Taqqu, and R. Pohl for
fruitful and stimulating discussions and M.P. Locher for useful
comments.
