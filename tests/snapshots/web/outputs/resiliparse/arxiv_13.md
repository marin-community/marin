# Duality relations for 𝑀 coupled Potts models

(January 2000)  

## Abstract

We establish explicit duality transformations for systems of $`M`$ $`q`$-state Potts models coupled through their local energy density, generalising known results for $`M=1,2,3`$. The $`M`$-dimensional space of coupling constants contains a selfdual sub-manifold of dimension $`D_M=[M/2]`$. For the case $`M=4`$, the variation of the effective central charge along the selfdual surface is investigated by numerical transfer matrix techniques. Evidence is given for the existence of a family of critical points, corresponding to conformal field theories with an extended $`S_M`$ symmetry algebra.

For several decades, the $`q`$-state Potts model has been used to model ferromagnetic materials , and an impressive number of results are known about it, especially in two dimensions . More recently, its random-bond counterpart has attracted considerable attention , primarily because it permits one to study how quenched randomness coupling to the local energy density can modify the nature of a phase transition.

But despite the remarkable successes of conformal invariance applied to pure two-dimensional systems, the amount of analytical results on the random-bond Potts model is rather scarce. Usually the disorder is dealt with by introducing $`M`$ replicas of the original model, with mutual energy-energy interactions, and taking the limit $`M0`$. The price to be paid is however that the resulting system loses many of the properties (such as unitarity) that lie at the heart of conventional conformal field theory .

Very recently, an alternative approach was suggested by Dotsenko et al . These authors point out that the perturbative renormalisation group  (effectively an expansion around the Ising model in the small parameter $`\epsilon =q2`$) predicts the existence of a non-trivial infrared fixed point at interlayer coupling $`g_{}\epsilon /(M2)+𝒪(\epsilon ^2)`$, so that the regions $`M<2`$ and $`M>2`$ are somehow dual upon changing the sign of the coupling constant<sup>1</sup><sup>1</sup>1The case $`M=2`$ is special: For $`q=2`$ (the Ashkin-Teller model) the coupling presents a marginal perturbation, giving rise to a halfline of critical points along which the critical exponents vary continuously . On the other hand, for $`q>2`$ where the perturbation is relevant, the model is still integrable, but now presents a mass generation leading to non-critical behaviour .. More interestingly, for $`M=3`$ they identify the exact lattice realisation of a critical theory with exponents consistent with those of the perturbative treatment, and they conjecture that this generalises to any integer $`M3`$. Their proposal is then to study this class of coupled models, which are now unitary by definition, and only take the limit $`M0`$ once the exact expressions for the various critical exponents have been worked out. One could hope to attack this task by means of extended conformal field theory, thus combining the $`Z_q`$ symmetry of the spin variable by a non-abelian $`S_M`$ symmetry upon permuting the replicas.

Clearly, a first step in this direction is to identify the lattice models corresponding to this series of critical theories, parametrised by the integer $`M3`$. For $`M=3`$ this was achieved  by working out the duality relations for $`M`$ coupled Potts models on the square lattice, within the $`M`$-dimensional space of coupling constants giving rise to $`S_M`$ symmetric interactions amongst the lattice energy operators of the replicas. Studying numerically the variation of the effective central charge  along the resulting selfdual line, using a novel and very powerful transfer matrix technique, the critical point was unambiguously identified with one of the endpoints of that line.

Unfortunately it was hard to see how such duality relations could be extended to the case of general $`M`$. The calculations in Ref.  relied on a particular version  of the method of lattice Fourier transforms , already employed for $`M=2`$ two decades ago . Though perfectly adapted to the case of linear combinations of cosinoidal interactions within a single (vector) Potts model , this approach led to increasingly complicated algebra when several coupled models were considered. Moreover, it seemed impossible to recast the end results in a reasonably simple form for larger $`M`$.

In the present publication we wish to assess whether such a scenario of a unique critical point with an extended $`S_M`$ symmetry can indeed be expected to persist in the general case of $`M3`$ symmetrically coupled models. We explicitly work out the duality transformations for any $`M`$, and show that they can be stated in a very simple form \[Eq. (9)\] after redefining the coupling constants.

The lattice identification of the $`M=3`$ critical point in Ref.  crucially relied on the existence of a one-parameter selfdual manifold, permitting only two possible directions of the initial flow away from the decoupling fixed point. We find in general a richer structure with an $`[M/2]`$-dimensional selfdual manifold. Nonetheless, from a numerical study of the case $`M=4`$ we end up concluding that the uniqueness of the non-trivial fixed point can be expected to persist, since the decoupling fixed point acts as a saddlepoint of the effective central charge.

Consider then a system of $`M`$ identical planar lattices, stacked on top of one another. On each lattice site $`i`$, and for each layer $`\mu =1,2,\mathrm{},M`$, we define a Potts spin $`\sigma _i^{(\mu )}`$ that can be in any of $`q=2,3,\mathrm{}`$ distinct states. The layers interact by means of the reduced hamiltonian

$$=\underset{ij}{}_{ij},$$
(1)   
where $`ij`$ denotes the set of lattice edges, and an $`S_M`$ symmetric nearest-neighbour interaction is defined as

$$_{ij}=\underset{m=1}{\overset{M}{}}K_m\underset{\mu _1\mu _2\mathrm{}\mu _m}{\overset{}{}}\underset{l=1}{\overset{m}{}}\delta (\sigma _i^{(\mu _l)},\sigma _j^{(\mu _l)}).$$
(2)   
By definition the primed summation runs over the $`\left(\genfrac{}{}{0pt}{}{M}{m}\right)`$ terms for which the indices $`1\mu _lM`$ with $`l=1,2,\mathrm{},m`$ are all different, and $`\delta (x,y)=1`$ if $`x=y`$ and zero otherwise.

For $`M=1`$ the model thus defined reduces to the conventional Potts model, whilst for $`M=2`$ it is identical to the Ashkin-Teller like model considered in Ref. , where the Potts models of either layer are coupled through their local energy density. For $`M>2`$, additional multi-energy interactions between several layers have been added, since such interactions are generated by the duality transformations, as we shall soon see. However, from the point of view of conformal field theory these supplementary interactions are irrelevant in the continuum limit. The case $`M=3`$ was discussed in Ref. .

By means of a generalised Kasteleyn-Fortuin transformation  the local Boltzmann weights can be recast as

$$\mathrm{exp}(_{ij})=\underset{m=1}{\overset{M}{}}\underset{\mu _1\mu _2\mathrm{}\mu _m}{\overset{}{}}\left[1+\left(\mathrm{e}^{K_m}1\right)\underset{l=1}{\overset{m}{}}\delta (\sigma _i^{(\mu _l)},\sigma _j^{(\mu _l)})\right].$$
(3)   
In analogy with the case of $`M=1`$, the products can now be expanded so as to transform the original Potts model into its associated random cluster model. To this end we note that Eq. (3) can be rewritten in the form

$$\mathrm{exp}(_{ij})=b_0+\underset{m=1}{\overset{M}{}}b_m\underset{\mu _1\mu _2\mathrm{}\mu _m}{\overset{}{}}\underset{l=1}{\overset{m}{}}\delta (\sigma _i^{(\mu _l)},\sigma _j^{(\mu _l)}),$$
(4)   
defining the coefficients $`\{b_m\}_{m=0}^M`$. The latter can be related to the physical coupling constants $`\{K_m\}_{m=1}^M`$ by evaluating Eqs. (3) and (4) in the situation where precisely $`m`$ out of the $`M`$ distinct Kronecker $`\delta `$-functions are non-zero. Clearly, in this case Eq. (3) is equal to $`\mathrm{e}^{J_m}`$, where

$$J_m=\underset{k=1}{\overset{m}{}}\left(\genfrac{}{}{0pt}{}{m}{k}\right)K_k$$
(5)   
for $`m1`$, and we set $`J_0=K_0=0`$. On the other hand, we find from Eq. (4) that this must be equated to $`_{k=0}^m\left(\genfrac{}{}{0pt}{}{m}{k}\right)b_k`$. This set of $`M+1`$ equations can be solved for the $`b_k`$ by recursion, considering in turn the cases $`m=0,1,\mathrm{},M`$. After some algebra, the edge weights $`b_k`$ (for $`k0`$) are then found as

$$b_k=\underset{m=0}{\overset{k}{}}(1)^{m+k}\left(\genfrac{}{}{0pt}{}{k}{m}\right)\mathrm{e}^{J_m}.$$
(6)   
The partition function in the spin representation

$$Z=\underset{\{\sigma \}}{}\underset{ij}{}\mathrm{exp}(_{ij})$$
(7)   
can now be transformed into the random cluster representation as follows. First, insert Eq. (4) on the right-hand side of the above equation, and imagine expanding the product over the lattice edges $`ij`$. To each term in the resulting sum we associate an edge colouring $`𝒢`$ of the $`M`$-fold replicated lattice, where an edge $`(ij)`$ in layer $`m`$ is considered to be coloured (occupied) if the term contains the factor $`\delta (\sigma _i^{(m)},\sigma _j^{(m)})`$, and uncoloured (empty) if it does not. \[In this language, the couplings $`J_k`$ correspond to the local energy density summed over all possible permutations of precisely $`k`$ simultaneously coloured edges.\]

The summation over the spin variables $`\{\sigma \}`$ is now trivially performed, yielding a factor of $`q`$ for each connected component (cluster) in the colouring graph. Keeping track of the prefactors multiplying the $`\delta `$-functions, using Eq. (4), we conclude that

$$Z=\underset{𝒢}{}\underset{m=1}{\overset{M}{}}q^{C_m}b_m^{B_m},$$
(8)   
where $`C_m`$ is the number of clusters in the $`m`$th layer, and $`B_m`$ is the number of occurencies in $`𝒢`$ of a situation where precisely $`m`$ ($`0mM`$) edges placed on top of one another have been simultaneously coloured.

It is worth noticing that the random cluster description of the model has the advantage that $`q`$ only enters as a parameter. By analytic continuation one can thus give meaning to a non-integer number of states. The price to be paid is that the $`C_m`$ are, a priori, non-local quantities.

In terms of the edge variables $`b_m`$ the duality transformation of the partition function is easily worked out. For simplicity we shall assume that the couplings constants $`\{K_m\}`$ are identical between all nearest-neighbour pairs of spins, the generalisation to an arbitrary inhomogeneous distribution of couplings being trivial. By analogy with the case $`M=1`$, a given colouring configuration $`𝒢`$ is taken to be dual to a colouring configuration $`\stackrel{~}{𝒢}`$ of the dual lattice obtained by applying the following duality rule: Each coloured edge intersects an uncoloured dual edge, and vice versa. In particular, the demand that the configuration $`𝒢_{\mathrm{full}}`$ with all lattice edges coloured be dual to the configuration $`𝒢_{\mathrm{empty}}`$ with no coloured (dual) edge fixes the constant entering the duality transformation. Indeed, from Eq. (8), we find that $`𝒢_{\mathrm{full}}`$ has weight $`q^Mb_M^E`$, where $`E`$ is the total number of lattice edges, and $`𝒢_{\mathrm{empty}}`$ is weighted by $`q^{MF}\stackrel{~}{b}_0^E`$, where $`F`$ is the number of faces, including the exterior one. We thus seek for a duality transformation of the form $`q^{MF}\stackrel{~}{b}_0^EZ(\{b_m\})=q^Mb_M^E\stackrel{~}{Z}(\{\stackrel{~}{b}_m\})`$, where for any configuration $`𝒢`$ the edge weights must transform so as to keep the same relative weight between $`𝒢`$ and $`𝒢_{\mathrm{full}}`$ as between $`\stackrel{~}{𝒢}`$ and $`𝒢_{\mathrm{empty}}`$.

An arbitrary colouring configuration $`𝒢`$ entering Eq. (8) can be generated by applying a finite number of changes to $`𝒢_{\mathrm{full}}`$, in which an edge of weight $`b_M`$ is changed into an edge of weight $`b_m`$ for some $`m=0,1,\mathrm{},M1`$. By such a change, in general, a number $`kMm`$ of pivotal bonds are removed from the colouring graph, thus creating $`k`$ new clusters, and the weight relative to that of $`𝒢_{\mathrm{full}}`$ will change by $`q^kb_m/b_M`$. On the other hand, in the dual configuration $`\stackrel{~}{𝒢}`$ a number $`Mmk`$ of clusters will be lost, since each of the $`k`$ new clusters mentioned above will be accompanied by the formation of a loop in $`\stackrel{~}{𝒢}`$. The weight change relative to $`𝒢_{\mathrm{empty}}`$ therefore amounts to $`\stackrel{~}{b}_{Mm}/(\stackrel{~}{b}_0q^{Mmk})`$. Comparing these two changes we see that the factors of $`q^k`$ cancel nicely, and after a change of variables $`mMm`$ the duality transformation takes the simple form

$$\stackrel{~}{b}_m=\frac{q^mb_{Mm}}{b_M}\text{for}m=0,1,\mathrm{},M,$$
(9)   
the relation with $`m=0`$ being trivial.

Selfdual solutions can be found by imposing $`\stackrel{~}{b}_m=b_m`$. However, this gives rise to only $`\left[\frac{M+1}{2}\right]`$ independent equations

$$b_{Mm}=q^{M/2m}b_m\text{for}m=0,1,\mathrm{},\left[\frac{M1}{2}\right],$$
(10)   
and the $`M`$-dimensional parameter space $`\{b_m\}_{m=1}^M`$, or $`\{K_m\}_{m=1}^M`$, thus has a selfdual sub-manifold of dimension $`D_M=\left[\frac{M}{2}\right]`$. In particular, the ordinary Potts model ($`M=1`$) has a unique selfdual point, whilst for $`M=2`$  and $`M=3`$  one has a line of selfdual solutions.

Our main result is constituted by Eqs. (5) and (6) relating the physical coupling constants $`\{K_m\}`$ to the edge weights $`\{b_m\}`$, in conjunction with Eqs. (9) and (10) giving the explicit (self)duality relations in terms of the latter.

Since the interaction energies entering Eq. (3) are invariant under a simultaneous shift of all Potts spins, an alternative way of establishing the duality transformations procedes by Fourier transformation of the energy gaps . This method was used in Refs.  and  to work out the cases $`M=2`$ and $`M=3`$ respectively. However, as $`M`$ increases this procedure very quickly becomes quite involved. To better appreciate the ease of the present approach, let us briefly pause to see how the parametrisations of the selfdual lines for $`M=2,3`$, expressed in terms of the couplings $`\{K_m\}`$, can be reproduced in a most expedient manner.

For $`M=2`$, Eq. (10) gives $`b_2=q`$, where from Eqs. (5) and (6) $`b_2=\mathrm{e}^{2K_1+K_2}2\mathrm{e}^{K_1}+1`$. Thus

$$\mathrm{e}^{K_2}=\frac{2\mathrm{e}^{K_1}+(q1)}{\mathrm{e}^{2K_1}},$$
(11)   
in accordance with Ref. . Similarly, for $`M=3`$ one has $`b_1=qb_2/b_3=b_2/\sqrt{q}`$ with $`b_1=\mathrm{e}^{K_1}1`$, $`b_2`$ as before, and $`b_3=\mathrm{e}^{3K_1+3K_2+K_3}3\mathrm{e}^{2K_1+K_2}+3\mathrm{e}^{K_1}1`$. This immediately leads to the result given in Ref. :

$`\mathrm{e}^{K_2}`$ $`=`$ $`{\displaystyle \frac{(2+\sqrt{q})\mathrm{e}^{K_1}(1+\sqrt{q})}{\mathrm{e}^{2K_1}}},`$ (12)   
$`\mathrm{e}^{K_3}`$ $`=`$ $`{\displaystyle \frac{3(\mathrm{e}^{K_1}1)(1+\sqrt{q})+q^{3/2}+1}{\left[(2+\sqrt{q})\mathrm{e}^{K_1}(1+\sqrt{q})\right]^3}}\mathrm{e}^{3K_1}.`$   
Returning now to the general case, we notice that the selfdual manifold always contains two special points for which the behaviour of the $`M`$ coupled models can be related to that of a single Potts model. At the first such point,

$$b_m=q^{m/2}\text{for}m=0,1,\mathrm{},\left[\frac{M}{2}\right],$$
(13)   
one has $`K_1=\mathrm{log}(1+\sqrt{q})`$ and $`K_m=0`$ for $`m=2,3,\mathrm{},M`$, whence the $`M`$ models simply decouple. The other point

$$b_m=\delta (m,0)\text{for}m=0,1,\mathrm{},\left[\frac{M}{2}\right]$$
(14)   
corresponds to $`K_m=0`$ for $`m=1,2,\mathrm{},M1`$ and $`K_M=\mathrm{log}(1+q^{M/2})`$, whence the resulting model is equivalent to a single $`q^M`$-state Potts model. Evidently, for $`M=1`$ these two special points coincide.

Specialising now to the case of a regular two-dimensional lattice, it is well-known that at the two special points the model undergoes a phase transition, which is continuous if the effective number of states ($`q`$ or $`q^M`$ as the case may be) is $`4`$ . In Ref.  the question was raised whether one in general can identify further non-trivial critical theories on the selfdual manifolds. In particular it was argued that for $`M=3`$ there is indeed such a point, supposedly corresponding to a conformal field theory with an extended $`S_3`$ symmetry.

To get an indication whether such results can be expected to generalise also to higher values of $`M`$, we have numerically computed the effective central charge of $`M=4`$ coupled models along the two-dimensional selfdual surface. We were able to diagonalise the transfer matrix for strips of width $`L=4,6,8`$ lattice constants in the equivalent loop model. Technical details of the simulations have been reported in Ref. . Relating the specific free energy $`f_0(L)`$ to the leading eigenvalue of the transfer matrix in the standard way, two estimates of the effective central charge, $`c(4,6)`$ and $`c(6,8)`$, were then obtained by fitting data for two consecutive strip widths according to 

$$f_0(L)=f_0(\mathrm{})\frac{\pi c}{6L^2}+\mathrm{}.$$
(15)   
A contour plot of $`c(6,8)`$, based on a grid of $`21\times 21`$ parameter values for $`(b_1,b_2)`$, is shown in Fig. 1. The data for $`c(4,6)`$ look qualitatively similar, but are less accurate due to finite-size effects. We should stress that even though the absolute values of $`c(6,8)`$ are some 4 % below what one would expect in the $`L\mathrm{}`$ limit, the variations in $`c`$ are supposed to be reproduced much more accurately . On the figure $`q=3`$, but other values of $`q`$ in the range $`2<q4`$ lead to similar results.

According to Zamolodchikov’s $`c`$-theorem , a system initially in the vicinity of the decoupled fixed point $`(b_1,b_2)=(\sqrt{q},q)`$, shown as an asterisk on the figure, will start flowing downhill in this central charge landscape. Fig. 1 very clearly indicates that the decoupled fixed point acts as a saddle point, and there are thus only two possibilities for the direction of the initial flow.

The first of these will take the system to the stable fixed point at the origin which trivially corresponds to one selfdual $`q^4`$-state Potts model. For $`q=3`$ this leads to the generation of a finite correlation length, consistent with $`c_{\mathrm{eff}}=0`$ in the limit of an infinitely large system. As expected, the flow starts out in the $`b_2`$ direction, meaning that it is the energy-energy coupling between layers ($`K_2`$) rather than the spin-spin coupling within each layer ($`K_1`$) that controls the initial flow.

More interestingly, if the system is started out in the opposite dirrection (i.e., with $`K_2`$ slightly positive) it will flow towards a third non-trivial fixed point, for which the edge weights tend to infinity in some definite ratios. \[Exactly what these ratios are is difficult to estimate, given that the asymptotic flow direction exhibits finite-size effects.\] Seemingly, at this point the central charge is only slightly lower than at the decoupled fixed point, as predicted by the perturbative renormalisation group . From the numerical data we would estimate the drop in the central charge as roughly $`\mathrm{\Delta }c=0.01`$$`0.02`$, in good agreement with the perturbative treatment which predicts $`\mathrm{\Delta }c=0.0168+𝒪(\epsilon ^5)`$ .

All of these facts are in agreement with the conjectures put forward in Ref. , and in particular one would think that this third fixed point corresponds to a conformal field theory with a non-abelian extended $`S_4`$ symmetry.

Finally, the numerics for $`q=2`$ (four coupled Ising models) is less conclusive, and we cannot rule out the possibility of a more involved fixed point structure. In particular, a $`c=2`$ theory is not only obtainable by decoupling the four models, but also by a pairwise coupling into two mutually decoupled four-state Potts (or Ashkin-Teller) models. Indeed, a similar phenomenon has already been observed for the case of three coupled Ising models .

Acknowledgments

The author is indebted to M. Picco for some very useful discussions.