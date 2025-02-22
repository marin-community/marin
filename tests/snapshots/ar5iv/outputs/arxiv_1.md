# References

Osterwalder–Schrader axioms—Wightman Axioms—The mathematical axiom systems for quantum field theory (QFT) grew out of Hilbert’s sixth problem , that of stating the problems of quantum theory in precise mathematical terms. There have been several competing mathematical systems of axioms, and here we shall deal with those of A.S. Wightman  and of K. Osterwalder and R. Schrader , stated in historical order. They are centered around group symmetry, relative to unitary representations of Lie groups in Hilbert space. We also mention how the Osterwalder–Schrader axioms have influenced the theory of unitary representations of groups, making connection with . Wightman’s axioms involve: (1) a unitary representation $`U`$ of $`G:=\mathrm{SL}(2,)^4`$ as a cover of the Poincaré group of relativity, and a vacuum state vector $`\psi _0`$ fixed by the representation, (2) quantum fields $`\phi _1\left(f\right),\mathrm{},\phi _n\left(f\right)`$, say, as operator-valued distributions, $`f`$ running over a specified space of test functions, and the operators $`\phi _i\left(f\right)`$ defined on a dense and invariant domain $`D`$ in $`𝐇`$ (the Hilbert space of quantum states), and $`\psi _0D`$, (3) a transformation law which states that $`U\left(g\right)\phi _j\left(f\right)U\left(g^1\right)`$ is a finite-dimensional representation $`R`$ of the group $`G`$ acting on the fields $`\phi _i\left(f\right)`$, i.e., $`_iR_{ji}\left(g^1\right)\phi _i\left(g\left[f\right]\right)`$, $`g`$ acting on space-time and $`g\left[f\right]\left(x\right)=f\left(g^1x\right)`$, $`x^4`$. (4) The fields $`\phi _j\left(f\right)`$ are assumed to satisfy locality and one of the two canonical commutation relations of $`[A,B]_\pm =AB\pm BA`$, for fermions, resp., bosons; and (5) it is assumed that there is scattering with asymptotic completeness, in the sense $`𝐇=𝐇^{\text{in}}=𝐇^{\text{out}}`$.

The Wightman axioms were the basis for many of the spectacular developments in QFT in the seventies, see, e.g., , and the Osterwalder–Schrader axioms  came in response to the dictates of path space measures. The constructive approach involved some variant of the Feynman measure. But the latter has mathematical divergences that can be resolved with an analytic continuation so that the mathematically well-defined Wiener measure becomes instead the basis for the analysis. Two analytical continuations were suggested in this connection: in the mass-parameter, and in the time-parameter, i.e., $`t\sqrt{1}t`$. With the latter, the Newtonian quadratic form on space-time turns into the form of relativity, $`x_1^2+x_2^2+x_3^2t^2`$. We get a stochastic process $`𝐗_t`$: symmetric, i.e., $`𝐗_t𝐗_t`$; stationary, i.e., $`𝐗_{t+s}𝐗_s`$; and Osterwalder–Schrader positive, i.e., $`_\mathrm{\Omega }f_1𝐗_{t_1}f_2𝐗_{t_2}\mathrm{}f_n𝐗_{t_n}𝑑P0`$, $`f_1,\mathrm{},f_n`$ test functions, $`\mathrm{}<t_1t_2\mathrm{}t_n<\mathrm{}`$, and $`P`$ denoting a path space measure.

Specifically: If $`t/2<t_1t_2\mathrm{}t_n<t/2`$, then

(1) 
$$\begin{array}{c}\mathrm{\Omega }A_1e^{\left(t_2t_1\right)\widehat{H}}A_2e^{\left(t_3t_2\right)\widehat{H}}A_3\mathrm{}A_n\mathrm{\Omega }\hfill \\ \hfill =\underset{t\mathrm{}}{lim}\underset{k=1}{\overset{n}{}}A_k\left(q\left(t_k\right)\right)d\mu _t\left(q()\right).\end{array}$$

By Minlos’ theorem, there is a measure $`\mu `$ on $`𝒟^{}`$ such that

(2) 
$$\underset{t\mathrm{}}{lim}e^{iq\left(f\right)}d\mu _t\left(q\right)=e^{iq\left(f\right)}d\mu \left(q\right)=:S\left(f\right)$$

for all $`f𝒟`$. Since $`\mu `$ is a positive measure, we have

$$\underset{k}{}\underset{l}{}\overline{c}_kc_lS\left(f_k\overline{f}_l\right)0$$

for all $`c_1,\mathrm{},c_n`$, and all $`f_1,\mathrm{},f_n𝒟`$. When combining (1) and (2), we note that this limit-measure $`\mu `$ then accounts for the time-ordered $`n`$-point functions which occur on the left-hand side in formula (1). This observation is further used in the analysis of the stochastic process $`𝐗_t`$, $`𝐗_t\left(q\right)=q\left(t\right)`$. But, more importantly, it can be checked from the construction that we also have the following reflection positivity: Let $`\left(\theta f\right)\left(s\right):=f\left(s\right)`$, $`f𝒟`$, $`s`$, and set

$$𝒟_+=\{f𝒟f\text{real valued,}f\left(s\right)=0\text{for}s<0\}.$$

Then

$$\underset{k}{}\underset{l}{}\overline{c}_kc_lS\left(\theta \left(f_k\right)f_l\right)0$$

for all $`c_1,\mathrm{},c_n`$, and all $`f_1,\mathrm{},f_n𝒟_+`$, which is one version of Osterwalder–Schrader positivity.

Since the Killing form of Lie theory may serve as a finite-dimensional metric, the Osterwalder–Schrader idea  turned out also to have implications for the theory of unitary representations of Lie groups. In , the authors associate to Riemannian symmetric spaces $`G/K`$ of tube domain type, a duality between complementary series representations of $`G`$ on one side, and highest weight representations of a $`c`$-dual $`G^c`$ on the other side. The duality $`GG^c`$ involves analytic continuation, in a sense which generalizes $`t\sqrt{1}t`$, and the reflection positivity of the Osterwalder–Schrader axiom system. What results is a new Hilbert space where the new representation of $`G^c`$ is “physical” in the sense that there is positive energy and causality, the latter concept being defined from certain cones in the Lie algebra of $`G`$.

A unitary representation $`\pi `$ acting on a Hilbert space $`𝐇(\pi )`$ is said to be reflection symmetric if there is a unitary operator $`J:𝐇(\pi )𝐇(\pi )`$ such that

* $`J^2=\text{id}`$.
* $`J\pi (g)=\pi (\tau (g))J,gG`$,

where $`\tau \mathrm{Aut}\left(G\right)`$, $`\tau ^2=id`$, and $`H:=\{gG\tau \left(g\right)=g\}`$.

A closed convex cone $`C𝔮`$ is hyperbolic if $`C^o\mathrm{}`$ and if $`\mathrm{ad}X`$ is semisimple with real eigenvalues for every $`XC^o`$.

Assume the following for $`(G,\pi ,\tau ,J)`$:

* $`\pi `$ is reflection symmetric with reflection $`J`$.
* There is an $`H`$-invariant hyperbolic cone $`C𝔮`$ such that $`S(C)=H\mathrm{exp}C`$ is a closed semigroup and $`S(C)^o=H\mathrm{exp}C^o`$ is diffeomorphic to $`H\times C^o`$.
* There is a subspace $`0𝐊_0𝐇(\pi )`$ invariant under $`S(C)`$ satisfying the positivity condition

$$v\text{}v_J:=v\text{}J(v)0,v𝐊_0.$$

Assume that $`(\pi ,C,𝐇,J)`$ satisfies (PR1)–(PR3). Then the following hold:

* $`S(C)`$ acts via $`s\stackrel{~}{\pi }(s)`$ by contractions on $`𝐊`$ ($`=`$ the Hilbert space obtained by completion of $`𝐊_0`$ in the norm from (PR3)).
* Let $`G^c`$ be the simply connected Lie group with Lie algebra $`𝔤^c`$. Then there exists a unitary representation $`\stackrel{~}{\pi }^c`$ of $`G^c`$ such that $`d\stackrel{~}{\pi }^c(X)=d\stackrel{~}{\pi }(X)`$ for $`X𝔥`$ and $`id\stackrel{~}{\pi }^c(Y)=d\stackrel{~}{\pi }(iY)`$ for $`YC`$, where $`𝔥:=\{X𝔤\tau \left(X\right)=X\}`$.
* The representation $`\stackrel{~}{\pi }^c`$ is irreducible if and only if $`\stackrel{~}{\pi }`$ is irreducible.

Palle E.T. Jorgensen: jorgen@math.uiowa.edu

Gestur Ólafsson: olafsson@math.lsu.edu
