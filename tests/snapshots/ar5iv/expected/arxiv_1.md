# References

Osterwalder–Schrader axioms—Wightman Axioms—The mathematical axiom systems for quantum field theory (QFT) grew out of Hilbert’s sixth problem, that of stating the problems of quantum theory in precise mathematical terms. There have been several competing mathematical systems of axioms, and here we shall deal with those of A.S. Wightman and of K. Osterwalder and R. Schrader, stated in historical order. They are centered around group symmetry, relative to unitary representations of Lie groups in Hilbert space. We also mention how the Osterwalder–Schrader axioms have influenced the theory of unitary representations of groups, making connection with. Wightman’s axioms involve: (1) a unitary representation $U$ of $G:=\mathrm{SL}\left(2,\mathbb{C}\right)\rtimes\mathbb{R}^{4}$ as a cover of the Poincaré group of relativity, and a vacuum state vector $\psi\_{0}$ fixed by the representation, (2) quantum fields $\varphi\_{1}\left(f\right),\dots,\varphi\_{n}\left(f\right)$, say, as operator-valued distributions, $f$ running over a specified space of test functions, and the operators $\varphi\_{i}\left(f\right)$ defined on a dense and invariant domain $D$ in $\mathbf{H}$ (the Hilbert space of quantum states), and $\psi\_{0}\in D$, (3) a transformation law which states that $U\left(g\right)\varphi\_{j}\left(f\right)U\left(g^{-1}\right)$ is a finite-dimensional representation $R$ of the group $G$ acting on the fields $\varphi\_{i}\left(f\right)$, i.e., $\sum\_{i}R\_{ji}\left(g^{-1}\right)\varphi\_{i}\left(g\left\[f\right\]\right)$, $g$ acting on space-time and $g\left\[f\right\]\left(x\right)=f\left(g^{-1}x\right)$, $x\in\mathbb{R}^{4}$. (4) The fields $\varphi\_{j}\left(f\right)$ are assumed to satisfy locality and one of the two canonical commutation relations of $\left\[A,B\right\]\_{\pm}=AB\pm BA$, for fermions, resp., bosons; and (5) it is assumed that there is scattering with asymptotic completeness, in the sense $\mathbf{H}=\mathbf{H}^{\text{in}}=\mathbf{H}^{\text{out}}$.

The Wightman axioms were the basis for many of the spectacular developments in QFT in the seventies, see, e.g.,, and the Osterwalder–Schrader axioms came in response to the dictates of path space measures. The constructive approach involved some variant of the Feynman measure. But the latter has mathematical divergences that can be resolved with an analytic continuation so that the mathematically well-defined Wiener measure becomes instead the basis for the analysis. Two analytical continuations were suggested in this connection: in the mass-parameter, and in the time-parameter, i.e., $t\mapsto\sqrt{-1}t$. With the latter, the Newtonian quadratic form on space-time turns into the form of relativity, $x\_{1}^{2}+x\_{2}^{2}+x\_{3}^{2}-t^{2}$. We get a stochastic process $\mathbf{X}\_{t}$: symmetric, i.e., $\mathbf{X}\_{t}\sim\mathbf{X}\_{-t}$; stationary, i.e., $\mathbf{X}\_{t+s}\sim\mathbf{X}\_{s}$; and Osterwalder–Schrader positive, i.e., $\int\_{\Omega}f\_{1}\circ\mathbf{X}\_{t\_{1}}\\,f\_{2}\circ\mathbf{X}\_{t\_{2}}\\,\cdots\\,f\_{n}\circ\mathbf{X}\_{t\_{n}}\\,dP\geq 0$, $f\_{1},\dots,f\_{n}$ test functions, $-\infty\<t\_{1}\leq t\_{2}\leq\dots\leq t\_{n}\<\infty$, and $P$ denoting a path space measure.

Specifically: If $-t/2\<t\_{1}\leq t\_{2}\leq\dots\leq t\_{n}\<t/2$, then

(1) 

$$\left\langle\Omega\mathrel{\mathchoice{\vrule height=7.5pt,width=0.25pt,depth=2.5pt}{\vrule height=7.5pt,width=0.25pt,depth=2.5pt}{\vrule height=7.5pt,width=0.25pt,depth=2.5pt}{\vrule height=7.5pt,width=0.25pt,depth=2.5pt}}A\_{1}e^{-\left(t\_{2}-t\_{1}\right)\hat{H}}A\_{2}e^{-\left(t\_{3}-t\_{2}\right)\hat{H}}A\_{3}\cdots A\_{n}\Omega\right\rangle\\ =\lim\_{t\rightarrow\infty}\int\prod\_{k=1}^{n}A\_{k}\left(q\left(t\_{k}\right)\right)\\,d\mu\_{t}\left(q\left(\\,\cdot\\,\right)\right).$$  

By Minlos’ theorem, there is a measure $\mu$ on $\mathcal{D}^{\prime}$ such that

(2) 

$$\lim\_{t\rightarrow\infty}\int e^{iq\left(f\right)}\\,d\mu\_{t}\left(q\right)=\int e^{iq\left(f\right)}\\,d\mu\left(q\right)=:S\left(f\right)$$  

for all $f\in\mathcal{D}$. Since $\mu$ is a positive measure, we have

$$\sum\_{k}\sum\_{l}\bar{c}\_{k}c\_{l}S\left(f\_{k}-\bar{f}\_{l}\right)\geq 0$$  

for all $c\_{1},\dots,c\_{n}\in\mathbb{C}$, and all $f\_{1},\dots,f\_{n}\in\mathcal{D}$. When combining (1) and (2), we note that this limit-measure $\mu$ then accounts for the time-ordered $n$-point functions which occur on the left-hand side in formula (1). This observation is further used in the analysis of the stochastic process $\mathbf{X}\_{t}$, $\mathbf{X}\_{t}\left(q\right)=q\left(t\right)$. But, more importantly, it can be checked from the construction that we also have the following reflection positivity: Let $\left(\theta f\right)\left(s\right):=f\left(-s\right)$, $f\in\mathcal{D}$, $s\in\mathbb{R}$, and set

$$\mathcal{D}\_{+}=\left\\{f\in\mathcal{D}\mid f\text{ real valued, }f\left(s\right)=0\text{ for }s\<0\right\\}\\,.$$  

Then

$$\sum\_{k}\sum\_{l}\bar{c}\_{k}c\_{l}S\left(\theta\left(f\_{k}\right)-f\_{l}\right)\geq 0$$  

for all $c\_{1},\dots,c\_{n}\in\mathbb{C}$, and all $f\_{1},\dots,f\_{n}\in\mathcal{D}\_{+}$, which is one version of Osterwalder–Schrader positivity.

Since the Killing form of Lie theory may serve as a finite-dimensional metric, the Osterwalder–Schrader idea turned out also to have implications for the theory of unitary representations of Lie groups. In, the authors associate to Riemannian symmetric spaces $G/K$ of tube domain type, a duality between complementary series representations of $G$ on one side, and highest weight representations of a $c$-dual $G^{c}$ on the other side. The duality $G\leftrightarrow G^{c}$ involves analytic continuation, in a sense which generalizes $t\mapsto\sqrt{-1}t$, and the reflection positivity of the Osterwalder–Schrader axiom system. What results is a new Hilbert space where the new representation of $G^{c}$ is “physical” in the sense that there is positive energy and causality, the latter concept being defined from certain cones in the Lie algebra of $G$.

A unitary representation $\pi$ acting on a Hilbert space $\mathbf{H}(\pi)$ is said to be reflection symmetric if there is a unitary operator $J:\mathbf{H}(\pi)\rightarrow\mathbf{H}(\pi)$ such that

* ${\displaystyle J^{2}=\mbox{\rm id}}$.
* ${\displaystyle J\pi(g)=\pi(\tau(g))J\\,,\quad g\in G}$,

where $\tau\in\operatorname{Aut}\left(G\right)$, $\tau^{2}=\operatorname\*{id}$, and $H:=\left\\{g\in G\mid\tau\left(g\right)=g\right\\}$.

A closed convex cone $C\subset\mathfrak{q}$ is hyperbolic if $C^{o}\not=\emptyset$ and if $\operatorname{ad}X$ is semisimple with real eigenvalues for every $X\in C^{o}$.

Assume the following for $(G,\pi,\tau,J)$:

* $\pi$ is reflection symmetric with reflection $J$.
* There is an $H$-invariant hyperbolic cone $C\subset\mathfrak{q}$ such that $S(C)=H\exp C$ is a closed semigroup and $S(C)^{o}=H\exp C^{o}$ is diffeomorphic to $H\times C^{o}$.
* There is a subspace ${0}\not=\mathbf{K}\_{0}\subset\mathbf{H}(\pi)$ invariant under $S(C)$ satisfying the positivity condition

$$\left\langle v\mathrel{\mathchoice{\vrule height=7.5pt,width=0.25pt,depth=2.5pt}{\vrule height=7.5pt,width=0.25pt,depth=2.5pt}{\vrule height=7.5pt,width=0.25pt,depth=2.5pt}{\vrule height=7.5pt,width=0.25pt,depth=2.5pt}}v\right\rangle\_{J}:=\left\langle v\mathrel{\mathchoice{\vrule height=7.5pt,width=0.25pt,depth=2.5pt}{\vrule height=7.5pt,width=0.25pt,depth=2.5pt}{\vrule height=7.5pt,width=0.25pt,depth=2.5pt}{\vrule height=7.5pt,width=0.25pt,depth=2.5pt}}J(v)\right\rangle\geq 0,\quad\forall v\in\mathbf{K}\_{0}\\,.$$

Assume that $(\pi,C,\mathbf{H},J)$ satisfies (PR1)–(PR3). Then the following hold:

* $S(C)$ acts via $s\mapsto\tilde{\pi}(s)$ by contractions on $\mathbf{K}$ ($=$ the Hilbert space obtained by completion of $\mathbf{K}\_{0}$ in the norm from (PR3)).
* Let $G^{c}$ be the simply connected Lie group with Lie algebra $\mathfrak{g}^{c}$. Then there exists a unitary representation $\tilde{\pi}^{c}$ of $G^{c}$ such that $d\tilde{\pi}^{c}(X)=d\tilde{\pi}(X)$ for $X\in\mathfrak{h}$ and $i\\,d\tilde{\pi}^{c}(Y)=d\tilde{\pi}(iY)$ for $Y\in C$, where $\mathfrak{h}:=\left\\{X\in\mathfrak{g}\mid\tau\left(X\right)=X\right\\}$.
* The representation $\tilde{\pi}^{c}$ is irreducible if and only if $\tilde{\pi}$ is irreducible.

Palle E.T. Jorgensen: jorgen@math.uiowa.edu

Gestur Ólafsson: olafsson@math.lsu.edu
