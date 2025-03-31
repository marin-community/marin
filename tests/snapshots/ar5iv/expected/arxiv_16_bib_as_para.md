# On Peakon Solutions of the Shallow Water Equation 11footnote 1Keywords: solitons, peakons, billiards, shallow water equation, Hamiltonian systems

## 1 Introduction

Camassa and Holm described classes of $n$-soliton peaked weak solutions, or “peakons,” for an integrable (SW) equation

$$U\_{t}+3UU\_{x}=U\_{xxt}+2U\_{x}U\_{xx}+UU\_{xxx}-2\kappa U\_{x}\\,,$$  

(1.1)   
arising in the context of shallow water theory. Of particular interest is their description of peakon dynamics in terms of a system of completely integrable Hamiltonian equations for the locations of the “peaks” of the solution, the points at which its spatial derivative changes sign. (Peakons have discontinuities in the $x$-derivative but both one-sided derivatives exist and differ only by a sign. This makes peakons different from cuspons considered earlier in the literature.) In other words, each peakon solution can be associated with a mechanical system of moving particles. Calogero and Calogero and Francoise further extended the class of mechanical systems of this type.

For the KdV equation, the spectral parameter $\lambda$ appears linearly in the potential of the corresponding Schrödinger equation: $V=u-\lambda$ in the context of the inverse scattering transform (IST) method (see Ablowitz and Segur). In contrast, the equation (1.1), as well as $N$-component systems in general, were shown to be connected to the energy dependent Schrödinger operators with potentials with poles in the spectral parameter.

Alber et al. showed that the presence of a pole in the potential is essential in a special limiting procedure that allows for the formation of “billiard solutions”. By using algebraic-geometric methods, one finds that these billiard solutions are related to finite dimensional integrable dynamical systems with reflections. This provides a short-cut to the study of quasi-periodic and solitonic billiard solutions of nonlinear PDE’s. This method can be used for a number of equations including the shallow water equation (1.1), the Dym type equation, as well as $N$-component systems with poles and the equations in their hierarchies. More information on algebraic-geometric methods for integrable systems can be found in and on billiards in.

In this paper we consider singular limits of quasi-periodic solutions when the spectral curve becomes singular and its arithmetic genus drops to zero. The solutions are then expressed in terms of purely exponential $\tau$-functions and they describe the finite time interaction of 2 solitary peakons of the shallow water equation (1.1). Namely, we invert the equations obtained by using a new parameterization. First a profile of the 2-peakon solution is described by considering different parameterizations for the associated Jacobi inversion problem on three subintervals of the $X$-axis and by gluing these pieces of the profile together. The dynamics of such solutions is then described by combining these profiles with the dynamics of the peaks of the solution in the form developed earlier in Alber et al.. This concludes a derivation in the context of the algebraic geometric approach of the $n$-peakon ansatz which was used in the initial papers for obtaining Hamiltonian systems for peaks. More recently $n$-peakon waves were studied in and.

The problem of describing complex traveling wave and quasi-periodic solutions of the equation (1.1) can be reduced to solving finite-dimensional Hamiltonian systems on symmetric products of hyperelliptic curves. Namely, according to Alber et al, such solutions can be represented in the case of two-phase quasi-periodic solutions in the following form

$$U(x,t)=\mu\_{1}+\mu\_{2}-M,$$  

(1.2)   
where $M$ is a constant and the evolution of the variables $\mu\_{1}$ and $\mu\_{2}$ is given by the equations

$$\sum\_{i=1}^{2}\frac{\mu\_{i}^{k}\\,{\rm d}\mu\_{i}}{\pm\sqrt{R(\mu\_{i})}}=\left\\{\begin{array}\[\]{ll}{\rm d}t&\mbox{$k=1,$}\\ {\rm d}x&\mbox{$k=2.$}\end{array}\right.$$  

(1.3)   
Here $R(\mu)$ is a polynomial of degree 6 of the form $R(\mu)=\mu\prod\_{i=1}^{5}(\mu-m\_{i})$. The constant from (1.2) takes the form $M=1/2\sum m\_{i}$. Notice that (1.3) describes quasi-periodic motion on tori of genus 2. In the limit $m\_{1}\rightarrow 0$, the solution develops peaks. (For details see Alber and Fedorov.)

### Interaction of Two Peakons.

In the limit when $m\_{2}\rightarrow m\_{3}\rightarrow a\_{1}$ and $m\_{4}\rightarrow m\_{5}\rightarrow a\_{2}$, we have 2 solitary peakons interacting with each other. For this 2 peakon case, we derive the general form of a profile for a fixed $t$ ($t=t\_{0},dt=0$) and then see how this profile changes with time knowing how the peaks evolve. Notice that the limit depends on the choice of the branches of the square roots present in (1.3) meaning choosing a particular sign $l\_{j}$ in front of each root. The problem of finding the profile, after applying the above limits to (1.3) gives

$\displaystyle l\_{1}\frac{d\mu\_{1}}{\mu\_{1}(\mu\_{1}-a\_{1})}+l\_{2}\frac{d\mu\_{2}}{\mu\_{2}(\mu\_{2}-a\_{1})}$ $\displaystyle=$ $\displaystyle a\_{2}\frac{dX}{\mu\_{1}\mu\_{2}}=a\_{2}dY\\,$ (1.4)   
$\displaystyle l\_{1}\frac{d\mu\_{1}}{\mu\_{1}(\mu\_{1}-a\_{2})}+l\_{2}\frac{d\mu\_{2}}{\mu\_{2}(\mu\_{2}-a\_{2})}$ $\displaystyle=$ $\displaystyle a\_{1}\frac{dX}{\mu\_{1}\mu\_{2}}=a\_{1}dY$ (1.5)   
where $Y$ is a new variable. This is a new parameterization of the Jacobi inversion problem (1.3) which makes the existence of three different branches of the solution obvious. In general, we consider three different cases: $(l\_{1}=1,l\_{2}=1)$, $(l\_{1}=1,l\_{2}=-1)$ and $(l\_{1}=-1,l\_{2}=-1)$. In each case we integrate and invert the integrals to calculate the symmetric polynomial ($\mu\_{1}+\mu\_{2}$). After substituting these expressions into the trace formula (1.2) for the solution, this results in three different parts of the profile defined on different subintervals on the real line. The union of these subintervals gives the whole line. On the last step these three parts are glued together to obtain a wave profile with two peaks.

The new parameterization $dX=\mu\_{1}\mu\_{2}dY$ plays an important role in our approach. In what follows each $\mu\_{i}(Y)$ will be defined on the whole real $Y$ line. However, the transformation from $Y$ back to $X$ is not surjective so that $\mu\_{i}(X)$ is only defined on a segment of the real axis. This is why different branches are needed to construct a solution on the entire real $X$ line.

In the case ($l\_{1}=l\_{2}=1$), if we assume that there is always one $\mu$ variable between $a\_{1}$ and $a\_{2}$ and one between 0 and $a\_{1}$ and that initial conditions are chosen so that $0\<\mu\_{1}^{0}\<a\_{1}\<\mu\_{2}^{0}\<a\_{2}$, then we find that: $\mu\_{1}+\mu\_{2}=a\_{1}+a\_{2}-(m\_{1}+n\_{1})a\_{1}a\_{2}e^{X}.$ This solution is valid on the domain

$$X\<-\log(a\_{1}n\_{1}+a\_{2}m\_{1})=X\_{1}^{-},$$  

where $n\_{1},m\_{1}$ are constants depending on $\mu\_{1}^{0},\mu\_{2}^{0}$. At the point $X\_{1}^{-}$,

$$\mu\_{1}(X\_{1}^{-})=0,\hskip 56.9055pt\mu\_{2}(X\_{1}^{-})=\frac{a\_{2}^{2}m\_{1}+a\_{1}^{2}n\_{1}}{a\_{2}m\_{1}+a\_{1}n\_{1}}.$$  

Now we consider ($l\_{1}=-1,l\_{2}=1$). Here we find the following expression for the symmetric polynomial

$$\mu\_{1}+\mu\_{2}=a\_{1}+a\_{2}-\frac{(a\_{2}-a\_{1})e^{-X}+m\_{2}n\_{2}(a\_{2}-a\_{1})e^{X}}{m\_{2}+n\_{2}},$$  

which is only defined on the interval

$$\log\frac{n\_{2}a\_{1}+m\_{2}a\_{2}}{m\_{2}n\_{2}(a\_{2}-a\_{1})}\>X\>\log\frac{a\_{2}-a\_{1}}{m\_{2}a\_{1}+n\_{2}a\_{2}}=X\_{1}^{+}.$$  

$m\_{2},n\_{2}$ are constants which must be chosen so that both $\mu\_{1}$ and $\mu\_{2}$ are continuous at $X\_{1}^{-}$ and that the ends of the branches match up, that is so that $X\_{1}^{-}=X\_{1}^{+}$. These conditions are satisfied if

$\displaystyle m\_{2}$ $\displaystyle=$ $\displaystyle\frac{a\_{2}}{a\_{1}}(a\_{2}-a\_{1})m\_{1},\\,$ (1.6)   
$\displaystyle n\_{2}$ $\displaystyle=$ $\displaystyle\frac{a\_{1}}{a\_{2}}(a\_{2}-a\_{1})n\_{1}.\ $ (1.7)   
Continuing in this fashion we arrive at the final 3 branched profile for a fixed $t$,

$\displaystyle U$ $\displaystyle=$ $\displaystyle-(a\_{1}M+a\_{2}N)e^{X}\hskip 14.22636pt{\rm if}\hskip 5.69054ptX\<-\log(N+M)\\,$ (1.8)   
$\displaystyle U$ $\displaystyle=$ $\displaystyle-\frac{a\_{1}a\_{2}e^{-X}+M\\;N\\;e^{X}(a\_{2}-a\_{1})^{2}}{a\_{2}M+a\_{1}N}\\,$ (1.9)   
$\displaystyle{\rm if}$ $\displaystyle\hskip 14.22636pt-\log(N+M)\<X\<\log\frac{a\_{2}^{2}M+a\_{1}^{2}N}{(a\_{2}-a\_{1})^{2}\\;M\\;N}\\,$ (1.10)   
$\displaystyle U$ $\displaystyle=$ $\displaystyle-e^{-X}\frac{a\_{2}^{3}M+a\_{1}^{3}N}{M\\;N(a\_{2}-a\_{1})^{2}}\hskip 28.45274pt{\rm if}\hskip 5.69054ptX\>\log\frac{a\_{2}^{2}M+a\_{1}^{2}N}{(a\_{2}-a\_{1})^{2}\\;M\\;N},\ $ (1.11)   
where we have made the substitution $M=a\_{2}m\_{1}$ and $N=a\_{1}n\_{1}$ and used the trace formula (1.2).

Please place the first figure near here.

### Time evolution.

So far only a profile has been derived. Now we will include the time evolution of the peaks to find the general solution for the two peakon case. To do this we use functions $q\_{i}(t)$ for $i=1,2$ introduced in Alber et al.

$$\mu\_{i}(x=q\_{i}(t),t)=0,$$  

for all $t$ and $i=1,2$ which describe the evolution of the peaks. All peaks belong to a zero level set: $\mu\_{i}=0$. Here the $\mu$-coordinates, generalized elliptic coordinates, are used to describe the positions of the peaks. This yields a connection between $x$ and $t$ along trajectories of the peaks resulting in a system of equations for the $q\_{i}(t)$. The solutions of this system are given by

$\displaystyle q\_{1}(t)$ $\displaystyle=$ $\displaystyle q\_{1}^{0}-a\_{2}t-\log|1-C\_{1}e^{(a\_{1}-a\_{2})t}|+\log(1-C\_{1})\\,$ (1.12)   
$\displaystyle q\_{2}(t)$ $\displaystyle=$ $\displaystyle q\_{2}^{0}-a\_{2}t+\log|1-C\_{2}e^{(a\_{2}-a\_{1})t}|-\log(1-C\_{2}),$ (1.13)   
where $C\_{i}=(q\_{i}^{\prime}(0)-a\_{1})/(q\_{i}^{\prime}(0)-a\_{2})$.

The solution defined in (1.8) has the peaks given in terms of the parameters $N$ and $M$. So to obtain the solution in terms of both $x$ and $t$, these parameters must be considered as functions of time. The complete solution now has the form

$\displaystyle U$ $\displaystyle=$ $\displaystyle-(a\_{1}M(t)+a\_{2}N(t))e^{X}\\,\hskip 14.22636pt{\rm if}\hskip 5.69054ptX\<-\log(N(t)+M(t))\\,$ (1.14)   
$\displaystyle U$ $\displaystyle=$ $\displaystyle-\frac{a\_{1}a\_{2}e^{-X}+M(t)\\;N(t)\\;e^{X}(a\_{2}-a\_{1})^{2}}{a\_{2}M(t)+a\_{1}N(t)}\\,$   
$\displaystyle{\rm if}$ $\displaystyle-\log(N(t)+M(t))\<X\<\log\frac{a\_{2}^{2}M(t)+a\_{1}^{2}N(t)}{(a\_{2}-a\_{1})^{2}\\;M(t)\\;N(t)}\\,$ (1.15)   
$\displaystyle U$ $\displaystyle=$ $\displaystyle-e^{-X}\frac{a\_{2}^{3}M(t)+a\_{1}^{3}N(t)}{M(t)\\;N(t)(a\_{2}-a\_{1})^{2}}\\,\hskip 14.22636pt{\rm if}\hskip 5.69054ptX\>\log\frac{a\_{2}^{2}M(t)+a\_{1}^{2}N(t)}{(a\_{2}-a\_{1})^{2}\\;M(t)\\;N(t)}.\ $ (1.16)   
where the functions $M(t),N(t)$ are determined by the relations

$\displaystyle N(t)+M(t)$ $\displaystyle=$ $\displaystyle e^{-q\_{1}(t)}=\frac{e^{-q\_{1}^{0}}|e^{a\_{2}t}-C\_{1}e^{a\_{1}t}|}{1-C\_{1}}\\,$ (1.17)   
$\displaystyle\frac{a\_{2}^{2}M(t)+a\_{1}^{2}N(t)}{M(t)\\;N(t)}$ $\displaystyle=$ $\displaystyle(a\_{2}-a\_{1})^{2}e^{q\_{2}(t)}=\frac{(a\_{2}-a\_{1})^{2}e^{q\_{2}^{0}}|e^{-a\_{2}t}-C\_{2}e^{-a\_{1}t}|}{(1-C\_{2})},$ (1.18)   
where $q\_{1}(t),q\_{2}(t)$ are taken from (1.12)-(1.13). This system can be solved to find that

$\displaystyle M(t)$ $\displaystyle=$ $\displaystyle\frac{a\_{1}^{2}-a\_{2}^{2}+A(t)B(t)\pm\sqrt{(a\_{1}^{2}-a\_{2}^{2})^{2}-2A(t)B(t)(a\_{1}^{2}+a\_{2}^{2})+A(t)^{2}B(t)^{2}}}{2B(t)}\\,$ (1.19)   
$\displaystyle N(t)$ $\displaystyle=$ $\displaystyle A(t)-M(t),$ (1.20)   
where $A(t)=e^{-q\_{1}(t)}$ and $B(t)=(a\_{2}-a\_{1})^{2}e^{q\_{2}(t)}$. These functions contain 4 parameters, but in fact these can be reduced to two parameters by using the following relations

$\displaystyle q\_{1}(0)$ $\displaystyle=$ $\displaystyle-\log(M(0)+N(0))\hskip 56.9055ptq\_{1}^{\prime}(0)=\frac{a\_{2}M(0)+a\_{1}N(0)}{M(0)+N(0)}\\,$ (1.21)   
$\displaystyle q\_{2}(0)$ $\displaystyle=$ $\displaystyle\log\frac{a\_{2}^{2}M(0)+a\_{1}^{2}N(0)}{(a\_{2}-a\_{1})^{2}\\;M(0)\\;N(0)}\hskip 31.2982ptq\_{2}^{\prime}(0)=\frac{a\_{1}a\_{2}(a\_{2}M(0)+a\_{1}N(0))}{a\_{2}^{2}M(0)+a\_{1}^{2}N(0)}.\ $ (1.22)   
Some care must be used in choosing the sign in (1.19). It is clear that for large negative $t$, $\mu\_{1}(q\_{1}(t),t)$ refers to the path of one peakon while for large positive $t$ it refers to the other. If this were not the case, simple asymptotic analysis of (1.12) would show that the peakons change speed which is not the case. Therefore $q\_{1}(t)$ represents the path of one of the peakons until some time $t^{\*}$ and the other one after this time. The opposite is true for $q\_{2}(t)$. At the time $t^{\*}$ we say that a change of identity has taken place. $t^{\*}$ can be found explicitly by using the fact that at this time, the two peaks must have the same height. But the peaks have the same height exactly when

$$a\_{2}M(t^{\*})=a\_{1}N(t^{\*}).$$  

(1.23)   
Without loss of generality we can rescale time such that $t^{\*}=0$. In this case (1.23), due to the original definitions of $m\_{1},n\_{1}$ given in terms of $\mu\_{1}^{0}$ $\mu\_{2}^{0}$, corresponds to a restriction on the choice of $\mu\_{1}^{0}$ and $\mu\_{2}^{0}$, namely

$$-a\_{2}^{2}\frac{\mu\_{1}^{0}-a\_{1}}{\mu\_{1}^{0}-a\_{2}}=a\_{1}^{2}\frac{\mu\_{2}^{0}-a\_{2}}{\mu\_{2}^{0}-a\_{1}}.$$  

(1.24)   
This condition is satisfied for example when ${\displaystyle\mu\_{1}^{0}=\frac{a\_{1}a\_{2}}{a\_{1}+a\_{2}}}$ and ${\displaystyle\mu\_{2}^{0}=\frac{a\_{1}+a\_{2}}{2}}$. Also notice that under this rescaling, the phase shift is simply $q\_{1}(0)-q\_{2}(0)$.

Please place the second figure near here

So we now have a procedure to make the change of identity occur at $t=0$, i.e. $\mu\_{1}$ goes from representing the first peakon to the second one at $t=0$. This change is represented by the change in the sign of the plus/minus in (1.19). That is, the sign is chosen as positive for $t\<0$ and negative for $t\>0$. However, $M$ remains continuous despite this sign change since the change of identity occurs precisely when the term under the square root is zero. Therefore (1.14)-(1.16) and (1.19) together describe the solution $U(X,t)$ of the SW equation as a function of $x$ and $t$ depending on two parameters $M(0)$, $N(0)$.

By using the approach of this paper weak billiard solutions can be obtained for the whole class of $n$-peakon solutions of $N$-component systems.

## Bibliography.

R. Camassa and D. Holm, An integrable shallow water equation with peaked solitons, Phys. Rev. Lett. 71 1661-1664 (1993).

F. Calogero, An integrable Hamiltonian system, Phys. Lett. A. 201 306-310 (1995).

F. Calogero and J. Francoise, Solvable quantum version of an integrable Hamiltonian system, J. Math. Phys. 37 (6) 2863-2871 (1996).

M. Ablowitz and H. Segur, Solitons and the Inverse Scattering Transform, SIAM, Philadelphia (1981).

M. Alber, R. Camassa, D. Holm and J. Marsden, The geometry of peaked solitons and billiard solutions of a class of integrable PDE’s, Lett. Math. Phys. 32 137-151 (1994).

M. Alber, R. Camassa, D. Holm, and J. Marsden, On the link between umbilic geodesics and soliton solutions of nonlinear PDE’s, Proc. Roy. Soc 450 677-692 (1995).

M. Alber and Y. Fedorov, Wave Solutions of Evolution Equations and Hamiltonian Flows on Nonlinear Subvarieties of Generalized Jacobians, (subm.) (1999).

E. Belokolos, A. Bobenko, V. Enol’sii, A. Its, and V. Matveev, Algebro-Geometric Approach to Nonlinear Integrable Equations., Springer-Verlag, Berlin;New York (1994).

M. Alber, R. Camassa, Y. Fedorov, D. Holm, and J. Marsden, The geometry of new classes of weak billiard solutions of nonlinear PDE’s. (subm.) (1999).

M. Alber, R. Camassa, Y. Fedorov, D. Holm and J. Marsden, On Billiard Solutions of Nonlinear PDE’s, Phys. Lett. A (to appear) (1999).

Y. Fedorov, Classical integrable systems and billiards related to generalized Jacobians, Acta Appl. Math., 55 (3) 151–201 (1999).

R. Camassa, D. Holm, and J. Hyman, A new integrable shallow water equation, Adv. Appl. Mech., 31 1–33 (1994).

R. Beals, D. Sattinger, J. Szmigielski, Multipeakons and a theorem of Stieltjes, Inverse Problems, 15 L1–L4 (1999).

Y. Li and P. Olver, Convergence of solitary-wave solutions in a perturbed bi-Hamiltonian dynamical system, Discrete and continuous dynamical systems, 4, 159–191 (1998).
