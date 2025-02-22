# On Peakon Solutions of the Shallow Water Equation 11footnote 1Keywords: solitons, peakons, billiards, shallow water equation, Hamiltonian systems

## 1 Introduction

Camassa and Holm  described classes of $`n`$-soliton peaked weak solutions, or “peakons,” for an integrable (SW) equation

$$U_t+3UU_x=U_{xxt}+2U_xU_{xx}+UU_{xxx}2\kappa U_x,$$
(1.1)   
arising in the context of shallow water theory. Of particular interest is their description of peakon dynamics in terms of a system of completely integrable Hamiltonian equations for the locations of the “peaks” of the solution, the points at which its spatial derivative changes sign. (Peakons have discontinuities in the $`x`$-derivative but both one-sided derivatives exist and differ only by a sign. This makes peakons different from cuspons considered earlier in the literature.) In other words, each peakon solution can be associated with a mechanical system of moving particles. Calogero  and Calogero and Francoise  further extended the class of mechanical systems of this type.

For the KdV equation, the spectral parameter $`\lambda `$ appears linearly in the potential of the corresponding Schrödinger equation: $`V=u\lambda `$ in the context of the inverse scattering transform (IST) method (see Ablowitz and Segur ). In contrast, the equation (1.1), as well as $`N`$-component systems in general, were shown to be connected to the energy dependent Schrödinger operators with potentials with poles in the spectral parameter.

Alber et al.  showed that the presence of a pole in the potential is essential in a special limiting procedure that allows for the formation of “billiard solutions”. By using algebraic-geometric methods, one finds that these billiard solutions are related to finite dimensional integrable dynamical systems with reflections. This provides a short-cut to the study of quasi-periodic and solitonic billiard solutions of nonlinear PDE’s. This method can be used for a number of equations including the shallow water equation (1.1), the Dym type equation, as well as $`N`$-component systems with poles and the equations in their hierarchies . More information on algebraic-geometric methods for integrable systems can be found in  and on billiards in .

In this paper we consider singular limits of quasi-periodic solutions when the spectral curve becomes singular and its arithmetic genus drops to zero. The solutions are then expressed in terms of purely exponential $`\tau `$-functions and they describe the finite time interaction of 2 solitary peakons of the shallow water equation (1.1). Namely, we invert the equations obtained by using a new parameterization. First a profile of the 2-peakon solution is described by considering different parameterizations for the associated Jacobi inversion problem on three subintervals of the $`X`$-axis and by gluing these pieces of the profile together. The dynamics of such solutions is then described by combining these profiles with the dynamics of the peaks of the solution in the form developed earlier in Alber et al. . This concludes a derivation in the context of the algebraic geometric approach of the $`n`$-peakon ansatz which was used in the initial papers  for obtaining Hamiltonian systems for peaks. More recently $`n`$-peakon waves were studied in  and .

The problem of describing complex traveling wave and quasi-periodic solutions of the equation (1.1) can be reduced to solving finite-dimensional Hamiltonian systems on symmetric products of hyperelliptic curves. Namely, according to Alber et al , such solutions can be represented in the case of two-phase quasi-periodic solutions in the following form

$$U(x,t)=\mu _1+\mu _2M,$$
(1.2)   
where $`M`$ is a constant and the evolution of the variables $`\mu _1`$ and $`\mu _2`$ is given by the equations

$$\underset{i=1}{\overset{2}{}}\frac{\mu _i^k\mathrm{d}\mu _i}{\pm \sqrt{R(\mu _i)}}=\{\begin{array}{cc}\mathrm{d}t\hfill & k=1,\hfill \\ \mathrm{d}x\hfill & k=2.\hfill \end{array}$$
(1.3)   
Here $`R(\mu )`$ is a polynomial of degree 6 of the form $`R(\mu )=\mu _{i=1}^5(\mu m_i)`$. The constant from (1.2) takes the form $`M=1/2m_i`$. Notice that (1.3) describes quasi-periodic motion on tori of genus 2. In the limit $`m_10`$, the solution develops peaks. (For details see Alber and Fedorov .)

### Interaction of Two Peakons.

In the limit when $`m_2m_3a_1`$ and $`m_4m_5a_2`$, we have 2 solitary peakons interacting with each other. For this 2 peakon case, we derive the general form of a profile for a fixed $`t`$ ($`t=t_0,dt=0`$) and then see how this profile changes with time knowing how the peaks evolve. Notice that the limit depends on the choice of the branches of the square roots present in (1.3) meaning choosing a particular sign $`l_j`$ in front of each root. The problem of finding the profile, after applying the above limits to (1.3) gives

$`l_1{\displaystyle \frac{d\mu _1}{\mu _1(\mu _1a_1)}}+l_2{\displaystyle \frac{d\mu _2}{\mu _2(\mu _2a_1)}}`$ $`=`$ $`a_2{\displaystyle \frac{dX}{\mu _1\mu _2}}=a_2dY`$ (1.4)   
$`l_1{\displaystyle \frac{d\mu _1}{\mu _1(\mu _1a_2)}}+l_2{\displaystyle \frac{d\mu _2}{\mu _2(\mu _2a_2)}}`$ $`=`$ $`a_1{\displaystyle \frac{dX}{\mu _1\mu _2}}=a_1dY`$ (1.5)   
where $`Y`$ is a new variable. This is a new parameterization of the Jacobi inversion problem (1.3) which makes the existence of three different branches of the solution obvious. In general, we consider three different cases: $`(l_1=1,l_2=1)`$, $`(l_1=1,l_2=1)`$ and $`(l_1=1,l_2=1)`$. In each case we integrate and invert the integrals to calculate the symmetric polynomial ($`\mu _1+\mu _2`$). After substituting these expressions into the trace formula (1.2) for the solution, this results in three different parts of the profile defined on different subintervals on the real line. The union of these subintervals gives the whole line. On the last step these three parts are glued together to obtain a wave profile with two peaks.

The new parameterization $`dX=\mu _1\mu _2dY`$ plays an important role in our approach. In what follows each $`\mu _i(Y)`$ will be defined on the whole real $`Y`$ line. However, the transformation from $`Y`$ back to $`X`$ is not surjective so that $`\mu _i(X)`$ is only defined on a segment of the real axis. This is why different branches are needed to construct a solution on the entire real $`X`$ line.

In the case ($`l_1=l_2=1`$), if we assume that there is always one $`\mu `$ variable between $`a_1`$ and $`a_2`$ and one between 0 and $`a_1`$ and that initial conditions are chosen so that $`0<\mu _1^0<a_1<\mu _2^0<a_2`$, then we find that: $`\mu _1+\mu _2=a_1+a_2(m_1+n_1)a_1a_2e^X.`$ This solution is valid on the domain

$$X<\mathrm{log}(a_1n_1+a_2m_1)=X_1^{},$$

where $`n_1,m_1`$ are constants depending on $`\mu _1^0,\mu _2^0`$. At the point $`X_1^{}`$,

$$\mu _1(X_1^{})=0,\mu _2(X_1^{})=\frac{a_2^2m_1+a_1^2n_1}{a_2m_1+a_1n_1}.$$

Now we consider ($`l_1=1,l_2=1`$). Here we find the following expression for the symmetric polynomial

$$\mu _1+\mu _2=a_1+a_2\frac{(a_2a_1)e^X+m_2n_2(a_2a_1)e^X}{m_2+n_2},$$

which is only defined on the interval

$$\mathrm{log}\frac{n_2a_1+m_2a_2}{m_2n_2(a_2a_1)}>X>\mathrm{log}\frac{a_2a_1}{m_2a_1+n_2a_2}=X_1^+.$$

$`m_2,n_2`$ are constants which must be chosen so that both $`\mu _1`$ and $`\mu _2`$ are continuous at $`X_1^{}`$ and that the ends of the branches match up, that is so that $`X_1^{}=X_1^+`$. These conditions are satisfied if

$`m_2`$ $`=`$ $`{\displaystyle \frac{a_2}{a_1}}(a_2a_1)m_1,`$ (1.6)   
$`n_2`$ $`=`$ $`{\displaystyle \frac{a_1}{a_2}}(a_2a_1)n_1.`$ (1.7)   
Continuing in this fashion we arrive at the final 3 branched profile for a fixed $`t`$,

$`U`$ $`=`$ $`(a_1M+a_2N)e^X\mathrm{if}X<\mathrm{log}(N+M)`$ (1.8)   
$`U`$ $`=`$ $`{\displaystyle \frac{a_1a_2e^X+MNe^X(a_2a_1)^2}{a_2M+a_1N}}`$ (1.9)   
$`\mathrm{if}`$ $`\mathrm{log}(N+M)<X<\mathrm{log}{\displaystyle \frac{a_2^2M+a_1^2N}{(a_2a_1)^2MN}}`$ (1.10)   
$`U`$ $`=`$ $`e^X{\displaystyle \frac{a_2^3M+a_1^3N}{MN(a_2a_1)^2}}\mathrm{if}X>\mathrm{log}{\displaystyle \frac{a_2^2M+a_1^2N}{(a_2a_1)^2MN}},`$ (1.11)   
where we have made the substitution $`M=a_2m_1`$ and $`N=a_1n_1`$ and used the trace formula (1.2).

Please place the first figure near here.

### Time evolution.

So far only a profile has been derived. Now we will include the time evolution of the peaks to find the general solution for the two peakon case. To do this we use functions $`q_i(t)`$ for $`i=1,2`$ introduced in Alber et al. 

$$\mu _i(x=q_i(t),t)=0,$$

for all $`t`$ and $`i=1,2`$ which describe the evolution of the peaks. All peaks belong to a zero level set: $`\mu _i=0`$. Here the $`\mu `$-coordinates, generalized elliptic coordinates, are used to describe the positions of the peaks. This yields a connection between $`x`$ and $`t`$ along trajectories of the peaks resulting in a system of equations for the $`q_i(t)`$. The solutions of this system are given by

$`q_1(t)`$ $`=`$ $`q_1^0a_2t\mathrm{log}|1C_1e^{(a_1a_2)t}|+\mathrm{log}(1C_1)`$ (1.12)   
$`q_2(t)`$ $`=`$ $`q_2^0a_2t+\mathrm{log}|1C_2e^{(a_2a_1)t}|\mathrm{log}(1C_2),`$ (1.13)   
where $`C_i=(q_i^{}(0)a_1)/(q_i^{}(0)a_2)`$.

The solution defined in (1.8) has the peaks given in terms of the parameters $`N`$ and $`M`$. So to obtain the solution in terms of both $`x`$ and $`t`$, these parameters must be considered as functions of time. The complete solution now has the form

$`U`$ $`=`$ $`(a_1M(t)+a_2N(t))e^X\mathrm{if}X<\mathrm{log}(N(t)+M(t))`$ (1.14)   
$`U`$ $`=`$ $`{\displaystyle \frac{a_1a_2e^X+M(t)N(t)e^X(a_2a_1)^2}{a_2M(t)+a_1N(t)}}`$   
$`\mathrm{if}`$ $`\mathrm{log}(N(t)+M(t))<X<\mathrm{log}{\displaystyle \frac{a_2^2M(t)+a_1^2N(t)}{(a_2a_1)^2M(t)N(t)}}`$ (1.15)   
$`U`$ $`=`$ $`e^X{\displaystyle \frac{a_2^3M(t)+a_1^3N(t)}{M(t)N(t)(a_2a_1)^2}}\mathrm{if}X>\mathrm{log}{\displaystyle \frac{a_2^2M(t)+a_1^2N(t)}{(a_2a_1)^2M(t)N(t)}}.`$ (1.16)   
where the functions $`M(t),N(t)`$ are determined by the relations

$`N(t)+M(t)`$ $`=`$ $`e^{q_1(t)}={\displaystyle \frac{e^{q_1^0}|e^{a_2t}C_1e^{a_1t}|}{1C_1}}`$ (1.17)   
$`{\displaystyle \frac{a_2^2M(t)+a_1^2N(t)}{M(t)N(t)}}`$ $`=`$ $`(a_2a_1)^2e^{q_2(t)}={\displaystyle \frac{(a_2a_1)^2e^{q_2^0}|e^{a_2t}C_2e^{a_1t}|}{(1C_2)}},`$ (1.18)   
where $`q_1(t),q_2(t)`$ are taken from (1.12)-(1.13). This system can be solved to find that

$`M(t)`$ $`=`$ $`{\displaystyle \frac{a_1^2a_2^2+A(t)B(t)\pm \sqrt{(a_1^2a_2^2)^22A(t)B(t)(a_1^2+a_2^2)+A(t)^2B(t)^2}}{2B(t)}}`$ (1.19)   
$`N(t)`$ $`=`$ $`A(t)M(t),`$ (1.20)   
where $`A(t)=e^{q_1(t)}`$ and $`B(t)=(a_2a_1)^2e^{q_2(t)}`$. These functions contain 4 parameters, but in fact these can be reduced to two parameters by using the following relations

$`q_1(0)`$ $`=`$ $`\mathrm{log}(M(0)+N(0))q_1^{}(0)={\displaystyle \frac{a_2M(0)+a_1N(0)}{M(0)+N(0)}}`$ (1.21)   
$`q_2(0)`$ $`=`$ $`\mathrm{log}{\displaystyle \frac{a_2^2M(0)+a_1^2N(0)}{(a_2a_1)^2M(0)N(0)}}q_2^{}(0)={\displaystyle \frac{a_1a_2(a_2M(0)+a_1N(0))}{a_2^2M(0)+a_1^2N(0)}}.`$ (1.22)   
Some care must be used in choosing the sign in (1.19). It is clear that for large negative $`t`$, $`\mu _1(q_1(t),t)`$ refers to the path of one peakon while for large positive $`t`$ it refers to the other. If this were not the case, simple asymptotic analysis of (1.12) would show that the peakons change speed which is not the case. Therefore $`q_1(t)`$ represents the path of one of the peakons until some time $`t^{}`$ and the other one after this time. The opposite is true for $`q_2(t)`$. At the time $`t^{}`$ we say that a change of identity has taken place. $`t^{}`$ can be found explicitly by using the fact that at this time, the two peaks must have the same height. But the peaks have the same height exactly when

$$a_2M(t^{})=a_1N(t^{}).$$
(1.23)   
Without loss of generality we can rescale time such that $`t^{}=0`$. In this case (1.23), due to the original definitions of $`m_1,n_1`$ given in terms of $`\mu _1^0`$ $`\mu _2^0`$, corresponds to a restriction on the choice of $`\mu _1^0`$ and $`\mu _2^0`$, namely

$$a_2^2\frac{\mu _1^0a_1}{\mu _1^0a_2}=a_1^2\frac{\mu _2^0a_2}{\mu _2^0a_1}.$$
(1.24)   
This condition is satisfied for example when $`\mu _1^0={\displaystyle \frac{a_1a_2}{a_1+a_2}}`$ and $`\mu _2^0={\displaystyle \frac{a_1+a_2}{2}}`$. Also notice that under this rescaling, the phase shift is simply $`q_1(0)q_2(0)`$.

Please place the second figure near here

So we now have a procedure to make the change of identity occur at $`t=0`$, i.e. $`\mu _1`$ goes from representing the first peakon to the second one at $`t=0`$. This change is represented by the change in the sign of the plus/minus in (1.19). That is, the sign is chosen as positive for $`t<0`$ and negative for $`t>0`$. However, $`M`$ remains continuous despite this sign change since the change of identity occurs precisely when the term under the square root is zero. Therefore (1.14)-(1.16) and (1.19) together describe the solution $`U(X,t)`$ of the SW equation as a function of $`x`$ and $`t`$ depending on two parameters $`M(0)`$, $`N(0)`$.

By using the approach of this paper weak billiard solutions can be obtained for the whole class of $`n`$-peakon solutions of $`N`$-component systems.

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
