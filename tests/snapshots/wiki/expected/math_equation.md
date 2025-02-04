# Method of undetermined coefficients - Wikipedia

Jump to content            Main menu move to sidebar hide    Navigation   * Main page
* Contents
* Current events
* Random article
* About Wikipedia
* Contact us

Contribute   * Help
* Learn to edit
* Community portal
* Recent changes
* Upload file

Search           Search         
* Donate
* Create account
* Log in

* Donate
* Create account
* Log in

Pages for logged out editors learn more   * Contributions
* Talk

## Contents

move to sidebar hide  * (Top)
* 1 Description of the method
* 2 Typical forms of the particular integral
* 3 Examples   Toggle Examples subsection 
    + 3.1 Example 1
    + 3.2 Example 2
    + 3.3 Example 3
* 4 References

# Method of undetermined coefficients

* Čeština
* 한국어
* Bahasa Indonesia
* עברית
* Қазақша
* Nederlands
* Português
* Русский
* Shqip
* Türkçe
* Українська
* 中文

Edit links           * Article
* Talk

* Read
* Edit
* View history

Tools move to sidebar hide    Actions   * Read
* Edit
* View history

General   * What links here
* Related changes
* Upload file
* Special pages
* Permanent link
* Page information
* Cite this page
* Get shortened URL
* Download QR code

Print/export   * Download as PDF
* Printable version

In other projects   * Wikidata item

Appearance move to sidebar hide           From Wikipedia, the free encyclopedia   Approach for finding solutions of nonhomogeneous ordinary differential equations 

| Differential equations |
| --- |
| Scope |
| Fields  * Natural sciences * Engineering  * Astronomy * Physics * Chemistry * <br>Biology * Geology  Applied mathematics  * Continuum mechanics * Chaos theory * Dynamical systems  Social sciences  * Economics * Population dynamics  ---  List of named differential equations |
| Classification |
| Types   * Ordinary * Partial * Differential-algebraic * Integro-differential * Fractional * Linear * Non-linear  By variable type  * Dependent and independent variables  * Autonomous * Coupled / Decoupled * Exact * Homogeneous / Nonhomogeneous  Features  * Order * Operator  * Notation |
| Relation to processes * Difference (discrete analogue)  * Stochastic     + Stochastic partial * Delay |
| Solution |
| Existence and uniqueness * Picard–Lindelöf theorem * Peano existence theorem * Carathéodory's existence theorem * Cauchy–Kowalevski theorem |
| General topics * Initial conditions * Boundary values     + Dirichlet     + Neumann     + Robin     + Cauchy problem * Wronskian * Phase portrait * Lyapunov / Asymptotic / Exponential stability * Rate of convergence * Series / Integral solutions * Numerical integration * Dirac delta function |
| Solution methods * Inspection * Method of characteristics * <br>Euler * Exponential response formula * Finite difference (Crank–Nicolson) * Finite element     + Infinite element * Finite volume * Galerkin     + Petrov–Galerkin * Green's function * Integrating factor * Integral transforms * Perturbation theory * Runge–Kutta  * Separation of variables * Undetermined coefficients * Variation of parameters |
| People |
| List * Isaac Newton * Gottfried Leibniz * Jacob Bernoulli * Leonhard Euler * Joseph-Louis Lagrange * Józef Maria Hoene-Wroński * Joseph Fourier * Augustin-Louis Cauchy * George Green * Carl David Tolmé Runge * Martin Kutta * Rudolf Lipschitz * Ernst Lindelöf * Émile Picard * Phyllis Nicolson * John Crank |
| *  *  * |

In mathematics, the **method of undetermined coefficients** is an approach to finding a particular solution to certain nonhomogeneous ordinary differential equations and recurrence relations. It is closely related to the annihilator method, but instead of using a particular kind of differential operator (the annihilator) in order to find the best possible form of the particular solution, an ansatz or 'guess' is made as to the appropriate form, which is then tested by differentiating the resulting equation. For complex equations, the annihilator method or variation of parameters is less time-consuming to perform. 

Undetermined coefficients is not as general a method as variation of parameters, since it only works for differential equations that follow certain forms.<sup>\[1\]</sup> 

## Description of the method

\[edit\] Consider a linear non-homogeneous ordinary differential equation of the form 

$`{\displaystyle \sum _{i=0}^{n}c_iy^{(i)}+y^{(n+1)}=g(x)}`$ where $`{\displaystyle y^{(i)}}`$ denotes the i-th derivative of $`{\displaystyle y}`$, and $`{\displaystyle c_i}`$ denotes a function of $`{\displaystyle x}`$. The method of undetermined coefficients provides a straightforward method of obtaining the solution to this ODE when two criteria are met:<sup>\[2\]</sup> 

1. $`{\displaystyle c_i}`$ are constants.
2. *g*(*x*) is a constant, a polynomial function, exponential function $`{\displaystyle e^{\alpha x}}`$, sine or cosine functions $`{\displaystyle \mathrm{sin}\beta x}`$ or $`{\displaystyle \mathrm{cos}\beta x}`$, or finite sums and products of these functions ($`{\displaystyle \alpha }`$, $`{\displaystyle \beta }`$ constants).

The method consists of finding the general homogeneous solution $`{\displaystyle y_c}`$ for the complementary linear homogeneous differential equation 

$`{\displaystyle \sum _{i=0}^{n}c_iy^{(i)}+y^{(n+1)}=0,}`$ and a particular integral $`{\displaystyle y_p}`$ of the linear non-homogeneous ordinary differential equation based on $`{\displaystyle g(x)}`$. Then the general solution $`{\displaystyle y}`$ to the linear non-homogeneous ordinary differential equation would be 

$`{\displaystyle y=y_c+y_p.}`$<sup>\[3\]</sup> If $`{\displaystyle g(x)}`$ consists of the sum of two functions $`{\displaystyle h(x)+w(x)}`$ and we say that $`{\displaystyle y_{p_1}}`$ is the solution based on $`{\displaystyle h(x)}`$ and $`{\displaystyle y_{p_2}}`$ the solution based on $`{\displaystyle w(x)}`$. Then, using a superposition principle, we can say that the particular integral $`{\displaystyle y_p}`$ is<sup>\[3\]</sup> 

$`{\displaystyle y_p=y_{p_1}+y_{p_2}.}`$ ## Typical forms of the particular integral

\[edit\] In order to find the particular integral, we need to 'guess' its form, with some coefficients left as variables to be solved for. This takes the form of the first derivative of the complementary function. Below is a table of some typical functions and the solution to guess for them. 

| Function of *x* | Form for *y* |
| --- | --- |
| $`{\displaystyle ke^{ax}}`$ | $`{\displaystyle Ce^{ax}}`$ |
| $`{\displaystyle kx^n,n=0,1,2,\dots }`$ | $`{\displaystyle \sum _{i=0}^{n}K_ix^i}`$ |
| $`{\displaystyle k\mathrm{cos}(ax)\text{ or }k\mathrm{sin}(ax)}`$ | $`{\displaystyle K\mathrm{cos}(ax)+M\mathrm{sin}(ax)}`$ |
| $`{\displaystyle ke^{ax}\mathrm{cos}(bx)\text{ or }ke^{ax}\mathrm{sin}(bx)}`$ | $`{\displaystyle e^{ax}(K\mathrm{cos}(bx)+M\mathrm{sin}(bx))}`$ |
| $`{\displaystyle \left(\sum _{i=0}^{n}k_ix^i\right)\mathrm{cos}(bx)\text{ or }\text{ }\left(\sum _{i=0}^{n}k_ix^i\right)\mathrm{sin}(bx)}`$ | $`{\displaystyle \left(\sum _{i=0}^{n}Q_ix^i\right)\mathrm{cos}(bx)+\left(\sum _{i=0}^{n}R_ix^i\right)\mathrm{sin}(bx)}`$ |
| $`{\displaystyle \left(\sum _{i=0}^{n}k_ix^i\right)e^{ax}\mathrm{cos}(bx)\text{ or }\left(\sum _{i=0}^{n}k_ix^i\right)e^{ax}\mathrm{sin}(bx)}`$ | $`{\displaystyle e^{ax}\left(\left(\sum _{i=0}^{n}Q_ix^i\right)\mathrm{cos}(bx)+\left(\sum _{i=0}^{n}R_ix^i\right)\mathrm{sin}(bx)\right)}`$ |

If a term in the above particular integral for *y* appears in the homogeneous solution, it is necessary to multiply by a sufficiently large power of *x* in order to make the solution independent. If the function of *x* is a sum of terms in the above table, the particular integral can be guessed using a sum of the corresponding terms for *y*.<sup>\[1\]</sup> 

## Examples

\[edit\] ### Example 1

\[edit\] Find a particular integral of the equation 

$`{\displaystyle y^″+y=t\mathrm{cos}t.}`$ The right side *t* cos *t* has the form 

$`{\displaystyle P_ne^{\alpha t}\mathrm{cos}\beta t}`$ with *n* = 2, *α* = 0, and *β* = 1. 

Since *α* \+ *iβ* = *i* is *a simple root* of the characteristic equation 

$`{\displaystyle \lambda ^2+1=0}`$ we should try a particular integral of the form 

$`{\displaystyle \begin{array}{rl}y_p& =t\left[F_1(t)e^{\alpha t}\mathrm{cos}\beta t+G_1(t)e^{\alpha t}\mathrm{sin}\beta t\right]\\ & =t\left[F_1(t)\mathrm{cos}t+G_1(t)\mathrm{sin}t\right]\\ & =t\left[\left(A_0t+A_1\right)\mathrm{cos}t+\left(B_0t+B_1\right)\mathrm{sin}t\right]\\ & =\left(A_0t^2+A_1t\right)\mathrm{cos}t+\left(B_0t^2+B_1t\right)\mathrm{sin}t.\end{array}}`$ Substituting *y*<sub>*p*</sub> into the differential equation, we have the identity 

$`{\displaystyle \begin{array}{rl}t\mathrm{cos}t& =y_p^″+y_p\\ & =\left[\left(A_0t^2+A_1t\right)\mathrm{cos}t+\left(B_0t^2+B_1t\right)\mathrm{sin}t\right]^″+\left[\left(A_0t^2+A_1t\right)\mathrm{cos}t+\left(B_0t^2+B_1t\right)\mathrm{sin}t\right]\\ & =\left[2A_0\mathrm{cos}t+2\left(2A_0t+A_1\right)(-\mathrm{sin}t)+\left(A_0t^2+A_1t\right)(-\mathrm{cos}t)+2B_0\mathrm{sin}t+2\left(2B_0t+B_1\right)\mathrm{cos}t+\left(B_0t^2+B_1t\right)(-\mathrm{sin}t)\right]\\ & +\left[\left(A_0t^2+A_1t\right)\mathrm{cos}t+\left(B_0t^2+B_1t\right)\mathrm{sin}t\right]\\ & =[4B_0t+(2A_0+2B_1)]\mathrm{cos}t+[-4A_0t+(-2A_1+2B_0)]\mathrm{sin}t.\end{array}}`$ Comparing both sides, we have 

$`{\displaystyle \{\begin{array}{l}1=4B_0\\ 0=2A_0+2B_1\\ 0=-4A_0\\ 0=-2A_1+2B_0\end{array}}`$ which has the solution 

$`{\displaystyle A_0=0,A_1=B_0=\frac{1}{4},B_1=0.}`$ We then have a particular integral 

$`{\displaystyle y_p=\frac{1}{4}t\mathrm{cos}t+\frac{1}{4}t^2\mathrm{sin}t.}`$ ### Example 2

\[edit\] Consider the following linear nonhomogeneous differential equation: 

$`{\displaystyle \frac{dy}{dx}=y+e^x.}`$ This is like the first example above, except that the nonhomogeneous part ($`{\displaystyle e^x}`$) is *not* linearly independent to the general solution of the homogeneous part ($`{\displaystyle c_1e^x}`$); as a result, we have to multiply our guess by a sufficiently large power of *x* to make it linearly independent. 

Here our guess becomes: 

$`{\displaystyle y_p=Axe^x.}`$ By substituting this function and its derivative into the differential equation, one can solve for *A*: 

$`{\displaystyle \frac{d}{dx}\left(Axe^x\right)=Axe^x+e^x}`$ $`{\displaystyle Axe^x+Ae^x=Axe^x+e^x}`$ $`{\displaystyle A=1.}`$ So, the general solution to this differential equation is: 

$`{\displaystyle y=c_1e^x+xe^x.}`$ ### Example 3

\[edit\] Find the general solution of the equation: 

$`{\displaystyle \frac{dy}{dt}=t^2-y}`$ $`{\displaystyle t^2}`$ is a polynomial of degree 2, so we look for a solution using the same form, 

$`{\displaystyle y_p=At^2+Bt+C,}`$ Plugging this particular function into the original equation yields, 

$`{\displaystyle 2At+B=t^2-(At^2+Bt+C),}`$ $`{\displaystyle 2At+B=(1-A)t^2-Bt-C,}`$ $`{\displaystyle (A-1)t^2+(2A+B)t+(B+C)=0.}`$ which gives: 

$`{\displaystyle A-1=0,2A+B=0,B+C=0.}`$ Solving for constants we get: 

$`{\displaystyle y_p=t^2-2t+2}`$ To solve for the general solution, 

$`{\displaystyle y=y_p+y_c}`$ where $`{\displaystyle y_c}`$ is the homogeneous solution $`{\displaystyle y_c=c_1e^{-t}}`$, therefore, the general solution is: 

$`{\displaystyle y=t^2-2t+2+c_1e^{-t}}`$ \[edit\]  

| *  *  *  Differential equations | |
| --- | --- |
| Classification | Operations  * Differential operator * Notation for differentiation * Ordinary * Partial * Differential-algebraic * Integro-differential * Fractional * Linear * Non-linear * Holonomic  Attributes of variables  * Dependent and independent variables * Homogeneous * Nonhomogeneous * Coupled * Decoupled * Order * Degree * Autonomous * Exact differential equation * On jet bundles  Relation to processes  * Difference (discrete analogue) * Stochastic     + Stochastic partial * Delay |
| Solutions | Existence/uniqueness  * Picard–Lindelöf theorem * Peano existence theorem * Carathéodory's existence theorem * Cauchy–Kowalevski theorem  Solution topics  * Wronskian * Phase portrait * Phase space * Lyapunov stability * Asymptotic stability * Exponential stability * Rate of convergence * Series solutions * Integral solutions * Numerical integration * Dirac delta function  Solution methods  * Inspection * Substitution * Separation of variables * Method of undetermined coefficients * Variation of parameters * Integrating factor * Integral transforms * Euler method * Finite difference method * Crank–Nicolson method * Runge–Kutta methods * Finite element method * Finite volume method * Galerkin method * Perturbation theory |
| Examples | * List of named differential equations * List of linear ordinary differential equations * List of nonlinear ordinary differential equations * List of nonlinear partial differential equations |
| Mathematicians | * Isaac Newton * Gottfried Wilhelm Leibniz * Leonhard Euler * Jacob Bernoulli * Émile Picard * Józef Maria Hoene-Wroński * Ernst Lindelöf * Rudolf Lipschitz * Joseph-Louis Lagrange * Augustin-Louis Cauchy * John Crank * Phyllis Nicolson * Carl David Tolmé Runge * Martin Kutta * Sofya Kovalevskaya |

* Boyce, W. E.; DiPrima, R. C. (1986). *Elementary Differential Equations and Boundary Value Problems* (4th ed.). John Wiley & Sons. ISBN 0-471-83824-1.
* Riley, K. F.; Bence, S. J. (2010). *Mathematical Methods for Physics and Engineering*. Cambridge University Press. ISBN 978-0-521-86153-3.
* Tenenbaum, Morris; Pollard, Harry (1985). *Ordinary Differential Equations*. Dover. ISBN 978-0-486-64940-5.
* de Oliveira, O. R. B. (2013). "A formula substituting the undetermined coefficients and the annihilator methods". *Int. J. Math. Educ. Sci. Technol*. **44** (3): 462–468. arXiv:1110.4425. Bibcode:2013IJMES..44..462R. doi:10.1080/0020739X.2012.714496. S2CID 55834468.

Retrieved from "https://en.wikipedia.org/w/index.php?title=Method\_of\_undetermined\_coefficients&oldid=1117728414" Category: * Ordinary differential equations
Hidden categories: * CS1 maint: multiple names: authors list
* Articles with short description
* Short description matches Wikidata
* This page was last edited on 23 October 2022, at 07:52 (UTC).
* Text is available under the Creative Commons Attribution-ShareAlike 4.0 License; additional terms may apply. By using this site, you agree to the Terms of Use and Privacy Policy. Wikipedia® is a registered trademark of the Wikimedia Foundation, Inc., a non-profit organization.

* Privacy policy
* About Wikipedia
* Disclaimers
* Contact Wikipedia
* Code of Conduct
* Developers
* Statistics
* Cookie statement
* Mobile view

* 
* 

Search                       
# Method of undetermined coefficients

From Wikipedia, the free encyclopediaIn mathematics, the **method of undetermined coefficients** is an approach to finding a particular solution to certain nonhomogeneous ordinary differential equations and recurrence relations. It is closely related to the annihilator method, but instead of using a particular kind of differential operator (the annihilator) in order to find the best possible form of the particular solution, an ansatz or 'guess' is made as to the appropriate form, which is then tested by differentiating the resulting equation. For complex equations, the annihilator method or variation of parameters is less time-consuming to perform. 

Undetermined coefficients is not as general a method as variation of parameters, since it only works for differential equations that follow certain forms.<sup>\[1\]</sup> 

## Description of the method

Consider a linear non-homogeneous ordinary differential equation of the form 

where $`{\displaystyle y^{(i)}}`$ denotes the i-th derivative of $`{\displaystyle y}`$, and $`{\displaystyle c_i}`$ denotes a function of $`{\displaystyle x}`$.The method of undetermined coefficients provides a straightforward method of obtaining the solution to this ODE when two criteria are met:<sup>\[2\]</sup> 

- $`{\displaystyle c_i}`$ are constants.
- *g*(*x*) is a constant, a polynomial function, exponential function $`{\displaystyle e^{\alpha x}}`$, sine or cosine functions $`{\displaystyle \mathrm{sin}\beta x}`$ or $`{\displaystyle \mathrm{cos}\beta x}`$, or finite sums and products of these functions ($`{\displaystyle \alpha }`$, $`{\displaystyle \beta }`$ constants).
The method consists of finding the general homogeneous solution $`{\displaystyle y_c}`$ for the complementary linear homogeneous differential equation 

and a particular integral $`{\displaystyle y_p}`$ of the linear non-homogeneous ordinary differential equation based on $`{\displaystyle g(x)}`$. Then the general solution $`{\displaystyle y}`$ to the linear non-homogeneous ordinary differential equation would be 

$`{\displaystyle y=y_c+y_p.}`$<sup>\[3\]</sup>If $`{\displaystyle g(x)}`$ consists of the sum of two functions $`{\displaystyle h(x)+w(x)}`$ and we say that $`{\displaystyle y_{p_1}}`$ is the solution based on $`{\displaystyle h(x)}`$ and $`{\displaystyle y_{p_2}}`$ the solution based on $`{\displaystyle w(x)}`$. Then, using a superposition principle, we can say that the particular integral $`{\displaystyle y_p}`$ is<sup>\[3\]</sup> 

## Typical forms of the particular integral

In order to find the particular integral, we need to 'guess' its form, with some coefficients left as variables to be solved for. This takes the form of the first derivative of the complementary function. Below is a table of some typical functions and the solution to guess for them. 

Function of *x* Form for *y* If a term in the above particular integral for *y* appears in the homogeneous solution, it is necessary to multiply by a sufficiently large power of *x* in order to make the solution independent. If the function of *x* is a sum of terms in the above table, the particular integral can be guessed using a sum of the corresponding terms for *y*.<sup>\[1\]</sup> 

## Examples

### Example 1

Find a particular integral of the equation 

The right side *t* cos *t* has the form 

with *n* = 2, *α* = 0, and *β* = 1. 

Since *α* \+ *iβ* = *i* is *a simple root* of the characteristic equation 

we should try a particular integral of the form 

Substituting *y*<sub>*p*</sub> into the differential equation, we have the identity 

Comparing both sides, we have 

which has the solution 

We then have a particular integral 

### Example 2

Consider the following linear nonhomogeneous differential equation: 

This is like the first example above, except that the nonhomogeneous part ($`{\displaystyle e^x}`$) is *not* linearly independent to the general solution of the homogeneous part ($`{\displaystyle c_1e^x}`$); as a result, we have to multiply our guess by a sufficiently large power of *x* to make it linearly independent. 

Here our guess becomes: 

By substituting this function and its derivative into the differential equation, one can solve for *A*: 

So, the general solution to this differential equation is: 

### Example 3

Find the general solution of the equation: 

$`{\displaystyle t^2}`$ is a polynomial of degree 2, so we look for a solution using the same form, 

Plugging this particular function into the original equation yields, 

which gives: 

Solving for constants we get: 

To solve for the general solution, 

where $`{\displaystyle y_c}`$ is the homogeneous solution $`{\displaystyle y_c=c_1e^{-t}}`$, therefore, the general solution is: 

