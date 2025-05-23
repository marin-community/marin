# Method of undetermined coefficients - Wikipedia

In mathematics, the **method of undetermined coefficients** is an approach to finding a particular solution to certain nonhomogeneous ordinary differential equations and recurrence relations. It is closely related to the annihilator method, but instead of using a particular kind of differential operator (the annihilator) in order to find the best possible form of the particular solution, an ansatz or 'guess' is made as to the appropriate form, which is then tested by differentiating the resulting equation. For complex equations, the annihilator method or variation of parameters is less time-consuming to perform. 

Undetermined coefficients is not as general a method as variation of parameters, since it only works for differential equations that follow certain forms. 

## Description of the method

Consider a linear non-homogeneous ordinary differential equation of the form 

$$\sum \_{i=0}^{n}c\_{i}y^{(i)}+y^{(n+1)}=g(x)$$  

where $y^{(i)}$ denotes the i-th derivative of $y$, and $c\_{i}$ denotes a function of $x$. 

The method of undetermined coefficients provides a straightforward method of obtaining the solution to this ODE when two criteria are met: 

1. $c\_{i}$ are constants.
2. *g*(*x*) is a constant, a polynomial function, exponential function $e^{\alpha x}$, sine or cosine functions $\sin {\beta x}$ or $\cos {\beta x}$, or finite sums and products of these functions ( $\alpha $, $\beta $ constants).

The method consists of finding the general homogeneous solution $y\_{c}$ for the complementary linear homogeneous differential equation 

$$\sum \_{i=0}^{n}c\_{i}y^{(i)}+y^{(n+1)}=0,$$  

and a particular integral $y\_{p}$ of the linear non-homogeneous ordinary differential equation based on $g(x)$. Then the general solution $y$ to the linear non-homogeneous ordinary differential equation would be 

$y=y\_{c}+y\_{p}.$ 

If $g(x)$ consists of the sum of two functions $h(x)+w(x)$ and we say that $y\_{1}$ is the solution based on $h(x)$ and $y\_{2}$ the solution based on $w(x)$. Then, using a superposition principle, we can say that the particular integral $y\_{p}$ is 

$$y\_{p}=y\_{1}}+y\_{2}}.$$  

## Typical forms of the particular integral

In order to find the particular integral, we need to 'guess' its form, with some coefficients left as variables to be solved for. This takes the form of the first derivative of the complementary function. Below is a table of some typical functions and the solution to guess for them. 

| Function of *x* | Form for *y* |
| --- | --- |
| $ke^{ax}$ | $Ce^{ax}$ |
| $kx^{n},\\;n=0,1,2,\ldots $ | $\sum \_{i=0}^{n}K\_{i}x^{i}$ |
| $k\cos(ax){\text{ or }}k\sin(ax)$ | $K\cos(ax)+M\sin(ax)$ |
| $ke^{ax}\cos(bx){\text{ or }}ke^{ax}\sin(bx)$ | $e^{ax}(K\cos(bx)+M\sin(bx))$ |
| $\left(\sum \_{i=0}^{n}k\_{i}x^{i}\right)\cos(bx){\text{ or }}\ \left(\sum \_{i=0}^{n}k\_{i}x^{i}\right)\sin(bx)$ | $\left(\sum \_{i=0}^{n}Q\_{i}x^{i}\right)\cos(bx)+\left(\sum \_{i=0}^{n}R\_{i}x^{i}\right)\sin(bx)$ |
| $\left(\sum \_{i=0}^{n}k\_{i}x^{i}\right)e^{ax}\cos(bx){\text{ or }}\left(\sum \_{i=0}^{n}k\_{i}x^{i}\right)e^{ax}\sin(bx)$ | $e^{ax}\left(\left(\sum \_{i=0}^{n}Q\_{i}x^{i}\right)\cos(bx)+\left(\sum \_{i=0}^{n}R\_{i}x^{i}\right)\sin(bx)\right)$ |

If a term in the above particular integral for *y* appears in the homogeneous solution, it is necessary to multiply by a sufficiently large power of *x* in order to make the solution independent. If the function of *x* is a sum of terms in the above table, the particular integral can be guessed using a sum of the corresponding terms for *y*. 

## Examples

### Example 1

Find a particular integral of the equation 

$$y''+y=t\cos t.$$  

The right side *t* cos *t* has the form 

$$P\_{n}e^{\alpha t}\cos {\beta t}$$  

with *n* = 2, *α* = 0, and *β* = 1. 

Since *α* \+  = *i* is *a simple root* of the characteristic equation 

$$\lambda ^{2}+1=0$$  

we should try a particular integral of the form 

$$\begin{aligned}y\_{p}&=t\left\[F\_{1}(t)e^{\alpha t}\cos {\beta t}+G\_{1}(t)e^{\alpha t}\sin {\beta t}\right\]\\\&=t\left\[F\_{1}(t)\cos t+G\_{1}(t)\sin t\right\]\\\&=t\left\[\left(A\_{0}t+A\_{1}\right)\cos t+\left(B\_{0}t+B\_{1}\right)\sin t\right\]\\\&=\left(A\_{0}t^{2}+A\_{1}t\right)\cos t+\left(B\_{0}t^{2}+B\_{1}t\right)\sin t.\end{aligned}$$  

Substituting *y*<sub>*p*</sub> into the differential equation, we have the identity 

$$\begin{aligned}t\cos t&=y\_{p}''+y\_{p}\\\&=\left\[\left(A\_{0}t^{2}+A\_{1}t\right)\cos t+\left(B\_{0}t^{2}+B\_{1}t\right)\sin t\right\]''+\left\[\left(A\_{0}t^{2}+A\_{1}t\right)\cos t+\left(B\_{0}t^{2}+B\_{1}t\right)\sin t\right\]\\\&=\left\[2A\_{0}\cos t+2\left(2A\_{0}t+A\_{1}\right)(-\sin t)+\left(A\_{0}t^{2}+A\_{1}t\right)(-\cos t)+2B\_{0}\sin t+2\left(2B\_{0}t+B\_{1}\right)\cos t+\left(B\_{0}t^{2}+B\_{1}t\right)(-\sin t)\right\]\\\&\qquad +\left\[\left(A\_{0}t^{2}+A\_{1}t\right)\cos t+\left(B\_{0}t^{2}+B\_{1}t\right)\sin t\right\]\\\&=\[4B\_{0}t+(2A\_{0}+2B\_{1})\]\cos t+\[-4A\_{0}t+(-2A\_{1}+2B\_{0})\]\sin t.\end{aligned}$$  

Comparing both sides, we have 

$$\begin{cases}1=4B\_{0}\\0=2A\_{0}+2B\_{1}\\0=-4A\_{0}\\0=-2A\_{1}+2B\_{0}\end{cases}$$  

which has the solution 

$$A\_{0}=0,\quad A\_{1}=B\_{0}={\frac {1}{4}},\quad B\_{1}=0.$$  

We then have a particular integral 

$$y\_{p}={\frac {1}{4}}t\cos t+{\frac {1}{4}}t^{2}\sin t.$$  

### Example 2

Consider the following linear nonhomogeneous differential equation: 

$$\frac {dy}{dx}}=y+e^{x}.$$  

This is like the first example above, except that the nonhomogeneous part ( $e^{x}$) is *not* linearly independent to the general solution of the homogeneous part ( $c\_{1}e^{x}$); as a result, we have to multiply our guess by a sufficiently large power of *x* to make it linearly independent. 

Here our guess becomes: 

$$y\_{p}=Axe^{x}.$$  

By substituting this function and its derivative into the differential equation, one can solve for *A*: 

$$\frac {d}{dx}}\left(Axe^{x}\right)=Axe^{x}+e^{x$$  

$$Axe^{x}+Ae^{x}=Axe^{x}+e^{x}$$  

$$A=1.$$  

So, the general solution to this differential equation is: 

$$y=c\_{1}e^{x}+xe^{x}.$$  

### Example 3

Find the general solution of the equation: 

$$\frac {dy}{dt}}=t^{2}-y$$  

$t^{2}$ is a polynomial of degree 2, so we look for a solution using the same form, 

$$y\_{p}=At^{2}+Bt+C,$$  

Plugging this particular function into the original equation yields, 

$$2At+B=t^{2}-(At^{2}+Bt+C),$$  

$$2At+B=(1-A)t^{2}-Bt-C,$$  

$$(A-1)t^{2}+(2A+B)t+(B+C)=0.$$  

which gives: 

$$A-1=0,\quad 2A+B=0,\quad B+C=0.$$  

Solving for constants we get: 

$$y\_{p}=t^{2}-2t+2$$  

To solve for the general solution, 

$$y=y\_{p}+y\_{c}$$  

where $y\_{c}$ is the homogeneous solution $y\_{c}=c\_{1}e^{-t}$, therefore, the general solution is: 

$$y=t^{2}-2t+2+c\_{1}e^{-t}$$  

* Boyce, W. E.; DiPrima, R. C. (1986). *Elementary Differential Equations and Boundary Value Problems* (4th ed.). John Wiley & Sons. ISBN 0-471-83824-1.
* Riley, K. F.; Bence, S. J. (2010). *Mathematical Methods for Physics and Engineering*. Cambridge University Press. ISBN 978-0-521-86153-3.
* Tenenbaum, Morris; Pollard, Harry (1985). *Ordinary Differential Equations*. Dover. ISBN 978-0-486-64940-5.
* de Oliveira, O. R. B. (2013). "A formula substituting the undetermined coefficients and the annihilator methods". *Int. J. Math. Educ. Sci. Technol*. **44** (3): 462–468. arXiv:1110.4425. Bibcode:2013IJMES..44..462R. doi:10.1080/0020739X.2012.714496. S2CID 55834468.
