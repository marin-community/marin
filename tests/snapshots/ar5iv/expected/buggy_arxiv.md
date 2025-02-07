# Untitled Document

Functional Inversion for Potentials

in Quantum Mechanics

Richard L. Hall

Department of Mathematics and Statistics,

Concordia University,

1455 de Maisonneuve Boulevard West,

MontrÃ©al, QuÃ©bec, Canada H3G 1M8.

email:Â Â rhall@cicma.concordia.ca

Abstract

Let $`E=Fâ(v)`$ be the ground-state eigenvalue of the SchrÃ¶dinger Hamiltonian $`H=â\mathrm{Î}+vâfâ(x),`$ where the potential shape $`fâ(x)`$ is symmetric and monotone increasing for $`x>0,`$ and the coupling parameter $`v`$ is positive. If the kinetic potential $`\stackrel{Â¯}{f}â(s)`$ associated with $`fâ(x)`$ is defined by the transformation $`\stackrel{Â¯}{f}â(s)=F^{â²}â(v),s=Fâ(v)âvâF^{â²}â(v),`$ then $`f`$ can be reconstructed from $`F`$ by the sequence $`f^{[n+1]}=\stackrel{Â¯}{f}â\stackrel{Â¯}{f}^{[n]^{â1}}âf^{[n]}.`$ Convergence is proved for special classes of potential shape; for other test cases it is demonstrated numerically. The seed potential shape $`f^{[0]}`$ need not be âcloseâ to the limit $`f.`$

PACSÂ Â 03 65 Ge

1.Â Â Introduction Â  We consider the SchrÃ¶dinger operator

$$H=â\mathrm{Î}+vâfâ(x)$$
$`(1.1)`$   
defined on some suitable domain in $`L^2â(\text{Â }\text{R}).`$ The potential has two aspects: an âattractiveâ potential shape $`fâ(x),`$ and a coupling parameter $`v>0.`$ We assume that $`fâ(x)`$ is symmetric, non-constant, and monotone increasing for $`x>0.`$ Elementary trial functions can be designed for such an operator to prove that for each $`v>0`$ there is a discrete eigenvalue $`E=Fâ(v)`$ at the bottom of the spectrum. If, in addition to the the above properties, we also assume that $`f`$ is continuous at $`x=0`$ and that it is piecewise analytic, then we are able to proveÂ \[1\] that $`f`$ is uniquely determined by $`F.`$ The subject of the present paper is the reconstruction of the potential shape $`f`$ from knowledge of the âenergy trajectoryâ $`F.`$ This is an example of what we call âgeometric spectral inversionâÂ \[1, 2\]. 

Geometric spectral inversion should be distinguished from the âinverse problem in the coupling constantâ which has been analysed in detail by Chadan et alÂ \[3-8\]. In the latter problem the discrete part of the input data consists of the set $`\{v_i\}`$ of values of the coupling that yield a given fixed energy $`E.`$ Inversion from excited-state energy trajectories $`F_kâ(v),k>0,`$ has also been studied, by a complete inversion of the WKB approximation for bound statesÂ \[9\]. For the ground-state energy trajectory $`Fâ(v)=F_0â(v)`$ a constructive numerical inversion algorithm has been devisedÂ \[10\], and an inversion inequality has been establishedÂ \[11\]. The work reported in the present paper also concerns inversion from the ground-state energy trajectory, but the approach uses functional methods which have a natural extension to the higher energy trajectories. 

Geometry is involved with this problem because we deal with a family of operators depending on a continuous parameter $`v.`$ This immediately leads to a family of spectral manifolds, and, more particularly, to the consideration of smooth transformations of potentials, and to the transformations which they in turn induce on the spectral manifolds. This is the environment in which we are able to construct the following functonal inversion sequence that is the central theme of the present paper:

$$f^{[n+1]}=\stackrel{Â¯}{f}â\stackrel{Â¯}{f}^{[n]^{â1}}âf^{[n]}â¡\stackrel{Â¯}{f}âK^{[n]}.$$
$`(1.2)`$   
A kinetic potential is the constrained mean value of the potential shape $`\stackrel{Â¯}{f}â(s)=<f>,`$ where the corresponding mean kinetic energy $`s=<â\mathrm{Î}>`$ is held constant. It turns out that kinetic potentials may be obtained from the corresponding energy trajectory $`F`$ by what is essentially a Legendre transformationÂ \[12\] $`\stackrel{Â¯}{f}âF`$ givenÂ \[13\] by 

$$\{\stackrel{Â¯}{f}â(s)=F^{â²}â(v),s=Fâ(v)âvâF^{â²}â(v)\}â\{Fâ(v)/v=\stackrel{Â¯}{f}â(s)âsâ\stackrel{Â¯}{f}^{â²}â(s),1/v=â\stackrel{Â¯}{f}^{â²}â(s)\}.$$
$`(1.3)`$   
As we shall explain in more detail in Section 2, these transformations are well defined because of the definite convexities of $`F`$ and $`\stackrel{Â¯}{f};`$ they complete the definition of the inversion sequence (1.2), up to the choice of a starting seed potential $`f^{[0]}â(x).`$ They differ from Legendre transformations only because of our choice of signs. The choice has been made so that the eigenvalue can be written (exactly) in the semi-classical forms

$$E=Fâ(v)=\underset{s>0}{\mathrm{min}}â¡\left\{s+vâ\stackrel{Â¯}{f}â(s)\right\}=\underset{x>0}{\mathrm{min}}â¡\left\{K^{[f]}â(x)+vâfâ(x)\right\}$$
$`(1.4)`$   
where the kinetic- and potential-energy terms have the âusualâ signs.

After more than 70 years of QM (and even more of the Sturm-Liouville problem) it may appear to be an extravagance to seek to rewrite the min-max characterization of the spectrum in slightly different forms, with kinetic potentials and K functions. The main reason for our doing this is that the new representations allows us to tackle the following problem: if $`g`$ is a smooth transformation, and we know the spectrum of $`â\mathrm{Î}+vâf^{[0]}â(x),`$ what is the spectrum of $`â\mathrm{Î}+vâgâ(f^{[0]}â(x))`$? In the forward direction (obtaining eigenvalues corresponding to a given potential), an approximation called the âenvelope methodâ has been developedÂ \[13, 14\]. The inversion sequence (1.2) was arrived at by an inversion of envelope theory, yielding a sequence of approximations for an initially unknown transformation $`g`$ satisfying $`fâ(x)=gâ(f^{[0]}â(x)).`$

In order to make this paper essentially self contained, the representation apparatus is outlined in Section 2. In Section 3 we use envelope theory to generate the inversion sequence. In Section 4 it is proved that the energy trajectory for a pure power potential is inverted from an arbitrary pure-power seed in only two steps: thus $`f^{[2]}=f`$ in these cases. In Section 5 we consider the exactly soluble problem of the sech-squared potential $`fâ(x)=â\mathrm{sech}^2â(x).`$ Starting from the seed $`f^{[0]}â(x)=â1+x^2/20,`$ we are able to construct the first iteration $`f^{[1]}`$ exactly; we then continue the sequence by using numerical methods. This illustration is interesting because the seed potential shape $`f^{[0]}`$ is very different from that of the target $`fâ(x)`$ and has a completely different, entirely discrete, spectrum. We consider also another sequence in which the seed is $`f^{[0]}â(x)=â1/(1+x/5).`$ Convergence, which, of course, cannot be proved with the aid of a computer, is strongly indicated by both of these examples.

2.Â Â Kinetic potentials and K functions Â  The term âkinetic potentialâ is short for âminimum mean iso-kinetic potentialâ. If the Hamiltonian is $`H=â\mathrm{Î}+vâfâ(x),`$ where $`fâ(x)`$ is potential shape, and $`\mathrm{ð}â(H)âL^2â(\text{Â }\text{R})`$ is the domain of $`H,`$ then the ground-state kinetic potential $`\stackrel{Â¯}{f}â(s)=\stackrel{Â¯}{f}_0â(s)`$ is definedÂ \[13, 14\] by the expression 

$$\stackrel{Â¯}{f}â(s)=\underset{\genfrac{}{}{0pt}{}{\genfrac{}{}{0pt}{}{\mathrm{Ï}â\mathrm{ð}â\left(H\right)}{(\mathrm{Ï},\mathrm{Ï})=1}}{(\mathrm{Ï},â\mathrm{Î}â\mathrm{Ï})=s}}{inf}(\mathrm{Ï},fâ\mathrm{Ï}).$$
$`(2.1)`$   
The extension of this definition to the higher discrete eigenvalues (for $`v`$ sufficiently large) is straightforwardÂ \[13\] but not explicitely needed in the present paper. The idea is that the min-max computation of the discrete eigenvalues is carried out in two stages: in the first stage (2.1) the mean potential shape is found for each fixed value of the mean kinetic energy $`s;`$ in the second and final stage we minimize over $`s.`$ Thus we have arrive at the semi-classical expression which is the first equality of Eq.(1.4). It is well known that $`Fâ(v)`$ is concave ($`F^{â²â²}â(v)<0`$) and it follows immediately that $`\stackrel{Â¯}{f}â(s)`$ is convex. More particularly, we haveÂ \[2\] 

$$F^{â²â²}â(v)â\stackrel{Â¯}{f}^{â²â²}â(s)=â\frac{1}{v^3}.$$
$`(2.2)`$   
Thus, although kinetic potentials are defined by (2.1), the transformations (1.3) may be used in practice to go back and forth between $`F`$ and $`\stackrel{Â¯}{f}.`$

Kinetic potentials have been used to study smooth transformations of potentials and also linear combinations. The present work is an application of the first kind. Our goal is to devise a method of searching for a transformation $`g,`$ which would convert the initial seed potential $`f^{[0]}â(x)`$ into the (unknown) goal $`fâ(x)=gâ(f^{[0]}).`$ We shall summarize briefly how one proceeds in the forward direction, to approximate $`F,`$ if we know $`fâ(x).`$ The $`K`$ functions are then introduced, by a change of variable, so that the potential $`fâ(x)`$ is exposed and can be extracted in a sequential inversion process.

In the forward direction we assume that the lowest eigenvalue $`F^{[0]}â(v)`$ of $`H^{[0]}=â\mathrm{Î}+vâf^{[0]}â(x)`$ is known for all $`v>0`$ and we assume that $`fâ(x)`$ is given; hence, since the potentials are symmetric and monotone for $`x>0,`$ we have defined the transformation function $`g.`$ âTangential potentialsâ to $`gâ(f^{[0]})`$ have the form $`a+bâf^{[0]}â(x),`$ where the coefficients $`aâ(t)`$ and $`bâ(t)`$ depend on the point of contact $`x=t`$ of the tangential potential to the graph of $`fâ(x).`$ Each one of these tangential potentials generates an energy trajectory of the form $`\mathrm{â\pm }â(v)=aâv+F^{[0]}â(bâv),`$ and the envelope of this family (with respect to $`t`$) forms an approximation $`F^Aâ(v)`$ to $`Fâ(v).`$ If the transformation $`g`$ has definite convexity, then $`F^Aâ(v)`$ will be either an upper or lower bound to $`Fâ(v).`$ It turns outÂ \[13\] that all the calculations implied by this envelope approximation can be summarized nicely by kinetic potentials. Thus the whole procedure just described corresponds exactly to the expression: 

$$\stackrel{Â¯}{f}â\stackrel{Â¯}{f}^A=gâ\stackrel{Â¯}{f}^{[0]},$$
$`(2.3)`$   
with $`â`$ being replaced by an inequality in case $`g`$ has definite convexity. Once we have an approximation $`\stackrel{Â¯}{f}^A,`$ we immediately recover the corresponding energy trajectory $`F^A`$ from the general minimization formula (1.4).

The formulation that reveals the potential shape is obtained when we use $`x`$ instead of $`s`$ as the minimization parameter. We achieve this by the following general definition of $`x`$ and of the $`K`$ function associated with $`f:`$

$$fâ(x)=\stackrel{Â¯}{f}â(s),K^{[f]}â(x)=\stackrel{Â¯}{f}^{â1}â(fâ(x)).$$
$`(2.4)`$   
The monotonicity of $`fâ(x)`$ and of $`\stackrel{Â¯}{f}`$ guarantee that $`x`$ and $`K`$ are well defined. Since $`\stackrel{Â¯}{f}^{â1}â(f)`$ is a convex function of $`f,`$ the second equality in (1.4) immediately followsÂ \[14\]. In terms of $`K`$ the envelope approximation (2.3) becomes simply

$$K^{[f]}âK^{\left[f^{[0]}\right]}.$$
$`(2.5)`$   
Thus the envelope approximation involves the use of an approximate $`K`$ function that no longer depends on $`f,`$ and there is now the possibility that we can invert (1.4) to extract an approximation for the potential shape.

We end this summary by listing some specific results that we shall need. First of all, the kinetic potentials and $`K`$ functions obeyÂ \[13, 14\] the following elementary shift and scaling laws: 

$$fâ(x)âa+bâfâ(x/t)â\left\{\stackrel{Â¯}{f}â(s)âa+bâ\stackrel{Â¯}{f}â(sât^2),K^{[f]}â(x)â\frac{1}{t^2}âK^{[f]}â\left(\frac{x}{t}\right)\right\}.$$
$`(2.6)`$   
Pure power potentials are important examples which have the following formulas:

$$fâ(x)=|x|^qâ\left\{\stackrel{Â¯}{f}â(s)=\left(\frac{P}{s^{\frac{1}{2}}}\right)^q,Kâ(x)=\left(\frac{P}{x}\right)^2\right\},$$
$`(2.7)`$   
where, if the bottom of the spectrum of $`â\mathrm{Î}+|x|^q`$ is $`Eâ(q),`$ then the $`P`$ numbers are givenÂ \[14\] by the folowing expressions with $`n=0:`$

$$P_nâ(q)=\left|E_nâ(q)\right|^{\frac{(2+q)}{2âq}}â\left[\frac{2}{2+q}\right]^{\frac{1}{q}}â\left[\frac{|q|}{2+q}\right]^{\frac{1}{2}},qâ 0.$$
$`(2.8)`$   
We have allowed for $`q<0`$ and for higher eigenvalues since the formulas are essentially the same. The $`P_nâ(q)`$ as functions of q are interesting in themselvesÂ \[14\]: they have been proved to be monotone increasing, they are probably concave, and $`P_nâ(0)`$ corresponds exactly to the $`\mathrm{log}`$ potential. By contrast the $`E_nâ(q)`$ are not so smooth: for example, they have infinite slopes at $`q=0.`$ But this is another story. An important observation is that the $`K`$ functions for the pure powers are all of the form $`(Pâ(q)/x)^2`$ and they are invariant with respect to both potential shifts and multipliers: thus $`a+bâ|x|^q`$ has the same $`K`$ function as does $`|x|^q.`$ For the harmonic oscillator $`P_nâ(2)=(n+\frac{1}{2})^2,n=0,1,2,\mathrm{â¦}.`$ Other specific examples may be found in the references cited.

The last formulas we shall need are those for the ground state of the sech-squared potential:

$$fâ(x)=â\mathrm{sech}^2â(x)â\left\{\stackrel{Â¯}{f}â(s)=â\frac{2âs}{(s+s^2)^{\frac{1}{2}}+s},Kâ(x)=\mathrm{sinh}^{â2}â(2âx)\right\}.$$
$`(2.9)`$   
3.Â Â The inversion sequence Â  The inversion sequence (1.2) is based on the following idea. The goal is to find a transformation $`g`$ so that $`f=gâf^{[0]}.`$ We choose a seed $`f^{[0]},`$ but, of course, $`f`$ is unknown. In so far as the envelope approximation with $`f^{[0]}`$ as a basis is âgoodâ, then an approximation $`g^{[1]}`$ for $`g`$ would be given by $`\stackrel{Â¯}{f}=g^{[1]}â\stackrel{Â¯}{f}^{[0]}.`$ Thus we have

$$gâg^{[1]}=\stackrel{Â¯}{f}â\stackrel{Â¯}{f}^{[0]^{â1}}.$$
$`(3.1)`$   
Applying this approximate transformation to the seed we find:

$$fâf^{[1]}=g^{[1]}âf^{[0]}=\stackrel{Â¯}{f}â\stackrel{Â¯}{f}^{[0]^{â1}}âf^{[0]}=\stackrel{Â¯}{f}âK^{[0]}.$$
$`(3.2)`$   
We now use $`f^{[1]}`$ as the basis for another envelope approximation, and, by repetition, we have the ansatz (1.2), that is to say

$$f^{[n+1]}=\stackrel{Â¯}{f}â\stackrel{Â¯}{f}^{[n]^{â1}}âf^{[n]}=\stackrel{Â¯}{f}âK^{[n]}.$$
$`(3.3)`$   
A useful practical device is to invert the second expression for $`F`$ given in (1.4) to obtain

$$K^{[f]}â(x)=\underset{v>0}{\mathrm{max}}â¡\left\{Fâ(v)âvâfâ(x)\right\}.$$
$`(3.4)`$   
The concavity of $`Fâ(v)`$ explains the $`\mathrm{max}`$ in this inversion, which, as it stands, is exact. In a situation where $`f`$ is unknown, we have $`f`$ on both sides and nothing can be done with this formal result. However, in the inversion sequence which we are considering, (3.4) is extremely useful. If we re-write (3.4) for stage \[n\] of the inversion sequence it becomes:

$$K^{[n]}â(x)=\underset{v>0}{\mathrm{max}}â¡\left\{F^{[n]}â(v)âvâf^{[n]}â(x)\right\}.$$
$`(3.5)`$   
In this application, the current potential shape $`f^{[n]}`$ and consequently $`F^{[n]}â(v)`$ can be found (by shooting methods) for each value of $`v.`$ The minimization can then be performed even without differentiation (for example, by using a Fibonacci search) and this is a much more effective method for $`K^{[n]}=\stackrel{Â¯}{f}^{[n]^{â1}}âf^{[n]}`$ than finding $`\stackrel{Â¯}{f}^{[n]}â(s),`$ finding the functional inverse, and applying the result to $`f^{[n]}.`$

4.Â Â Inversion for pure powers Â  We now treat the case of pure-power potentials given by

$$fâ(x)=A+Bâ|x|^q,q>0,$$
$`(4.1)`$   
where $`A`$ and $`B>0`$ are arbitrary and fixed. We shall prove that, starting from another pure power as a seed, the inversion sequence converges in just two steps. The exact energy trajectory $`Fâ(v)`$ for the potential (4.1) is assumed known. Hence, so is the exact kinetic potential given by (2.7) and the general scaling rule (2.6), that is to say

$$\stackrel{Â¯}{f}â(s)=A+Bâ\left(\frac{Pâ(q)}{s^{\frac{1}{2}}}\right)^q.$$
$`(4.2)`$   
We now suppose that a pure power is also used as a seed, thus we have

$$f^{[0]}â(x)=a+bâ|x|^pâK^{[0]}â(x)=\left(\frac{Pâ(p)}{x}\right)^2,$$
$`(4.3)`$   
where the parameters $`a,b>0,p>0`$ are arbitrary and fixed. The first step of the inversion (1.4) therefore yields

$$f^{[1]}â(x)=\left(\stackrel{Â¯}{f}âK^{[0]}\right)â(x)=A+Bâ\left(\frac{Pâ(q)â|x|}{Pâ(p)}\right)^q.$$
$`(4.4)`$   
The approximate potential $`f^{[1]}â(x)`$ now has the correct $`x`$ power dependence but has the wrong multiplying factor. Because of the invariance of the $`K`$ functions to multipliers, this error is completely corrected at the next step, yielding:

$$K^{[1]}â(x)=\left(\frac{Pâ(q)}{x}\right)^2âf^{[2]}â(x)=\left(\stackrel{Â¯}{f}âK^{[1]}\right)â(x)=A+Bâ|x|^q.$$
$`[4.5]`$   
This establishes our claim that power potentials are inverted without error in exactly two steps.

The implications of this result are a little wider than one might first suspect. If the potential that is being reconstructed has the asymptotic form of a pure power for small or large $`x,`$ say, then we know that the inversion sequence will very quickly produce an accurate approximation for that part of the potential shape. More generally, since the first step of the inversion process involves the construction of $`K^{[0]},`$ the general invariance property $`K^{[a+bâf]}=K^{[f]}`$ given in (2.6) means that the seed potential $`f^{[0]}`$ may be chosen without special consideration to gross features of $`f`$ already arrived at by other methods. For example, the area (if the potential has area), or the starting value $`fâ(0)`$ need not be incorporated in $`f^{[0]},`$ say, by adjusting $`a`$ and $`b.`$

5.Â Â More general examples Â  We consider the problem of reconstructing the sech-squared potential $`fâ(x)=â\mathrm{sech}^2â(x).`$ We assume that the corresponding exact energy trajectory $`Fâ(v)`$ and, consequently, the kinetic potential $`\stackrel{Â¯}{f}â(s)`$ are known. ThusÂ \[14\]: 

$$fâ(x)=â\mathrm{sech}^2â(x)â\left\{Fâ(v)=â\left((v+\frac{1}{4})^{\frac{1}{2}}â\frac{1}{2}\right)^2,\stackrel{Â¯}{f}â(s)=â\frac{2âs}{(s+s^2)^{\frac{1}{2}}+s}\right\}.$$
$`(5.1)`$   
We study two seeds. The first seed is essentially $`x^2,`$ but we use a scaled version of this for the purpose of illustration in Fig.(1). Thus we have

$$f^{[0]}=â1+\frac{x^2}{20}âK^{[0]}â(x)=\frac{1}{4âx^2}$$
$`(5.1)`$   
This potential generates the exact eigenvalue

$$F^{[0]}â(v)=âv+\left(\frac{v}{20}\right)^{\frac{1}{2}}$$
$`(5.2)`$   
which, like the potential itself, is very different from that of the goal. After the first iteration we obtain

$$f^{[1]}â(x)=\stackrel{Â¯}{f}â\left(K^{[0]}â(x)\right)=â\frac{2}{1+(1+4âx^2)^{\frac{1}{2}}}.$$
$`(5.3)`$   
A graph of this potential is shown as $`fâ1`$ in Fig.(1). In order to continue analytically we would need to solve the problem with Hamiltonian $`H^{[1]}=â\mathrm{Î}+vâf^{[1]}â(x)`$ exactly to find an expression for $`F^{[1]}â(v).`$ We know no way of doing this. However, it can be done numerically, with the aid of the inversion formula (3.5) for $`K.`$ The first 5 iterations shown in Fig.(1) suggest convergence of the series.

As a second example we consider the initial potential given by

$$f^{[0]}â(x)=â\frac{1}{1+|x|/5}.$$
$`(5.4)`$   
In this case none of the steps can be carried out exactly. In Fig.(2) we show the first five iterations. Again, convergence is indicated by this sequence, with considerable progress being made in the first step.

The numerical technique needed to solve these problems is not the main point of the present work. However, a few remarks about this are perhaps appropriate. As we showed in Ref.\[10\], by looking at three large values of $`v`$ we can determine the best model of the form $`fâ(x)=A+Bâ|x|^q`$ that fits the given $`Fâ(v)`$ for small $`x<x_a.`$ We have used this method here with $`v=10000Ã\{1,\frac{1}{2},\frac{1}{4}\}`$ and $`x_a=0.2`$ in all cases. As indicated above, the inversion (3.5) was used to determine each $`K^{[n]}`$ function from the corresponding $`F^{[n]}.`$ For all the graphs shown in the two figures, the range of $`v`$ still to be considered for $`x>x_a`$ turned out to be $`0.0008<v<175.`$ With 40 points beyond $`x_a`$ on each polygonal approximation for $`f^{[n]}â(x),`$ a complete set of graphs for one figure took about $`4â¤\frac{1}{2}`$ minutes to compute with a program written in C++ on a PentiumPro running at 200MHz. The exact expression for $`f^{[1]}`$ arising from the harmonic-oscillator starting point was very useful for verifying the behaviour of the program.

6.Â Â Conclusion Â  Progress has certainly been made with geometric spectral inversion. The results reported in this paper suggest strongly that in some suitable topology, the inversion sequence (1.4) converges to $`f.`$ The ânaturalâ extension of (1.4) to excited states leads to the conjecture that each of the following sequences converges to $`f:`$

$$f_k^{[n+1]}=\stackrel{Â¯}{f}_kâ\stackrel{Â¯}{f}_k^{[n]^{â1}}âf_k^{[n]}â¡\stackrel{Â¯}{f}âK_k^{[n]},k=0,1,2,\mathrm{â¦}$$
$`(6.1)`$   
For the examples studied by the inversion of the WKB approximation, inversion improved rapidly as $`k`$ increased and the view of the problem became more âclassicalâ.

If an energy trajectory $`Fâ(v)`$ is derived from a potential which vanishes at large distances, is bounded, and has area, it is straightforwardÂ \[1\] to ânormalizeâ the function $`Fâ(v)`$ by scaling so that it corresponds to a potential with area $`2`$ and lowest point $`â1.`$ The graphs of $`F_0â(v)`$ for normalized potentials with square-well, exponential, and sech-squared shapes look very similar: for small $`v`$ they are asymptotically like $`âv^2,`$ and for large $`v`$ they satisfy $`lim_{vâ\mathrm{â}}\{F_0â(v)/v\}=â1.`$ These asymptotic features are the same for all such normalized potentials. We now know that encoded in the details of these $`F_0â(v)`$ curves for intermediate values of $`v`$ are complete recipes for the corresponding potentials. If, as the WKB studies strongly suggest, the code could be unravelled for the excited states too, the situation would become very interesting. What this would mean is that, given any one energy trajectory $`F_kâ(v),`$ we could reconstruct from it the underlying potential shape $`fâ(x)`$ and then, by solving the problem in the forward direction, go on to find all the other trajectories $`\{F_jâ(v)\}_{jâ k},`$ and all the scattering data. For large $`k`$ this would imply that a classical view alone could determine the potential and hence all the quantum phenomena which it might generate.

Acknowledgment Â  Partial financial support of this work under Grant No. GP3438 from the Natural Sciences and Engineering Research Council of Canada is gratefully acknowledged. References 

\[1\]âR. L. Hall, J. Phys. A:Math. Gen 28, 1771 (1995).

\[2\]âR. L. Hall, Phys. Rev. A 50, 2876 (1995).

\[3\]âK. Chadan, C. R. Acad. Sci. Paris SÃ¨r. II 299, 271 (1984).

\[4\]âK. Chadan and H. Grosse, C. R. Acad. Sci. Paris SÃ¨r. II 299, 1305 (1984).

\[5\]âK. Chadan and R. Kobayashi, C. R. Acad. Sci. Paris SÃ¨r. II 303, 329 (1986).

\[6\]âK. Chadan and M. Musette, C. R. Acad. Sci. Paris SÃ¨r. II 305, 1409 (1987).

\[7\]âK. Chadan and P. C. Sabatier, Inverse Problems in Quantum Scattering Theory (Springer, New York, 1989).Â The âinverse problem in the coupling constantâ is discussed on p406

\[8\]âB. N. Zakhariev and A.A.Suzko, Direct and Inverse Problems: Potentials in Quantum Scattering Theory (Springer, Berlin, 1990).Â The âinverse problem in the coupling constantâ is mentioned on p53

\[9\]âR. L. Hall, Phys. Rev A 51, 1787 (1995).

\[10\]âR. L. Hall, J. Math. Phys. 40, 669 (1999).

\[11\]âR. L. Hall, J. Math. Phys. 40, 2254 (1999).

\[12\]âI. M. Gelfand and S. V. Fomin, Calculus of Variations (Prentice-Hall, Englewood Cliffs, 1963).Â Legendre transformations are discussed on p 72.

\[13\]âR. L. Hall, J. Math. Phys. 25, 2708 (1984).

\[14\]âR. L. Hall, J. Math. Phys. 34, 2779 (1993).Â 

FigureÂ (1)Â Â The energy trajectory $`F`$ for the sech-squared potential $`fâ(x)=â\mathrm{sech}^2â(x)`$ is approximately inverted starting from the seed $`f^{[0]}â(x)=â1+x^2/20.`$ The first step can be completed analytically yielding $`fâ1=f^{[1]}â(x)=â2/\{1+\sqrt{1+4âx^2}\}.`$ Four more steps $`\{fâk=f^{[k]}\}_{k=2}^5`$ of the inversion sequence approaching $`f`$ are performed numerically.

FigureÂ (2)Â Â The energy trajectory $`F`$ for the sech-squared potential $`fâ(x)=â\mathrm{sech}^2â(x)`$ is approximately inverted starting from the seed $`fâ0=f^{[0]}â(x)=â1/(1+x/5).`$ The first 5 steps $`\{fâk=f^{[k]}\}_{k=1}^5`$ of the inversion sequence approaching $`f`$ are performed numerically.