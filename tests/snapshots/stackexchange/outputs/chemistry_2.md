# Question
Title: Hierarchy of electronic wavefunctions
*The previous question contained too much unnecessary information and was edited.*

I am wondering about the "hierarchy" of wavefunctions. If one can combine atomic orbitals (AO) into molecular orbitals (MO) through the LCAO method, could one combine MOs to form a *supra* wavefunction, or an LCMO?

I've done some research and found a paper on an approach to finding the MO of molecules by effectively taking a LCMO of smaller molecules or constituents. While interesting, this is not what I'm referring to as a "supra" wavefunction, if there's such a thing.

# Answer
> 12 votes
I've been reading up on this, and will take a shot at the answer. It is going to be an interesting answer, and probably not what you expect. There has been an interesting conversation going on in the *Journal of Chemical Education* in response to this letter advocating the retirement of the localized hybrid orbitals because they "aren't real" and cannot be used to predict "all properties" of molecules. Rebuttals included comments that delocalized molecular orbitals also fail to predict "all properties". However, the final word in the discussion came from this letter, which reminds us what orbitals are.

Orbitals are not real. The electrons that populate orbitals are real. Orbitals are mathematical constructs created to describe the curious wave-like probabilistic quantum mechanical properties of electrons. When we talk about atomic orbitals, we generally mean the 1-electron hydrogenic atomic orbitals (HAOs), which are mathematically represented as a wavefunction ($\Psi$) and graphically represented by the probability density function ($\Psi ^ * \Psi$) at either 90% or 95% level. Despite valiant and massive efforts, wavefunctions for multi-electron atoms cannot be solved exactly. They are approximated as a parameterized HAOs. When we talk about the atomic orbitals of carbon, we usually mean the HAOs that closely approximate what we think the AOs of carbon should be. 

Molecules and their orbitals are even more troublesome. The LCAO approach creates a bunch of nice 1-electron wavefunctions out of the set of HAOs. These methods give rise to both the canonical delocalized molecules orbitals (CMOs), and the localized valence bond orbitals (VBOs). As the authors of that letter point out:

> The term canonical implies that CMOs are not the only possible MOs, and there are many orbitals that generate an identical “chemical reality”.

The approximate polyelectron wavefunction of a molecule (at least in HF theory) is the Slater determinant of a series of 1-electron CMOs $\psi \_i$: $$ \psi \_1 \overline{\psi} \_1 \psi \_2 \overline{\psi} \_2 \psi \_3 \overline{\psi} \_3 \psi \_4 \overline{\psi} \_4 ... $$

An interesting property of Slater determinants is that unitary operations (like rotations) on the set of CMOs return a different set of MOs, which produce the same expectation values for the polyelectron wave function as original set:

$$ \psi \_1 \overline{\psi} \_1 \psi \_2 \overline{\psi} \_2 \psi \_3 \overline{\psi} \_3 \psi \_4 \overline{\psi} \_4 ...~~ \underrightarrow{unitary~~transformation} ~~\psi \_a \overline{\psi} \_a \psi \_b \overline{\psi} \_b \psi \_c \overline{\psi} \_c \psi \_d \overline{\psi} \_d ...$$

Mathematically, it seems to me, that a linear combination of MOs that are themselves linear combinations of AOs would still be a linear combination of AOs, just a different one than you started with. If that is true, then all you have is a different set of MOs than you started with, and according to HF theory, that should give you the same expectation values. 

Thus a set of "LCMOs" are no more or less valid than the original set of LCAOs. They likely do not produce any additional insights into the structure of the molecule. However, the approach might be computationally more efficient when trying to determine the MOs of very large molecules and polymers.

# Answer
> 5 votes
As a supplement to Ben Norris's answer, I thought I'd add the following:

When you use a basis to construct an orbital, your original basis functions can already be represented as linear combinations of your whole set:

$\chi(x,y,z,s)=\chi(\mathbf{x})= c\_1\varphi\_1 + c\_2\varphi\_2+c\_3\varphi\_3\dots \\ \varphi\_1 = 1\varphi\_1 + 0\varphi\_2 + 0 \varphi\_3\dots$ 

(where $\chi$ is an MO and $\varphi$ is a basis function)

So, altering the coefficients or linearly combining MOs doesn't increase the complexity of your result: $\chi\_a + \chi\_b = (c\_{a,1} + c\_{b,1})\varphi\_1 + (c\_{a,2}+c\_{b,2})\varphi\_2 + (c\_{a,3}+c\_{b,3})\varphi\_3\dots$

When you construct a wave function from basis functions, the usual method is to use a Slater determinant - the determinant of a matrix as follows:

$\Phi(\mathbf{x}\_1,\mathbf{x}\_2,\dots\mathbf{x}\_N) = \frac{1}{\sqrt{N!}}\left| \begin{matrix} \chi\_1(\mathbf{x}\_1) & \chi\_2(\mathbf{x}\_1) & \cdots & \chi\_N(\mathbf{x}\_1) \\ \chi\_1(\mathbf{x}\_2) & \chi\_2(\mathbf{x}\_2) & \cdots & \chi\_N(\mathbf{x}\_2) \\ \vdots & \vdots & \ddots & \vdots \\ \chi\_1(\mathbf{x}\_N) & \chi\_2(\mathbf{x}\_N) & \cdots & \chi\_N(\mathbf{x}\_N) \end{matrix} \right|$

(from the Wikipedia article)

This gives you an expression involving products of your MOs rather than just linear sums, increasing the complexity. This is kind of awkward looking for the n-function case, so instead, for a 3 MO case: $\Phi(\mathbf{x}\_1,\mathbf{x}\_2,\mathbf{x}\_3) = \frac{1}{\sqrt{3!}} ( -\chi\_3(\mathbf{x}\_1)\chi\_{2}(\mathbf{x}\_2)\chi\_1(\mathbf{x}\_3) + \chi\_2(\mathbf{x}\_1)\chi\_{3}(\mathbf{x}\_2)\chi\_3(\mathbf{x}\_1) + \chi\_3(\mathbf{x}\_1)\chi\_{1}(\mathbf{x}\_2)\chi\_2(\mathbf{x}\_3) + \chi\_1(\mathbf{x}\_1)\chi\_{1}(\mathbf{x}\_2)\chi\_3(\mathbf{x}\_3) + \chi\_2(\mathbf{x}\_1)\chi\_{1}(\mathbf{x}\_2)\chi\_3(\mathbf{x}\_3) ) $

Then, just as you can use AOs to build MOs, you can either use that alone as your wave function, or you can use linear combinations of them to obtain a *multiconfigurational* wave function - the MCSCF methods mentioned in jjj's answer: $\Psi = c\_1\Phi\_1 + c\_2\Phi\_2 + c\_3\Phi\_3 \dots$

I'm not sure if this is the sort of thing you were looking for.

# Answer
> 3 votes
I am writing this as a comment, since all the replies are fine, but I cannot write comments. I hope though that the answer is not fully redundant.

Quantum Mechanics is linear. In the space of a particle you can write the wave function as a linear combination of whatever functions. So you can write MOs (molecular orbitals) as LC (linear combination) of AOs (atomic orbitals) or AOs as LC of MOs. The distinction between these set of basis is physical, not mathematical.

One should realise that the electronic wave function of a molecule is not an orbital: it "dwells" in a 3N space (not including spin) where N is the number of electrons, and it must by antisymmetrized. Again, since Quantum Mechanics is linear, the LC of wave functions of 3N electrons is an electronic wave function of 3N electrons. You can always approximate one in terms of others that are simpler and indeed this is the basis of all methods beyond Hartree-Fock.

Therefore, the LC of MOs is another MO of a single electron in the molecule, not the wave function of a "molecule" or "super molecule". The LC process does not move you away from the initial space where the function is defined.

# Answer
> 2 votes
(atomic orbitals -\> molecular orbitals) -\> ("Slater determinats" -\> MCSCF wave functions) -\> mixed state N-electron density matrices -\> grand canonical ensemble (Fock space) density matrices

the first two are 1-electron objects, the next two are N-electron objects representing pure states at different level of approximations

---
Tags: physical-chemistry, computational-chemistry, orbitals, molecular-orbital-theory
---
