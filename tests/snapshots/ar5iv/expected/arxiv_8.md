# Pluricanonical systems on surfaces with small ð¾Â².

## Introduction

The main purpose of this paper is studying of the properties of pluricanonical systems on surfaces with $`K^2â¤4`$.

The first part is devoted to the study of the spannedness of bicanonical systems on surfaces of general type with small $`K^2`$ or, which is the same, on canonical models of such surfaces.

The study of the bicanonical system on a canonical surface has two advantages. Namely, it is well known that linear systems tend to have less singular points as base points (e.g., see Theorem 1.1). The second advantage is that studying of an ample line bundle is easier than that of a nef one.

The main result of this part is the following

###### Theorem 0.1

Let $`X`$ be a canonical surface with $`K_X^2=4`$. Then $`|2âK_X|`$ has no fixed part and every base point of $`X`$ is smooth. In particular, the bicanonical system on the minimal model of $`X`$ has no base component.

A well-known theorem says that $`|2âK_X|`$ is base point free if $`K_X^2>4`$ or $`p_gâ(X)>0`$ but our knowledge of base points, or even base components, in the case when $`2â¤K_X^2â¤4`$ and $`p_g=0`$ is very limited (see e.g.Â \[CT\] and \[We\]).

To prove Theorem 0.1 we use our earlier work on adjoint linear series (\[La1\] and \[La2\]) generalizing Reiderâs criterion to normal surfaces to exclude singular base points and all but one possibility for the fixed component and then we use the bilinear map lemma and Cliffordâs theorem for arbitrary CohenâMacaulay curves (see Section 2) to exclude the remaining case.

In the second part of the paper we describe which degree $`2`$ clusters are contracted by $`\mathrm{Ï}_{4âK_X}`$ on a numerical Godeaux surface and by $`\mathrm{Ï}_{3âK_X}`$ on a Campedelli surface. Here we have the following theorems (looking at them one should remember that the torsion group $`\mathrm{Tors}â¡X`$ is finite; see 1.6):

###### Theorem 0.2

Let $`X`$ be a canonical numerical Godeaux surface. If $`|4âK_X|`$ contracts a degree $`2`$ cluster $`\mathrm{Î\P }`$, then $`\mathrm{Î\P }`$ is contained in a curve $`Dâ|K_X+\mathrm{Ï}|`$, where $`0â \mathrm{Ï}â\mathrm{Tors}â¡X`$ and the morphism $`\mathrm{Ï}_{4âK_X}`$ restricted to $`D`$ is either:

1.âan embedding apart from the contracted cluster $`\mathrm{Î\P }`$ if $`2â\mathrm{Ï}â 0`$ in $`\mathrm{Tors}â¡X`$,or

2.âa double covering of $`\mathrm{â}^1`$, if $`2â\mathrm{Ï}=0`$ in $`\mathrm{Tors}â¡X`$.

###### Theorem 0.3

Let $`X`$ be a canonical model of a numerical Campedelli surface. If a degree $`2`$ cluster $`\mathrm{Î\P }`$ is contracted by $`|3âK_X|`$, then one of the following holds:

1.âThere exists an honestly hyperelliptic curve $`Câ|2âK_X|`$ containing $`\mathrm{Î\P }`$ and $`\mathrm{Ï}_{3âK_X}|_C`$ is a double covering of $`\mathrm{â}^1`$.

2.â$`\mathrm{Î\P }`$ is a scheme-theoretic intersection of two curves $`B_1â|K_X+\mathrm{Ï}|`$ and $`B_2â|K_Xâ\mathrm{Ï}|`$, for some $`\mathrm{Ï}â\mathrm{Tors}â¡X`$ such that $`2â\mathrm{Ï}â 0`$.

After writing down this paper (apart from the part after Lemma 5.4, which was inspired by the paper of Kotschick) we learnt about a related paper of Kotschick \[Ko\], who however considered mainly the case of torsion free numerically Godeaux or Campedelli surfaces. Our Theorems 0.2 and 0.3 (together with Proposition 5.5 giving a topological interpretation of curves appearing in case 1 of Theorem 0.3) generalize Theorems 1 and 2, \[ibid\].

He also proved (\[Theorem 3, ibid\]) a criterion for spannedness of $`|2âK_X|`$ in terms of the fundamental group of $`X`$ and the second Stifel-Whitney class of the tangent bundle of $`X`$. Since even the fundamental group is known only in explicit examples and can vary (and for some examples it is not finite), the criterion seems to be only of theoretical interest.

## 1. Preliminaries

All varieties are assumed to be defined over $`\mathrm{â}`$ (except in Section 2). In this section we will state some results generalizing Reiderâs criterion. First let us recall a special case of Corollary 5.1.4, \[La2\]:

###### Theorem 1.1

(\[La2\], Corollary 5.1.4). Let $`X`$ be a normal projective surface with only quotient singularities and $`L`$ be a nef Weil divisor on $`X`$ such that $`K_X+L`$ is Cartier. Assume that $`L^2>\frac{4}{r}`$, where $`r`$ is the order of the local fundamental group around a given point $`x`$. Then $`K_X+L`$ is not globally generated at $`x`$ if and only if there exists a connected curve $`D`$ containing $`x`$ such that $`\mathrm{ðª}_Dâ(K_X+L)`$ is not globally generated at $`x`$. Moreover, one can choose $`D`$ satisfying the following conditions:

1.âThere exists an injection $`m_xâ\mathrm{ðª}_Dâ(K_X+L)âª\mathrm{Ï}_D`$,

2.â$`LâDâ\frac{1}{r}â¤D^2â¤\frac{(LâD)^2}{L^2}`$ and $`0â¤LâD<\frac{2}{r}`$,

3.âIf $`X`$ has only Du Val singularities then $`m_xâ\mathrm{ðª}_Dâ(K_X+L)â\mathrm{Ï}_D`$. In particular, $`2âp_aâD=Dâ(K_X+L)+1`$.

Sketch of the proof. If $`K_X+L`$ is not globally generated at $`x`$ then by Serreâs construction there exists a rank $`2`$ reflexive sheaf $`\mathrm{â°}â\mathrm{Ext}^1â¡(m_xâ\mathrm{ðª}â(K_X+L),\mathrm{Ï}_X)`$. This sheaf is locally at the point $`x`$ isomorphic with $`\mathrm{Î\copyright }_X^{ââ}`$. Using this fact the theorem can be proved by using Bogomolovâs instability theorem for surfaces with only quotient singularities (see \[Ka, Lemma 2.5\]).

The details of the proof together with a generalization of the theorem can be found in \[La2\] or in a forthcoming paper of the author.

###### Corollary 1.2

Let $`x`$ be a singular point of the canonical surface $`X`$. Then $`|2âK_X|`$ is globally generated at $`x`$ unless

1.â$`K_X^2=1`$ and $`x`$ is of type $`A_1`$, $`A_2`$, $`A_3`$, or

2.â$`K_X^2=2`$ and $`x`$ is of type $`A_1`$.

A special case of this corollary was proved in \[We\].

###### Definition 1.3

Assume that a normal surface $`X`$ has rational singularity at $`x`$ and let $`f:YâX`$ be the minimal resolution at $`x`$. Let $`Z`$ denote the fundamental cycle (or the exceptional divisor if $`x`$ is smooth). The Seshadri constant of a nef divisor $`L`$ at $`x`$ is the real number

$$\mathrm{Ï\mu }â(L,x)=sup\{\mathrm{Ï\mu }â\yen 0|f^{â}âLâ\mathrm{Ï\mu }âZâ\text{Â is nef}\}.$$

In the proof of the next proposition we will need the following

###### Lemma 1.4

Let $`D`$ be a Weil divisor on a normal surface. If $`D^2â\yen 0`$ and $`DâL>0`$ for some nef divisor $`L`$, then $`D`$ is pseudoeffective. If moreover $`D`$ is not big then $`D^2=0`$ and $`D`$ is nef.

Proof. First part of the lemma follows easily from the Hodge index theorem. The second one follows from the Zariski decomposition for $`D`$, Q.E.D.

###### Proposition 1.5

Let $`X`$ be a canonical surface and $`x`$ a base point of $`|2âK_X|`$. Then

1.âIf $`K_X^2=4`$, then $`x`$ is smooth and $`\mathrm{Ï\mu }â(K_X,x)=2.`$

2.âIf $`K_X^2=2`$ and $`x`$ is singular, then $`x`$ is of type $`A_1`$ and $`\mathrm{Ï\mu }â(K_X,x)=1`$.

Proof. This is just a simple corollary to Corollary 1.2 and \[La3, Corollary 3.2\].

1.6. Let $`X`$ be a canonical surface with $`p_g=q=0`$. Recall that $`X`$ is called numerical Godeaux (numerical Campedelli) if $`K_X^2=1`$ ($`K_X^2=2`$, respectively). Note that usually these notions are defined for a minimal model but as we said before it is more convenient to use the canonical model.

The torsion group $`\mathrm{Tors}â¡X`$ (i.e., a torsion subgroup of $`\text{Pic}âX`$; it is the same for $`X`$ and its minimal model) of a numerical Godeaux or Campedelli surface is finite. In fact if $`X`$ is numerical Godeaux then $`\mathrm{Tors}â¡X`$ is cyclic of order at most $`5`$ (and all these possibilities occur). If $`X`$ is numerical Campedelli then instead one can consider $`\mathrm{Ï}_1^{aâlâg}â(X)`$ which is a related group. This group is finite of order at most $`9`$. Moreover, if $`|\mathrm{Ï}_1^{aâlâg}â(X)|=9`$ then $`\mathrm{Tors}â¡X=\mathrm{Ï}_1^{aâlâg}â(X)=\mathrm{Ï}_1â(X)=\mathrm{â¤}_3â\mathrm{â¤}_3`$. If $`|\mathrm{Ï}_1^{aâlâg}â(X)|=8`$ then $`\mathrm{Ï}_1^{aâlâg}â(X)=\mathrm{Ï}_1â(X)`$ and we have the following possibilities:

a)â$`\mathrm{Ï}_1â(X)=\mathrm{Tors}â¡X=\mathrm{â¤}_2â\mathrm{â¤}_2â\mathrm{â¤}_2`$, $`\mathrm{â¤}_2â\mathrm{â¤}_4`$ or $`Z_8`$,

b)â$`\mathrm{Ï}_1â(X)=\mathrm{â}`$ (the quaternion group) and $`\mathrm{Tors}â¡X=\mathrm{â¤}_2â\mathrm{â¤}_2`$,

and all of them occur (this follows from explicit description of those surfaces in \[Re\]).

1.7. Let us recall (see e.g., \[CFHR\]) that a Gorenstein curve $`C`$ is called honestly hyperelliptic if there exists a finite morphism $`\mathrm{Ï}:Câ\mathrm{â}^1`$ of degree $`2`$. Clearly every irreducible reduced curve of genus $`2`$ is honestly hyperellipic.

## 2. Cliffordâs lemma

Let us recall the following well-known bilinear map lemma:

###### Lemma 2.1

(Hopf, \[Ha, Lemma 5.1\]) Let $`\mathrm{Ï}:V_1ÃV_2â\P W`$ be a bilinear map of nonzero finite-dimensional vector spaces (over an algebraically closed field $`k`$), which is nondegenerate, i.e., for each $`v_1â 0`$ in $`V_1`$ and each $`v_2â 0`$ in $`V_2`$, $`\mathrm{Ï}â(v_1,v_2)â 0`$. Then

$$dâiâmâWâ\yen dâiâmâV_1+dâiâmâV_2â1.$$

###### Corollary 2.2

Let $`X`$ be an integral scheme defined over an algebraically closed field $`k`$. Let $`\mathrm{â}`$ and $`\mathrm{â³}`$ be two coherent subsheaves of the sheaf of total quotient rings $`\mathrm{ð¦}_X`$ such that $`h^0â(\mathrm{â})â\yen 1`$ and $`h^0â(\mathrm{â³})â\yen 1`$. Then

$$h^0â(\mathrm{â}â\mathrm{â³})â\yen h^0â(\mathrm{â})+h^0â(\mathrm{â³})â1,$$

where $`\mathrm{â}â\mathrm{â³}`$ is the product of sheaves in $`\mathrm{ð¦}_X`$.

Proof. We have a natural multiplication map $`\mathrm{â}â(X)Ã\mathrm{â³}â(X)â\P (\mathrm{â}â\mathrm{â³})â(X)`$. If $`s_1â\mathrm{â}â(X)`$ and $`s_2â\mathrm{â³}â(X)`$ then $`s_1âs_2â 0`$ in $`(\mathrm{â}â\mathrm{â³})â(X)â\mathrm{ð¦}â(X)`$ and we can apply the bilinear map lemma, Q.E.D.

###### Definition 2.3

A sheaf $`L`$ on a scheme $`X`$ is called invertible in codimension $`0`$, or generically invertible, if it is locally isomorphic to $`\mathrm{ðª}_X`$ at every generic point of $`X`$.

###### Lemma 2.4

(Clifford). Let $`C`$ be a Cohen-Macaulay, projective curve over an algebraically closed field $`k`$. Let $`\mathrm{â}`$ be a generically invertible torsion free coherent sheaf such that $`\mathrm{deg}â¡\mathrm{â}â¤2âp_aâC+h^0â(\mathrm{ðª}_C)â1`$. Then there exists a generically Gorenstein subcurve $`BâC`$ such that $`h^0â(B,\mathrm{â}|_B)=0`$ or

$$2âh^0â(B,\mathrm{â}|_B)â¤\mathrm{deg}â¡\mathrm{â}|_B+h^0â(\mathrm{ðª}_B)+1.$$

Proof. By the definition of degree and Serre duality we have:

$$\mathrm{Ï}â(\mathrm{â})=h^0â(\mathrm{â})âh^0â(\mathrm{â}âoâmâ(\mathrm{â},\mathrm{Ï}_C))=\mathrm{deg}â¡\mathrm{â}+\mathrm{Ï}â(\mathrm{ðª}_C)$$

Now if $`h^0â(\mathrm{â}âoâmâ(\mathrm{â},\mathrm{Ï}_C))=0`$, then $`2âh^0â(\mathrm{â})=2â\mathrm{deg}â¡\mathrm{â}+2â\mathrm{Ï}â(\mathrm{ðª}_C)â¤\mathrm{deg}â¡\mathrm{â}+h^0â(\mathrm{ðª}_C)+1`$, because $`\mathrm{deg}â¡\mathrm{â}â¤2âp_aâC+h^0â(\mathrm{ðª}_C)â1`$.

Hence we can assume that $`h^0â(\mathrm{â}âoâmâ(\mathrm{â},\mathrm{Ï}_C))â 0`$ and $`h^0â(\mathrm{â})â 0`$. There exists a subcurve $`BâC`$ such that $`\mathrm{Hom}â¡(\mathrm{â},\mathrm{Ï}_C)â 0`$ and every non-zero homomorphism $`\mathrm{Ï}:\mathrm{â}|_Bâ\mathrm{Ï}_B`$ is generically onto. Indeed, if $`\mathrm{Ï}`$$`â\mathrm{Hom}â¡(\mathrm{â},\mathrm{Ï}_C)`$ is not generically onto then we can choose a subcurve $`B^{â²}âC`$ defined by $`\mathrm{Ann}â¡\mathrm{Ï}`$, such that $`\mathrm{Ï}`$ has factorisation

$$\mathrm{â}â\mathrm{â}|_{B^{â²}}â\stackrel{\mathrm{Ï}_{B^{â²}}}{â}\mathrm{Ï}_{B^{â²}}â\mathrm{Ï}_C,$$

where $`\mathrm{Ï}_{B^{â²}}`$ is generically onto (see \[CFHR, Lemma 2.4\]). If necessary we continue this process for other homomorphisms $`\mathrm{â}|_{B^{â²}}â\mathrm{Ï}_{B^{â²}}`$ until we get the required curve.

Obviously we can assume that $`h^0â(B,\mathrm{â}|_B)â 0`$. Then we have a natural pairing $`H^0â(B,\mathrm{â}|_B)Ã\mathrm{Hom}â¡(\mathrm{â}|_B,\mathrm{Ï}_B)âH^0â(\mathrm{Ï}_B)`$ and by the above we see that assumptions of the bilinear map lemma are satisfied. Therefore

$$h^0â(B,\mathrm{â}|_B)+h^0â(\mathrm{â}âoâmâ(\mathrm{â}|_B,\mathrm{Ï}_B))â¤h^0â(\mathrm{Ï}_B)+1.$$

Hence

$$2âh^0â(B,\mathrm{â}|_B)â\mathrm{deg}â¡\mathrm{â}|_Bâ\mathrm{Ï}â(\mathrm{ðª}_B)=h^0â(B,\mathrm{â}|_B)+h^0â(\mathrm{â}âoâmâ(\mathrm{â}|_B,\mathrm{Ï}_B))â¤h^0â(\mathrm{Ï}_B)+1,$$

which proves the lemma.

We will usually apply Cliffordâs lemma for rank one torsion-free sheaves on a reduced, irreducible curve.

## 3. Proof of Theorem 0.1

The second part of the theorem is contained in Corollary 1.2.

Let us write $`|2âK_X|=|M|+V`$, where $`V`$ is a fixed part. We can assume that $`p_gâ(X)=qâ(X)=0`$, since otherwise $`2âK_X`$ is base point free. Then $`|M|`$ is not composed with a pencil (see \[Xi\]). Therefore a general member of $`|M|`$ is irreducible and reduced. By an abuse of notation we will write $`M`$ for the general member of $`|M|`$.

###### Lemma 3.1

$$2âK_X^2â¤p_aâMâ¤\frac{1}{2}â(K_XâM+M^2)+1$$

Proof. From the long cohomology exact sequence corresponding to

$$0â\P \mathrm{ðª}_Xâ(K_XâM)â\P \mathrm{ðª}â(K_X)â\P \mathrm{ðª}_Mâ(K_X)â\P 0,$$

one can get

$$0â\P H^1â(K_X|_M)â\P H^2â(K_XâM)â\P H^2â(K_X)â\P 0.$$

Hence by Serre duality we have

$$h^1â(K_X|_M)=h^2â(K_XâM)âh^2â(K_X)=h^0â(M)â1=h^0â(2âK_X)â1=K_X^2.$$

Similarly one can compute

$$h^1â(2âK_X|_M)=h^2â(2âK_XâM)âh^2â(2âK_X)=h^0â(K_XâV)=0$$

since $`h^0â(K_X)=0`$ by assumption. Now using the RiemannâRoch theorem on $`M`$ we get:

$$h^0â(K_X|_M)=\mathrm{Ï}â(K_X|_M)+h^1â(K_X|_M)=\mathrm{Ï}â(\mathrm{ðª}_M)+K_XâM+K_X^2$$

and

$$h^0â(2âK_X|_M)=\mathrm{Ï}â(2âK_X|_M)+h^1â(2âK_X|_M)=\mathrm{Ï}â(\mathrm{ðª}_M)+2âK_XâM.$$

Using Corollary 2.2, we obtain:

$$2âh^0â(K_X|_M)â1â¤h^0â(2âK_X|_M).$$

Substituting the above equalities for both sides of the previous inequality, we get $`2âK_X^2â¤p_aâM`$. The inequality $`2âp_aâ(M)â2â¤(K_X+M)âM`$ (a ânumerical subadjunctionâ) is a consequence of the RiemannâRoch theorem for surfaces with at most Du Val singularities (see e.g., \[La1, Theorem 2.1\]). In fact, from this theorem it follows that an equality holds if and only if $`M`$ is Cartier, Q.E.D.

Remark.  The lemma works also for surfaces $`K_X^2=2`$ or $`3`$ and $`p_gâ(X)=0`$. It limits the number and intersection numbers of the possible base components of the bicanonical system.

From Lemma 1.4 it follows that $`K_XâVâ\yen 2âmâuâlât_xâV`$ for any $`xâV`$. In particular $`K_XâVâ\yen 2`$.

Since $`V`$ does not pass through the singular points of $`X`$ by Corollary 1.2, $`M=2âK_XâV`$ is a Cartier divisor and hence $`2âp_aâMâ2=(K_X+M)âM`$. Using this equality and Lemma 3.1 we get $`M^2â\yen 14âKâM=6+KâVâ\yen 8`$. By the Hodge index theorem $`(KâM)^2â\yen K^2âM^2â\yen 4â8`$, hence $`KâMâ\yen 6`$. Using $`KâM+KâV=2âK^2=8`$, we are left with only one case: $`KâV=2`$, $`KâM=6`$. Since $`M^2=2âp_aâMâ2âK_XâM`$ is divisible by $`2`$ and $`8â¤M^2â¤(KâM)^2/K^2=9`$, we get $`M^2=8`$, $`MâV=4`$ and $`V^2=0`$.

Now it is easy to see that there is only one possibility: $`K_XâV=2`$, $`K_XâM=6`$, $`M^2=8`$, $`MâV=4`$, $`V^2=0`$.

In this case we have a sequence:

$$0â\P \mathrm{ðª}â(MâV)â\P \mathrm{ðª}â(M)â\P \mathrm{ðª}_Vâ(M)â\P 0$$

and

$$h^0â(\mathrm{ðª}_Vâ(M))â¤\frac{1}{2}âMâV+1=3$$

by Cliffordâs theorem, since $`MâVâ¤2âp_aâ(V)=4`$. Hence we get $`h^0â(\mathrm{ðª}â(MâV))â\yen h^0â(\mathrm{ðª}â(M))âh^0â(\mathrm{ðª}_Vâ(M))â\yen 2.`$ Let $`G`$ be a generic member of $`|MâV|`$. We will prove that $`h^0â(\mathrm{ðª}_A)â¤2`$ for every subcurve $`AâG`$. Then, since $`4=MâGâ¤2âp_aâG=6`$, using Cliffordâs theorem, we get

$$4=h^0â(M)âh^0â(V)â¤h^0â(\mathrm{ðª}_Aâ(M))â¤\frac{1}{2}â(MâA+h^0â(\mathrm{ðª}_A)+1)<\frac{1}{2}âMâ(MâV)+2=4,$$

hence a contradiction.

Because $`MâV=2â(K_XâV)`$, $`K_XâV`$ is pseudoeffective. If it is big, then we can apply \[La1, Theorem 3.6\], since $`h^1â(K_X+(K_XâV))=h^1â(M)=h^0â(\mathrm{ðª}_Vâ(2âK_X))â 0`$. Therefore there exist divisors $`A`$ and $`B`$ such that $`K_XâV=A+B`$, $`AâB`$ is pseudoeffective (and numerically nontrivial) and $`B`$ is effective. Hence $`1â¤BâK_X<AâK_X`$ and we get a contradiction with $`(A+B)âK_X=(K_XâV)âK_X=2`$.

Hence $`MâV`$ is not big and it is easy to see that it is nef. Therefore if we write $`|MâV|=|D|+F`$, where $`F`$ is the fixed part of $`|MâV|`$, then $`Dâ(D+F)=Fâ(D+F)=(MâV)^2=0`$. Hence $`DâFâ¤D^2+DâF=0`$ and we have a contradiction unless $`F=0`$. In the latter case $`|D|`$ is composed with a pencil. Because $`qâ(X)=0`$, $`|D|`$ is composed with a rational pencil. Moreover, $`|D|`$ has no base points since $`D^2=0`$. Therefore $`D=f^{â}â\mathrm{ðª}_{\mathrm{â}^1}â(2)`$, where $`f:Xâ\mathrm{â}^1`$ is the morphism defined by $`|D|`$, and $`h^0â(\mathrm{ðª}_D)=2`$, Q.E.D.

## 4. Numerical Godeaux surfaces

###### Proposition 4.1

Let $`X`$ be a canonical numerical Godeaux surface. Then $`|4âK_X|`$ contracts a degree $`2`$ cluster $`\mathrm{Î\P }`$ if and only if $`\mathrm{Î\P }`$ is a scheme-theoretical intersection of two curves $`D_1â|K_X+\mathrm{Ï}|`$ and $`D_2â|2âK_Xâ\mathrm{Ï}|`$, where $`0â \mathrm{Ï}â\mathrm{Tors}â¡X`$.

Proof. By \[La1, Theorem 0.2\] applied to $`L=3âK_X`$ one can see that if $`\mathrm{Î\P }`$ is contracted by $`|4âK_X|`$, then there exists an effective Cartier divisor $`D`$ passing through $`\mathrm{Î\P }`$ such that $`K_XâD=1`$, $`p_aâD=2`$, $`Dâ¡K_X`$ and $`\mathrm{Ï}_Dâ\mathrm{â}_{\mathrm{Î\P }}â\mathrm{ðª}_Dâ(4âK_X)`$. Moreover, from the construction of the divisor $`D`$, we see that the bundle $`\mathrm{â°}â\mathrm{Ext}^1â¡(\mathrm{â}_{\mathrm{Î\P }}â\mathrm{ðª}â(4âK_X),\mathrm{Ï}_X)`$ sits in an exact sequence

$$0â\mathrm{ðª}â(K_X+A)â\P \mathrm{â°}â\P \mathrm{ðª}â(K_X+D)â0$$

(since $`\mathrm{Ï}â(\mathrm{â°})=\mathrm{Ï}â(K_X+A)+\mathrm{Ï}â(K_X+D)`$). But $`\mathrm{Ext}^1â¡(\mathrm{ðª}â(K_X+D),\mathrm{ðª}â(K_X+A))=H^1â(AâD)`$. Because $`A+Dâ¼3âK_X`$, we have $`AâDâ¡K_X`$. Now one can easily see that $`h^1â(AâD)=0`$ and $`\mathrm{â°}â\mathrm{ðª}â(K_X+A)â\mathrm{ðª}â(K_X+D)`$. Recall that we have a surjection $`\mathrm{Î\pm }:\mathrm{â°}â\mathrm{â}_{\mathrm{Î\P }}â\mathrm{ðª}â(4âK_X)`$. Since $`\mathrm{ðª}â(âA)=\mathrm{im}â¡\mathrm{Î\pm }|_{\mathrm{ðª}â(K_X+D)}â\mathrm{ðª}â(â4âK_X)âª\mathrm{â}_{\mathrm{Î\P }}`$, we can assume that $`A`$ is an effective divisor and $`\mathrm{â}_{\mathrm{Î\P }}=\mathrm{â}_A+\mathrm{â}_D`$.

If we have two effective divisors $`D_1â¡K_X`$ and $`D_2â¡2âK_X`$ intersecting at $`\mathrm{Î\P }`$ and $`D_1+D_2â¼3âK_X`$, then $`\mathrm{ðª}â(K_X+D_1)â\mathrm{ðª}â(K_X+D_2)`$ gives a non-trivial extension of $`\mathrm{â}_{\mathrm{Î\P }}â\mathrm{ðª}â(4âK_X)`$ by $`\mathrm{Ï}_X`$. Therefore from Serreâs construction $`\mathrm{Î\P }`$ is contracted by $`|4âK_X|`$, Q.E.D.

4.2. Proof of Theorem 0.2.

From the preceding proposition $`\mathrm{Î\P }`$ is contained in a curve $`Dâ|K_X+\mathrm{Ï}|`$, where $`0â \mathrm{Ï}â\mathrm{Tors}â¡X`$. One can easily see that $`h^0â(K_X+\mathrm{Ï})=1`$ and $`D`$ is irreducible, reduced (because $`K_XâD=1`$) of genus $`2`$. Moreover, the trace of $`|4âK_X|`$ on $`D`$ is a complete linear system.

If $`2â\mathrm{Ï}=0`$, then $`\mathrm{ðª}_Dâ(4âK_X)=\mathrm{ðª}_Dâ(2âK_D)`$ and because the curve $`D`$ is honestly hyperelliptic we get 2 of the theorem.

If $`2â\mathrm{Ï}â 0`$, then from the sequence

$$0â\mathrm{ðª}â(K_Xâ2â\mathrm{Ï})â\P \mathrm{ðª}â(2âK_Xâ\mathrm{Ï})â\P \mathrm{ðª}_Dâ(2âK_Xâ\mathrm{Ï})â0$$

one gets $`h^0â(\mathrm{ðª}_Dâ(2âK_Xâ\mathrm{Ï}))=1`$ and $`\mathrm{Î\P }`$ is a zero set of the unique section of $`\mathrm{ðª}_Dâ(2âK_Xâ\mathrm{Ï})`$. Therefore in $`|2âK_Xâ\mathrm{Ï}|`$ there is a unique divisor containing $`D`$ and all the other divisors in $`|2âK_Xâ\mathrm{Ï}|`$ intersect $`D`$ exactly at $`\mathrm{Î\P }`$, Q.E.D.

###### Corollary 4.3

Let $`X`$ be any canonical surface. Then $`\mathrm{Ï}_{4âK_X}`$ is an embedding if and only if $`K_X^2â\yen 2`$ or $`X`$ is a torsion free Godeaux surface.

The following theorem is a generalization of \[Bo, Theorem 7.1\] to the case when the canonical model of a Godeaux surface has singularities (remark: the proof given in \[Bo\] cannot be easily generalized because it uses the fact that if the image of $`\mathrm{Ï}_{4âK_X}â(X)`$ is singular then $`\mathrm{Ï}_{4âK_X}`$ is not biholomorphic at some points).

###### Corollary 4.4

If $`X`$ is a Godeaux surface (i.e., a canonical numerical Godeaux surface with $`\mathrm{Tors}â¡X=\mathrm{â¤}_5`$), then $`\mathrm{Ï}_{4âK_X}`$ contracts only $`4`$ tangent vectors (contained in $`|K_X+\mathrm{Ï}|`$, $`0â \mathrm{Ï}â\mathrm{Tors}â¡X`$) at the base points of $`|2âK_X|`$.

Proof. By Theorem 0.2 $`\mathrm{Ï}`$ contracts only $`4`$ degree $`2`$ clusters, which are scheme-theoretic intersections of curves $`D_1â|K_X+\mathrm{Ï}|`$ and $`2âD_2`$, where $`D_2â|K_X+2â\mathrm{Ï}|`$. But $`D_1âD_2=1`$, hence $`\mathrm{Î\P }`$ is a tangent vector at the point $`P=D_1â\copyright D_2`$ and one can easily see that $`P`$ is a base point of $`|2âK_X|`$, Q.E.D.

## 5. Numerical Campedelli surfaces

###### Theorem 5.1

Let $`C`$ be a Gorenstein curve. If a degree $`2`$ cluster $`\mathrm{Î\P }`$ is contracted by a linear system $`|K_C|`$ and each element of $`\mathrm{Hom}â¡(I_{\mathrm{Î\P }},\mathrm{ðª}_C)`$ is an injection, then $`C`$ is honestly hyperelliptic.

Proof. The proof is the same as the proof of \[CFHR, Theorem 3.6\].

Let us also recall the following

###### Proposition 5.2

(\[Re\]) Let $`X`$ be a numerical Campedelli surface. Then for every $`\mathrm{Ï}â\mathrm{Tors}â¡X`$ we have $`h^1â(\mathrm{Ï})=0`$. In particular, $`h^0â(K_X+\mathrm{Ï})=1`$ for $`\mathrm{Ï}â 0`$.

The proposition follows easily (by passing to the universal covering) from the fact that $`qâ(X)=0`$ and every Ã©tale Galois covering of $`X`$ has bounded degree ($`â¤9`$, see \[Re, Theorem I\] or \[Be, Remarque 5.9\] with bound $`â¤10`$).

5.3. Proof of Theorem 0.3.

The idea of the first part of the proof is stolen from \[CFHR\].

Let $`Câ|\mathrm{â}_{\mathrm{Î\P }}â\mathrm{ðª}â(2âK_X)|`$ and assume that $`\mathrm{Î\P }`$ is contracted by $`|3âK_X|`$. Then we have a surjection

$$H^0â(\mathrm{ðª}â(3âK_X))â\P H^0â(\mathrm{ðª}_Câ(3âK_X))=H^0â(\mathrm{Ï}_C),$$

so $`|\mathrm{Ï}_C|`$ also contracts $`\mathrm{Î\P }`$. Therefore $`dim\mathrm{Hom}â¡(\mathrm{â}_{\mathrm{Î\P },C},\mathrm{ðª}_C)=dim\mathrm{Hom}â¡(\mathrm{â}_{\mathrm{Î\P }}â\mathrm{Ï}_C,\mathrm{Ï}_C)=`$

$`h^1â(\mathrm{â}_{\mathrm{Î\P }}â\mathrm{Ï}_C)=h^1â(\mathrm{Ï}_C)+1=2`$.

Because of Theorem 5.1 we can assume that there exists a nonzero section $`s:\mathrm{â}_{\mathrm{Î\P },C}â\mathrm{ðª}_C`$, which is not an injection (otherwise $`\mathrm{Ï}_{3âK_X}|_C`$ would be a double covering of $`\mathrm{â}^1`$). The section $`s`$ vanishes on some subcurve $`BâC`$ and by the automatic adjunction (\[CFHR, Lemma 2.4\]) we have an injection $`\mathrm{â}_{\mathrm{Î\P }}â\mathrm{ðª}_Bâ(3âK_X)âª\mathrm{Ï}_B`$, which is generically a surjection. Therefore

$$\mathrm{deg}â¡\mathrm{â}_{\mathrm{Î\P }}â\mathrm{ðª}_Bâ(3âK_X)=3âK_XâBâ\mathrm{deg}â¡(\mathrm{Î\P }â\copyright B)â¤\mathrm{deg}â¡\mathrm{Ï}_B,$$

i.e.,

$$3âK_XâBâ\mathrm{deg}â¡\mathrm{Ï}_Bâ¤\mathrm{deg}â¡(\mathrm{Î\P }â\copyright B)â¤2.$$

By the Bombieri connectedness theorem $`C`$ is not numerically 3-connected, $`Bâ¡K_X`$ and $`\mathrm{deg}â¡(\mathrm{Î\P }â\copyright B)=2`$ (see the proof of \[CFHR, Lemma 4.2\]). But this means that $`\mathrm{Î\P }`$ is contained in $`B`$. Now it is sufficient to prove the following

###### Lemma 5.4

In a notation as above $`\mathrm{Î\P }`$ is a scheme-theoretic intersection of two Cartier divisors $`B_1â|K_X+\mathrm{Ï}|`$ and $`B_2â|K_Xâ\mathrm{Ï}|`$, for some $`\mathrm{Ï}â\mathrm{Tors}â¡X`$ such that $`2â\mathrm{Ï}â 0`$.

Proof. There exists a reflexive sheaf $`\mathrm{â°}`$ sitting in the exact sequence

$$0â\mathrm{Ï}_Xâ\P \mathrm{â°}â\stackrel{\mathrm{Î\pm }}{â\P }\mathrm{â}_{\mathrm{Î\P }}â\mathrm{ðª}â(3âK_X)â0.$$

We already have one curve $`B_1=Bâ¡K_X`$ containing $`\mathrm{Î\P }`$. This gives an embedding $`\mathrm{ðª}â(3âK_XâB_1)âª\mathrm{â}_{\mathrm{Î\P }}â\mathrm{ðª}â(3âK_X)`$. Because $`\mathrm{Ext}^1â¡(\mathrm{ðª}â(3âK_XâB_1),\mathrm{Ï}_X)=0`$, this embedding lifts to $`\mathrm{ðª}â(3âK_XâB_1)ââª^{\mathrm{Î²}}\mathrm{â°}`$. One can easily see that $`\mathrm{coker}â¡\mathrm{Î²}`$ is torsion-free (otherwise there would exist a curve $`B_1^{â²}<B`$ containing $`\mathrm{Î\P }`$, which gives a contradiction). Moreover, $`(\mathrm{coker}â¡\mathrm{Î²})^{ââ}=\mathrm{ðª}â(K_X+B_1)`$ and $`\mathrm{Ï}â(\mathrm{â°})=\mathrm{Ï}â(K_X+B_1)+\mathrm{Ï}â(3âK_XâB_1)`$, so $`\mathrm{coker}â¡\mathrm{Î²}=\mathrm{ðª}â(K_X+B_1)`$. Now $`dim\mathrm{Ext}^1â¡(\mathrm{ðª}â(K_X+B_1),\mathrm{ðª}â(3âK_XâB_1))=h^1â(\mathrm{ðª}â(2â\mathrm{Ï}))=0`$, so $`\mathrm{â°}â\mathrm{ðª}â(K_X+B_1)â\mathrm{ðª}â(3âK_XâB_1)`$. The image of $`\mathrm{Î\pm }|_{\mathrm{ðª}â(K_X+B_1)}`$ gives a divisor $`B_2=2âK_XâB_1`$ containing $`\mathrm{Î\P }`$ and such that $`\mathrm{â}_{B_1}+\mathrm{â}_{B_2}=\mathrm{â}_{\mathrm{Î\P }}`$. If $`B_1â|K_X+\mathrm{Ï}|`$, then $`2â\mathrm{Ï}â 0`$, because $`h^0â(K_X+\mathrm{Ï})=1`$, Q.E.D.

Remarks.

(1) Combining Proposition 5.2 and Theorem 0.3 we see that there is only a finite number of degree $`2`$ clusters which are contracted by $`|3âK_X|`$ and are not contained in curves from 1 in Theorem 0.3. It allows for a simple proof of the fact that $`\mathrm{Ï}_{3âK_X}`$ is birational (pass to the 4-th point of the proof in \[BC\]).

(2) Theorem 0.3 could be proven by somewhat longer arguments but similar as in the proof of Theorem 0.2. Namely, if $`\mathrm{Î\P }`$ is contracted by $`|3âK_X|`$ one can construct a rank $`2`$ reflexive sheaf $`\mathrm{â°}â\mathrm{Ext}^1â¡(\mathrm{â}_{\mathrm{Î\P }}â\mathrm{ðª}â(3âK_X),\mathrm{Ï}_X)`$. A curve $`Câ|\mathrm{â}_{\mathrm{Î\P }}â\mathrm{ðª}â(2âK_X)|`$ defines an injection $`\mathrm{Ï}_Xâª\mathrm{â}_{\mathrm{Î\P }}â\mathrm{ðª}â(3âK_X)`$ lifting to $`\mathrm{Ï}_Xâ\mathrm{â°}`$, since $`\mathrm{Ext}^1â¡(\mathrm{Ï}_X,\mathrm{Ï}_X)=0`$. Consider all the linear combinations $`L`$ of our two maps from $`\mathrm{Ï}_X`$ to $`\mathrm{â°}`$. If a cokernel of any of them has torsion then one can easily prove that we are in case 2 of Theorem 0.3. Otherwise one can prove that the curve $`C`$ is honestly hyperelliptic using the linear system $`L`$. In this last case the sheaf $`\mathrm{â°}`$, which occurs to be a bundle, is stable and by Donaldson theorem from gauge theory it (or rather the bundle $`\mathrm{â°}â(â2âK_X)`$ which has trivial Chern classes) corresponds to an irreducible $`\text{SU}â(2)`$-representation of $`\mathrm{Ï}_1â(X)`$ (in fact one should pull back $`\mathrm{â°}`$ to the minimal model of $`X`$ since Donaldsonâs theorem holds for smooth surfaces; the reverse is slightly harder). Therefore we have the following proposition:

###### Proposition 5.5

There is a bijection between the set of honestly hyperelliptic curves $`Câ|2âK_X|`$ and the set of irreducible $`\text{SU}â(2)`$-representations of $`\mathrm{Ï}_1â(X)`$.

This proposition together with Theorem 0.3 shows that all the clusters contracted by $`|3âK_X|`$ depend on the topology of $`X`$: either they come from the torsion group $`H_1â(X,\mathrm{â¤})`$ or from the representations of the fundamental group. This generalizes \[Ko, Theorem 2\].

5.6. Examples.

(1) Note that if $`X`$ is numerical Campedelli with $`\mathrm{Ï}_1^{aâlâg}â(X)=\mathrm{â¤}_2â\mathrm{â¤}_2â\mathrm{â¤}_2`$ (i.e., $`X`$ is a Campedelli surface) then Proposition 5.5 together with Theorem 0.3 imply that $`\mathrm{Ï}_{3âK_X}`$ is an embedding (since in this case $`\mathrm{Ï}_1â(X)=\mathrm{Ï}_1^{aâlâg}â(X)`$ has no irreducible $`\text{SU}â(2)`$-representations). This was not known even though we knew an explicit description of $`X`$ (see \[Pe, Remark after Theorem 2\]).

(2) If $`X`$ is numerical Campedelli with $`\mathrm{Ï}_1^{aâlâg}â(X)=\mathrm{â}`$ then there are no contracted clusters coming from torsion (see 1.6) but $`\mathrm{Ï}_1â(X)=\mathrm{â}`$ has an irreducible $`\text{SU}â(2)`$-representation so $`\mathrm{Ï}_{3âK_X}`$ is not an embedding.

Summarizing all the known results, we get the following:

###### Corollary 5.7

Let $`X`$ be any canonical surface. Then $`\mathrm{Ï}_{3âK_X}`$ is an embedding is an embedding if and only if $`K_X^2>2`$ or $`X`$ is a Campedelli surface or a numerical Campedelli surface with $`\mathrm{Ï}_1^{aâlâg}âX=\mathrm{â¤}_2â\mathrm{â¤}_2`$, $`\mathrm{â¤}_2`$ or $`0`$ and such that $`\mathrm{Ï}_1â(X)`$ has no irreducible $`\text{SU}â(2)`$-representations.

In view of this corollary it would be very interesting to prove the following

###### Conjecture 5.8

For any numerical Campedelli surface $`\mathrm{Ï}_1^{aâlâg}â(X)=\mathrm{Ï}_1â(X)`$.