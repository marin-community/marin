# Learning What’s in a Name with Graphical Models — Gradient

# Learning What’s in a Name with Graphical Models

An overview of the structure and statistical assumptions behind linear-chain graphical models — effective and interpretable approaches to sequence labeling problems such as named entity recognition.

Which of these words refer to a named entity?

Starting prediction server…

## I. Probabilistic Graphical Models

### Factorizing Joint Distributions

$$p(a,b)=p(a)p(ba)p(a, b) = p(a) \cdot p(b|a)$$

Notation: the shorthand *p(a)* means *p(A = a)*, that is, the probability of variable A taking value a.

$$p(a,b,c,d)=p(a)p(ba)p(c)p(da,b,c)p(a, b, c, d) = p(a) \cdot p(b|a) \cdot p(c) \cdot p(d|a, b, c)$$

### Directed Acyclic Graphs

Sampled Population

## II. Hidden Markov Models

### The Hidden Layer

$$p({s}_{i}{s}_{1},{s}_{2},,{s}_{i1})=p({s}_{i}{s}_{i1})\text{for all }i{2,,N}p(s_i | s_1, s_2,…, s_{i-1}) = p(s_i | s_{i-1}) \\ \footnotesize\textrm{for all $i\,{\scriptscriptstyle \in}\,\{2,…, N\}$}$$

$$p({s}_{i}{s}_{i1})=p({s}_{i+1}{s}_{i})\text{for all }i{2,,N1}p(s_i | s_{i-1}) = p(s_{i+1} | s_i) \\ \footnotesize\textrm{for all $i\,{\scriptscriptstyle \in}\,\{2,…, N-1\}$}$$

### The Observed Layer

$$p({o}_{i}{s}_{1},{s}_{2},,{s}_{N})=p({o}_{i}{s}_{i})\text{for all }i{1,2,,N}p(o_i | s_1, s_2,…, s_N) = p(o_i | s_i) \\ \footnotesize\textrm{for all $i\,{\scriptscriptstyle \in}\,\{1, 2,…, N\}$}$$

### Representing Named Entities

Representation: rather than labeling each node using the name of the variable it represents (X₁, Y₁) as we have until this point, we'll instead display the value of that variable (“O”, “Great”). This helps make the graphs easier to read.

### Training

### Inference

### Results

Name tag predictions by HMM:

Starting prediction server…

Accuracy

90.1%

Precision

64.2%

Recall

55.8%

F₁ Score

59.7%

Precision

| Tag | No OOV | 1+ OOV |
| --- | --- | --- |
| ORG | 0.8 | 0.21 |
| PER | 0.85 | 0.62 |
| LOC | 0.87 | 0.06 |
| MISC | 0.78 | 0.12 |
| ALL | 0.84 | 0.39 |

Recall

| Tag | No OOV | 1+ OOV |
| --- | --- | --- |
| ORG | 0.64 | 0.33 |
| PER | 0.58 | 0.59 |
| LOC | 0.71 | 0.05 |
| MISC | 0.54 | 0.06 |
| ALL | 0.64 | 0.41 |

### Limitations

| Tag | Entity Length 1 | Entity Length 2 | Entity Length 3 | Entity Length 4 | Entity Length 5 |
| --- | --- | --- | --- | --- | --- |
| ORG | 0.61 | 0.68 | 0.3 | 0.12 | 0.28 |
| PER | 0.96 | 0.67 | 0.7 |  |  |
| LOC | 0.88 | 0.36 |  |  |  |
| MISC | 0.9 | 0.46 | 0.24 | 0.12 |  |
| ALL | 0.77 | 0.61 | 0.31 | 0.12 | 0.29 |

| Entity Length | OOV Rate 0 | OOV Rate 0.2 | OOV Rate 0.25 | OOV Rate 0.33 | OOV Rate 0.4 | OOV Rate 0.5 | OOV Rate 0.6 | OOV Rate 0.66 | OOV Rate 0.75 | OOV Rate 0.8 | OOV Rate 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.86 |  |  |  |  |  |  |  |  |  | 0.3 |
| 2 | 0.8 |  |  |  |  | 0.5 |  |  |  |  | 0.52 |
| 3 | 0.78 |  |  | 0.15 |  |  |  | 0.06 |  |  | 0 |
| 4 | 0.42 |  | 0.22 |  |  | 0 |  |  | 0 |  | 0 |
| 5 | 0.67 |  |  |  | 0.22 |  | 0 |  |  | 0.14 |  |

## III. Maximum Entropy Markov Models

### Discriminative Structure

### Word Features

$$b({o}_{t})={\begin{matrix} \text{1 if }{o}_{t}\text{ has shape “Xxx”} \\ \text{0 otherwise} \end{matrix}b(o_t) = \begin{cases} \textrm{1 if $o_t$ has shape “Xxx”} \\ \textrm{0 otherwise} \end{cases}$$

$${f}_{b,s}({o}_{t},{s}_{t})={\begin{matrix} \text{1 if }b({o}_{t})=1\text{ and }{s}_{t}=s \\ \text{0 otherwise} \end{matrix}f_{\langle b, s \rangle}(o_t, s_t) = \begin{cases} \textrm{1 if $b(o_t) = 1$ and $s_t = s$} \\ \textrm{0 otherwise} \end{cases}$$

### State Transitions

$${p}_{s}(so)=\frac{1}{Z(o,s)}\exp (\underset{a}{}{\lambda}_{a}\text{}{f}_{a}(o,s))p_{s\prime}(s|o) = \frac{1}{Z(o, s\prime)} \exp\left(\sum_{a}{\lambda_a \, f_a(o, s)}\right)$$

Calculating p<sub>O</sub>(B-LOC | “UK”)

With weights retrieved from a MEMM trained on CoNLL-2003 data. Numbers are rounded to 3 decimal places for clarity.

| Feature-State Pair (a) | λ<sub>a</sub> | f<sub>a</sub> | λ<sub>a</sub>f<sub>a</sub> |
| --- | --- | --- | --- |

p<sub>O</sub>(B-LOC | “UK”) = e<sup>SUM(λ<sub>a</sub>f<sub>a</sub>)</sup> / Z  
≈ e<sup>0</sup> / 1  
≈ 0

### Training & Inference

### Results

Most Informative Features when Previous State is 

| Current Word Feature | Current State | Weight |
| --- | --- | --- |
| word='germany' | B-LOC | 11.492 |
| word='van' | B-PER | 8.972 |
| word='wall' | B-ORG | 8.525 |
| word='della' | B-PER | 7.86 |
| lowercase='della' | B-PER | 7.86 |
| is\_not\_title\_case | B-PER | -6.949 |
| word='de' | B-PER | 6.781 |
| shape='X.X.' | O | -6.713 |
| shape='xxxx' | B-ORG | -6.642 |
| word='CLINTON' | B-ORG | 6.456 |

Name tag predictions by MEMM:

Starting prediction server…

Accuracy

93.1%

Precision

72.9%

Recall

63.5%

F₁ Score

67.9%

Precision

| Tag | No OOV | 1+ OOV |
| --- | --- | --- |
| ORG | 0.81 | 0.36 |
| PER | 0.82 | 0.8 |
| LOC | 0.82 | 0.17 |
| MISC | 0.74 | 0.14 |
| ALL | 0.8 | 0.54 |

Recall

| Tag | No OOV | 1+ OOV |
| --- | --- | --- |
| ORG | 0.68 | 0.12 |
| PER | 0.72 | 0.57 |
| LOC | 0.89 | 0.29 |
| MISC | 0.78 | 0.02 |
| ALL | 0.79 | 0.37 |

### Advantage Over HMMs

| Entity Length | OOV Rate 0 | OOV Rate 0.2 | OOV Rate 0.25 | OOV Rate 0.33 | OOV Rate 0.4 | OOV Rate 0.5 | OOV Rate 0.6 | OOV Rate 0.66 | OOV Rate 0.75 | OOV Rate 0.8 | OOV Rate 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.83 |  |  |  |  |  |  |  |  |  | 0.34 |
| 2 | 0.76 |  |  |  |  | 0.72 |  |  |  |  | 0.55 |
| 3 | 0.6 |  |  | 0.56 |  |  |  | 0.59 |  |  | 0.5 |
| 4 | 0.16 |  | 0.2 |  |  | 0 |  |  |  |  |  |
| 5 | 0.6 | 0.5 |  |  |  |  |  |  |  |  |  |

| Tag | Entity Length 1 | Entity Length 2 | Entity Length 3 | Entity Length 4 | Entity Length 5 |
| --- | --- | --- | --- | --- | --- |
| ORG | 0.76 | 0.69 | 0.84 | 0.36 | 0.8 |
| PER | 0.59 | 0.91 | 0.66 | 0.25 |  |
| LOC | 0.8 | 0.33 | 0.35 | 0 |  |
| MISC | 0.82 | 0.57 | 0.29 | 0.18 |  |
| ALL | 0.77 | 0.7 | 0.58 | 0.16 | 0.43 |

### Label Bias Problem

## IV. Conditional Random Fields

### Markov Random Fields

$$\varphi(X,Y)={\begin{matrix} \text{3 if }X=1\text{ and }Y=1 \\ \text{2 if }X=0\text{ and }Y=0 \\ \text{1 otherwise} \end{matrix}\phi(X, Y) = \begin{cases} \textrm{3 if $X = 1$ and $Y = 1$} \\ \textrm{2 if $X = 0$ and $Y = 0$} \\ \textrm{1 otherwise} \end{cases}$$

$$p(A,B,C)=\frac{1}{Z}\text{}\varphi(A,B)\text{}\varphi(B,C)\text{}\varphi(C,A)\text{where Z is a normalization factor}p(A,B,C) = \frac{1}{Z}\,\phi(A,B)\,\phi(B,C)\,\phi(C,A) \\ \footnotesize\textrm{where Z is a normalization factor}$$

$$p({x}_{1},{x}_{2},)=\frac{1}{Z}\underset{c\text{}\text{}C}{}{\varphi}_{c}({x}_{c})\text{where Z is a normalization factor}\text{and C is the set of cliques in }\mathcal{G}p(x_1, x_2,…) = \frac{1}{Z}\prod_{c\,{\scriptscriptstyle \in}\,C}{\phi_c(x_c)} \\ \footnotesize\textrm{where Z is a normalization factor} \\ \footnotesize\textrm{and C is the set of cliques in $\mathcal{G}$}$$

Calculating p(A, B, C, D, E, F)

p(1, 1, 1, 0, 0, 0)  
=1/Z  
⨯ ɸ<sub><sub>ABC</sub></sub>(1, 1, 1)  
⨯ ɸ<sub><sub>AB</sub></sub>(1, 1)  
⨯ ɸ<sub><sub>BC</sub></sub>(1, 1)  
⨯ ɸ<sub><sub>AC</sub></sub>(1, 1)  
⨯ ɸ<sub><sub>CD</sub></sub>(1, 0)  
⨯ ɸ<sub><sub>DEF</sub></sub>(0, 0, 0)  
⨯ ɸ<sub><sub>DE</sub></sub>(0, 0)  
⨯ ɸ<sub><sub>EF</sub></sub>(0, 0)  
⨯ ɸ<sub><sub>DF</sub></sub>(0, 0)  
=1 / 28,915  
⨯ 3  
⨯ 3  
⨯ 3  
⨯ 3  
⨯ 1  
⨯ 2  
⨯ 2  
⨯ 2  
⨯ 2  
0.0448

ɸ<sub><sub>ABC</sub></sub> = ɸ<sub><sub>AB</sub></sub> = ɸ(x) = 3 if x<sub>1</sub> = x<sub>2</sub> = … = 1  
2 if x<sub>1</sub> = x<sub>2</sub> = … = 0  
1 otherwise

### Conditional Form

$$p(yx)=\frac{1}{Z(x)}\underset{c\text{}\text{}C}{}{\varphi}_{c}({y}_{c},{x}_{c})\text{where Z is a normalization factor}\text{and C is the set of cliques in the}\text{graph }\mathcal{G}\text{ representing the labels }yp(y|x) = \frac{1}{Z(x)}\prod_{c\,{\scriptscriptstyle \in}\,C}{\phi_c(y_c, x_c)} \\ \footnotesize\textrm{where Z is a normalization factor} \\ \footnotesize\textrm{and C is the set of cliques in the} \\ \footnotesize\textrm{graph $\mathcal{G}$ representing the labels $y$}$$
Linear-chain CRF where the hidden layer depends on the current, previous, and future observations. 

### Exponential Factors

$${\varphi}_{c}({y}_{c},{x}_{c})=\exp (\underset{a}{}{\lambda}_{a}\text{}{f}_{a}({y}_{c},{x}_{c}))\text{where }{f}_{a}\text{ is a feature function defined for clique }c\text{and }{\lambda}_{a}\text{ is the weight parameter for }{f}_{a}\phi_c(y_c, x_c) = \exp\left(\sum_{a}{\lambda_a \, f_a(y_c, x_c)}\right) \\ \footnotesize\textrm{where $f_a$ is a feature function defined for clique $c$} \\ \footnotesize\textrm{and $\lambda_a$ is the weight parameter for $f_a$}$$

## References

1. **MUC-7 Named Entity Task Definition (Version 3.5)** [PDF](https://aclanthology.org/M98-1028.pdf "")  
Nancy Chinchor. 1998. In Seventh Message Understanding Conference (MUC-7).
2. **Probabilistic Graphical Models: Principles and Techniques - Adaptive Computation and Machine Learning**  
Daphne Koller and Nir Friedman. 2009. The MIT Press.
3. **Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference**  
Judea Pearl. 1988. Morgan Kaufmann Publishers Inc, San Francisco, CA, USA.
4. **Genes, Themes and Microarrays: Using Information Retrieval for Large-Scale Gene Analysis**  
Hagit Shatkay, Stephen Edwards, W John Wilbur, and Mark Boguski. 2000. In Proceedings of the International Conference on Intelligent Systems for Molecular Biology, 317–328.
5. **Information Extraction Using Hidden Markov Models**  
Timothy Robert Leek. 1997. Master’s Thesis, UC San Diego.
6. **Information Extraction with HMMs and Shrinkage** [PDF](https://www.aaai.org/Papers/Workshops/1999/WS-99-11/WS99-11-006.pdf "")  
Dayne Freitag and Andrew McCallum. 1999. In Papers from the AAAI-99 Workshop on Machine Learning for Information Extraction (AAAI Techinical Report WS-99-11), 31–36.
7. **A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition**  
Lawrence R Rabiner. 1989. Proceedings of the IEEE 77, 2: 257–286. https://doi.org/10.1109/5.18626
8. **An Algorithm that Learns What’s in a Name** [PDF](https://link.springer.com/content/pdf/10.1023/A:1007558221122.pdf "")  
Daniel M. Bikel, Richard Schwartz, and Ralph M. Weischedel. 1999. Machine Learning 34, 1: 211–231. https://doi.org/10.1023/A:1007558221122
9. **Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding Algorithm**  
A. Viterbi. 1967. IEEE Transactions on Information Theory 13, 2: 260–269.
10. **Appendix A.4 — Decoding: The Viterbi Algorithm** [PDF](https://web.stanford.edu/~jurafsky/slp3/A.pdf "")  
Daniel Jurafsky and James H. Martin. 2021. In Speech and Language Processing. 8–10.
11. **Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition** [PDF](https://aclanthology.org/W03-0419.pdf "")  
Erik F. Tjong Kim Sang and Fien De Meulder. 2003. In Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003, 142–147.
12. **Maximum Entropy Markov Models for Information Extraction and Segmentation** [PDF](http://www.ai.mit.edu/courses/6.891-nlp/READINGS/maxent.pdf "")  
Andrew McCallum, Dayne Freitag, and Fernando C. N. Pereira. 2000. In Proceedings of the Seventeenth International Conference on Machine Learning (ICML ’00), 591–598.
13. **Maximum Entropy Models for Antibody Diversity** [Link](https://www.pnas.org/doi/abs/10.1073/pnas.1001705107 "")  
Thierry Mora, Aleksandra M. Walczak, William Bialek, and Curtis G. Callan. 2010. Proceedings of the National Academy of Sciences 107, 12: 5405–5410. https://doi.org/10.1073/pnas.1001705107
14. **Human Behavior Modeling with Maximum Entropy Inverse Optimal Control** [PDF](https://www.aaai.org/Papers/Symposia/Spring/2009/SS-09-04/SS09-04-016.pdf "")  
Brian Ziebart, Andrew Maas, J. Bagnell, and Anind Dey. 2009. In Papers from the 2009 AAAI Spring Symposium, Technical Report SS-09-04, Stanford, California, USA, 92–97.
15. **On Discriminative vs. Generative Classifiers: A comparison of logistic regression and naive Bayes** [PDF](https://proceedings.neurips.cc/paper/2001/file/7b7a53e239400a13bd6be6c91c4f6c4e-Paper.pdf "")  
Andrew Ng and Michael Jordan. 2001. In Advances in Neural Information Processing Systems.
16. **Inducing Features of Random Fields**  
S. Della Pietra, V. Della Pietra, and J. Lafferty. 1997. IEEE Transactions on Pattern Analysis and Machine Intelligence 19, 4: 380–393. https://doi.org/ 10.1109/34.588021
17. **Une Approche théorique de l’Apprentissage Connexionniste: Applications à la Reconnaissance de la Parole**  
Léon Bottou. 1991. Université de Paris X.
18. **Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data** [PDF](http://www.aladdin.cs.cmu.edu/papers/pdfs/y2001/crf.pdf "")  
John D. Lafferty, Andrew McCallum, and Fernando C. N. Pereira. 2001. In Proceedings of the Eighteenth International Conference on Machine Learning (ICML ’01), 282–289.
19. **The Label Bias Problem** [Link](https://awni.github.io/label-bias/ "")  
Awni Hannun. 2019. Awni Hannun — Writing About Machine Learning.
20. **Discriminative Probabilistic Models for Relational Data** [Link](https://arxiv.org/abs/1301.0604 "")  
Ben Taskar, Pieter Abbeel, and Daphne Koller. 2013. https://doi.org/10.48550/ARXIV.1301.0604
21. **Accurate Information Extraction from Research Papers using Conditional Random Fields** [Link](https://aclanthology.org/N04-1042 "")  
Fuchun Peng and Andrew McCallum. 2004. In Proceedings of the Human Language Technology Conference of the North American Chapter of the Association for Computational Linguistics: HLT-NAACL 2004, 329–336.
22. **Discriminative Fields for Modeling Spatial Dependencies in Natural Images** [PDF](https://proceedings.neurips.cc/paper/2003/file/92049debbe566ca5782a3045cf300a3c-Paper.pdf "")  
Sanjiv Kumar and Martial Hebert. 2003. In Advances in Neural Information Processing Systems.
23. **Multiscale Conditional Random Fields for Image Labeling** [Link](https://ieeexplore.ieee.org/document/1315232 "")  
Xuming He, R.S. Zemel, and M.A. Carreira-Perpinan. 2004. In Proceedings of the 2004 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2004, CVPR 2004, II–II. https://doi.org/10.1109/CVPR.2004.1315232
24. **Conditional Random Fields as Recurrent Neural Networks** [Link](https://ieeexplore.ieee.org/document/7410536 "")  
Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes, Vibhav Vineet, Zhizhong Su, Dalong Du, Chang Huang, and Philip H. S. Torr. 2015. In 2015 IEEE International Conference on Computer Vision (ICCV). https://doi.org/10.1109/iccv.2015.179
25. **Convolutional CRFs for Semantic Segmentation** [Link](https://arxiv.org/abs/1805.04777 "")  
Marvin T. T. Teichmann and Roberto Cipolla. 2018. https://doi.org/10.48550/arxiv.1805.04777
26. **RNA Secondary Structural Alignment with Conditional Random Fields** [Link](https://academic.oup.com/bioinformatics/article/21/suppl_2/ii237/227803?login=false "")  
Kengo Sato and Yasubumi Sakakibara. 2005. Bioinformatics 21: ii237–ii242. https://doi.org/10.1093/bioinformatics/bti1139
27. **Protein Fold Recognition Using Segmentation Conditional Random Fields (SCRFs)**  
Yan Liu, Jaime Carbonell, Peter Weigele, and Vanathi Gopalakrishnan. 2006. J. Comput. Biol. 13, 2: 394–406.
28. **Introduction to Markov Random Fields**  
Andrew Blake and Pushmeet Kohli. 2011. In Markov Random Fields for Vision and Image Processing. The MIT Press.
