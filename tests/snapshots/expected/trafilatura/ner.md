Learning What’s in a Name with Graphical Models — Gradient

Learning What’s in a Name with Graphical Models
An overview of the structure and statistical assumptions behind linear-chain graphical models — effective and interpretable approaches to sequence labeling problems such as named entity recognition.
Starting prediction server…
“The UK” alone is a country, but “The UK Department of Transport” is an organization within said country. In a named entity recognition (NER) task, where we want to label each word with a name tag (organization/person/location/other/not a name) [1], how can a computer model know one from the other?
In such cases, contextual information is key. In the second example, the fact that “The UK” is followed by “Department” is compelling evidence that when taken together the phrase refers to an organization. Sequence models — machine learning systems designed to take sequences of data as input, can recognize and put such relationships to productive use. Rather than making isolated predictions based on individual words or word groups, they take the given sequence as a combined unit, model the dependencies between words in that sequences, and depending on the problem can return the most likely label sequence.
In this article, we’ll explore three sequence models that are remarkably successful at NER: Hidden Markov Models (HMMs), Maximum-Entropy Markov Models (MEMMs), and Conditional Random Fields (CRFs). All three are probabilistic graphical models, which we’ll cover in the next section.
Graphical modeling is a robust framework for representing probabilistic models. Complex multivariate probability distributions can be expressed with compact graphs that are vastly easier to understand and interpret.
Let’s start with a simple example with two random variables, and . Assume that is conditionally dependent on . Through a canonical application of the chain rule, the joint distribution of and is:
Notation: the shorthand p(a) means p(A = a), that is, the probability of variable A taking value a.
This is a simple enough example, with 2 factors in the right hand side. Add more variables, however, and the result can get messy fast. To see this, assume that there are two more variables, and , and that is conditionally dependent on , , and . The factorization becomes:
The relationship between variables is more opaque, hidden behind second-order dependencies. For example, while it’s clear that is directly dependent on , we may miss the fact that there is another, second-order dependency between the two ( is dependent on , which in turn is dependent on ).
Directed Acyclic Graphs, or DAGs, offer a natural remedy to this problem. Each factor in the equation can be represented by a node. An arrow indicates conditional dependence. The resulting graph would look like:
With this graph, it’s easier to construct a generative story of how , , and are sampled. The process proceeds in topological order, for example → → → , to ensure that all dependencies have been resolved by the time each variable is sampled.
Below is what a sampled population of the given distributions would look like. For the sake of demonstration, many distribution parameters are modifiable — in reality these are the quantities that need to be learned from training data.
Sampled Population
For more detailed accounts of probabilistic graphical models, consider reading the textbooks Probabilistic Graphical Models: Principles and Techniques by Daphne Koller and Nir Friedman [2] and Probabilistic Reasoning in Intelligent Systems by Judea Pearl [3].
Hidden Markov Models (HMMs) are an early class of probabilistic graphical models representing partially hidden (unobserved) sequences of events. Structurally, they are built with two main layers, one hidden () and one observed ():
HMMs have been successfully applied to a wide range of problems, including gene analysis [4], information extraction [5, 6], speech recognition [7], and named entity recognition [8].
The hidden layer is assumed to be a Markov process: a chain of events in which each event’s probability depends only on the state of the preceding event. More formally, given a sequence of random events , ,…, , the Markov assumption holds that:
In a graph, this translates to a linear chain of events where each event has one arrow pointing towards it (except for the first event) and one pointing away from it (except for the last):
A second assumption that HMMs make is time-homogeneity: that the probability of transition from one event's state to the next is constant over time. In formal terms:
is called the transition probability and is one of the two key parameters to be learned during training.
The assumptions about the hidden layer — Markov and time-homogeneity — hold up in various time-based systems where the hidden, unobserved events occur sequentially, one after the other. Together, they meaningfully reduce the computational complexity of both learning and inference.
The hidden and observed layer are connected via a one-to-one mapping relationship. The probability of each observation is assumed to depend only on the state of the hidden event at the same time step. Given a sequence of hidden events , ,…, and observed events , ,…, we have:
In a graph, this one-to-one relationship looks like:
The conditional probability , called the emission probability, is also assumed to be time-homogenous, further reducing the model's complexity. It is the second key parameter to be learned, alongside the transition probability.
HMMs’ chain structure is particularly useful in sequence labeling problems like NER. For each input text sequence, the observed layer represents known word tokens, while the hidden layer contains their respective name tags:
Representation: rather than labeling each node using the name of the variable it represents (X₁, Y₁) as we have until this point, we'll instead display the value of that variable (“O”, “Great”). This helps make the graphs easier to read.
There are 9 possible name tags. Each, apart from the “O” tag, has either a B- (beginning) or I- (inside) prefix, to eliminate confusion about when an entity stops and the next one begins.
Between any two consecutive hidden states, there are 9² = 81 possible transitions. Each transition has its own probability, :
Higher opacity indicates higher relative probability.
In the observed layer, each node can have any value from the vocabulary, whose size ranges anywhere from the thousands to the hundreds of thousands. The vocabulary created for the HMM in this article contains 23,622 tokens. Let N be the number of tokens in the vocabulary. The number of possible emission probabilities is 9N ().
Higher opacity indicates higher relative probability.
There are three sets of parameters to be learned during training: the transition, emission, and start probabilities. All can be computed as normalized rates of occurrence from the training data.
For example, to get the transition probability from state “O ” to state “B-LOC”, , we need two numbers: the number of times state “O” is followed by any other state (that is, it isn't the last state in the sequence), as , and the number of times state “O” is followed by state “B-LOC”, as . The desired transition probability is . The same calculation can be done for each of the remaining probabilities.
In the context of HMMs, inference involves answering useful questions about hidden states given observed values, or about missing values given a partially observed sequence. In NER, we are focused on the first type of inference. Specifically, we want to perform maximum a posteriori (MAP) inference to identify the most likely state sequence conditioned on observed values.
There is usually an intractably large number of candidate state sequences. For any two consecutive states, there are 81 potential transition paths. For three states there are 82² paths. This number continues to grow exponentially as the number of states increases.
Luckily, there is an efficient dynamic algorithm that returns the most likely path with relatively low computational complexity: the Viterbi algorithm [9]. It moves through the input sequence from left to right, at each step identifying and saving the most likely path in a trellis-shaped memory structure. For more details, refer to the excellent description of the Viterbi algorithm in the book Speech and Language Processing by Jurafsky & Martin [10].
An HMM with the structure outlined above was trained on the CoNLL-2003 English dataset [11]. The train set contains 14,987 sentences and a total of 203,621 word tokens. Here's the model in action:
Starting prediction server…
Evaluated against a test set, the model achieves satisfactory per-word accuracy:
Accuracy
90.1%
However, precision and recall — calculated per entity [11] — are decidedly low:
Precision
64.2%
Recall
55.8%
F₁ Score
59.7%
These metrics are lower than per-word accuracy because they are entity-level evaluations that count only exact matches as true positives. Long, multi-word entities are considered incorrect if one or more of their constituent words are misidentified, in effect ignoring the other correctly identified words in the entity.
A closer look at the results reveals a discrepancy between entities with known words and those with at least one out-of-vocabulary (OOV) words:
Precision
| Tag | No OOV | 1+ OOV |
|---|---|---|
| ORG | 0.8 | 0.21 |
| PER | 0.85 | 0.62 |
| LOC | 0.87 | 0.06 |
| MISC | 0.78 | 0.12 |
| ALL | 0.84 | 0.39 |
| Tag | No OOV | 1+ OOV |
Recall
| Tag | No OOV | 1+ OOV |
|---|---|---|
| ORG | 0.64 | 0.33 |
| PER | 0.58 | 0.59 |
| LOC | 0.71 | 0.05 |
| MISC | 0.54 | 0.06 |
| ALL | 0.64 | 0.41 |
| Tag | No OOV | 1+ OOV |
This makes sense: we expect the model to perform worse on tokens that it wasn't trained on. And because the CoNLL-2003 train set has a large number of OOV tokens — 80.6% of all test entities contain at least one OOV token — across-the-board precision and recall are heavily impacted.
HMMs are by no means perfect candidates for NER or text labeling problems in general, for at least three reasons. The first two have to do with the statistical assumptions that underlie HMMs’ chain structure, while the third relates to their sole reliance on word identity in the observed layer.
First is the Markov assumption. We don't compose words and their respective labels (“Noun”, “Adjective”, “B-ORG”, “I-ORG”) in a tidy, unidirectional manner. A word may link to another many places before or many more further down the line. In “South African Breweries Ltd”, both “South African” and “Ltd” provide crucial context to “Breweries”, which would otherwise register as not a name. HMMs fail to capture these tangled interdependencies, instead assuming that information passes from left to right in a single, neat chain.
An indication, although it is by no means proof, of such limitation lies in the fast deterioration of the precision score as the entity length — counted as the number of tokens inside each entity — increases (the recall scores are much more noisy and thus less conclusive):
| Tag | Entity Length 1 | Entity Length 2 | Entity Length 3 | Entity Length 4 | Entity Length 5 |
|---|---|---|---|---|---|
| ORG | 0.61 | 0.68 | 0.3 | 0.12 | 0.28 |
| PER | 0.96 | 0.67 | 0.7 | ||
| LOC | 0.88 | 0.36 | |||
| MISC | 0.9 | 0.46 | 0.24 | 0.12 | |
| ALL | 0.77 | 0.61 | 0.31 | 0.12 | 0.29 |
| Tag | 1 | 2 | 3 | 4 | 5 |
The second concern has to do with the emissions assumption. When composing sentences, we don’t go about forming a chain of labels (such as “B-ORG” – “I-ORG” – “I-ORG” – “O”…) before generating words from each label in that chain.
Semantic and grammatical rules may restrict the range of words that can appear in any given observation, but those restrictions are far from the strong emissions assumption made by HMMs. Beyond the current word label, there is a whole host of other factors that together help determine which words are chosen. Additionally, while there is a cognizable link between part-of-speech tags and the words that are chosen, with name tags that link is less clear.
The third concern relates to HMMs sole reliance on word identity in the observed layer. There are various word features that can provide additional information on the hidden label. Capitalization is a strong signal of named entities. Word shape may help account for common name patterns. HMMs take none of these features into account (in fact, they can’t: their generative chain structure requires that all observations be independent of each other).
With the only information available being word identities, HMMs end up stumbling over out-of-vocabulary words. While the models can wring bits of useful information out of nearby predicted name tags — the Viterbi algorithm maximizes likelihood over the whole sequence — the results are nonetheless discouraging:
| Entity Length | OOV Rate 0 | OOV Rate 0.2 | OOV Rate 0.25 | OOV Rate 0.33 | OOV Rate 0.4 | OOV Rate 0.5 | OOV Rate 0.6 | OOV Rate 0.66 | OOV Rate 0.75 | OOV Rate 0.8 | OOV Rate 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.86 | 0.3 | |||||||||
| 2 | 0.8 | 0.5 | 0.52 | ||||||||
| 3 | 0.78 | 0.15 | 0.06 | 0 | |||||||
| 4 | 0.42 | 0.22 | 0 | 0 | 0 | ||||||
| 5 | 0.67 | 0.22 | 0 | 0.14 | |||||||
| Entity Length | 0 | 0.2 | 0.25 | 0.33 | 0.4 | 0.5 | 0.6 | 0.66 | 0.75 | 0.8 | 1 |
The concerns detailed above are direct consequences of HMMs’ generative graph structure. Below, we’ll consider another class of probabilistic graphical models with a different, discriminative structure that will hopefully address some of those concerns and deliver better performance on long, OOV-heavy entities.
Maximum Entropy Markov Models (MEMMs) [12] resolve some of the concerns we had with HMMs by way of a simple yet meaningful modification to their graphical structure :
The arrows connecting observations with their respective states have switched directions. We'll discuss the radical implications of such a change in the sections below.
Similar to HMMs, MEMMs have been successfully applied to a wide range of sequence modeling problems [12, 13, 14].
There are two main approaches to building classification models: generative and discriminative [15]. Suppose there is a system with two variables, and . We want to make predictions about based on observations on . A generative classifier would do that by first learning the prior distribution and then applying Bayes' rule to find the posterior . This can be thought of as reconstructing the process that generated the observed data. A discriminative classifier, on the other hand, would model the posterior directly based on training data.
HMMs are generative classifiers, while MEMMs are discriminative. The former are generative because they model the joint distribution over both observations and hidden states (as a product of transition and emission probabilities) before using that joint distribution to find the most likely state sequence given observations (or solve some other inference problem). MEMMs, on the other hand, directly model the conditional probabilities without any intermediary.
Notably, MEMMs’ discriminative structure allows them to model overlapping word features. Two features can overlap when they contain the same or similar pieces of information, like word shape (“Xxx”) and capitalization. HMMs don’t allow overlapping features, since as a result of their generative structure they require that all events in the observation layer be independent of one another. MEMMs, on the other hand, are discriminative and able to relax the independence requirement, so they can use arbitrary overlapping word features [12].
Common practice is to use binary features, such as:
These features are then paired with the current state to form feature-state pairs :
Feature-state pairs provide useful additional information how which features and states go together and which don’t. For example, we can expect pairs like "is_capitalized" + “B-ORG” to occur together frequently, capturing the fact that in English named entities are often capitalized.
MEMMs’ state transition distributions have exponential form and contain a weighted sum of all feature-state pairs:
where and are the previous and current state, is the current observation, is a feature-state pair, is the learned weight for , and is a normalizing term to make the distribution sum to one across all next states .
Calculating pO(B-LOC | “UK”)
With weights retrieved from a MEMM trained on CoNLL-2003 data. Numbers are rounded to 3 decimal places for clarity.| Feature-State Pair (a) | λa | fa | λafa |
|---|
pO(B-LOC | “UK”) = eSUM(λafa) / Z
≈ e0 / 1
≈ 0
Those familiar with neural networks will recognize that the function above is a softmax. Its exponential form is a result of the core principle of maximum entropy that underlies MEMMs’ statistical structure and gives them their name. Maximum entropy states that the model that best represents our knowledge about a system is one that makes the fewest possible assumptions except for certain constraints derived from prior data from that system [12, 16].
The training step involves learning the weights that satisfy MEMMs’ maximum entropy constraint [12]. Learning is done through Generalized Iterative Scaling, which iteratively updates the values in order to nudge the expected value of all features closer to their train set average. Convergence at a global optimum is guaranteed given the exponential form of the transition distribution.
As with HMMs, the Viterbi algorithm makes MAP inference tractable [12, 9]. The variable transition probability takes the place of HMMs’ fixed transition and emission probabilities.
A MEMM was trained on the CoNLL-2003 English dataset [11]. In addition to word identity, features used for training include the word’s lowercase version (“Algeria” → “algeria”), shape (“Xxxx”), whether it’s in title/upper case, and whether it contains only digits.
A list of the most informative features — those with the largest absolute weights — offers valuable insights into how the model found and remembers linguistic patterns:
Most Informative Features when Previous State is
| Current Word Feature | Current State | Weight |
|---|---|---|
| word='germany' | B-LOC | 11.492 |
| word='van' | B-PER | 8.972 |
| word='wall' | B-ORG | 8.525 |
| word='della' | B-PER | 7.86 |
| lowercase='della' | B-PER | 7.86 |
| is_not_title_case | B-PER | -6.949 |
| word='de' | B-PER | 6.781 |
| shape='X.X.' | O | -6.713 |
| shape='xxxx' | B-ORG | -6.642 |
| word='CLINTON' | B-ORG | 6.456 |
Many of these features are word identities. This makes intuitive sense: certain words, like “Germany”, are almost always used as names irrespective of what comes before or after them.
Other features relate to established linguistic patterns. For example, if the current word has shape “X.X.”, such as “U.S.” and “U.N.”, it’s unlikely to have the “O” tag — the feature-state pair’s weight is a large negative number. This means the word is likely a named entity, most probably two-letter initialisms.
Here’s a live version of the trained model:
Starting prediction server…
The model has better performance than its HMM counterpart. Per-word accuracy is higher than the HMM’s 90.1%:
Accuracy
93.1%
Per-entity precision and recall are notably higher, up from the HMM’s 64.2% and 55.8%, respectively:
Precision
72.9%
Recall
63.5%
F₁ Score
67.9%
A large part of the performance boost is attributable to higher precision on entities with at least one OOV word:
Precision
| Tag | No OOV | 1+ OOV |
|---|---|---|
| ORG | 0.81 | 0.36 |
| PER | 0.82 | 0.8 |
| LOC | 0.82 | 0.17 |
| MISC | 0.74 | 0.14 |
| ALL | 0.8 | 0.54 |
| Tag | No OOV | 1+ OOV |
Recall
| Tag | No OOV | 1+ OOV |
|---|---|---|
| ORG | 0.68 | 0.12 |
| PER | 0.72 | 0.57 |
| LOC | 0.89 | 0.29 |
| MISC | 0.78 | 0.02 |
| ALL | 0.79 | 0.37 |
| Tag | No OOV | 1+ OOV |
The ability to model word features allows MEMMs to fare better with OOV-dense name entities than HMMs. Faced with words that they have never seen before during training, these models can easily stumble. Word identity alone provides no useful information. In those cases, derived features such as word shape and capitalization can function as imperfect yet doubtlessly helpful proxies for word identity, allowing MEMMs to make better guesses at the name tag, resulting higher precision and recall scores:
| Entity Length | OOV Rate 0 | OOV Rate 0.2 | OOV Rate 0.25 | OOV Rate 0.33 | OOV Rate 0.4 | OOV Rate 0.5 | OOV Rate 0.6 | OOV Rate 0.66 | OOV Rate 0.75 | OOV Rate 0.8 | OOV Rate 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.83 | 0.34 | |||||||||
| 2 | 0.76 | 0.72 | 0.55 | ||||||||
| 3 | 0.6 | 0.56 | 0.59 | 0.5 | |||||||
| 4 | 0.16 | 0.2 | 0 | ||||||||
| 5 | 0.6 | 0.5 | |||||||||
| Entity Length | 0 | 0.2 | 0.25 | 0.33 | 0.4 | 0.5 | 0.6 | 0.66 | 0.75 | 0.8 | 1 |
With stronger predictive power on OOV words, we can additionally expect better performance on long, multi-word entities. That’s because OOV words are dangerous information gaps inside named entities. They’re easy to misclassify, and when they are the entire entity prediction is counted as incorrect. MEMMs are able to fill those gaps to an extent by using word features. As a result, we don’t see as drastic of a performance deterioration for longer entities as observed with the HMM:
| Tag | Entity Length 1 | Entity Length 2 | Entity Length 3 | Entity Length 4 | Entity Length 5 |
|---|---|---|---|---|---|
| ORG | 0.76 | 0.69 | 0.84 | 0.36 | 0.8 |
| PER | 0.59 | 0.91 | 0.66 | 0.25 | |
| LOC | 0.8 | 0.33 | 0.35 | 0 | |
| MISC | 0.82 | 0.57 | 0.29 | 0.18 | |
| ALL | 0.77 | 0.7 | 0.58 | 0.16 | 0.43 |
| Tag | 1 | 2 | 3 | 4 | 5 |
MEMMs’ discriminative structure confers great benefits, but there’s a downside: it makes them susceptible to the label bias problem. First recorded by Bottou [17], this problem mostly affects discriminative models, causing certain states to effectively ignore their observations, biasing predictions toward less likely transition paths. While the label bias problem doesn’t render models useless, it still has a notable effect on predictions, causing demonstrably higher error rates [18].
What’s important to know is that MEMMs fall victim to the label bias problem because they have local probability normalization. The normalization factor ensures that transition probabilities between neighboring states sum up to one. Local normalization forces every state to transfer all of its probability mass onto the next state, regardless of how likely or unlikely the current observation is. Hannun [19] provides an excellent, detailed explanation of how this happens.
We can consider getting rid of local normalization to avoid the problem. That would lead us to Conditional Random Fields — a class of globally-normalized, undirected probabilistic models, which we’ll cover next.
Conditional Random Fields (CRFs) are a class of undirected probabilistic models. They have proved to be powerful models with a wide range of applications, including text processing [18, 20, 21], image recognition [22, 23, 24, 25], and bioinformatics [26, 27].
While CRFs can have any graph structure, in this article we’ll focus on the linear-chain version:
CRFs are a type of Markov Random Fields (MRFs) — probability distributions over random variables defined by undirected graphs [28]:
Undirected graphs are appropriate for when it’s difficult or implausible to establish causal, generative relationships between random variables. Social networks are a good example of undirected relationships. We can think of , , and in the graph above as people in a simple network. and are friends and tend to share similar beliefs. The same goes for and as well as and . We might, for example, want to model how each person in the network thinks about a specific topic.
Acyclic Directed graphs fail to adequately represent the mutual belief propagation that occurs within the group. For example, we might have an edge from to but no path from back to — there will always be at least one such exclusion in an acyclic directed graph.
Rather than assuming a generative relationship between variables, MRFs model their mutual relationships with non-negative scoring functions , called factors, that assign higher scores if the variables’ values are in agreement, for example:
Unlike conditional probabilities, there is no assumed directionality in scoring functions. These functions simply return higher scores if the variables agree and lower scores if they disagree. They model pairwise correlation, not causation.
The joint probability of all variables in the graph is:
The factors promote assignments in which their constituent variables ( and in the case of ) agree with each other. The assignment 1-1-1 would receive a higher score and thus have higher probability than say 1-1-0, since there is more agreement in the former case.
More generally, MRFs are probability distributions over random variables , ,… that are defined by an undirected graph and have the form:
Calculating p(A, B, C, D, E, F)
p(1, 1, 1, 0, 0, 0)
=1/Z
⨯ ɸABC(1, 1, 1)
⨯ ɸAB(1, 1)
⨯ ɸBC(1, 1)
⨯ ɸAC(1, 1)
⨯ ɸCD(1, 0)
⨯ ɸDEF(0, 0, 0)
⨯ ɸDE(0, 0)
⨯ ɸEF(0, 0)
⨯ ɸDF(0, 0)
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
≈0.0448
ɸABC = ɸAB = ɸ…(x) = 3 if x1 = x2 = … = 1
2 if x1 = x2 = … = 0
1 otherwise
MRFs have a generalized form of which the directed models we’ve seen so far — HMMs and MEMMs — are special cases. The factors can be defined as conditional probabilities, for example , and act as the transition and emission probabilities that characterize HMMs and MEMMs.
The additional level of generality comes at a cost, however: the normalization factors are often difficult to compute. They require summing over an exponential number of potential assignments, an infeasible task if the network is large enough. Fortunately, there are configurations that can be solved using efficient decoding algorithms. That includes linear-chain CRFs, which can be decoded with the Viterbi algorithm.
CRFs are random fields globally conditioned on a set of observations [18] and have the form:
The distribution is parameterized by . When we replace all the values in the right hand side with real values, what remains has the same form as an MRF. In fact, we get a new MRF for every observation sequence .
CRFs are globally conditioned on . They directly model the probability of the label sequence — — rather than local transition/emission probabilities or .
Global conditioning on means that the hidden states can depend not only on the current observation but also any other observation in the sequence. Adding more such dependencies to the model does not increase the computational complexity of inference tasks, since we don’t have to model the marginal probabilities at train/test time.
The factors have an exponential form [18] that’s similar MEMMs’ transition function:
References
- MUC-7 Named Entity Task Definition (Version 3.5) PDF
Nancy Chinchor. 1998. In Seventh Message Understanding Conference (MUC-7). - Probabilistic Graphical Models: Principles and Techniques - Adaptive Computation and Machine Learning
Daphne Koller and Nir Friedman. 2009. The MIT Press. - Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference
Judea Pearl. 1988. Morgan Kaufmann Publishers Inc, San Francisco, CA, USA. - Genes, Themes and Microarrays: Using Information Retrieval for Large-Scale Gene Analysis
Hagit Shatkay, Stephen Edwards, W John Wilbur, and Mark Boguski. 2000. In Proceedings of the International Conference on Intelligent Systems for Molecular Biology, 317–328. - Information Extraction Using Hidden Markov Models
Timothy Robert Leek. 1997. Master’s Thesis, UC San Diego. - Information Extraction with HMMs and Shrinkage PDF
Dayne Freitag and Andrew McCallum. 1999. In Papers from the AAAI-99 Workshop on Machine Learning for Information Extraction (AAAI Techinical Report WS-99-11), 31–36. - A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition
Lawrence R Rabiner. 1989. Proceedings of the IEEE 77, 2: 257–286. https://doi.org/10.1109/5.18626 - An Algorithm that Learns What’s in a Name PDF
Daniel M. Bikel, Richard Schwartz, and Ralph M. Weischedel. 1999. Machine Learning 34, 1: 211–231. https://doi.org/10.1023/A:1007558221122 - Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding Algorithm
A. Viterbi. 1967. IEEE Transactions on Information Theory 13, 2: 260–269. - Appendix A.4 — Decoding: The Viterbi Algorithm PDF
Daniel Jurafsky and James H. Martin. 2021. In Speech and Language Processing. 8–10. - Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition PDF
Erik F. Tjong Kim Sang and Fien De Meulder. 2003. In Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003, 142–147. - Maximum Entropy Markov Models for Information Extraction and Segmentation PDF
Andrew McCallum, Dayne Freitag, and Fernando C. N. Pereira. 2000. In Proceedings of the Seventeenth International Conference on Machine Learning (ICML ’00), 591–598. - Maximum Entropy Models for Antibody Diversity Link
Thierry Mora, Aleksandra M. Walczak, William Bialek, and Curtis G. Callan. 2010. Proceedings of the National Academy of Sciences 107, 12: 5405–5410. https://doi.org/10.1073/pnas.1001705107 - Human Behavior Modeling with Maximum Entropy Inverse Optimal Control PDF
Brian Ziebart, Andrew Maas, J. Bagnell, and Anind Dey. 2009. In Papers from the 2009 AAAI Spring Symposium, Technical Report SS-09-04, Stanford, California, USA, 92–97. - On Discriminative vs. Generative Classifiers: A comparison of logistic regression and naive Bayes PDF
Andrew Ng and Michael Jordan. 2001. In Advances in Neural Information Processing Systems. - Inducing Features of Random Fields
S. Della Pietra, V. Della Pietra, and J. Lafferty. 1997. IEEE Transactions on Pattern Analysis and Machine Intelligence 19, 4: 380–393. https://doi.org/ 10.1109/34.588021 - Une Approche théorique de l’Apprentissage Connexionniste: Applications à la Reconnaissance de la Parole
Léon Bottou. 1991. Université de Paris X. - Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data PDF
John D. Lafferty, Andrew McCallum, and Fernando C. N. Pereira. 2001. In Proceedings of the Eighteenth International Conference on Machine Learning (ICML ’01), 282–289. - The Label Bias Problem Link
Awni Hannun. 2019. Awni Hannun — Writing About Machine Learning. - Discriminative Probabilistic Models for Relational Data Link
Ben Taskar, Pieter Abbeel, and Daphne Koller. 2013. https://doi.org/10.48550/ARXIV.1301.0604 - Accurate Information Extraction from Research Papers using Conditional Random Fields Link
Fuchun Peng and Andrew McCallum. 2004. In Proceedings of the Human Language Technology Conference of the North American Chapter of the Association for Computational Linguistics: HLT-NAACL 2004, 329–336. - Discriminative Fields for Modeling Spatial Dependencies in Natural Images PDF
Sanjiv Kumar and Martial Hebert. 2003. In Advances in Neural Information Processing Systems. - Multiscale Conditional Random Fields for Image Labeling Link
Xuming He, R.S. Zemel, and M.A. Carreira-Perpinan. 2004. In Proceedings of the 2004 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2004, CVPR 2004, II–II. https://doi.org/10.1109/CVPR.2004.1315232 - Conditional Random Fields as Recurrent Neural Networks Link
Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes, Vibhav Vineet, Zhizhong Su, Dalong Du, Chang Huang, and Philip H. S. Torr. 2015. In 2015 IEEE International Conference on Computer Vision (ICCV). https://doi.org/10.1109/iccv.2015.179 - Convolutional CRFs for Semantic Segmentation Link
Marvin T. T. Teichmann and Roberto Cipolla. 2018. https://doi.org/10.48550/arxiv.1805.04777 - RNA Secondary Structural Alignment with Conditional Random Fields Link
Kengo Sato and Yasubumi Sakakibara. 2005. Bioinformatics 21: ii237–ii242. https://doi.org/10.1093/bioinformatics/bti1139 - Protein Fold Recognition Using Segmentation Conditional Random Fields (SCRFs)
Yan Liu, Jaime Carbonell, Peter Weigele, and Vanathi Gopalakrishnan. 2006. J. Comput. Biol. 13, 2: 394–406. - Introduction to Markov Random Fields
Andrew Blake and Pushmeet Kohli. 2011. In Markov Random Fields for Vision and Image Processing. The MIT Press.
