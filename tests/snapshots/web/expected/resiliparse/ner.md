Learning What’s in a Name with Graphical Models — Gradient

Learning What’s in a Name with Graphical Models

Learning What’s in a Name with Graphical Models

An overview of the structure and statistical assumptions behind linear-chain graphical models — effective and interpretable approaches to sequence labeling problems such as named entity recognition.

Which of these words refer to a named entity?

Starting prediction server…

hmm
memm
crf
O
 = not a name
ORG
 = organization
PER
 = person
LOC
 = location
MISC
 = miscellaneous

“The UK” alone is a country, but “The UK Department of Transport” is an organization within said country. In a named entity recognition (NER) task, where we want to label each word with a name tag (organization/person/location/other/not a name) [1], how can a computer model know one from the other?

In such cases, contextual information is key. In the second example, the fact that “The UK” is followed by “Department” is compelling evidence that when taken together the phrase refers to an organization. Sequence models — machine learning systems designed to take sequences of data as input, can recognize and put such relationships to productive use. Rather than making isolated predictions based on individual words or word groups, they take the given sequence as a combined unit, model the dependencies between words in that sequences, and depending on the problem can return the most likely label sequence.

In this article, we’ll explore three sequence models that are remarkably successful at NER: Hidden Markov Models (HMMs), Maximum-Entropy Markov Models (MEMMs), and Conditional Random Fields (CRFs). All three are probabilistic graphical models, which we’ll cover in the next section.

I. Probabilistic Graphical Models

Graphical modeling is a robust framework for representing probabilistic models. Complex multivariate probability distributions can be expressed with compact graphs that are vastly easier to understand and interpret.

Factorizing Joint Distributions

Let’s start with a simple example with two random variables, AAA and BBB. Assume that BBB is conditionally dependent on AAA. Through a canonical application of the chain rule, the joint distribution of AAA and BBB is:

p(a,b)=p(a)⋅p(b∣a)p(a, b) = p(a) \cdot p(b|a)p(a,b)=p(a)⋅p(b∣a)

Notation: the shorthand p(a) means p(A = a), that is, the probability of variable A taking value a.

This is a simple enough example, with 2 factors in the right hand side. Add more variables, however, and the result can get messy fast. To see this, assume that there are two more variables, CCC and DDD, and that DDD is conditionally dependent on AAA, BBB, and CCC. The factorization becomes:

p(a,b,c,d)=p(a)⋅p(b∣a)⋅p(c)⋅p(d∣a,b,c)p(a, b, c, d) = p(a) \cdot p(b|a) \cdot p(c) \cdot p(d|a, b, c)p(a,b,c,d)=p(a)⋅p(b∣a)⋅p(c)⋅p(d∣a,b,c)

The relationship between variables is more opaque, hidden behind second-order dependencies. For example, while it’s clear that DDD is directly dependent on AAA, we may miss the fact that there is another, second-order dependency between the two (DDD is dependent on BBB, which in turn is dependent on AAA).

Directed Acyclic Graphs

Directed Acyclic Graphs, or DAGs, offer a natural remedy to this problem. Each factor in the equation can be represented by a node. An arrow indicates conditional dependence. The resulting graph would look like:

With this graph, it’s easier to construct a generative story of how AAA, BBB, CCC and DDD are sampled. The process proceeds in topological order, for example AAA → CCC → BBB → DDD, to ensure that all dependencies have been resolved by the time each variable is sampled.

Below is what a sampled population of the given distributions would look like. For the sake of demonstration, many distribution parameters are modifiable — in reality these are the quantities that need to be learned from training data.

Sampled Population

For more detailed accounts of probabilistic graphical models, consider reading the textbooks Probabilistic Graphical Models: Principles and Techniques by Daphne Koller and Nir Friedman [2] and Probabilistic Reasoning in Intelligent Systems by Judea Pearl [3].

II. Hidden Markov Models

Hidden Markov Models (HMMs) are an early class of probabilistic graphical models representing partially hidden (unobserved) sequences of events. Structurally, they are built with two main layers, one hidden (SiS_iSi​) and one observed (OiO_iOi​):

HMMs have been successfully applied to a wide range of problems, including gene analysis [4], information extraction [5, 6], speech recognition [7], and named entity recognition [8].

The Hidden Layer

The hidden layer is assumed to be a Markov process: a chain of events in which each event’s probability depends only on the state of the preceding event. More formally, given a sequence of NNN random events S1S_1S1​, S2S_2S2​,…, SNS_NSN​, the Markov assumption holds that:

p(si∣s1,s2,…,si−1)=p(si∣si−1)for all i∈{2,…,N}p(s_i | s_1, s_2,…, s_{i-1}) = p(s_i | s_{i-1}) \\ \footnotesize\textrm{for all $i\,{\scriptscriptstyle \in}\,\{2,…, N\}$}p(si​∣s1​,s2​,…,si−1​)=p(si​∣si−1​)for all i∈{2,…,N}

In a graph, this translates to a linear chain of events where each event has one arrow pointing towards it (except for the first event) and one pointing away from it (except for the last):

A second assumption that HMMs make is time-homogeneity: that the probability of transition from one event's state to the next is constant over time. In formal terms:

p(si∣si−1)=p(si+1∣si)for all i∈{2,…,N−1}p(s_i | s_{i-1}) = p(s_{i+1} | s_i) \\ \footnotesize\textrm{for all $i\,{\scriptscriptstyle \in}\,\{2,…, N-1\}$}p(si​∣si−1​)=p(si+1​∣si​)for all i∈{2,…,N−1}

p(si∣si−1)p(s_i | s_{i-1})p(si​∣si−1​) is called the transition probability and is one of the two key parameters to be learned during training.

The assumptions about the hidden layer — Markov and time-homogeneity — hold up in various time-based systems where the hidden, unobserved events occur sequentially, one after the other. Together, they meaningfully reduce the computational complexity of both learning and inference.

The Observed Layer

The hidden and observed layer are connected via a one-to-one mapping relationship. The probability of each observation is assumed to depend only on the state of the hidden event at the same time step. Given a sequence of NNN hidden events S1S_1S1​, S2S_2S2​,…, SNS_NSN​ and observed events O1O_1O1​, O2O_2O2​,…, ONO_NON​ we have:

p(oi∣s1,s2,…,sN)=p(oi∣si)for all i∈{1,2,…,N}p(o_i | s_1, s_2,…, s_N) = p(o_i | s_i) \\ \footnotesize\textrm{for all $i\,{\scriptscriptstyle \in}\,\{1, 2,…, N\}$}p(oi​∣s1​,s2​,…,sN​)=p(oi​∣si​)for all i∈{1,2,…,N}

In a graph, this one-to-one relationship looks like:

The conditional probability p(oi∣si)p(o_i | s_i)p(oi​∣si​), called the emission probability, is also assumed to be time-homogenous, further reducing the model's complexity. It is the second key parameter to be learned, alongside the transition probability.

Representing Named Entities

HMMs’ chain structure is particularly useful in sequence labeling problems like NER. For each input text sequence, the observed layer represents known word tokens, while the hidden layer contains their respective name tags:

Representation: rather than labeling each node using the name of the variable it represents (X₁, Y₁) as we have until this point, we'll instead display the value of that variable (“O”, “Great”). This helps make the graphs easier to read.

There are 9 possible name tags. Each, apart from the “O” tag, has either a B- (beginning) or I- (inside) prefix, to eliminate confusion about when an entity stops and the next one begins.

Between any two consecutive hidden states, there are 9² = 81 possible transitions. Each transition has its own probability, p(xi∣xi−1)p(x_i|x_{i-1})p(xi​∣xi−1​):

Transition paths between hidden states.
Higher opacity indicates higher relative probability.

In the observed layer, each node can have any value from the vocabulary, whose size ranges anywhere from the thousands to the hundreds of thousands. The vocabulary created for the HMM in this article contains 23,622 tokens. Let N be the number of tokens in the vocabulary. The number of possible emission probabilities is 9N (nstates⋅ntokensn_{states} \cdot n_{tokens}nstates​⋅ntokens​).

Emission paths from hidden states to observations.
Higher opacity indicates higher relative probability.

Training

There are three sets of parameters to be learned during training: the transition, emission, and start probabilities. All can be computed as normalized rates of occurrence from the training data.

For example, to get the transition probability from state “O” to state “B-LOC”, p(B−LOC∣O)p(B{-}LOC | O)p(B−LOC∣O), we need two numbers: the number of times state “O” is followed by any other state (that is, it isn't the last state in the sequence), as NON_ONO​, and the number of times state “O” is followed by state “B-LOC”, as NO→B−LOCN_{O→B-LOC}NO→B−LOC​. The desired transition probability is NO→B−LOCNO\frac{N_{O→B-LOC}}{N_O}NO​NO→B−LOC​​. The same calculation can be done for each of the remaining probabilities.

Inference

In the context of HMMs, inference involves answering useful questions about hidden states given observed values, or about missing values given a partially observed sequence. In NER, we are focused on the first type of inference. Specifically, we want to perform maximum a posteriori (MAP) inference to identify the most likely state sequence conditioned on observed values.

There is usually an intractably large number of candidate state sequences. For any two consecutive states, there are 81 potential transition paths. For three states there are 82² paths. This number continues to grow exponentially as the number of states increases.

Transition paths between hidden states

Luckily, there is an efficient dynamic algorithm that returns the most likely path with relatively low computational complexity: the Viterbi algorithm [9]. It moves through the input sequence from left to right, at each step identifying and saving the most likely path in a trellis-shaped memory structure. For more details, refer to the excellent description of the Viterbi algorithm in the book Speech and Language Processing by Jurafsky & Martin [10].

Results

An HMM with the structure outlined above was trained on the CoNLL-2003 English dataset [11]. The train set contains 14,987 sentences and a total of 203,621 word tokens. Here's the model in action:

Name tag predictions by HMM:

Starting prediction server…

hmm
O
 = not a name
ORG
 = organization
PER
 = person
LOC
 = location
MISC
 = miscellaneous

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

TagNo OOV1+ OOV
ORG
0.8
0.21
PER
0.85
0.62
LOC
0.87
0.06
MISC
0.78
0.12
ALL
0.84
0.39
TagNo OOV1+ OOV

Recall

TagNo OOV1+ OOV
ORG
0.64
0.33
PER
0.58
0.59
LOC
0.71
0.05
MISC
0.54
0.06
ALL
0.64
0.41
TagNo OOV1+ OOV

This makes sense: we expect the model to perform worse on tokens that it wasn't trained on. And because the CoNLL-2003 train set has a large number of OOV tokens — 80.6% of all test entities contain at least one OOV token — across-the-board precision and recall are heavily impacted.

Limitations

HMMs are by no means perfect candidates for NER or text labeling problems in general, for at least three reasons. The first two have to do with the statistical assumptions that underlie HMMs’ chain structure, while the third relates to their sole reliance on word identity in the observed layer.

First is the Markov assumption. We don't compose words and their respective labels (“Noun”, “Adjective”, “B-ORG”, “I-ORG”) in a tidy, unidirectional manner. A word may link to another many places before or many more further down the line. In “South African Breweries Ltd”, both “South African” and “Ltd” provide crucial context to “Breweries”, which would otherwise register as not a name. HMMs fail to capture these tangled interdependencies, instead assuming that information passes from left to right in a single, neat chain.

An indication, although it is by no means proof, of such limitation lies in the fast deterioration of the precision score as the entity length — counted as the number of tokens inside each entity — increases (the recall scores are much more noisy and thus less conclusive):

TagEntity Length 1Entity Length 2Entity Length 3Entity Length 4Entity Length 5
ORG
0.61
0.68
0.3
0.12
0.28
PER
0.96
0.67
0.7
LOC
0.88
0.36
MISC
0.9
0.46
0.24
0.12
ALL
0.77
0.61
0.31
0.12
0.29
Tag12345

Entity Length

The second concern has to do with the emissions assumption. When composing sentences, we don’t go about forming a chain of labels (such as “B-ORG” – “I-ORG” – “I-ORG” – “O”…) before generating words from each label in that chain.

Semantic and grammatical rules may restrict the range of words that can appear in any given observation, but those restrictions are far from the strong emissions assumption made by HMMs. Beyond the current word label, there is a whole host of other factors that together help determine which words are chosen. Additionally, while there is a cognizable link between part-of-speech tags and the words that are chosen, with name tags that link is less clear.

The third concern relates to HMMs sole reliance on word identity in the observed layer. There are various word features that can provide additional information on the hidden label. Capitalization is a strong signal of named entities. Word shape may help account for common name patterns. HMMs take none of these features into account (in fact, they can’t: their generative chain structure requires that all observations be independent of each other).

With the only information available being word identities, HMMs end up stumbling over out-of-vocabulary words. While the models can wring bits of useful information out of nearby predicted name tags — the Viterbi algorithm maximizes likelihood over the whole sequence — the results are nonetheless discouraging:

Entity Length

Entity LengthOOV Rate 0OOV Rate 0.2OOV Rate 0.25OOV Rate 0.33OOV Rate 0.4OOV Rate 0.5OOV Rate 0.6OOV Rate 0.66OOV Rate 0.75OOV Rate 0.8OOV Rate 1
1
0.86
0.3
2
0.8
0.5
0.52
3
0.78
0.15
0.06
0
4
0.42
0.22
0
0
0
5
0.67
0.22
0
0.14
Entity Length00.20.250.330.40.50.60.660.750.81

OOV Rate

The concerns detailed above are direct consequences of HMMs’ generative graph structure. Below, we’ll consider another class of probabilistic graphical models with a different, discriminative structure that will hopefully address some of those concerns and deliver better performance on long, OOV-heavy entities.

III. Maximum Entropy Markov Models

Maximum Entropy Markov Models (MEMMs) [12] resolve some of the concerns we had with HMMs by way of a simple yet meaningful modification to their graphical structure :

The arrows connecting observations with their respective states have switched directions. We'll discuss the radical implications of such a change in the sections below.

Similar to HMMs, MEMMs have been successfully applied to a wide range of sequence modeling problems [12, 13, 14].

Discriminative Structure

There are two main approaches to building classification models: generative and discriminative [15]. Suppose there is a system with two variables, XXX and YYY. We want to make predictions about YYY based on observations on XXX. A generative classifier would do that by first learning the prior distribution p(X,Y)p(X, Y)p(X,Y) and then applying Bayes' rule to find the posterior p(Y∣X)p(Y|X)p(Y∣X). This can be thought of as reconstructing the process that generated the observed data. A discriminative classifier, on the other hand, would model the posterior p(Y∣X)p(Y|X)p(Y∣X) directly based on training data.

HMMs are generative classifiers, while MEMMs are discriminative. The former are generative because they model the joint distribution over both observations and hidden states (as a product of transition and emission probabilities) before using that joint distribution to find the most likely state sequence given observations (or solve some other inference problem). MEMMs, on the other hand, directly model the conditional probabilities p(state∣observation,prevstate)p(state|observation, prev state)p(state∣observation,prevstate) without any intermediary.

Word Features

Notably, MEMMs’ discriminative structure allows them to model overlapping word features. Two features can overlap when they contain the same or similar pieces of information, like word shape (“Xxx”) and capitalization. HMMs don’t allow overlapping features, since as a result of their generative structure they require that all events in the observation layer be independent of one another. MEMMs, on the other hand, are discriminative and able to relax the independence requirement, so they can use arbitrary overlapping word features [12].

Common practice is to use binary features, such as:

b(ot)={1 if ot has shape “Xxx”0 otherwiseb(o_t) = \begin{cases} \textrm{1 if $o_t$ has shape “Xxx”} \\ \textrm{0 otherwise} \end{cases}b(ot​)={1 if ot​ has shape “Xxx”0 otherwise​

These features are then paired with the current state sss to form feature-state pairs a=⟨b,s⟩a = \langle b, s \ranglea=⟨b,s⟩:

f⟨b,s⟩(ot,st)={1 if b(ot)=1 and st=s0 otherwisef_{\langle b, s \rangle}(o_t, s_t) = \begin{cases} \textrm{1 if $b(o_t) = 1$ and $s_t = s$} \\ \textrm{0 otherwise} \end{cases}f⟨b,s⟩​(ot​,st​)={1 if b(ot​)=1 and st​=s0 otherwise​

Feature-state pairs provide useful additional information how which features and states go together and which don’t. For example, we can expect pairs like "is_capitalized" + “B-ORG” to occur together frequently, capturing the fact that in English named entities are often capitalized.

State Transitions

MEMMs’ state transition distributions have exponential form and contain a weighted sum of all feature-state pairs:

ps′(s∣o)=1Z(o,s′)exp⁡(∑aλa fa(o,s))p_{s\prime}(s|o) = \frac{1}{Z(o, s\prime)} \exp\left(\sum_{a}{\lambda_a \, f_a(o, s)}\right)ps′​(s∣o)=Z(o,s′)1​exp(a∑​λa​fa​(o,s))

where s′s\primes′ and sss are the previous and current state, ooo is the current observation, a=⟨b,s⟩a = \langle b, s \ranglea=⟨b,s⟩ is a feature-state pair, λa\lambda_aλa​ is the learned weight for aaa, and Z(o,s′)Z(o, s\prime)Z(o,s′) is a normalizing term to make the distribution ps′p_{s\prime}ps′​ sum to one across all next states sss.

Calculating pO(B-LOC | “UK”)

With weights retrieved from a MEMM trained on CoNLL-2003 data. Numbers are rounded to 3 decimal places for clarity.
Feature-State Pair (a)λafaλafa
……………
……………
……………
……………
……………
……………
……………
……………
……………
SUM(λafa)0

pO(B-LOC | “UK”) = eSUM(λafa)  / Z
≈ e0 / 1
≈ 0

Those familiar with neural networks will recognize that the function above is a softmax. Its exponential form is a result of the core principle of maximum entropy that underlies MEMMs’ statistical structure and gives them their name. Maximum entropy states that the model that best represents our knowledge about a system is one that makes the fewest possible assumptions except for certain constraints derived from prior data from that system [12, 16].

Training & Inference

The training step involves learning the weights λa\lambda_aλa​ that satisfy MEMMs’ maximum entropy constraint [12]. Learning is done through Generalized Iterative Scaling, which iteratively updates the values λa\lambda_aλa​ in order to nudge the expected value of all features closer to their train set average. Convergence at a global optimum is guaranteed given the exponential form of the transition distribution.

As with HMMs, the Viterbi algorithm makes MAP inference tractable [12, 9]. The variable transition probability ps′(s∣o)p_{s\prime}(s|o)ps′​(s∣o) takes the place of HMMs’ fixed transition and emission probabilities.

Results

A MEMM was trained on the CoNLL-2003 English dataset [11]. In addition to word identity, features used for training include the word’s lowercase version (“Algeria” → “algeria”), shape (“Xxxx”), whether it’s in title/upper case, and whether it contains only digits.

A list of the most informative features — those with the largest absolute weights — offers valuable insights into how the model found and remembers linguistic patterns:

Most Informative Features when Previous State is 

Current Word FeatureCurrent StateWeight
word='germany'B-LOC11.492
word='van'B-PER8.972
word='wall'B-ORG8.525
word='della'B-PER7.86
lowercase='della'B-PER7.86
is_not_title_caseB-PER-6.949
word='de'B-PER6.781
shape='X.X.'O-6.713
shape='xxxx'B-ORG-6.642
word='CLINTON'B-ORG6.456

Many of these features are word identities. This makes intuitive sense: certain words, like “Germany”, are almost always used as names irrespective of what comes before or after them.

Other features relate to established linguistic patterns. For example, if the current word has shape “X.X.”, such as “U.S.” and “U.N.”, it’s unlikely to have the “O” tag — the feature-state pair’s weight is a large negative number. This means the word is likely a named entity, most probably two-letter initialisms.

Here’s a live version of the trained model:

Name tag predictions by MEMM:

Starting prediction server…

memm
O
 = not a name
ORG
 = organization
PER
 = person
LOC
 = location
MISC
 = miscellaneous

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

TagNo OOV1+ OOV
ORG
0.81
0.36
PER
0.82
0.8
LOC
0.82
0.17
MISC
0.74
0.14
ALL
0.8
0.54
TagNo OOV1+ OOV

Recall

TagNo OOV1+ OOV
ORG
0.68
0.12
PER
0.72
0.57
LOC
0.89
0.29
MISC
0.78
0.02
ALL
0.79
0.37
TagNo OOV1+ OOV

Advantage Over HMMs

The ability to model word features allows MEMMs to fare better with OOV-dense name entities than HMMs. Faced with words that they have never seen before during training, these models can easily stumble. Word identity alone provides no useful information. In those cases, derived features such as word shape and capitalization can function as imperfect yet doubtlessly helpful proxies for word identity, allowing MEMMs to make better guesses at the name tag, resulting higher precision and recall scores:

Entity Length

Entity LengthOOV Rate 0OOV Rate 0.2OOV Rate 0.25OOV Rate 0.33OOV Rate 0.4OOV Rate 0.5OOV Rate 0.6OOV Rate 0.66OOV Rate 0.75OOV Rate 0.8OOV Rate 1
1
0.83
0.34
2
0.76
0.72
0.55
3
0.6
0.56
0.59
0.5
4
0.16
0.2
0
5
0.6
0.5
Entity Length00.20.250.330.40.50.60.660.750.81

OOV Rate

With stronger predictive power on OOV words, we can additionally expect better performance on long, multi-word entities. That’s because OOV words are dangerous information gaps inside named entities. They’re easy to misclassify, and when they are the entire entity prediction is counted as incorrect. MEMMs are able to fill those gaps to an extent by using word features. As a result, we don’t see as drastic of a performance deterioration for longer entities as observed with the HMM:

TagEntity Length 1Entity Length 2Entity Length 3Entity Length 4Entity Length 5
ORG
0.76
0.69
0.84
0.36
0.8
PER
0.59
0.91
0.66
0.25
LOC
0.8
0.33
0.35
0
MISC
0.82
0.57
0.29
0.18
ALL
0.77
0.7
0.58
0.16
0.43
Tag12345

Entity Length

Label Bias Problem

MEMMs’ discriminative structure confers great benefits, but there’s a downside: it makes them susceptible to the label bias problem. First recorded by Bottou [17], this problem mostly affects discriminative models, causing certain states to effectively ignore their observations, biasing predictions toward less likely transition paths. While the label bias problem doesn’t render models useless, it still has a notable effect on predictions, causing demonstrably higher error rates [18].

What’s important to know is that MEMMs fall victim to the label bias problem because they have local probability normalization. The normalization factor Z(o,s′)Z(o, s\prime)Z(o,s′) ensures that transition probabilities between neighboring states sum up to one. Local normalization forces every state to transfer all of its probability mass onto the next state, regardless of how likely or unlikely the current observation is. Hannun [19] provides an excellent, detailed explanation of how this happens.

We can consider getting rid of local normalization to avoid the problem. That would lead us to Conditional Random Fields — a class of globally-normalized, undirected probabilistic models, which we’ll cover next.

IV. Conditional Random Fields

Conditional Random Fields (CRFs) are a class of undirected probabilistic models. They have proved to be powerful models with a wide range of applications, including text processing [18, 20, 21], image recognition [22, 23, 24, 25], and bioinformatics [26, 27].

While CRFs can have any graph structure, in this article we’ll focus on the linear-chain version:

Markov Random Fields

CRFs are a type of Markov Random Fields (MRFs) — probability distributions over random variables defined by undirected graphs [28]:

Undirected graphs are appropriate for when it’s difficult or implausible to establish causal, generative relationships between random variables. Social networks are a good example of undirected relationships. We can think of AAA, BBB, and CCC in the graph above as people in a simple network. AAA and BBB are friends and tend to share similar beliefs. The same goes for BBB and CCC as well as CCC and AAA. We might, for example, want to model how each person in the network thinks about a specific topic.

Acyclic Directed graphs fail to adequately represent the mutual belief propagation that occurs within the group. For example, we might have an edge from AAA to BBB but no path from BBB back to AAA — there will always be at least one such exclusion in an acyclic directed graph.

Rather than assuming a generative relationship between variables, MRFs model their mutual relationships with non-negative scoring functions ϕ\phiϕ, called factors, that assign higher scores if the variables’ values are in agreement, for example:

ϕ(X,Y)={3 if X=1 and Y=12 if X=0 and Y=01 otherwise\phi(X, Y) = \begin{cases} \textrm{3 if $X = 1$ and $Y = 1$} \\ \textrm{2 if $X = 0$ and $Y = 0$} \\ \textrm{1 otherwise} \end{cases}ϕ(X,Y)=⎩⎨⎧​3 if X=1 and Y=12 if X=0 and Y=01 otherwise​

Unlike conditional probabilities, there is no assumed directionality in scoring functions. These functions simply return higher scores if the variables agree and lower scores if they disagree. They model pairwise correlation, not causation.

The joint probability of all variables in the graph is:

p(A,B,C)=1Z ϕ(A,B) ϕ(B,C) ϕ(C,A)where Z is a normalization factorp(A,B,C) = \frac{1}{Z}\,\phi(A,B)\,\phi(B,C)\,\phi(C,A) \\ \footnotesize\textrm{where Z is a normalization factor}p(A,B,C)=Z1​ϕ(A,B)ϕ(B,C)ϕ(C,A)where Z is a normalization factor

The factors ϕ\phiϕ promote assignments in which their constituent variables (AAA and BBB in the case of ϕ(A,B)\phi(A, B)ϕ(A,B)) agree with each other. The assignment 1-1-1 would receive a higher score and thus have higher probability than say 1-1-0, since there is more agreement in the former case.

More generally, MRFs are probability distributions ppp over random variables x1x_1x1​, x2x_2x2​,… that are defined by an undirected graph G\mathcal{G}G and have the form:

p(x1,x2,…)=1Z∏c ∈ Cϕc(xc)where Z is a normalization factorand C is the set of cliques in Gp(x_1, x_2,…) = \frac{1}{Z}\prod_{c\,{\scriptscriptstyle \in}\,C}{\phi_c(x_c)} \\ \footnotesize\textrm{where Z is a normalization factor} \\ \footnotesize\textrm{and C is the set of cliques in $\mathcal{G}$}p(x1​,x2​,…)=Z1​c∈C∏​ϕc​(xc​)where Z is a normalization factorand C is the set of cliques in G

Calculating p(A, B, C, D, E, F)

p(1, 1, 1, 0, 0, 0)
=1/Z
⨯ ɸABC(1, 1, 1)
⨯ ɸAB(1, 1)
⨯ ɸBC(1, 1)
⨯ ɸAC(1, 1)
⨯ ɸCD(1, 0)
⨯ ɸDEF(0, 0, 0)
⨯ ɸDE(0, 0)
⨯ ɸEF(0, 0)
⨯ ɸDF(0, 0)

=1 / 28,915
⨯ 3
⨯ 3
⨯ 3
⨯ 3
⨯ 1
⨯ 2
⨯ 2
⨯ 2
⨯ 2

≈0.0448

ɸABC = ɸAB = ɸ…(x) = 3 if x1 = x2 = … = 1
2 if x1 = x2 = … = 0
1 otherwise

MRFs have a generalized form of which the directed models we’ve seen so far — HMMs and MEMMs — are special cases. The factors ϕc\phi_cϕc​ can be defined as conditional probabilities, for example ϕc(x1,x2)=p(x2∣x1)\phi_c(x_1,x_2) = p(x_2|x_1)ϕc​(x1​,x2​)=p(x2​∣x1​), and act as the transition and emission probabilities that characterize HMMs and MEMMs.

The additional level of generality comes at a cost, however: the normalization factors ZZZ are often difficult to compute. They require summing over an exponential number of potential assignments, an infeasible task if the network is large enough. Fortunately, there are configurations that can be solved using efficient decoding algorithms. That includes linear-chain CRFs, which can be decoded with the Viterbi algorithm.

Conditional Form

CRFs are random fields globally conditioned on a set of observations xxx [18] and have the form:

p(y∣x)=1Z(x)∏c ∈ Cϕc(yc,xc)where Z is a normalization factorand C is the set of cliques in thegraph G representing the labels yp(y|x) = \frac{1}{Z(x)}\prod_{c\,{\scriptscriptstyle \in}\,C}{\phi_c(y_c, x_c)} \\ \footnotesize\textrm{where Z is a normalization factor} \\ \footnotesize\textrm{and C is the set of cliques in the} \\ \footnotesize\textrm{graph $\mathcal{G}$ representing the labels $y$}p(y∣x)=Z(x)1​c∈C∏​ϕc​(yc​,xc​)where Z is a normalization factorand C is the set of cliques in thegraph G representing the labels y

The distribution p(y∣x)p(y|x)p(y∣x) is parameterized by xxx. When we replace all the values xix_ixi​ in the right hand side with real values, what remains has the same form as an MRF. In fact, we get a new MRF for every observation sequence xxx.

CRFs are globally conditioned on xxx. They directly model the probability of the label sequence yyy — p(y∣x)p(y|x)p(y∣x) — rather than local transition/emission probabilities p(yi∣yi−1)p(y_i|y_{i-1})p(yi​∣yi−1​) or p(yi∣xi)p(y_i|x_i)p(yi​∣xi​).

Global conditioning on xxx means that the hidden states yiy_iyi​ can depend not only on the current observation but also any other observation in the sequence. Adding more such dependencies to the model does not increase the computational complexity of inference tasks, since we don’t have to model the marginal probabilities p(xi)p(x_i)p(xi​) at train/test time.

Linear-chain CRF where the hidden layer depends on the current, previous, and future observations.

Exponential Factors

The factors ϕc\phi_cϕc​ have an exponential form [18] that’s similar MEMMs’ transition function:

ϕc(yc,xc)=exp⁡(∑aλa fa(yc,xc))where fa is a feature function defined for clique cand λa is the weight parameter for fa\phi_c(y_c, x_c) = \exp\left(\sum_{a}{\lambda_a \, f_a(y_c, x_c)}\right) \\ \footnotesize\textrm{where $f_a$ is a feature function defined for clique $c$} \\ \footnotesize\textrm{and $\lambda_a$ is the weight parameter for $f_a$}ϕc​(yc​,xc​)=exp(a∑​λa​fa​(yc​,xc​))where fa​ is a feature function defined for clique cand λa​ is the weight parameter for fa​

References

  1. MUC-7 Named Entity Task Definition (Version 3.5) PDF
    Nancy Chinchor. 1998. In Seventh Message Understanding Conference (MUC-7).
  2. Probabilistic Graphical Models: Principles and Techniques - Adaptive Computation and Machine Learning
    Daphne Koller and Nir Friedman. 2009. The MIT Press.
  3. Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference
    Judea Pearl. 1988. Morgan Kaufmann Publishers Inc, San Francisco, CA, USA.
  4. Genes, Themes and Microarrays: Using Information Retrieval for Large-Scale Gene Analysis
    Hagit Shatkay, Stephen Edwards, W John Wilbur, and Mark Boguski. 2000. In Proceedings of the  International Conference on Intelligent Systems for Molecular Biology, 317–328.
  5. Information Extraction Using Hidden Markov Models
    Timothy Robert Leek. 1997. Master’s Thesis, UC San Diego.
  6. Information Extraction with HMMs and Shrinkage PDF
    Dayne Freitag and Andrew McCallum. 1999. In Papers from the AAAI-99 Workshop on Machine Learning for Information Extraction (AAAI Techinical Report WS-99-11), 31–36.
  7. A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition
    Lawrence R Rabiner. 1989. Proceedings of the IEEE 77, 2: 257–286. https://doi.org/10.1109/5.18626
  8. An Algorithm that Learns What’s in a Name PDF
    Daniel M. Bikel, Richard Schwartz, and Ralph M. Weischedel. 1999. Machine Learning 34, 1: 211–231. https://doi.org/10.1023/A:1007558221122
  9. Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding Algorithm
    A. Viterbi. 1967. IEEE Transactions on Information Theory 13, 2: 260–269.
  10. Appendix A.4 — Decoding: The Viterbi Algorithm PDF
    Daniel Jurafsky and James H. Martin. 2021. In Speech and Language Processing. 8–10.
  11. Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition PDF
    Erik F. Tjong Kim Sang and Fien De Meulder. 2003. In Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003, 142–147.
  12. Maximum Entropy Markov Models for Information Extraction and Segmentation PDF
    Andrew McCallum, Dayne Freitag, and Fernando C. N. Pereira. 2000. In Proceedings of the Seventeenth International Conference on Machine Learning (ICML ’00), 591–598.
  13. Maximum Entropy Models for Antibody Diversity Link
    Thierry Mora, Aleksandra M. Walczak, William Bialek, and Curtis G. Callan. 2010. Proceedings of the National Academy of Sciences 107, 12: 5405–5410. https://doi.org/10.1073/pnas.1001705107
  14. Human Behavior Modeling with Maximum Entropy Inverse Optimal Control PDF
    Brian Ziebart, Andrew Maas, J. Bagnell, and Anind Dey. 2009. In Papers from the 2009 AAAI Spring Symposium, Technical Report SS-09-04, Stanford, California, USA, 92–97.
  15. On Discriminative vs. Generative Classifiers: A comparison of logistic regression and naive Bayes PDF
    Andrew Ng and Michael Jordan. 2001. In Advances in Neural Information Processing Systems.
  16. Inducing Features of Random Fields
    S. Della Pietra, V. Della Pietra, and J. Lafferty. 1997. IEEE Transactions on Pattern Analysis and Machine Intelligence 19, 4: 380–393. https://doi.org/ 10.1109/34.588021
  17. Une Approche théorique de l’Apprentissage Connexionniste: Applications à la Reconnaissance de la Parole
    Léon Bottou. 1991. Université de Paris X.
  18. Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data PDF
    John D. Lafferty, Andrew McCallum, and Fernando C. N. Pereira. 2001. In Proceedings of the Eighteenth International Conference on Machine Learning (ICML ’01), 282–289.
  19. The Label Bias Problem Link
    Awni Hannun. 2019. Awni Hannun — Writing About Machine Learning.
  20. Discriminative Probabilistic Models for Relational Data Link
    Ben Taskar, Pieter Abbeel, and Daphne Koller. 2013. https://doi.org/10.48550/ARXIV.1301.0604
  21. Accurate Information Extraction from Research Papers using Conditional Random Fields Link
    Fuchun Peng and Andrew McCallum. 2004. In Proceedings of the Human Language Technology Conference of the North American Chapter of the Association for Computational Linguistics: HLT-NAACL 2004, 329–336.
  22. Discriminative Fields for Modeling Spatial Dependencies in Natural Images PDF
    Sanjiv Kumar and Martial Hebert. 2003. In Advances in Neural Information Processing Systems.
  23. Multiscale Conditional Random Fields for Image Labeling Link
    Xuming He, R.S. Zemel, and M.A. Carreira-Perpinan. 2004. In Proceedings of the 2004 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2004, CVPR 2004, II–II. https://doi.org/10.1109/CVPR.2004.1315232
  24. Conditional Random Fields as Recurrent Neural Networks Link
    Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes, Vibhav Vineet, Zhizhong Su, Dalong Du, Chang Huang, and Philip H. S. Torr. 2015. In 2015 IEEE International Conference on Computer Vision (ICCV). https://doi.org/10.1109/iccv.2015.179
  25. Convolutional CRFs for Semantic Segmentation Link
    Marvin T. T. Teichmann and Roberto Cipolla. 2018. https://doi.org/10.48550/arxiv.1805.04777
  26. RNA Secondary Structural Alignment with Conditional Random Fields Link
    Kengo Sato and Yasubumi Sakakibara. 2005. Bioinformatics 21: ii237–ii242. https://doi.org/10.1093/bioinformatics/bti1139
  27. Protein Fold Recognition Using Segmentation Conditional Random Fields (SCRFs)
    Yan Liu, Jaime Carbonell, Peter Weigele, and Vanathi Gopalakrishnan. 2006. J. Comput. Biol. 13, 2: 394–406.
  28. Introduction to Markov Random Fields
    Andrew Blake and Pushmeet Kohli. 2011. In Markov Random Fields for Vision and Image Processing. The MIT Press.

© Gradient 2023
