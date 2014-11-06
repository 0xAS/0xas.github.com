---
layout : post
category : ReadingNotes
tags : [DeepLearning, DBN]
---
{% include JB/setup %}

*"Learning Deep Architectures for AI"* [link](http://www.iro.umontreal.ca/~bengioy/papers/ftml_book.pdf) gives a brief introduction about one main type of deep learning algorithms, *Deep Belief Networks (DBNs)*. It starts with the motivation for deep architecture, after which is the brief theoretical introduction of *DBNs*. This reading notes tries to summarize the arguements, theorems, proofs, fomulars and algorithms in section wise.


Ch1. Introduction

- *Concepts*
    + *Depth of architectures:* the number of levels of composition of non-linear operations in the function learned. (shallow architectures: 1, 2, or 3 levels). Mammal brain is in a deep architecture [173].
    + *Distributed representation [68,156]:* the information is not localized in a particular neuron but distributed.
- *Arguments*
    + The *motivation of deep learning* is to *automatically discover abstractions from the lowest level features to the highest level abstract concepts*. The ideal learning algorithms are desired to discover these features with as little human effort as possible, without having to manually define all necessary abstractions or having to provide a huge set of relevant hand-labeled samples.
    + The *aim of deep learning methods* is to learn feature hierarchies with features from higher levels of the hierarchy formed by the composition of lower level features.
    + Automatical feature learning at multiple levels of abstraction allows a system to learn complex (non-linear) functions mapping the input to the output directly from data, without depending completely on human-crafted features. *Important for high-level features*: since humans often do not know how to specify explicitly in terms of raw sensory input.
<!--more-->
    + **Breakthrough in 2006**
        * *Hinton et al [73], greedy layer-wise learning algorithm exploiting an unsupervised learning algorithm for each RBM layer [51]*
        * *auto-encoders proposed [17, 153]. guiding the training of intermediate levels of representation using unsupervised learning, which can be performed locally at each level.*
        * *deep convolutional neural network*
    + Observation found with many successful experiment: once a good representation has been found at each level, it can be used to initialize and successfully train a deep neural network by supervised gradient-based optimization.
    + Brain has **sparse** and **distributed representation** to process visual information.
    + Deep architectures natually provide sharing and re-use components for multi-task learning: low-level visual features and intermediate-level visual features are useful for a large group of visual tasks. Deep learning algorithms are based on learning intermediate representations which can be shared across tasks.
    + Learning about a large set of interrelated concepts might provide a key to the kind of broad generalizations that human appear able to do.
- *Bibliography*
    + [173] [link]() biological side of how human brain processes the information. The brain appears to process information through multiple stages of transformation and representation, especially in primate visual system.
    + [17,153] [link]() auto-encoder based deep architecture learning algorithm
    + [37] [link]() poverty of labeled data solved by dln.

Ch2. Theoretical Advantages of Deep Architectures

- *Concepts*
    + *Computational elements*: logical gates (AND, OR, NOT), affine transformation, kernel computation.
    + *Compact function*: the expression of a function is compact when it has few computational elements.
    + *artificial neuron*: an affine transformation followed by a non-linearity.
    + *Monotone weighted threshold circuits*: multi-layer neural networks with linear threshold units and positive weights.
- *Arguments*
    + The formal arguments for the power of deep architectures investigate into two directions:
        * **Computational Complexity of Circuits**: Deep architectures can **compactly represent** highly varying functions which would otherwise require a very large size to be represented with an inappropriate architecture, normally in *exponential times*.
        * A function which can be expressed by the composition of computational elements from a given set, can be pictured as a graph formalizing the composition with one node per computational element. *Depth of architecture refers to the depth of that graph*.
        * The composition of computational units in a small but deep circuit can actually be seen as an efficient factorization of a large but shallow circuit.
        * The depth of architectures is very important for statistical efficiency.
- *Bibliography* 
    + [156]: multilayer neural network: putting artificial neurons into the set of computational elements.
    + [63]: Theorem 2.1: monotone weighted threshold circuits 
    + [19] & [191]: discussion on the power of deep architectures and their potential for AI
    + [140] an early survey of theoretical results in computational complexity relevant to machine learning algorithms.

#### Ch3. Local vs. Non-local Generalization

