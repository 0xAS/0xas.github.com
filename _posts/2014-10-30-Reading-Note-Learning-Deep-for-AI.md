---
layout : post
category : ReadingNotes
tags : [DeepLearning, DBN]
---
{% include JB/setup %}

*"Learning Deep Architectures for AI"* [link](http://www.iro.umontreal.ca/~bengioy/papers/ftml_book.pdf) gives a brief introduction about one main type of deep learning algorithms, *Deep Belief Networks (DBNs)*. It starts with the motivation for deep architecture, after which is the brief theoretical introduction of *DBNs*. This reading notes tries to summarize the arguements, theorems, proofs, fomulars and algorithms in section wise.


### Ch 1. Introduction
 
- *Arguments*
    + The *motivation of deep learning* is to *automatically discover abstractions from the lowest level features to the highest level abstract concepts*. The ideal learning algorithms are desired to discover these features with as little human effort as possible, without having to manually define all necessary abstractions or having to provide a huge set of relevant hand-labeled samples.
    + The *aim of deep learning methods* is to learn feature hierarchies with features from higher levels of the hierarchy formed by the composition of lower level features.
    + Automatical feature learning at multiple levels of abstraction allows a system to learn complex (non-linear) functions mapping the input to the output directly from data, without depending completely on human-crafted features. *Important for high-level features*: since humans often do not know how to specify explicitly in terms of raw sensory input.
    + **Breakthrough in 2006**
        * *Hinton et al [73], greedy layer-wise learning algorithm exploiting an unsupervised learning algorithm for each RBM layer [51]*
        * *auto-encoders proposed [17, 153]. guiding the training of intermediate levels of representation using unsupervised learning, which can be performed locally at each level.*
        * *deep convolutional neural network*
<!--more-->
    + Observation found with many successful experiment: once a good representation has been found at each level, it can be used to initialize and successfully train a deep neural network by supervised gradient-based optimization.
    + Brain has **sparse** and **distributed representation** to process visual information.
    + Deep architectures natually provide sharing and re-use components for multi-task learning: low-level visual features and intermediate-level visual features are useful for a large group of visual tasks. Deep learning algorithms are based on learning intermediate representations which can be shared across tasks.
    + Learning about a large set of interrelated concepts might provide a key to the kind of broad generalizations that human appear able to do.