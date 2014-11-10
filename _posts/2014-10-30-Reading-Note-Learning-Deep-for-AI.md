---
layout : post
category : ReadingNotes
tags : [DeepLearning, DBN]
---
{% include JB/setup %}

*"Learning Deep Architectures for AI"* [link](http://www.iro.umontreal.ca/~bengioy/papers/ftml_book.pdf) gives a brief introduction about one main type of deep learning algorithms, *Deep Belief Networks (DBNs)*. It starts with the motivation for deep architecture, after which is the brief theoretical introduction of *DBNs*. This reading notes tries to summarize the arguements, theorems, proofs, fomulars and algorithms in section wise.


**Ch1. Introduction**

- *Concepts*
    + *Depth of architectures:* the number of levels of composition of non-linear operations in the function learned. (shallow architectures: 1, 2, or 3 levels). Mammal brain is in a deep architecture [173].
    + *Distributed representation [68,156]:* the information is not localized in a particular neuron but distributed.
- *Arguments*
    + The *motivation of deep learning* is to *automatically discover abstractions from the lowest level features to the highest level abstract concepts*. The ideal learning algorithms are desired to discover these features with as little human effort as possible, without having to manually define all necessary abstractions or having to provide a huge set of relevant hand-labeled samples.
    + The *aim of deep learning methods* is to learn feature hierarchies with features from higher levels of the hierarchy formed by the composition of lower level features.
    + Automatical feature learning at multiple levels of abstraction allows a system to learn complex (non-linear) functions mapping the input to the output directly from data, without depending completely on human-crafted
<!--more-->
    features. *Important for high-level features*: since humans often do not know how to specify explicitly in terms of raw sensory input.
    + **Breakthrough in 2006**
        * <cite>Hinton et al [73], greedy layer-wise learning algorithm exploiting an unsupervised learning algorithm for each RBM layer [51]</cite>
        * <cite>auto-encoders proposed [17, 153]. guiding the training of intermediate levels of representation using unsupervised learning, which can be performed locally at each level.</cite>
        * <cite>deep convolutional neural network</cite>
    + Observation found with many successful experiment: once a good representation has been found at each level, it can be used to initialize and successfully train a deep neural network by supervised gradient-based optimization.
    + Brain has **sparse** and **distributed representation** to process visual information.
    + Deep architectures natually provide sharing and re-use components for multi-task learning: low-level visual features and intermediate-level visual features are useful for a large group of visual tasks. Deep learning algorithms are based on learning intermediate representations which can be shared across tasks.
    + Learning about a large set of interrelated concepts might provide a key to the kind of broad generalizations that human appear able to do.
- *Bibliography*
    + <cite>[173] [link]() biological side of how human brain processes the information. The brain appears to process information through multiple stages of transformation and representation, especially in primate visual system.</cite>
    + <cite>[17,153] [link]() auto-encoder based deep architecture learning algorithm</cite>
    + <cite>[37] [link]() poverty of labeled data solved by dln.</cite>

**Ch2. Theoretical Advantages of Deep Architectures**

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
    + <cite>[156]: multilayer neural network: putting artificial neurons into the set of computational elements.</cite>
    + <cite>[63]: Theorem 2.1: monotone weighted threshold circuits </cite>
    + <cite>[19] & [191]: discussion on the power of deep architectures and their potential for AI</cite>
    + <cite>[140] an early survey of theoretical results in computational complexity relevant to machine learning algorithms.</cite>

**Ch3. Local vs. Non-local Generalization**

- *Concepts*
    + *Local in input space estimator*: an estimator obtains good generalization for a new input $$x$$ by mostly exploiting training samples in the neighborhood of $$x$$.
    + *Parity function*: a boolean function whose value is 1 if and only if the input vector has an odd number of ones.
    + *Local kernel*: a kernel is locak when $$k(x, x_i) > row$$ is true only for $$x$$ in some connected region around $$x_i$$ (for some threshold row). $$x_i$$ is he training sample, while $$x$$ is the input sample to be learned/classified.
    + *Distributed representation*: the input pattern is represented by a set of features that are not mutually exclusive and might even be statistically independent.
- *Arguments*
    + *Curse of dimension*: what matters for generalization is not dimensionality, but instead the number of "variations" of the function we wish to obtain after learning.
    + Architectures based on matching local templates can be thought of as having two levels:
        * $$1^{st} level$$: input sample will be matched to a set of templates, with an output of a value indicating the degree of matching.
        * $$2^{nd} level$$: perform a kind of interpolation to predit with template the input sample mostly fits in.
            $$ f(x) = b + \sum_i {\alpha_i K (x, x_i)}$$
    + Kernal machine is the *prototypical example* of local matching architecture. It yields generalization by exploiting the *smoothness prior*.
    + For a maximally varying function such as the parity function, the number of examples necessary to achieve some error rate with a Gaussian kernel machine is *exponential* in the input dimension.
    + *Biggest disadvantage of supervised, semi-supervised learning algorithms based on neighbor graph*: they need as many samples as there are variations of interest in the target function, and they *cannot generalize to new variations not covered in the training set*.
    + Local representation for a vector can be represented in a much more compact way with the help of distributed representation (*exponentially*).
- *Bibliography*
    + <cite>[160] a Gaussian Process kernel machine can be improved using a Deep Belief Network to learn a feature space</cite>
    + <cite>[41] computational complexity literature tells that the number of training examples necessary to achieve a given error rate is exponential in the input dimension.</cite>
    
**Ch4. Neural Networks for Deep Architectures**

- *Concepts*
    + *Neural Network*: multilayer perceptron.
        * *equation of each layer*: 
        $$h^k = tanh(b^k + W^k h^{k-1})$$
        This equation performs an additional nonlinear computation on top of the affine transformation. The hyperbolic tangent non-linearity can be replaced by sign function, softamx function, etc..
        * *loss function*: 
        $$L(h^l, y) = -logP(Y = y|x) = -log h_{y}^{l}$$
        Loss function is typically *convex* in $$b^l + W^l h^{l-1}$$.
        * *training rule*: The target is to reduce the loss of the prediction. By that, it means to minimize the log-likelihood of the conditional probability of $$y$$ given $$x$$.
    + Sigmoid belief network: units in each layer are independent given the values of the units in the layer above.
    + Deep belief network: similar to sigmoid networks, but top two layers are RBM.
    + LeCun's Convolutional neural networks: convolutional layers, subsampling layers, error gradient. Each neuron is associated with a fixed two dimenional position that corresponds to a location in the input image, along with a receptive field.
    + Auto-encoder: encode the input $$x$$ into some representation $$c(x)$$ so that the input can be reconstructed from that representation.
    + Training distribution: the emperical distribution of the training set, or the generating distribution for our training samples.
    + Model distribution: the probability distribution of the trained model.
- *Arguements*
    + *Difficulties* in training deep architectures:
        * random initialization of network parameters often leads to "*apparent local minima or plateaus*"
        * the deeper the network goes, the worse the generalization we get.
    + Greedy layer-wise unsupervised training (RBM, auto-encoder) helps to initialize the network's parameters, which improves the generalization performance.
    + Possible explanation for the improvements brought by unsupervised pre-training.
        * These unsupervised training algorithms have layer-local unsupervised criteria. That helps guide the parameters of that layer towards better regions in parameter space.
        * Unsupervised pre-training can be seen as *a form of regularizer* (and prior): unsupervised pre-training amounts to a constraint on the region in parameter space where a solution is allowed.
        * The effect of unsupervised learning is most marked for the lower layers of a deep architecture, which means it is more in the "optimization" direction.
        * Unsupervised pre-training helps generalization by allowing for a better tuning of lower layers a deep architecture. WIth unsupervised pre-training, the lower layers are constrained to capture regularities of the input distribution.
        * The unsupervised pre-training does not only help to regularize the nework, but also helps to optimize the weights of the lower layers in the deep architecture.
        * The gradient descent criterion defined at the output layer becomes less useful as it propagates backwards to the lower layer. That's why the generalization error is poor with random parameter initialization. Because the criterion cannot help optimize the lower level layer parameters.
        * Generally speaking, *unsupervised learning could help reduce the dependency on the unreliable update direction given by the gradient of a supervised criterion*. It also helps to decompose the problem into sub-problems associated with different levels of abstraction.
    + Convolutional neural networks is not difficult to train even without unsupervised pre-training.
    + One issue with *auto-encoder*: if there is no other constraint, the auto-encoder with n-dimensional inout and an encoding of dimension at least n could potentially just learn the identity function.
        * This might be avoided by using the stochastic gradient descent with early stopping, which is similar to an $$l_2$$ regularization of the parameters.
        * *adding noises* to the encoding may help avoid this problem as well.
    + *Denoising auto-encoder* not only tries to *encode the input* but also to *capture the statistical structure in the input*, by approximately *maximizing the likelihood of a generative model*. This maximizes a lower bound on the log-likelihood of a generative model.
- *Bibliography*
    + <cite>[73] initial experiments using RBM for each layer</cite>
    + <cite>[17,98,99] statistical comparisons to prove the advantage of unsupervised pre-training versus random initialization</cite>
    + <cite>[50] unsupervised pre-training acts more like a data dependent "regularizer".</cite>
    + <cite>[91] generative models can often be represented as graphical models</cite>
    + <cite>[83] convolutional nets were inspired by the visual system's structure.</cite>
    + <cite>[101,104] LeCun's state of art convolutional neural network on visual classification tasks. error-gradient</cite>
    + <cite>[45, 111] imported convolutonal structure into DBN, design of a generative version of pooling/subsampling units.</cite>

**Ch5. Energy based Models and RBM**

- *Concepts*
    + *Energy based model*: The model associates a scalar energy to each configuration of the variables of interst.
        $$P(x) = \frac{e^{-Energy(x)}}{Z}$$, where $$Z = \sum_x e^{-Energy(x)}$$ is the partition function as a sum running over the input space.
    + *Boltzmann machine*: Energy function of a BM is $$Energy(x,h) =  -b^{'}x - c^{'}h - h^{'}Wx - x^{'}Ux - h^{'}Vh$$. These parameters (offsets and weights) are collectively denoted by $$\theta$$.
    + *Restricted Boltzmann Machine (RBM)*: Building block of the DBN. It shares parametrization with individual layers of a DBN.
        * Hidden units are independent of each other, when conditioning on visible units $$x$$. Likewise for visible units.
        * Energy function:
            $$Energy(x,h) = -b^{'}x - c^{'} - h^{'}Wx$$
          Free energy function:
            $$ FreeEnergy(x) = -b^{'}x - \sum_i log \sum_{h_i}e^{j_i (c_i + W_ix)}$$
        * Conditional probability: 
            $$P(h_i = 1|x) = \prod_{i} P(h_i | x) P(x | h) = \prod_{i}P(x_i | h) $$, in binary case: 
            $$ P(h_i = 1 | x) = \frac{e^{c_i + W_ix}}{1 + e^{c_i + W_ix}} = sigm(c_i + W_ix)$$
            $$ P(w_i = 1 | h) = \frac{e^{b_i + W_ih}}{1 + e^{b_i + W_ih}} = sigm(b_i + W_ih)$$
    + *CD-K*: the idea of k-step contrastive divergence involves two approximations:
        * Replace average over all possible inputs by a simple sample
        * Run the MCMC (Monte Carlo Markov Chain) chain $$x_1, x_2, ... , x_{k+1}$$ for only $$k$$ steps starting from the observed example $$x_1 = x$$.
        * Equation: $$ \Delta \theta \propto \frac{\partial{FreeEnergy(x)}}{\partial{\theta}} - \frac{\partial{FreeEnergy(\wildtilde{x})}}{\partial{\theta}}$$
    + *Persistent MCMC for Negative Phase*: instead of CD-k for updating parameters, an MCMC chain is kept in the background to obtain the negative phase samples $$(x,h)$$.
    + *Score matching*: the score function of a distribution is defined as $\phi = \frac{\partial{logp(x)}}{\partial{x}}$. This does not depend on the normalization constant. The idea is to match the score function of the model with the score function of the empirical density.
- *Arguements*
    + Boltzmann Machine is defined by an energy function $$P(x)=e^{-E(x)/Z)}$$. Due to the quadratic interaction in $$h$$, *an Monte Carlo Markov Chain sampling procedure can be applied to obtain a stochastic estimator of the gradient (log-likelihood gradient).*
    $$ \frac{\partial{logP(x)}}{\theta}  = \frac{\partial{log \sum_h e^{-Energy(x,h)}}}{\theta} - \frac{\partial{log\sum_{\wildtilde{x},h}e^{-Energy(\wildtilde{x},h)}}}{\theta}$$ 
- *Bibliography*
