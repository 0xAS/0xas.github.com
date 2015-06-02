---
layout : post
category : LearningNote
tags : [MachineLearning, Basis]
---
{% include JB/setup %}

**Chapter 1. Introduction**

- *Definition*:
    + *Machine Learning*: A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E. - *Tom Mitchell*.
    + *Supervised Learning*: given a dataset and knowing what the correct output should look like, supervised learning tries to find the relationship between the input and output.
        * It is categorized into "*regression*" and "*classification*" problems.
    + *Unsupervised Learning*: Unsupervised learning derives structure from data where we don't necessarily know the effect of the variables, approaching problems with little or no idea what our results should look like.
**Chapter 2. Linear Regression with One Variable**

- *Definition*:
    + *Univariable linear regression*: linear regression with one variable is also known as "univariable linear regression". It is used to predict a signle output value from a single input value.
- *Notes*:
    + Hypothesis function:\\
        $$ h_{\theta}(x) = \theta_0 + \theta_1 \times x$$