---
layout : post
category : LearningNote
tags : [EffectiveC++, C++, Basis]
---
{% include JB/setup %}

**Introduction**

A short review of the terminology, the C++ vocabulary:
- *signature*: the parameters and return types of a function, revealed by each function's declaration.
- *explicit*: constructors declared explicit are usually preferable to non-explicit ones, because they prevent compilers from performing unexpected type conversions.

**Chapter 1**

- *Item 1*: view C++ as a federation of languages  
    + the primary sublanguages of C++:
        * C. C++ is still based on C deep down.
        * Object-Oriented: classes, encapsulation, inheritance, polymorphism, virtual functions (dynamic binding), etc.
        * Template C++: the generic programming part of C++.
        * The STL: standard template library.
- *Item 2*: prefer consts, enums, and inlines to #defines. This can be called "prefer compiler to preprocessor" as well.
    + #define doesn't respect scope. Once a macro is defined, it's in force for the rest of the compilation.
    + When replacing #define with constant, two special cases are worth mentioning:
        * 1. defining cosntant pointers.
        * 2. To limit the scope of a constant to a class, make it a memer. To ensure there's at most one copy of the constant, make it a *static* member.
    + When you need the value of a class constant during compilation of the class, use "*the enum hack*". It takes the advantage of the fact that the values of an enumerated type can be used where ints are expected.
    + *For simple constants, prefer const objects or enums to #defines*.
    + *For function-like macros, prefer inline functions to #defines*.