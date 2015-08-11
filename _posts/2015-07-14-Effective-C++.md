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
    + *#define* doesn't respect scope. Once a macro is defined, it's in force for the rest of the compilation.
    + When replacing #define with constant, two special cases are worth mentioning:
        * Defining cosntant pointers.
        * To limit the scope of a constant to a class, make it a memer. To ensure there's at most one copy of the constant, make it a *static* member.
    + When you need the value of a class constant during compilation of the class, use "*the enum hack*". It takes the advantage of the fact that the values of an enumerated type can be used where ints are expected.
    + *For simple constants, prefer const objects or enums to #defines*.
    + *For function-like macros, prefer inline functions to #defines*.
- *Item 3*: use *const* whenever possible.
    + The purpose of *const* om member functions is to identify which member functions may be invoked on *const* objects.
        * 1. Make the interface of a class easier to understand which functions may modify an object and which may not. 
        * 2. Make it possible to work with *const* objects.
    + Declearing something *const* helps compilers detect usage errors.
    + Compilers enforce bitwise constness, but you should program using conceptual constness.
    + When *const* and *non-const* member functions have essentially identical implementations, code duplication can be avoided by having the *non-const* version call the *const* version.
- *Item 4*: make sure that objects are initialized before they are used.
    + Always initialize the object before using it.
        * Make sure that all constructors initialize everything in the object.
        * The order in which an object's data is initialized is not fickle: base classes are initialized before derived classes, and within a class, data members are initialized in the order in which they are declared.
        * A translation unit is the source code giving rise to a single object file.
        * The relative order of initialization of non-local static objects defined in different translation units is undefined.
    + Manually initialize objects of built-in type.
    + Prefer use of the member initialization list to assignment inside the body of the constructor. List data members in the initialization list in the same order they are declared in the class.
    + Avoid initialization order problems across translation units by replacing non-local static objects with local static objects.
- *Item 5*: Compilers may implicitly generate a class's default constructor, copy constructor, copy assignment operator, and destructor.
    + If you want to support assignment in a class containing a reference member, you must define the copy assignment operator yourself.
- *Item 6*: To disallow functionality automatically provided by compilers, declare the corresponding member functions *private* and give no implementations. Using a base class like *uncopyable* is one way to do this.
- *Item 7*: Declare virtual destructors in polymorphic base classes.
    + If a class has any virtual functions, it should have a virtual destructor.
    + Classes not designed to be base classes or not designed to be used polymorphically should not declare virtual destructors.
- *Item 8*: Destructors should never emit exceptions.
    + If functions called in a destructor may throw, the destructor should catch any exceptions, then swallow them or terminate the program.
- *Item 9*: Don't call virtual functions during construction or destruction, since such calls will never go to a more derived class than that of the currently executing constructor or destructor.
- *Item 10*: Assignment operators should return a reference to its left-hand argument. 
    + While defining your own assignment operators, the last line code should always be ''' return *this'''.