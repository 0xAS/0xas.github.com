---
layout : post
category : LearningNote
tags : [C++Primer, Basis]
---
{% include JB/setup %}

```java
string.empty()          // returns true if string is empty; otherwise false.
string.size()           // returns number of characters in the string
string[n]               // access the n+1 th element in the string. string index starts from 0 to string.size()-1
s1 + s2                 // concatenats s1 and s2
                        // **when mixing strings and string literals, at least one operand to each '+' operator must be of *string* type**
s1 = s2                 // replace characters in s1 by a copy of s2
s1 == s2                // returns trun if v1 and v2 are equal, otherwise false
!=, <, <=, >, >=        // comparison between strings
```

**Chapter 2: Variables and Basic Types**


- Primitive Built-in Types
    + Integral Types: arithmetic types representing integers, characters and boolean values.
        * a byte = 8 bits, a word = 4 bytes = 32 bits.
        * when the assigned value exceeds the range which a data type can hold, the compiler will let the assigned value modulo the max range value.
    + Floating-point Types: *float (32 bits), double (64 bits), long double (96 or 128 bits)*
- Literal Constants: 
    + Literal: we can only speak of it in terms of its value
    + Constant: it cannot be changed.
    + 3 ways of notations for *integral literals*: decimal, octal, or hexadecimal.
        * Suffix: by adding a suffix, we can force the type of a literal integer constant to be type *long (L)* or *unsigned (u)* or *unsigned long (UL)*.
    + *floating-point literals*: 2 ways of notations: decimal (L), or scientific (E, or e).
    + *Boolean and Character Literals*:
        * boolean: true, or false
        * character: '' + character.
    
>| signed | unsigned | boolean | 
>|--------|:---------:|--------:|
>|both postive and negative numbers | only valus greater than 0 | true or false |
>| sign bit = 1, value < 0 | | |
>| sign bit = 0, value >= 0| | |

<!--more-->

- constructor:
    + default constructor: constructor with no arguments, usually called when there is no specific initializer available.


**Chapter 3: Library Types**

- Library *string* type
    + *string* supports variable-length character strings
    + common ways to initialize a string
    {% highlight cpp %}
    std::string s1;              // Default constructor; s1 is the empty string
    std::string s2(s1);          // initialize s2 as a copy of s1
    std::string s3("value");     // initialize s3 as a copy of the string literal
    std::string s4(n, "c");      // initialize s4 with n copies of the character 'c'
    {% endhighlight %} 

    + reading and writing of *string*('>>' & '<<') always discard the whitespace(space).
    + '*getline*' reads the entire line, but discard the newline sign('\n').
    + operations defined for *string*
    
    ```cpp
    string.empty()          // returns true if string is empty; otherwise false.
    string.size()           // returns number of characters in the string
    string[n]               // access the n+1 th element in the string. string index starts from 0 to string.size()-1
    s1 + s2                 // concatenats s1 and s2
                            // **when mixing strings and string literals, at least one operand to each '+' operator must be of *string* type**
    s1 = s2                 // replace characters in s1 by a copy of s2
    s1 == s2                // returns trun if v1 and v2 are equal, otherwise false
    !=, <, <=, >, >=        // comparison between strings
    ```


