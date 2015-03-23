---
layout : post
category : LearningNote
tags : [C++Primer, Basis]
---
{% include JB/setup %}

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
    
    >{% highlight cpp %}
      std::string s1;              // Default constructor; s1 is the empty string
    std::string s2(s1);          // initialize s2 as a copy of s1
    std::string s3("value");     // initialize s3 as a copy of the string literal
    std::string s4(n, "c");      // initialize s4 with n copies of the character 'c'
    {% endhighlight %} 

    + reading and writing of *string*('>>' & '<<') always discard the whitespace(space).
    + '*getline*' reads the entire line, but discard the newline sign('\n').
    + operations defined for *string*
    + when mixing strings and string literals, at least one operand to each '+' operator must be of *string* type
    
    >{% highlight cpp %}
      string.empty()          // returns true if string is empty; otherwise false.
    string.size()           // returns number of characters in the string
    string[n]               // access the n+1 th element in the string. string index starts from 0 to string.size()-1
    s1 + s2                 // concatenats s1 and s2
    s1 = s2                 // replace characters in s1 by a copy of s2
    s1 == s2                // returns trun if v1 and v2 are equal, otherwise false
    !=, <, <=, >, >=        // comparison between strings
    {% endhighlight %} 


**Chapter 4: Arrays and Pointers**

- Array
    + *initialization*
    + *operation*

- Pointer
    + *pointer* is a compound type, pointing to an object of some other type. It offers indirect access to the object to which it points.
    + It can point to an object, instead of a single element in an array.
    + "*" is the pointer operator, "&" is the *address-of* operator.
    + 4 way to initialize/assign:
        * constant expression with value 0
        * address of an object of an appropriate type
        * address on past the end of another object
        * another valid pointer of the same type
    + *Difference between Pointers and References*
        * Reference is another name (alias) to the object, however pointer is just an indirective access of the object. 
        * Once initialized, reference is fixed to one object. However, reference can be changed to different addresses of different objects.
    + When you define a const object, you must initialize the const object. However, the const pointer doesn't have to be initialized.

- C-Style character strings
    + strcpy: copy entire strings
    + strncpy: copy an assigned number of strings

**Chapter 5: Expressions**

- Definitions:
    + *Expression* is composed of one or more operands that are combined by operators. 
    + *unary* and *binary* operators. 
        * unary operators have the highest precedence.
    + *overflow*: a situation whereby the calculated result size exceeds that which can be defined by the data type.
    + *relational and logical* operators return values of *bool*.
        * *logical* operator treats the operands as conditions. If the value is 0, then the result of logical operator is *false*. Otherwise, it is *true*.
        * *short circuite evaluation*: evalute the left operands first. If the left operands can not determine the condition, it will evaluate the rest operands.
    + *Bitwise* operators: take operands of integral type. These operators treat their operands as a collection of bits, providing operations to test and set individual bit.
        * it is strongly recommended to use *unsigned int* while using bitwise operators.
        * *bitset* operations are more direct, easier to read, easier to write and more likely to be used correctly.
        * *I/O operators '>>' & '<<'* are left associative.
        * *shift operators '>>' & '<<'* have middel-level precedence. LOWER that *arithmetic* operators, but HIGHER than *relational, assignment, conditional* operators.
    + *assignment* operators: left-hand operands of assignment operators must be *nonconst* lvalues.
        * assignment operator is right-associative.
        * compound operators: *a op= b* is essentially equal to *a = a op b*. The left-hand operand is only evaluated once.
    + *incremental '++' & decremental '--'* operators: they provide a shorthand for adding or subtracting 1 from an object.
        * *prefix operators* yield a *changed result/value* and also a *changed operand*.
        * *postfix operators* yield an *unchanged result/value* but *changed operand*.
        * mostly, we use prefix operators.
        * *postfix operators* have higher precedence than *dereference '*'* operator.
    + *Arrow '->'* operator: 
        (*p).foo  // dereference p to get an object, and then fetch its member named foo.
        p->foo    // equavilant way to fetch the foo from the object to which p points.
    + *conditional operator*: the only ternary operator in c++.
        cond ? expression1 (true) : expression2 (false);
    + *sizeof* operator: returns the size of defined data type in bytes.
    + When in doubt, parenthesize expressions to force the grouping that the logic of your program requires.
    + If you change the value of an operand, don't use that operand elsewhere in the same statement.