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

**Chapter 6: Statements**

- Definitions:
    + Null statement: empty statement ends with a semicolon. It is useful when the language requires a statement but the programmer's logic doesn't.
    + Block: a compound statement, a sequence of statements surrounded by a pair of curly braces. A block is a scope.

- Notes:
    + dangling else problem: it occurs when a statement contains more if clauses than else clauses. It can be resolved by *matching the else with the last occuring unmatached if*.
    + *switch*: a deeply nested *if ... else ...*. 
        * It is usually to end the case label statement with "break", except the following case: you want to excute the same statement under multiple cases.
        * It is also necessary to always add *"default"* case label at the end of the switch block.
        * always **constant** for the case labels, **no variables** for the case labels.
    + *while*: expression in *while(condition)* will be converted into bool after being executed.
        * a *return* or *break* can help avoiding the endless loop.
    + *for*: it is a must to initialize the condition variable.
    + *do while*: 
        * The statement block *do* will firstly be executed, then the condition in *while* will be evaluated.
        * it ends with an *semicolon ;* at the end, after *while*.
        * it is not allowed to declare a new variable in the *while(condition)*.
    + *try catch*
    + *preprocessor*: *-D* is the compile flage for preprocessor information.


**Chapter 7: Functions**

- Definitions:
    + *Function*: Functions are named units of computation and are essential to structuring even modest programs. it is a programmer-defined operation. It is represented by a name and a set of operand types. 
        * The operands, *parameters*, are specified in a *comma-separated* list enclosed in parentheses.
        * *function body*: Actions that the function performs, specified in a block.
        * *return type*: every function has an associated return type.
    + *call operator*: a pair of parentheses.
    + *arguments*: the value used to initialize a parameter is the corresponding argument passed in the call. They might be variables, literal constants, or expressions.
    + *recursive function*: function that calls itself directly or indirectly.

- Notes:
    + a variable local to a function may not use the same name as the name of any of the function's parameters.
    + *Nonreference parameters* are the local *copies* of the corresponding argument. Changes are made to the local copy. Once the function terminates, the local values are gone. 
    + *Reference parameters* refer directly to the objects to which they are bound rather than to copies of those objects. It directly works on the variables instead of the local copies.
        * In C++, it is safer and more natural to use reference parameters.
        * When the only reason to make a parameter a reference is to avoid copying the argument, the parameter should be *const* reference.
    + *Vector parameters*: in practice, C++ program tend to pass containers by passing iterators to the elements we want to pass.
    + *Array parameters*: 
        * for nonreference parameter: when we pass the array, the argument is a pointer to the first element in the array.
        * for reference parameter: the compiler passes the reference to the array itself. The array size is part of the parameter and argument types.
    + *Return*: return statement terminates the functin that is currently executing and returns control to the functin that called the now-terminated function.
        * *void*: no return values.
        * return a value: the type of returned value should be declared in the function declaaration.
        * *NEVER return a reference to a local variable*. 
        * *NEVER return a pointer to a local object*.
        * a *recursive* function must always define a stopping condition. (ex 7.20)
    + Function prototype: return type, function name, and parameter list.
        * function declarations go in header files.
        * default argument: the value that is expected to be used most of the time.
        * While using a function, if some of the parameters already have default values, arguments for them are not needed.
        * A parameter can have its default argument specified only once in a file.
        * Default arguments ordinarily should be specified with the declaration for the function and placed in an appropriate header.
        * Parameters with default arguments should be *behind* the parameters without default arguments.
    + The *Scope* of a name is the part of the program's text in which that name is known.
    + The *lifetime* of an object is the time during the program's execution that the object exists.
    + *automatic objects*: only exist while a function is executing. They are created and destroyed on each call to a function.
    + *static local objects*: guaranteed to be initialized no later than the first time that program execution passes through the object's definition.
        * local *statics* will not be destroyed when the function ends.
        * local *statics* continue to exist and hold their values across calls to the function.
    + *Inline Function*: optimize small, straight-line functions that are called frequently.
        * *inline* is only a request to the compiler. The compiler may choose to ignore the request.
        * When calling a function is slower than evaluating the equivalent expression, we use *inline*.
        * *inline functions* should be *defined* in *header* files.
        * whenever an *inline* function is added or changed in the header file, every source file that uses the header file must be recompiled.
    + *Class Member Functions*:
        * function prototype must be defined within the class body, while the body of the function may be defined within the class itself, or outside the class body.
        * A member function that is defined inside the class is implicitly treated as an *inline* function.
        * A member function may access the private members of its class.
        * *this*: implicit parameter for each member function.
            - When a member function is called, the *this* parameter is initialized with the address of the object on which the function was invoked.
            - *const member functions*: *const* modifies the type of the implicit *this* parameter. *this* will return a *const* object.
        * Member functions defined outside the class definition must indicate that they are members of the class *class::member_function*.
        * *constructor*: data members are initialized through the constructor.
            - constructor is a special member function that is distinguished from other member funtions by having the same name as its class. It has no return type.
            - *default constructor*: takes no arguments. It says what happens when we define an object but do not supply an explicit initializer.
            - *constructor initializer list*: the colon and the following text up to the open curly is the constructor initializer list. It specifies the initial values for one or more data members of the class. They are separated by **commas**.
    + *Overloaded Functions*: two functions in the same scope are overloaded if they have the same name but different parameter lists.
        * Functions cannot be overloaded based only on differences in the return type.
        * declarations for every version of an overloaded function must appear in the same scope.
        * In C++, name lookup happens before type checking.
        * Function *overload resolution*: a function call is associated with a specific function from a set of overloaded functions.
        * For nonreference parameters, *const* and non-const are the same parameters. It's illegal to add *const* in front of the params to overload the function.
        * *Three Steps in Overload Resolution*
            - *Candidate Functions*: identify the set of overloaded functions considered for the call. The functions in this set are called candidate functions.
            - *Determining the viable functions*: Select the functions from the set of candidate functions that can be called with the arguments specified in the call. These selected functions are called *viable functions*.
                + The same number of parameters
                + The type of each argument must match or be convertible to the type of its corresponding parameter.
            - *Finding the best match, if any*: The closer the types of the arguments and parameter are to each other, the better the match.
                + The rank for conversions:
                    * Exact match
                    * Match through a promotion
                    * Match through a standard conversion
                    * Match through a class-type conversion
        * When using overloaded functions with *enum* parameters: Two enumeration types may behave quite differently during function overload resolution, depending on the value of their enumeration constants.
        * Whether a parameter is *const* only matters when the parameter is a reference or pointer.
    + *Pointers to Functions*: a function pointer is a pointer that denotes a unction rather than an object.
        * A function's type is determined by its *return type* and its *parameter list*.
        * use *Typedefs* to simplify function pointer definitions.
        * initializing a function pointer to zero indicates that the pointer does not point to any function.
        * Only pointers that have been initialized or assigned to refer to a function can be safely used to call a function.
        * A function parameter can be a function pointer.
        * A function can return a pointer to function. 
   
**Chapter 8: The IO Library**

- Definitions:
    + *condition states*: A set of condition state members that indicate whether a given IO object is in a usable state, or has encountered a particular kind of error.
- Notes:
    + The standard IO types are defined in three separate headers:
        * *iostream*: the types used to read and write to a console.
        * *fstream*: the types used to read and write named files.
        * *sstream*: the types used to read and write in-memory strings.
        * each type of *fstream* and *sstream* is derived from a corresponding type defined in the *iostream* header.
        * Because of the inheritance, we can pass an object of a derived type to that function, which takes a reference to a base-class type.
        * No copy of Assign for IO objects.
        * If we need to pass or return an IO object, it must be passed or returned as a pointer or reference.
    + To assure that the stream is loaded successfully, use *if* or *while* to test the state of the stream.
    + Each IO stream objects contains a condition state member that is managed through the *setstate* and *clear* operations. (*ex 8.3 & 8.4*)
        * *badbit*: a system level failure. Not possible to continue using a stream after such an error.
        * *failbit*: set after a recoverable error. 
        * *eofbit*: set when an end-of-file is encountered-
        * The state of the stream is revealed by the *bad*, *fail*, *eof* and *good* operations.
        * *clear*: put the stream back in its valid state. Called after we have remedied whatever problem occurred, and we want to set the stream to valid state.
        * *setstate*: turns on the specified condition to indicate that a problem occurred. It leaves existing state variables unchanged except that it adds the additional indicated state(s).
        * *rdstate*: returns an *iostate* value that corresponds to the entire current condition state of the stream.
    + Managing a buffer:
        * conditions causing the buffer flushed:
            - program completes normally.
            - at some indeterminate time, the buffer becomes full, in which case it will be flushed before writing to the next value.
            - using a manipulator: *endl*. It writes a newline and flushes the buffer.
            - use *unitbuf* manipulator to set the stream's internal state to empty the buffer after each output operation. Empty every output.
            - *tie* the output stream to an input stream. The output buffer is flushed whenever the associated input stream is read. *The library ties cout to cin*.
                + To break existing tie,  we pass in an argument of 0. (cin.tie(0)).
        * Buffers are not flushed if the program crashes. Thus we should use *endl* when writing output.