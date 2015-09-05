---
layout : post
category : LearningNote
tags : [C++, Basis]
---

{% include JB/setup %}

This post summarizes the note of some basic C++ syntax, which make me confused.

- *heap vs stack* ([link](http://stackoverflow.com/questions/79923/what-and-where-are-the-stack-and-heap))
    + The *stack* is the memory set aside as *scratch space for a thread of executation*. 
        * The stack is always reserved in a *LIFO (Last In First Out)* order. The most recently reserved block is always the next block to be freed.
        * When a function is called, a block is reserved on the top of the stack for local variables and some bookkeeping data.
        * When the function returns, the block becomes unused and can be used the next time a function is called.
    + The *heap* is the memory set aside for *dynamic allocation*. 
        * Unlike the stack, there is no enforced pattern to the allocation and deallocation of blocks from the heap. You can allocate a block at any time and free it at any time.
        * It is more complex to keep track of which parts of the heap are allocated or free at any given time.
    + Each thread gets a stack, while there is typically only one heap for the application.
    + In summary, the stack is attached to a thread when a thread is claimed, the heap is typically allocated at application startup by the runtime.
    + The size of the stack is set when a thread is created. The size of the heap is set on application startup, but can grow as space is needed.
    + A demonstration for "Stack vs. Heap".
    ![Stack vs. Heap ([source](http://vikashazrati.wordpress.com/2007/10/01/quicktip-java-basics-stack-and-heap/))](http://i.stack.imgur.com/i6k0Z.png)

- *lock. mutex*
- *thread*

- *template*
- *share_ptr*
- *smart pointers*
- *pointer to implementation*