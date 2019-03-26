#ifndef UNIFIED_H
#define UNIFIED_H
#include <iostream>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>

class UnifiedMemoryClass
{
public:
    void* operator new(size_t);

    void operator delete(void*);
};
#endif
