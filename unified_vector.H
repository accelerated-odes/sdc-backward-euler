#ifndef DEVICE_VECTOR_H
#define DEVICE_VECTOR_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include "unified.H"

template<class T> class UnifiedVector : public UnifiedMemoryClass {

public:
    size_t filled_size;
    size_t allocated_size;
    T* data_ptr;

    UnifiedVector(size_t create_size = 1) {
        cudaError_t cuda_status;
        cuda_status = cudaMallocManaged(&data_ptr, sizeof(T)*create_size);
        assert(cuda_status == cudaSuccess);
        allocated_size = create_size;
        filled_size = 0;
    }

    ~UnifiedVector() {
        cudaError_t cuda_status = cudaFree(data_ptr);
        allocated_size = 0;
        assert(cuda_status == cudaSuccess);
    }

    AMREX_GPU_HOST_DEVICE T* dataPtr() {
        return begin();
    }

    AMREX_GPU_HOST_DEVICE T* begin() {
        return data_ptr;
    }

    AMREX_GPU_HOST_DEVICE T* end() {
        return data_ptr + filled_size;
    }

    AMREX_GPU_HOST_DEVICE T& operator[] (unsigned int i) {
        // need to check this isn't OOB!!
        return *(data_ptr + i);
    }

    AMREX_GPU_HOST_DEVICE
    UnifiedVector<T>& operator=(UnifiedVector<T>& other) {
#ifdef __CUDA_ARCH__
        assert(other.filled_size <= this->allocated_size);
#else
        if (other.filled_size > this->allocated_size) resize(other.filled_size);
#endif
        for (int i = 0; i < other.filled_size; i++) {
            *(data_ptr + i) = other[i];
        }
        filled_size = other.filled_size;
        return *this;
    }

    AMREX_GPU_HOST_DEVICE T* get_pointer_to (unsigned int i) {
        return (data_ptr + i);
    }

    void push_back(T element) {
        if (filled_size+1 > allocated_size) resize(max(2*allocated_size, static_cast<size_t>(1)));
        new (data_ptr + filled_size) T(element);
        filled_size++;
    }

    void resize(size_t new_size) {
        if (new_size == 0) filled_size = 0;
        else {
            T* new_data_ptr;
            cudaError_t cuda_status;

            cuda_status = cudaMallocManaged(&new_data_ptr, sizeof(T)*new_size);
            assert(cuda_status == cudaSuccess);

            size_t num_elements_to_copy = min(new_size, filled_size);
            memcpy(new_data_ptr, data_ptr, sizeof(T)*num_elements_to_copy);

            cuda_status = cudaFree(data_ptr);
            assert(cuda_status == cudaSuccess);

            data_ptr = new_data_ptr;
            allocated_size = new_size;
        }
    }

    AMREX_GPU_HOST_DEVICE size_t size() {
        return filled_size;
    }
};
#endif
