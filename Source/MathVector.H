#ifndef _MATH_VECTOR_H
#define _MATH_VECTOR_H
#include <iostream>
#include <cassert>
#include "AMReX_GpuQualifiers.H"
#include "AMReX_Extension.H"
#include "VectorGpuMacros.H"

template<class MathType, size_t Nvec> class MathVector {
public:

  using MV = MathVector<MathType, Nvec>;
  
  MathType data[Nvec];

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MathVector() {}

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ~MathVector() {}

  // begin() and end() are iterators for sets
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MathType* begin() {
    return data;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MathType* end() {
    return data + Nvec;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MathType* dataPtr() {
    return begin();
  }

  virtual void print() {
    size_t i = 0;
    for (size_t i = 0; i < Nvec; i++) {
      std::cout << this->data[i];
      if (i < Nvec - 1) std::cout << " ";
      i++;
    }
  }  
  
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator=(MV& source) {
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& i) {
                    data[i] = source[i];
                  });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator=(MathType scalar) {
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& i) {
                    data[i] = scalar;
                  });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& load(MathType* array, size_t array_max_size) {
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& i) {
                     #ifdef __CUDA_ARCH__
                       size_t k = i + Nvec * blockIdx.x;
                       if (k < array_max_size)
                         data[i] = array[k];
                     #else
                       if (i < array_max_size)
                         data[i] = array[i];
                     #endif
                  });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& save(MathType* array, size_t array_max_size) {
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& i) {
                     #ifdef __CUDA_ARCH__
                       size_t k = i + Nvec * blockIdx.x;
                       if (k < array_max_size)
                         array[k] = data[i];
                     #else
                       if (i < array_max_size)
                         array[i] = data[i];
                     #endif
                  });
    return *this;
  }

  // The indexing operator returns the indexed entry
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MathType& operator[](unsigned int i) {
    return data[i];
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator+=(MV& y) {
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& i) {
                    data[i] += y[i];
                  });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator+=(MathType scalar) {
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& i) {
                    data[i] += scalar;
                  });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator-=(MV& y) {
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& i) {
                    data[i] -= y[i];
                  });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator-=(MathType scalar) {
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& i) {
                    data[i] -= scalar;
                  });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator*=(MV& y) {
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& i) {
                    data[i] *= y[i];
                  });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator*=(MathType scalar) {
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& i) {
                    data[i] *= scalar;
                  });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator/=(MV& y) {
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& i) {
                    data[i] /= y[i];
                  });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator/=(MathType scalar) {
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& i) {
                    data[i] /= scalar;
                  });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& negate() {
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& i) {
                    data[i] = -data[i];
                  });
    return *this;
  }
};
#endif
