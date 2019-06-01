#ifndef _MATH_VECTOR_H
#define _MATH_VECTOR_H
#include <iostream>
#include <cassert>
#include "AMReX_GpuQualifiers.H"
#include "AMReX_Extension.H"
#include "VectorParallelUtil.H"

#ifndef AMREX_USE_CUDA
using std::min;
using std::max;
#endif

template<class MathType, size_t Nvec, class StorageType>
class MathVector : public StorageType {
public:

  using StorageType::data;
  using StorageType::map;
  using StorageType::save;
  using StorageType::stored_size;
  using StorageType::operator[];

  using MV = MathVector<MathType, Nvec, StorageType>;

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
    return data + stored_size;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MathType* dataPtr() {
    return begin();
  }

  virtual void print() {
    size_t i = 0;
    for (size_t i = 0; i < stored_size; i++) {
      std::cout << this->data[i];
      if (i < stored_size - 1) std::cout << " ";
      i++;
    }
  }

  template<class SourceStoreClass>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator=(MathVector<MathType, Nvec, SourceStoreClass>& source) {
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& i) {
                    data[i] = source[i];
                  });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator=(MathType scalar) {
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& i) {
                    data[i] = scalar;
                  });
    return *this;
  }

  template<class SourceStoreClass>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator+=(MathVector<MathType, Nvec, SourceStoreClass>& y) {
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& i) {
                    data[i] += y[i];
                  });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator+=(MathType scalar) {
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& i) {
                    data[i] += scalar;
                  });
    return *this;
  }

  template<class SourceStoreClass>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator-=(MathVector<MathType, Nvec, SourceStoreClass>& y) {
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& i) {
                    data[i] -= y[i];
                  });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator-=(MathType scalar) {
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& i) {
                    data[i] -= scalar;
                  });
    return *this;
  }

  template<class SourceStoreClass>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator*=(MathVector<MathType, Nvec, SourceStoreClass>& y) {
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& i) {
                    data[i] *= y[i];
                  });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator*=(MathType scalar) {
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& i) {
                    data[i] *= scalar;
                  });
    return *this;
  }

  template<class SourceStoreClass>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator/=(MathVector<MathType, Nvec, SourceStoreClass>& y) {
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& i) {
                    data[i] /= y[i];
                  });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator/=(MathType scalar) {
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& i) {
                    data[i] /= scalar;
                  });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& negate() {
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& i) {
                    data[i] = -data[i];
                  });
    return *this;
  }
};
#endif
