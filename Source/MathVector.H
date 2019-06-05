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

  template<typename AnyMathType, typename AnyStorageType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator=(MathVector<AnyMathType, Nvec, AnyStorageType>& source) {
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& i) {
                    data[i] = static_cast<MathType>(source[i]);
                  });
    return *this;
  }

  template<typename AnyMathType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator=(AnyMathType scalar) {
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& i) {
                    data[i] = static_cast<MathType>(scalar);
                  });
    return *this;
  }

  template<typename AnyMathType, typename AnyStorageType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator+=(MathVector<AnyMathType, Nvec, AnyStorageType>& y) {
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& i) {
                    data[i] += static_cast<MathType>(y[i]);
                  });
    return *this;
  }

  template<typename AnyMathType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator+=(AnyMathType scalar) {
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& i) {
                    data[i] += static_cast<MathType>(scalar);
                  });
    return *this;
  }

  template<typename AnyMathType, typename AnyStorageType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator-=(MathVector<AnyMathType, Nvec, AnyStorageType>& y) {
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& i) {
                    data[i] -= static_cast<MathType>(y[i]);
                  });
    return *this;
  }

  template<typename AnyMathType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator-=(AnyMathType scalar) {
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& i) {
                    data[i] -= static_cast<MathType>(scalar);
                  });
    return *this;
  }

  template<typename AnyMathType, typename AnyStorageType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator*=(MathVector<AnyMathType, Nvec, AnyStorageType>& y) {
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& i) {
                    data[i] *= static_cast<MathType>(y[i]);
                  });
    return *this;
  }

  template<typename AnyMathType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator*=(AnyMathType scalar) {
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& i) {
                    data[i] *= static_cast<MathType>(scalar);
                  });
    return *this;
  }

  template<typename AnyMathType, typename AnyStorageType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator/=(MathVector<AnyMathType, Nvec, AnyStorageType>& y) {
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& i) {
                    data[i] /= static_cast<MathType>(y[i]);
                  });
    return *this;
  }

  template<typename AnyMathType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& operator/=(AnyMathType scalar) {
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& i) {
                    data[i] /= static_cast<MathType>(scalar);
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

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& abs() {
    // apply absolute value to all elements of the vector
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& i) {
                    data[i] = fabs(data[i]);
                  });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& inv() {
    // invert all the elements of the vector
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& i) {
                    data[i] = 1.0 / data[i];
                  });
    return *this;
  }

  template<typename AnyMathType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& max_norm(AnyMathType& result) {
    // calculate the max norm of elements in the vector and store in result
    SINGLE_LAMBDA([&]() { result = 0.0; });
    WORKER_SYNC();
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& i) {
                    atomicMax(&result, static_cast<AnyMathType>(fabs(data[i])));
                  });
    WORKER_SYNC();
    return *this;
  }

  template<typename AnyMathType1, typename AnyStorageType1, typename AnyMathType2>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& wrms_norm(MathVector<AnyMathType1, Nvec, AnyStorageType1>& weights, AnyMathType2& result) {
    // calculate the wrms norm of this vector with respect to the weight vector
    SINGLE_LAMBDA([&]() { result = 0.0; });
    WORKER_SYNC();
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& k) {
                    atomicAdd(&result,
                              static_cast<AnyMathType2>(data[k] * data[k] *
                                static_cast<MathType>(weights[k] * weights[k])));
                  });
    WORKER_SYNC();
    SINGLE_LAMBDA([&]() { result = sqrt(result / Nvec); });
    WORKER_SYNC();
    return *this;
  }

  template<typename AnyMathType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& rms_norm(AnyMathType& result) {
    // calculate the rms norm of this vector
    SINGLE_LAMBDA([&]() { result = 0.0; });
    WORKER_SYNC();
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& k) {
                    atomicAdd(&result, static_cast<AnyMathType>(data[k] * data[k]));
                  });
    WORKER_SYNC();
    SINGLE_LAMBDA([&]() { result = sqrt(result / Nvec); });
    WORKER_SYNC();
    return *this;
  }

  template<typename AnyMathType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& frobenius_norm(AnyMathType& result) {
    // calculate the frobenius norm of this vector
    SINGLE_LAMBDA([&]() { result = 0.0; });
    WORKER_SYNC();
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& k) {
                    atomicAdd(&result, static_cast<AnyMathType>(data[k] * data[k]));
                  });
    WORKER_SYNC();
    SINGLE_LAMBDA([&]() { result = sqrt(result); });
    WORKER_SYNC();
    return *this;
  }

  template<typename AnyMathType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MV& minimum(AnyMathType& result) {
    // calculate the min of elements in the vector and store in result
    SINGLE_LAMBDA([&]() { result = static_cast<AnyMathType>(data[0]); });
    WORKER_SYNC();
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& k) {
                    atomicMin(&result, static_cast<AnyMathType>(data[k]));
                  });
    WORKER_SYNC();
    return *this;
  }
};
#endif
