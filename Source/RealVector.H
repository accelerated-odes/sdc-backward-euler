#ifndef _REAL_VECTOR_H
#define _REAL_VECTOR_H
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include "AMReX_REAL.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_Extension.H"
#include "VectorGpuMacros.H"
#include "MathVector.H"

using namespace amrex;

template<size_t Nvec, class StorageType>
class RealVector : public MathVector<Real, Nvec, StorageType> {
public:

  using MathVector<Real, Nvec, StorageType>::data;
  using MathVector<Real, Nvec, StorageType>::begin;
  using MathVector<Real, Nvec, StorageType>::end;
  typedef RealVector<Nvec, StorageType> ThisType;

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ThisType& abs(size_t comp) {
    // apply absolute value to a component of the vector
    data[comp] = fabs(data[comp]);
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ThisType& abs() {
    // apply absolute value to all elements of the vector
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& i) {
                    abs(i);
                  });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ThisType& inv(size_t comp) {
    // invert all the elements of the vector
    data[comp] = 1.0/data[comp];
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ThisType& inv() {
    // invert all the elements of the vector
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& i) {
                    inv(i);
                  });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ThisType& max_norm(Real& result) {
    // calculate the max norm of elements in the vector and store in result
    SINGLE_LAMBDA([&]() { result = 0.0; });
    WORKER_SYNC();
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& i) {
                    atomicMax(&result, fabs(data[i]));
                  });
    WORKER_SYNC();
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ThisType& wrms_norm(ThisType& weights, Real& result) {
    // calculate the wrms norm of this vector with respect to the weight vector
    SINGLE_LAMBDA([&]() { result = 0.0; });
    WORKER_SYNC();
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& k) {
                    atomicAdd(&result, pow(data[k] * weights[k], 2.0));
                  });
    WORKER_SYNC();
    SINGLE_LAMBDA([&]() { result = sqrt(result / Nvec); });
    WORKER_SYNC();
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ThisType& rms_norm(Real& result) {
    // calculate the rms norm of this vector
    SINGLE_LAMBDA([&]() { result = 0.0; });
    WORKER_SYNC();
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& k) {
                    atomicAdd(&result, pow(data[k], 2.0));
                  });
    WORKER_SYNC();
    SINGLE_LAMBDA([&]() { result = sqrt(result / Nvec); });
    WORKER_SYNC();
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ThisType& frobenius_norm(Real& result) {
    // calculate the frobenius norm of this vector
    SINGLE_LAMBDA([&]() { result = 0.0; });
    WORKER_SYNC();
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& k) {
                    atomicAdd(&result, pow(data[k], 2.0));
                  });
    WORKER_SYNC();
    SINGLE_LAMBDA([&]() { result = sqrt(result); });
    WORKER_SYNC();
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ThisType& min(Real& result) {
    // calculate the min of elements in the vector and store in result
    SINGLE_LAMBDA([&]() { result = data[0]; });
    WORKER_SYNC();
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& k) {
                    atomicMin(&result, data[k]);
                  });
    WORKER_SYNC();
    return *this;
  }
};
#endif
