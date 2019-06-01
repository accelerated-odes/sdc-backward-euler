#ifndef _REAL_VECTOR_SET_H
#define _REAL_VECTOR_SET_H
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>

#ifdef AMREX_USE_CUDA
#include "cuda_runtime.h"
#endif

#include "AMReX_REAL.H"

#include "VectorParallelUtil.H"
#include "MathVectorSet.H"
#include "RealVector.H"

using namespace amrex;

template<size_t Nset, size_t Nvec, class StorageType>
class RealVectorSet : public MathVectorSet<Real, Nset, Nvec, StorageType> {
public:

  using MathVectorSet<Real, Nset, Nvec, StorageType>::data;
  using MathVectorSet<Real, Nset, Nvec, StorageType>::begin;
  using MathVectorSet<Real, Nset, Nvec, StorageType>::end;
  using MathVectorSet<Real, Nset, Nvec, StorageType>::operator=;
  using MathVectorSet<Real, Nset, Nvec, StorageType>::operator+=;
  using MathVectorSet<Real, Nset, Nvec, StorageType>::operator-=;
  using MathVectorSet<Real, Nset, Nvec, StorageType>::operator*=;
  using MathVectorSet<Real, Nset, Nvec, StorageType>::operator/=;
  using MathVectorSet<Real, Nset, Nvec, StorageType>::negate;
  using MathVectorSet<Real, Nset, Nvec, StorageType>::stored_size;

  typedef RealVectorSet<Nset, Nvec, StorageType> ThisType;

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ThisType& abs(int comp = -1) {
    // apply absolute value to the desired component, or all components
    if (comp >= 0)
      data[comp].abs();
    else
      VECTOR_SET_LAMBDA(Nset, stored_size,
                        [&](size_t& iset,size_t& ivec) {
                          data[iset].abs(ivec);
                        });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ThisType& inv(int comp = -1) {
    // apply inverse to the desired component, or all components
    if (comp >= 0)
      data[comp].inv();
    else
      VECTOR_SET_LAMBDA(Nset, stored_size,
                        [&](size_t& iset,size_t& ivec) {
                          data[iset].inv(ivec);
                        });
    return *this;
  }

  template<class AnyStorage1, class AnyStorage2>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ThisType& dot(RealVectorSet<Nset, Nvec, AnyStorage1>& y,
                RealVector<Nvec, AnyStorage2>& result) {
    // calculate the dot product with y with respect to components and store it in result
    result = 0.0;
    WORKER_SYNC();
    size_t looplen = min(stored_size, y.stored_size);
    VECTOR_SET_LAMBDA(Nset, looplen,
                      [&](size_t& iset, size_t& ivec) {
                        atomicAdd(&result[ivec], data[iset][ivec] * y[iset][ivec]);
                      });
    WORKER_SYNC();
    return *this;
  }

  template<class AnyStorage>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ThisType& max_norm(RealVector<Nvec, AnyStorage>& result) {
    // calculate the max norm with respect to components and store it in result
    result = 0.0;
    WORKER_SYNC();
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        atomicMax(&result[ivec], fabs(data[iset][ivec]));
                      });
    WORKER_SYNC();
    return *this;
  }

  template<class AnyStorage1, class AnyStorage2>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ThisType& wrms_norm(RealVectorSet<Nset, Nvec, AnyStorage1>& weights,
                      RealVector<Nvec, AnyStorage2>& result) {
    // calculate the component-wise wrms norm with respect to weights and store it in result
    result = 0.0;
    WORKER_SYNC();
    size_t looplen = min(stored_size, weights.stored_size);
    VECTOR_SET_LAMBDA(Nset, looplen,
                      [&](size_t& iset, size_t& ivec) {
                        atomicAdd(&result[ivec],
                                  pow(data[iset][ivec] * weights[iset][ivec], 2.0));
                      });
    WORKER_SYNC();
    VECTOR_LAMBDA(looplen,
                  [&](size_t& k) {
                    result[k] = sqrt(result[k] / Nset);
                  });
    WORKER_SYNC();
    return *this;
  }

  template<class AnyStorage>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ThisType& rms_norm(RealVector<Nvec, AnyStorage>& result) {
    // calculate the component-wise rms norm and store it in result
    result = 0.0;
    WORKER_SYNC();
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        atomicAdd(&result[ivec], pow(data[iset][ivec], 2.0));
                      });
    WORKER_SYNC();
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& k) {
                    result[k] = sqrt(result[k] / Nset);
                  });
    WORKER_SYNC();
    return *this;
  }

  template<class AnyStorage>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ThisType& frobenius_norm(RealVector<Nvec, AnyStorage>& result) {
    // calculate the component-wise frobenius norm and store it in result
    result = 0.0;
    WORKER_SYNC();
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        atomicAdd(&result[ivec], pow(data[iset][ivec], 2.0));
                      });
    WORKER_SYNC();
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& k) {
                    result[k] = sqrt(result[k]);
                  });
    WORKER_SYNC();
    return *this;
  }

  template<class AnyStorage>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ThisType& min(RealVector<Nvec, AnyStorage>& result) {
    // calculate the min with respect to components and store it in result
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& k) {
                    result[k] = data[0][k];
                  });
    WORKER_SYNC();
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        atomicMin(&result[ivec], data[iset][ivec]);
                      });
    WORKER_SYNC();
    return *this;
  }
};
#endif
