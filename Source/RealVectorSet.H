#ifndef _REAL_VECTOR_SET_H
#define _REAL_VECTOR_SET_H
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include "cuda_runtime.h"
#include "VectorGpuMacros.H"
#include "AMReX_REAL.H"
#include "MathVectorSet.H"
#include "RealVector.H"

using namespace amrex;

template<size_t Nset, size_t Nvec> class RealVectorSet : public MathVectorSet<Real, Nset, Nvec> {
public:

  using MathVectorSet<Real, Nset, Nvec>::data;
  using MathVectorSet<Real, Nset, Nvec>::begin;
  using MathVectorSet<Real, Nset, Nvec>::end;
  using MathVectorSet<Real, Nset, Nvec>::operator=;
  using MathVectorSet<Real, Nset, Nvec>::operator+=;
  using MathVectorSet<Real, Nset, Nvec>::operator-=;
  using MathVectorSet<Real, Nset, Nvec>::operator*=;
  using MathVectorSet<Real, Nset, Nvec>::operator/=;
  using MathVectorSet<Real, Nset, Nvec>::negate;

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  RealVectorSet<Nset, Nvec>& abs(int comp = -1) {
    // apply absolute value to the desired component, or all components
    if (comp >= 0)
      data[comp].abs();
    else
      VECTOR_SET_LAMBDA(Nset, Nvec,
                        [&](size_t& iset,size_t& ivec) {
                          data[iset].abs(ivec);
                        });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  RealVectorSet<Nset, Nvec>& inv(int comp = -1) {
    // apply inverse to the desired component, or all components
    if (comp >= 0)
      data[comp].inv();
    else
      VECTOR_SET_LAMBDA(Nset, Nvec,
                        [&](size_t& iset,size_t& ivec) {
                          data[iset].inv(ivec);
                        });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  RealVectorSet<Nset, Nvec>& dot(RealVectorSet<Nset, Nvec>& y,
                                 RealVector<Nvec>& result) {
    // calculate the dot product with y with respect to components and store it in result
    result = 0.0;
    WORKER_SYNC();
    VECTOR_SET_LAMBDA(Nset, Nvec,
                      [&](size_t& iset, size_t& ivec) {
                        atomicAdd(&result[ivec], data[iset][ivec] * y[iset][ivec]);
                      });
    WORKER_SYNC();
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  RealVectorSet<Nset, Nvec>& max_norm(RealVector<Nvec>& result) {
    // calculate the max norm with respect to components and store it in result
    result = 0.0;
    WORKER_SYNC();
    VECTOR_SET_LAMBDA(Nset, Nvec,
                      [&](size_t& iset, size_t& ivec) {
                        atomicMax(&result[ivec], fabs(data[iset][ivec]));
                      });
    WORKER_SYNC();
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  RealVectorSet<Nset, Nvec>& wrms_norm(RealVectorSet<Nset, Nvec>& weights,
                                       RealVector<Nvec>& result) {
    // calculate the component-wise wrms norm with respect to weights and store it in result
    result = 0.0;
    WORKER_SYNC();
    VECTOR_SET_LAMBDA(Nset, Nvec,
                      [&](size_t& iset, size_t& ivec) {
                        atomicAdd(&result[ivec],
                                  pow(data[iset][ivec] * weights[iset][ivec], 2.0));
                      });
    WORKER_SYNC();
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& k) {
                    result[k] = sqrt(result[k] / Nset);
                  });
    WORKER_SYNC();
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  RealVectorSet<Nset, Nvec>& rms_norm(RealVector<Nvec>& result) {
    // calculate the component-wise rms norm and store it in result
    result = 0.0;
    WORKER_SYNC();
    VECTOR_SET_LAMBDA(Nset, Nvec,
                      [&](size_t& iset, size_t& ivec) {
                        atomicAdd(&result[ivec], pow(data[iset][ivec], 2.0));
                      });
    WORKER_SYNC();
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& k) {
                    result[k] = sqrt(result[k] / Nset);
                  });
    WORKER_SYNC();
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  RealVectorSet<Nset, Nvec>& frobenius_norm(RealVector<Nvec>& result) {
    // calculate the component-wise frobenius norm and store it in result
    result = 0.0;
    WORKER_SYNC();
    VECTOR_SET_LAMBDA(Nset, Nvec,
                      [&](size_t& iset, size_t& ivec) {
                        atomicAdd(&result[ivec], pow(data[iset][ivec], 2.0));
                      });
    WORKER_SYNC();
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& k) {
                    result[k] = sqrt(result[k]);
                  });
    WORKER_SYNC();
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  RealVectorSet<Nset, Nvec>& min(RealVector<Nvec>& result) {
    // calculate the min with respect to components and store it in result
    VECTOR_LAMBDA(Nvec,
                  [&](size_t& k) {
                    result[k] = data[0][k];
                  });
    WORKER_SYNC();
    VECTOR_SET_LAMBDA(Nset, Nvec,
                      [&](size_t& iset, size_t& ivec) {
                        atomicMin(&result[ivec], data[iset][ivec]);
                      });
    WORKER_SYNC();
    return *this;
  }
};
#endif
