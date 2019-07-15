#ifndef _SPARSE_GAUSS_JORDAN_H
#define _SPARSE_GAUSS_JORDAN_H
#include "AMReX_REAL.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_Extension.H"

template<class SystemClass, size_t vector_length>
class SparseGaussJordan {
public:

  template<template<typename,size_t> typename AnyStorageType1,
           template<typename,size_t> typename AnyStorageType2,
           template<typename,size_t> typename AnyStorageType3,
           template<typename,size_t> typename AnyStorageType4>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  static void solve(RealSparseMatrixSet<SystemClass::neqs, vector_length, SystemClass::nnz,
                      AnyStorageType1>& Av,
                    MathVectorSet<Real, vector_length, SystemClass::neqs,
                                  AnyStorageType2<Real, SystemClass::neqs>>& xv,
                    MathVectorSet<Real, vector_length, SystemClass::neqs,
                                  AnyStorageType3<Real, SystemClass::neqs>>& bv,
                    MathVector<size_t, vector_length,
                               AnyStorageType4<size_t, vector_length>>& active_map,
                    unsigned int num_active) {

    VECTOR_LAMBDA(static_cast<size_t>(num_active),
                  [&](const size_t& k_active) {
                    const size_t& k = active_map[k_active];

                    auto A = Av[k];
                    auto b = bv[k];
                    auto x = xv[k];

                    Real scratch_0 = 1.0/A[0];
                    Real scratch_1 = b[0]*scratch_0;
                    Real scratch_2 = A[2]*scratch_0;
                    Real scratch_3 = A[1]*scratch_0;
                    Real scratch_4 = 1.0/(-A[3]*scratch_3 + A[4]);
                    Real scratch_5 = scratch_4*(-A[3]*scratch_2 + A[5]);
                    Real scratch_6 = scratch_4*(-A[3]*scratch_1 + b[1]);
                    Real scratch_7 = (-A[6]*scratch_6 + b[2])/(-A[6]*scratch_5 + A[7]);

                    x[0] = scratch_1 - scratch_3*scratch_6 - scratch_7*(scratch_2 - scratch_3*scratch_5);
                    x[1] = -scratch_5*scratch_7 + scratch_6;
                    x[2] = scratch_7;
                  });

    WORKER_SYNC();
  }
};

#endif
