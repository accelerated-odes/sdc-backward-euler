#ifndef _REAL_VECTOR_H
#define _REAL_VECTOR_H
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include "AMReX_REAL.H"
#include "MathVector.H"

using namespace amrex;

template<size_t N> class RealVector : public MathVector<Real,N> {
public:
  using MathVector<Real,N>::begin;
  using MathVector<Real,N>::end;

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  static void abs(RealVector<N>& v) {
    for (size_t i = 0; i < N; i++) {
      v.data[i] = fabs(v.data[i]);
    }
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  static void inv(RealVector<N>& v) {
    for (size_t i = 0; i < N; i++) {
      v.data[i] = 1.0/v.data[i];
    }
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  static Real dot(RealVector<N>& x, RealVector<N>& y) {
    Real sum = 0.0;
    for (size_t i = 0; i < N; i++) {
      sum += x.data[i] * y.data[i];
    }
    return sum;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  static Real max_norm(RealVector<N>& v) {
    Real max = 0.0;
    for (size_t i = 0; i < N; i++) {
      max = (fabs(v.data[i]) > max) ? fabs(v.data[i]) : max;
    }
    return max;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  static Real wrms_norm(RealVector<N>& v, RealVector<N>& weights) {
    Real sum = 0.0;
    for (size_t i = 0; i < N; i++) {
      sum += pow(v.data[i] * weights.data[i], 2.0);
    }
    sum = sqrt(sum / N);
    return sum;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  static Real rms_norm(RealVector<N>& v) {
    Real sum = 0.0;
    for (size_t i = 0; i < N; i++) {
      sum += pow(v.data[i], 2.0);
    }
    sum = sqrt(sum / N);
    return sum;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  static Real frobenius_norm(RealVector<N>& v) {
    Real sum = 0.0;
    for (size_t i = 0; i < N; i++) {
      sum += pow(v.data[i], 2.0);
    }
    sum = sqrt(sum);
    return sum;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  static Real min(RealVector<N>& v) {
    Real xmin = v.data[0];
    for (size_t i = 0; i < N; i++) {
      xmin = (v.data[i] < xmin) ? v.data[i] : xmin;
    }
    return xmin;
  }
};
#endif
